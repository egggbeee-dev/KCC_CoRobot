# phases.py
#
# PASS/INFORM/handoff 완전 제거.
# 에이전트 간 협력은 depends_on으로만 표현한다.
#
# Phase 1 : Offer Generation      — 이미지 기반 capability 선언 + 교환
# Phase 2 : Local Planning        — 각자 독립 플랜 생성
# Phase 2b: Coordination Injection— offer 매칭으로 cross-agent depends_on 자동 삽입
# Phase 3 : Conflict Detection    — 5가지 rule-based 탐지
# Phase 4 : P2P Negotiation       — 구조화 제안, 최대 3라운드
# Phase 5 : Convergence Check     — rule-based 수렴 판단
# Phase 6 : Deferred Human Query  — 수렴 실패 시에만, VLM 기반
# Finalize: Rule-based merge      — LLM 없음

from __future__ import annotations

import json
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Dict, List, Optional, Set, Tuple

from p2p_config import (
    AGENT_B_STEP_OFFSET, AUTO_HQ_ANSWER, FUZZY_STOPWORDS,
    HQ_TOP_K, MAX_CAN_DO, MAX_CANNOT_DO, MAX_NEGOTIATION_ROUNDS,
    UNCERTAINTY_THRESH, VALID_AGENTS, VALID_PROPOSAL_FIELDS,
)
from p2p_models import (
    CannotEntry, ConflictEntry, ConflictType, ConvergenceResult,
    HQEntry, LocalPlan, NegotiationProposal, NegotiationRound,
    Offer, PlanStep,
)
from p2p_utils import (
    _banner, _fuzzy_match, _fuzzy_match_soft, _log,
    _match_conf, _norm_agent, _norm_depends, _norm_reason,
    clamp01, compute_plan_uncertainty, compute_token_uncertainty,
    extract_json, jdump, safe_int,
)
from p2p_vlm import run_vlm


# ── 병렬 VLM 호출 ─────────────────────────────────────────────────────────────

def _run_parallel(calls: List[Tuple]) -> List[Tuple[str, List[float]]]:
    with ThreadPoolExecutor(max_workers=len(calls)) as ex:
        futs = [ex.submit(run_vlm, *c) for c in calls]
    return [f.result() for f in futs]


# ── 직렬화 ────────────────────────────────────────────────────────────────────

def offer_to_dict(o: Offer) -> Dict:
    return {
        "agent_id":        o.agent_id,
        "room_type":       o.room_type,
        "observation":     o.observation,
        "obs_scope":       o.obs_scope,
        "can_do":          o.can_do,
        "cannot_do":       [{"action": c.action, "reason": c.reason} for c in o.cannot_do],
        "conf":            o.conf,
        "can_provide":     o.can_provide,
        "need_from_other": o.need_from_other,
    }


def local_plan_to_dict(lp: LocalPlan) -> Dict:
    return {
        "agent_id": lp.agent_id,
        "U_plan":   round(lp.U_plan, 3),
        "steps":    [asdict(s) for s in lp.steps],
        "hq_list":  [asdict(h) for h in lp.hq_list],
    }


def plan_steps_to_dicts(steps: List[PlanStep]) -> List[Dict]:
    return [asdict(s) for s in steps]


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: OBSERVATION & OFFER GENERATION
# ══════════════════════════════════════════════════════════════════════════════

_PHASE1_PROMPT_EXAMPLE = """
EXAMPLE — kitchen agent:
<JSON>
{
  "room_type": "kitchen",
  "observation": "Kitchen with countertop, stove, fruits on the island, bread basket.",
  "obs_scope": "counter, island, stove, sink, shelf, fruits, bread basket",
  "can_do": [
    "place apple and orange from island onto serving tray",
    "arrange bread from basket onto plate",
    "wipe counter surface with cloth",
    "clean visible sink with sponge"
  ],
  "cannot_do": [
    {"action": "arrange living room seating", "reason": "NO_OBJECT"},
    {"action": "adjust TV settings", "reason": "NO_OBJECT"}
  ],
  "conf": {
    "place apple and orange from island onto serving tray": 0.9,
    "arrange bread from basket onto plate": 0.85,
    "wipe counter surface with cloth": 0.95,
    "clean visible sink with sponge": 0.9
  },
  "can_provide": ["snack tray with fruits and bread"],
  "need_from_other": ["confirmation that living room table is cleared for snacks"]
}
</JSON>
""".strip()


def _build_phase1_prompt(task: str) -> str:
    # 에이전트가 더 구체적인 자원 중심(Resource-oriented) 선언을 하도록 프롬프트 강화
    return f"""You are an embodied home agent. Observe your room image carefully.

Global task: "{task}"

RULES:
1. can_provide: List tangible items or specific completion signals you can deliver to the other agent.
   (e.g., "delivered glass of water", "placed tray on the counter")
2. need_from_other: List specific items or actions you require from the other agent to finish the task.
   (e.g., "need the table cleared", "need a bottle of juice from the kitchen")
3. Use ONLY objects actually visible in your image.
4. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "room_type": "...",
  "observation": "one concise sentence",
  "obs_scope": "comma-separated list of visible objects",
  "can_do": ["action 1", "action 2"],
  "cannot_do": [{{"action": "...", "reason": "NO_OBJECT"}}],
  "conf": {{"action 1": 0.9, "action 2": 0.8}},
  "can_provide": ["item/signal you provide"],
  "need_from_other": ["item/action you need"]
}}
</JSON>"""



def _parse_offer(raw: str, agent_id: str) -> Offer:
    data = extract_json(raw)

    cannot_do: List[CannotEntry] = []
    uncertain_count = 0
    for item in data.get("cannot_do", [])[:MAX_CANNOT_DO]:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip()
        reason = _norm_reason(item.get("reason", "UNCERTAIN"))
        if action:
            if reason == "UNCERTAIN":
                uncertain_count += 1
            cannot_do.append(CannotEntry(action, reason))

    cannot_set = {c.action.lower() for c in cannot_do}
    seen: set = set()
    can_do: List[str] = []
    for x in data.get("can_do", []):
        a = str(x).strip()
        if not a or a.lower() in seen:
            continue
        if any(_fuzzy_match(a, c, min_overlap=2) for c in cannot_set):
            continue
        seen.add(a.lower())
        can_do.append(a)
        if len(can_do) >= MAX_CAN_DO:
            break

    raw_scope = data.get("obs_scope", "")
    obs_scope = (
        ", ".join(str(x).strip() for x in raw_scope)
        if isinstance(raw_scope, list)
        else str(raw_scope).strip()
    )

    conf_raw = {str(k): clamp01(v) for k, v in data.get("conf", {}).items()}

    return Offer(
        agent_id        = agent_id,
        room_type       = str(data.get("room_type", "")).strip(),
        observation     = str(data.get("observation", "")).strip(),
        obs_scope       = obs_scope,
        can_do          = can_do,
        cannot_do       = cannot_do,
        conf            = _match_conf(conf_raw, can_do),
        can_provide     = [str(x).strip() for x in data.get("can_provide", []) if str(x).strip()],
        need_from_other = [str(x).strip() for x in data.get("need_from_other", []) if str(x).strip()],
        uncertain_count = uncertain_count,
    )


def phase1_offer(
    img_a: str, img_b: str, task: str, verbose: str = "full"
) -> Tuple[Offer, Offer]:
    _banner("PHASE 1 — OBSERVATION & OFFER GENERATION")
    prompt  = _build_phase1_prompt(task)
    results = _run_parallel([(img_a, prompt, False), (img_b, prompt, False)])
    raw_a, _ = results[0]
    raw_b, _ = results[1]

    if verbose == "full":
        _log("A RAW OFFER", raw_a)
        _log("B RAW OFFER", raw_b)

    offer_a = _parse_offer(raw_a, "agent_A")
    offer_b = _parse_offer(raw_b, "agent_B")

    if verbose in ("full", "summary"):
        _log("OFFER A", jdump(offer_to_dict(offer_a)))
        _log("OFFER B", jdump(offer_to_dict(offer_b)))

    print(f"\n  A: room={offer_a.room_type} | can_do={len(offer_a.can_do)} "
          f"| provide={len(offer_a.can_provide)} | need={len(offer_a.need_from_other)}")
    print(f"  B: room={offer_b.room_type} | can_do={len(offer_b.can_do)} "
          f"| provide={len(offer_b.can_provide)} | need={len(offer_b.need_from_other)}")
    return offer_a, offer_b


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: LOCAL PLANNING
# ══════════════════════════════════════════════════════════════════════════════

_PHASE2_EXAMPLE = """
EXAMPLE — kitchen agent, task "prepare movie night":
<JSON>
{
  "plan_steps": [
    {"step_id": 1, "time_min": 0,  "action": "place apple and orange from island onto serving tray",
     "preconditions": [], "depends_on": [], "uncertainty": 0.1, "notes": ""},
    {"step_id": 2, "time_min": 5,  "action": "arrange bread from basket onto plate",
     "preconditions": [], "depends_on": [], "uncertainty": 0.1, "notes": ""},
    {"step_id": 3, "time_min": 10, "action": "wipe counter surface with cloth",
     "preconditions": [], "depends_on": [], "uncertainty": 0.1, "notes": ""},
    {"step_id": 4, "time_min": 15, "action": "clean visible sink with sponge",
     "preconditions": [], "depends_on": [], "uncertainty": 0.1, "notes": ""}
  ]
}
</JSON>
""".strip()


def _build_phase2_prompt(my: Offer, other: Offer, task: str, use_offer: bool) -> str:
    # 1. 협력 기회 파악 (기존 로직 활용)
    i_provide = [p for p in my.can_provide if any(_fuzzy_match_soft(p, n) for n in other.need_from_other)]
    other_provides = [p for p in other.can_provide if any(_fuzzy_match_soft(p, n) for n in my.need_from_other)]

    ctx = f"""YOUR ROOM: {my.room_type} ({my.agent_id})
OTHER AGENT: {other.room_type} ({other.agent_id})

PEER AGENT CAPABILITIES:
- Can provide to you: {json.dumps(other.can_provide, ensure_ascii=False)}
- Needs from you: {json.dumps(other.need_from_other, ensure_ascii=False)}

COORDINATION GUIDELINES:
1. HELP: If the peer needs something you have, add a step to prepare/deliver it.
2. REQUEST: If you need something the peer provides, add a step with "depends_on": [999].
   (999 is a placeholder meaning "I wait for the other agent")"""

    return f"""You are the {my.room_type} agent ({my.agent_id}).
Global task: "{task}"

{ctx}

INSTRUCTIONS:
1. Generate 4-6 steps for YOUR room only.
2. Cross-agent dependencies: Use "depends_on": [999] ONLY when you must wait for the peer.
3. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "plan_steps": [
    {{"step_id": 1, "time_min": 0, "action": "...", "depends_on": [], "uncertainty": 0.1, "notes": ""}},
    {{"step_id": 2, "time_min": 5, "action": "...", "depends_on": [999], "uncertainty": 0.1, "notes": "waiting for peer's item"}}
  ]
}}
</JSON>"""


def _stem(word: str) -> str:
    """간단한 stemming: 복수형 s 제거."""
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _resource_keywords(action: str) -> Set[str]:
    words = set(re.findall(r"\w+", action.lower())) - FUZZY_STOPWORDS
    return {_stem(w) for w in words}


def _parse_local_plan(
    raw: str, log_probs: List[float], my: Offer, step_offset: int = 0,
) -> LocalPlan:
    data      = extract_json(raw)
    raw_steps = data.get("plan_steps", [])
    if not isinstance(raw_steps, list):
        raw_steps = []

    token_unc     = compute_token_uncertainty(log_probs)
    steps:        List[PlanStep] = []
    hq_list:      List[HQEntry]  = []
    seen_ids:     set = set()
    seen_actions: set = set()

    for i, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip()
        if not action:
            continue

        # 중복 액션 제거
        akey = frozenset(_resource_keywords(action))
        if akey and akey in seen_actions:
            continue
        seen_actions.add(akey)

        raw_sid = safe_int(item.get("step_id", i), i)
        sid     = raw_sid + step_offset
        while sid in seen_ids:
            sid += 1
        seen_ids.add(sid)

        json_unc    = clamp01(item.get("uncertainty", 0.2))
        action_conf = max(
            (v for k, v in my.conf.items() if _fuzzy_match_soft(action, k)),
            default=0.7,
        )
        step_unc = clamp01(json_unc * 0.5 + token_unc * 0.2 + (1 - action_conf) * 0.3)

        # _parse_local_plan 함수 내부의 해당 루프를 아래와 같이 수정
        raw_deps = _norm_depends(item.get("depends_on"))
        deps = []
        for d in raw_deps:
            if d == 999:
                deps.append(999)  # 999는 상대방을 가리키는 특수 번호로 유지
            else:
                deps.append(d + step_offset)

        # time_min 파싱: step_id와 혼동 방지
        raw_time   = safe_int(item.get("time_min", 0), 0)
        if raw_time > 25 and raw_time == raw_sid:
            raw_time = 0  # step_id가 잘못 들어온 경우

        step = PlanStep(
            step_id       = sid,
            time_min      = max(0, min(30, raw_time)),
            room          = my.room_type,
            agent_id      = my.agent_id,
            action        = action,
            preconditions = [str(x).strip() for x in item.get("preconditions", [])
                             if str(x).strip()],
            depends_on    = deps,
            uncertainty   = step_unc,
            notes         = str(item.get("notes", "")).strip(),
        )
        steps.append(step)

        if step_unc >= UNCERTAINTY_THRESH:
            hq_list.append(HQEntry(sid, f"Is '{action}' feasible?", step_unc))

    steps.sort(key=lambda s: (s.time_min, s.step_id))
    all_unc = [s.uncertainty for s in steps] if steps else [token_unc]
    return LocalPlan(my.agent_id, steps, compute_plan_uncertainty(all_unc), hq_list)


def phase2_local_plan(
    offer_a: Offer, offer_b: Offer,
    img_a: str, img_b: str, task: str,
    use_offer: bool = True,
    verbose: str = "full",
) -> Tuple[LocalPlan, LocalPlan]:
    _banner("PHASE 2 — LOCAL PLANNING")
    prompt_a = _build_phase2_prompt(offer_a, offer_b, task, use_offer)
    prompt_b = _build_phase2_prompt(offer_b, offer_a, task, use_offer)

    results = _run_parallel([(img_a, prompt_a, True), (img_b, prompt_b, True)])
    raw_a, logp_a = results[0]
    raw_b, logp_b = results[1]

    plan_a = _parse_local_plan(raw_a, logp_a, offer_a, step_offset=0)
    plan_b = _parse_local_plan(raw_b, logp_b, offer_b, step_offset=AGENT_B_STEP_OFFSET)

    # [수정] 기존의 강제 Injection을 주석 처리합니다.
    # if use_offer:
    #     _banner("PHASE 2b — COORDINATION INJECTION (DEPRECATED)")
    #     plan_a, plan_b = _inject_coordination(plan_a, plan_b, offer_a, offer_b)

    return plan_a, plan_b
    

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: CONFLICT DETECTION (rule-based)
# ══════════════════════════════════════════════════════════════════════════════

def detect_conflicts(
    plan_a: LocalPlan, plan_b: LocalPlan,
    offer_a: Offer, offer_b: Offer,
) -> List[ConflictEntry]:
    conflicts: List[ConflictEntry] = []
    steps_a   = plan_a.steps
    steps_b   = plan_b.steps
    all_steps = [(s, offer_a) for s in steps_a] + [(s, offer_b) for s in steps_b]

    # ── 1. TEMPORAL ──────────────────────────────────────────────────────────
    time_slots: Dict[int, List[PlanStep]] = {}
    for s in steps_a + steps_b:
        time_slots.setdefault(s.time_min, []).append(s)

    for t, slot in time_slots.items():
        for i in range(len(slot)):
            for j in range(i + 1, len(slot)):
                si, sj = slot[i], slot[j]
                if si.agent_id == sj.agent_id or si.room != sj.room:
                    continue
                overlap = _resource_keywords(si.action) & _resource_keywords(sj.action)
                if overlap:
                    conflicts.append(ConflictEntry(
                        conflict_type = ConflictType.TEMPORAL,
                        step_ids      = [si.step_id, sj.step_id],
                        agent_ids     = [si.agent_id, sj.agent_id],
                        description   = (
                            f"T={t}m same room: step{si.step_id} & step{sj.step_id} "
                            f"share resource {overlap}"
                        ),
                        fix_hint = f"Shift one step's time_min away from T={t}m.",
                    ))

    # ── 2. DEPENDENCY ─────────────────────────────────────────────────────────
    # A가 can_provide하는데 B가 관련 스텝을 먼저 실행하는 경우 탐지
    all_step_ids_a = {s.step_id for s in steps_a}
    all_step_ids_b = {s.step_id for s in steps_b}

    for provide in offer_a.can_provide:
        pkw = _resource_keywords(provide)
        prep_steps_a = [s for s in steps_a if pkw & _resource_keywords(s.action)]
        use_steps_b  = [s for s in steps_b if pkw & _resource_keywords(s.action)]
        if not prep_steps_a or not use_steps_b:
            continue
        last_prep_time = max(s.time_min for s in prep_steps_a)
        last_prep_ids  = {s.step_id for s in prep_steps_a}
        for bs in use_steps_b:
            already_linked = any(d in last_prep_ids for d in bs.depends_on)
            if not already_linked and bs.time_min <= last_prep_time:
                conflicts.append(ConflictEntry(
                    conflict_type = ConflictType.DEPENDENCY,
                    step_ids      = [bs.step_id],
                    agent_ids     = ["agent_A", "agent_B"],
                    description   = (
                        f"B step{bs.step_id} '{bs.action[:40]}' (T={bs.time_min}m) "
                        f"uses '{provide}' prepared by A at T={last_prep_time}m, "
                        f"but runs before/without depends_on."
                    ),
                    fix_hint = (
                        f"Add depends_on={sorted(last_prep_ids)} to B step{bs.step_id} "
                        f"and set time_min > {last_prep_time}."
                    ),
                ))

    for provide in offer_b.can_provide:
        pkw = _resource_keywords(provide)
        prep_steps_b = [s for s in steps_b if pkw & _resource_keywords(s.action)]
        use_steps_a  = [s for s in steps_a if pkw & _resource_keywords(s.action)]
        if not prep_steps_b or not use_steps_a:
            continue
        last_prep_time = max(s.time_min for s in prep_steps_b)
        last_prep_ids  = {s.step_id for s in prep_steps_b}
        for as_ in use_steps_a:
            already_linked = any(d in last_prep_ids for d in as_.depends_on)
            if not already_linked and as_.time_min <= last_prep_time:
                conflicts.append(ConflictEntry(
                    conflict_type = ConflictType.DEPENDENCY,
                    step_ids      = [as_.step_id],
                    agent_ids     = ["agent_A", "agent_B"],
                    description   = (
                        f"A step{as_.step_id} '{as_.action[:40]}' (T={as_.time_min}m) "
                        f"uses '{provide}' prepared by B at T={last_prep_time}m."
                    ),
                    fix_hint = (
                        f"Add depends_on={sorted(last_prep_ids)} to A step{as_.step_id} "
                        f"and set time_min > {last_prep_time}."
                    ),
                ))

    # ── 3. REDUNDANCY (inter-agent) ───────────────────────────────────────────
    for sa in steps_a:
        for sb in steps_b:
            if _fuzzy_match(sa.action, sb.action, min_overlap=3):
                conflicts.append(ConflictEntry(
                    conflict_type = ConflictType.REDUNDANCY,
                    step_ids      = [sa.step_id, sb.step_id],
                    agent_ids     = ["agent_A", "agent_B"],
                    description   = (
                        f"Duplicate: A-step{sa.step_id} '{sa.action[:35]}' "
                        f"~= B-step{sb.step_id} '{sb.action[:35]}'"
                    ),
                    fix_hint = "Delete one of the duplicate steps.",
                ))

    # ── 4. REDUNDANCY (intra-agent) ───────────────────────────────────────────
    for agent_steps in [steps_a, steps_b]:
        for i in range(len(agent_steps)):
            for j in range(i + 1, len(agent_steps)):
                si, sj = agent_steps[i], agent_steps[j]
                if _fuzzy_match(si.action, sj.action, min_overlap=3):
                    conflicts.append(ConflictEntry(
                        conflict_type = ConflictType.REDUNDANCY,
                        step_ids      = [si.step_id, sj.step_id],
                        agent_ids     = [si.agent_id],
                        description   = (
                            f"Intra-agent duplicate ({si.agent_id}): "
                            f"step{si.step_id} ~= step{sj.step_id}"
                        ),
                        fix_hint = f"Delete step{sj.step_id} (keep step{si.step_id}).",
                    ))

    # ── 5. CANNOT_DO ──────────────────────────────────────────────────────────
    for step, offer in all_steps:
        for c in offer.cannot_do:
            if _fuzzy_match(step.action, c.action, min_overlap=2):
                conflicts.append(ConflictEntry(
                    conflict_type = ConflictType.CANNOT_DO,
                    step_ids      = [step.step_id],
                    agent_ids     = [step.agent_id],
                    description   = (
                        f"{step.agent_id} step{step.step_id} '{step.action[:40]}' "
                        f"violates cannot_do '{c.action}'"
                    ),
                    fix_hint = f"Delete or reassign step{step.step_id}.",
                ))

    # ── 6. OBSERVABILITY ──────────────────────────────────────────────────────
    for step, offer in all_steps:
        scope_kw  = set(re.findall(r"\w+", offer.obs_scope.lower()))
        can_do_kw: Set[str] = set()
        for cd in offer.can_do:
            can_do_kw |= _resource_keywords(cd)
        pool = scope_kw | can_do_kw
        act_kw = _resource_keywords(step.action)
        if act_kw and pool and not (act_kw & pool):
            conflicts.append(ConflictEntry(
                conflict_type = ConflictType.OBSERV,
                step_ids      = [step.step_id],
                agent_ids     = [step.agent_id],
                description   = (
                    f"{step.agent_id} step{step.step_id} '{step.action[:40]}' "
                    f"references objects outside observable scope"
                ),
                fix_hint = f"Modify or delete step{step.step_id}.",
            ))

    return conflicts



# ── 7. MISSING/PLACEHOLDER DEPENDENCY (추가) ──────────────────────────
    # [999] 플레이스홀더가 남아있는 경우 갈등으로 표시하여 수정을 유도
    for step in plan_a.steps + plan_b.steps:
        if 999 in step.depends_on:
            conflicts.append(ConflictEntry(
                conflict_type = "DEPENDENCY_MISSING",
                step_ids      = [step.step_id],
                agent_ids     = [step.agent_id],
                description   = (
                    f"{step.agent_id} step{step.step_id} has a placeholder [999]. "
                    f"Must specify a real step ID from the peer agent's plan."
                ),
                fix_hint = "Identify which peer step you are waiting for and update depends_on."
            ))

    return conflicts

# ── 헬퍼 함수 추가 (Phase 4에서 사용됨) ──────────────────────────────────
def detect_conflicts_from_dicts(steps_a: List[Dict], steps_b: List[Dict], offer_a: Offer, offer_b: Offer):
    """
    협상 중에 사전(dict) 형태의 플랜 데이터를 받아 갈등을 체크하기 위한 브릿지 함수
    """
    # PlanStep 객체로 복원하여 기존 detect_conflicts 호출
    tmp_a = LocalPlan("agent_A", [PlanStep(**s) for s in steps_a], 0.0, [])
    tmp_b = LocalPlan("agent_B", [PlanStep(**s) for s in steps_b], 0.0, [])
    return detect_conflicts(tmp_a, tmp_b, offer_a, offer_b)
    


def phase3_conflict_detection(
    plan_a: LocalPlan, plan_b: LocalPlan,
    offer_a: Offer, offer_b: Offer,
    verbose: str = "full",
) -> List[ConflictEntry]:
    _banner("PHASE 3 — CONFLICT DETECTION")
    conflicts = detect_conflicts(plan_a, plan_b, offer_a, offer_b)

    if not conflicts:
        print("  ✓ No conflicts detected.")
    else:
        by_type: Dict[str, List[ConflictEntry]] = {}
        for c in conflicts:
            by_type.setdefault(c.conflict_type, []).append(c)
        for ctype, clist in by_type.items():
            print(f"\n  [{ctype}] ×{len(clist)}")
            if verbose in ("full", "summary"):
                for c in clist:
                    print(f"    {c.description}")
                    if c.fix_hint:
                        print(f"    → {c.fix_hint}")

    return conflicts


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: P2P NEGOTIATION
# ══════════════════════════════════════════════════════════════════════════════

def _build_negotiation_prompt(
    my_agent: str, my_offer: Offer,
    cur_a: List[Dict], cur_b: List[Dict],
    conflicts: List[ConflictEntry],
    locked: Set[int], round_num: int,
    prev_props: List[NegotiationProposal],
    task: str,
) -> str:
    other_id   = "agent_B" if my_agent == "agent_A" else "agent_A"
    my_plan    = cur_a if my_agent == "agent_A" else cur_b
    other_plan = cur_b if my_agent == "agent_A" else cur_a

    # 갈등 목록 텍스트화
    c_text = "\n".join(
        f"  [{c.conflict_type}] {c.description}"
        + (f"\n    → FIX HINT: {c.fix_hint}" if c.fix_hint else "")
        for c in conflicts
    ) or "  (none)"

    return f"""You are {my_agent} ({my_offer.room_type}). NEGOTIATION ROUND {round_num}/{MAX_NEGOTIATION_ROUNDS}.
Task: "{task}"

YOUR PLAN: {json.dumps(my_plan, ensure_ascii=False)}
{other_id}'s PLAN: {json.dumps(other_plan, ensure_ascii=False)}

ACTIVE CONFLICTS:
{c_text}

NEGOTIATION RULES:
1. RESOLVE 999: If your step has "depends_on": [999], find the relevant step ID from {other_id}'s PLAN and update it.
2. ALIGN TIME: If you depend on {other_id}'s step, your "time_min" MUST be greater than theirs.
3. FIX CONFLICTS:
   - TEMPORAL: Change time_min to avoid same-room/same-time overlaps.
   - REDUNDANCY: Delete duplicate steps using field="delete".
4. ACCEPTANCE: To agree with {other_id}'s previous proposal, use reason="ACCEPT".

OUTPUT FORMAT:
Return ONLY valid JSON with a "proposals" list.

<JSON>
{{
  "proposals": [
    {{
      "step_id": 102,
      "agent_id": "{my_agent}",
      "field": "depends_on",
      "new_value": "[5]",
      "reason": "Replaced 999 with {other_id}'s step 5"
    }}
  ]
}}
</JSON>"""


def _parse_proposals(raw: str, my_agent: str) -> List[NegotiationProposal]:
    data   = extract_json(raw)
    result = []
    for item in data.get("proposals", []):
        if not isinstance(item, dict):
            continue
        sid       = safe_int(item.get("step_id", -1), -1)
        agent_id  = str(item.get("agent_id", my_agent)).strip()
        field     = str(item.get("field", "")).strip().lower()
        new_value = str(item.get("new_value", "")).strip()
        reason    = str(item.get("reason", "")).strip()
        if sid < 0 or field not in VALID_PROPOSAL_FIELDS or not new_value:
            continue
        if agent_id not in VALID_AGENTS:
            agent_id = my_agent
        result.append(NegotiationProposal(sid, agent_id, field, new_value, reason))
    return result


def _apply_proposal(
    cur_a: List[Dict], cur_b: List[Dict],
    prop: NegotiationProposal, locked: Set[int],
) -> bool:
    if prop.step_id in locked:
        return False
    plan    = cur_a if prop.agent_id == "agent_A" else cur_b
    sid_map = {s["step_id"]: i for i, s in enumerate(plan)}
    if prop.step_id not in sid_map:
        return False
    idx = sid_map[prop.step_id]

    if prop.field == "delete":
        plan.pop(idx)
        return True
    if prop.field == "time_min":
        t = safe_int(prop.new_value, -1)
        if 0 <= t <= 30:
            plan[idx]["time_min"] = t
            return True
    elif prop.field == "action":
        if prop.new_value:
            plan[idx]["action"] = prop.new_value
            return True
# _apply_proposal 함수 내 depends_on 처리 부분 수정
    elif prop.field == "depends_on":
        try:
            # "[5]" 혹은 "5" 형태 모두 처리
            val = prop.new_value.strip()
            if val.startswith("[") and val.endswith("]"):
                deps = json.loads(val)
            else:
                deps = [int(val)]
            
            if isinstance(deps, list):
                plan[idx]["depends_on"] = [int(d) for d in deps if d != 999]
                return True
        except Exception:
            pass
    return False


def _lock_steps(
    props_a: List[NegotiationProposal],
    props_b: List[NegotiationProposal],
    conflict_sids: Set[int],
    existing: Set[int],
) -> Set[int]:
    acc_b  = {p.step_id for p in props_b if p.reason.upper() == "ACCEPT"}
    acc_a  = {p.step_id for p in props_a if p.reason.upper() == "ACCEPT"}
    prop_a = {p.step_id for p in props_a if p.reason.upper() != "ACCEPT"}
    prop_b = {p.step_id for p in props_b if p.reason.upper() != "ACCEPT"}
    agreed     = (prop_a & acc_b) | (prop_b & acc_a)
    mentioned  = {p.step_id for p in props_a + props_b}
    uncontested = conflict_sids - mentioned
    return existing | agreed | uncontested


def phase4_negotiation(
    plan_a: LocalPlan, plan_b: LocalPlan,
    offer_a: Offer, offer_b: Offer,
    conflicts: List[ConflictEntry],
    img_a: str, img_b: str, task: str,
    verbose: str = "full",
) -> Tuple[List[Dict], List[Dict], List[NegotiationRound]]:
    _banner("PHASE 4 — P2P NEGOTIATION")

    if not conflicts:
        print("  No conflicts → skip.")
        return plan_steps_to_dicts(plan_a.steps), plan_steps_to_dicts(plan_b.steps), []

    conflict_sids: Set[int] = {sid for c in conflicts for sid in c.step_ids}
    print(f"  Conflict step IDs: {sorted(conflict_sids)}")
    print(f"  Max rounds: {MAX_NEGOTIATION_ROUNDS}")

    cur_a: List[Dict] = plan_steps_to_dicts(plan_a.steps)
    cur_b: List[Dict] = plan_steps_to_dicts(plan_b.steps)
    locked: Set[int]  = set()
    rounds: List[NegotiationRound] = []
    prev_a: List[NegotiationProposal] = []
    prev_b: List[NegotiationProposal] = []
    last_val: Dict[Tuple[int, str], str] = {}  # 핑퐁 방지

    for rnd in range(1, MAX_NEGOTIATION_ROUNDS + 1):
        remaining = [
            c for c in conflicts
            if not c.step_ids or not all(sid in locked for sid in c.step_ids)
        ]
        if not remaining:
            print(f"\n  Round {rnd}: all resolved → stop early.")
            break

        print(f"\n  -- Round {rnd}/{MAX_NEGOTIATION_ROUNDS} "
              f"(remaining={len(remaining)}, locked={sorted(locked)}) --")

        prompt_a = _build_negotiation_prompt(
            "agent_A", offer_a, cur_a, cur_b, remaining, locked, rnd, prev_b, task)
        prompt_b = _build_negotiation_prompt(
            "agent_B", offer_b, cur_a, cur_b, remaining, locked, rnd, prev_a, task)

        results  = _run_parallel([(img_a, prompt_a, False), (img_b, prompt_b, False)])
        raw_a, _ = results[0]
        raw_b, _ = results[1]

        props_a = _parse_proposals(raw_a, "agent_A")
        props_b = _parse_proposals(raw_b, "agent_B")

        # 핑퐁 필터
        def _filter(props: List[NegotiationProposal]) -> List[NegotiationProposal]:
            out = []
            for p in props:
                key = (p.step_id, p.field)
                if last_val.get(key) == p.new_value:
                    continue
                out.append(p)
                last_val[key] = p.new_value
            return out

        props_a = _filter(props_a)
        props_b = _filter(props_b)

        if verbose in ("full", "summary"):
            for p in props_a:
                print(f"  [A→{p.agent_id}] step{p.step_id} .{p.field}="
                      f"'{p.new_value[:40]}' ({p.reason[:50]})")
            for p in props_b:
                print(f"  [B→{p.agent_id}] step{p.step_id} .{p.field}="
                      f"'{p.new_value[:40]}' ({p.reason[:50]})")

        for prop in props_a + props_b:
            changed = _apply_proposal(cur_a, cur_b, prop, locked)
            if changed and verbose in ("full", "summary"):
                print(f"  [APPLIED {prop.agent_id[:1]}] step{prop.step_id}.{prop.field}")

        locked = _lock_steps(props_a, props_b, conflict_sids, locked)
        rounds.append(NegotiationRound(rnd, props_a, props_b, sorted(locked)))
        print(f"  → Locked: {sorted(locked)}")
        prev_a, prev_b = props_a, props_b

    print(f"\n  Negotiation complete. Total rounds: {len(rounds)}")
    return cur_a, cur_b, rounds


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: CONVERGENCE CHECK (rule-based)
# ══════════════════════════════════════════════════════════════════════════════

def _has_cycle(steps: List[Dict]) -> bool:
    indegree = {s["step_id"]: 0 for s in steps}
    adj: Dict[int, List[int]] = {s["step_id"]: [] for s in steps}
    for s in steps:
        for dep in s.get("depends_on", []):
            if dep in adj:
                adj[dep].append(s["step_id"])
                indegree[s["step_id"]] += 1
    q = deque(sid for sid, d in indegree.items() if d == 0)
    visited = 0
    while q:
        node = q.popleft()
        visited += 1
        for nxt in adj[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)
    return visited != len(steps)


def phase5_convergence_check(
    steps_a: List[Dict], steps_b: List[Dict],
    offer_a: Offer, offer_b: Offer,
    conflicts: List[ConflictEntry],
) -> ConvergenceResult:
    _banner("PHASE 5 — CONVERGENCE CHECK")
    all_steps = steps_a + steps_b

    # 1. Dependency Cycle 체크 (기존 유지)
    no_cycle = not _has_cycle(all_steps)

    # 2. Observability 체크 (기존 유지)
    # ... (기존 obs_ok 로직) ...

    # 3. [추가/수정] Placeholder & Dependency 체크
    # 아직 [999]가 남아있거나, 필요한 의존성이 비어있는지 확인합니다.
    missing_deps: List[int] = []
    has_placeholder: List[int] = []
    
    for s in all_steps:
        deps = s.get("depends_on", [])
        if 999 in deps:
            has_placeholder.append(s["step_id"])
        
        # 만약 Conflict 리스트에 해당 스텝의 DEPENDENCY 갈등이 남아있다면 FAIL 처리
        if any(c.conflict_type == ConflictType.DEPENDENCY and s["step_id"] in c.step_ids for c in conflicts):
            missing_deps.append(s["step_id"])

    no_placeholder = len(has_placeholder) == 0
    no_missing = len(missing_deps) == 0

    # 4. 최종 수렴 판단
    unresolved = [c for c in conflicts 
                  if c.conflict_type in (ConflictType.REDUNDANCY, ConflictType.CANNOT_DO, ConflictType.TEMPORAL)]
    
    # 모든 조건이 만족되어야 수렴(Converged)으로 인정
    converged = no_cycle and obs_ok and no_missing and no_placeholder and not unresolved

    print(f"  No dep cycle    : {'OK' if no_cycle else 'FAIL'}")
    print(f"  Observability   : {'OK' if obs_ok else 'FAIL'}")
    print(f"  No placeholders : {'OK' if no_placeholder else f'FAIL (steps={has_placeholder})'}")
    print(f"  No missing deps : {'OK' if no_missing else f'FAIL (steps={missing_deps})'}")
    print(f"  Residual Confl. : {'OK' if not unresolved else f'FAIL ({len(unresolved)} remaining)'}")
    print(f"  → Converged     : {'YES ✓' if converged else 'NO ✗'}")

    return ConvergenceResult(
        converged            = converged,
        no_dep_cycle         = no_cycle,
        observability_ok     = obs_ok,
        no_missing_deps      = no_missing and no_placeholder,
        unresolved_conflicts = unresolved,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6: DEFERRED HUMAN QUERY (VLM 기반, 수렴 실패 시에만)
# ══════════════════════════════════════════════════════════════════════════════

_HQ_TEMPLATES: Dict[str, str] = {
    "DEP_CYCLE":    "A dependency cycle was detected. Which step should be reordered?",
    "DEPENDENCY":   "A step is waiting for the peer (ID 999) but the specific link is unclear. Which peer step is it waiting for?", # 수정
    "REDUNDANCY":   "Two agents are doing the same task. Which should handle it?",
    "CANNOT_DO":    "An agent planned something outside its capability. Should it be removed or reassigned?",
    "UNMATCHED":    "An agent needs something no one can provide. How should this be handled?",
    "OBSERVABILITY":"A step references objects outside the agent's visible scope. Modify or remove?",
}

def _generate_hq_question(
    trigger_type: str, detail: str,
    offer_a: Offer, offer_b: Offer, img: str,
) -> str:
    template = _HQ_TEMPLATES.get(trigger_type, "How should the agents handle this issue?")
    prompt = f"""You are an expert home robotics coordinator. 
Looking at the scene, there is a coordination failure between Agent A and B.

CONTEXT:
- Agent A ({offer_a.room_type})
- Agent B ({offer_b.room_type})
- Issue: {detail}

TASK:
Based on the image, write a BRIEF question (max 15 words) to the human to resolve this.
Example: "Should Agent A wait for Agent B to bring the tray before cleaning?"
No preamble. Just the question."""
    # ... (이하 run_vlm 호출 로직 유지) ...
    try:
        q, _ = run_vlm(img, prompt)
        q = q.strip().strip('"').strip("'")
        if 10 < len(q) < 400:
            return q
    except Exception as e:
        print(f"  [HQ VLM error] {e}")
    return f"{template}\nContext: {detail[:100]}"


def phase6_human_query(
    plan_a: LocalPlan, plan_b: LocalPlan,
    offer_a: Offer, offer_b: Offer,
    convergence: ConvergenceResult,
    img_a: str, img_b: str,
    use_human_query: bool = True,
) -> Tuple[Dict[str, str], List[str], List[str]]:
    _banner("PHASE 6 — DEFERRED HUMAN QUERY")

    if not use_human_query:
        print("  [ABLATION] disabled.")
        return {}, [], []

    if convergence.converged and not convergence.unresolved_conflicts:
        print("  Plan converged → no query needed.")
        return {}, [], []

    raw_triggers: List[Tuple[str, str, float]] = []
    triggered:    List[str] = []

    if not convergence.no_dep_cycle:
        d = "Dependency cycle detected in joint plan."
        triggered.append(f"[DEP_CYCLE] {d}")
        raw_triggers.append(("DEP_CYCLE", d, 0.90))

    if not convergence.no_missing_deps:
        # 999 플레이스홀더가 남아있는지 체크하여 더 구체적인 메시지 생성
        d = "Some steps still have placeholder '999' dependencies that were not resolved during negotiation."
        triggered.append(f"[DEPENDENCY] {d}")
        raw_triggers.append(("DEPENDENCY", d, 0.95)) # 우선순위를 높게 설정

    if not convergence.observability_ok:
        d = "Some steps reference objects outside the agent's visible scope."
        triggered.append(f"[OBSERVABILITY] {d}")
        raw_triggers.append(("OBSERVABILITY", d, 0.75))

    for c in convergence.unresolved_conflicts:
        triggered.append(f"[{c.conflict_type}] {c.description}")
        raw_triggers.append((c.conflict_type, c.description, 0.80))

    all_provides = offer_a.can_provide + offer_b.can_provide
    for need in offer_a.need_from_other + offer_b.need_from_other:
        if not any(_fuzzy_match_soft(need, p) for p in all_provides):
            d = f"No agent can provide: '{need}'"
            triggered.append(f"[UNMATCHED] {d}")
            raw_triggers.append(("UNMATCHED", d, 0.85))

    if not triggered:
        print("  No query needed.")
        return {}, [], []

    print(f"  Triggers ({len(triggered)}):")
    for t in triggered:
        print(f"    {t}")

    raw_triggers.sort(key=lambda x: -x[2])
    answers: Dict[str, str] = {}
    asked:   List[str]      = []

    for i, (ttype, detail, pri) in enumerate(raw_triggers[:HQ_TOP_K], 1):
        print(f"\n  Generating Q{i} [{ttype}, priority={pri:.2f}]...", end=" ", flush=True)
        q = _generate_hq_question(ttype, detail, offer_a, offer_b, img_a)
        print("done")
        print(f"  Q{i}: {q}")
        asked.append(q)

        if AUTO_HQ_ANSWER is not None:
            ans = AUTO_HQ_ANSWER
            print(f"  A (auto): {ans}")
        else:
            try:
                ans = input("  A: ").strip()
            except EOFError:
                ans = ""

        if ans:
            answers[q] = ans

    return answers, triggered, asked


# ══════════════════════════════════════════════════════════════════════════════
# FINALIZE: RULE-BASED MERGE (LLM 없음)
# ══════════════════════════════════════════════════════════════════════════════

def phase_finalize(
    steps_a: List[Dict], steps_b: List[Dict],
    offer_a: Offer, offer_b: Offer,
    human_answers: Dict[str, str],
    convergence: ConvergenceResult,
    verbose: str = "full",
) -> List[Dict]:
    _banner("FINALIZE — RULE-BASED MERGE")

    # 1. [보강] Human Answer 반영 로직
    # 인간의 답변(human_answers)에 특정 액션에 대한 지시가 있다면 계획을 수정합니다.
    # 예: "Agent A should wait for B" -> B의 특정 스텝을 찾아 A의 depends_on에 강제 주입
    if human_answers:
        for q, a in human_answers.items():
            a_lower = a.lower()
            if "yes" in a_lower or "agree" in a_lower:
                # 999 플레이스홀더가 남아있는 스텝들을 찾아 적절히 처리 (간단한 휴리스틱)
                for s in steps_a + steps_b:
                    if 999 in s.get("depends_on", []):
                        # 999를 제거하여 최소한 계획이 멈추지 않게 함
                        s["depends_on"] = [d for d in s["depends_on"] if d != 999]
                        s["notes"] = (s.get("notes", "") + " [Resolved by Human]").strip()

    # 2. [추가] Safety-net: 정렬 전 남아있는 모든 999 제거
    # 재번호 매기기(old_to_new) 시 999가 있으면 에러가 발생할 수 있으므로 미리 청소합니다.
    for s in steps_a + steps_b:
        if "depends_on" in s:
            s["depends_on"] = [d for d in s["depends_on"] if d != 999]

    # 3. 기존 병합 및 정렬 로직 (동일)
    merged = list(steps_a) + list(steps_b)
    merged.sort(key=lambda s: (s.get("time_min", 0), s.get("step_id", 0)))

    # 4. ID 재매핑 (기존 유지)
    old_to_new: Dict[int, int] = {}
    for new_id, s in enumerate(merged, start=1):
        old_to_new[s["step_id"]] = new_id

    for s in merged:
        s["step_id"] = old_to_new[s["step_id"]]
        # d가 old_to_new에 없는 경우(999 등)는 위에서 이미 처리했으므로 안전함
        s["depends_on"] = [old_to_new[d] for d in s.get("depends_on", []) if d in old_to_new]
        
        new_preconds = []
        for p in s.get("preconditions", []):
            m = re.match(r"step (\d+) completed", p)
            if m and int(m.group(1)) in old_to_new:
                new_preconds.append(f"step {old_to_new[int(m.group(1))]} completed")
            else:
                new_preconds.append(p)
        s["preconditions"] = new_preconds

    if verbose in ("full", "summary"):
        print(f"\n  {len(steps_a)} A-steps + {len(steps_b)} B-steps = {len(merged)} total")
    return merged
