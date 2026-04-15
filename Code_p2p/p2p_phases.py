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
    return f"""You are an embodied home agent. Observe your room image carefully.

Global task: "{task}"

{_PHASE1_PROMPT_EXAMPLE}

Generate an Offer for YOUR room only.

RULES:
1. Use ONLY objects actually visible in the image.
2. can_do: max {MAX_CAN_DO} actions. Format: "<verb> <specific visible object>"
3. cannot_do: max {MAX_CANNOT_DO}. reason: NO_OBJECT | NO_CAPABILITY | UNCERTAIN
4. can_do and cannot_do must NOT overlap.
5. conf: confidence [0.0-1.0] for each can_do item.
6. can_provide: results or items you can prepare/deliver that the other agent needs.
   Only include tangible items (food, drink, objects) or clear status notifications.
   Do NOT include room conditions like "cleaned sink" or "wiped counter".
7. need_from_other: what you specifically need from the other agent.
8. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "room_type": "...",
  "observation": "one concise sentence",
  "obs_scope": "comma-separated list of visible objects and areas",
  "can_do": ["..."],
  "cannot_do": [{{"action": "...", "reason": "NO_OBJECT"}}],
  "conf": {{"action text": 0.9}},
  "can_provide": ["..."],
  "need_from_other": ["..."]
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
    if use_offer:
        # offer 매칭으로 coordination 기회 파악
        i_provide = [p for p in my.can_provide
                     if any(_fuzzy_match_soft(p, n) for n in other.need_from_other)
                     or any(_resource_keywords(p) & _resource_keywords(cd)
                            for cd in other.can_do)]
        other_provides = [p for p in other.can_provide
                          if any(_fuzzy_match_soft(p, n) for n in my.need_from_other)
                          or any(_resource_keywords(p) & _resource_keywords(cd)
                                 for cd in my.can_do)]

        ctx = f"""YOUR OFFER:
- room: {my.room_type}
- obs_scope: {my.obs_scope}
- can_do: {json.dumps(my.can_do, ensure_ascii=False)}
- can_provide: {json.dumps(my.can_provide, ensure_ascii=False)}
- need_from_other: {json.dumps(my.need_from_other, ensure_ascii=False)}

OTHER AGENT ({other.room_type}):
- can_provide: {json.dumps(other.can_provide, ensure_ascii=False)}
- need_from_other: {json.dumps(other.need_from_other, ensure_ascii=False)}

COORDINATION CONTEXT:
- What YOU will prepare for the other agent: {json.dumps(i_provide, ensure_ascii=False)}
- What the other agent will prepare for YOU: {json.dumps(other_provides, ensure_ascii=False)}"""
    else:
        ctx = f"YOUR ROOM: {my.room_type}\nOTHER AGENT ROOM: {other.room_type}"

    return f"""You are the {my.room_type} agent ({my.agent_id}).

Global task: "{task}"

{ctx}

{_PHASE2_EXAMPLE}

Generate YOUR local plan. Follow these rules:
1. Steps ONLY in YOUR room ({my.room_type}), using ONLY visible objects.
2. Generate 4-6 steps over 0-25 minutes. No repeated actions.
3. Do NOT include any handoff_type or target_agent fields.
4. Just plan your own actions. Cross-agent coordination will be handled separately.
5. uncertainty: 0.0=certain, 1.0=very uncertain.
6. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "plan_steps": [
    {{"step_id": 1, "time_min": 0, "action": "verb + specific object",
      "preconditions": [], "depends_on": [], "uncertainty": 0.1, "notes": ""}}
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

        raw_deps = _norm_depends(item.get("depends_on"))
        deps     = [d + step_offset for d in raw_deps]

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


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2b: COORDINATION INJECTION (rule-based)
# 핵심: offer 매칭으로 cross-agent depends_on을 자동 삽입
# LLM은 액션만 생성하고, 코드가 협력 구조를 보장한다
# ══════════════════════════════════════════════════════════════════════════════

def _inject_coordination(
    plan_a: LocalPlan, plan_b: LocalPlan,
    offer_a: Offer, offer_b: Offer,
) -> Tuple[LocalPlan, LocalPlan]:
    """
    A→B, B→A 두 방향으로 coordination을 삽입한다.

    패턴:
      sender.can_provide ↔ receiver.need_from_other 매칭 시
      → receiver 플랜의 관련 스텝에 depends_on + time_min 조정
      → sender 플랜에 notify 스텝 추가 (선택적)
    """

    def _inject_one_direction(
        sender_plan: LocalPlan, receiver_plan: LocalPlan,
        sender_offer: Offer, receiver_offer: Offer,
        sender_id: str, receiver_id: str,
    ) -> Tuple[LocalPlan, LocalPlan]:

        for provide in sender_offer.can_provide:
            provide_kw = _resource_keywords(provide)

            # 1. receiver가 이것을 필요로 하는지 확인 (여러 방법으로)
            need_match = any(_fuzzy_match_soft(provide, n) for n in receiver_offer.need_from_other)
            can_do_match = any(provide_kw & _resource_keywords(cd) for cd in receiver_offer.can_do)
            if not need_match and not can_do_match and not provide_kw:
                continue

            # 2. sender 플랜에서 이 provide와 관련된 마지막 prep step 찾기
            prep_steps = [s for s in sender_plan.steps if provide_kw & _resource_keywords(s.action)]
            if not prep_steps:
                prep_steps = list(sender_plan.steps)
            if not prep_steps:
                continue
            last_prep = max(prep_steps, key=lambda s: s.time_min)

            # 3. receiver 플랜에서 연결할 스텝 찾기 (4단계 탐색)
            candidates = [s for s in receiver_plan.steps]

            # 1단계: keyword 직접 겹침
            targets = [s for s in candidates if provide_kw & _resource_keywords(s.action)]

            # 2단계: need_from_other fuzzy match
            if not targets:
                targets = [s for s in candidates
                           if any(_fuzzy_match_soft(s.action, n)
                                  for n in receiver_offer.need_from_other)]

            # 3단계: food/item 관련 배치 동사
            if not targets:
                _FOOD_KW = {"snack", "food", "drink", "item", "tray", "bowl",
                            "cup", "plate", "fruit", "bread", "water", "bottle"}
                _PLACE_VERBS = {"place", "put", "lay", "arrange", "bring",
                                "serve", "deliver", "receive", "set"}
                if provide_kw & _FOOD_KW:
                    targets = [s for s in candidates
                               if set(re.findall(r"\w+", s.action.lower())) & _PLACE_VERBS
                               and _resource_keywords(s.action) & _FOOD_KW]

            # 4단계: last_prep 이후 시간의 첫 번째 스텝
            if not targets:
                later = [s for s in candidates if s.time_min >= last_prep.time_min]
                if later:
                    targets = [min(later, key=lambda s: s.time_min)]

            if not targets:
                continue

            # 4. 연결: sender의 last_prep step_id를 receiver target의 depends_on에 추가
            coord_time = last_prep.time_min
            linked = False
            for rs in targets:
                if last_prep.step_id in rs.depends_on:
                    linked = True
                    continue
                rs.depends_on = sorted(set(rs.depends_on + [last_prep.step_id]))
                if rs.time_min <= coord_time:
                    rs.time_min = coord_time + 1
                print(f"  [COORD] {receiver_id} step{rs.step_id} "
                      f"'{rs.action[:45]}'\n"
                      f"          ← depends on {sender_id} step{last_prep.step_id} "
                      f"'{last_prep.action[:35]}' (T={rs.time_min}m)")
                linked = True

        return sender_plan, receiver_plan

    # A→B
    plan_a, plan_b = _inject_one_direction(
        plan_a, plan_b, offer_a, offer_b, "agent_A", "agent_B"
    )
    # B→A (A가 필요한 것을 B가 제공하는 경우)
    plan_b, plan_a = _inject_one_direction(
        plan_b, plan_a, offer_b, offer_a, "agent_B", "agent_A"
    )
    return plan_a, plan_b


def phase2_local_plan(
    offer_a: Offer, offer_b: Offer,
    img_a: str, img_b: str, task: str,
    use_offer: bool = True,
    verbose: str = "full",
) -> Tuple[LocalPlan, LocalPlan]:
    _banner("PHASE 2 — LOCAL PLANNING")
    prompt_a = _build_phase2_prompt(offer_a, offer_b, task, use_offer)
    prompt_b = _build_phase2_prompt(offer_b, offer_a, task, use_offer)

    results       = _run_parallel([(img_a, prompt_a, True), (img_b, prompt_b, True)])
    raw_a, logp_a = results[0]
    raw_b, logp_b = results[1]

    if verbose == "full":
        _log("A RAW PLAN", raw_a)
        _log("B RAW PLAN", raw_b)

    plan_a = _parse_local_plan(raw_a, logp_a, offer_a, step_offset=0)
    plan_b = _parse_local_plan(raw_b, logp_b, offer_b, step_offset=AGENT_B_STEP_OFFSET)

    # Coordination Injection
    if use_offer:
        _banner("PHASE 2b — COORDINATION INJECTION (rule-based)")
        plan_a, plan_b = _inject_coordination(plan_a, plan_b, offer_a, offer_b)

    if verbose in ("full", "summary"):
        _log("PLAN A", jdump(local_plan_to_dict(plan_a)))
        _log("PLAN B", jdump(local_plan_to_dict(plan_b)))

    cross_a = sum(1 for s in plan_a.steps if any(d >= AGENT_B_STEP_OFFSET for d in s.depends_on))
    cross_b = sum(1 for s in plan_b.steps if any(d < AGENT_B_STEP_OFFSET for d in s.depends_on))
    print(f"\n  A: steps={len(plan_a.steps)} U={plan_a.U_plan:.3f} cross_deps={cross_a}")
    print(f"  B: steps={len(plan_b.steps)} U={plan_b.U_plan:.3f} cross_deps={cross_b}")
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

    c_text = "\n".join(
        f"  [{c.conflict_type}] {c.description}"
        + (f"\n    → FIX HINT: {c.fix_hint}" if c.fix_hint else "")
        for c in conflicts
    ) or "  (none)"

    prev_text = "\n".join(
        f"  step{p.step_id}[{p.agent_id}] .{p.field}='{p.new_value}' ({p.reason})"
        for p in prev_props
    ) or "  (none)"

    return f"""You are {my_agent} ({my_offer.room_type}). NEGOTIATION ROUND {round_num}/{MAX_NEGOTIATION_ROUNDS}.
Task: "{task}"

YOUR PLAN:
{jdump(my_plan)}

{other_id}'s PLAN:
{jdump(other_plan)}

CONFLICTS TO RESOLVE:
{c_text}

LOCKED steps (do not modify): {sorted(locked) or '(none)'}

{other_id}'s previous proposals:
{prev_text}

INSTRUCTIONS:
1. Propose ONE fix per conflict. Prefer fixing YOUR OWN steps first.
2. DEPENDENCY: adjust time_min so dependent step runs AFTER the step it depends on.
   Add depends_on=[<prep_step_id>] to the dependent step.
3. REDUNDANCY: delete one of the duplicate steps.
4. TEMPORAL: shift time_min to avoid overlap.
5. OBSERVABILITY: delete or modify the step.
6. To accept other agent's proposal: reason="ACCEPT".
7. Do NOT modify locked steps.
8. Allowed fields: "time_min" | "action" | "depends_on" | "delete"
9. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "proposals": [
    {{
      "step_id": 103,
      "agent_id": "agent_B",
      "field": "depends_on",
      "new_value": "[2]",
      "reason": "DEPENDENCY: B step103 must wait for A step2 to complete"
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
    elif prop.field == "depends_on":
        try:
            deps = json.loads(prop.new_value)
            if isinstance(deps, list):
                plan[idx]["depends_on"] = [int(d) for d in deps]
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

    # 조건 1: dependency cycle 없음
    no_cycle = not _has_cycle(all_steps)

    # 조건 2: observability
    scope_a   = set(re.findall(r"\w+", offer_a.obs_scope.lower()))
    scope_b   = set(re.findall(r"\w+", offer_b.obs_scope.lower()))
    can_kw_a: Set[str] = set()
    can_kw_b: Set[str] = set()
    for cd in offer_a.can_do:
        can_kw_a |= _resource_keywords(cd)
    for cd in offer_b.can_do:
        can_kw_b |= _resource_keywords(cd)

    obs_ok = True
    for s in all_steps:
        pool = (can_kw_a | scope_a) if s.get("agent_id") == "agent_A" else (can_kw_b | scope_b)
        kw   = _resource_keywords(s.get("action", ""))
        if kw and pool and not (kw & pool):
            obs_ok = False
            break

    # 조건 3: cross-agent DEPENDENCY conflicts 해결 여부
    dep_conflicts_after = [c for c in conflicts if c.conflict_type == ConflictType.DEPENDENCY]
    missing_deps: List[int] = []
    for c in dep_conflicts_after:
        for sid in c.step_ids:
            step = next((s for s in all_steps if s["step_id"] == sid), None)
            if step and not step.get("depends_on"):
                missing_deps.append(sid)
    no_missing = len(missing_deps) == 0

    unresolved = [c for c in conflicts
                  if c.conflict_type in (ConflictType.REDUNDANCY, ConflictType.CANNOT_DO)]
    converged  = no_cycle and obs_ok and no_missing

    print(f"  No dep cycle   : {'OK' if no_cycle else 'FAIL'}")
    print(f"  Observability  : {'OK' if obs_ok else 'FAIL'}")
    print(f"  Missing deps   : {'OK' if no_missing else f'FAIL (steps={missing_deps})'}")
    print(f"  → Converged    : {'YES ✓' if converged else 'NO ✗'}")
    if unresolved:
        print(f"  Residual ({len(unresolved)}): {[c.conflict_type for c in unresolved]}")

    return ConvergenceResult(
        converged            = converged,
        no_dep_cycle         = no_cycle,
        observability_ok     = obs_ok,
        no_missing_deps      = no_missing,
        unresolved_conflicts = unresolved,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6: DEFERRED HUMAN QUERY (VLM 기반, 수렴 실패 시에만)
# ══════════════════════════════════════════════════════════════════════════════

_HQ_TEMPLATES: Dict[str, str] = {
    "DEP_CYCLE":    "A dependency cycle was detected. Which step should be reordered?",
    "DEPENDENCY":   "A step uses results from another agent but runs before them. How should timing be adjusted?",
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
    prompt = f"""You are coordinating two home agents.

Agent A ({offer_a.room_type}) can do: {json.dumps(offer_a.can_do[:3], ensure_ascii=False)}
Agent B ({offer_b.room_type}) can do: {json.dumps(offer_b.can_do[:3], ensure_ascii=False)}

Issue ({trigger_type}): {detail[:200]}

Write ONE specific, actionable question for the human operator.
Maximum 2 sentences. No preamble."""

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
        d = "Some steps are missing cross-agent depends_on links."
        triggered.append(f"[DEPENDENCY] {d}")
        raw_triggers.append(("DEPENDENCY", d, 0.85))

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

    if human_answers and verbose in ("full", "summary"):
        print("  Human answers:")
        for q, a in human_answers.items():
            print(f"    Q: {q[:65]}...")
            print(f"    A: {a}")

    # 합산 → time_min 정렬 → step_id 재번호 → depends_on 재매핑
    merged = list(steps_a) + list(steps_b)
    merged.sort(key=lambda s: (s.get("time_min", 0), s.get("step_id", 0)))

    old_to_new: Dict[int, int] = {}
    for new_id, s in enumerate(merged, start=1):
        old_to_new[s["step_id"]] = new_id

    for s in merged:
        s["step_id"]    = old_to_new[s["step_id"]]
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
