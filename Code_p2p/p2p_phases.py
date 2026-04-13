# phases.py
#
# Phase 1 : Observation & Offer Generation
# Phase 2 : Local Planning (각 에이전트 독립)
# Phase 3 : Conflict Detection
# Phase 4 : P2P Negotiation (구조화 제안, add_step 지원, 핑퐁 방지)
# Phase 5 : Convergence Check (rule-based)
# Phase 6 : Deferred Human Query (template 기반, VLM 미사용)
# Finalize: Rule-based merge (LLM 없음)
#
# 변경사항:
#   - Phase 1/2/4: ThreadPoolExecutor로 A/B 병렬 VLM 호출
#   - Phase 3 TEMPORAL: 다른 방이면 탐지 안 함 (false positive 제거)
#   - Phase 3 DEPENDENCY: sender/receiver 방향을 명시적으로 분리
#   - Phase 4: add_step 제안 지원 (새 PASS step 삽입)
#   - Phase 4: 핑퐁 방지 (이전 라운드와 반대 값 제안 시 무시)
#   - Phase 4: DEPENDENCY conflict 전용 안내 추가
#   - Phase 6: VLM 호출 제거, template 기반 질문 생성

from __future__ import annotations

import json
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Tuple

from p2p_config import (
    AGENT_B_STEP_OFFSET,
    AUTO_HQ_ANSWER,
    FUZZY_STOPWORDS,
    HQ_TOP_K,
    MAX_CAN_DO,
    MAX_CANNOT_DO,
    MAX_NEGOTIATION_ROUNDS,
    UNCERTAINTY_THRESH,
    VALID_AGENTS,
    VALID_PROPOSAL_FIELDS,
)
from p2p_models import (
    CannotEntry, ConflictEntry, ConflictType,
    ConvergenceResult, HQEntry, Handoff,
    LocalPlan, NegotiationProposal, NegotiationRound,
    Offer, PlanStep,
)
from p2p_utils import (
    _banner, _fuzzy_match, _fuzzy_match_soft, _log,
    _match_conf, _norm_agent, _norm_depends, _norm_handoff, _norm_reason,
    clamp01, compute_plan_uncertainty, compute_token_uncertainty,
    extract_json, jdump, safe_int,
)
from p2p_vlm import run_vlm


# ──────────────────────────────────────────────────────────────────────────────
# 병렬 VLM 호출 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _run_vlm_parallel(
    calls: List[Tuple],
) -> List[Tuple[str, List[float]]]:
    """
    여러 VLM 호출을 ThreadPoolExecutor로 병렬 실행.
    calls: [(image_path, prompt, return_logprobs), ...]
    returns: [(text, log_probs), ...]
    """
    with ThreadPoolExecutor(max_workers=len(calls)) as ex:
        futures = [ex.submit(run_vlm, *c) for c in calls]
    return [f.result() for f in futures]


# ──────────────────────────────────────────────────────────────────────────────
# 직렬화 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

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
        "uncertain_count": o.uncertain_count,
    }


def local_plan_to_dict(lp: LocalPlan) -> Dict:
    return {
        "agent_id": lp.agent_id,
        "U_plan":   round(lp.U_plan, 3),
        "steps":    [asdict(s) for s in lp.steps],
        "handoffs": [asdict(h) for h in lp.handoffs],
        "hq_list":  [asdict(h) for h in lp.hq_list],
    }


def plan_steps_to_dicts(steps: List[PlanStep]) -> List[Dict]:
    return [asdict(s) for s in steps]


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1: OBSERVATION & OFFER GENERATION
# ──────────────────────────────────────────────────────────────────────────────

PHASE1_FEW_SHOT = """
EXAMPLE - kitchen agent observing a room:
<JSON>
{
  "room_type": "kitchen",
  "observation": "Kitchen with countertops, a stove, fruits on the island, and a bread basket visible.",
  "obs_scope": "counter surface, island, stove, sink, shelf, microwave, fruits, bread basket",
  "can_do": [
    "place apple and orange from island onto serving tray",
    "arrange bread from basket onto plate",
    "wipe counter surface with cloth",
    "clean visible sink with sponge",
    "organize items on shelf"
  ],
  "cannot_do": [
    {"action": "check contents of closed cabinets", "reason": "NO_OBJECT"},
    {"action": "arrange living room seating", "reason": "NO_OBJECT"}
  ],
  "conf": {
    "place apple and orange from island onto serving tray": 0.9,
    "arrange bread from basket onto plate": 0.85,
    "wipe counter surface with cloth": 0.95,
    "clean visible sink with sponge": 0.9,
    "organize items on shelf": 0.85
  },
  "can_provide": ["prepared fruit snacks on tray", "arranged bread plate"],
  "need_from_other": ["confirmation that living room table is ready to receive snacks"]
}
</JSON>

IMPORTANT:
- cannot_do: only things NOT VISIBLE in your camera view.
- obs_scope: comma-separated string, NOT a list.
- reason: exactly one of NO_OBJECT | NO_CAPABILITY | UNCERTAIN
""".strip()


def build_phase1_prompt(task: str) -> str:
    return f"""You are an embodied home agent. Observe your room through the camera image.

Global task: "{task}"

{PHASE1_FEW_SHOT}

Analyze YOUR room and produce an Offer.

STRICT RULES:
1. Use ONLY objects directly visible in the image.
2. can_do: max {MAX_CAN_DO} unique items.
3. cannot_do: max {MAX_CANNOT_DO} items. reason MUST be NO_OBJECT | NO_CAPABILITY | UNCERTAIN.
4. can_do and cannot_do must NOT share any action.
5. conf: confidence per action [0.0-1.0].
6. can_provide: concrete items you can hand off to the other agent.
7. need_from_other: concrete items you need from the other agent.
8. obs_scope: comma-separated string.
9. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "room_type": "your room type",
  "observation": "one concise sentence",
  "obs_scope": "comma-separated visible areas",
  "can_do": ["verb + specific visible object"],
  "cannot_do": [{{"action": "...", "reason": "NO_OBJECT"}}],
  "conf": {{"action": 0.9}},
  "can_provide": ["concrete item for the other agent"],
  "need_from_other": ["concrete item needed from the other agent"]
}}
</JSON>"""


def _parse_offer(raw: str, agent_id: str) -> Offer:
    data = extract_json(raw)
    cannot_do: List[CannotEntry] = []
    uncertain_count = 0

    for item in data.get("cannot_do", [])[:MAX_CANNOT_DO]:
        if isinstance(item, dict):
            action = str(item.get("action", "")).strip()
            reason = _norm_reason(item.get("reason", "UNCERTAIN"))
            if action:
                if reason == "UNCERTAIN":
                    uncertain_count += 1
                cannot_do.append(CannotEntry(action, reason))

    cannot_kw = {c.action for c in cannot_do}
    seen: set = set()
    can_do: List[str] = []

    for x in data.get("can_do", []):
        a = str(x).strip()
        if not a or a.lower() in seen:
            continue
        if any(_fuzzy_match(a, c, min_overlap=2) for c in cannot_kw):
            continue
        seen.add(a.lower())
        can_do.append(a)
        if len(can_do) >= MAX_CAN_DO:
            break

    raw_scope = data.get("obs_scope", "")
    obs_scope = ", ".join(str(x).strip() for x in raw_scope) if isinstance(raw_scope, list) else str(raw_scope).strip()

    conf_raw = {str(k).strip(): clamp01(v) for k, v in data.get("conf", {}).items()}
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
    _banner("PHASE 1 - OBSERVATION & OFFER GENERATION")
    prompt = build_phase1_prompt(task)

    # 병렬 VLM 호출
    results = _run_vlm_parallel([
        (img_a, prompt, False),
        (img_b, prompt, False),
    ])
    raw_a, _ = results[0]
    raw_b, _ = results[1]

    if verbose == "full":
        _log("AGENT A RAW OFFER", raw_a)
        _log("AGENT B RAW OFFER", raw_b)

    offer_a = _parse_offer(raw_a, "agent_A")
    offer_b = _parse_offer(raw_b, "agent_B")

    if verbose in ("full", "summary"):
        _log("PARSED OFFER A", jdump(offer_to_dict(offer_a)))
        _log("PARSED OFFER B", jdump(offer_to_dict(offer_b)))

    print(f"\n  A: room={offer_a.room_type} | can_do={len(offer_a.can_do)} | provide={len(offer_a.can_provide)} | need={len(offer_a.need_from_other)}")
    print(f"  B: room={offer_b.room_type} | can_do={len(offer_b.can_do)} | provide={len(offer_b.can_provide)} | need={len(offer_b.need_from_other)}")
    return offer_a, offer_b


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2: OFFER-CONDITIONED LOCAL PLANNING
# ──────────────────────────────────────────────────────────────────────────────

PHASE2_FEW_SHOT = """
EXAMPLE - kitchen agent preparing snacks for delivery:
<JSON>
{
  "plan_steps": [
    {
      "step_id": 1, "time_min": 0,
      "action": "place apple and orange from island onto serving tray",
      "preconditions": [], "depends_on": [],
      "handoff_type": null, "target_agent": null,
      "uncertainty": 0.1, "notes": ""
    },
    {
      "step_id": 2, "time_min": 5,
      "action": "arrange bread from basket onto plate",
      "preconditions": [], "depends_on": [],
      "handoff_type": null, "target_agent": null,
      "uncertainty": 0.1, "notes": ""
    },
    {
      "step_id": 3, "time_min": 10,
      "action": "carry snack tray to kitchen doorway for agent_B pickup",
      "preconditions": ["snacks arranged on tray"], "depends_on": [1, 2],
      "handoff_type": "PASS", "target_agent": "agent_B",
      "uncertainty": 0.15, "notes": "snack tray ready at doorway"
    },
    {
      "step_id": 4, "time_min": 15,
      "action": "wipe counter surface with cloth",
      "preconditions": [], "depends_on": [],
      "handoff_type": null, "target_agent": null,
      "uncertainty": 0.1, "notes": ""
    }
  ]
}
</JSON>

PASS rules:
- PASS = carry item to room boundary. Must have depends_on pointing to your own prep steps.
- At most 1-2 PASS steps per plan.
- Preparation steps (place, arrange) must NOT have handoff_type=PASS.
- If RECEIVING: add receive step with depends_on=[PASS_step_id], time_min AFTER the PASS.
- Do NOT repeat the same action twice.
""".strip()


def build_phase2_prompt(
    my: Offer, other: Offer, task: str, use_offer: bool = True
) -> str:
    matched_needs    = [n for n in my.need_from_other if any(_fuzzy_match_soft(n, p) for p in other.can_provide)]
    matched_provides = [p for p in my.can_provide if any(_fuzzy_match_soft(p, n) for n in other.need_from_other)]

    if use_offer:
        context = f"""YOUR OFFER:
{jdump(offer_to_dict(my))}

OTHER AGENT'S OFFER ({other.room_type}):
{jdump(offer_to_dict(other))}

MATCHED HANDOFF OPPORTUNITIES:
- Items YOU will PASS to other agent: {json.dumps(matched_provides, ensure_ascii=False)}
- Items other agent will PASS to YOU: {json.dumps(matched_needs, ensure_ascii=False)}"""
    else:
        context = f"YOUR ROOM: {my.room_type}\nOTHER AGENT'S ROOM: {other.room_type}"

    return f"""You are the {my.room_type} agent ({my.agent_id}).

Global task: "{task}"

{context}

{PHASE2_FEW_SHOT}

Generate your LOCAL PLAN (4-6 steps, 0-25 minutes):
1. Steps ONLY in your room. Use only visible objects.
2. Each step must be unique.
3. If you have items to provide: include a PASS step (carry to boundary) with depends_on=[prep_step_id].
4. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "plan_steps": [
    {{
      "step_id": 1, "time_min": 0,
      "action": "verb + specific visible object + detail",
      "preconditions": [], "depends_on": [],
      "handoff_type": null, "target_agent": null,
      "uncertainty": 0.1, "notes": ""
    }}
  ]
}}
</JSON>"""


def _normalize_pass_steps(steps: List[PlanStep]) -> List[PlanStep]:
    """Rule-based PASS 정제: 잘못된 PASS 제거."""
    my_step_ids   = {s.step_id for s in steps}
    valid_passes: List[PlanStep] = []

    for s in steps:
        if s.handoff_type != "PASS":
            continue

        if not s.depends_on:
            print(f"  [NORM] step{s.step_id} PASS removed: no depends_on")
            s.handoff_type = None
            s.target_agent = None
            continue

        valid_deps = [d for d in s.depends_on if d in my_step_ids]
        if not valid_deps:
            print(f"  [NORM] step{s.step_id} PASS removed: deps not in own plan")
            s.handoff_type = None
            s.target_agent = None
            continue

        is_dup = any(_fuzzy_match(s.action, prev.action, min_overlap=3) for prev in valid_passes)
        if is_dup:
            print(f"  [NORM] step{s.step_id} PASS removed: duplicate PASS")
            s.handoff_type = None
            s.target_agent = None
            continue

        valid_passes.append(s)

    return steps


def _parse_local_plan(
    raw: str, log_probs: List[float], my: Offer,
    use_handoff: bool = True, step_offset: int = 0,
) -> LocalPlan:
    data      = extract_json(raw)
    raw_steps = data.get("plan_steps", [])
    if not isinstance(raw_steps, list):
        raw_steps = []

    token_unc     = compute_token_uncertainty(log_probs)
    steps:        List[PlanStep] = []
    hq_list:      List[HQEntry]  = []
    handoffs:     List[Handoff]  = []
    seen_ids:     set = set()
    seen_actions: set = set()

    for i, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip()
        if not action:
            continue

        action_key = frozenset(_resource_keywords(action))
        if action_key and action_key in seen_actions:
            print(f"  [DEDUP] Skipping duplicate: '{action}'")
            continue
        seen_actions.add(action_key)

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

        handoff_type = _norm_handoff(item.get("handoff_type")) if use_handoff else None
        target       = _norm_agent(item.get("target_agent"))   if use_handoff else None
        raw_deps     = _norm_depends(item.get("depends_on"))
        deps         = [d + step_offset for d in raw_deps]

        step = PlanStep(
            step_id       = sid,
            time_min      = max(0, min(30, safe_int(item.get("time_min", 0), 0))),
            room          = my.room_type,
            agent_id      = my.agent_id,
            action        = action,
            preconditions = [str(x).strip() for x in item.get("preconditions", []) if str(x).strip()],
            depends_on    = deps,
            handoff_type  = handoff_type,
            target_agent  = target,
            uncertainty   = step_unc,
            notes         = str(item.get("notes", "")).strip(),
        )
        steps.append(step)

        if step_unc >= UNCERTAINTY_THRESH:
            hq_list.append(HQEntry(sid, f"Is '{action}' feasible?", step_unc))

    if use_handoff:
        steps = _normalize_pass_steps(steps)

    for s in steps:
        if s.handoff_type:
            payload = s.notes if s.handoff_type == "INFORM" else ""
            handoffs.append(Handoff(s.step_id, s.action, s.handoff_type, s.target_agent, payload))

    steps.sort(key=lambda s: (s.time_min, s.step_id))
    all_unc = [s.uncertainty for s in steps] if steps else [token_unc]
    return LocalPlan(my.agent_id, steps, compute_plan_uncertainty(all_unc), hq_list, handoffs)


def phase2_local_plan(
    offer_a: Offer, offer_b: Offer,
    img_a: str, img_b: str, task: str,
    use_offer: bool = True, use_handoff: bool = True,
    verbose: str = "full",
) -> Tuple[LocalPlan, LocalPlan]:
    _banner("PHASE 2 - INDEPENDENT LOCAL PLANNING")

    prompt_a = build_phase2_prompt(offer_a, offer_b, task, use_offer)
    prompt_b = build_phase2_prompt(offer_b, offer_a, task, use_offer)

    # 병렬 VLM 호출
    results = _run_vlm_parallel([
        (img_a, prompt_a, True),
        (img_b, prompt_b, True),
    ])
    raw_a, logp_a = results[0]
    raw_b, logp_b = results[1]

    if verbose == "full":
        _log("AGENT A RAW LOCAL PLAN", raw_a)
        _log("AGENT B RAW LOCAL PLAN", raw_b)

    plan_a = _parse_local_plan(raw_a, logp_a, offer_a, use_handoff, step_offset=0)
    plan_b = _parse_local_plan(raw_b, logp_b, offer_b, use_handoff, step_offset=AGENT_B_STEP_OFFSET)

    if verbose in ("full", "summary"):
        _log("PARSED LOCAL PLAN A", jdump(local_plan_to_dict(plan_a)))
        _log("PARSED LOCAL PLAN B", jdump(local_plan_to_dict(plan_b)))

    print(f"\n  A: U_plan={plan_a.U_plan:.3f} | steps={len(plan_a.steps)} | handoffs={len(plan_a.handoffs)}")
    print(f"  B: U_plan={plan_b.U_plan:.3f} | steps={len(plan_b.steps)} | handoffs={len(plan_b.handoffs)}")
    for tag, plan in [("A", plan_a), ("B", plan_b)]:
        for h in plan.handoffs:
            print(f"  [{tag}->{h.handoff_type}] step={h.step_id} target={h.target_agent} | {h.action}")

    return plan_a, plan_b


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3: CONFLICT DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def _resource_keywords(action: str) -> Set[str]:
    words = set(re.findall(r"\w+", action.lower()))
    return words - FUZZY_STOPWORDS


def detect_conflicts(
    plan_a: LocalPlan,
    plan_b: LocalPlan,
    offer_a: Offer,
    offer_b: Offer,
) -> List[ConflictEntry]:
    conflicts: List[ConflictEntry] = []
    steps_a   = plan_a.steps
    steps_b   = plan_b.steps
    all_steps = [(s, offer_a) for s in steps_a] + [(s, offer_b) for s in steps_b]

    # ── 1. TEMPORAL ─────────────────────────────────────────────────────────
    # FIX: 다른 방이면 같은 키워드라도 실제 충돌 아님 → 같은 방일 때만 탐지
    time_slots: Dict[int, List[PlanStep]] = {}
    for s in steps_a + steps_b:
        time_slots.setdefault(s.time_min, []).append(s)

    for t, slot_steps in time_slots.items():
        if len(slot_steps) < 2:
            continue
        for i in range(len(slot_steps)):
            for j in range(i + 1, len(slot_steps)):
                si, sj = slot_steps[i], slot_steps[j]
                if si.agent_id == sj.agent_id:
                    continue
                # 같은 방에 있을 때만 TEMPORAL conflict
                if si.room != sj.room:
                    continue
                ki = _resource_keywords(si.action)
                kj = _resource_keywords(sj.action)
                overlap = ki & kj
                if overlap:
                    conflicts.append(ConflictEntry(
                        conflict_type = ConflictType.TEMPORAL,
                        step_ids      = [si.step_id, sj.step_id],
                        agent_ids     = [si.agent_id, sj.agent_id],
                        description   = (
                            f"T={t}m same room '{si.room}': "
                            f"step{si.step_id}({si.agent_id}) and "
                            f"step{sj.step_id}({sj.agent_id}) share {overlap}"
                        ),
                    ))

    # ── 2. DEPENDENCY ────────────────────────────────────────────────────────
    # FIX: sender/receiver 방향을 명시적으로 분리해서 description에 기록
    pass_targets_from_b = {h.target_agent for h in plan_b.handoffs if h.handoff_type == "PASS"}
    pass_targets_from_a = {h.target_agent for h in plan_a.handoffs if h.handoff_type == "PASS"}

    for provide in offer_a.can_provide:
        matched_need = next((n for n in offer_b.need_from_other if _fuzzy_match_soft(provide, n)), None)
        if matched_need and "agent_B" not in pass_targets_from_a:
            conflicts.append(ConflictEntry(
                conflict_type = ConflictType.DEPENDENCY,
                step_ids      = [],
                agent_ids     = ["agent_A", "agent_B"],
                description   = (
                    f"agent_A can provide '{provide}' which agent_B needs ('{matched_need}'), "
                    f"but agent_A has NO PASS step to agent_B. "
                    f"RESOLUTION: agent_A must add a PASS step carrying '{provide}' to room boundary."
                ),
            ))

    for provide in offer_b.can_provide:
        matched_need = next((n for n in offer_a.need_from_other if _fuzzy_match_soft(provide, n)), None)
        if matched_need and "agent_A" not in pass_targets_from_b:
            conflicts.append(ConflictEntry(
                conflict_type = ConflictType.DEPENDENCY,
                step_ids      = [],
                agent_ids     = ["agent_A", "agent_B"],
                description   = (
                    f"agent_B can provide '{provide}' which agent_A needs ('{matched_need}'), "
                    f"but agent_B has NO PASS step to agent_A. "
                    f"RESOLUTION: agent_B must add a PASS step carrying '{provide}' to room boundary."
                ),
            ))

    # ── 3. REDUNDANCY inter-agent ─────────────────────────────────────────────
    for sa in steps_a:
        for sb in steps_b:
            if _fuzzy_match(sa.action, sb.action, min_overlap=3):
                conflicts.append(ConflictEntry(
                    conflict_type = ConflictType.REDUNDANCY,
                    step_ids      = [sa.step_id, sb.step_id],
                    agent_ids     = ["agent_A", "agent_B"],
                    description   = (
                        f"Inter-agent duplicate: A-step{sa.step_id} '{sa.action}' ~= "
                        f"B-step{sb.step_id} '{sb.action}'"
                    ),
                ))

    # ── 4. REDUNDANCY intra-agent ─────────────────────────────────────────────
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
                    ))

    # ── 5. CANNOT_DO violation ────────────────────────────────────────────────
    for step, offer in all_steps:
        for c in offer.cannot_do:
            if _fuzzy_match(step.action, c.action, min_overlap=2):
                conflicts.append(ConflictEntry(
                    conflict_type = ConflictType.CANNOT_DO,
                    step_ids      = [step.step_id],
                    agent_ids     = [step.agent_id],
                    description   = (
                        f"{step.agent_id} step{step.step_id} '{step.action}' "
                        f"violates cannot_do '{c.action}'"
                    ),
                ))

    # ── 6. OBSERVABILITY violation ────────────────────────────────────────────
    for step, offer in all_steps:
        scope_kw  = set(re.findall(r"\w+", offer.obs_scope.lower()))
        action_kw = _resource_keywords(step.action)
        if step.action.lower().startswith(("inform", "receive", "carry")):
            continue
        if step.handoff_type == "PASS":
            continue
        if action_kw and scope_kw and not (action_kw & scope_kw):
            conflicts.append(ConflictEntry(
                conflict_type = ConflictType.OBSERV,
                step_ids      = [step.step_id],
                agent_ids     = [step.agent_id],
                description   = (
                    f"{step.agent_id} step{step.step_id} '{step.action}' "
                    f"references objects outside obs_scope"
                ),
            ))

    # ── 7. HANDOFF mismatch ───────────────────────────────────────────────────
    all_pass_step_ids = {
        h.step_id for h in plan_a.handoffs + plan_b.handoffs
        if h.handoff_type == "PASS"
    }
    all_depends = {dep for s in steps_a + steps_b for dep in s.depends_on}

    for sid in all_pass_step_ids:
        if sid not in all_depends:
            conflicts.append(ConflictEntry(
                conflict_type = ConflictType.HANDOFF,
                step_ids      = [sid],
                agent_ids     = ["agent_A", "agent_B"],
                description   = f"PASS step{sid} has no matching receiver step",
            ))

    return conflicts


def phase3_conflict_detection(
    plan_a: LocalPlan, plan_b: LocalPlan,
    offer_a: Offer, offer_b: Offer,
    verbose: str = "full",
) -> List[ConflictEntry]:
    _banner("PHASE 3 - CONFLICT DETECTION")
    conflicts = detect_conflicts(plan_a, plan_b, offer_a, offer_b)

    if not conflicts:
        print("  No conflicts detected.")
    else:
        by_type: Dict[str, List[ConflictEntry]] = {}
        for c in conflicts:
            by_type.setdefault(c.conflict_type, []).append(c)
        for ctype, clist in by_type.items():
            print(f"\n  [{ctype}] x {len(clist)}")
            for c in clist:
                print(f"    steps={c.step_ids} agents={c.agent_ids}")
                if verbose in ("full", "summary"):
                    print(f"      -> {c.description}")

    return conflicts


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 4: PEER-TO-PEER NEGOTIATION
# ──────────────────────────────────────────────────────────────────────────────

def _conflict_to_str(c: ConflictEntry) -> str:
    return f"[{c.conflict_type}] steps={c.step_ids} - {c.description}"


def _build_negotiation_prompt(
    my_agent:        str,
    my_offer:        Offer,
    other_offer:     Offer,
    current_plan_a:  List[Dict],
    current_plan_b:  List[Dict],
    conflicts:       List[ConflictEntry],
    locked_step_ids: Set[int],
    round_num:       int,
    prev_proposals:  List[NegotiationProposal],
    task:            str,
) -> str:
    conflict_text = "\n".join(f"  - {_conflict_to_str(c)}" for c in conflicts) or "  (none)"
    locked_text   = str(sorted(locked_step_ids)) if locked_step_ids else "(none)"
    prev_text     = (
        "\n".join(
            f"  step{p.step_id}[{p.agent_id}] {p.field}='{p.new_value}' ({p.reason})"
            for p in prev_proposals
        ) or "  (none)"
    )

    my_plan    = current_plan_a if my_agent == "agent_A" else current_plan_b
    other_plan = current_plan_b if my_agent == "agent_A" else current_plan_a
    other_id   = "agent_B" if my_agent == "agent_A" else "agent_A"

    # DEPENDENCY conflict가 있으면 전용 안내 추가
    dep_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.DEPENDENCY]
    dep_guide = ""
    if dep_conflicts:
        dep_guide = f"""
DEPENDENCY RESOLUTION GUIDE:
For DEPENDENCY conflicts, the SENDER agent must add a new PASS step.
Use field="add_step" with new_value as a JSON object:
{{
  "step_id": <new_unique_id>,
  "time_min": <after_last_prep_step>,
  "action": "carry <item> to room boundary for {other_id} pickup",
  "depends_on": [<last_prep_step_id>],
  "handoff_type": "PASS",
  "target_agent": "{other_id}"
}}
If you are the RECEIVER, add a receive step with depends_on=[PASS_step_id].
"""

    return f"""You are {my_agent} ({my_offer.room_type}). NEGOTIATION ROUND {round_num}/{MAX_NEGOTIATION_ROUNDS}.

Global task: "{task}"

YOUR CURRENT PLAN:
{jdump(my_plan)}

{other_id}'s CURRENT PLAN:
{jdump(other_plan)}

DETECTED CONFLICTS:
{conflict_text}

LOCKED STEPS (do NOT modify):
{locked_text}

{other_id}'s PREVIOUS PROPOSALS:
{prev_text}
{dep_guide}
YOUR CAPABILITIES:
- can_do: {json.dumps(my_offer.can_do, ensure_ascii=False)}
- can_provide: {json.dumps(my_offer.can_provide, ensure_ascii=False)}
- obs_scope: {my_offer.obs_scope}

INSTRUCTIONS:
1. For each conflict, propose ONE minimal fix.
2. Do NOT modify locked steps.
3. Prefer modifying YOUR OWN steps.
4. Accept {other_id}'s proposal with reason="ACCEPT" if you agree.
5. Fields available:
   - "time_min"    : shift timing (string integer)
   - "action"      : rewrite action text
   - "handoff_type": "PASS" | "INFORM" | "null"
   - "depends_on"  : JSON array string e.g. "[3, 5]"
   - "delete"      : remove step (new_value="true")
   - "add_step"    : insert new step (new_value=JSON object string)
6. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "proposals": [
    {{
      "step_id": -1,
      "agent_id": "{my_agent}",
      "field": "add_step",
      "new_value": "{{\\"step_id\\": 99, \\"time_min\\": 20, \\"action\\": \\"carry snacks to doorway for {other_id}\\", \\"depends_on\\": [1], \\"handoff_type\\": \\"PASS\\", \\"target_agent\\": \\"{other_id}\\"}}",
      "reason": "DEPENDENCY: I am the sender and must add PASS step"
    }}
  ]
}}
</JSON>"""


def _parse_proposals(raw: str, my_agent: str) -> List[NegotiationProposal]:
    data   = extract_json(raw)
    result = []
    valid_fields = VALID_PROPOSAL_FIELDS | {"add_step"}

    for item in data.get("proposals", []):
        if not isinstance(item, dict):
            continue
        sid       = safe_int(item.get("step_id", -1), -1)
        agent_id  = str(item.get("agent_id", my_agent)).strip()
        field     = str(item.get("field", "")).strip().lower()
        new_value = str(item.get("new_value", "")).strip()
        reason    = str(item.get("reason", "")).strip()

        if field not in valid_fields or not new_value:
            continue
        if field != "add_step" and sid < 0:
            continue
        if agent_id not in VALID_AGENTS:
            agent_id = my_agent

        result.append(NegotiationProposal(sid, agent_id, field, new_value, reason))
    return result


def _apply_proposal(
    plan_a: List[Dict],
    plan_b: List[Dict],
    proposal: NegotiationProposal,
    locked_ids: Set[int],
    offer_a: Offer,
    offer_b: Offer,
) -> bool:
    """
    제안을 플랜에 직접 적용. add_step 지원 추가.
    변경이 이루어지면 True 반환.
    """
    # add_step: 새 스텝 삽입
    if proposal.field == "add_step":
        try:
            new_step_data = json.loads(proposal.new_value)
            if not isinstance(new_step_data, dict):
                return False
            target_plan = plan_a if proposal.agent_id == "agent_A" else plan_b
            offer       = offer_a if proposal.agent_id == "agent_A" else offer_b

            # step_id 중복 방지
            existing_ids = {s["step_id"] for s in target_plan}
            new_sid      = safe_int(new_step_data.get("step_id", -1), -1)
            if new_sid < 0 or new_sid in existing_ids:
                max_id  = max(existing_ids, default=0)
                new_sid = max_id + 1

            target_plan.append({
                "step_id":      new_sid,
                "time_min":     max(0, min(30, safe_int(new_step_data.get("time_min", 25), 25))),
                "room":         offer.room_type,
                "agent_id":     proposal.agent_id,
                "action":       str(new_step_data.get("action", "")).strip(),
                "preconditions":[],
                "depends_on":   _norm_depends(new_step_data.get("depends_on", [])),
                "handoff_type": _norm_handoff(new_step_data.get("handoff_type")),
                "target_agent": _norm_agent(new_step_data.get("target_agent")),
                "uncertainty":  0.15,
                "notes":        "negotiation-added step",
            })
            target_plan.sort(key=lambda s: (s["time_min"], s["step_id"]))
            print(f"  [ADD_STEP] step{new_sid} added to {proposal.agent_id}'s plan")
            return True
        except Exception as e:
            print(f"  [ADD_STEP ERROR] {e}")
            return False

    # 기존 스텝 수정
    if proposal.step_id in locked_ids:
        return False

    target_plan = plan_a if proposal.agent_id == "agent_A" else plan_b
    sid_map     = {s["step_id"]: i for i, s in enumerate(target_plan)}
    if proposal.step_id not in sid_map:
        return False

    idx = sid_map[proposal.step_id]

    if proposal.field == "delete":
        target_plan.pop(idx)
        return True

    if proposal.field == "time_min":
        new_t = safe_int(proposal.new_value, -1)
        if 0 <= new_t <= 30:
            target_plan[idx]["time_min"] = new_t
            return True

    elif proposal.field == "action":
        if proposal.new_value:
            target_plan[idx]["action"] = proposal.new_value
            return True

    elif proposal.field == "handoff_type":
        ht = _norm_handoff(proposal.new_value)
        target_plan[idx]["handoff_type"] = ht
        if ht is None:
            target_plan[idx]["target_agent"] = None
        return True

    elif proposal.field == "depends_on":
        try:
            deps = json.loads(proposal.new_value)
            if isinstance(deps, list):
                target_plan[idx]["depends_on"] = [int(d) for d in deps]
                return True
        except Exception:
            pass

    return False


def _lock_agreed_steps(
    proposals_a:       List[NegotiationProposal],
    proposals_b:       List[NegotiationProposal],
    conflict_step_ids: Set[int],
    existing_locked:   Set[int],
) -> Set[int]:
    accepted_by_b   = {p.step_id for p in proposals_b if p.reason.upper() == "ACCEPT"}
    accepted_by_a   = {p.step_id for p in proposals_a if p.reason.upper() == "ACCEPT"}
    proposed_by_a   = {p.step_id for p in proposals_a if p.reason.upper() != "ACCEPT"}
    proposed_by_b   = {p.step_id for p in proposals_b if p.reason.upper() != "ACCEPT"}
    mutually_agreed = (proposed_by_a & accepted_by_b) | (proposed_by_b & accepted_by_a)
    all_mentioned   = {p.step_id for p in proposals_a + proposals_b}
    uncontested     = conflict_step_ids - all_mentioned
    return existing_locked | mutually_agreed | uncontested


def _detect_pingpong(
    proposals: List[NegotiationProposal],
    prev_values: Dict[Tuple[int, str], str],
) -> Tuple[List[NegotiationProposal], Dict[Tuple[int, str], str]]:
    """
    핑퐁 방지: 이전 라운드에서 적용된 값과 반대 방향 제안이면 무시.
    prev_values: {(step_id, field): last_applied_value}
    """
    filtered = []
    new_prev  = dict(prev_values)

    for p in proposals:
        key      = (p.step_id, p.field)
        last_val = prev_values.get(key)

        # add_step은 핑퐁 체크 불필요
        if p.field == "add_step":
            filtered.append(p)
            continue

        # 이전 값과 동일하면 skip (아무 변화 없음)
        if last_val is not None and last_val == p.new_value:
            print(f"  [PINGPONG SKIP] step{p.step_id}.{p.field}='{p.new_value}' (same as prev)")
            continue

        filtered.append(p)
        new_prev[key] = p.new_value

    return filtered, new_prev


def phase4_negotiation(
    plan_a:      LocalPlan,
    plan_b:      LocalPlan,
    offer_a:     Offer,
    offer_b:     Offer,
    conflicts:   List[ConflictEntry],
    img_a:       str,
    img_b:       str,
    task:        str,
    use_handoff: bool = True,
    verbose:     str  = "full",
) -> Tuple[List[Dict], List[Dict], List[NegotiationRound]]:
    _banner("PHASE 4 - PEER-TO-PEER NEGOTIATION")

    if not conflicts:
        print("  No conflicts -> skipping negotiation.")
        return plan_steps_to_dicts(plan_a.steps), plan_steps_to_dicts(plan_b.steps), []

    conflict_step_ids: Set[int] = {sid for c in conflicts for sid in c.step_ids}
    print(f"  Conflict step IDs: {sorted(conflict_step_ids)}")
    print(f"  Max rounds: {MAX_NEGOTIATION_ROUNDS}")

    cur_a: List[Dict] = plan_steps_to_dicts(plan_a.steps)
    cur_b: List[Dict] = plan_steps_to_dicts(plan_b.steps)
    locked: Set[int]  = set()
    rounds: List[NegotiationRound] = []

    prev_props_a: List[NegotiationProposal] = []
    prev_props_b: List[NegotiationProposal] = []

    # 핑퐁 추적: {(step_id, field): last_applied_value}
    applied_vals_a: Dict[Tuple[int, str], str] = {}
    applied_vals_b: Dict[Tuple[int, str], str] = {}

    for rnd in range(1, MAX_NEGOTIATION_ROUNDS + 1):
        remaining = [
            c for c in conflicts
            if not all(sid in locked for sid in c.step_ids) or not c.step_ids
        ]
        if not remaining:
            print(f"\n  Round {rnd}: All conflicts resolved -> stopping early.")
            break

        print(f"\n  -- Round {rnd}/{MAX_NEGOTIATION_ROUNDS} "
              f"(remaining: {len(remaining)}, locked: {sorted(locked)}) --")

        prompt_a = _build_negotiation_prompt(
            "agent_A", offer_a, offer_b, cur_a, cur_b,
            remaining, locked, rnd, prev_props_b, task,
        )
        prompt_b = _build_negotiation_prompt(
            "agent_B", offer_b, offer_a, cur_a, cur_b,
            remaining, locked, rnd, prev_props_a, task,
        )

        # 병렬 VLM 호출
        results = _run_vlm_parallel([
            (img_a, prompt_a, False),
            (img_b, prompt_b, False),
        ])
        raw_a, _ = results[0]
        raw_b, _ = results[1]

        props_a = _parse_proposals(raw_a, "agent_A")
        props_b = _parse_proposals(raw_b, "agent_B")

        if verbose == "full":
            _log(f"ROUND {rnd} AGENT A RAW", raw_a)
            _log(f"ROUND {rnd} AGENT B RAW", raw_b)

        # 핑퐁 필터링
        props_a, applied_vals_a = _detect_pingpong(props_a, applied_vals_a)
        props_b, applied_vals_b = _detect_pingpong(props_b, applied_vals_b)

        if verbose in ("full", "summary"):
            for p in props_a:
                print(f"  [A->{p.agent_id}] step{p.step_id} {p.field}='{p.new_value[:50]}' ({p.reason})")
            for p in props_b:
                print(f"  [B->{p.agent_id}] step{p.step_id} {p.field}='{p.new_value[:50]}' ({p.reason})")

        for prop in props_a:
            changed = _apply_proposal(cur_a, cur_b, prop, locked, offer_a, offer_b)
            if changed and verbose in ("full", "summary"):
                print(f"  [APPLIED A] step{prop.step_id}.{prop.field}")

        for prop in props_b:
            changed = _apply_proposal(cur_a, cur_b, prop, locked, offer_a, offer_b)
            if changed and verbose in ("full", "summary"):
                print(f"  [APPLIED B] step{prop.step_id}.{prop.field}")

        locked = _lock_agreed_steps(props_a, props_b, conflict_step_ids, locked)
        rounds.append(NegotiationRound(
            round_num       = rnd,
            proposals_a     = props_a,
            proposals_b     = props_b,
            locked_step_ids = sorted(locked),
        ))
        print(f"  -> Locked after round {rnd}: {sorted(locked)}")

        prev_props_a = props_a
        prev_props_b = props_b

    print(f"\n  Negotiation complete. Total rounds: {len(rounds)}")
    return cur_a, cur_b, rounds


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 5: CONVERGENCE CHECK (rule-based)
# ──────────────────────────────────────────────────────────────────────────────

def _has_cycle(steps: List[Dict]) -> bool:
    indegree = {s["step_id"]: 0 for s in steps}
    adj: Dict[int, List[int]] = {s["step_id"]: [] for s in steps}

    for s in steps:
        for dep in s.get("depends_on", []):
            if dep in adj:
                adj[dep].append(s["step_id"])
                indegree[s["step_id"]] += 1

    q = deque(sid for sid, deg in indegree.items() if deg == 0)
    visited = 0
    while q:
        node = q.popleft()
        visited += 1
        for nxt in adj.get(node, []):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)

    return visited != len(steps)


def phase5_convergence_check(
    steps_a:   List[Dict],
    steps_b:   List[Dict],
    offer_a:   Offer,
    offer_b:   Offer,
    conflicts: List[ConflictEntry],
) -> ConvergenceResult:
    _banner("PHASE 5 - CONVERGENCE CHECK (rule-based)")
    all_steps = steps_a + steps_b

    pass_step_ids  = {s["step_id"] for s in all_steps if s.get("handoff_type") == "PASS"}
    all_depends    = {dep for s in all_steps for dep in s.get("depends_on", [])}
    unmatched_pass = pass_step_ids - all_depends
    pass_matched   = len(unmatched_pass) == 0
    no_dep_cycle   = not _has_cycle(all_steps)

    scope_a = set(re.findall(r"\w+", offer_a.obs_scope.lower()))
    scope_b = set(re.findall(r"\w+", offer_b.obs_scope.lower()))
    obs_ok  = True

    for s in all_steps:
        action = s.get("action", "").lower()
        if action.startswith(("inform", "receive", "carry")):
            continue
        if s.get("handoff_type") == "PASS":
            continue
        scope = scope_a if s.get("agent_id") == "agent_A" else scope_b
        kw    = _resource_keywords(s.get("action", ""))
        if kw and scope and not (kw & scope):
            obs_ok = False
            break

    unresolved = [
        c for c in conflicts
        if c.conflict_type in (ConflictType.CANNOT_DO, ConflictType.REDUNDANCY)
    ]

    converged = pass_matched and no_dep_cycle and obs_ok

    print(f"  PASS matched     : {'OK' if pass_matched else 'FAIL'} (unmatched={unmatched_pass})")
    print(f"  No dep cycle     : {'OK' if no_dep_cycle else 'FAIL'}")
    print(f"  Observability OK : {'OK' if obs_ok else 'FAIL'}")
    print(f"  -> Converged     : {'YES' if converged else 'NO'}")
    if unresolved:
        print(f"  Residual conflicts ({len(unresolved)}):")
        for c in unresolved:
            print(f"    [{c.conflict_type}] {c.description}")

    return ConvergenceResult(
        converged            = converged,
        pass_matched         = pass_matched,
        no_dep_cycle         = no_dep_cycle,
        observability_ok     = obs_ok,
        unresolved_conflicts = unresolved,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 6: DEFERRED HUMAN QUERY (VLM 미사용, template 기반)
# ──────────────────────────────────────────────────────────────────────────────

HQ_TEMPLATES: Dict[str, str] = {
    "UNMATCHED_PASS": (
        "A PASS handoff has no matching receiver step. "
        "Should the receiving agent add a step to pick up the item at the room boundary?"
    ),
    "DEP_CYCLE": (
        "A circular dependency exists in the plan. "
        "Which step should be executed first to break the cycle?"
    ),
    "DEPENDENCY": (
        "An agent can provide an item the other agent needs, but no PASS step exists. "
        "Should the providing agent carry the item to the room boundary?"
    ),
    "REDUNDANCY": (
        "Two agents are performing similar actions. "
        "Which agent should handle this task, and should the other skip it?"
    ),
    "CANNOT_DO": (
        "An agent is attempting an action outside its capability. "
        "Should this step be removed or reassigned to the other agent?"
    ),
    "UNMATCHED_NEED": (
        "An agent needs something that neither agent can provide. "
        "How should this unmet need be handled?"
    ),
    "HIGH_UNCERTAINTY": (
        "Some plan steps have high uncertainty. "
        "Should the agents proceed with the current plan or request clarification?"
    ),
}


def _generate_hq_question(
    trigger_type: str,
    detail: str,
    offer_a: Offer,
    offer_b: Offer,
) -> str:
    """Template 기반 질문 생성 — VLM 호출 없음."""
    template = HQ_TEMPLATES.get(trigger_type, "How should the agents handle this issue?")
    return f"{template}\nContext: {detail[:120]}"


def phase6_human_query(
    plan_a:          LocalPlan,
    plan_b:          LocalPlan,
    offer_a:         Offer,
    offer_b:         Offer,
    convergence:     ConvergenceResult,
    img_a:           str,
    img_b:           str,
    use_human_query: bool = True,
) -> Tuple[Dict[str, str], List[str], List[str]]:
    _banner("PHASE 6 - DEFERRED HUMAN QUERY")

    if not use_human_query:
        print("  [ABLATION] Human query disabled.")
        return {}, [], []

    if convergence.converged and not convergence.unresolved_conflicts:
        print("  Plan converged -> no human query needed.")
        return {}, [], []

    triggered: List[str] = []
    raw_triggers: List[Tuple[str, str, float]] = []

    if not convergence.pass_matched:
        detail = "Some PASS handoffs have no matching receiver step"
        triggered.append(f"[HQ1] {detail}")
        raw_triggers.append(("UNMATCHED_PASS", detail, 0.90))

    if not convergence.no_dep_cycle:
        detail = "Dependency cycle detected"
        triggered.append(f"[HQ2] {detail}")
        raw_triggers.append(("DEP_CYCLE", detail, 0.85))

    for c in convergence.unresolved_conflicts:
        triggered.append(f"[HQ3-{c.conflict_type}] {c.description}")
        raw_triggers.append((c.conflict_type, c.description, 0.80))

    all_provides = offer_a.can_provide + offer_b.can_provide
    for need in offer_a.need_from_other + offer_b.need_from_other:
        if not any(_fuzzy_match_soft(need, p) for p in all_provides):
            detail = f"No agent can provide: '{need}'"
            triggered.append(f"[HQ4] {detail}")
            raw_triggers.append(("UNMATCHED_NEED", detail, 0.85))

    if not triggered:
        print("  No human query needed.")
        return {}, [], []

    print(f"  Triggers:")
    for t in triggered:
        print(f"    {t}")

    raw_triggers.sort(key=lambda x: -x[2])
    answers: Dict[str, str] = {}
    asked:   List[str]      = []

    for i, (ttype, detail, u) in enumerate(raw_triggers[:HQ_TOP_K], 1):
        # Template 기반 질문 생성 (VLM 미사용)
        question = _generate_hq_question(ttype, detail, offer_a, offer_b)
        print(f"\n  Q{i} [{ttype}, priority={u:.2f}]:")
        print(f"  {question}")
        asked.append(question)

        if AUTO_HQ_ANSWER is not None:
            ans = AUTO_HQ_ANSWER
            print(f"  A (auto): {ans}")
        else:
            try:
                ans = input("  A: ").strip()
            except EOFError:
                ans = ""

        if ans:
            answers[question] = ans

    return answers, triggered, asked


# ──────────────────────────────────────────────────────────────────────────────
# FINALIZE: RULE-BASED MERGE (LLM 없음)
# ──────────────────────────────────────────────────────────────────────────────

def _auto_add_pass_receivers(
    steps: List[Dict],
    offer_a: Offer,
    offer_b: Offer,
) -> List[Dict]:
    """
    PASS step에 receiver가 없으면 자동 추가.
    반드시 step_id 재번호 이전에 호출해야 함.
    """
    room_map      = {"agent_A": offer_a.room_type, "agent_B": offer_b.room_type}
    pass_steps    = {s["step_id"]: s for s in steps if s.get("handoff_type") == "PASS"}
    pass_received = {dep for s in steps for dep in s.get("depends_on", []) if dep in pass_steps}

    additions = []
    max_id    = max((s["step_id"] for s in steps), default=0)

    for sid, pass_s in pass_steps.items():
        if sid in pass_received:
            continue
        target = pass_s.get("target_agent")
        if not target or target not in room_map:
            continue

        max_id += 1
        receiver_time = min(30, pass_s["time_min"] + 1)
        additions.append({
            "step_id":       max_id,
            "time_min":      receiver_time,
            "room":          room_map[target],
            "agent_id":      target,
            "action":        f"receive and place: {pass_s['action']}",
            "preconditions": [f"step {sid} completed"],
            "depends_on":    [sid],
            "handoff_type":  None,
            "target_agent":  None,
            "notes":         "auto-added PASS receiver",
        })
        print(f"  [AUTO-PASS] receiver step{max_id} added for {target} <- PASS step{sid}")

    return steps + additions


def phase_finalize(
    steps_a:       List[Dict],
    steps_b:       List[Dict],
    offer_a:       Offer,
    offer_b:       Offer,
    human_answers: Dict[str, str],
    convergence:   ConvergenceResult,
    verbose:       str = "full",
) -> List[Dict]:
    _banner("FINALIZE - RULE-BASED JOINT PLAN MERGE")

    if human_answers:
        print("  Human query answers:")
        for q, a in human_answers.items():
            print(f"    Q: {q[:60]}...")
            print(f"    A: {a}")

    # 1. 합산
    merged = list(steps_a) + list(steps_b)

    # 2. PASS receiver 자동 보완 (재번호 이전)
    merged = _auto_add_pass_receivers(merged, offer_a, offer_b)

    # 3. time_min 순 정렬
    merged.sort(key=lambda x: (x.get("time_min", 0), x.get("step_id", 0)))

    # 4-5. step_id 재번호 + 재매핑
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
        print(f"\n  Merged {len(steps_a)} A-steps + {len(steps_b)} B-steps = {len(merged)} total steps")

    return merged
