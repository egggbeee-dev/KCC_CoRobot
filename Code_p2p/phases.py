# ══════════════════════════════════════════════════════════════════════════════
# phases.py
#
# Phase 1 : Observation & Offer Generation
# Phase 2 : Local Planning (각 에이전트 독립)
# Phase 3 : Conflict Detection  (temporal / dependency / redundancy / ...)
# Phase 4 : P2P Negotiation     (최대 MAX_NEGOTIATION_ROUNDS 라운드)
# Phase 5 : Convergence Check   (rule-based, LLM 판단 없음)
# Phase 6 : Deferred Human Query (필요할 때만)
# Finalize: Joint Plan 확정
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Dict, List, Optional, Set, Tuple

from config import (
    ALPHA, BETA, GAMMA, DELTA,
    HQ_TOP_K, MAX_CAN_DO, MAX_CANNOT_DO,
    MAX_NEGOTIATION_ROUNDS,
    UNCERTAINTY_THRESH, VALID_AGENTS,
)
from models import (
    CannotEntry, ConflictEntry, ConflictType,
    ConvergenceResult, HQEntry, Handoff, LeaderResult,
    LocalPlan, NegotiationProposal, NegotiationRound,
    Offer, PlanStep,
)
from utils import (
    _banner, _fuzzy_match, _fuzzy_match_soft, _log,
    _match_conf, _norm_agent, _norm_depends, _norm_handoff, _norm_reason,
    clamp01, compute_plan_uncertainty, compute_token_uncertainty,
    extract_json, jdump, safe_int,
)
from vlm import run_vlm


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
EXAMPLE — kitchen agent observing a room:
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
    {"action": "arrange living room seating", "reason": "NO_OBJECT"},
    {"action": "assess floor area outside camera view", "reason": "NO_OBJECT"}
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
- Food items visible in the image (fruits, bread, bowls, containers, bottles)
  CAN be used as snack or drink ingredients. Include preparation actions in can_do.
- cannot_do should ONLY list things NOT VISIBLE in your current camera view.
- obs_scope must be a comma-separated string, NOT a list.
- reason must be exactly one of: NO_OBJECT | NO_CAPABILITY | UNCERTAIN
""".strip()


def build_phase1_prompt(task: str) -> str:
    return f"""You are an embodied home agent. Observe your room through the camera image.

Global task: "{task}"

{PHASE1_FEW_SHOT}

Analyze YOUR room and produce an Offer.

STRICT RULES:
1. Use ONLY objects directly visible in the image.
2. can_do: max {MAX_CAN_DO} unique items. Format: "<verb> <specific visible object> [detail]"
   - If you see food items (fruits, bread, snacks, drinks, bowls), include
     preparation or serving actions using those specific objects.
3. cannot_do: max {MAX_CANNOT_DO} items. reason MUST be one of: NO_OBJECT | NO_CAPABILITY | UNCERTAIN
   - Only list things NOT VISIBLE in your current camera view.
   - Do NOT use "NO_ROOM" or any other reason string.
4. can_do and cannot_do must NOT share any action.
5. conf: confidence per action [0.0–1.0]. Keys must exactly match can_do items.
6. can_provide: concrete items/results you can hand off to the other agent.
7. need_from_other: concrete items you need the other agent to provide.
8. obs_scope: write as a comma-separated string (e.g. "counter, sink, shelf, island").
9. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "room_type": "your room type",
  "observation": "one concise sentence describing what you see",
  "obs_scope": "comma-separated list of visible areas and objects",
  "can_do": ["verb + specific visible object + detail"],
  "cannot_do": [{{"action": "specific action", "reason": "NO_OBJECT"}}],
  "conf": {{"action string": 0.9}},
  "can_provide": ["concrete result or item for the other agent"],
  "need_from_other": ["concrete item or confirmation needed from the other agent"]
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
    if isinstance(raw_scope, list):
        obs_scope = ", ".join(str(x).strip() for x in raw_scope)
    else:
        obs_scope = str(raw_scope).strip()

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
    _banner("PHASE 1 — OBSERVATION & OFFER GENERATION")
    prompt = build_phase1_prompt(task)
    raw_a, _ = run_vlm(img_a, prompt)
    raw_b, _ = run_vlm(img_b, prompt)

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
# PHASE 2: OFFER-CONDITIONED LOCAL PLANNING (두 에이전트 독립적으로)
# ──────────────────────────────────────────────────────────────────────────────

PHASE2_FEW_SHOT = """
EXAMPLE — kitchen agent preparing snacks for delivery:
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
      "uncertainty": 0.15, "notes": "snack tray ready at kitchen doorway"
    },
    {
      "step_id": 4, "time_min": 15,
      "action": "wipe counter surface with cloth",
      "preconditions": [], "depends_on": [],
      "handoff_type": null, "target_agent": null,
      "uncertainty": 0.1, "notes": ""
    },
    {
      "step_id": 5, "time_min": 20,
      "action": "INFORM agent_B: kitchen preparation complete",
      "preconditions": [], "depends_on": [3],
      "handoff_type": "INFORM", "target_agent": "agent_B",
      "uncertainty": 0.1, "notes": "kitchen ready"
    }
  ]
}
</JSON>

HANDOFF RULES:

PASS — physical item transfer between rooms:
  ✅ CORRECT: YOU prepare item → carry to boundary → PASS to other agent
  ❌ WRONG — receiver declaring PASS (you are RECEIVER, not sender)
  ❌ WRONG — PASS without preparation step

INFORM — status/completion notification (no physical item):
  Use INFORM to notify the other agent that your task is done or ready.

CRITICAL:
- uncertainty: 0.0=certain, 1.0=uncertain. Confident actions → LOW uncertainty (0.1–0.2).
- Steps must be in YOUR OWN ROOM ONLY.
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
- Items YOU can PASS to other agent (you are SENDER): {json.dumps(matched_provides, ensure_ascii=False)}
- Items other agent will PASS to YOU (you are RECEIVER): {json.dumps(matched_needs, ensure_ascii=False)}"""
    else:
        context = f"YOUR ROOM: {my.room_type}\nOTHER AGENT'S ROOM: {other.room_type}"

    return f"""You are the {my.room_type} agent ({my.agent_id}).

Global task: "{task}"

{context}

{PHASE2_FEW_SHOT}

Generate your LOCAL PLAN independently:
1. OBSERVABILITY: steps ONLY for your room ({my.room_type}). Use only objects visible in your image.
2. EXECUTABILITY: every action must be physically possible in your room.
3. Generate 4–6 steps spread over 0–25 minutes.
4. UNCERTAINTY [0.0–1.0]: set LOW (0.1–0.2) for actions you are confident about.
5. HANDOFF — two types only:
   - PASS: ONLY if YOU are SENDING a physical item to the other agent.
   - INFORM: notify the other agent of your completion or status.
6. If you are RECEIVING an item, add a receive step with depends_on=[sender's PASS step_id] but NO handoff_type.
7. Return ONLY valid JSON inside <JSON> tags.

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


def _parse_local_plan(
    raw: str, log_probs: List[float], my: Offer, use_handoff: bool = True
) -> LocalPlan:
    data      = extract_json(raw)
    raw_steps = data.get("plan_steps", [])
    if not isinstance(raw_steps, list):
        raw_steps = []

    token_unc = compute_token_uncertainty(log_probs)
    steps:    List[PlanStep] = []
    hq_list:  List[HQEntry]  = []
    handoffs: List[Handoff]  = []
    seen_ids: set = set()

    for i, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip()
        if not action:
            continue

        sid = safe_int(item.get("step_id", i), i)
        while sid in seen_ids:
            sid += 1
        seen_ids.add(sid)

        json_unc    = clamp01(item.get("uncertainty", 0.2))
        action_conf = max(
            (v for k, v in my.conf.items() if _fuzzy_match_soft(action, k)),
            default=0.7
        )
        step_unc = clamp01(
            json_unc         * 0.5
            + token_unc      * 0.2
            + (1 - action_conf) * 0.3
        )

        handoff_type = _norm_handoff(item.get("handoff_type")) if use_handoff else None
        target       = _norm_agent(item.get("target_agent"))   if use_handoff else None

        step = PlanStep(
            step_id       = sid,
            time_min      = max(0, min(30, safe_int(item.get("time_min", 0), 0))),
            room          = my.room_type,
            agent_id      = my.agent_id,
            action        = action,
            preconditions = [str(x).strip() for x in item.get("preconditions", []) if str(x).strip()],
            depends_on    = _norm_depends(item.get("depends_on")),
            handoff_type  = handoff_type,
            target_agent  = target,
            uncertainty   = step_unc,
            notes         = str(item.get("notes", "")).strip(),
        )
        steps.append(step)

        if step_unc >= UNCERTAINTY_THRESH:
            hq_list.append(HQEntry(sid, f"Is '{action}' feasible?", step_unc))
        if handoff_type:
            payload = step.notes if handoff_type == "INFORM" else ""
            handoffs.append(Handoff(sid, action, handoff_type, target, payload))

    steps.sort(key=lambda s: (s.time_min, s.step_id))
    all_unc = [s.uncertainty for s in steps] if steps else [token_unc]
    return LocalPlan(my.agent_id, steps, compute_plan_uncertainty(all_unc), hq_list, handoffs)


def phase2_local_plan(
    offer_a: Offer, offer_b: Offer,
    img_a: str, img_b: str, task: str,
    use_offer: bool = True, use_handoff: bool = True,
    verbose: str = "full",
) -> Tuple[LocalPlan, LocalPlan]:
    _banner("PHASE 2 — INDEPENDENT LOCAL PLANNING")
    prompt_a = build_phase2_prompt(offer_a, offer_b, task, use_offer)
    prompt_b = build_phase2_prompt(offer_b, offer_a, task, use_offer)

    raw_a, logp_a = run_vlm(img_a, prompt_a, return_logprobs=True)
    raw_b, logp_b = run_vlm(img_b, prompt_b, return_logprobs=True)

    if verbose == "full":
        _log("AGENT A RAW LOCAL PLAN", raw_a)
        _log("AGENT B RAW LOCAL PLAN", raw_b)

    plan_a = _parse_local_plan(raw_a, logp_a, offer_a, use_handoff)
    plan_b = _parse_local_plan(raw_b, logp_b, offer_b, use_handoff)

    if verbose in ("full", "summary"):
        _log("PARSED LOCAL PLAN A", jdump(local_plan_to_dict(plan_a)))
        _log("PARSED LOCAL PLAN B", jdump(local_plan_to_dict(plan_b)))

    print(f"\n  A: U_plan={plan_a.U_plan:.3f} | steps={len(plan_a.steps)} | handoffs={len(plan_a.handoffs)}")
    print(f"  B: U_plan={plan_b.U_plan:.3f} | steps={len(plan_b.steps)} | handoffs={len(plan_b.handoffs)}")
    for tag, plan in [("A", plan_a), ("B", plan_b)]:
        for h in plan.handoffs:
            print(f"  [{tag}→{h.handoff_type}] step={h.step_id} target={h.target_agent} | {h.action}")

    return plan_a, plan_b


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3: CONFLICT DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def _resource_keywords(action: str) -> Set[str]:
    """액션에서 자원 키워드 추출 (불용어 제거)."""
    from config import FUZZY_STOPWORDS
    words = set(re.findall(r"\w+", action.lower()))
    return words - FUZZY_STOPWORDS


def detect_conflicts(
    plan_a: LocalPlan,
    plan_b: LocalPlan,
    offer_a: Offer,
    offer_b: Offer,
) -> List[ConflictEntry]:
    """
    두 로컬 플랜을 비교하여 충돌을 분류한다.

    탐지 항목:
      1. TEMPORAL    : 같은 time_min에 공유 자원(키워드 겹침)을 사용하는 스텝
      2. DEPENDENCY  : need_from_other ↔ can_provide 연결은 있지만 PASS handoff 미선언
      3. REDUNDANCY  : 두 에이전트가 사실상 동일한 액션을 시도
      4. CANNOT_DO   : 에이전트가 자신의 cannot_do 위반 액션 포함
      5. OBSERVABILITY: 에이전트 액션이 자기 obs_scope 키워드와 교집합 없음
      6. HANDOFF     : PASS sender에 대응하는 receiver 스텝 없음
    """
    conflicts: List[ConflictEntry] = []

    steps_a = plan_a.steps
    steps_b = plan_b.steps
    all_steps = [(s, offer_a) for s in steps_a] + [(s, offer_b) for s in steps_b]

    # ── 1. TEMPORAL conflict ─────────────────────────────────────────────────
    # 같은 time_min 슬롯에서 두 에이전트가 같은 자원 키워드를 쓰는 경우
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
                ki = _resource_keywords(si.action)
                kj = _resource_keywords(sj.action)
                overlap = ki & kj
                if overlap:
                    conflicts.append(ConflictEntry(
                        conflict_type = ConflictType.TEMPORAL,
                        step_ids      = [si.step_id, sj.step_id],
                        agent_ids     = [si.agent_id, sj.agent_id],
                        description   = (
                            f"T={t}m: step {si.step_id}({si.agent_id}) and "
                            f"step {sj.step_id}({sj.agent_id}) share resource keywords {overlap}"
                        ),
                    ))

    # ── 2. DEPENDENCY conflict ───────────────────────────────────────────────
    # A가 B의 결과를 필요로 하는데 B의 PASS handoff가 없거나 A의 receive step이 없는 경우
    pass_senders_a = {h.target_agent for h in plan_a.handoffs if h.handoff_type == "PASS"}
    pass_senders_b = {h.target_agent for h in plan_b.handoffs if h.handoff_type == "PASS"}

    for need in offer_a.need_from_other:
        matched_provide = any(_fuzzy_match_soft(need, p) for p in offer_b.can_provide)
        has_pass = "agent_A" in pass_senders_b   # B가 A에게 PASS 선언
        if matched_provide and not has_pass:
            conflicts.append(ConflictEntry(
                conflict_type = ConflictType.DEPENDENCY,
                step_ids      = [],
                agent_ids     = ["agent_A", "agent_B"],
                description   = (
                    f"agent_A needs '{need}' from agent_B "
                    f"but agent_B declared no PASS handoff to agent_A"
                ),
            ))

    for need in offer_b.need_from_other:
        matched_provide = any(_fuzzy_match_soft(need, p) for p in offer_a.can_provide)
        has_pass = "agent_B" in pass_senders_a   # A가 B에게 PASS 선언
        if matched_provide and not has_pass:
            conflicts.append(ConflictEntry(
                conflict_type = ConflictType.DEPENDENCY,
                step_ids      = [],
                agent_ids     = ["agent_A", "agent_B"],
                description   = (
                    f"agent_B needs '{need}' from agent_A "
                    f"but agent_A declared no PASS handoff to agent_B"
                ),
            ))

    # ── 3. REDUNDANCY conflict ───────────────────────────────────────────────
    for sa in steps_a:
        for sb in steps_b:
            if _fuzzy_match(sa.action, sb.action, min_overlap=3):
                conflicts.append(ConflictEntry(
                    conflict_type = ConflictType.REDUNDANCY,
                    step_ids      = [sa.step_id, sb.step_id],
                    agent_ids     = ["agent_A", "agent_B"],
                    description   = (
                        f"Duplicate actions: A-step{sa.step_id} '{sa.action}' ≈ "
                        f"B-step{sb.step_id} '{sb.action}'"
                    ),
                ))

    # ── 4. CANNOT_DO violation ───────────────────────────────────────────────
    for step, offer in all_steps:
        for c in offer.cannot_do:
            if _fuzzy_match(step.action, c.action, min_overlap=2):
                conflicts.append(ConflictEntry(
                    conflict_type = ConflictType.CANNOT_DO,
                    step_ids      = [step.step_id],
                    agent_ids     = [step.agent_id],
                    description   = (
                        f"{step.agent_id} step {step.step_id} '{step.action}' "
                        f"violates cannot_do '{c.action}' (reason: {c.reason})"
                    ),
                ))

    # ── 5. OBSERVABILITY violation ───────────────────────────────────────────
    for step, offer in all_steps:
        scope_kw = set(re.findall(r"\w+", offer.obs_scope.lower()))
        action_kw = _resource_keywords(step.action)
        # INFORM/receive 스텝은 skip
        if step.action.lower().startswith(("inform", "receive")):
            continue
        if step.handoff_type == "PASS":
            continue
        if action_kw and scope_kw and not (action_kw & scope_kw):
            conflicts.append(ConflictEntry(
                conflict_type = ConflictType.OBSERV,
                step_ids      = [step.step_id],
                agent_ids     = [step.agent_id],
                description   = (
                    f"{step.agent_id} step {step.step_id} '{step.action}' "
                    f"references objects outside obs_scope"
                ),
            ))

    # ── 6. HANDOFF mismatch ──────────────────────────────────────────────────
    all_pass_step_ids = {
        h.step_id for h in plan_a.handoffs + plan_b.handoffs
        if h.handoff_type == "PASS"
    }
    all_depends = {
        dep
        for s in steps_a + steps_b
        for dep in s.depends_on
    }
    for sid in all_pass_step_ids:
        if sid not in all_depends:
            # PASS sender step에 대응하는 receiver 스텝 없음
            # (step_id는 각 로컬플랜 내 id라 전체 비교는 approximation)
            conflicts.append(ConflictEntry(
                conflict_type = ConflictType.HANDOFF,
                step_ids      = [sid],
                agent_ids     = ["agent_A", "agent_B"],
                description   = (
                    f"PASS step {sid} has no matching receiver step "
                    f"(no depends_on referencing it in the other agent's plan)"
                ),
            ))

    return conflicts


def phase3_conflict_detection(
    plan_a: LocalPlan,
    plan_b: LocalPlan,
    offer_a: Offer,
    offer_b: Offer,
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
            print(f"\n  [{ctype}] × {len(clist)}")
            for c in clist:
                print(f"    steps={c.step_ids} agents={c.agent_ids}")
                if verbose in ("full", "summary"):
                    print(f"      → {c.description}")

    return conflicts


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 4: PEER-TO-PEER NEGOTIATION
# ──────────────────────────────────────────────────────────────────────────────

def _conflict_to_str(c: ConflictEntry) -> str:
    return f"[{c.conflict_type}] steps={c.step_ids} — {c.description}"


def _build_negotiation_prompt(
    my_agent:        str,
    my_offer:        Offer,
    other_offer:     Offer,
    current_plan_a:  List[Dict],
    current_plan_b:  List[Dict],
    conflicts:       List[ConflictEntry],
    locked_step_ids: Set[int],
    round_num:       int,
    prev_proposals:  List[NegotiationProposal],   # 상대방이 이전에 제안한 것
    task:            str,
) -> str:
    conflict_text = "\n".join(f"  - {_conflict_to_str(c)}" for c in conflicts) or "  (none)"
    locked_text   = str(sorted(locked_step_ids)) if locked_step_ids else "(none)"
    prev_text = (
        "\n".join(
            f"  step {p.step_id}: '{p.proposed_change}' — reason: {p.reason}"
            for p in prev_proposals
        )
        or "  (none)"
    )
    my_plan    = current_plan_a if my_agent == "agent_A" else current_plan_b
    other_plan = current_plan_b if my_agent == "agent_A" else current_plan_a
    other_id   = "agent_B" if my_agent == "agent_A" else "agent_A"

    return f"""You are {my_agent} ({my_offer.room_type}). This is NEGOTIATION ROUND {round_num}/{MAX_NEGOTIATION_ROUNDS}.

Global task: "{task}"

YOUR CURRENT PLAN:
{jdump(my_plan)}

{other_id}'s CURRENT PLAN:
{jdump(other_plan)}

DETECTED CONFLICTS (negotiation targets only):
{conflict_text}

LOCKED STEPS (already agreed — do NOT modify):
{locked_text}

{other_id}'s PREVIOUS PROPOSALS (consider accepting or counter-proposing):
{prev_text}

YOUR CAPABILITIES:
- can_do: {json.dumps(my_offer.can_do, ensure_ascii=False)}
- cannot_do: {json.dumps([c.action for c in my_offer.cannot_do], ensure_ascii=False)}
- obs_scope: {my_offer.obs_scope}

INSTRUCTIONS:
1. Review the detected conflicts above.
2. For EACH conflict, propose a minimal change to ONE step that resolves it.
   - You may only propose changes to steps that are NOT locked.
   - Prefer modifying your own steps; only propose changes to {other_id}'s steps if truly necessary.
   - If you accept {other_id}'s proposal, include it as-is with reason "ACCEPT".
3. Output proposals as a JSON list. Each proposal: {{step_id, proposed_change, reason}}.
4. If no change is needed for a conflict (already resolved), skip it.
5. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "proposals": [
    {{
      "step_id": 3,
      "proposed_change": "shift time_min from 10 to 15 to avoid resource conflict",
      "reason": "TEMPORAL conflict with step 2 at T=10"
    }}
  ]
}}
</JSON>"""


def _parse_proposals(raw: str) -> List[NegotiationProposal]:
    data = extract_json(raw)
    result = []
    for item in data.get("proposals", []):
        if not isinstance(item, dict):
            continue
        sid     = safe_int(item.get("step_id", -1), -1)
        change  = str(item.get("proposed_change", "")).strip()
        reason  = str(item.get("reason", "")).strip()
        if sid >= 0 and change:
            result.append(NegotiationProposal(sid, change, reason))
    return result


def _apply_proposals_to_plan(
    plan_steps: List[Dict],
    proposals:  List[NegotiationProposal],
    locked_ids: Set[int],
) -> List[Dict]:
    """
    제안을 플랜에 적용한다. 잠긴 스텝은 수정하지 않는다.
    proposals는 자연어로 변경을 기술하므로 LLM 호출 없이 가능한 부분만 적용:
      - time_min 조정: "shift time_min to X" 패턴 탐지
      - 그 외 변경은 notes에 기록 (joint plan 시 LLM이 참고)
    """
    plan = [dict(s) for s in plan_steps]
    sid_map = {s["step_id"]: i for i, s in enumerate(plan)}

    for prop in proposals:
        if prop.step_id in locked_ids:
            continue
        if prop.step_id not in sid_map:
            continue
        idx = sid_map[prop.step_id]

        # time_min 조정 파턴 탐지
        m = re.search(r"time_min\s+(?:from\s+\d+\s+)?to\s+(\d+)", prop.proposed_change, re.I)
        if m:
            new_t = max(0, min(30, int(m.group(1))))
            plan[idx]["time_min"] = new_t

        # notes에 제안 내용 추가 (finalize 시 참고)
        existing_notes = plan[idx].get("notes", "")
        plan[idx]["notes"] = (
            (existing_notes + " | " if existing_notes else "")
            + f"[NEG R{prop.step_id}] {prop.proposed_change}"
        )

    return plan


def _lock_agreed_steps(
    proposals_a: List[NegotiationProposal],
    proposals_b: List[NegotiationProposal],
    conflict_step_ids: Set[int],
    existing_locked: Set[int],
) -> Set[int]:
    """
    두 에이전트가 같은 step_id에 대해 제안이 없거나 ACCEPT한 경우 lock.
    즉, conflict_step_ids 중 양측이 이의를 제기하지 않은 step은 lock.
    """
    contested = {p.step_id for p in proposals_a + proposals_b}
    newly_locked = conflict_step_ids - contested
    return existing_locked | newly_locked


def phase4_negotiation(
    plan_a:    LocalPlan,
    plan_b:    LocalPlan,
    offer_a:   Offer,
    offer_b:   Offer,
    conflicts: List[ConflictEntry],
    img_a:     str,
    img_b:     str,
    task:      str,
    use_handoff: bool = True,
    verbose:   str = "full",
) -> Tuple[List[Dict], List[Dict], List[NegotiationRound]]:
    """
    P2P 협상: 최대 MAX_NEGOTIATION_ROUNDS 라운드.
    충돌이 있는 스텝만 협상 대상으로 삼고, 합의된 스텝은 라운드마다 lock한다.

    Returns:
        (negotiated_steps_a, negotiated_steps_b, rounds_history)
    """
    _banner("PHASE 4 — PEER-TO-PEER NEGOTIATION")

    if not conflicts:
        print("  No conflicts → skipping negotiation.")
        return (
            plan_steps_to_dicts(plan_a.steps),
            plan_steps_to_dicts(plan_b.steps),
            [],
        )

    conflict_step_ids: Set[int] = {sid for c in conflicts for sid in c.step_ids}
    print(f"  Conflict step IDs to negotiate: {sorted(conflict_step_ids)}")
    print(f"  Max rounds: {MAX_NEGOTIATION_ROUNDS}")

    cur_a: List[Dict] = plan_steps_to_dicts(plan_a.steps)
    cur_b: List[Dict] = plan_steps_to_dicts(plan_b.steps)
    locked: Set[int]  = set()
    rounds: List[NegotiationRound] = []

    prev_props_a: List[NegotiationProposal] = []
    prev_props_b: List[NegotiationProposal] = []

    for rnd in range(1, MAX_NEGOTIATION_ROUNDS + 1):
        # 남은 충돌만 협상
        remaining = [
            c for c in conflicts
            if not all(sid in locked for sid in c.step_ids) or not c.step_ids
        ]
        if not remaining:
            print(f"\n  Round {rnd}: All conflicts resolved → stopping early.")
            break

        print(f"\n  ── Round {rnd}/{MAX_NEGOTIATION_ROUNDS} "
              f"(remaining conflicts: {len(remaining)}, locked: {sorted(locked)}) ──")

        # Agent A의 제안 생성
        prompt_a = _build_negotiation_prompt(
            "agent_A", offer_a, offer_b,
            cur_a, cur_b,
            remaining, locked, rnd, prev_props_b, task,
        )
        raw_a, _ = run_vlm(img_a, prompt_a)
        props_a  = _parse_proposals(raw_a)

        # Agent B의 제안 생성
        prompt_b = _build_negotiation_prompt(
            "agent_B", offer_b, offer_a,
            cur_a, cur_b,
            remaining, locked, rnd, prev_props_a, task,
        )
        raw_b, _ = run_vlm(img_b, prompt_b)
        props_b  = _parse_proposals(raw_b)

        if verbose == "full":
            _log(f"ROUND {rnd} AGENT A RAW", raw_a)
            _log(f"ROUND {rnd} AGENT B RAW", raw_b)

        if verbose in ("full", "summary"):
            print(f"  A proposals ({len(props_a)}): "
                  + ", ".join(f"step{p.step_id}→'{p.proposed_change[:40]}...'" for p in props_a))
            print(f"  B proposals ({len(props_b)}): "
                  + ", ".join(f"step{p.step_id}→'{p.proposed_change[:40]}...'" for p in props_b))

        # 제안 적용
        cur_a = _apply_proposals_to_plan(cur_a, props_b, locked)  # A plan ← B 제안
        cur_b = _apply_proposals_to_plan(cur_b, props_a, locked)  # B plan ← A 제안

        # 이번 라운드에서 합의된 step lock
        locked = _lock_agreed_steps(props_a, props_b, conflict_step_ids, locked)

        round_result = NegotiationRound(
            round_num       = rnd,
            proposals_a     = props_a,
            proposals_b     = props_b,
            locked_step_ids = sorted(locked),
        )
        rounds.append(round_result)

        print(f"  → Locked after round {rnd}: {sorted(locked)}")

        prev_props_a = props_a
        prev_props_b = props_b

    print(f"\n  Negotiation complete. Total rounds: {len(rounds)}")
    print(f"  Final locked steps: {sorted(locked)}")
    return cur_a, cur_b, rounds


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 5: CONVERGENCE CHECK (rule-based)
# ──────────────────────────────────────────────────────────────────────────────

def _has_cycle(steps: List[Dict]) -> bool:
    """Kahn's algorithm으로 depends_on cycle 탐지."""
    from collections import deque
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
    steps_a:  List[Dict],
    steps_b:  List[Dict],
    offer_a:  Offer,
    offer_b:  Offer,
    conflicts: List[ConflictEntry],
) -> ConvergenceResult:
    """
    Rule-based 수렴 판단 (LLM 미사용).

    조건 1 — PASS sender-receiver 매칭:
        모든 PASS step의 target_agent 쪽 플랜에
        depends_on으로 그 step_id를 참조하는 스텝이 존재해야 한다.

    조건 2 — dependency cycle 없음:
        합산 플랜에서 depends_on 그래프에 cycle이 없어야 한다.

    조건 3 — observability:
        각 에이전트의 모든 액션이 자기 obs_scope 키워드와
        최소 1개 이상 겹치거나, INFORM/receive 스텝이어야 한다.
    """
    _banner("PHASE 5 — CONVERGENCE CHECK (rule-based)")
    all_steps = steps_a + steps_b

    # -- 조건 1: PASS 매칭 -------------------------------------------------------
    pass_step_ids = {
        s["step_id"] for s in all_steps
        if s.get("handoff_type") == "PASS"
    }
    all_depends = {dep for s in all_steps for dep in s.get("depends_on", [])}
    unmatched_pass = pass_step_ids - all_depends
    pass_matched = len(unmatched_pass) == 0

    # -- 조건 2: cycle 없음 -------------------------------------------------------
    no_dep_cycle = not _has_cycle(all_steps)

    # -- 조건 3: observability ---------------------------------------------------
    scope_a = set(re.findall(r"\w+", offer_a.obs_scope.lower()))
    scope_b = set(re.findall(r"\w+", offer_b.obs_scope.lower()))
    obs_ok  = True

    for s in all_steps:
        action = s.get("action", "").lower()
        if action.startswith(("inform", "receive")):
            continue
        if s.get("handoff_type") == "PASS":
            continue
        scope = scope_a if s.get("agent_id") == "agent_A" else scope_b
        kw    = _resource_keywords(s.get("action", ""))
        if kw and scope and not (kw & scope):
            obs_ok = False
            break

    # -- 미해결 충돌 목록 ----------------------------------------------------------
    # 수렴 후에도 남은 CANNOT_DO / REDUNDANCY만 경고
    unresolved = [
        c for c in conflicts
        if c.conflict_type in (ConflictType.CANNOT_DO, ConflictType.REDUNDANCY)
    ]

    converged = pass_matched and no_dep_cycle and obs_ok

    print(f"  PASS matched     : {'✓' if pass_matched else '✗'} (unmatched={unmatched_pass})")
    print(f"  No dep cycle     : {'✓' if no_dep_cycle else '✗'}")
    print(f"  Observability OK : {'✓' if obs_ok else '✗'}")
    print(f"  → Converged      : {'YES ✓' if converged else 'NO — will auto-finalize anyway'}")
    if unresolved:
        print(f"  ⚠ Residual conflicts ({len(unresolved)}):")
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
# PHASE 6: DEFERRED HUMAN QUERY (필요 시에만)
# ──────────────────────────────────────────────────────────────────────────────

def _detect_contradiction(offer_a: Offer, offer_b: Offer) -> List[str]:
    result = []
    b_cannot = [c.action for c in offer_b.cannot_do]
    a_cannot = [c.action for c in offer_a.cannot_do]
    for action in offer_a.can_do:
        matches = [bc for bc in b_cannot if _fuzzy_match(action, bc, min_overlap=3)]
        if matches:
            result.append(f"A can_do '{action}' conflicts with B cannot_do '{matches[0]}'")
    for action in offer_b.can_do:
        matches = [ac for ac in a_cannot if _fuzzy_match(action, ac, min_overlap=3)]
        if matches:
            result.append(f"B can_do '{action}' conflicts with A cannot_do '{matches[0]}'")
    return result


def _generate_hq_question(
    trigger_type: str,
    detail: str,
    offer_a: Offer,
    offer_b: Offer,
    leader_img: str,
) -> str:
    prompt = f"""You are coordinating two home agents working on a task.

Agent A ({offer_a.room_type}) can do: {json.dumps(offer_a.can_do, ensure_ascii=False)}
Agent B ({offer_b.room_type}) can do: {json.dumps(offer_b.can_do, ensure_ascii=False)}

Issue detected ({trigger_type}): {detail}

Generate ONE clear, specific, actionable question to ask the human operator.
The question should help resolve the issue above.
Write ONLY the question, no preamble or explanation.
Keep it under 2 sentences."""

    question, _ = run_vlm(leader_img, prompt)
    question = question.strip().strip('"').strip("'")
    if not question or len(question) > 300:
        return f"[{trigger_type}] {detail} — How should the agents handle this?"
    return question


def phase6_human_query(
    plan_a:     LocalPlan,
    plan_b:     LocalPlan,
    offer_a:    Offer,
    offer_b:    Offer,
    convergence: ConvergenceResult,
    img_a:      str,
    img_b:      str,
    use_human_query: bool = True,
) -> Tuple[Dict[str, str], List[str], List[str]]:
    """
    수렴 체크 후에도 해결되지 않은 충돌이나 불확실성이 높은 경우에만 human query를 발동한다.
    수렴했으면 skip.
    """
    _banner("PHASE 6 — DEFERRED HUMAN QUERY")

    if not use_human_query:
        print("  [ABLATION] Human query disabled.")
        return {}, [], []

    if convergence.converged and not convergence.unresolved_conflicts:
        print("  Plan converged with no residual issues → no human query needed.")
        return {}, [], []

    # leader는 score 높은 쪽으로 (간단히 plan_a 기준)
    leader_img = img_a
    triggered: List[str] = []
    raw_triggers: List[Tuple[str, str, float]] = []

    # 미매칭 PASS
    if not convergence.pass_matched:
        detail = "Some PASS handoffs have no matching receiver step after negotiation"
        triggered.append(f"[HQ1] {detail}")
        raw_triggers.append(("UNMATCHED_PASS", detail, 0.90))

    # cycle 존재
    if not convergence.no_dep_cycle:
        detail = "Dependency cycle detected in the joint plan after negotiation"
        triggered.append(f"[HQ2] {detail}")
        raw_triggers.append(("DEP_CYCLE", detail, 0.85))

    # 잔존 충돌
    for c in convergence.unresolved_conflicts:
        detail = c.description
        triggered.append(f"[HQ3-{c.conflict_type}] {detail}")
        raw_triggers.append((c.conflict_type, detail, 0.80))

    # 높은 불확실성
    if plan_a.U_plan > UNCERTAINTY_THRESH or plan_b.U_plan > UNCERTAINTY_THRESH:
        detail = f"Plan uncertainty remains high (A:{plan_a.U_plan:.3f}, B:{plan_b.U_plan:.3f})"
        triggered.append(f"[HQ4] {detail}")
        raw_triggers.append(("HIGH_UNCERTAINTY", detail, 0.70))

    # 미충족 need
    all_provides = offer_a.can_provide + offer_b.can_provide
    for need in offer_a.need_from_other + offer_b.need_from_other:
        if not any(_fuzzy_match_soft(need, p) for p in all_provides):
            detail = f"No agent can provide: '{need}'"
            triggered.append(f"[HQ5] Unmatched need: '{need}'")
            raw_triggers.append(("UNMATCHED_NEED", detail, 0.85))

    if not triggered:
        print("  No human query needed.")
        return {}, [], []

    print(f"  Triggers:")
    for t in triggered:
        print(f"    {t}")

    raw_triggers.sort(key=lambda x: -x[2])
    answers: Dict[str, str] = {}
    asked: List[str] = []

    for i, (ttype, detail, u) in enumerate(raw_triggers[:HQ_TOP_K], 1):
        print(f"\n  Generating Q{i} [{ttype}, priority={u:.2f}]...", end=" ", flush=True)
        question = _generate_hq_question(ttype, detail, offer_a, offer_b, leader_img)
        print("done")
        print(f"  Q{i}: {question}")
        asked.append(question)
        try:
            ans = input("  A: ").strip()
        except EOFError:
            ans = ""
        if ans:
            answers[question] = ans

    return answers, triggered, asked


# ──────────────────────────────────────────────────────────────────────────────
# FINALIZE: JOINT PLAN 확정 (rule-based auto-finalize)
# ──────────────────────────────────────────────────────────────────────────────

FINALIZE_FEW_SHOT = """
EXAMPLE JOINT PLAN (agent_A sends snacks to agent_B):
<JSON>
{
  "joint_plan": [
    {
      "step_id": 1, "time_min": 0,
      "room": "kitchen", "agent_id": "agent_A",
      "action": "place fruits from island onto serving tray",
      "preconditions": [], "depends_on": [],
      "handoff_type": null, "target_agent": null, "notes": "PARALLEL with step 2"
    },
    {
      "step_id": 2, "time_min": 0,
      "room": "living room", "agent_id": "agent_B",
      "action": "arrange cushions on sofa",
      "preconditions": [], "depends_on": [],
      "handoff_type": null, "target_agent": null, "notes": "PARALLEL with step 1"
    },
    {
      "step_id": 3, "time_min": 10,
      "room": "kitchen", "agent_id": "agent_A",
      "action": "carry snack tray to kitchen doorway for agent_B pickup",
      "preconditions": ["snacks arranged on tray"], "depends_on": [1],
      "handoff_type": "PASS", "target_agent": "agent_B",
      "notes": "snack tray ready at doorway"
    },
    {
      "step_id": 4, "time_min": 11,
      "room": "living room", "agent_id": "agent_B",
      "action": "receive snack tray and place on coffee table",
      "preconditions": ["snack tray at doorway"], "depends_on": [3],
      "handoff_type": null, "target_agent": null,
      "notes": "receives from agent_A PASS step 3"
    },
    {
      "step_id": 5, "time_min": 15,
      "room": "kitchen", "agent_id": "agent_A",
      "action": "INFORM agent_B: kitchen preparation complete",
      "preconditions": [], "depends_on": [3],
      "handoff_type": "INFORM", "target_agent": "agent_B",
      "notes": "kitchen ready"
    }
  ]
}
</JSON>

CRITICAL HANDOFF RULES:
- PASS is declared by the SENDER only. Receiver adds a step with depends_on=[PASS step_id], NO handoff_type.
- INFORM is for status notification only, no physical item transfer.
- room: EXACT room name from offer
- agent_id: EXACTLY "agent_A" or "agent_B"
- Total steps: 6–10 only.
- Both agents must finish within 5 minutes of each other.
""".strip()


def _build_finalize_prompt(
    steps_a:       List[Dict],
    steps_b:       List[Dict],
    offer_a:       Offer,
    offer_b:       Offer,
    human_answers: Dict[str, str],
    convergence:   ConvergenceResult,
    task:          str,
) -> str:
    hq_str = (
        "\n".join(f"  Q: {q}\n  A: {a}" for q, a in human_answers.items())
        if human_answers else "  (none)"
    )
    convergence_notes = []
    if not convergence.pass_matched:
        convergence_notes.append("⚠ Some PASS steps still unmatched — add receiver steps.")
    if not convergence.no_dep_cycle:
        convergence_notes.append("⚠ Dependency cycle detected — remove circular depends_on.")
    if not convergence.observability_ok:
        convergence_notes.append("⚠ Some steps exceed obs_scope — restrict to visible objects.")
    conv_str = "\n".join(convergence_notes) if convergence_notes else "✓ All convergence conditions met."

    return f"""You are finalizing a collaborative plan for: "{task}"

AGENT A ({offer_a.room_type}):
  can_do   : {json.dumps(offer_a.can_do, ensure_ascii=False)}
  obs_scope: {offer_a.obs_scope}

AGENT B ({offer_b.room_type}):
  can_do   : {json.dumps(offer_b.can_do, ensure_ascii=False)}
  obs_scope: {offer_b.obs_scope}

NEGOTIATED PLAN A (post-negotiation):
{jdump(steps_a)}

NEGOTIATED PLAN B (post-negotiation):
{jdump(steps_b)}

CONVERGENCE STATUS:
{conv_str}

HUMAN QUERY ANSWERS:
{hq_str}

{FINALIZE_FEW_SHOT}

Merge the two plans into a single coherent Joint Plan:
1. COMPLETENESS   : cover all objectives from both plans.
2. EXECUTABILITY  : each step within the assigned agent's can_do.
3. OBSERVABILITY  : agent_A uses ONLY {offer_a.room_type}; agent_B ONLY {offer_b.room_type}.
4. SEQUENTIAL     : depends_on must reference valid step_ids.
5. LOAD BALANCE   : both agents finish within 5 minutes of each other.
6. PASS RESOLVE   : every PASS sender must have a receiver step (depends_on=[PASS step_id]).
7. Incorporate HUMAN QUERY ANSWERS to resolve any remaining ambiguities.
8. Fix any remaining convergence issues noted above.

Output rules:
- step_ids: sequential integers (1, 2, 3, ...), no duplicates
- time_min: integer in [0, 30], spread evenly
- Total steps: 6–10
- Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "joint_plan": [
    {{
      "step_id": 1, "time_min": 0,
      "room": "{offer_a.room_type}", "agent_id": "agent_A",
      "action": "verb + specific object + detail",
      "preconditions": [], "depends_on": [],
      "handoff_type": null, "target_agent": null, "notes": ""
    }}
  ]
}}
</JSON>"""


def _parse_joint(
    raw: str, offer_a: Offer, offer_b: Offer
) -> List[Dict]:
    data     = extract_json(raw)
    raw_plan = data.get("joint_plan", [])
    if not isinstance(raw_plan, list):
        return []

    room_map = {"agent_A": offer_a.room_type, "agent_B": offer_b.room_type}
    seen_ids: set = set()
    cleaned: List[Dict] = []

    for i, step in enumerate(raw_plan, start=1):
        if not isinstance(step, dict):
            continue
        action    = str(step.get("action", "")).strip()
        raw_agent = str(step.get("agent_id", "")).strip()
        raw_room  = str(step.get("room", "")).strip()
        if not action or "|" in raw_agent or "|" in raw_room:
            continue
        if raw_agent not in VALID_AGENTS:
            continue

        sid = safe_int(step.get("step_id", i), i)
        while sid in seen_ids:
            sid += 1
        seen_ids.add(sid)

        cleaned.append({
            "step_id":       sid,
            "time_min":      max(0, min(30, safe_int(step.get("time_min", 0), 0))),
            "room":          raw_room,
            "agent_id":      raw_agent,
            "action":        action,
            "preconditions": [str(x).strip() for x in step.get("preconditions", []) if str(x).strip()],
            "depends_on":    _norm_depends(step.get("depends_on")),
            "handoff_type":  _norm_handoff(step.get("handoff_type")),
            "target_agent":  _norm_agent(step.get("target_agent")),
            "notes":         str(step.get("notes", "")).strip(),
        })

    # PASS 수신 스텝 자동 보완
    pass_steps    = {s["step_id"]: s for s in cleaned if s.get("handoff_type") == "PASS"}
    pass_received = {dep for s in cleaned for dep in s.get("depends_on", []) if dep in pass_steps}

    for sid, pass_s in pass_steps.items():
        if sid not in pass_received and pass_s.get("target_agent") in room_map:
            target = pass_s["target_agent"]
            auto = {
                "step_id":       max((s["step_id"] for s in cleaned), default=0) + 1,
                "time_min":      min(30, pass_s["time_min"] + 1),
                "room":          room_map[target],
                "agent_id":      target,
                "action":        f"receive and place: {pass_s['action']}",
                "preconditions": [f"step {sid} completed"],
                "depends_on":    [sid],
                "handoff_type":  None,
                "target_agent":  None,
                "notes":         "auto-generated PASS receiver",
            }
            cleaned.append(auto)
            print(f"  [AUTO-PASS] step {auto['step_id']} added for {target}")

    # step_id 재번호 (time_min 순)
    sorted_plan = sorted(cleaned, key=lambda x: (x["time_min"], x["step_id"]))
    old_to_new: Dict[int, int] = {}
    for new_id, s in enumerate(sorted_plan, start=1):
        old_to_new[s["step_id"]] = new_id
    for s in sorted_plan:
        s["step_id"]    = old_to_new[s["step_id"]]
        s["depends_on"] = [old_to_new[d] for d in s["depends_on"] if d in old_to_new]
        s["preconditions"] = [
            f"step {old_to_new[int(m.group(1))]} completed"
            if (m := re.match(r"step (\d+) completed", p)) and int(m.group(1)) in old_to_new
            else p
            for p in s["preconditions"]
        ]

    return sorted_plan


def phase_finalize(
    steps_a:       List[Dict],
    steps_b:       List[Dict],
    offer_a:       Offer,
    offer_b:       Offer,
    human_answers: Dict[str, str],
    convergence:   ConvergenceResult,
    task:          str,
    img_a:         str,
    img_b:         str,
    verbose:       str = "full",
) -> List[Dict]:
    """
    협상된 두 플랜을 단일 Joint Plan으로 확정한다.
    수렴 조건 충족 여부와 human query 답변을 반영한다.
    """
    _banner("FINALIZE — JOINT PLAN")

    prompt = _build_finalize_prompt(
        steps_a, steps_b, offer_a, offer_b, human_answers, convergence, task
    )
    # 리더 역할 없이 agent_A 이미지로 finalize (P2P 구조에서 leader 불필요)
    MAX_RETRY = 3
    raw = ""
    for attempt in range(MAX_RETRY):
        raw, _ = run_vlm(img_a, prompt)
        refusal_kw = ["sorry", "can't assist", "cannot assist", "i'm not able", "unable to"]
        if not any(kw in raw.lower() for kw in refusal_kw):
            break
        print(f"  [RETRY {attempt+1}/{MAX_RETRY}] model refused, retrying...")

    if verbose == "full":
        _log("RAW JOINT PLAN", raw)

    joint = _parse_joint(raw, offer_a, offer_b)

    if verbose in ("full", "summary"):
        _log("PARSED JOINT PLAN", jdump(joint))

    return joint
