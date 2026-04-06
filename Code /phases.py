# ══════════════════════════════════════════════════════════════════════════════
# phases.py
# Phase 1: Observation & Offer
# Phase 2: Leader Election
# Phase 3: Local Planning
# Phase 4a: Human Query
# Phase 4b: Joint Planning
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from config import (
    ALPHA, BETA, GAMMA, DELTA,
    HQ_TOP_K, MAX_CAN_DO, MAX_CANNOT_DO,
    UNCERTAINTY_THRESH, VALID_AGENTS,
)
from models import (
    CannotEntry, HQEntry, Handoff, LeaderResult,
    LocalPlan, Offer, PlanStep,
)
from utils import (
    _banner, _fuzzy_match, _fuzzy_match_soft, _log,
    _match_conf, _norm_agent, _norm_depends, _norm_handoff, _norm_reason,
    clamp01, compute_plan_uncertainty, compute_token_uncertainty,
    extract_json, jdump, safe_int,
)
from vlm import run_vlm


# ──────────────────────────────────────────────────────────────────────────────
# Offer 직렬화 헬퍼
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


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1: OBSERVATION & OFFER GENERATION
# ──────────────────────────────────────────────────────────────────────────────

PHASE1_FEW_SHOT = """
EXAMPLE — home agent observing a room:
<JSON>
{
  "room_type": "describe the room type you see (e.g. kitchen, bedroom, bathroom, living room)",
  "observation": "Wooden shelf with decorative items visible. Appliances and counter surfaces present.",
  "obs_scope": "counter surface, shelf, appliances, sink, cabinets, floor area",
  "can_do": [
    "wipe counter surface with cloth",
    "organize items on shelf",
    "check for sharp objects on visible surfaces",
    "secure loose items on shelf",
    "clean visible sink with sponge"
  ],
  "cannot_do": [
    {"action": "check contents of closed cabinets", "reason": "NO_OBJECT"},
    {"action": "inspect items behind appliances", "reason": "NO_OBJECT"},
    {"action": "assess floor area outside camera view", "reason": "NO_OBJECT"}
  ],
  "conf": {
    "wipe counter surface with cloth": 0.95,
    "organize items on shelf": 0.85,
    "check for sharp objects on visible surfaces": 0.9,
    "secure loose items on shelf": 0.85,
    "clean visible sink with sponge": 0.9
  },
  "can_provide": ["secured visible surface area", "organized shelf"],
  "need_from_other": ["confirmation that other room hazards are addressed"]
}
</JSON>

IMPORTANT — cannot_do should list things you CANNOT do because the object
is NOT VISIBLE in your current camera view (not simply because it belongs to another room).
""".strip()


def build_phase1_prompt(task: str) -> str:
    return f"""You are an embodied home agent. Observe your room through the camera image.

Global task: "{task}"

{PHASE1_FEW_SHOT}

Analyze YOUR room and produce an Offer.

STRICT RULES:
1. Use ONLY objects directly visible in the image.
2. can_do: max {MAX_CAN_DO} unique items. Format: "<verb> <specific visible object> [detail]"
3. cannot_do: max {MAX_CANNOT_DO} items. Reason: NO_OBJECT | NO_CAPABILITY | UNCERTAIN
   Focus on things NOT VISIBLE in your current view.
4. can_do and cannot_do must NOT share any action.
5. conf: confidence per action [0.0–1.0]. Keys must exactly match can_do items.
6. can_provide / need_from_other: concrete items for inter-agent coordination.
8. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "room_type": "your room type",
  "observation": "one concise sentence",
  "obs_scope": "list of observable areas and objects",
  "can_do": ["verb + specific visible object + detail"],
  "cannot_do": [{{"action": "specific action", "reason": "NO_OBJECT"}}],
  "conf": {{"action": 0.9}},
  "can_provide": ["concrete result"],
  "need_from_other": ["concrete item needed"]
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

    conf_raw = {str(k).strip(): clamp01(v) for k, v in data.get("conf", {}).items()}
    return Offer(
        agent_id        = agent_id,
        room_type       = str(data.get("room_type", "")).strip(),
        observation     = str(data.get("observation", "")).strip(),
        obs_scope       = str(data.get("obs_scope", "")).strip(),
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
# PHASE 2: TASK-ADAPTIVE LEADER ELECTION
# ──────────────────────────────────────────────────────────────────────────────

def _leader_score(o: Offer) -> float:
    coverage  = len(o.can_do) / MAX_CAN_DO
    mean_conf = sum(o.conf.values()) / max(len(o.conf), 1)
    unc_ratio = o.uncertain_count / max(len(o.can_do), 1)
    self_suff = len(o.can_provide) / (len(o.can_provide) + len(o.need_from_other) + 1)
    return round(ALPHA * coverage + BETA * mean_conf - GAMMA * unc_ratio + DELTA * self_suff, 4)


def phase2_leader(
    offer_a: Offer, offer_b: Offer, use_leader_election: bool = True
) -> LeaderResult:
    _banner("PHASE 2 — TASK-ADAPTIVE LEADER ELECTION")
    sa = _leader_score(offer_a)
    sb = _leader_score(offer_b)

    for o, s in [(offer_a, sa), (offer_b, sb)]:
        mc = round(sum(o.conf.values()) / max(len(o.conf), 1), 3)
        ss = round(len(o.can_provide) / (len(o.can_provide) + len(o.need_from_other) + 1), 3)
        print(f"  [{o.agent_id} | {o.room_type}]  score={s}")
        print(f"    can_do={len(o.can_do)}/{MAX_CAN_DO}, mean_conf={mc}, uncertain={o.uncertain_count}, self_suff={ss}")

    if not use_leader_election:
        print("\n  [ABLATION] Leader election disabled. agent_A fixed as leader.")
        return LeaderResult("agent_A", "agent_B", sa, sb, "fixed=agent_A")

    if sa >= sb:
        lid, fid, reason = "agent_A", "agent_B", f"agent_A elected (score {sa} >= {sb})"
    else:
        lid, fid, reason = "agent_B", "agent_A", f"agent_B elected (score {sb} > {sa})"

    print(f"\n  → LEADER  : {lid}")
    print(f"  → FOLLOWER: {fid}")
    print(f"  → {reason}")
    return LeaderResult(lid, fid, sa, sb, reason)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3: OFFER-CONDITIONED LOCAL PLANNING
# ──────────────────────────────────────────────────────────────────────────────

PHASE3_FEW_SHOT = """
EXAMPLE (home agent, LEADER):
<JSON>
{
  "plan_steps": [
    {
      "step_id": 1, "time_min": 0,
      "action": "organize shelf to remove fragile or hazardous items",
      "preconditions": [], "depends_on": [],
      "handoff_type": null, "target_agent": null,
      "uncertainty": 0.1, "human_query": null, "notes": ""
    },
    {
      "step_id": 2, "time_min": 5,
      "action": "check for sharp objects on visible surfaces",
      "preconditions": [], "depends_on": [],
      "handoff_type": null, "target_agent": null,
      "uncertainty": 0.15, "human_query": null, "notes": ""
    },
    {
      "step_id": 3, "time_min": 15,
      "action": "INFORM agent_B: this room secured, safe to proceed",
      "preconditions": ["sharp objects removed"], "depends_on": [2],
      "handoff_type": "INFORM", "target_agent": "agent_B",
      "uncertainty": 0.1, "human_query": null,
      "notes": "payload: room hazards cleared at T=15min"
    }
  ]
}
</JSON>

CRITICAL — uncertainty: 0.0=certain, 1.0=uncertain. High confidence → LOW uncertainty (0.1–0.3).
Only write human_query if uncertainty >= 0.5.
""".strip()


def build_phase3_prompt(
    my: Offer, other: Offer, leader_id: str, task: str, use_offer: bool = True
) -> str:
    role = "LEADER" if my.agent_id == leader_id else "FOLLOWER"
    matched_needs    = [n for n in my.need_from_other if any(_fuzzy_match_soft(n, p) for p in other.can_provide)]
    matched_provides = [p for p in my.can_provide if any(_fuzzy_match_soft(p, n) for n in other.need_from_other)]

    if use_offer:
        context = f"""YOUR OFFER:
{jdump(offer_to_dict(my))}

OTHER AGENT'S OFFER ({other.room_type}):
{jdump(offer_to_dict(other))}

MATCHED HANDOFF OPPORTUNITIES:
Items other agent can provide to YOU: {json.dumps(matched_needs, ensure_ascii=False)}
Items YOU can provide to other agent: {json.dumps(matched_provides, ensure_ascii=False)}"""
    else:
        context = f"YOUR ROOM: {my.room_type}\nOTHER AGENT'S ROOM: {other.room_type}"

    return f"""You are the {my.room_type} agent. Role: {role}.

Global task: "{task}"

{context}

{PHASE3_FEW_SHOT}

Generate your LOCAL PLAN:
1. OBSERVABILITY: steps ONLY for your room ({my.room_type}).
2. EXECUTABILITY: every action physically possible in your room.
3. Generate 4–6 steps spread over 0–30 minutes.
4. UNCERTAINTY [0.0–1.0]: set LOW for confident actions.
5. HANDOFFS — use ONLY when truly needed:
   - RELAY: use ONLY when a physical item must cross room boundaries
     (e.g. agent_A prepares food → agent_B picks it up in their room)
     DO NOT use RELAY for actions you perform entirely within your own room.
   - INFORM: use to notify the other agent of completed status or key info.
6. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "plan_steps": [
    {{
      "step_id": 1, "time_min": 0,
      "action": "verb + specific visible object + detail",
      "preconditions": [], "depends_on": [],
      "handoff_type": null, "target_agent": null,
      "uncertainty": 0.1, "human_query": null, "notes": ""
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

        json_unc = clamp01(item.get("uncertainty", 0.2))
        step_unc = clamp01(json_unc * 0.7 + token_unc * 0.3)

        hq_raw = item.get("human_query")
        if isinstance(hq_raw, str) and hq_raw.strip().lower() in {"", "null", "none"}:
            hq_raw = None

        handoff_type = _norm_handoff(item.get("handoff_type")) if use_handoff else None
        target       = _norm_agent(item.get("target_agent"))   if use_handoff else None

        step = PlanStep(
            step_id=sid,
            time_min=max(0, min(30, safe_int(item.get("time_min", 0), 0))),
            room=my.room_type,
            agent_id=my.agent_id,
            action=action,
            preconditions=[str(x).strip() for x in item.get("preconditions", []) if str(x).strip()],
            depends_on=_norm_depends(item.get("depends_on")),
            handoff_type=handoff_type,
            target_agent=target,
            uncertainty=step_unc,
            notes=str(item.get("notes", "")).strip(),
        )
        steps.append(step)

        if step_unc >= UNCERTAINTY_THRESH or hq_raw:
            hq_list.append(HQEntry(sid, hq_raw or f"Is '{action}' feasible?", step_unc))
        if handoff_type:
            payload = step.notes if handoff_type == "INFORM" else ""
            handoffs.append(Handoff(sid, action, handoff_type, target, payload))

    steps.sort(key=lambda s: (s.time_min, s.step_id))
    all_unc = [s.uncertainty for s in steps] if steps else [token_unc]
    return LocalPlan(my.agent_id, steps, compute_plan_uncertainty(all_unc), hq_list, handoffs)


def phase3_local_plan(
    offer_a: Offer, offer_b: Offer, leader: LeaderResult,
    img_a: str, img_b: str, task: str,
    use_offer: bool = True, use_handoff: bool = True,
    verbose: str = "full",
) -> Tuple[LocalPlan, LocalPlan]:
    _banner("PHASE 3 — OFFER-CONDITIONED LOCAL PLANNING")
    prompt_a = build_phase3_prompt(offer_a, offer_b, leader.leader_id, task, use_offer)
    prompt_b = build_phase3_prompt(offer_b, offer_a, leader.leader_id, task, use_offer)

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

    print(f"\n  A: U_plan={plan_a.U_plan:.3f} | steps={len(plan_a.steps)} | handoffs={len(plan_a.handoffs)} | HQ={len(plan_a.hq_list)}")
    print(f"  B: U_plan={plan_b.U_plan:.3f} | steps={len(plan_b.steps)} | handoffs={len(plan_b.handoffs)} | HQ={len(plan_b.hq_list)}")
    for tag, plan in [("A", plan_a), ("B", plan_b)]:
        for h in plan.handoffs:
            print(f"  [{tag}→{h.handoff_type}] step={h.step_id} target={h.target_agent} | {h.action}")

    return plan_a, plan_b


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 4a: DEFERRED HUMAN QUERY
# ──────────────────────────────────────────────────────────────────────────────

def _detect_contradiction(offer_a: Offer, offer_b: Offer) -> List[str]:
    result = []
    b_cannot = [c.action for c in offer_b.cannot_do]
    a_cannot = [c.action for c in offer_a.cannot_do]
    for action in offer_a.can_do:
        matches = [bc for bc in b_cannot if _fuzzy_match(action, bc, min_overlap=2)]
        if matches:
            result.append(f"A can_do '{action}' conflicts with B cannot_do '{matches[0]}'")
    for action in offer_b.can_do:
        matches = [ac for ac in a_cannot if _fuzzy_match(action, ac, min_overlap=2)]
        if matches:
            result.append(f"B can_do '{action}' conflicts with A cannot_do '{matches[0]}'")
    return result


def phase4a_human_query(
    plan_a: LocalPlan, plan_b: LocalPlan,
    offer_a: Offer, offer_b: Offer,
    leader_id: str, use_human_query: bool = True,
) -> Tuple[Dict[str, str], List[str], List[str]]:
    _banner("PHASE 4a — DEFERRED HUMAN QUERY")

    if not use_human_query:
        print("  [ABLATION] Human query disabled. Skipping.")
        return {}, [], []

    triggered: List[str] = []
    all_hq = sorted(plan_a.hq_list + plan_b.hq_list, key=lambda h: -h.u_step)

    if plan_a.U_plan > UNCERTAINTY_THRESH or plan_b.U_plan > UNCERTAINTY_THRESH:
        triggered.append(f"[Q1] High uncertainty A:{plan_a.U_plan:.3f} B:{plan_b.U_plan:.3f}")

    unknown = [h for h in plan_a.handoffs + plan_b.handoffs if h.target_agent is None]
    if unknown:
        triggered.append(f"[Q2] {len(unknown)} handoff(s) with unknown target")
        for h in unknown:
            all_hq.append(HQEntry(h.step_id, f"Who handles '{h.action}'?", 0.8))

    all_provides = offer_a.can_provide + offer_b.can_provide
    seen_needs: set = set()
    for need in offer_a.need_from_other + offer_b.need_from_other:
        if need.lower() in {"none", "nothing", ""} or need in seen_needs:
            continue
        seen_needs.add(need)
        if not any(_fuzzy_match_soft(need, p) for p in all_provides):
            triggered.append(f"[Q3] Unmatched need: '{need}'")
            all_hq.append(HQEntry(0, f"No agent can provide '{need}'. How to handle?", 0.9))

    contradictions = _detect_contradiction(offer_a, offer_b)
    if contradictions:
        triggered.append(f"[Q4] {len(contradictions)} contradiction(s)")
        for c in contradictions[:2]:
            all_hq.append(HQEntry(0, f"Conflict: {c}. Which agent is correct?", 0.85))

    if not triggered:
        print(f"  Leader ({leader_id}): no query needed.")
        return {}, [], []

    print(f"  Leader ({leader_id}) triggered:")
    for c in triggered:
        print(f"    {c}")

    seen_q: set = set()
    deduped: List[HQEntry] = []
    for hq in sorted(all_hq, key=lambda h: -h.u_step):
        if hq.question_nl not in seen_q:
            seen_q.add(hq.question_nl)
            deduped.append(hq)

    answers: Dict[str, str] = {}
    asked_questions: List[str] = []
    for i, hq in enumerate(deduped[:HQ_TOP_K], 1):
        print(f"\n  Q{i} [u={hq.u_step:.2f}]: {hq.question_nl}")
        asked_questions.append(hq.question_nl)
        try:
            ans = input("  A: ").strip()
        except EOFError:
            ans = ""
        if ans:
            answers[hq.question_nl] = ans

    return answers, triggered, asked_questions


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 4b: LEADER-DRIVEN JOINT PLANNING
# ──────────────────────────────────────────────────────────────────────────────

PHASE4_FEW_SHOT = """
EXAMPLE JOINT PLAN:
<JSON>
{
  "joint_plan": [
    {
      "step_id": 1, "time_min": 0,
      "room": "room_A_name", "agent_id": "agent_A",
      "action": "organize shelf to remove fragile or hazardous items",
      "preconditions": [], "depends_on": [],
      "handoff_type": null, "target_agent": null, "notes": "PARALLEL with step 2"
    },
    {
      "step_id": 2, "time_min": 0,
      "room": "room_B_name", "agent_id": "agent_B",
      "action": "arrange seating area for safe use",
      "preconditions": [], "depends_on": [],
      "handoff_type": null, "target_agent": null, "notes": "PARALLEL with step 1"
    },
    {
      "step_id": 5, "time_min": 15,
      "room": "room_A_name", "agent_id": "agent_A",
      "action": "RELAY: place item at room boundary for agent_B pickup",
      "preconditions": ["room_A secured"], "depends_on": [3],
      "handoff_type": "RELAY", "target_agent": "agent_B", "notes": ""
    },
    {
      "step_id": 6, "time_min": 16,
      "room": "room_B_name", "agent_id": "agent_B",
      "action": "receive item and place in designated spot",
      "preconditions": ["item placed by agent_A at step 5"], "depends_on": [5],
      "handoff_type": null, "target_agent": null, "notes": "RELAY receive"
    }
  ],
  "validity": {
    "completeness": "all objectives covered",
    "executability": "all actions within agent can_do",
    "sequential_consistency": "all depends_on resolved",
    "observability": "each agent uses only their own room objects",
    "load_balance": "both agents finish around same time",
    "handoff_resolution": "RELAY step5->step6 resolved"
  }
}
</JSON>

CRITICAL:
- room: use the EXACT room name from each agent's offer (room_type field) — NEVER use "|"
- agent_id: EXACTLY "agent_A" or "agent_B" — NEVER use "|"
- Every RELAY sender MUST have a receiver step with depends_on=[sender_step_id]
""".strip()


def build_phase4_prompt(
    plan_a: LocalPlan, plan_b: LocalPlan,
    offer_a: Offer, offer_b: Offer,
    leader: LeaderResult,
    human_answers: Dict[str, str],
    task: str,
) -> str:
    all_handoffs = plan_a.handoffs + plan_b.handoffs
    handoff_str = "\n".join(
        f"  [{h.handoff_type}] from={'A' if h in plan_a.handoffs else 'B'}: {h.action} → target={h.target_agent}"
        for h in all_handoffs
    ) or "  (none)"
    hq_str = (
        "\n".join(f"  Q: {q}\n  A: {a}" for q, a in human_answers.items())
        if human_answers
        else "  (none)"
    )

    return f"""You are the LEADER agent ({leader.leader_id}).
Merge two local plans into one valid Joint Plan for: "{task}"

AGENT CAPABILITIES:
agent_A ({offer_a.room_type}) can_do: {json.dumps(offer_a.can_do, ensure_ascii=False)}
agent_B ({offer_b.room_type}) can_do: {json.dumps(offer_b.can_do, ensure_ascii=False)}

LOCAL PLAN A:
{jdump(local_plan_to_dict(plan_a))}

LOCAL PLAN B:
{jdump(local_plan_to_dict(plan_b))}

DECLARED HANDOFFS:
{handoff_str}

HUMAN QUERY ANSWERS:
{hq_str}

{PHASE4_FEW_SHOT}

Conditions:
1. COMPLETENESS   : cover all objectives from both local plans.
2. EXECUTABILITY  : each step within the assigned agent's can_do.
3. SEQUENTIAL     : preconditions met by prior steps.
4. OBSERVABILITY  : agent_A uses ONLY {offer_a.room_type}; agent_B uses ONLY {offer_b.room_type}.
5. LOAD BALANCE   : both agents finish within 5 min of each other.
6. HANDOFF RESOLVE: every RELAY sender must have a receiver step.

Rules:
- step_ids: sequential (1, 2, 3, ...), no duplicates
- time_min: [0, 30]
- room: use EXACTLY the room name from each agent's offer — agent_A room is "{offer_a.room_type}", agent_B room is "{offer_b.room_type}"
- agent_id: EXACTLY "agent_A" or "agent_B"
- Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "joint_plan": [
    {{
      "step_id": 1, "time_min": 0,
      "room": "kitchen", "agent_id": "agent_A",
      "action": "verb + specific object + detail",
      "preconditions": [], "depends_on": [],
      "handoff_type": null, "target_agent": null, "notes": ""
    }}
  ],
  "validity": {{
    "completeness": "...", "executability": "...",
    "sequential_consistency": "...", "observability": "...",
    "load_balance": "...", "handoff_resolution": "..."
  }}
}}
</JSON>"""


def _parse_joint(
    raw: str, offer_a: Offer, offer_b: Offer
) -> Tuple[List[Dict], Dict]:
    data     = extract_json(raw)
    raw_plan = data.get("joint_plan", [])
    validity = data.get("validity", {})
    if not isinstance(raw_plan, list):
        return [], validity

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

    # RELAY 수신 스텝 자동 보완
    relay_steps    = {s["step_id"]: s for s in cleaned if s.get("handoff_type") == "RELAY"}
    relay_received = {dep for s in cleaned for dep in s.get("depends_on", []) if dep in relay_steps}

    for sid, relay_s in relay_steps.items():
        if sid not in relay_received and relay_s.get("target_agent") in room_map:
            target = relay_s["target_agent"]
            auto = {
                "step_id":       max((s["step_id"] for s in cleaned), default=0) + 1,
                "time_min":      min(30, relay_s["time_min"] + 1),
                "room":          room_map[target],
                "agent_id":      target,
                "action":        f"receive and complete: {relay_s['action']}",
                "preconditions": [f"step {sid} completed"],
                "depends_on":    [sid],
                "handoff_type":  None,
                "target_agent":  None,
                "notes":         "auto-generated RELAY receiver",
            }
            cleaned.append(auto)
            print(f"  [AUTO-RELAY] step {auto['step_id']} added for {target}")

    return sorted(cleaned, key=lambda x: (x["time_min"], x["step_id"])), validity


def phase4b_joint_plan(
    plan_a: LocalPlan, plan_b: LocalPlan,
    offer_a: Offer, offer_b: Offer,
    leader: LeaderResult,
    human_answers: Dict[str, str],
    task: str,
    img_a: str, img_b: str,
    verbose: str = "full",
) -> Tuple[List[Dict], Dict]:
    _banner("PHASE 4b — LEADER-DRIVEN JOINT PLANNING")
    prompt     = build_phase4_prompt(plan_a, plan_b, offer_a, offer_b, leader, human_answers, task)
    leader_img = img_a if leader.leader_id == "agent_A" else img_b
    raw, _     = run_vlm(leader_img, prompt)

    if verbose == "full":
        _log("RAW JOINT PLAN", raw)

    joint, validity = _parse_joint(raw, offer_a, offer_b)

    if verbose in ("full", "summary"):
        _log("PARSED JOINT PLAN", jdump(joint))

    return joint, validity
