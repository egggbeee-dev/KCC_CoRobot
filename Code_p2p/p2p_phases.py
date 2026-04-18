# phases.py
#
# PASS/INFORM 방식 복원. 핵심 수정사항:
#   - can_provide를 물리적 아이템만으로 제한 (Phase 1 프롬프트)
#   - _ensure_pass_steps: offer 매칭으로 PASS 누락 시 코드가 보완
#   - _auto_add_receivers: PASS에 대응하는 receive step 자동 삽입
#   - _normalize_pass: 비정상 PASS 제거 (공간/상태, 중복, target없음)
#   - format_joint_plan: 깔끔한 자연어 출력

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
    NON_PASSABLE_KW, UNCERTAINTY_THRESH, VALID_AGENTS,
    VALID_HANDOFFS, VALID_PROPOSAL_FIELDS,
)
from p2p_models import (
    CannotEntry, ConflictEntry, ConflictType, ConvergenceResult,
    Handoff, HQEntry, LocalPlan, NegotiationProposal,
    NegotiationRound, Offer, PlanStep,
)
from p2p_utils import (
    _banner, _fuzzy_match, _fuzzy_match_soft, _log,
    _match_conf, _norm_agent, _norm_depends, _norm_handoff,
    _norm_reason, clamp01, compute_plan_uncertainty,
    compute_token_uncertainty, extract_json, jdump, safe_int,
)
from p2p_vlm import run_vlm


# ── 병렬 VLM ─────────────────────────────────────────────────────────────────

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
        "handoffs": [asdict(h) for h in lp.handoffs],
    }


def plan_steps_to_dicts(steps: List[PlanStep]) -> List[Dict]:
    return [asdict(s) for s in steps]


# ── 키워드 유틸 ───────────────────────────────────────────────────────────────

def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _kw(text: str) -> Set[str]:
    return {_stem(w) for w in set(re.findall(r"\w+", text.lower())) - FUZZY_STOPWORDS}


def _is_passable(item: str) -> bool:
    """물리적으로 들고 이동 가능한 아이템인지 판단."""
    return not bool(_kw(item) & NON_PASSABLE_KW)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: OBSERVATION & OFFER GENERATION
# ══════════════════════════════════════════════════════════════════════════════

_OBS_DRAFT_EXAMPLE = """
EXAMPLE — kitchen agent, task "prepare for sick person staying home":
<JSON>
{
  "offer": {
    "room_type": "kitchen",
    "observation": "Kitchen with fruits on island, bread basket, coffee maker, kettle.",
    "obs_scope": "island, counter, sink, stove, fruits, bread basket, coffee maker, kettle, mugs",
    "can_do": [
      "slice fruit from island and place on plate",
      "arrange bread from basket onto plate",
      "pour hot water from kettle into mug",
      "place mug and plate on serving tray"
    ],
    "cannot_do": [
      {"action": "adjust bedroom lighting", "reason": "NO_OBJECT"}
    ],
    "conf": {
      "slice fruit from island and place on plate": 0.9,
      "arrange bread from basket onto plate": 0.85,
      "pour hot water from kettle into mug": 0.9,
      "place mug and plate on serving tray": 0.9
    },
    "can_provide": ["light meal tray with fruit, bread, and warm drink"],
    "need_from_other": []
  },
  "draft_plan": {
    "plan_steps": [
      {"step_id":1,"time_min":1,"action":"slice fruit from island and place on plate",
       "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,"uncertainty":0.1,"notes":""},
      {"step_id":2,"time_min":2,"action":"arrange bread from basket onto tray",
       "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,"uncertainty":0.1,"notes":""},
      {"step_id":3,"time_min":3,"action":"pour hot water from kettle into mug and place on tray",
       "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,"uncertainty":0.1,"notes":""},
      {"step_id":4,"time_min":4,"action":"carry meal tray to kitchen doorway for agent_B pickup",
       "preconditions":["tray ready"],"depends_on":[1,2,3],
       "handoff_type":"PASS","target_agent":"agent_B","uncertainty":0.15,"notes":"meal tray at doorway"}
    ]
  }
}
</JSON>

EXAMPLE — bedroom agent, same task:
<JSON>
{
  "offer": {
    "room_type": "bedroom",
    "observation": "Bedroom with bed, pillow, blanket, lamp, dresser, alarm clock.",
    "obs_scope": "bed, pillow, blanket, lamp, dresser, alarm clock, window, curtain",
    "can_do": [
      "fluff pillow and arrange on bed",
      "fold blanket for easy access",
      "dim lamp for restful lighting",
      "clear dresser surface for tray placement"
    ],
    "cannot_do": [
      {"action": "prepare food or drinks", "reason": "NO_OBJECT"}
    ],
    "conf": {
      "fluff pillow and arrange on bed": 0.95,
      "fold blanket for easy access": 0.95,
      "dim lamp for restful lighting": 0.9,
      "clear dresser surface for tray placement": 0.9
    },
    "can_provide": [],
    "need_from_other": ["light meal tray with food and warm drink from kitchen"]
  },
  "draft_plan": {
    "plan_steps": [
      {"step_id":101,"time_min":1,"action":"fluff pillow and arrange on bed",
       "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,"uncertainty":0.1,"notes":""},
      {"step_id":102,"time_min":2,"action":"clear dresser surface for tray placement",
       "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,"uncertainty":0.1,"notes":""},
      {"step_id":103,"time_min":3,"action":"dim lamp for restful lighting",
       "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,"uncertainty":0.1,"notes":""},
      {"step_id":104,"time_min":5,"action":"receive meal tray from kitchen and place on dresser",
       "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,"uncertainty":0.15,"notes":"wait for kitchen agent"}
    ]
  }
}
</JSON>
""".strip()


def _build_obs_draft_prompt(task: str, step_offset: int = 0) -> str:
    """OBSERVATION 단계: offer + draft plan을 한 번에 생성."""
    offset_note = f"Use step_id starting from {step_offset+1}." if step_offset else "Use step_id starting from 1."
    return f"""You are a home agent. Look at the room image carefully.

Task: "{task}"

{_OBS_DRAFT_EXAMPLE}

{_P2_HANDOFF_RULES}

STRICT VISIBILITY RULE:
- List ONLY objects you can literally see in the image in obs_scope.
- Every can_do action must use a specific visible object from obs_scope.
- Use all available visible objects actively — do not ignore useful items.
- Do NOT imagine or infer objects not visible.

Generate BOTH your offer AND your draft plan in one response.

OFFER RULES:
1. obs_scope: every object you can actually see (comma-separated).
2. can_do: up to {MAX_CAN_DO} actions — use visible objects actively for the task.
3. can_provide: ONE item the other agent genuinely needs from you ([] if none).
4. need_from_other: ONE item you need delivered ([] if independent).

DRAFT PLAN RULES:
1. 2–5 steps using ONLY objects in obs_scope.
2. Every step must directly serve the global task goal. No filler steps.
3. If can_provide is NOT empty → add prep steps + PASS step (mandatory).
4. If need_from_other is NOT empty → add receive step (mandatory).
5. {offset_note}

Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "offer": {{
    "room_type": "kitchen or bedroom or living room",
    "observation": "one sentence describing the room",
    "obs_scope": "comma-separated visible objects",
    "can_do": ["verb + visible object"],
    "cannot_do": [{{"action": "...", "reason": "NO_OBJECT"}}],
    "conf": {{"action": 0.9}},
    "can_provide": [],
    "need_from_other": []
  }},
  "draft_plan": {{
    "plan_steps": [
      {{"step_id":1,"time_min":1,"action":"verb + visible object",
        "preconditions":[],"depends_on":[],"handoff_type":null,
        "target_agent":null,"uncertainty":0.1,"notes":""}}
    ]
  }}
}}
</JSON>"""


def _parse_offer(raw: str, agent_id: str) -> Offer:
    data = extract_json(raw)
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        data = {}

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
    seen: Set[str] = set()
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

    raw_provides = [str(x).strip() for x in data.get("can_provide", []) if str(x).strip()]
    can_provide  = [p for p in raw_provides if _is_passable(p)]
    filtered     = [p for p in raw_provides if not _is_passable(p)]
    if filtered:
        print(f"  [OFFER] non-passable items filtered from can_provide: {filtered}")

    return Offer(
        agent_id        = agent_id,
        room_type       = str(data.get("room_type", "")).strip(),
        observation     = str(data.get("observation", "")).strip(),
        obs_scope       = obs_scope,
        can_do          = can_do,
        cannot_do       = cannot_do,
        conf            = _match_conf(conf_raw, can_do),
        can_provide     = can_provide,
        need_from_other = [str(x).strip() for x in data.get("need_from_other", [])
                           if str(x).strip()],
        uncertain_count = uncertain_count,
    )


def _parse_obs_draft(raw: str, agent_id: str, step_offset: int = 0) -> tuple:
    """통합 obs+draft JSON 파싱 → (Offer, LocalPlan) 반환."""
    data = extract_json(raw)
    if not isinstance(data, dict):
        data = {}

    offer_data = data.get("offer", {})
    draft_data = data.get("draft_plan", {})

    # offer 파싱
    offer_raw = json.dumps(offer_data) if offer_data else "{}"
    offer = _parse_offer(offer_raw, agent_id)

    # draft plan 파싱
    if draft_data:
        plan_raw = json.dumps(draft_data)
    else:
        plan_raw = "{}"
    plan = _parse_local_plan(plan_raw, [], offer, step_offset=step_offset)
    return offer, plan


def observe_and_draft(
    img_a: str, img_b: str, task: str,
    verbose: str = "full",
) -> tuple:
    """OBSERVATION: offer + draft plan을 한 번에 생성 (VLM × 2)."""
    _banner("OBSERVATION — offer + draft plan")
    prompt_a = _build_obs_draft_prompt(task, step_offset=0)
    prompt_b = _build_obs_draft_prompt(task, step_offset=AGENT_B_STEP_OFFSET)

    results  = _run_parallel([(img_a, prompt_a, False), (img_b, prompt_b, False)])
    raw_a, _ = results[0]
    raw_b, _ = results[1]

    if verbose == "full":
        _log("A RAW", raw_a)
        _log("B RAW", raw_b)

    offer_a, draft_a = _parse_obs_draft(raw_a, "agent_A", step_offset=0)
    offer_b, draft_b = _parse_obs_draft(raw_b, "agent_B", step_offset=AGENT_B_STEP_OFFSET)

    if verbose in ("full", "summary"):
        _log("OFFER A", jdump(offer_to_dict(offer_a)))
        _log("OFFER B", jdump(offer_to_dict(offer_b)))

    print(f"  A: room={offer_a.room_type} | can_do={len(offer_a.can_do)} "
          f"| provide={len(offer_a.can_provide)} | draft={len(draft_a.steps)} steps")
    print(f"  B: room={offer_b.room_type} | can_do={len(offer_b.can_do)} "
          f"| provide={len(offer_b.can_provide)} | draft={len(draft_b.steps)} steps")
    return offer_a, offer_b, draft_a, draft_b





# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: LOCAL PLANNING
# ══════════════════════════════════════════════════════════════════════════════

_P2_EXAMPLE = """
EXAMPLE — kitchen agent, task "prepare for sick person":
Context: can_provide=["light meal tray"], other needs=["light meal tray from kitchen"]
<JSON>
{"plan_steps": [
  {"step_id":1,"time_min":1,"action":"slice apple and arrange on plate",
   "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,"uncertainty":0.1,"notes":""},
  {"step_id":2,"time_min":2,"action":"arrange bread from basket onto tray",
   "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,"uncertainty":0.1,"notes":""},
  {"step_id":3,"time_min":3,"action":"pour hot water into mug and place on tray",
   "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,"uncertainty":0.1,"notes":""},
  {"step_id":4,"time_min":4,"action":"carry meal tray to kitchen doorway for agent_B pickup",
   "preconditions":["tray ready"],"depends_on":[1,2,3],
   "handoff_type":"PASS","target_agent":"agent_B","uncertainty":0.15,"notes":"meal tray at doorway"}
]}
</JSON>

EXAMPLE — bedroom agent, same task:
Context: can_provide=[], kitchen agent will PASS meal tray to you
<JSON>
{"plan_steps": [
  {"step_id":101,"time_min":1,"action":"fluff pillow and arrange on bed",
   "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,"uncertainty":0.1,"notes":""},
  {"step_id":102,"time_min":2,"action":"clear dresser surface for tray placement",
   "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,"uncertainty":0.1,"notes":""},
  {"step_id":103,"time_min":3,"action":"dim lamp for restful lighting",
   "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,"uncertainty":0.1,"notes":""},
  {"step_id":104,"time_min":5,"action":"receive meal tray from kitchen and place on dresser",
   "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,"uncertainty":0.15,"notes":"wait for kitchen PASS"}
]}
</JSON>
""".strip()

_P2_HANDOFF_RULES = """
HANDOFF RULES:

PASS — physical delivery to room boundary:
  USE WHEN: you physically carry an item to the doorway for the other agent.
  ACTION must start with: "carry" or "bring"
  CORRECT: {"action":"carry snack tray to doorway","handoff_type":"PASS",
             "target_agent":"agent_B","depends_on":[1,2]}
  WRONG: PASS on preparation steps (place, arrange, set up, organize)
  WRONG: PASS on non-physical items (sink, counter, status, confirmation)
  MAXIMUM: 1–2 PASS steps total. Only for items in your can_provide list.

INFORM — status notification (no physical movement):
  USE WHEN: you want to notify the other agent of completion.
  CORRECT: {"action":"notify agent_B: snacks are ready at doorway",
             "handoff_type":"INFORM","target_agent":"agent_B"}

KEY: "carry X to doorway" → PASS | "notify agent_B" → INFORM | all others → null
""".strip()


def _build_phase2_prompt(my: Offer, other: Offer, task: str, use_offer: bool, hq_context: str = "") -> str:
    if use_offer:
        passable = [p for p in my.can_provide if _is_passable(p)]
        ctx = f"""YOUR OFFER:
- room: {my.room_type} ({my.agent_id})
- can_provide (items to PASS): {json.dumps(passable, ensure_ascii=False)}
- need_from_other: {json.dumps(my.need_from_other, ensure_ascii=False)}

OTHER AGENT ({other.room_type}, {other.agent_id}):
- can_provide: {json.dumps(other.can_provide, ensure_ascii=False)}
- need_from_other: {json.dumps(other.need_from_other, ensure_ascii=False)}"""
    else:
        ctx = f"YOUR ROOM: {my.room_type}\nOTHER ROOM: {other.room_type}"

    hq_block = f"\nHUMAN CLARIFICATION (use this to improve your plan):\n{hq_context}\n" if hq_context else ""

    return f"""You are the {my.room_type} agent ({my.agent_id}).
Global task: "{task}"{hq_block}

{ctx}

{_P2_EXAMPLE}

{_P2_HANDOFF_RULES}

STRICT VISIBILITY RULE:
- Every action must use ONLY objects visible in YOUR room image.
- Do NOT invent objects. Do NOT reference items from the other room.
- If you cannot physically see it, do not include it.

Think before writing:
1. FINAL GOAL: What END STATE does this task want?
2. MY ROLE: What is MY specific contribution from {my.room_type}?
3. GIVE: If can_provide is NOT empty → you MUST add prep steps + PASS step.
4. RECEIVE: If need_from_other is NOT empty → you MUST add a receive step.

PLANNING RULES:
1. Steps ONLY in {my.room_type}, using ONLY objects from obs_scope.
2. 2–5 steps. Every step must directly serve the global task goal.
   NO filler steps (do not add steps unrelated to the task).
3. PASS — if can_provide is NOT empty (MANDATORY):
   - 1–2 prep steps using visible objects
   - ONE PASS: "carry [item] to {my.room_type} doorway for [agent] pickup"
   - PASS MUST have depends_on=[prep step ids]
4. RECEIVE — if need_from_other is NOT empty (MANDATORY):
   - "receive [item] from other room and place at [specific location in your room]"
   - depends_on=[] (system links automatically)
5. Return ONLY valid JSON inside <JSON> tags. No other text.

<JSON>
{{
  "plan_steps": [
    {{"step_id":1,"time_min":1,"action":"verb + visible object",
      "preconditions":[],"depends_on":[],"handoff_type":null,
      "target_agent":null,"uncertainty":0.1,"notes":""}}
  ]
}}
</JSON>"""


def _parse_local_plan(
    raw: str, log_probs: List[float], my: Offer, step_offset: int = 0,
) -> LocalPlan:
    data = extract_json(raw)
    if isinstance(data, list):
        # LLM이 plan_steps 배열을 바로 반환한 경우
        data = {"plan_steps": data}
    if not isinstance(data, dict):
        data = {}
    raw_steps = data.get("plan_steps", [])
    if not isinstance(raw_steps, list):
        raw_steps = []

    token_unc = compute_token_uncertainty(log_probs)
    steps:    List[PlanStep] = []
    hq_list:  List[HQEntry]  = []
    seen_ids: Set[int]        = set()
    seen_act: Set[frozenset]  = set()

    for i, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip()
        if not action:
            continue
        # action 이름에 포함된 PASS/NOTIFY 텍스트 제거
        action = re.sub(r"\s*\[[→→]\s*PASS to \w+\]", "", action, flags=re.IGNORECASE).strip()
        action = re.sub(r"\s*\[[→→]\s*NOTIFY \w+\]", "", action, flags=re.IGNORECASE).strip()
        action = re.sub(r"\s*\(PASS\)", "", action, flags=re.IGNORECASE).strip()

        akey = frozenset(_kw(action))
        if akey and akey in seen_act:
            continue
        seen_act.add(akey)

        raw_sid = safe_int(item.get("step_id", i), i)
        raw_time = safe_int(item.get("time_min", 0), 0)
        if raw_time > 25 and raw_time == raw_sid:
            raw_time = 0

        sid = raw_sid + step_offset
        while sid in seen_ids:
            sid += 1
        seen_ids.add(sid)

        json_unc    = clamp01(item.get("uncertainty", 0.2))
        action_conf = max(
            (v for k, v in my.conf.items() if _fuzzy_match_soft(action, k)),
            default=0.7,
        )
        step_unc = clamp01(json_unc * 0.5 + token_unc * 0.2 + (1 - action_conf) * 0.3)

        raw_deps  = _norm_depends(item.get("depends_on"))
        deps      = [d + step_offset for d in raw_deps]
        handoff   = _norm_handoff(item.get("handoff_type")) if item.get("handoff_type") else None
        target    = _norm_agent(item.get("target_agent"))

        # carry/bring 동사인데 INFORM이면 PASS로 교정
        first_word = action.lower().split()[0] if action.strip() else ""
        if handoff == "INFORM" and first_word in {"carry", "bring", "deliver", "transport"}:
            handoff = "PASS"

        step = PlanStep(
            step_id       = sid,
            time_min      = max(0, min(30, raw_time)),
            room          = my.room_type,
            agent_id      = my.agent_id,
            action        = action,
            preconditions = [str(x).strip() for x in item.get("preconditions", [])
                             if str(x).strip()],
            depends_on    = deps,
            handoff_type  = handoff,
            target_agent  = target,
            uncertainty   = step_unc,
            notes         = str(item.get("notes", "")).strip(),
        )
        steps.append(step)

        if step_unc >= UNCERTAINTY_THRESH:
            hq_list.append(HQEntry(sid, f"Is '{action}' feasible?", step_unc))

    steps.sort(key=lambda s: (s.time_min, s.step_id))
    steps = _normalize_pass(steps)

    handoffs = [
        Handoff(s.step_id, s.action, s.handoff_type, s.target_agent,
                s.notes if s.handoff_type == "INFORM" else "",
                my.agent_id)
        for s in steps if s.handoff_type
    ]

    all_unc = [s.uncertainty for s in steps] if steps else [token_unc]
    return LocalPlan(my.agent_id, steps, compute_plan_uncertainty(all_unc), hq_list, handoffs)


def _normalize_pass(steps: List[PlanStep]) -> List[PlanStep]:
    """비정상 PASS 제거."""
    my_ids = {s.step_id for s in steps}
    seen_pass: List[PlanStep] = []

    # carry/bring 동사 아닌데 PASS면 제거
    _CARRY = {"carry", "bring", "deliver", "transport", "move", "transfer"}
    # 배치/수신 동사인데 PASS면 제거
    _RECV  = {"place", "set", "organize", "receive", "pick", "get", "put", "sort"}

    for s in steps:
        if s.handoff_type != "PASS":
            continue

        first = s.action.lower().split()[0] if s.action.strip() else ""

        if not s.target_agent or s.target_agent not in VALID_AGENTS:
            print(f"  [NORM] step{s.step_id} PASS removed: no valid target_agent")
            s.handoff_type = None; s.target_agent = None; continue

        if first in _RECV:
            print(f"  [NORM] step{s.step_id} PASS removed: receiver verb '{first}'")
            s.handoff_type = None; s.target_agent = None; continue

        if not s.depends_on:
            # deps 없으면 같은 agent의 이전 prep step 자동 연결
            prep = [p for p in steps
                    if p.step_id < s.step_id
                    and p.agent_id == s.agent_id
                    and p.handoff_type is None]
            if prep:
                s.depends_on = [p.step_id for p in prep]
                print(f"  [NORM] step{s.step_id} PASS auto-linked deps={s.depends_on}")
            else:
                print(f"  [NORM] step{s.step_id} PASS removed: no depends_on")
                s.handoff_type = None; s.target_agent = None; continue

        if not [d for d in s.depends_on if d in my_ids]:
            print(f"  [NORM] step{s.step_id} PASS removed: deps not in own plan")
            s.handoff_type = None; s.target_agent = None; continue

        # cross-agent deps 제거
        s.depends_on = [d for d in s.depends_on if d in my_ids]

        # 중복 PASS 제거
        if any(_fuzzy_match(s.action, prev.action, min_overlap=3) for prev in seen_pass):
            print(f"  [NORM] step{s.step_id} PASS removed: duplicate")
            s.handoff_type = None; s.target_agent = None; continue

        seen_pass.append(s)

    return steps


def _ensure_pass(
    plan_a: LocalPlan, plan_b: LocalPlan,
    offer_a: Offer, offer_b: Offer,
) -> Tuple[LocalPlan, LocalPlan]:
    """
    offer 매칭 기반으로 PASS가 누락됐으면 삽입하고
    receiver 플랜의 관련 스텝에 receive step을 추가한다.
    """
    _FOOD_KW = {
        "snack", "food", "drink", "tray", "bowl", "cup", "plate",
        "fruit", "bread", "water", "bottle", "popcorn", "soda", "nut",
        "juice", "meal", "cookie", "candy",
    }

    def _inject(
        sender: LocalPlan, receiver: LocalPlan,
        s_offer: Offer, r_offer: Offer,
        sid: str, rid: str,
    ) -> Tuple[LocalPlan, LocalPlan]:
        # 이미 유효한 PASS가 있으면 스킵
        existing = [s for s in sender.steps
                    if s.handoff_type == "PASS" and s.target_agent == rid]
        if existing:
            # 기존 PASS에 receiver step 연결만 확인
            for pass_step in existing:
                _link_receiver(pass_step, receiver, s_offer, r_offer, rid)
            return sender, receiver

        # passable item 찾기
        passable = [p for p in s_offer.can_provide if _is_passable(p)]
        if not passable:
            return sender, receiver

        provide = passable[0]
        pkw     = _kw(provide)

        # sender 플랜에서 prep step 찾기
        prep = [s for s in sender.steps if pkw & _kw(s.action) and not s.handoff_type]
        if not prep:
            prep = [s for s in sender.steps if not s.handoff_type]
        if not prep:
            return sender, receiver

        last_prep = max(prep, key=lambda s: s.time_min)

        # PASS step 생성
        all_ids = {s.step_id for s in sender.steps} | {s.step_id for s in receiver.steps}
        new_sid = max(all_ids, default=0) + 1
        while new_sid in all_ids:
            new_sid += 1

        pass_time = min(30, last_prep.time_min + 5)
        pass_step = PlanStep(
            step_id      = new_sid,
            time_min     = pass_time,
            room         = s_offer.room_type,
            agent_id     = sid,
            action       = f"carry {provide} to {s_offer.room_type} doorway for {rid} pickup",
            preconditions= [f"step {last_prep.step_id} completed"],
            depends_on   = [last_prep.step_id],
            handoff_type = "PASS",
            target_agent = rid,
            uncertainty  = 0.15,
            notes        = f"{provide} ready at doorway",
        )
        sender.steps.append(pass_step)
        sender.steps.sort(key=lambda s: (s.time_min, s.step_id))
        sender.handoffs.append(
            Handoff(new_sid, pass_step.action, "PASS", rid, "", sid)
        )
        print(f"  [ENSURE] {sid}: PASS step{new_sid} injected "
              f"(T={pass_time}m) '{provide}' → {rid}")

        _link_receiver(pass_step, receiver, s_offer, r_offer, rid)
        return sender, receiver

    def _link_receiver(
        pass_step: PlanStep,
        receiver: LocalPlan,
        s_offer: Offer, r_offer: Offer,
        rid: str,
    ) -> None:
        """PASS step에 대응하는 receiver step 찾아서 depends_on 연결."""
        pkw = _kw(pass_step.notes or pass_step.action)
        _FOOD_KW_LOCAL = {
            "snack","food","drink","tray","bowl","cup","plate",
            "fruit","bread","water","bottle","popcorn","soda",
            "nut","juice","meal","cookie",
        }
        _PLACE = {"place","put","lay","bring","serve","deliver","receive","arrange"}

        _RECV_VERBS = {"receive","collect","pick","get","take","accept","grab"}

        # 1단계: receive 동사가 있는 스텝만 우선
        targets = [s for s in receiver.steps
                   if not s.handoff_type
                   and set(re.findall(r"\w+", s.action.lower())) & _RECV_VERBS]

        # 2단계: keyword 직접 겹침 (receive 동사 없어도)
        if not targets:
            targets = [s for s in receiver.steps
                       if not s.handoff_type and pkw & _kw(s.action)]

        # 3단계: food item + 배치 동사 (receive 없는 경우)
        if not targets and pkw & _FOOD_KW_LOCAL:
            targets = [s for s in receiver.steps
                       if not s.handoff_type
                       and set(re.findall(r"\w+", s.action.lower())) & _PLACE
                       and _kw(s.action) & _FOOD_KW_LOCAL]

        # 4단계: need_from_other fuzzy match
        if not targets:
            targets = [s for s in receiver.steps
                       if not s.handoff_type
                       and any(_fuzzy_match_soft(s.action, n)
                               for n in r_offer.need_from_other)
                       and set(re.findall(r"\w+", s.action.lower())) & _PLACE]

        # receiver step이 없으면 자동 추가
        if not targets:
            _add_receive_step(pass_step, receiver, s_offer, rid)
            return

        coord_time = pass_step.time_min
        for rs in targets:
            if pass_step.step_id not in rs.depends_on:
                rs.depends_on = sorted(set(rs.depends_on + [pass_step.step_id]))
            if rs.time_min <= coord_time:
                rs.time_min = coord_time + 1
            print(f"  [ENSURE] {rid} step{rs.step_id} "
                  f"'{rs.action[:40]}' ← PASS step{pass_step.step_id} "
                  f"(T={rs.time_min}m)")

    def _add_receive_step(
        pass_step: PlanStep,
        receiver: LocalPlan,
        s_offer: Offer,
        rid: str,
    ) -> None:
        """receiver 플랜에 receive step 자동 추가."""
        all_ids = {s.step_id for s in receiver.steps}
        new_sid = max(all_ids, default=0) + 1
        while new_sid in all_ids:
            new_sid += 1

        # 아이템 이름 추출
        m = re.search(r"carry (.+?) (?:to|for)", pass_step.action, re.IGNORECASE)
        item = m.group(1).strip() if m else "item"
        if len(item) > 35:
            item = item[:35].rsplit(" ", 1)[0]

        recv_step = PlanStep(
            step_id      = new_sid,
            time_min     = min(30, pass_step.time_min + 1),
            room         = r_offer.room_type,
            agent_id     = rid,
            action       = f"receive {item} from {s_offer.room_type} and bring into room",
            preconditions= [f"step {pass_step.step_id} completed"],
            depends_on   = [pass_step.step_id],
            handoff_type = None,
            target_agent = None,
            uncertainty  = 0.15,
            notes        = "auto-added receive step",
        )
        receiver.steps.append(recv_step)
        receiver.steps.sort(key=lambda s: (s.time_min, s.step_id))
        print(f"  [ENSURE] {rid}: receive step{new_sid} auto-added "
              f"(T={recv_step.time_min}m)")

    # A→B
    plan_a, plan_b = _inject(plan_a, plan_b, offer_a, offer_b, "agent_A", "agent_B")
    # B→A: A의 need_from_other가 물리적 아이템인 경우만
    _INFO_KW = {"confirmation","confirm","clear","ready","status",
                "notify","check","verified","done","complete","that"}
    a_needs_physical = any(
        not (_kw(n) & _INFO_KW)
        for n in offer_a.need_from_other
    )
    if a_needs_physical:
        plan_b, plan_a = _inject(plan_b, plan_a, offer_b, offer_a, "agent_B", "agent_A")
    else:
        print(f"  [ENSURE] B→A skipped: A only needs confirmation-type info")
    return plan_a, plan_b


def _build_phase2b_prompt(my: Offer, other: Offer, draft: LocalPlan, task: str, hq_context: str = "") -> str:
    """Phase 2b: 상대방 draft를 보고 최종 plan 생성."""
    other_summary = []
    for i, s in enumerate(draft.steps, 1):
        line = f"  Step {i}: {s.action}"
        if s.handoff_type == "PASS":
            line += f"  [→ PASS to {s.target_agent}]"
        elif s.handoff_type == "INFORM":
            line += f"  [→ NOTIFY {s.target_agent}]"
        other_summary.append(line)

    passable = [p for p in my.can_provide if _is_passable(p)]

    hq_block2 = f"\nHUMAN CLARIFICATION (incorporate into your plan):\n{hq_context}\n" if hq_context else ""

    return f"""You are {my.agent_id} ({my.room_type}).
Global task: "{task}"{hq_block2}

YOUR can_provide: {json.dumps(passable, ensure_ascii=False)}
YOUR need_from_other: {json.dumps(my.need_from_other, ensure_ascii=False)}

{other.agent_id} ({other.room_type}) DRAFT PLAN:
{chr(10).join(other_summary) if other_summary else "  (no steps yet)"}

STEP 1 — Understand YOUR role in this task:
  Global task: "{task}"
  Your room: {my.room_type}
  Ask yourself: "What does THIS ROOM need to do to achieve the task goal?"
  → List 2–3 actions your room must perform (independent of the other agent).

STEP 2 — Handle incoming PASS:
  Does {other.agent_id}'s draft have "[→ PASS to {my.agent_id}]"?
  If YES → add a RECEIVE step AFTER your room-preparation steps:
    action: "receive [specific item] from {other.room_type} and place at [location]"
    depends_on: [] (system links automatically)

STEP 3 — Handle outgoing PASS:
  Is YOUR can_provide non-empty?
  If YES → add prep steps + PASS step with depends_on=[prep step ids]

STEP 4 — Remove duplicates:
  Skip any action already in {other.agent_id}'s draft.

STRICT VISIBILITY RULE:
- Use ONLY objects you can literally see in your room image.
- Every action must match the task goal for YOUR room.

RULES:
1. Steps ONLY in {my.room_type}, 3–5 steps total.
2. ROOM PREPARATION steps come FIRST (what your room must do for the task).
3. RECEIVE step comes AFTER room preparation (if incoming PASS exists).
4. PASS step is MANDATORY if can_provide is not empty.
5. Return ONLY valid JSON inside <JSON> tags. No other text.

WRONG — bedroom only has receive step:
  Step 1: receive meal tray  ← too few, room not prepared

CORRECT — bedroom prepares AND receives:
  Step 1: fluff pillows and arrange bed for comfort
  Step 2: dim lamp for restful lighting
  Step 3: clear dresser surface for tray placement
  Step 4: receive meal tray from kitchen and place on dresser  ← RECEIVE last

<JSON>
{{"plan_steps": [
  {{"step_id":1,"time_min":1,"action":"verb + visible object",
    "preconditions":[],"depends_on":[],"handoff_type":null,
    "target_agent":null,"uncertainty":0.1,"notes":""}}
]}}
</JSON>"""


# phase2_local_plan kept as alias for backward compat
def coordinate(
    offer_a: Offer, offer_b: Offer,
    draft_a: LocalPlan, draft_b: LocalPlan,
    img_a: str, img_b: str, task: str,
    use_offer: bool = True,
    verbose: str = "full",
) -> Tuple[LocalPlan, LocalPlan]:
    """COORDINATION: 상대방 draft를 보고 final plan 생성 (VLM × 2)."""

    _banner("COORDINATION — mutual awareness final plan")
    pa2 = _build_phase2b_prompt(offer_a, offer_b, draft_b, task)
    pb2 = _build_phase2b_prompt(offer_b, offer_a, draft_a, task)
    res_b = _run_parallel([(img_a, pa2, True), (img_b, pb2, True)])
    raw_a2, lp_a2 = res_b[0]
    raw_b2, lp_b2 = res_b[1]
    if verbose == "full":
        _log("A FINAL RAW", raw_a2); _log("B FINAL RAW", raw_b2)
    plan_a = _parse_local_plan(raw_a2, lp_a2, offer_a, step_offset=0)
    plan_b = _parse_local_plan(raw_b2, lp_b2, offer_b, step_offset=AGENT_B_STEP_OFFSET)

    # HANDOFF SYNC: rule-based PASS 보완
    if use_offer:
        _banner("HANDOFF SYNC — rule-based coordination")
        plan_a, plan_b = _ensure_pass(plan_a, plan_b, offer_a, offer_b)

    if verbose in ("full", "summary"):
        _log("PLAN A (final)", jdump(local_plan_to_dict(plan_a)))
        _log("PLAN B (final)", jdump(local_plan_to_dict(plan_b)))

    pass_a = sum(1 for s in plan_a.steps if s.handoff_type == "PASS")
    pass_b = sum(1 for s in plan_b.steps if s.handoff_type == "PASS")
    print(f"\n  A: steps={len(plan_a.steps)} U={plan_a.U_plan:.3f} PASS={pass_a}")
    print(f"  B: steps={len(plan_b.steps)} U={plan_b.U_plan:.3f} PASS={pass_b}")
    for h in plan_a.handoffs + plan_b.handoffs:
        print(f"  [{h.agent_id}->{h.handoff_type}] step{h.step_id} → {h.target_agent} | {h.action[:55]}")
    return plan_a, plan_b


def phase2_local_plan(
    offer_a: Offer, offer_b: Offer,
    img_a: str, img_b: str, task: str,
    use_offer: bool = True,
    verbose: str = "full",
    draft_a: LocalPlan = None,
    draft_b: LocalPlan = None,
) -> Tuple[LocalPlan, LocalPlan]:
    """Backward compat wrapper."""
    if draft_a is None or draft_b is None:
        # draft 없으면 빈 플랜 생성 (호환성)
        from p2p_models import LocalPlan as LP
        draft_a = draft_a or LP("agent_A", [], 0.2, [], [])
        draft_b = draft_b or LP("agent_B", [], 0.2, [], [])
    return coordinate(offer_a, offer_b, draft_a, draft_b, img_a, img_b, task, use_offer, verbose)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: CONFLICT DETECTION
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
    slots: Dict[int, List[PlanStep]] = {}
    for s in steps_a + steps_b:
        slots.setdefault(s.time_min, []).append(s)
    for t, slot in slots.items():
        for i in range(len(slot)):
            for j in range(i + 1, len(slot)):
                si, sj = slot[i], slot[j]
                if si.agent_id == sj.agent_id or si.room != sj.room:
                    continue
                overlap = _kw(si.action) & _kw(sj.action)
                if overlap:
                    conflicts.append(ConflictEntry(
                        conflict_type = ConflictType.TEMPORAL,
                        step_ids      = [si.step_id, sj.step_id],
                        agent_ids     = [si.agent_id, sj.agent_id],
                        description   = (
                            f"T={t}m: step{si.step_id} & step{sj.step_id} "
                            f"share resource {overlap} in same room"
                        ),
                        fix_hint = f"Shift one step away from T={t}m.",
                    ))

    # ── 2. DEPENDENCY: PASS sender가 있는데 receiver deps 없음 ────────────────
    all_ids_b = {s.step_id for s in steps_b}
    all_ids_a = {s.step_id for s in steps_a}

    pass_steps_a = [s for s in steps_a if s.handoff_type == "PASS"]
    pass_steps_b = [s for s in steps_b if s.handoff_type == "PASS"]

    recv_deps_b = {dep for s in steps_b for dep in s.depends_on if dep in all_ids_a}
    recv_deps_a = {dep for s in steps_a for dep in s.depends_on if dep in all_ids_b}

    for ps in pass_steps_a:
        if ps.step_id not in recv_deps_b:
            # B에 receive step이 없음
            related_b = [s for s in steps_b
                         if _kw(ps.action) & _kw(s.action)
                         or _kw(ps.notes) & _kw(s.action)]
            if not related_b:
                conflicts.append(ConflictEntry(
                    conflict_type = ConflictType.DEPENDENCY,
                    step_ids      = [ps.step_id],
                    agent_ids     = ["agent_A", "agent_B"],
                    description   = (
                        f"A step{ps.step_id} is a PASS to agent_B "
                        f"but agent_B has no step depending on it."
                    ),
                    fix_hint = (
                        f"Add a receive/use step to agent_B's plan "
                        f"with depends_on=[{ps.step_id}]."
                    ),
                ))

    for ps in pass_steps_b:
        if ps.step_id not in recv_deps_a:
            related_a = [s for s in steps_a
                         if _kw(ps.action) & _kw(s.action)
                         or _kw(ps.notes) & _kw(s.action)]
            if not related_a:
                conflicts.append(ConflictEntry(
                    conflict_type = ConflictType.DEPENDENCY,
                    step_ids      = [ps.step_id],
                    agent_ids     = ["agent_A", "agent_B"],
                    description   = (
                        f"B step{ps.step_id} is a PASS to agent_A "
                        f"but agent_A has no step depending on it."
                    ),
                    fix_hint = (
                        f"Add a receive/use step to agent_A's plan "
                        f"with depends_on=[{ps.step_id}]."
                    ),
                ))

    # ── 3. REDUNDANCY (inter) ─────────────────────────────────────────────────
    for sa in steps_a:
        for sb in steps_b:
            if sa.handoff_type == "PASS" or sb.handoff_type == "PASS":
                continue
            if _fuzzy_match(sa.action, sb.action, min_overlap=3):
                conflicts.append(ConflictEntry(
                    conflict_type = ConflictType.REDUNDANCY,
                    step_ids      = [sa.step_id, sb.step_id],
                    agent_ids     = ["agent_A", "agent_B"],
                    description   = (
                        f"Duplicate: A-step{sa.step_id} '{sa.action[:35]}' "
                        f"≈ B-step{sb.step_id} '{sb.action[:35]}'"
                    ),
                    fix_hint = "Delete one of the duplicate steps.",
                ))

    # ── 4. REDUNDANCY (intra) ─────────────────────────────────────────────────
    for agent_steps in [steps_a, steps_b]:
        for i in range(len(agent_steps)):
            for j in range(i + 1, len(agent_steps)):
                si, sj = agent_steps[i], agent_steps[j]
                one_pass = (si.handoff_type == "PASS") != (sj.handoff_type == "PASS")
                if one_pass:
                    continue
                if _fuzzy_match(si.action, sj.action, min_overlap=3):
                    conflicts.append(ConflictEntry(
                        conflict_type = ConflictType.REDUNDANCY,
                        step_ids      = [si.step_id, sj.step_id],
                        agent_ids     = [si.agent_id],
                        description   = (
                            f"Intra-agent duplicate ({si.agent_id}): "
                            f"step{si.step_id} ≈ step{sj.step_id}"
                        ),
                        fix_hint = f"Delete step{sj.step_id}.",
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
                        f"violates cannot_do"
                    ),
                    fix_hint = f"Delete or reassign step{step.step_id}.",
                ))

    # ── 6. OBSERVABILITY ──────────────────────────────────────────────────────
    for step, offer in all_steps:
        # PASS / receive / notify / inform / confirm 스텝은 면제
        if step.handoff_type in ("PASS", "INFORM"):
            continue
        act_lower = step.action.lower()
        if act_lower.startswith(("receive", "notify", "inform", "confirm",
                                  "auto-added", "carry")):
            continue
        scope_kw = set(re.findall(r"\w+", offer.obs_scope.lower()))
        can_kw: Set[str] = set()
        for cd in offer.can_do:
            can_kw |= _kw(cd)
        pool   = scope_kw | can_kw
        act_kw = _kw(step.action)
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
    _banner("CONFLICT CHECK")
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
        + (f"\n    → HINT: {c.fix_hint}" if c.fix_hint else "")
        for c in conflicts
    ) or "  (none)"

    prev_text = "\n".join(
        f"  step{p.step_id}[{p.agent_id}] .{p.field}='{p.new_value}' ({p.reason})"
        for p in prev_props
    ) or "  (none)"

    return f"""You are {my_agent} ({my_offer.room_type}). ROUND {round_num}/{MAX_NEGOTIATION_ROUNDS}.
Task: "{task}"

YOUR PLAN: {jdump(my_plan)}
{other_id}'s PLAN: {jdump(other_plan)}

CONFLICTS: {c_text}
LOCKED (do not modify): {sorted(locked) or '(none)'}
{other_id}'s previous proposals: {prev_text}

RULES:
1. Fix YOUR steps first. One proposal per conflict.
2. DEPENDENCY: add depends_on + shift time_min after the PASS step.
3. REDUNDANCY: delete one duplicate (field="delete").
4. TEMPORAL: shift time_min to avoid overlap.
5. OBSERVABILITY: delete or fix the step.
6. To ACCEPT other's proposal: reason="ACCEPT".
7. Allowed fields: "time_min" | "action" | "depends_on" | "delete"

<JSON>
{{"proposals":[
  {{"step_id":103,"agent_id":"{my_agent}","field":"depends_on",
    "new_value":"[5]","reason":"DEPENDENCY: must wait for PASS step 5"}}
]}}
</JSON>"""


def _parse_proposals(raw: str, my_agent: str) -> List[NegotiationProposal]:
    data   = extract_json(raw)
    # LLM이 {"proposals":[...]} 대신 [...] 를 바로 반환하는 경우 처리
    if isinstance(data, list):
        data = {"proposals": data}
    if not isinstance(data, dict):
        return []
    result = []
    for item in data.get("proposals", []):
        if not isinstance(item, dict):
            continue
        sid      = safe_int(item.get("step_id", -1), -1)
        agent_id = str(item.get("agent_id", my_agent)).strip()
        field    = str(item.get("field", "")).strip().lower()
        new_val  = str(item.get("new_value", "")).strip()
        reason   = str(item.get("reason", "")).strip()
        if sid < 0 or field not in VALID_PROPOSAL_FIELDS or not new_val:
            continue
        if agent_id not in VALID_AGENTS:
            agent_id = my_agent
        result.append(NegotiationProposal(sid, agent_id, field, new_val, reason))
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
        plan.pop(idx); return True
    if prop.field == "time_min":
        t = safe_int(prop.new_value, -1)
        if 0 <= t <= 30:
            plan[idx]["time_min"] = t; return True
    elif prop.field == "action":
        if prop.new_value:
            plan[idx]["action"] = prop.new_value; return True
    elif prop.field == "depends_on":
        try:
            val = prop.new_value.strip()
            deps = json.loads(val) if val.startswith("[") else [int(val)]
            if isinstance(deps, list):
                plan[idx]["depends_on"] = [int(d) for d in deps]
                return True
        except Exception:
            pass
    return False


def _lock_steps(
    props_a: List[NegotiationProposal],
    props_b: List[NegotiationProposal],
    conflict_sids: Set[int], existing: Set[int],
) -> Set[int]:
    acc_b = {p.step_id for p in props_b if p.reason.upper() == "ACCEPT"}
    acc_a = {p.step_id for p in props_a if p.reason.upper() == "ACCEPT"}
    prop_a = {p.step_id for p in props_a if p.reason.upper() != "ACCEPT"}
    prop_b = {p.step_id for p in props_b if p.reason.upper() != "ACCEPT"}
    agreed    = (prop_a & acc_b) | (prop_b & acc_a)
    # uncontested는 lock 안 함 — 미해결 conflict는 다음 라운드에서 처리
    return existing | agreed


def phase4_negotiation(
    plan_a: LocalPlan, plan_b: LocalPlan,
    offer_a: Offer, offer_b: Offer,
    conflicts: List[ConflictEntry],
    img_a: str, img_b: str, task: str,
    verbose: str = "full",
) -> Tuple[List[Dict], List[Dict], List[NegotiationRound]]:
    _banner("NEGOTIATION — P2P")

    if not conflicts:
        print("  No conflicts → skip.")
        return plan_steps_to_dicts(plan_a.steps), plan_steps_to_dicts(plan_b.steps), []

    conflict_sids: Set[int] = {sid for c in conflicts for sid in c.step_ids}
    print(f"  Conflict step IDs: {sorted(conflict_sids)}")

    cur_a  = plan_steps_to_dicts(plan_a.steps)
    cur_b  = plan_steps_to_dicts(plan_b.steps)
    locked: Set[int] = set()
    rounds: List[NegotiationRound] = []
    prev_a: List[NegotiationProposal] = []
    prev_b: List[NegotiationProposal] = []
    last_val: Dict[Tuple[int, str], str] = {}

    for rnd in range(1, MAX_NEGOTIATION_ROUNDS + 1):
        remaining = [c for c in conflicts
                     if not c.step_ids or not all(s in locked for s in c.step_ids)]
        if not remaining:
            print(f"\n  Round {rnd}: all resolved early.")
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
                print(f"  [A→{p.agent_id}] step{p.step_id}.{p.field}="
                      f"'{p.new_value[:35]}' ({p.reason[:40]})")
            for p in props_b:
                print(f"  [B→{p.agent_id}] step{p.step_id}.{p.field}="
                      f"'{p.new_value[:35]}' ({p.reason[:40]})")

        for prop in props_a + props_b:
            if _apply_proposal(cur_a, cur_b, prop, locked) and verbose in ("full", "summary"):
                print(f"  [APPLIED] step{prop.step_id}.{prop.field}")

        locked = _lock_steps(props_a, props_b, conflict_sids, locked)
        rounds.append(NegotiationRound(rnd, props_a, props_b, sorted(locked)))
        print(f"  → Locked: {sorted(locked)}")
        prev_a, prev_b = props_a, props_b

    print(f"\n  Total rounds: {len(rounds)}")
    return cur_a, cur_b, rounds


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: CONVERGENCE CHECK
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
    _banner("PLAN QUALITY CHECK")
    all_steps = steps_a + steps_b

    no_cycle = not _has_cycle(all_steps)

    # observability — 완화: 동사/일반어 제외, 핵심 명사만 체크
    _OBS_SKIP = {"the","a","an","and","or","to","for","of","in","on","at",
                 "place","set","put","move","carry","bring","arrange","adjust",
                 "prepare","organize","clear","clean","check","make","create",
                 "receive","notify","pick","get","take","use"}
    def _obs_pool(offer: Offer) -> Set[str]:
        pool = set(re.findall(r"\w+", offer.obs_scope.lower()))
        for cd in offer.can_do:
            pool |= _kw(cd)
        return pool - _OBS_SKIP

    pool_a = _obs_pool(offer_a)
    pool_b = _obs_pool(offer_b)
    obs_ok = True
    obs_violations = []
    for s in all_steps:
        if s.get("handoff_type") in ("PASS", "INFORM"):
            continue
        pool = pool_a if s.get("agent_id") == "agent_A" else pool_b
        kw   = _kw(s.get("action", "")) - _OBS_SKIP
        if len(kw) >= 2 and pool and not (kw & pool):
            obs_ok = False
            obs_violations.append(s["step_id"])

    # PASS-receive 매칭
    pass_ids_a = {s["step_id"] for s in steps_a if s.get("handoff_type") == "PASS"}
    pass_ids_b = {s["step_id"] for s in steps_b if s.get("handoff_type") == "PASS"}
    deps_in_b  = {d for s in steps_b for d in s.get("depends_on", []) if d in pass_ids_a}
    deps_in_a  = {d for s in steps_a for d in s.get("depends_on", []) if d in pass_ids_b}
    truly_unmatched = (pass_ids_a - deps_in_b) | (pass_ids_b - deps_in_a)
    no_missing = len(truly_unmatched) == 0

    unresolved = [c for c in conflicts
                  if c.conflict_type in (ConflictType.REDUNDANCY, ConflictType.CANNOT_DO)]
    converged  = no_cycle and obs_ok and no_missing

    print(f"  Dep cycle      : {'OK' if no_cycle else 'FAIL'}")
    print(f"  PASS matched   : {'OK' if no_missing else f'FAIL (unmatched={truly_unmatched})'}")
    print(f"  Observability  : {'OK' if obs_ok else f'WARN (step_ids={obs_violations})'}")

    return ConvergenceResult(
        converged            = converged,
        no_dep_cycle         = no_cycle,
        observability_ok     = obs_ok,
        no_missing_deps      = no_missing,
        unresolved_conflicts = unresolved,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6: DEFERRED HUMAN QUERY (VLM 기반)
# ══════════════════════════════════════════════════════════════════════════════

_HQ_TEMPLATES: Dict[str, str] = {
    "DEP_CYCLE":    "A dependency cycle was detected. Which step should be reordered?",
    "DEPENDENCY":   "A PASS step has no matching receive step. Should the receiving agent add one?",
    "REDUNDANCY":   "Two agents are doing the same thing. Which should handle it?",
    "CANNOT_DO":    "An agent planned something it cannot do. Remove or reassign?",
    "OBSERVABILITY":"A step references objects outside visible scope. Modify or remove?",
    "UNMATCHED":    "An agent needs something no one can provide. How to handle this?",
}


def _generate_hq_question(
    trigger_type: str, detail: str,
    offer_a: Offer, offer_b: Offer, img: str,
) -> str:
    template = _HQ_TEMPLATES.get(trigger_type, "How should the agents handle this?")

    # conflict 타입별 directive — step 번호 대신 액션 내용 설명
    if trigger_type == "OBSERVABILITY":
        directive = (
            f"An action references objects not visible in the room image: '{detail[:100]}'. "
            f"Should this action be DELETED, or MODIFIED to use only visible objects?"
        )
    elif trigger_type == "REDUNDANCY":
        directive = (
            f"Two agents planned the same action: '{detail[:120]}'. "
            f"Which agent should handle this, and which should remove their duplicate?"
        )
    elif trigger_type == "DEPENDENCY":
        directive = (
            f"A handoff was planned but the receiving agent has no step to collect it: '{detail[:120]}'. "
            f"Should the receiving agent add a pickup step?"
        )
    elif trigger_type == "UNMATCHED":
        directive = (
            f"An agent needs something neither agent can provide: '{detail[:120]}'. "
            f"Can this need be fulfilled, or should the plan proceed without it?"
        )
    else:
        directive = detail[:200]

    prompt = f"""Two home agents are planning together.
Agent A is in the {offer_a.room_type}. Agent B is in the {offer_b.room_type}.

Issue: {directive}

Write ONE clear YES/NO question for the human operator.
Describe what the action IS (not step numbers). One sentence only."""

    try:
        q, _ = run_vlm(img, prompt)
        q = q.strip().strip('"').strip("'")
        # 마크다운/불필요한 블록 제거
        import re as _re
        q = _re.sub(r'\*\*.*?\*\*:?\s*', '', q)       # **...**
        q = _re.sub(r'Action:.*', '', q, flags=_re.S)  # Action: 이후 전부
        q = _re.sub(r'Evaluate.*', '', q, flags=_re.S) # Evaluate 이후 전부
        q = q.strip()
        if 10 < len(q) < 400:
            return q
    except Exception as e:
        print(f"  [HQ VLM error] {e}")
    return f"{template} — {directive[:80]}"


def phase6_human_query(
    plan_a: LocalPlan, plan_b: LocalPlan,
    offer_a: Offer, offer_b: Offer,
    img_a: str, img_b: str,
    task: str = "",
    use_human_query: bool = True,
    unresolved_conflicts: List = None,
) -> Tuple[Dict[str, str], List[str], List[str]]:
    """
    HQ 발동 조건 (둘 중 하나):
    1. 협상 후에도 해결 안 된 DEPENDENCY conflict가 남아있음
    2. 두 에이전트 모두 제공 못 하는 need가 있고,
       두 방의 이미지 어디에도 관련 객체가 없음 (진짜 외부 도움 필요)
    """
    _banner("HUMAN QUERY")

    if not use_human_query:
        print("  [ABLATION] disabled.")
        return {}, [], []

    unresolved_conflicts = unresolved_conflicts or []
    hq_candidates = []

    # ── 조건 1: 협상 후에도 남은 DEPENDENCY conflict ──────────────────────────
    for c_entry in unresolved_conflicts:
        ct = str(c_entry.conflict_type) if hasattr(c_entry, "conflict_type") else ""
        if "DEPENDENCY" in ct or "DEP_CYCLE" in ct:
            hq_candidates.append(c_entry)

    # ── 조건 2: 두 에이전트 모두 제공 못 하는 need ───────────────────────────
    # 단, 두 방 이미지(obs_scope + can_do)에서도 찾을 수 없는 경우만
    _INFO_KW = {"confirmation","confirm","ready","status","notify",
                "check","verified","done","complete","whether"}
    all_provides   = offer_a.can_provide + offer_b.can_provide
    all_can_do_kw  = set()
    for cd in offer_a.can_do + offer_b.can_do:
        all_can_do_kw |= _kw(cd)
    obs_kw = _kw(offer_a.obs_scope) | _kw(offer_b.obs_scope)
    all_visible = all_can_do_kw | obs_kw

    for need in offer_a.need_from_other + offer_b.need_from_other:
        if _kw(need) & _INFO_KW:
            continue
        need_kw = _kw(need)
        # 두 에이전트 모두 provide 못 함
        if any(_fuzzy_match_soft(need, p) for p in all_provides):
            continue
        # 두 방 이미지 어디에도 관련 객체 없음
        if need_kw & all_visible:
            continue
        # 진짜 외부 도움이 필요한 상황
        class _FakeConflict:
            conflict_type = "UNMATCHED"
            description   = f"Neither agent can provide '{need}' and it is not visible in either room."
            step_ids      = []
            agent_ids     = []
            fix_hint      = "Human operator may need to provide this."
        hq_candidates.append(_FakeConflict())

    if not hq_candidates:
        print("  No HQ needed — agents can resolve all issues.")
        return {}, [], []

    print(f"  HQ triggers ({len(hq_candidates)}):")
    for h in hq_candidates:
        print(f"    [{h.conflict_type}] {h.description[:80]}")

    # ── Human에게 질문 ──────────────────────────────────────────────────────
    answers: Dict[str, str] = {}
    asked:   List[str]      = []
    triggered = [f"[{h.conflict_type}] {h.description}" for h in hq_candidates]

    for i, c_entry in enumerate(hq_candidates[:HQ_TOP_K], 1):
        print(f"\n  Generating Q{i}...", end=" ", flush=True)
        ctype = str(c_entry.conflict_type) if isinstance(c_entry.conflict_type, str)                 else c_entry.conflict_type
        q = _generate_hq_question(
            ctype, c_entry.description,
            offer_a, offer_b, img_a,
        )
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

    # HQ 답변 반영: VLM으로 각 에이전트 플랜 polish (2회 호출)
    if answers:
        _banner("HUMAN QUERY — PLAN POLISH")
        hq_ctx = "\n".join(f"Q: {q}\nA: {a}" for q, a in answers.items())
        print(f"  Polishing plans with human context...")

        def _polish(plan: LocalPlan, offer: Offer, img: str) -> LocalPlan:
            steps_summary = "\n".join(
                f"  {i+1}. {s.action}"
                + (f" [PASS→{s.target_agent}]" if s.handoff_type == "PASS" else "")
                for i, s in enumerate(plan.steps)
            )
            prompt = f"""You are {offer.agent_id} ({offer.room_type}).
Task: "{task}"

Human clarification:
{hq_ctx}

Your current plan:
{steps_summary}

Update your plan to incorporate the human's answer.
- Modify or add steps based on the human's clarification.
- Keep steps that are already correct.
- Use ONLY objects visible in your room image.
- 2–5 steps, no filler.
- Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{"plan_steps": [
  {{"step_id":1,"time_min":1,"action":"...","preconditions":[],"depends_on":[],
    "handoff_type":null,"target_agent":null,"uncertainty":0.1,"notes":""}}
]}}
</JSON>"""
            try:
                raw, logp = run_vlm(img, prompt)
                polished = _parse_local_plan(
                    raw, logp, offer,
                    step_offset=plan.steps[0].step_id - 1 if plan.steps else 0
                )
                if polished.steps:
                    print(f"  [{offer.agent_id}] polished: {len(polished.steps)} steps")
                    return polished
            except Exception as e:
                print(f"  [{offer.agent_id}] polish failed: {e}")
            return plan

        with __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor(2) as ex:
            fa = ex.submit(_polish, plan_a, offer_a, img_a)
            fb = ex.submit(_polish, plan_b, offer_b, img_b)
            pa_new = fa.result()
            pb_new = fb.result()

        plan_a.steps[:] = pa_new.steps
        plan_a.handoffs[:] = pa_new.handoffs
        plan_b.steps[:] = pb_new.steps
        plan_b.handoffs[:] = pb_new.handoffs
        print("  Polish complete.")

    return answers, triggered, asked


# ══════════════════════════════════════════════════════════════════════════════
# FINALIZE: RULE-BASED MERGE
# ══════════════════════════════════════════════════════════════════════════════

def phase_finalize(
    steps_a: List[Dict], steps_b: List[Dict],
    offer_a: Offer, offer_b: Offer,
    human_answers: Dict[str, str],
    verbose: str = "full",
) -> List[Dict]:
    _banner("MERGE — final joint plan")

    if human_answers and verbose in ("full", "summary"):
        print("  Human answers:")
        for q, a in human_answers.items():
            print(f"    Q: {q[:65]}...")
            print(f"    A: {a}")

    # HQ 답변을 joint plan에 반영 — VLM으로 필요한 스텝 추가
    if human_answers:
        hq_ctx = "\n".join(f"Q: {q}\nA: {a}" for q, a in human_answers.items())
        # "A should do X" 형태 답변에서 agent_A 스텝 추가
        a_lower_all = " ".join(human_answers.values()).lower()
        _A_KW = {"agent a", "kitchen", "a should", "kitchen agent", "agent_a"}
        _B_KW = {"agent b", "bedroom", "b should", "bedroom agent", "agent_b"}
        for food_kw in ["meal", "food", "snack", "drink", "coffee", "breakfast", "lunch"]:
            if food_kw in a_lower_all:
                # A가 음식을 준비해야 하는 경우
                if any(kw in a_lower_all for kw in _A_KW):
                    already = any(food_kw in s.get("action","").lower() for s in steps_a)
                    if not already:
                        new_sid = max((s["step_id"] for s in steps_a + steps_b), default=0) + 1
                        steps_a.append({
                            "step_id": new_sid, "time_min": 1,
                            "room": steps_a[0].get("room","kitchen") if steps_a else "kitchen",
                            "agent_id": "agent_A",
                            "action": f"prepare {food_kw} using visible kitchen items",
                            "preconditions": [], "depends_on": [],
                            "handoff_type": None, "target_agent": None,
                            "uncertainty": 0.2, "notes": "added per human clarification",
                        })
                        print(f"  [FINALIZE] Added A step: 'prepare {food_kw}' per HQ")
                break

    merged = list(steps_a) + list(steps_b)
    merged.sort(key=lambda s: (s.get("time_min", 0), s.get("step_id", 0)))

    old_to_new: Dict[int, int] = {}
    for new_id, s in enumerate(merged, start=1):
        old_to_new[s["step_id"]] = new_id

    for s in merged:
        s["step_id"]    = old_to_new[s["step_id"]]
        s["depends_on"] = [old_to_new[d] for d in s.get("depends_on", [])
                           if d in old_to_new]
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


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMAT (자연어, 깔끔한 출력)
# ══════════════════════════════════════════════════════════════════════════════


merge_plans = phase_finalize  # alias

def format_joint_plan(plan: List[Dict], task: str = "") -> str:
    if not plan:
        return "  (empty)"

    id_to_step: Dict[int, Dict] = {s["step_id"]: s for s in plan}
    steps_a = [s for s in plan if s.get("agent_id") == "agent_A"]
    steps_b = [s for s in plan if s.get("agent_id") == "agent_B"]
    room_a  = steps_a[0].get("room", "Room A") if steps_a else "Room A"
    room_b  = steps_b[0].get("room", "Room B") if steps_b else "Room B"
    n_pass  = sum(1 for s in plan if s.get("handoff_type") == "PASS")
    n_info  = sum(1 for s in plan if s.get("handoff_type") == "INFORM")

    SEP = "━" * 68
    lines: List[str] = [SEP]
    if task:
        lines.append(f'  "{(task[:60]+"…") if len(task)>60 else task}"')
    lines.append(f"  {room_a.upper()} (agent_A)  +  {room_b.upper()} (agent_B)")
    lines.append(f"  {len(plan)} steps  |  {n_pass} handoff(s)  |  {n_info} notify")
    lines.append(SEP)
    lines.append("")

    # 위상 정렬: depends_on 기반으로 순서 결정 (time_min 무시)
    def _topo_sort(steps):
        from collections import defaultdict, deque
        id_map = {s["step_id"]: s for s in steps}
        in_deg = {s["step_id"]: 0 for s in steps}
        graph  = defaultdict(list)
        for s in steps:
            for d in s.get("depends_on", []):
                if d in id_map:
                    graph[d].append(s["step_id"])
                    in_deg[s["step_id"]] += 1
        queue = deque(sorted(
            [sid for sid, deg in in_deg.items() if deg == 0],
            key=lambda sid: id_map[sid].get("step_id", 0)
        ))
        result = []
        while queue:
            sid = queue.popleft()
            result.append(id_map[sid])
            for nxt in sorted(graph[sid]):
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    queue.append(nxt)
        # cycle 있으면 나머지 추가
        done = {s["step_id"] for s in result}
        result += [s for s in steps if s["step_id"] not in done]
        return result
    sorted_plan = _topo_sort(plan)
    for idx, s in enumerate(sorted_plan, 1):
        agent  = s.get("agent_id","?")
        room   = s.get("room","?")
        action = s.get("action","")
        ht     = s.get("handoff_type")
        tgt    = s.get("target_agent","")
        deps   = s.get("depends_on",[])

        suffix = ""
        badge  = "  "
        if ht == "PASS":
            suffix = f"  ──────► {tgt}"
            badge  = "[PASS→]"
        elif ht == "INFORM":
            suffix = f"  ~~~~~~~► {tgt}"
            badge  = "[NOTIFY→]"

        # receive step 감지 — cross-agent PASS에 depend
        cross_pass = [id_to_step[d] for d in deps
                      if d in id_to_step
                      and id_to_step[d].get("agent_id") != agent
                      and id_to_step[d].get("handoff_type") == "PASS"]
        if cross_pass:
            badge = "[←RECV]"

        lines.append(f"  {idx:>2}. [{room}] [{agent}] {badge}")
        lines.append(f"      {action}{suffix}")

        cross_all = [id_to_step[d] for d in deps
                     if d in id_to_step and id_to_step[d].get("agent_id") != agent]
        for dep in cross_all:
            da = dep["action"]
            ds = (da[:45]+"…") if len(da)>45 else da
            lines.append("      → waits for " + dep.get('agent_id','?') + ": '" + ds + "'")
        lines.append("")

    coord: List[str] = []
    for s in sorted_plan:
        ht = s.get("handoff_type")
        if not ht: continue
        src = s.get("agent_id","?")
        tgt = s.get("target_agent","?")
        act = s.get("action","")
        short = (act[:50]+"…") if len(act)>50 else act
        recv = [r for r in plan
                if s["step_id"] in r.get("depends_on",[])
                and r.get("agent_id") != src]
        if ht == "PASS":
            ra = recv[0]["action"] if recv else "no receive step"
            rs = (ra[:45]+"…") if len(ra)>45 else ra
            coord += [
                f"  [PASS]   {src} ──────► {tgt}",
                "     sent:  " + short,
                f"  [RECV] {tgt} ◄──────",
                "     recv:  " + rs,
                "",
            ]
        elif ht == "INFORM":
            coord += [
                f"  [INFORM] {src} ~~~~~~~► {tgt}",
                "     msg:   " + short,
                "",
            ]

    if coord:
        lines.append("─"*68)
        lines.append("  COORDINATION SUMMARY")
        lines.append("")
        lines.extend(coord)

    lines.append(SEP)
    return "\n".join(lines)

# ── 함수 alias (하위 호환 + 새 이름) ──────────────────────────────────────
run_negotiation       = phase4_negotiation
check_plan_quality    = phase5_convergence_check
run_human_query       = phase6_human_query
