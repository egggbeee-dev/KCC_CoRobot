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

_P1_EXAMPLE = """
EXAMPLE — kitchen agent, task "prepare movie night":
<JSON>
{
  "room_type": "kitchen",
  "observation": "Kitchen with fruits on island, bread basket, countertops, sink.",
  "obs_scope": "island, counter, shelf, sink, stove, fruits, bread basket",
  "can_do": [
    "place apple and orange from island onto serving tray",
    "arrange bread from basket onto plate",
    "fill water glass from tap",
    "wipe counter surface with cloth",
    "clean visible sink with sponge"
  ],
  "cannot_do": [
    {"action": "arrange living room seating", "reason": "NO_OBJECT"},
    {"action": "adjust TV lighting", "reason": "NO_OBJECT"}
  ],
  "conf": {
    "place apple and orange from island onto serving tray": 0.9,
    "arrange bread from basket onto plate": 0.85,
    "fill water glass from tap": 0.9,
    "wipe counter surface with cloth": 0.95,
    "clean visible sink with sponge": 0.9
  },
  "can_provide": ["snack tray with fruits and bread"],
  "need_from_other": ["living room table cleared for snacks"]
}
</JSON>
""".strip()


def _build_phase1_prompt(task: str) -> str:
    return f"""You are an embodied home agent observing your room.

Global task: "{task}"

{_P1_EXAMPLE}

Generate your Offer for YOUR room only. Be faithful to what is actually visible.

RULES:
1. can_do: max {MAX_CAN_DO} actions using ONLY visible objects.
   - Prioritize actions that DIRECTLY contribute to the global task.
   - Format: "verb + specific visible object + purpose"
2. cannot_do: max {MAX_CANNOT_DO}. reason: NO_OBJECT | NO_CAPABILITY | UNCERTAIN
3. conf: confidence [0.0–1.0] per can_do item.
4. can_provide: items you can PHYSICALLY CARRY to the room boundary for the other agent.
   - ONLY tangible objects: food tray, drink, meal, document, tool
   - NOT: "cleaned sink", "confirmation", "status", "organized shelf"
   - Keep to 1–2 items maximum. Only what the OTHER agent actually needs.
5. need_from_other: 1–2 things you genuinely need from the other agent to complete the task.
   Focus on physical items or critical information, not generic confirmations.
6. Think about COLLABORATION: what can you prepare that helps the other agent?
7. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "room_type": "...",
  "observation": "one concise sentence describing the room",
  "obs_scope": "comma-separated list of visible objects and areas",
  "can_do": ["verb + specific object + purpose"],
  "cannot_do": [{{"action": "...", "reason": "NO_OBJECT"}}],
  "conf": {{"action text": 0.9}},
  "can_provide": ["max 2 tangible items for the other agent"],
  "need_from_other": ["max 2 specific needs"]
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

    # can_provide: 물리적으로 전달 가능한 아이템만 허용
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


def _is_vlm_refusal(raw: str) -> bool:
    """VLM이 거절/오류 응답을 반환했는지 판단."""
    _REFUSAL_PHRASES = [
        "i'm sorry", "i cannot", "i can't", "i apologize",
        "as an ai", "not able to", "unable to",
    ]
    lower = raw.strip().lower()
    # JSON 태그가 없고 거절 문구가 있으면 거절로 판단
    has_json = "<json>" in lower or "{" in lower
    if has_json:
        return False
    return any(p in lower for p in _REFUSAL_PHRASES)


def _vlm_with_retry(img: str, prompt: str, log_probs: bool,
                    max_retries: int = 2) -> Tuple[str, List[float]]:
    """VLM 호출 + 거절 시 재시도."""
    for attempt in range(max_retries + 1):
        raw, logp = run_vlm(img, prompt, log_probs)
        if not _is_vlm_refusal(raw):
            return raw, logp
        print(f"  [RETRY] VLM refusal detected (attempt {attempt+1}/{max_retries+1}), retrying...")
    print(f"  [WARN] VLM still refusing after {max_retries} retries, using empty response.")
    return raw, logp


def phase1_offer(
    img_a: str, img_b: str, task: str, verbose: str = "full",
) -> Tuple[Offer, Offer]:
    _banner("PHASE 1 — OBSERVATION & OFFER GENERATION")
    prompt = _build_phase1_prompt(task)

    with __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor(max_workers=2) as ex:
        fut_a = ex.submit(_vlm_with_retry, img_a, prompt, False)
        fut_b = ex.submit(_vlm_with_retry, img_b, prompt, False)
        raw_a, _ = fut_a.result()
        raw_b, _ = fut_b.result()

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

_P2_EXAMPLE = """
EXAMPLE — kitchen agent:
<JSON>
{
  "plan_steps": [
    {"step_id":1,"time_min":0,"action":"place apple and orange from island onto serving tray",
     "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,
     "uncertainty":0.1,"notes":""},
    {"step_id":2,"time_min":5,"action":"arrange bread from basket onto plate",
     "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,
     "uncertainty":0.1,"notes":""},
    {"step_id":3,"time_min":10,"action":"carry snack tray to kitchen doorway for agent_B pickup",
     "preconditions":["snacks on tray"],"depends_on":[1,2],
     "handoff_type":"PASS","target_agent":"agent_B",
     "uncertainty":0.15,"notes":"snack tray ready at doorway"},
    {"step_id":4,"time_min":15,"action":"wipe counter surface with cloth",
     "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,
     "uncertainty":0.1,"notes":""}
  ]
}
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


def _build_phase2_prompt(my: Offer, other: Offer, task: str, use_offer: bool) -> str:
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

    return f"""You are the {my.room_type} agent ({my.agent_id}).
Global task: "{task}"

{ctx}

{_P2_EXAMPLE}

{_P2_HANDOFF_RULES}

Generate YOUR local plan. Think step by step:
1. What does the global task require from YOUR room specifically?
2. What can you prepare for the other agent (see can_provide above)?
3. What do you need from the other agent (see need_from_other above)?

PLANNING RULES:
1. Steps ONLY in your room ({my.room_type}), using ONLY visible objects.
2. Generate 4–6 steps over 0–25 minutes. NO repeated actions.
3. Prioritize actions that DIRECTLY contribute to the global task.
4. HANDOFF — if can_provide is NOT empty:
   - Prepare the item first (1–2 prep steps)
   - Then add ONE PASS step: "carry [item] to [room] doorway for [other_agent] pickup"
   - PASS step must have depends_on=[prep step ids]
5. INFORM — if you want to notify completion:
   - "notify [other_agent]: [what is ready]"
   - handoff_type="INFORM", target_agent=[other_agent]
6. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "plan_steps": [
    {{"step_id":1,"time_min":0,"action":"verb + specific object",
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
        # step_offset 적용 후 자기 플랜의 step_id 범위 내 deps만 허용
        # cross-agent deps(상대 에이전트 step_id 참조)는 제거
        # agent_A: step_offset=0 → 1~99 범위
        # agent_B: step_offset=100 → 101~199 범위
        # LLM이 상대 step_id(예: A가 B의 101 참조)를 넣는 경우 필터링
        _my_offset   = step_offset
        _other_start = 100 if step_offset == 0 else 0
        _other_end   = 199 if step_offset == 0 else 99
        deps = [
            d + step_offset for d in raw_deps
            if not (_other_start <= d <= _other_end)  # 상대 범위 제외
        ]
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
            prev_steps = [p for p in steps if p.step_id < s.step_id and not p.handoff_type]
            if prev_steps:
                s.depends_on = [max(prev_steps, key=lambda p: p.step_id).step_id]
                print(f"  [NORM] step{s.step_id} PASS: auto-linked depends_on={s.depends_on}")
            else:
                print(f"  [NORM] step{s.step_id} PASS removed: no depends_on")
                s.handoff_type = None; s.target_agent = None; continue

        if not [d for d in s.depends_on if d in my_ids]:
            print(f"  [NORM] step{s.step_id} PASS removed: deps not in own plan")
            s.handoff_type = None; s.target_agent = None; continue

        # cross-agent deps 제거
        s.depends_on = [d for d in s.depends_on if d in my_ids]

        # 중복 PASS 제거 — payload 키워드가 다르면 중복 아님
        def _ppkw(action):
            m = re.search(r"(?:carry|bring|deliver|transport)\s+(.+?)\s+(?:to |for )", action, re.I)
            return _kw(m.group(1)) if m else _kw(action)
        s_pl = _ppkw(s.action)
        truly_dup = any(
            _fuzzy_match(s.action, prev.action, min_overlap=3)
            and bool(s_pl & _ppkw(prev.action))
            and not bool(s_pl - _ppkw(prev.action))
            for prev in seen_pass
        )
        if truly_dup:
            print(f"  [NORM] step{s.step_id} PASS removed: duplicate payload")
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

        # 수령/배치 관련 동사만 target 후보로 — close/adjust 등 무관한 step 제외
        _LINK_RECV_VERBS = {"receive","accept","collect","get","take","pick",
                            "place","put","set","arrange","bring","use","set up"}
        def _is_recv_candidate(s: PlanStep) -> bool:
            first = s.action.lower().split()[0] if s.action.strip() else ""
            return first in _LINK_RECV_VERBS

        # 1단계: keyword 직접 겹침 + 수령 관련 동사
        targets = [s for s in receiver.steps
                   if not s.handoff_type
                   and pkw & _kw(s.action)
                   and _is_recv_candidate(s)]

        # 1.5단계: keyword 강하게 겹침 + 수령 관련 동사 (동사 무관하면 오탐)
        if not targets:
            targets = [s for s in receiver.steps
                       if not s.handoff_type
                       and len(pkw & _kw(s.action)) >= 2
                       and _is_recv_candidate(s)]

        # 2단계: food item + 배치 동사
        if not targets and pkw & _FOOD_KW_LOCAL:
            targets = [s for s in receiver.steps
                       if not s.handoff_type
                       and _is_recv_candidate(s)
                       and _kw(s.action) & _FOOD_KW_LOCAL]

        # 3단계: need_from_other와 keyword 실질 겹침 (min 2개) + 수령 동사
        # fuzzy_match_soft 단독은 범용 need("workspace setup")에서 오탐 위험
        if not targets:
            for need in r_offer.need_from_other:
                need_kw = _kw(need)
                matched = [s for s in receiver.steps
                           if not s.handoff_type
                           and _is_recv_candidate(s)
                           and len(need_kw & _kw(s.action)) >= 2]
                if matched:
                    targets = matched
                    break

        # receiver step이 없으면 자동 추가
        if not targets:
            _add_receive_step(pass_step, receiver, s_offer, r_offer, rid)
            return

        coord_time = pass_step.time_min
        # receiver 플랜 step_id 집합 — 이 범위 + PASS step_id만 deps로 허용
        receiver_ids = {s.step_id for s in receiver.steps}
        # sender 플랜의 step_id도 추출 (A가 B PASS 받는 경우, sender=B이므로 B ids 제외)
        # 즉 rs.depends_on에서 receiver 자신 ids가 아닌 외부 ids는 PASS step_id 하나만 남김
        for rs in targets:
            # 기존 cross-agent deps 제거 (receiver 자신 ids + PASS step_id만 유지)
            clean_deps = [
                d for d in rs.depends_on
                if d in receiver_ids  # receiver 자신 step
            ]
            # PASS step_id 추가 (중복 방지)
            if pass_step.step_id not in clean_deps:
                clean_deps = sorted(clean_deps + [pass_step.step_id])
            rs.depends_on = clean_deps
            if rs.time_min <= coord_time:
                rs.time_min = coord_time + 1
            print(f"  [ENSURE] {rid} step{rs.step_id} "
                  f"'{rs.action[:40]}' ← PASS step{pass_step.step_id} "
                  f"(T={rs.time_min}m)")

    def _add_receive_step(
        pass_step: PlanStep,
        receiver: LocalPlan,
        s_offer: Offer,
        r_offer: Offer,
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

    if use_offer:
        _banner("PHASE 2b — HANDOFF COORDINATION (rule-based)")
        plan_a, plan_b = _ensure_pass(plan_a, plan_b, offer_a, offer_b)

    if verbose in ("full", "summary"):
        _log("PLAN A", jdump(local_plan_to_dict(plan_a)))
        _log("PLAN B", jdump(local_plan_to_dict(plan_b)))

    pass_a = sum(1 for s in plan_a.steps if s.handoff_type == "PASS")
    pass_b = sum(1 for s in plan_b.steps if s.handoff_type == "PASS")
    print(f"\n  A: steps={len(plan_a.steps)} U={plan_a.U_plan:.3f} PASS={pass_a}")
    print(f"  B: steps={len(plan_b.steps)} U={plan_b.U_plan:.3f} PASS={pass_b}")
    for h in plan_a.handoffs + plan_b.handoffs:
        tag = h.handoff_type
        print(f"  [{h.agent_id}->{tag}] step{h.step_id} → {h.target_agent} | {h.action[:55]}")
    return plan_a, plan_b


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
    # PASS payload 키워드 수집 — PASS에 포함된 아이템은 receiver에서 나와도 중복 아님
    _pass_payload_kw: Set[str] = set()
    for s in steps_a + steps_b:
        if s.handoff_type == "PASS":
            _pass_payload_kw |= _kw(s.action) | _kw(s.notes or "")

    _COMMON_VERBS = {
        "collect","gather","prepare","arrange","set","place","put","move",
        "clear","clean","organize","pick","get","take","bring","carry",
        "check","make","create","use","open","close","turn","adjust",
    }
    _LOCATION_KW = {
        "doorway","pickup","room","living","bedroom","kitchen","bathroom",
        "hallway","boundary","entrance","door","area","space","table","floor",
    }
    _STRIP_KW = _COMMON_VERBS | _LOCATION_KW

    def _payload_kw(action):
        m = re.search(r"(?:carry|bring|deliver|transport)\s+(.+?)\s+(?:to |for )", action, re.I)
        return _kw(m.group(1)) if m else (_kw(action) - _STRIP_KW)

    _recv_v = {"receive","collect","pick","get","take","accept","notify","inform","wait"}

    for sa in steps_a:
        for sb in steps_b:
            if sa.handoff_type == "PASS" or sb.handoff_type == "PASS":
                continue
            if set(sa.action.lower().split()[:1]) & _recv_v:
                continue
            if set(sb.action.lower().split()[:1]) & _recv_v:
                continue
            kw_a = _kw(sa.action) - _STRIP_KW
            kw_b = _kw(sb.action) - _STRIP_KW
            if kw_a & _pass_payload_kw or kw_b & _pass_payload_kw:
                continue
            overlap = kw_a & kw_b
            if len(overlap) >= 3:
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
                verb_i = si.action.lower().split()[0] if si.action.strip() else ""
                verb_j = sj.action.lower().split()[0] if sj.action.strip() else ""
                if verb_i != verb_j:
                    continue
                if si.handoff_type == "PASS":
                    kw_i, kw_j = _payload_kw(si.action), _payload_kw(sj.action)
                else:
                    kw_i = _kw(si.action) - _STRIP_KW
                    kw_j = _kw(sj.action) - _STRIP_KW
                if len(kw_i & kw_j) >= 3:
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
    # PASS/receive 스텝은 cannot_do 체크 제외
    # min_overlap=3으로 강화 (false positive 방지)
    _recv_verbs = {"receive","accept","get","take","pick"}
    for step, offer in all_steps:
        if step.handoff_type in ("PASS", "INFORM"):
            continue
        act_first = step.action.lower().split()[0] if step.action.strip() else ""
        if act_first in _recv_verbs:
            continue
        for c in offer.cannot_do:
            if _fuzzy_match(step.action, c.action, min_overlap=3):
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
    # 동의어 사전 — 같은 물건의 다른 표현을 하나로 매핑
    _SYNONYMS: Dict[str, Set[str]] = {
        "nightstand":  {"side table", "bedside table", "night table", "nightstand"},
        "side table":  {"nightstand", "bedside table", "night table", "side table"},
        "couch":       {"sofa", "couch", "settee"},
        "sofa":        {"sofa", "couch", "settee"},
        "fridge":      {"refrigerator", "fridge"},
        "refrigerator":{"refrigerator", "fridge"},
        "tv":          {"television", "tv", "monitor", "screen"},
        "television":  {"television", "tv", "monitor"},
        "worktop":     {"counter", "countertop", "countertops", "worktop", "surface"},
        "counter":     {"counter", "countertop", "countertops", "worktop", "surface"},
        "countertop":  {"counter", "countertop", "countertops", "worktop", "surface"},
        "countertops": {"counter", "countertop", "countertops", "worktop", "surface"},
        "blanket":     {"blanket", "duvet", "comforter", "quilt"},
        "duvet":       {"blanket", "duvet", "comforter", "quilt"},
        "pillow":      {"pillow", "cushion"},
        "cushion":     {"pillow", "cushion"},
        "desk":        {"desk", "table", "workstation"},
        "table":       {"desk", "table", "workstation"},
    }

    def _expand_pool(pool: Set[str]) -> Set[str]:
        """pool의 각 단어에 동의어를 추가해 확장."""
        expanded = set(pool)
        for w in pool:
            for syn_set in _SYNONYMS.values():
                if w in syn_set:
                    expanded |= {s.replace(" ", "_") for s in syn_set}
                    expanded |= {t for s in syn_set for t in s.split()}
        return expanded

    def _build_detect_obs_pool(offer: Offer) -> Set[str]:
        scope_kw = set(re.findall(r"\w+", offer.obs_scope.lower()))
        can_kw: Set[str] = set()
        for cd in offer.can_do:
            can_kw |= _kw(cd)
        for cd in offer.conf:
            can_kw |= _kw(cd)
        _BEV_D  = {"beverage","drink","drinks","coffee","tea","juice",
                   "water","cup","glass","mug","bottle","carafe","flask"}
        _FOOD_D = {"snack","food","fruit","bread","plate","tray","bowl",
                   "meal","cookie","nut","snacks"}
        cdt = " ".join(offer.can_do).lower()
        if any(w in cdt for w in {"drink","beverage","water","coffee","tea","juice"}):
            can_kw |= _BEV_D
        if any(w in cdt for w in {"snack","food","fruit","bread","prepare","serve"}):
            can_kw |= _FOOD_D
        return _expand_pool(scope_kw | can_kw)

    _OBS_SKIP_VERBS = {"receive","accept","notify","inform","wait","pick","take","get","collect"}
    for step, offer in all_steps:
        if step.handoff_type in ("PASS", "INFORM"):
            continue
        fw = step.action.lower().split()[0] if step.action.strip() else ""
        if fw in _OBS_SKIP_VERBS:
            continue
        if step.action.lower().startswith(("auto-added", "carry")):
            continue
        pool   = _build_detect_obs_pool(offer)
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
        + (f"\n    → HINT: {c.fix_hint}" if c.fix_hint else "")
        for c in conflicts
    ) or "  (none)"

    prev_text = "\n".join(
        f"  step{p.step_id}[{p.agent_id}] .{p.field}='{p.new_value}' ({p.reason})"
        for p in prev_props
    ) or "  (none)"

    # conflict별 구체적 컨텍스트 생성
    conflict_details: List[str] = []
    my_step_ids = {s["step_id"] for s in my_plan}

    for c in conflicts:
        ct = str(c.conflict_type)
        sids = c.step_ids

        if "DEPENDENCY" in ct:
            # PASS step을 찾아서 구체적으로 알려줌
            pass_steps = [s for s in (cur_a + cur_b)
                          if s["step_id"] in sids and s.get("handoff_type") == "PASS"]
            recv_agent = "agent_B" if my_agent == "agent_A" else "agent_A"
            if pass_steps:
                ps = pass_steps[0]
                conflict_details.append(
                    f"[DEPENDENCY] step{ps['step_id']} ({ps.get('agent_id')}) "
                    f"PASS action: '{ps['action'][:50]}'\n"
                    f"  → {recv_agent} MUST add a receive step with depends_on=[{ps['step_id']}].\n"
                    f"  → ONLY modify the RECEIVE step in {recv_agent}'s plan, not other steps."
                )
            else:
                conflict_details.append(f"[DEPENDENCY] {c.description}\n  → {c.fix_hint}")

        elif "REDUNDANCY" in ct:
            # 중복 스텝 양쪽 action을 명확히 보여줌
            dup_steps = [s for s in (cur_a + cur_b) if s["step_id"] in sids]
            if len(dup_steps) >= 2:
                conflict_details.append(
                    f"[REDUNDANCY] Two agents planned nearly identical actions:\n"
                    f"  step{dup_steps[0]['step_id']} ({dup_steps[0].get('agent_id')}): "
                    f"'{dup_steps[0]['action'][:50]}'\n"
                    f"  step{dup_steps[1]['step_id']} ({dup_steps[1].get('agent_id')}): "
                    f"'{dup_steps[1]['action'][:50]}'\n"
                    f"  → DELETE one of these TWO steps. Use field='delete'."
                )
            else:
                conflict_details.append(f"[REDUNDANCY] {c.description}\n  → {c.fix_hint}")

        elif "TEMPORAL" in ct:
            t_steps = [s for s in (cur_a + cur_b) if s["step_id"] in sids]
            conflict_details.append(
                f"[TEMPORAL] Time conflict at same time slot:\n"
                + "\n".join(
                    f"  step{s['step_id']} ({s.get('agent_id')}): '{s['action'][:50]}' "
                    f"at time_min={s.get('time_min')}"
                    for s in t_steps
                )
                + f"\n  → Shift ONE step's time_min to a different value. Use field='time_min'."
            )

        elif "CANNOT" in ct:
            bad_steps = [s for s in (cur_a + cur_b) if s["step_id"] in sids]
            if bad_steps:
                bs = bad_steps[0]
                conflict_details.append(
                    f"[CANNOT_DO] step{bs['step_id']} ({bs.get('agent_id')}): "
                    f"'{bs['action'][:50]}' — this agent CANNOT do this.\n"
                    f"  → DELETE step{bs['step_id']} using field='delete'."
                )

        else:
            conflict_details.append(f"[{ct}] {c.description}\n  → {c.fix_hint}")

    conflict_block = "\n\n".join(conflict_details) or "  (none)"

    return f"""You are {my_agent} ({my_offer.room_type}). ROUND {round_num}/{MAX_NEGOTIATION_ROUNDS}.
Task: "{task}"

YOUR PLAN (step_ids you can propose changes for):
{jdump(my_plan)}

{other_id}'s PLAN:
{jdump(other_plan)}

=== CONFLICTS TO RESOLVE ===
{conflict_block}

LOCKED step_ids (DO NOT touch): {sorted(locked) or '(none)'}
{other_id}'s previous proposals: {prev_text}

=== HOW TO RESPOND ===
- Make ONE proposal per conflict.
- ONLY modify the exact step_ids mentioned in the conflict description above.
- DO NOT modify unrelated steps.
- DEPENDENCY → field="depends_on", new_value="[PASS_STEP_ID]"
- REDUNDANCY → field="delete", new_value="true" for the duplicate step
- TEMPORAL   → field="time_min", new_value="NEW_TIME"
- CANNOT_DO  → field="delete", new_value="true"
- If you AGREE with the other agent's proposal, echo it with reason="ACCEPT".
  Without ACCEPT, the change is NOT finalized and appears again next round.

<JSON>
{{"proposals":[
  {{"step_id": <EXACT step_id from conflict>,
    "agent_id": "<agent who owns that step>",
    "field": "depends_on",
    "new_value": "[<PASS_STEP_ID>]",
    "reason": "DEPENDENCY: receive step must wait for PASS step"}}
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
    acc_b  = {p.step_id for p in props_b if "ACCEPT" in p.reason.upper()}
    acc_a  = {p.step_id for p in props_a if "ACCEPT" in p.reason.upper()}
    prop_a = {p.step_id for p in props_a if "ACCEPT" not in p.reason.upper()}
    prop_b = {p.step_id for p in props_b if "ACCEPT" not in p.reason.upper()}
    agreed        = (prop_a & acc_b) | (prop_b & acc_a)
    uncontested_a = prop_a - prop_b
    uncontested_b = prop_b - prop_a
    return existing | agreed | uncontested_a | uncontested_b


def _dicts_to_localplan(steps: List[Dict], offer: Offer) -> LocalPlan:
    from p2p_utils import compute_plan_uncertainty
    ps = [PlanStep(
        step_id=s["step_id"], time_min=s.get("time_min",0),
        room=s.get("room", offer.room_type), agent_id=s.get("agent_id", offer.agent_id),
        action=s.get("action",""), preconditions=s.get("preconditions",[]),
        depends_on=s.get("depends_on",[]), handoff_type=s.get("handoff_type"),
        target_agent=s.get("target_agent"), uncertainty=s.get("uncertainty",0.1),
        notes=s.get("notes",""),
    ) for s in steps]
    unc = compute_plan_uncertainty([s.uncertainty for s in ps]) if ps else 0.0
    return LocalPlan(offer.agent_id, ps, unc, [], [])


def phase4_negotiation(
    plan_a: LocalPlan, plan_b: LocalPlan,
    offer_a: Offer, offer_b: Offer,
    conflicts: List[ConflictEntry],
    img_a: str, img_b: str, task: str,
    verbose: str = "full",
) -> Tuple[List[Dict], List[Dict], List[NegotiationRound]]:
    _banner("PHASE 4 — P2P NEGOTIATION")

    cur_a = plan_steps_to_dicts(plan_a.steps)
    cur_b = plan_steps_to_dicts(plan_b.steps)

    initial = detect_conflicts(plan_a, plan_b, offer_a, offer_b)
    if not initial:
        print("  No conflicts detected → skip.")
        return cur_a, cur_b, []
    print(f"  Initial conflicts: {len(initial)}")

    locked: Set[int] = set()
    rounds: List[NegotiationRound] = []
    prev_a: List[NegotiationProposal] = []
    prev_b: List[NegotiationProposal] = []
    last_val: Dict[Tuple[int, str], str] = {}

    for rnd in range(1, MAX_NEGOTIATION_ROUNDS + 1):
        lp_a = _dicts_to_localplan(cur_a, offer_a)
        lp_b = _dicts_to_localplan(cur_b, offer_b)
        remaining = detect_conflicts(lp_a, lp_b, offer_a, offer_b)

        if not remaining:
            print(f"\n  Round {rnd}: all conflicts resolved.")
            break

        print(f"\n  -- Round {rnd}/{MAX_NEGOTIATION_ROUNDS} "
              f"(conflicts={len(remaining)}, locked={sorted(locked)}) --")
        if verbose in ("full", "summary"):
            for c in remaining:
                print(f"    [{c.conflict_type}] {c.description}")

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
                if "ACCEPT" in p.reason.upper():
                    out.append(p); continue
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

        applied = 0
        for prop in props_a + props_b:
            if _apply_proposal(cur_a, cur_b, prop, locked):
                applied += 1
                if verbose in ("full", "summary"):
                    print(f"  [APPLIED] step{prop.step_id}.{prop.field}")

        conflict_sids: Set[int] = {sid for c in remaining for sid in c.step_ids}
        locked = _lock_steps(props_a, props_b, conflict_sids, locked)
        rounds.append(NegotiationRound(rnd, props_a, props_b, sorted(locked)))
        print(f"  → Applied: {applied} | Locked: {sorted(locked)}")
        prev_a, prev_b = props_a, props_b

        if applied == 0 and rnd > 1:
            print(f"  No progress → early stop.")
            break

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
    _banner("PHASE 5 — CONVERGENCE CHECK")
    all_steps = steps_a + steps_b

    no_cycle = not _has_cycle(all_steps)

    # observability
    _DERIVED_OBS: Dict[str, Set[str]] = {
        "water":    {"water","drink","beverage","glass","cup","bottle"},
        "drink":    {"drink","beverage","cup","glass","water","bottle"},
        "beverage": {"beverage","drink","cup","glass","water","bottle"},
        "coffee":   {"coffee","cup","mug","drink","beverage"},
        "tea":      {"tea","cup","mug","drink","beverage"},
        "kettle":   {"kettle","water","drink","beverage","cup"},
        "snack":    {"snack","plate","tray","food","fruit","bread"},
        "food":     {"food","snack","plate","tray","meal","fruit"},
        "fruit":    {"fruit","snack","plate","tray","food"},
        "bread":    {"bread","snack","plate","tray","food"},
        "tray":     {"tray","plate","bowl","snack","food","drink"},
        "refrigerator": {"refrigerator","drink","beverage","water","food","snack"},
        "stove":    {"stove","pot","pan","water","food"},
        "sink":     {"sink","water","glass","cup","beverage"},
        "counter":  {"counter","countertop","countertops","tray","plate","snack"},
        "countertop":{"counter","countertop","countertops","tray","plate","snack"},
        "cabinet":  {"cabinet","plate","bowl","cup","glass","tray"},
    }

    def _obs_pool(offer: Offer) -> Set[str]:
        # obs_scope 키워드
        scope_kw = set(re.findall(r"\w+", offer.obs_scope.lower()))
        # obs_scope 기반 파생만 확장 (can_do 기반 파생은 과도한 완화 방지)
        derived: Set[str] = set()
        for w in scope_kw:
            if w in _DERIVED_OBS:
                derived |= _DERIVED_OBS[w]
        # can_do 행동 키워드는 파생 없이 직접 추가
        can_do_kw: Set[str] = set()
        for cd in offer.can_do:
            can_do_kw |= _kw(cd)
        return scope_kw | derived | can_do_kw

    pool_a = _obs_pool(offer_a)
    pool_b = _obs_pool(offer_b)
    _P5_OBS_SKIP = {
        "receive","accept","notify","inform","wait",
        "pick","take","get","collect","gather",
    }
    obs_ok = True
    for s in all_steps:
        if s.get("handoff_type") in ("PASS", "INFORM"):
            continue
        # auto-added receive step은 obs 체크 제외
        if "auto-added" in s.get("notes","").lower():
            continue
        fw = s.get("action","").lower().split()[0] if s.get("action","").strip() else ""
        if fw in _P5_OBS_SKIP:
            continue
        pool = pool_a if s.get("agent_id") == "agent_A" else pool_b
        kw   = _kw(s.get("action", ""))
        if kw and pool and not (kw & pool):
            obs_ok = False
            break

    # missing deps: PASS step이 있는데 receiver side에 대응 step이 없음
    pass_ids_a = {s["step_id"] for s in steps_a if s.get("handoff_type") == "PASS"}
    pass_ids_b = {s["step_id"] for s in steps_b if s.get("handoff_type") == "PASS"}
    deps_in_b  = {d for s in steps_b for d in s.get("depends_on", []) if d in pass_ids_a}
    deps_in_a  = {d for s in steps_a for d in s.get("depends_on", []) if d in pass_ids_b}

    _RECV_VERBS = {"receive","accept","pick","collect","get","take"}

    unmatched = (pass_ids_a - deps_in_b) | (pass_ids_b - deps_in_a)
    truly_unmatched: Set[int] = set()
    for sid in unmatched:
        ps = next((s for s in all_steps if s["step_id"] == sid), None)
        if not ps:
            continue
        target = ps.get("target_agent")
        target_steps = [s for s in all_steps if s.get("agent_id") == target]
        # 완화 1: target에 PASS 이후 시간의 step이 있으면 OK
        if any(s["time_min"] > ps["time_min"] for s in target_steps):
            continue
        # 완화 2: target에 receive 동사 step이 있으면 OK
        if any(
            s.get("action","").lower().split()[0] in _RECV_VERBS
            for s in target_steps if s.get("action","").strip()
        ):
            continue
        truly_unmatched.add(sid)

    no_missing = len(truly_unmatched) == 0

    all_remaining_ids = {s["step_id"] for s in steps_a + steps_b}
    unresolved = [
        c for c in conflicts
        if c.conflict_type in (ConflictType.REDUNDANCY, ConflictType.CANNOT_DO)
        and all(sid in all_remaining_ids for sid in c.step_ids)
    ]
    converged  = no_cycle and obs_ok and no_missing

    print(f"  No dep cycle   : {'OK' if no_cycle else 'FAIL'}")
    print(f"  Observability  : {'OK' if obs_ok else 'FAIL'}")
    print(f"  PASS matched   : {'OK' if no_missing else f'FAIL (unmatched={truly_unmatched})'}")
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
    prompt = f"""You are coordinating two home agents.
Agent A is in the {offer_a.room_type}. Agent B is in the {offer_b.room_type}.
Issue ({trigger_type}): {detail[:200]}

Write ONE clear question for the human operator that:
- Names which agent and step is involved
- Asks for a concrete decision
One sentence only. No preamble."""
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
        d = "Dependency cycle detected."
        triggered.append(f"[DEP_CYCLE] {d}")
        raw_triggers.append(("DEP_CYCLE", d, 0.90))

    if not convergence.no_missing_deps:
        d = "PASS step has no matching receive step."
        triggered.append(f"[DEPENDENCY] {d}")
        raw_triggers.append(("DEPENDENCY", d, 0.85))

    if not convergence.observability_ok:
        d = "A step references objects outside visible scope."
        triggered.append(f"[OBSERVABILITY] {d}")
        raw_triggers.append(("OBSERVABILITY", d, 0.75))

    for c in convergence.unresolved_conflicts:
        triggered.append(f"[{c.conflict_type}] {c.description}")
        raw_triggers.append((c.conflict_type, c.description, 0.80))

    _INFO_KW = {"confirmation","confirm","ready","status","notify",
                "check","verified","done","complete","that","whether"}
    all_provides = offer_a.can_provide + offer_b.can_provide
    for need in offer_a.need_from_other + offer_b.need_from_other:
        # 정보성 need (confirmation 류)는 UNMATCHED 탐지 제외
        if _kw(need) & _INFO_KW:
            continue
        if not any(_fuzzy_match_soft(need, p) for p in all_provides):
            d = f"No agent can provide: '{need}'"
            triggered.append(f"[UNMATCHED] {d}")
            raw_triggers.append(("UNMATCHED", d, 0.75))

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
        print(f"\n  Generating Q{i} [{ttype}]...", end=" ", flush=True)
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
# FINALIZE: RULE-BASED MERGE
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

    # HQ 답변 반영: "delete", "remove", "skip" → 관련 스텝 삭제
    #               "yes", "keep", "add" → 현재 유지
    _DELETE_HINTS = {"delete", "remove", "skip", "drop", "ignore", "no"}
    for q, a in human_answers.items():
        a_lower = a.lower()
        a_words = set(a_lower.split())
        if a_words & _DELETE_HINTS:
            # 질문에서 step_id 추출 시도
            import re as _re
            sids = [int(x) for x in _re.findall(r"step[\s_]?(\\d+)", q, _re.IGNORECASE)]
            for sid in sids:
                for plan in [steps_a, steps_b]:
                    before = len(plan)
                    plan[:] = [s for s in plan if s.get("step_id") != sid]
                    if len(plan) < before:
                        print(f"  [FINALIZE] step{sid} removed per human answer: '{a[:40]}'")

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

def format_joint_plan(plan: List[Dict], task: str = "") -> str:
    """
    Joint plan을 자연어 줄글 형식으로 출력.
    타임라인 순으로 각 스텝을 설명하고, PASS/INFORM은 명시적으로 표시.
    """
    if not plan:
        return "  (empty)"

    id_to_step: Dict[int, Dict] = {s["step_id"]: s for s in plan}

    steps_a = [s for s in plan if s.get("agent_id") == "agent_A"]
    steps_b = [s for s in plan if s.get("agent_id") == "agent_B"]
    room_a  = steps_a[0].get("room", "Room A") if steps_a else "Room A"
    room_b  = steps_b[0].get("room", "Room B") if steps_b else "Room B"

    n_pass   = sum(1 for s in plan if s.get("handoff_type") == "PASS")
    n_inform = sum(1 for s in plan if s.get("handoff_type") == "INFORM")
    max_t    = max((s.get("time_min", 0) for s in plan), default=0)

    SEP  = "━" * 68
    SEP2 = "─" * 68

    def _dep_note(s: Dict) -> str:
        deps = s.get("depends_on", [])
        if not deps:
            return ""
        cross = [id_to_step[d] for d in deps
                 if d in id_to_step and id_to_step[d].get("agent_id") != s.get("agent_id")]
        if cross:
            acts = ", ".join(f'"{x["action"][:35]}…"' if len(x["action"]) > 35
                             else f'"{x["action"]}"' for x in cross[:2])
            return f"\n       (waits for: {acts})"
        return ""

    lines: List[str] = []
    lines.append(SEP)
    task_str = task[:60] + "…" if len(task) > 60 else task
    if task_str:
        lines.append(f'  "{task_str}"')
    lines.append(
        f"  {room_a.upper()} (agent_A) + {room_b.upper()} (agent_B)  |  "
        f"{len(plan)} steps  |  {n_pass} handoff  |  {n_inform} notify  |  ~{max_t} min"
    )
    lines.append(SEP)
    lines.append("")

    # 타임라인 순서로 출력
    all_times = sorted({s.get("time_min", 0) for s in plan})

    for t in all_times:
        slot = sorted(
            [s for s in plan if s.get("time_min") == t],
            key=lambda s: s.get("agent_id", "")
        )
        lines.append(f"  T = {t:>2} min")
        lines.append(f"  {SEP2[:50]}")

        for s in slot:
            agent  = s.get("agent_id", "?")
            room   = s.get("room", "?")
            action = s.get("action", "")
            ht     = s.get("handoff_type")
            tgt    = s.get("target_agent", "")

            # 핸드오프 표시
            if ht == "PASS":
                marker = f"  ──► PASS to {tgt}"
            elif ht == "INFORM":
                marker = f"  ~~► NOTIFY {tgt}"
            else:
                marker = ""

            dep_note = _dep_note(s)
            lines.append(f"  [{room}] {action}{marker}{dep_note}")

        lines.append("")

    # COORDINATION SUMMARY
    coord = []
    for s in sorted(plan, key=lambda x: x.get("time_min", 0)):
        ht  = s.get("handoff_type")
        if not ht:
            continue
        src = s.get("agent_id", "?")
        tgt = s.get("target_agent", "?")
        t   = s.get("time_min", 0)
        act = s.get("action", "")
        if len(act) > 50:
            act = act[:50] + "…"
        receivers = [r for r in plan
                     if s["step_id"] in r.get("depends_on", [])
                     and r.get("agent_id") != src]
        if ht == "PASS":
            recv_t = receivers[0].get("time_min", t+1) if receivers else t+1
            coord.append(f"  {src} ──[PASS]──► {tgt}  |  '{act}'  (T={t}m → T={recv_t}m)")
        elif ht == "INFORM":
            coord.append(f"  {src} ~~[NOTIFY]~~► {tgt}  |  '{act}'  (T={t}m)")

    if coord:
        lines.append(SEP2)
        lines.append("  COORDINATION")
        lines.extend(coord)
    lines.append(SEP)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# COMPATIBILITY WRAPPERS — p2p_main.py 인터페이스 맞춤
# ══════════════════════════════════════════════════════════════════════════════
from concurrent.futures import ThreadPoolExecutor as _TPE


def observe_and_draft(img_a: str, img_b: str, task: str, verbose: str = "full"):
    """OBSERVATION: offer + draft plan → (offer_a, offer_b, draft_a, draft_b)"""
    offer_a, offer_b = phase1_offer(img_a, img_b, task, verbose=verbose)
    _banner("OBSERVATION — DRAFT PLAN")
    prompt_a = _build_phase2_prompt(offer_a, offer_b, task, use_offer=True)
    prompt_b = _build_phase2_prompt(offer_b, offer_a, task, use_offer=True)
    with _TPE(max_workers=2) as ex:
        fut_a = ex.submit(_vlm_with_retry, img_a, prompt_a, True)
        fut_b = ex.submit(_vlm_with_retry, img_b, prompt_b, True)
        raw_a, logp_a = fut_a.result()
        raw_b, logp_b = fut_b.result()
    if verbose == "full":
        _log("A DRAFT RAW", raw_a); _log("B DRAFT RAW", raw_b)
    draft_a = _parse_local_plan(raw_a, logp_a, offer_a, step_offset=0)
    draft_b = _parse_local_plan(raw_b, logp_b, offer_b, step_offset=AGENT_B_STEP_OFFSET)
    if verbose in ("full", "summary"):
        _log("DRAFT PLAN A", jdump(local_plan_to_dict(draft_a)))
        _log("DRAFT PLAN B", jdump(local_plan_to_dict(draft_b)))
    return offer_a, offer_b, draft_a, draft_b


def coordinate(
    offer_a, offer_b, draft_a, draft_b,
    img_a: str, img_b: str, task: str,
    use_offer: bool = True, verbose: str = "full",
):
    """MUTUAL AWARE LOCAL PLANNING: draft → final local plan → (plan_a, plan_b)"""
    _banner("MUTUAL AWARE LOCAL PLANNING")
    draft_a_json = jdump(plan_steps_to_dicts(draft_a.steps))
    draft_b_json = jdump(plan_steps_to_dicts(draft_b.steps))

    def _build_coord_prompt(my, other, my_draft, other_draft):
        passable = [p for p in my.can_provide if _is_passable(p)]
        return f"""You are the {my.room_type} agent ({my.agent_id}).
Global task: "{task}"

YOUR DRAFT PLAN (improve this):
{my_draft}

OTHER AGENT ({other.room_type}, {other.agent_id}) DRAFT PLAN:
{other_draft}

YOUR OFFER:
- can_provide: {json.dumps(passable, ensure_ascii=False)}
- need_from_other: {json.dumps(my.need_from_other, ensure_ascii=False)}

{_P2_HANDOFF_RULES}

Produce your FINAL local plan. Refine your draft:
1. Avoid redundancy with the other agent's plan.
2. Add a PASS step if can_provide is not empty.
3. Add a receive/use step if the other agent provides what you need.
4. IMPORTANT: depends_on must ONLY reference your OWN step_ids. Never reference the other agent's step_ids.
5. Generate 4-6 steps. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "plan_steps": [
    {{"step_id":1,"time_min":0,"action":"verb + specific object",
      "preconditions":[],"depends_on":[],"handoff_type":null,
      "target_agent":null,"uncertainty":0.1,"notes":""}}
  ]
}}
</JSON>"""

    prompt_a = _build_coord_prompt(offer_a, offer_b, draft_a_json, draft_b_json)
    prompt_b = _build_coord_prompt(offer_b, offer_a, draft_b_json, draft_a_json)
    with _TPE(max_workers=2) as ex:
        fut_a = ex.submit(_vlm_with_retry, img_a, prompt_a, True)
        fut_b = ex.submit(_vlm_with_retry, img_b, prompt_b, True)
        raw_a, logp_a = fut_a.result()
        raw_b, logp_b = fut_b.result()
    if verbose == "full":
        _log("A COORD RAW", raw_a); _log("B COORD RAW", raw_b)
    plan_a = _parse_local_plan(raw_a, logp_a, offer_a, step_offset=0)
    plan_b = _parse_local_plan(raw_b, logp_b, offer_b, step_offset=AGENT_B_STEP_OFFSET)
    if use_offer:
        _banner("HANDOFF SYNC — PASS COORDINATION")
        plan_a, plan_b = _ensure_pass(plan_a, plan_b, offer_a, offer_b)
    if verbose in ("full", "summary"):
        _log("LOCAL PLAN A", jdump(local_plan_to_dict(plan_a)))
        _log("LOCAL PLAN B", jdump(local_plan_to_dict(plan_b)))
    pass_a = sum(1 for s in plan_a.steps if s.handoff_type == "PASS")
    pass_b = sum(1 for s in plan_b.steps if s.handoff_type == "PASS")
    print(f"\n  A: steps={len(plan_a.steps)} U={plan_a.U_plan:.3f} PASS={pass_a}")
    print(f"  B: steps={len(plan_b.steps)} U={plan_b.U_plan:.3f} PASS={pass_b}")
    return plan_a, plan_b


# ── phase6_human_query 어댑터 ─────────────────────────────────────────────────
# p2p_main.py는 convergence.unresolved_conflicts만 넘기므로
# converged 여부를 정확히 재구성하기 위해 실제 convergence 객체를 캐싱
_last_convergence_result = None

def _store_convergence(result):
    """phase5 결과를 캐시 — phase6 어댑터에서 활용."""
    global _last_convergence_result
    _last_convergence_result = result
    return result

_phase5_original = phase5_convergence_check

def phase5_convergence_check(
    steps_a, steps_b, offer_a, offer_b, conflicts,
):
    result = _phase5_original(steps_a, steps_b, offer_a, offer_b, conflicts)
    return _store_convergence(result)

_phase6_original = phase6_human_query

def phase6_human_query(
    plan_a, plan_b, offer_a, offer_b,
    img_a: str = "", img_b: str = "",
    use_human_query: bool = True,
    task: str = "",
    unresolved_conflicts=None,
    convergence=None,
):
    if hasattr(img_a, "converged"):
        # 원본 직접 호출: (plan_a, plan_b, offer_a, offer_b, convergence_obj, img_a, img_b)
        convergence, img_a = img_a, img_b
        img_b = use_human_query if isinstance(use_human_query, str) else ""
        use_human_query = True
    else:
        # p2p_main.py 호출: unresolved_conflicts만 넘어옴
        # → 캐시된 phase5 결과를 우선 사용, 없으면 재구성
        if convergence is None:
            if _last_convergence_result is not None:
                convergence = _last_convergence_result
            else:
                from p2p_models import ConvergenceResult
                uc = unresolved_conflicts or []
                convergence = ConvergenceResult(
                    converged            = len(uc) == 0,
                    no_dep_cycle         = True,
                    observability_ok     = True,
                    no_missing_deps      = True,
                    unresolved_conflicts = uc,
                )
    return _phase6_original(plan_a, plan_b, offer_a, offer_b,
                            convergence, img_a, img_b, use_human_query=use_human_query)


# ── phase_finalize 어댑터 ─────────────────────────────────────────────────────
_phase_finalize_original = phase_finalize

def phase_finalize(
    steps_a, steps_b, offer_a, offer_b,
    human_answers, convergence=None, verbose: str = "full",
):
    if convergence is None:
        from p2p_models import ConvergenceResult
        convergence = ConvergenceResult(
            converged=True, no_dep_cycle=True,
            observability_ok=True, no_missing_deps=True,
            unresolved_conflicts=[],
        )
    return _phase_finalize_original(steps_a, steps_b, offer_a, offer_b,
                                    human_answers, convergence, verbose)


# ── format_joint_plan — 번호 목록 형식 ───────────────────────────────────────
def format_joint_plan(plan: List[Dict], task: str = "") -> str:
    if not plan:
        return "  (empty)"
    steps_a = [s for s in plan if s.get("agent_id") == "agent_A"]
    steps_b = [s for s in plan if s.get("agent_id") == "agent_B"]
    room_a  = steps_a[0].get("room", "Room A") if steps_a else "Room A"
    room_b  = steps_b[0].get("room", "Room B") if steps_b else "Room B"
    n_pass   = sum(1 for s in plan if s.get("handoff_type") == "PASS")
    n_inform = sum(1 for s in plan if s.get("handoff_type") == "INFORM")
    max_t    = max((s.get("time_min", 0) for s in plan), default=0)
    SEP = "━" * 68
    pass_sids = {s["step_id"] for s in plan if s.get("handoff_type") == "PASS"}
    # 명확한 수령 동사 — 무조건 RECV
    _EXPLICIT_RECV = {"receive", "accept"}
    # 문맥 의존 수령 동사 — from agent/doorway 등 수령 문맥 있을 때만 RECV
    _CTX_RECV = {"pick", "get", "take", "collect", "gather"}
    # 절대 RECV 아닌 동사 — depends_on에 PASS가 있어도 제외
    _NEVER_RECV = {
        "close", "open", "adjust", "wipe", "clean", "set", "turn",
        "arrange", "prepare", "place", "put", "check", "make", "clear",
        "organize", "unplug", "plug", "remove", "notify", "inform",
    }
    _RECV_CTX_KW = {
        "doorway", "pickup", "handover", "handoff",
        "from agent", "from kitchen", "from bedroom",
        "from living", "from other", "agent_a", "agent_b",
    }
    def _is_recv(s: Dict) -> bool:
        action = s.get("action","").lower()
        first  = action.split()[0] if action.strip() else ""
        # 절대 RECV 아닌 동사 → 무조건 False
        if first in _NEVER_RECV:
            return False
        # 명확 수령 동사 → True
        if first in _EXPLICIT_RECV:
            return True
        # depends_on에 PASS가 있고 + 수령 문맥 키워드 있으면 True
        if any(d in pass_sids for d in s.get("depends_on", [])):
            return first in _CTX_RECV and any(ctx in action for ctx in _RECV_CTX_KW)
        # depends_on에 PASS 없어도 수령 문맥 키워드 있으면 True
        if first in _CTX_RECV:
            return any(ctx in action for ctx in _RECV_CTX_KW)
        return False
    sorted_plan = sorted(plan, key=lambda s: (s.get("time_min", 0), s.get("agent_id", "")))
    lines: List[str] = [SEP]
    task_str = task[:65] + "…" if len(task) > 65 else task
    if task_str:
        lines.append(f'  Task : "{task_str}"')
    lines.append(
        f"  {room_a.upper()} (agent_A)  +  {room_b.upper()} (agent_B)"
        f"  |  {len(plan)} steps  |  {n_pass} handoff  |  {n_inform} notify  |  ~{max_t} min"
    )
    lines.append(SEP)
    lines.append("")
    for i, s in enumerate(sorted_plan, start=1):
        room   = s.get("room", "?")
        action = s.get("action", "")
        ht     = s.get("handoff_type")
        if ht == "PASS":       marker = "[PASS→] "
        elif ht == "INFORM":   marker = "[NOTIFY→] "
        elif _is_recv(s):      marker = "[←RECV] "
        else:                  marker = ""
        lines.append(f"  {i}. [{room}] {marker}{action}")
    lines.append("")
    lines.append(SEP)
    return "\n".join(lines)
