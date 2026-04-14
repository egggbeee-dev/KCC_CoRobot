# phases.py
#
# Phase 1 : Observation & Offer Generation
# Phase 2 : Local Planning (각 에이전트 독립)
# Phase 3 : Conflict Detection  (temporal / dependency / redundancy / ...)
# Phase 4 : P2P Negotiation     (최대 MAX_NEGOTIATION_ROUNDS 라운드, 구조화 제안)
# Phase 5 : Convergence Check   (rule-based, LLM 판단 없음)
# Phase 6 : Deferred Human Query (필요할 때만)
# Finalize: Rule-based merge (LLM 호출 없음)

from __future__ import annotations

import json
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Dict, List, Optional, Set, Tuple

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
4. can_do and cannot_do must NOT share any action.
5. conf: confidence per action [0.0-1.0]. Keys must exactly match can_do items.
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
    _banner("PHASE 1 - OBSERVATION & OFFER GENERATION")
    prompt = build_phase1_prompt(task)
    # 병렬 VLM 호출 (A/B 동시)
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_a = ex.submit(run_vlm, img_a, prompt, False)
        fut_b = ex.submit(run_vlm, img_b, prompt, False)
    raw_a, _ = fut_a.result()
    raw_b, _ = fut_b.result()

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
#
# LLM이 handoff 포함 플랜 생성 (기존 방식 유지)
# + _normalize_pass_steps()로 잘못된 PASS를 파싱 후 rule-based 제거
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

HANDOFF RULES:

PASS - physical item delivery to room boundary:
  CORRECT: prepare item first -> then carry to boundary -> PASS
  WRONG: PASS on a preparation step (place, arrange, set up)
  WRONG: PASS without depends_on pointing to your own preparation steps
  WRONG: more than one PASS for the same item
  -> A plan should have AT MOST 1-2 PASS steps total.

TIMING:
  If you RECEIVE an item via PASS, add a receive step with
  depends_on=[PASS_step_id] and time_min AFTER the PASS step.

UNIQUENESS: Do NOT repeat the same action twice.
""".strip()


def build_phase2_prompt(
    my: Offer, other: Offer, task: str, use_offer: bool = True
) -> str:
    matched_needs    = [n for n in my.need_from_other
                        if any(_fuzzy_match_soft(n, p) for p in other.can_provide)]
    matched_provides = [p for p in my.can_provide
                        if any(_fuzzy_match_soft(p, n) for n in other.need_from_other)]

    if use_offer:
        context = f"""YOUR OFFER:
{jdump(offer_to_dict(my))}

OTHER AGENT'S OFFER ({other.room_type}):
{jdump(offer_to_dict(other))}

MATCHED HANDOFF OPPORTUNITIES:
- Items YOU will PASS to other agent (YOU MUST create a PASS step if this is not empty):
  {json.dumps(matched_provides, ensure_ascii=False)}
- Items other agent will PASS to YOU (add a receive step if not empty):
  {json.dumps(matched_needs, ensure_ascii=False)}"""
    else:
        context = f"YOUR ROOM: {my.room_type}\nOTHER AGENT'S ROOM: {other.room_type}"

    return f"""You are the {my.room_type} agent ({my.agent_id}).

Global task: "{task}"

{context}

{PHASE2_FEW_SHOT}

Generate your LOCAL PLAN:
1. Steps ONLY in your room ({my.room_type}). Use only visible objects.
2. Generate 4-6 steps over 0-25 minutes.
3. Each step must be unique — no repeated actions.
4. **CRITICAL — PASS handoff**:
   - Items YOU will PASS to other agent: {{json.dumps(matched_provides, ensure_ascii=False)}}
   - If this list is NOT empty, you MUST add a PASS step.
   - PASS step: carry the prepared item to the room boundary.
     handoff_type="PASS", target_agent=<receiver_id>, depends_on=[<your prep step ids>]
   - The PASS step must come AFTER all prep steps for that item.
5. If receiving an item (Items other agent will PASS to YOU: {{json.dumps(matched_needs, ensure_ascii=False)}}):
   - Add a "receive" step: depends_on=[<PASS step id>], time_min AFTER the PASS step.
6. Return ONLY valid JSON inside <JSON> tags.

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
    """
    Rule-based post-processing: 잘못된 PASS를 제거한다.

    제거 조건:
    1. depends_on이 비어 있음 (준비 스텝 없이 바로 PASS)
    2. depends_on이 자기 플랜 내 step_id를 참조하지 않음
    3. 이미 유효한 PASS와 action이 유사한 중복 PASS
    """
    my_step_ids  = {s.step_id for s in steps}
    valid_passes: List[PlanStep] = []

    for s in steps:
        if s.handoff_type != "PASS":
            continue

        # 조건 0: target_agent 없으면 제거 (방향 불명확)
        if not s.target_agent or s.target_agent not in {"agent_A", "agent_B"}:
            print(f"  [NORM] step{s.step_id} PASS removed: no valid target_agent")
            s.handoff_type = None
            s.target_agent = None
            continue

        # 조건 1: depends_on 없으면 제거 (준비 스텝 없이 바로 PASS)
        if not s.depends_on:
            print(f"  [NORM] step{s.step_id} PASS removed: no depends_on")
            s.handoff_type = None
            s.target_agent = None
            continue

        # 조건 2: depends_on이 자기 플랜 내 스텝 참조하는지
        #         (cross-agent dep만 있는 경우 → receiver 역할이므로 PASS 제거)
        valid_deps = [d for d in s.depends_on if d in my_step_ids]
        if not valid_deps:
            print(f"  [NORM] step{s.step_id} PASS removed: deps not in own plan (likely receiver, not sender)")
            s.handoff_type = None
            s.target_agent = None
            continue

        # 조건 3: 이미 유효한 PASS와 action 유사하면 중복 제거
        is_dup = any(
            _fuzzy_match(s.action, prev.action, min_overlap=3)
            for prev in valid_passes
        )
        if is_dup:
            print(f"  [NORM] step{s.step_id} PASS removed: duplicate of existing PASS")
            s.handoff_type = None
            s.target_agent = None
            continue

        valid_passes.append(s)

    return steps


def _parse_local_plan(
    raw: str, log_probs: List[float], my: Offer,
    use_handoff: bool = True, step_offset: int = 0,
) -> LocalPlan:
    """
    LLM 출력 파싱.
    use_handoff=True: handoff_type 읽고 _normalize_pass_steps() 적용
    use_handoff=False: handoff_type 무시 (ablation용)
    step_offset: Agent B step_id 충돌 방지 (A=0, B=AGENT_B_STEP_OFFSET)
    """
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

        # 중복 액션 필터링
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
        step_unc = clamp01(
            json_unc * 0.5 + token_unc * 0.2 + (1 - action_conf) * 0.3
        )

        handoff_type = _norm_handoff(item.get("handoff_type")) if use_handoff else None
        target       = _norm_agent(item.get("target_agent"))   if use_handoff else None

        raw_deps = _norm_depends(item.get("depends_on"))
        deps     = [d + step_offset for d in raw_deps]

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

    # Rule-based PASS 정규화 (use_handoff=True일 때만)
    if use_handoff:
        steps = _normalize_pass_steps(steps)

    # handoffs 리스트 재구성 (normalize 후 기준)
    for s in steps:
        if s.handoff_type:
            payload = s.notes if s.handoff_type == "INFORM" else ""
            handoffs.append(Handoff(s.step_id, s.action, s.handoff_type, s.target_agent, payload))

    steps.sort(key=lambda s: (s.time_min, s.step_id))
    all_unc = [s.uncertainty for s in steps] if steps else [token_unc]
    return LocalPlan(my.agent_id, steps, compute_plan_uncertainty(all_unc), hq_list, handoffs)




def _ensure_pass_steps(
    plan_a: LocalPlan, plan_b: LocalPlan,
    offer_a: Offer, offer_b: Offer,
) -> Tuple[LocalPlan, LocalPlan]:
    """
    Phase 2 이후 PASS 보완: offer 매칭 기반으로 PASS가 없으면 자동 삽입.

    LLM이 PASS를 아예 생성하지 않은 경우를 보완한다.
    A→B: offer_a.can_provide ↔ offer_b.need_from_other 매칭 → A에 PASS 삽입
    B→A: offer_b.can_provide ↔ offer_a.need_from_other 매칭 → B에 PASS 삽입
    """
    def _inject_if_missing(
        sender_plan: LocalPlan, receiver_plan: LocalPlan,
        sender_offer: Offer, receiver_offer: Offer,
        sender_id: str, receiver_id: str,
    ) -> LocalPlan:
        # sender 플랜에 이미 유효한 PASS가 있으면 스킵
        has_pass = any(
            s.handoff_type == "PASS" and s.target_agent == receiver_id
            for s in sender_plan.steps
        )
        if has_pass:
            return sender_plan

        # can_provide ↔ 매칭 확인 (3단계, 점차 완화)
        matched_provides = []
        for p in sender_offer.can_provide:
            pkw = _resource_keywords(p)
            # 1차: need_from_other fuzzy match
            if any(_fuzzy_match_soft(p, n) for n in receiver_offer.need_from_other):
                matched_provides.append(p)
                continue
            # 2차: receiver can_do와 keyword 겹침 (stemming 기반)
            if any(pkw & _resource_keywords(cd) for cd in receiver_offer.can_do):
                matched_provides.append(p)
                continue
            # 3차: can_provide 자체가 있으면 inject 시도
            #      (LLM이 need_from_other를 애매하게 표현해도 보완)
            if p:
                matched_provides.append(p)

        if not matched_provides:
            return sender_plan

        provide = matched_provides[0]

        # sender 플랜에서 이 아이템과 관련된 prep step 찾기 (stemming 기반)
        provide_kw = _resource_keywords(provide)
        prep_steps = [
            s for s in sender_plan.steps
            if provide_kw & _resource_keywords(s.action)
            and s.handoff_type is None
        ]
        if not prep_steps:
            # fallback: 마지막 non-PASS step
            prep_steps = [s for s in sender_plan.steps if s.handoff_type is None]
        if not prep_steps:
            return sender_plan

        last_prep = max(prep_steps, key=lambda s: s.time_min)

        # PASS step 생성
        all_ids = {s.step_id for s in sender_plan.steps} | {s.step_id for s in receiver_plan.steps}
        new_sid  = max(all_ids, default=0) + 1
        while new_sid in all_ids:
            new_sid += 1

        pass_time = min(30, last_prep.time_min + 5)
        pass_step = PlanStep(
            step_id       = new_sid,
            time_min      = pass_time,
            room          = sender_offer.room_type,
            agent_id      = sender_id,
            action        = f"carry {provide} to room boundary for {receiver_id} pickup",
            preconditions = [f"step {last_prep.step_id} completed"],
            depends_on    = [last_prep.step_id],
            handoff_type  = "PASS",
            target_agent  = receiver_id,
            uncertainty   = 0.15,
            notes         = f"auto-injected PASS: {provide}",
        )
        sender_plan.steps.append(pass_step)
        sender_plan.steps.sort(key=lambda s: (s.time_min, s.step_id))
        sender_plan.handoffs.append(
            Handoff(new_sid, pass_step.action, "PASS", receiver_id, "")
        )
        print(f"  [ENSURE_PASS] {sender_id}: PASS step{new_sid} injected "
              f"(T={pass_time}m) for '{provide}' → {receiver_id}")

        # PASS 삽입 후 receiver 플랜에서 관련 스텝에 depends_on 자동 연결
        provide_kw = _resource_keywords(provide)
        for rs in receiver_plan.steps:
            # notify/receive/PASS 스텝은 건너뜀
            if rs.handoff_type == "PASS" or rs.action.lower().startswith(("receive", "notify")):
                continue
            act_kw = _resource_keywords(rs.action)
            # keyword 겹침으로 관련 스텝 탐지
            if provide_kw & act_kw:
                if new_sid not in rs.depends_on:
                    rs.depends_on = sorted(set(rs.depends_on + [new_sid]))
                    if rs.time_min <= pass_time:
                        rs.time_min = pass_time + 1
                    print(f"  [ENSURE_PASS] {receiver_id} step{rs.step_id} "
                          f"'{rs.action[:40]}' → depends_on=[...,{new_sid}], T={rs.time_min}m")

        return sender_plan

    plan_a = _inject_if_missing(plan_a, plan_b, offer_a, offer_b, "agent_A", "agent_B")
    plan_b = _inject_if_missing(plan_b, plan_a, offer_b, offer_a, "agent_B", "agent_A")
    return plan_a, plan_b

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
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_a = ex.submit(run_vlm, img_a, prompt_a, True)
        fut_b = ex.submit(run_vlm, img_b, prompt_b, True)
    raw_a, logp_a = fut_a.result()
    raw_b, logp_b = fut_b.result()

    if verbose == "full":
        _log("AGENT A RAW LOCAL PLAN", raw_a)
        _log("AGENT B RAW LOCAL PLAN", raw_b)

    plan_a = _parse_local_plan(raw_a, logp_a, offer_a, use_handoff, step_offset=0)
    plan_b = _parse_local_plan(raw_b, logp_b, offer_b, use_handoff, step_offset=AGENT_B_STEP_OFFSET)

    # PASS 자동 보완: offer 매칭 기반으로 PASS 없으면 삽입
    if use_handoff and use_offer:
        plan_a, plan_b = _ensure_pass_steps(plan_a, plan_b, offer_a, offer_b)

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

def _stem(word: str) -> str:
    """간단한 rule-based stemming: 복수형 s 제거 (snacks→snack, fruits→fruit)."""
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _resource_keywords(action: str) -> Set[str]:
    """액션에서 자원 키워드 추출 (불용어 제거 + stemming)."""
    words = set(re.findall(r"\w+", action.lower())) - FUZZY_STOPWORDS
    return {_stem(w) for w in words}


def detect_conflicts(
    plan_a: LocalPlan,
    plan_b: LocalPlan,
    offer_a: Offer,
    offer_b: Offer,
) -> List[ConflictEntry]:
    """
    두 로컬 플랜을 비교하여 충돌을 분류한다.

    탐지 항목:
      1. TEMPORAL          : 같은 time_min에 공유 자원 사용 (에이전트 간)
      2. DEPENDENCY        : need_from_other <-> can_provide 매칭은 있지만 PASS 미선언
      3. REDUNDANCY (inter): 두 에이전트가 동일 액션 시도
      4. REDUNDANCY (intra): 같은 에이전트 내 중복 액션   <- FIX
      5. CANNOT_DO         : cannot_do 위반
      6. OBSERVABILITY     : obs_scope 밖 액션
      7. HANDOFF           : PASS sender에 receiver 없음
    """
    conflicts: List[ConflictEntry] = []

    steps_a = plan_a.steps
    steps_b = plan_b.steps
    all_steps = [(s, offer_a) for s in steps_a] + [(s, offer_b) for s in steps_b]

    # ── 1. TEMPORAL (에이전트 간) ────────────────────────────────────────────
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
                # 다른 방이면 같은 키워드라도 실제 자원 충돌 아님
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
                            f"T={t}m: step {si.step_id}({si.agent_id}) and "
                            f"step {sj.step_id}({sj.agent_id}) share resource keywords {overlap}"
                        ),
                    ))

    # ── 2. DEPENDENCY ────────────────────────────────────────────────────────
    pass_targets_from_b = {h.target_agent for h in plan_b.handoffs if h.handoff_type == "PASS"}
    pass_targets_from_a = {h.target_agent for h in plan_a.handoffs if h.handoff_type == "PASS"}

    for need in offer_a.need_from_other:
        matched_provide = any(_fuzzy_match_soft(need, p) for p in offer_b.can_provide)
        has_pass = "agent_A" in pass_targets_from_b
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
        has_pass = "agent_B" in pass_targets_from_a
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

    # ── 3. REDUNDANCY inter-agent ────────────────────────────────────────────
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

    # ── 4. REDUNDANCY intra-agent (FIX) ─────────────────────────────────────
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
                            f"step{si.step_id} '{si.action}' ~= "
                            f"step{sj.step_id} '{sj.action}'"
                        ),
                    ))

    # ── 5. CANNOT_DO violation ───────────────────────────────────────────────
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

    # ── 6. OBSERVABILITY violation ───────────────────────────────────────────
    for step, offer in all_steps:
        scope_kw  = set(re.findall(r"\w+", offer.obs_scope.lower()))
        action_kw = _resource_keywords(step.action)
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

    # ── 7. HANDOFF mismatch ──────────────────────────────────────────────────
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

    prev_text = (
        "\n".join(
            f"  step {p.step_id} [{p.agent_id}]: field='{p.field}' -> new_value='{p.new_value}' | reason: {p.reason}"
            for p in prev_proposals
        )
        or "  (none)"
    )

    my_plan    = current_plan_a if my_agent == "agent_A" else current_plan_b
    other_plan = current_plan_b if my_agent == "agent_A" else current_plan_a
    other_id   = "agent_B" if my_agent == "agent_A" else "agent_A"

    return f"""You are {my_agent} ({my_offer.room_type}). NEGOTIATION ROUND {round_num}/{MAX_NEGOTIATION_ROUNDS}.

Global task: "{task}"

YOUR CURRENT PLAN (agent_id={my_agent}):
{jdump(my_plan)}

{other_id}'s CURRENT PLAN:
{jdump(other_plan)}

DETECTED CONFLICTS (only these need resolution):
{conflict_text}

LOCKED STEPS (already agreed - do NOT propose changes to these):
{locked_text}

{other_id}'s PREVIOUS PROPOSALS (consider accepting with reason="ACCEPT" or counter-proposing):
{prev_text}

YOUR CAPABILITIES:
- can_do: {json.dumps(my_offer.can_do, ensure_ascii=False)}
- cannot_do: {json.dumps([c.action for c in my_offer.cannot_do], ensure_ascii=False)}
- obs_scope: {my_offer.obs_scope}

INSTRUCTIONS:
1. For each conflict, propose ONE minimal change that resolves it.
2. Only propose changes to steps NOT in the locked list.
3. Prefer modifying your OWN steps. Only modify {other_id}'s steps if truly necessary.
4. If you accept {other_id}'s proposal unchanged, set reason="ACCEPT".
5. Each proposal MUST use one of these fields:
   - "time_min"     : shift timing (new_value = integer as string, e.g. "15")
   - "action"       : rewrite the action text (new_value = new action string)
   - "handoff_type" : add/change handoff ("PASS" | "INFORM" | "null")
   - "depends_on"   : set dependency list (new_value = JSON array string, e.g. "[3, 5]")
   - "delete"       : remove the step entirely (new_value = "true")
6. If no change is needed, return an empty proposals list.
7. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "proposals": [
    {{
      "step_id": 3,
      "agent_id": "{my_agent}",
      "field": "time_min",
      "new_value": "15",
      "reason": "TEMPORAL conflict with step {AGENT_B_STEP_OFFSET + 2} at T=10"
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


def _apply_proposal(plan: List[Dict], proposal: NegotiationProposal, locked_ids: Set[int]) -> bool:
    if proposal.step_id in locked_ids:
        return False

    sid_map = {s["step_id"]: i for i, s in enumerate(plan)}
    if proposal.step_id not in sid_map:
        return False

    idx = sid_map[proposal.step_id]

    if proposal.field == "delete":
        plan.pop(idx)
        return True

    if proposal.field == "time_min":
        new_t = safe_int(proposal.new_value, -1)
        if 0 <= new_t <= 30:
            plan[idx]["time_min"] = new_t
            return True

    elif proposal.field == "action":
        if proposal.new_value:
            plan[idx]["action"] = proposal.new_value
            return True

    elif proposal.field == "handoff_type":
        ht = _norm_handoff(proposal.new_value)
        plan[idx]["handoff_type"] = ht
        if ht is None:
            plan[idx]["target_agent"] = None
        return True

    elif proposal.field == "depends_on":
        try:
            deps = json.loads(proposal.new_value)
            if isinstance(deps, list):
                plan[idx]["depends_on"] = [int(d) for d in deps]
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
    """
    합의 기준:
    1. 한쪽이 제안하고 상대방이 ACCEPT -> lock
    2. 양쪽 모두 언급 없는 conflict step -> 이미 OK로 lock
    """
    accepted_by_b = {p.step_id for p in proposals_b if p.reason.upper() == "ACCEPT"}
    accepted_by_a = {p.step_id for p in proposals_a if p.reason.upper() == "ACCEPT"}
    proposed_by_a = {p.step_id for p in proposals_a if p.reason.upper() != "ACCEPT"}
    proposed_by_b = {p.step_id for p in proposals_b if p.reason.upper() != "ACCEPT"}

    mutually_agreed = (proposed_by_a & accepted_by_b) | (proposed_by_b & accepted_by_a)
    all_mentioned   = {p.step_id for p in proposals_a + proposals_b}
    uncontested     = conflict_step_ids - all_mentioned

    return existing_locked | mutually_agreed | uncontested


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
        return (
            plan_steps_to_dicts(plan_a.steps),
            plan_steps_to_dicts(plan_b.steps),
            [],
        )

    conflict_step_ids: Set[int] = {sid for c in conflicts for sid in c.step_ids}
    print(f"  Conflict step IDs: {sorted(conflict_step_ids)}")
    print(f"  Max rounds: {MAX_NEGOTIATION_ROUNDS}")

    cur_a: List[Dict] = plan_steps_to_dicts(plan_a.steps)
    cur_b: List[Dict] = plan_steps_to_dicts(plan_b.steps)
    locked: Set[int]  = set()
    rounds: List[NegotiationRound] = []

    prev_props_a: List[NegotiationProposal] = []
    prev_props_b: List[NegotiationProposal] = []

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
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_a = ex.submit(run_vlm, img_a, prompt_a, False)
            fut_b = ex.submit(run_vlm, img_b, prompt_b, False)
        raw_a, _ = fut_a.result()
        raw_b, _ = fut_b.result()
        props_a  = _parse_proposals(raw_a, "agent_A")
        props_b  = _parse_proposals(raw_b, "agent_B")

        if verbose == "full":
            _log(f"ROUND {rnd} AGENT A RAW", raw_a)
            _log(f"ROUND {rnd} AGENT B RAW", raw_b)

        if verbose in ("full", "summary"):
            for p in props_a:
                print(f"  [A->{p.agent_id}] step{p.step_id} {p.field}='{p.new_value}' ({p.reason})")
            for p in props_b:
                print(f"  [B->{p.agent_id}] step{p.step_id} {p.field}='{p.new_value}' ({p.reason})")

        for prop in props_a:
            target_plan = cur_a if prop.agent_id == "agent_A" else cur_b
            changed = _apply_proposal(target_plan, prop, locked)
            if changed and verbose in ("full", "summary"):
                print(f"  [APPLIED A] step{prop.step_id}.{prop.field}={prop.new_value}")

        for prop in props_b:
            target_plan = cur_a if prop.agent_id == "agent_A" else cur_b
            changed = _apply_proposal(target_plan, prop, locked)
            if changed and verbose in ("full", "summary"):
                print(f"  [APPLIED B] step{prop.step_id}.{prop.field}={prop.new_value}")

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
    print(f"  Final locked steps: {sorted(locked)}")
    return cur_a, cur_b, rounds


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 5: CONVERGENCE CHECK (rule-based)
# ──────────────────────────────────────────────────────────────────────────────

def _has_cycle(steps: List[Dict]) -> bool:
    """Kahn's algorithm으로 depends_on cycle 탐지."""
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

    # -- 조건 1: PASS 매칭 -------------------------------------------------------
    # 완화된 조건: PASS step_id가 다른 스텝의 depends_on에 있거나,
    # target_agent 플랜에 PASS 이후 시간의 스텝이 존재하면 OK로 간주
    pass_steps_map = {s["step_id"]: s for s in all_steps if s.get("handoff_type") == "PASS"}
    all_depends    = {dep for s in all_steps for dep in s.get("depends_on", [])}

    unmatched_pass = set()
    for sid, ps in pass_steps_map.items():
        target = ps.get("target_agent")
        # 1차: depends_on으로 직접 참조
        if sid in all_depends:
            continue
        # 2차 완화: target 플랜에 PASS 이후 시간의 스텝이 있으면 수렴으로 간주
        # (Finalize에서 auto_add_pass_receivers가 receiver를 추가하므로)
        target_has_later_step = any(
            s.get("agent_id") == target and s.get("time_min", 0) >= ps.get("time_min", 0)
            and s.get("step_id") != sid
            for s in all_steps
        )
        if not target_has_later_step:
            unmatched_pass.add(sid)

    pass_matched = len(unmatched_pass) == 0
    pass_step_ids = set(pass_steps_map.keys())

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



# ── Phase 6 Human Query Template ─────────────────────────────────────────────
_HQ_TEMPLATES: Dict[str, str] = {
    "UNMATCHED_PASS": (
        "A PASS handoff has no matching receiver step. "
        "Should the receiving agent add a step to pick up the item at the room boundary?"
    ),
    "DEP_CYCLE": (
        "A circular dependency was detected in the plan. "
        "Which step should be reordered to break the cycle?"
    ),
    "DEPENDENCY": (
        "An agent can provide an item the other agent needs, but no PASS step exists. "
        "Should the providing agent carry the item to the room boundary?"
    ),
    "REDUNDANCY": (
        "Two agents are attempting the same task. "
        "Which agent should handle it, and should the other skip it?"
    ),
    "CANNOT_DO": (
        "An agent is attempting something outside its capability. "
        "Should this step be removed or reassigned to the other agent?"
    ),
    "UNMATCHED_NEED": (
        "An agent needs something that neither agent can currently provide. "
        "How should this unmet need be handled?"
    ),
    "OBSERVABILITY": (
        "A step references objects outside the agent's visible scope. "
        "Should this step be modified or removed?"
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 6: DEFERRED HUMAN QUERY
# ──────────────────────────────────────────────────────────────────────────────

def _generate_hq_question(
    trigger_type: str,
    detail: str,
    offer_a: Offer,
    offer_b: Offer,
    img: str,
) -> str:
    """Template 기반 질문 생성 — VLM 호출 없음 (비용 절감)."""
    template = _HQ_TEMPLATES.get(trigger_type, "How should the agents handle this issue?")
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
        print("  Plan converged with no residual issues -> no human query needed.")
        return {}, [], []

    triggered: List[str] = []
    raw_triggers: List[Tuple[str, str, float]] = []

    if not convergence.pass_matched:
        detail = "Some PASS handoffs have no matching receiver step after negotiation"
        triggered.append(f"[HQ1] {detail}")
        raw_triggers.append(("UNMATCHED_PASS", detail, 0.90))

    if not convergence.no_dep_cycle:
        detail = "Dependency cycle detected in the joint plan after negotiation"
        triggered.append(f"[HQ2] {detail}")
        raw_triggers.append(("DEP_CYCLE", detail, 0.85))

    for c in convergence.unresolved_conflicts:
        triggered.append(f"[HQ3-{c.conflict_type}] {c.description}")
        raw_triggers.append((c.conflict_type, c.description, 0.80))

    all_provides = offer_a.can_provide + offer_b.can_provide
    for need in offer_a.need_from_other + offer_b.need_from_other:
        if not any(_fuzzy_match_soft(need, p) for p in all_provides):
            detail = f"No agent can provide: '{need}'"
            triggered.append(f"[HQ4] Unmatched need: '{need}'")
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
        question = _generate_hq_question(ttype, detail, offer_a, offer_b, img_a)
        print("done")
        print(f"  Q{i}: {question}")
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
# FINALIZE: RULE-BASED MERGE (LLM 호출 없음)
# ──────────────────────────────────────────────────────────────────────────────

def _auto_add_pass_receivers(
    steps: List[Dict],
    offer_a: Offer,
    offer_b: Offer,
) -> List[Dict]:
    """
    PASS step에 대응하는 receiver step이 없으면 자동으로 추가한다.

    FIX: 이 함수는 반드시 step_id 재번호 이전에 호출해야 한다.
         original step_id를 기준으로 pass_received를 계산한다.
    """
    room_map = {"agent_A": offer_a.room_type, "agent_B": offer_b.room_type}

    pass_steps = {s["step_id"]: s for s in steps if s.get("handoff_type") == "PASS"}

    # receiver 존재 여부: depends_on에서 PASS step_id를 참조하는 스텝이 있는지
    pass_received = {
        dep
        for s in steps
        for dep in s.get("depends_on", [])
        if dep in pass_steps
    }

    additions = []
    max_id = max((s["step_id"] for s in steps), default=0)

    for sid, pass_s in pass_steps.items():
        if sid in pass_received:
            continue
        target = pass_s.get("target_agent")
        if not target or target not in room_map:
            continue

        max_id += 1
        receiver_time = min(30, pass_s["time_min"] + 1)
        # receiver action: PASS step에서 아이템 이름 추출해서 자연스럽게 생성
        pass_action = pass_s["action"]
        # "carry X to Y" 패턴에서 X 추출
        import re as _re
        carry_m = _re.search(r"carry (.+?) to ", pass_action, _re.IGNORECASE)
        item_name = carry_m.group(1) if carry_m else "item"
        sender_id = pass_s.get("agent_id", "other agent")
        receiver_action = f"receive {item_name} from {sender_id} and place in {room_map[target]}"

        additions.append({
            "step_id":       max_id,
            "time_min":      receiver_time,
            "room":          room_map[target],
            "agent_id":      target,
            "action":        receiver_action,
            "preconditions": [f"step {sid} completed"],
            "depends_on":    [sid],  # original PASS step_id (재번호 전)
            "handoff_type":  None,
            "target_agent":  None,
            "notes":         "auto-added PASS receiver",
        })
        print(f"  [AUTO-PASS] receiver step {max_id} added for {target} <- PASS step {sid} (T={receiver_time}m)")

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
    """
    협상된 두 플랜을 단일 Joint Plan으로 확정한다.
    LLM 호출 없이 rule-based merge만 수행.

    FIX - 올바른 순서:
      1. 두 플랜 합산
      2. PASS receiver 자동 보완  <- 재번호 이전 (original step_id 기준)
      3. time_min 순 정렬
      4. step_id 재번호 (1부터 순차)
      5. depends_on / preconditions 재매핑
    """
    _banner("FINALIZE - RULE-BASED JOINT PLAN MERGE")

    if human_answers:
        print("  Human query answers incorporated:")
        for q, a in human_answers.items():
            print(f"    Q: {q[:60]}...")
            print(f"    A: {a}")

    # 1. 합산
    merged = list(steps_a) + list(steps_b)

    # 2. PASS receiver 자동 보완 (재번호 이전 - FIX)
    merged = _auto_add_pass_receivers(merged, offer_a, offer_b)

    # 3. time_min 순 정렬
    merged.sort(key=lambda x: (x.get("time_min", 0), x.get("step_id", 0)))

    # 4-5. step_id 재번호 및 의존성 재매핑
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
