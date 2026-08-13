# p2p_phase_nagent.py
#
# N-agent 확장 파이프라인. 기존 p2p_phase.py(2-agent KCC 코드)를 최대한 그대로
# import해서 재사용하고, agent 개수/식별자에 의존하는 부분만 새로 구현한다.
#
# 설계 원칙 (지도교수님 방향: "P2P 유지, 방법론 변경 없이 N=4로"):
#   1. Offer 구조, PASS/INFORM 규칙, negotiation 프로토콜(최대 3라운드, step lock),
#      conflict 판정 규칙(TEMPORAL/DEPENDENCY/REDUNDANCY/CANNOT_DO/OBSERVABILITY) —
#      전부 p2p_phase.py의 원본 함수를 그대로 호출해서 재사용한다. 새로 만들지 않는다.
#   2. Conflict Detection은 "중앙 실행 주체"가 아니라, 이미 서로의 plan을 교환받은
#      각 agent 쌍이 동일한 규칙을 로컬에서 대칭적으로 계산하는 것으로 재해석한다
#      (phase3_conflict_detection_n 참고).
#   3. Negotiation은 전체 N명이 한 테이블에 앉는 게 아니라, conflict가 실제로
#      존재하는 쌍끼리만 원본 2자간 P2P negotiation 루프를 그대로 재사용해 수행한다.

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor as _TPE
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

from p2p_config import HQ_TOP_K, MAX_NEGOTIATION_ROUNDS
from p2p_config_nagent import make_agent_ids, step_offset
from p2p_models import (
    ConflictEntry, ConflictType, ConvergenceResult, Handoff, HQEntry,
    LocalPlan, NegotiationProposal, NegotiationRound, Offer, PlanStep,
)
from p2p_utils import (
    _banner, _fuzzy_match, _fuzzy_match_soft, _log, _norm_depends,
    _norm_handoff, clamp01, compute_plan_uncertainty, compute_token_uncertainty,
    extract_json, jdump, safe_int,
)
from p2p_phases import _kw  # stemming 포함된 keyword 추출기 — p2p_phases.py에만 정의됨
from p2p_utils_nagent import make_norm_agent
from p2p_vlm import run_vlm

# 원본 2-agent 모듈 — 알고리즘을 바꾸지 않고 그대로 재사용하기 위해 통째로 import
import p2p_phases as _p2


# ══════════════════════════════════════════════════════════════════════════
# PHASE 1: OFFER GENERATION (N-agent broadcast)
# ══════════════════════════════════════════════════════════════════════════

def phase1_offer_n(
    images: List[str], agent_ids: List[str], task: str, verbose: str = "full",
) -> Dict[str, Offer]:
    """images: agent 수만큼의 이미지 경로 리스트. 각 agent는 자기 이미지(=자기 구역)만 본다."""
    _banner(f"PHASE 1 — OFFER GENERATION (N={len(agent_ids)})")
    if len(images) != len(agent_ids):
        raise ValueError(f"images 개수({len(images)})가 agent 수({len(agent_ids)})와 일치해야 합니다.")

    prompt = _p2._build_phase1_prompt(task)  # 원본 프롬프트 그대로 재사용

    with _TPE(max_workers=len(agent_ids)) as ex:
        futs = [ex.submit(_p2._vlm_with_retry, img, prompt, False) for img in images]
        raws = [f.result()[0] for f in futs]

    offers: Dict[str, Offer] = {}
    for aid, raw in zip(agent_ids, raws):
        if verbose == "full":
            _log(f"{aid} RAW OFFER", raw)
        offers[aid] = _p2._parse_offer(raw, aid)  # 원본 파서 그대로 재사용

    for aid in agent_ids:
        o = offers[aid]
        print(f"  {aid}: room={o.room_type} | can_do={len(o.can_do)} "
              f"| provide={len(o.can_provide)} | need={len(o.need_from_other)}")
    return offers


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2: LOCAL PLANNING (peer-aware, N-agent)
# ══════════════════════════════════════════════════════════════════════════

def _peer_summary(offer: Offer) -> Dict:
    """전체 draft plan이 아니라 offer 요약만 넘김 — N=4에서 프롬프트 폭발 방지."""
    return {
        "agent_id": offer.agent_id,
        "room_type": offer.room_type,
        "can_provide": offer.can_provide,
        "need_from_other": offer.need_from_other,
    }


def _build_local_plan_prompt_n(my: Offer, others: List[Offer], task: str) -> str:
    peers = json.dumps([_peer_summary(o) for o in others], ensure_ascii=False, indent=2)
    other_ids = [o.agent_id for o in others]
    return f"""You are the {my.room_type} agent ({my.agent_id}).
Global task: "{task}"

There are {len(others) + 1} agents total, each in a separate room, coordinating this task together.

YOUR OFFER:
- can_do: {json.dumps(my.can_do, ensure_ascii=False)}
- can_provide (items you can PASS to others): {json.dumps(my.can_provide, ensure_ascii=False)}
- need_from_other: {json.dumps(my.need_from_other, ensure_ascii=False)}

OTHER AGENTS (summary only — decide who to PASS to / request from):
{peers}

{_p2._P2_HANDOFF_RULES}

Generate YOUR local plan. Think step by step:
1. What does the global task require from YOUR room specifically?
2. Looking at the other agents' can_provide/need_from_other above, is there something you
   should PASS to a SPECIFIC agent, or expect to RECEIVE from a specific agent?
3. If you PASS or expect to receive, target_agent MUST be exactly one of: {other_ids}

PLANNING RULES:
1. Steps ONLY in your room ({my.room_type}), using ONLY visible objects.
2. Generate 4-6 steps over 0-25 minutes. NO repeated actions.
3. Prioritize actions that DIRECTLY contribute to the global task.
4. HANDOFF - if can_provide is NOT empty and a specific other agent needs it:
   - Only do this if the item is genuinely relevant to something in that agent's
     need_from_other — don't hand off an item just because you happen to have one spare.
   - Prepare the item first (1-2 prep steps)
   - Then add ONE PASS step: "carry [item] to [room] doorway for [target_agent] pickup"
   - target_agent MUST name the specific agent_id who needs it (one of {other_ids}), not "other".
   - PASS step must have depends_on=[prep step ids] (your OWN step_ids only)
5. INFORM - if you want to notify a specific agent of completion:
   - "notify [target_agent]: [what is ready]"
   - handoff_type="INFORM", target_agent=[one of {other_ids}]
6. depends_on must ONLY reference YOUR OWN step_ids. Never another agent's step_ids.
7. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "plan_steps": [
    {{"step_id":1,"time_min":0,"action":"verb + specific object",
      "preconditions":[],"depends_on":[],"handoff_type":null,
      "target_agent":null,"uncertainty":0.1,"notes":""}}
  ]
}}
</JSON>"""


def _normalize_pass_n(steps: List[PlanStep], agent_ids: List[str]) -> List[PlanStep]:
    """원본 _normalize_pass()와 동일한 규칙, target_agent 유효성만 N-agent 리스트로 일반화."""
    my_ids = {s.step_id for s in steps}
    seen_pass: List[PlanStep] = []
    _CARRY = {"carry", "bring", "deliver", "transport", "move", "transfer"}
    _RECV = {"place", "set", "organize", "receive", "pick", "get", "put", "sort"}

    for s in steps:
        if s.handoff_type != "PASS":
            continue
        first = s.action.lower().split()[0] if s.action.strip() else ""
        if first not in _CARRY:
            s.handoff_type = None; s.target_agent = None; continue
        if not s.target_agent or s.target_agent not in agent_ids:
            s.handoff_type = None; s.target_agent = None; continue
        if first in _RECV:
            s.handoff_type = None; s.target_agent = None; continue
        if not s.depends_on:
            prev_steps = [p for p in steps if p.step_id < s.step_id and not p.handoff_type]
            if prev_steps:
                s.depends_on = [max(prev_steps, key=lambda p: p.step_id).step_id]
            else:
                s.handoff_type = None; s.target_agent = None; continue
        if not [d for d in s.depends_on if d in my_ids]:
            s.handoff_type = None; s.target_agent = None; continue
        s.depends_on = [d for d in s.depends_on if d in my_ids]

        def _ppkw(action: str) -> set:
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
            s.handoff_type = None; s.target_agent = None; continue
        seen_pass.append(s)
    return steps


def _parse_local_plan_n(
    raw: str, log_probs: List[float], my: Offer,
    agent_index: int, agent_ids: List[str], norm_agent,
) -> LocalPlan:
    data = extract_json(raw)
    if isinstance(data, list):
        data = {"plan_steps": data}
    if not isinstance(data, dict):
        data = {}
    raw_steps = data.get("plan_steps", [])
    if not isinstance(raw_steps, list):
        raw_steps = []

    offset = step_offset(agent_index)
    token_unc = compute_token_uncertainty(log_probs)
    steps: List[PlanStep] = []
    hq_list: List[HQEntry] = []
    seen_ids: Set[int] = set()
    seen_act: Set[frozenset] = set()
    pending_deps: Dict[int, List[int]] = {}

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

        sid = raw_sid + offset
        while sid in seen_ids:
            sid += 1
        seen_ids.add(sid)

        json_unc = clamp01(item.get("uncertainty", 0.2))
        action_conf = max(
            (v for k, v in my.conf.items() if _fuzzy_match_soft(action, k)),
            default=0.7,
        )
        step_unc = clamp01(json_unc * 0.5 + token_unc * 0.2 + (1 - action_conf) * 0.3)

        raw_deps = _norm_depends(item.get("depends_on"))
        pending_deps[sid] = [d + offset for d in raw_deps]

        handoff = _norm_handoff(item.get("handoff_type")) if item.get("handoff_type") else None
        target = norm_agent(item.get("target_agent"))

        # target_agent 필드와 action 텍스트("for {agent} pickup" / "notify {agent}:")가
        # 서로 다른 agent를 가리키는 경우가 있다 — VLM이 문장은 맞게 쓰고 구조화된
        # target_agent 필드는 잘못 채우는 경우. 텍스트 쪽이 우리가 지정한 프롬프트
        # 포맷을 그대로 따르므로 더 신뢰할 수 있어, 불일치 시 텍스트를 우선한다.
        text_target = None
        m = re.search(r"for\s+(agent_[A-Za-z]+)\s+pickup", action, re.IGNORECASE)
        if not m:
            m = re.search(r"notify\s+(agent_[A-Za-z]+)\s*:", action, re.IGNORECASE)
        if m:
            text_target = norm_agent(m.group(1))
        if text_target and text_target != my.agent_id and text_target != target:
            print(f"  [WARN] {my.agent_id} step {i}: target_agent field said "
                  f"'{item.get('target_agent')}' but action text says '{text_target}' -> using text")
            target = text_target

        first_word = action.lower().split()[0] if action.strip() else ""
        if handoff == "INFORM" and first_word in {"carry", "bring", "deliver", "transport"}:
            handoff = "PASS"
        if target == my.agent_id:  # 자기 자신을 target으로 잘못 지정한 경우 제거
            target, handoff = None, None

        steps.append(PlanStep(
            step_id=sid, time_min=max(0, min(30, raw_time)), room=my.room_type,
            agent_id=my.agent_id, action=action,
            preconditions=[str(x).strip() for x in item.get("preconditions", []) if str(x).strip()],
            depends_on=[], handoff_type=handoff, target_agent=target,
            uncertainty=step_unc, notes=str(item.get("notes", "")).strip(),
        ))
        if step_unc >= 0.50:
            hq_list.append(HQEntry(sid, f"Is '{action}' feasible?", step_unc))

    final_ids = {s.step_id for s in steps}
    for s in steps:
        s.depends_on = sorted(d for d in pending_deps.get(s.step_id, []) if d in final_ids)

    steps.sort(key=lambda s: (s.time_min, s.step_id))
    steps = _normalize_pass_n(steps, agent_ids)

    handoffs = [
        Handoff(s.step_id, s.action, s.handoff_type, s.target_agent,
                s.notes if s.handoff_type == "INFORM" else "", my.agent_id)
        for s in steps if s.handoff_type
    ]
    all_unc = [s.uncertainty for s in steps] if steps else [token_unc]
    return LocalPlan(my.agent_id, steps, compute_plan_uncertainty(all_unc), hq_list, handoffs)


def _ensure_pass_n(
    plans: Dict[str, LocalPlan], offers: Dict[str, Offer], agent_ids: List[str],
) -> Dict[str, LocalPlan]:
    """원본 _ensure_pass()의 sender-receiver 1:1 매칭 로직을 N명 전체 쌍으로 확장한
    rule-based fallback. VLM이 놓친 PASS/receive를 keyword 매칭으로 보정한다."""
    _banner("PHASE 2b - HANDOFF SYNC (rule-based, N-agent)")

    _LINK_RECV_VERBS = {
        "receive", "accept", "collect", "get", "take", "pick",
        "place", "put", "set", "arrange", "bring", "use", "set up",
    }

    def _is_recv_candidate(s: PlanStep) -> bool:
        first = s.action.lower().split()[0] if s.action.strip() else ""
        return first in _LINK_RECV_VERBS

    # 원본 FUZZY_STOPWORDS(p2p_config.py)에는 "of", "from", "into" 같은 전치사가
    # 빠져있어서, 이 함수의 keyword-overlap 매칭에서 "of" 한 단어만 겹쳐도
    # 매칭된 걸로 착각하는 문제가 있었다 (예: coffee PASS가 엉뚱하게 sandwich
    # receive step에 흡수됨). 매칭 전용으로 전치사류를 추가로 걸러낸다.
    _EXTRA_STOPWORDS_N = {
        "of", "from", "into", "onto", "by", "as", "for", "this", "that",
        "these", "those", "then", "than", "also", "your", "their", "its",
    }

    def _mkw(text: str) -> Set[str]:
        return _kw(text) - _EXTRA_STOPWORDS_N

    for sender in agent_ids:
        s_offer = offers[sender]
        passable = [p for p in s_offer.can_provide if _p2._is_passable(p)]
        if not passable:
            continue
        already_targets = {s.target_agent for s in plans[sender].steps if s.handoff_type == "PASS"}

        for receiver in agent_ids:
            if receiver == sender or receiver in already_targets:
                continue
            r_offer = offers[receiver]

            matched_item = None
            for item in passable:
                ikw = _mkw(item)
                if any(ikw & _mkw(need) for need in r_offer.need_from_other):
                    matched_item = item
                    break
            if not matched_item:
                continue

            sender_plan, receiver_plan = plans[sender], plans[receiver]
            pkw = _mkw(matched_item)
            prep = [s for s in sender_plan.steps if pkw & _mkw(s.action) and not s.handoff_type]
            if not prep:
                prep = [s for s in sender_plan.steps if not s.handoff_type]
            if not prep:
                continue
            last_prep = max(prep, key=lambda s: s.time_min)

            all_ids = {s.step_id for p in plans.values() for s in p.steps}
            new_sid = max(all_ids, default=0) + 1
            while new_sid in all_ids:
                new_sid += 1

            pass_step = PlanStep(
                step_id=new_sid, time_min=min(30, last_prep.time_min + 5),
                room=s_offer.room_type, agent_id=sender,
                action=f"carry {matched_item} to {s_offer.room_type} doorway for {receiver} pickup",
                preconditions=[f"step {last_prep.step_id} completed"],
                depends_on=[last_prep.step_id], handoff_type="PASS", target_agent=receiver,
                uncertainty=0.15, notes=f"{matched_item} ready at doorway",
            )
            sender_plan.steps.append(pass_step)
            sender_plan.steps.sort(key=lambda s: (s.time_min, s.step_id))
            sender_plan.handoffs.append(Handoff(new_sid, pass_step.action, "PASS", receiver, "", sender))
            print(f"  [ENSURE] {sender}->{receiver}: PASS step{new_sid} injected '{matched_item}'")

            targets = [
                s for s in receiver_plan.steps
                if not s.handoff_type and pkw & _mkw(s.action) and _is_recv_candidate(s)
            ]
            if targets:
                for rs in targets:
                    if new_sid not in rs.depends_on:
                        rs.depends_on = sorted(rs.depends_on + [new_sid])
                    if rs.time_min <= pass_step.time_min:
                        rs.time_min = pass_step.time_min + 1
                    print(f"  [ENSURE] {receiver} step{rs.step_id} <- PASS step{new_sid}")
            else:
                recv_ids = {s.step_id for s in receiver_plan.steps}
                recv_sid = max(recv_ids, default=0) + 1
                while recv_sid in recv_ids:
                    recv_sid += 1
                recv_step = PlanStep(
                    step_id=recv_sid, time_min=min(30, pass_step.time_min + 1),
                    room=r_offer.room_type, agent_id=receiver,
                    action=f"receive {matched_item} from {s_offer.room_type} and bring into room",
                    preconditions=[f"step {new_sid} completed"], depends_on=[new_sid],
                    handoff_type=None, target_agent=None, uncertainty=0.15,
                    notes="auto-added receive step",
                )
                receiver_plan.steps.append(recv_step)
                receiver_plan.steps.sort(key=lambda s: (s.time_min, s.step_id))
                print(f"  [ENSURE] {receiver}: receive step{recv_sid} auto-added")

            already_targets.add(receiver)

    # ── 추가 검증: VLM이 스스로 만든 PASS도 받는 쪽에 receive step이 있는지 확인 ──
    # 위 루프는 "이 함수가 새로 주입하는 PASS"에만 receive를 짝지어줬다. 그런데 VLM이
    # 자기 plan에 이미 PASS를 만들어놓고, 정작 상대방 plan에는 그걸 받는 step이
    # 없는 경우(= convergence check에서 "PASS matched: FAIL"로 잡히는 원인)가
    # 생길 수 있다. 이 경우도 동일한 규칙으로 보정한다.
    for sender in agent_ids:
        for s in list(plans[sender].steps):
            if s.handoff_type != "PASS" or not s.target_agent:
                continue
            receiver = s.target_agent
            if receiver not in plans:
                continue
            receiver_plan = plans[receiver]
            already_linked = any(s.step_id in rs.depends_on for rs in receiver_plan.steps)
            if already_linked:
                continue

            pkw = _mkw(s.action)
            targets = [
                rs for rs in receiver_plan.steps
                if not rs.handoff_type and pkw & _mkw(rs.action) and _is_recv_candidate(rs)
            ]
            if targets:
                for rs in targets:
                    if s.step_id not in rs.depends_on:
                        rs.depends_on = sorted(rs.depends_on + [s.step_id])
                    if rs.time_min <= s.time_min:
                        rs.time_min = s.time_min + 1
                print(f"  [ENSURE] {receiver} step linked to existing PASS step{s.step_id} from {sender}")
            else:
                r_offer = offers[receiver]
                recv_ids = {rs.step_id for rs in receiver_plan.steps}
                recv_sid = max(recv_ids, default=0) + 1
                while recv_sid in recv_ids:
                    recv_sid += 1
                item_guess = s.action.split(" to ")[0].replace("carry ", "").strip() or "item"
                recv_step = PlanStep(
                    step_id=recv_sid, time_min=min(30, s.time_min + 1),
                    room=r_offer.room_type, agent_id=receiver,
                    action=f"receive {item_guess} from {sender} and bring into room",
                    preconditions=[f"step {s.step_id} completed"], depends_on=[s.step_id],
                    handoff_type=None, target_agent=None, uncertainty=0.15,
                    notes="auto-added receive step (existing VLM-authored PASS)",
                )
                receiver_plan.steps.append(recv_step)
                receiver_plan.steps.sort(key=lambda rs: (rs.time_min, rs.step_id))
                print(f"  [ENSURE] {receiver}: receive step{recv_sid} auto-added "
                      f"for existing PASS step{s.step_id} from {sender}")

    return plans


def phase2_local_plan_n(
    offers: Dict[str, Offer], images: List[str], agent_ids: List[str],
    task: str, verbose: str = "full",
) -> Dict[str, LocalPlan]:
    """images: Phase 1과 동일한 agent별 이미지 리스트."""
    _banner(f"PHASE 2 - LOCAL PLANNING (N={len(agent_ids)})")

    room_to_agent = {
        offers[aid].room_type.lower(): aid for aid in agent_ids if offers[aid].room_type
    }
    norm_agent = make_norm_agent(agent_ids, room_to_agent)

    prompts = []
    for aid in agent_ids:
        others = [offers[o] for o in agent_ids if o != aid]
        prompts.append(_build_local_plan_prompt_n(offers[aid], others, task))

    with _TPE(max_workers=len(agent_ids)) as ex:
        futs = [ex.submit(_p2._vlm_with_retry, img, p, True) for img, p in zip(images, prompts)]
        results = [f.result() for f in futs]

    plans: Dict[str, LocalPlan] = {}
    for idx, aid in enumerate(agent_ids):
        raw, logp = results[idx]
        if verbose == "full":
            _log(f"{aid} RAW PLAN", raw)
        plans[aid] = _parse_local_plan_n(raw, logp, offers[aid], idx, agent_ids, norm_agent)

    plans = _ensure_pass_n(plans, offers, agent_ids)

    for aid in agent_ids:
        lp = plans[aid]
        n_pass = sum(1 for s in lp.steps if s.handoff_type == "PASS")
        print(f"  {aid}: steps={len(lp.steps)} U={lp.U_plan:.3f} PASS={n_pass}")
    return plans


# ══════════════════════════════════════════════════════════════════════════
# PHASE 3: DECENTRALIZED CONFLICT DETECTION
# ══════════════════════════════════════════════════════════════════════════
#
# detect_conflicts()는 순수 결정론적 규칙 함수(VLM 호출 없음)이므로, 두 agent가
# 각자 로컬에서 동일한 입력(자기 plan + 이미 교환받은 상대 plan)에 대해 이 함수를
# 돌리면 항상 같은 결과가 나온다. 즉 "중앙 조율자가 대신 계산"하는 게 아니라
# "당사자 두 agent가 각자 계산해도 결과가 자동으로 일치"하는 구조로 해석할 수 있다.
# 원본 함수는 조금도 수정하지 않고, 결과 라벨(agent_A/agent_B로 하드코딩된 부분)만
# 실제 agent_id로 되돌려 붙여주는 얇은 wrapper만 추가한다.

def _remap_conflict_labels(conflicts: List[ConflictEntry], real_a: str, real_b: str) -> List[ConflictEntry]:
    for c in conflicts:
        c.agent_ids = [
            real_a if x == "agent_A" else (real_b if x == "agent_B" else x)
            for x in c.agent_ids
        ]
        c.description = c.description.replace("agent_A", real_a).replace("agent_B", real_b)
    return conflicts


def phase3_conflict_detection_n(
    plans: Dict[str, LocalPlan], offers: Dict[str, Offer], agent_ids: List[str], verbose: str = "full",
) -> Tuple[List[ConflictEntry], Dict[Tuple[str, str], List[ConflictEntry]]]:
    _banner(f"PHASE 3 - CONFLICT DETECTION (decentralized, {len(agent_ids)} agents)")

    all_conflicts: List[ConflictEntry] = []
    by_pair: Dict[Tuple[str, str], List[ConflictEntry]] = {}

    for a, b in combinations(agent_ids, 2):
        pair_conflicts = _p2.detect_conflicts(plans[a], plans[b], offers[a], offers[b])
        pair_conflicts = _remap_conflict_labels(pair_conflicts, a, b)
        if pair_conflicts:
            by_pair[(a, b)] = pair_conflicts
            all_conflicts.extend(pair_conflicts)
            print(f"  [{a} <-> {b}] {len(pair_conflicts)} conflict(s)")
        elif verbose == "full":
            print(f"  [{a} <-> {b}] no conflicts")

    n_total_pairs = len(agent_ids) * (len(agent_ids) - 1) // 2
    print(f"\n  Total conflicts: {len(all_conflicts)} across "
          f"{len(by_pair)}/{n_total_pairs} pair(s)")
    return all_conflicts, by_pair


# ══════════════════════════════════════════════════════════════════════════
# PHASE 4: P2P NEGOTIATION (conflict-pair scoped)
# ══════════════════════════════════════════════════════════════════════════
#
# N명 전체를 한 협상 테이블에 앉히지 않는다. 원본 phase4_negotiation의 2자간
# 프로토콜(최대 3라운드, step lock, ACCEPT 기반 합의)을 그대로, conflict가
# 실제로 존재하는 쌍에 대해서만 순차 실행한다.

_VALID_PROPOSAL_FIELDS = {"time_min", "action", "depends_on", "delete"}


def _build_negotiation_prompt_n(
    my_agent: str, other_agent: str, my_offer: Offer,
    cur_x: List[Dict], cur_y: List[Dict],
    conflicts: List[ConflictEntry], locked: Set[int], round_num: int,
    prev_props: List[NegotiationProposal], task: str,
) -> str:
    details: List[str] = []
    for c in conflicts:
        ct = str(c.conflict_type)
        sids = c.step_ids
        if "DEPENDENCY" in ct:
            pass_steps = [s for s in (cur_x + cur_y) if s["step_id"] in sids and s.get("handoff_type") == "PASS"]
            if pass_steps:
                ps = pass_steps[0]
                recv_agent = other_agent if ps.get("agent_id") == my_agent else my_agent
                details.append(
                    f"[DEPENDENCY] step{ps['step_id']} ({ps.get('agent_id')}) PASS action: '{ps['action'][:50]}'\n"
                    f"  -> {recv_agent} MUST add a receive step with depends_on=[{ps['step_id']}].\n"
                    f"  -> ONLY modify the RECEIVE step in {recv_agent}'s plan, not other steps."
                )
            else:
                details.append(f"[DEPENDENCY] {c.description}\n  -> {c.fix_hint}")
        elif "REDUNDANCY" in ct:
            dup = [s for s in (cur_x + cur_y) if s["step_id"] in sids]
            if len(dup) >= 2:
                details.append(
                    "[REDUNDANCY] Two agents planned nearly identical actions:\n"
                    f"  step{dup[0]['step_id']} ({dup[0].get('agent_id')}): '{dup[0]['action'][:50]}'\n"
                    f"  step{dup[1]['step_id']} ({dup[1].get('agent_id')}): '{dup[1]['action'][:50]}'\n"
                    "  -> DELETE one of these TWO steps. Use field='delete'."
                )
            else:
                details.append(f"[REDUNDANCY] {c.description}\n  -> {c.fix_hint}")
        elif "TEMPORAL" in ct:
            t_steps = [s for s in (cur_x + cur_y) if s["step_id"] in sids]
            details.append(
                "[TEMPORAL] Time conflict at same time slot:\n"
                + "\n".join(
                    f"  step{s['step_id']} ({s.get('agent_id')}): '{s['action'][:50]}' at time_min={s.get('time_min')}"
                    for s in t_steps
                )
                + "\n  -> Shift ONE step's time_min to a different value. Use field='time_min'."
            )
        elif "CANNOT" in ct:
            bad = [s for s in (cur_x + cur_y) if s["step_id"] in sids]
            if bad:
                bs = bad[0]
                details.append(
                    f"[CANNOT_DO] step{bs['step_id']} ({bs.get('agent_id')}): '{bs['action'][:50]}' - cannot do this.\n"
                    f"  -> DELETE step{bs['step_id']} using field='delete'."
                )
        else:
            details.append(f"[{ct}] {c.description}\n  -> {c.fix_hint}")

    conflict_block = "\n\n".join(details) or "  (none)"
    prev_text = "\n".join(
        f"  step{p.step_id}[{p.agent_id}] .{p.field}='{p.new_value}' ({p.reason})" for p in prev_props
    ) or "  (none)"

    return f"""You are {my_agent} ({my_offer.room_type}). ROUND {round_num}/{MAX_NEGOTIATION_ROUNDS}.
Task: "{task}"
NOTE: You are negotiating one-on-one (P2P) with {other_agent} to resolve conflicts between
your two plans. Other agents in the team are not part of this negotiation.

YOUR PLAN (step_ids you can propose changes for):
{jdump(cur_x)}

{other_agent}'s PLAN:
{jdump(cur_y)}

=== CONFLICTS TO RESOLVE (between you and {other_agent}) ===
{conflict_block}

LOCKED step_ids (DO NOT touch): {sorted(locked) or '(none)'}
{other_agent}'s previous proposals: {prev_text}

=== HOW TO RESPOND ===
- Make ONE proposal per conflict.
- ONLY modify the exact step_ids mentioned in the conflict description above.
- DO NOT modify unrelated steps.
- DEPENDENCY -> field="depends_on", new_value="[PASS_STEP_ID]"
- REDUNDANCY -> field="delete", new_value="true" for the duplicate step
- TEMPORAL   -> field="time_min", new_value="NEW_TIME"
- CANNOT_DO  -> field="delete", new_value="true"
- If you AGREE with {other_agent}'s proposal, echo it with reason="ACCEPT".
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


def _parse_proposals_n(raw: str, my_agent: str, valid_pair: Set[str]) -> List[NegotiationProposal]:
    data = extract_json(raw)
    if isinstance(data, list):
        data = {"proposals": data}
    if not isinstance(data, dict):
        return []
    result = []
    for item in data.get("proposals", []):
        if not isinstance(item, dict):
            continue
        sid = safe_int(item.get("step_id", -1), -1)
        agent_id = str(item.get("agent_id", my_agent)).strip()
        field = str(item.get("field", "")).strip().lower()
        new_val = str(item.get("new_value", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if sid < 0 or field not in _VALID_PROPOSAL_FIELDS or not new_val:
            continue
        if agent_id not in valid_pair:
            agent_id = my_agent
        result.append(NegotiationProposal(sid, agent_id, field, new_val, reason))
    return result


def _apply_proposal_n(
    cur_x: List[Dict], cur_y: List[Dict], agent_x: str,
    prop: NegotiationProposal, locked: Set[int],
) -> bool:
    if prop.step_id in locked:
        return False
    plan = cur_x if prop.agent_id == agent_x else cur_y
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


def _lock_steps_n(
    props_x: List[NegotiationProposal], props_y: List[NegotiationProposal], existing: Set[int],
) -> Set[int]:
    acc_y = {p.step_id for p in props_y if "ACCEPT" in p.reason.upper()}
    acc_x = {p.step_id for p in props_x if "ACCEPT" in p.reason.upper()}
    prop_x = {p.step_id for p in props_x if "ACCEPT" not in p.reason.upper()}
    prop_y = {p.step_id for p in props_y if "ACCEPT" not in p.reason.upper()}
    agreed = (prop_x & acc_y) | (prop_y & acc_x)
    uncontested_x = prop_x - prop_y
    uncontested_y = prop_y - prop_x
    return existing | agreed | uncontested_x | uncontested_y


def _negotiate_pair(
    agent_x: str, agent_y: str,
    plans: Dict[str, LocalPlan], offers: Dict[str, Offer], images_map: Dict[str, str],
    conflicts: List[ConflictEntry], task: str, verbose: str = "full",
) -> Tuple[List[Dict], List[Dict], List[NegotiationRound]]:
    cur_x = _p2.plan_steps_to_dicts(plans[agent_x].steps)
    cur_y = _p2.plan_steps_to_dicts(plans[agent_y].steps)
    valid_pair = {agent_x, agent_y}

    locked: Set[int] = set()
    rounds: List[NegotiationRound] = []
    prev_x: List[NegotiationProposal] = []
    prev_y: List[NegotiationProposal] = []
    last_val: Dict[Tuple[int, str], str] = {}

    for rnd in range(1, MAX_NEGOTIATION_ROUNDS + 1):
        lp_x = _p2._dicts_to_localplan(cur_x, offers[agent_x])
        lp_y = _p2._dicts_to_localplan(cur_y, offers[agent_y])
        remaining = _remap_conflict_labels(
            _p2.detect_conflicts(lp_x, lp_y, offers[agent_x], offers[agent_y]), agent_x, agent_y,
        )
        if not remaining:
            print(f"    [{agent_x}<->{agent_y}] Round {rnd}: all conflicts resolved.")
            break
        print(f"    [{agent_x}<->{agent_y}] -- Round {rnd}/{MAX_NEGOTIATION_ROUNDS} "
              f"(conflicts={len(remaining)}) --")

        prompt_x = _build_negotiation_prompt_n(
            agent_x, agent_y, offers[agent_x], cur_x, cur_y, remaining, locked, rnd, prev_y, task)
        prompt_y = _build_negotiation_prompt_n(
            agent_y, agent_x, offers[agent_y], cur_y, cur_x, remaining, locked, rnd, prev_x, task)

        with _TPE(max_workers=2) as ex:
            fx = ex.submit(run_vlm, images_map[agent_x], prompt_x)
            fy = ex.submit(run_vlm, images_map[agent_y], prompt_y)
            raw_x, _ = fx.result()
            raw_y, _ = fy.result()

        props_x = _parse_proposals_n(raw_x, agent_x, valid_pair)
        props_y = _parse_proposals_n(raw_y, agent_y, valid_pair)

        def _filter(props: List[NegotiationProposal]) -> List[NegotiationProposal]:
            out = []
            for p in props:
                if "ACCEPT" in p.reason.upper():
                    out.append(p); continue
                key = (p.step_id, p.field)
                if last_val.get(key) == p.new_value:
                    continue
                out.append(p); last_val[key] = p.new_value
            return out

        props_x, props_y = _filter(props_x), _filter(props_y)

        applied = 0
        for prop in props_x + props_y:
            if _apply_proposal_n(cur_x, cur_y, agent_x, prop, locked):
                applied += 1

        conflict_sids = {sid for c in remaining for sid in c.step_ids}
        locked = _lock_steps_n(props_x, props_y, locked)
        rounds.append(NegotiationRound(rnd, props_x, props_y, sorted(locked)))
        print(f"    [{agent_x}<->{agent_y}] -> applied={applied} locked={sorted(locked)}")
        prev_x, prev_y = props_x, props_y

        if applied == 0 and rnd > 1:
            print(f"    [{agent_x}<->{agent_y}] no progress -> early stop.")
            break

    return cur_x, cur_y, rounds


def phase4_negotiation_n(
    plans: Dict[str, LocalPlan], offers: Dict[str, Offer], images_map: Dict[str, str],
    conflicts_by_pair: Dict[Tuple[str, str], List[ConflictEntry]],
    agent_ids: List[str], task: str,
    use_negotiation: bool = True, verbose: str = "full",
) -> Tuple[Dict[str, List[Dict]], Dict[Tuple[str, str], List[NegotiationRound]]]:
    _banner("PHASE 4 - P2P NEGOTIATION (conflict-pair scoped)")

    cur_steps: Dict[str, List[Dict]] = {
        aid: _p2.plan_steps_to_dicts(plans[aid].steps) for aid in agent_ids
    }
    all_rounds: Dict[Tuple[str, str], List[NegotiationRound]] = {}

    if not use_negotiation:
        print("  [ABLATION] Negotiation disabled - conflicting plans pass through unresolved.")
        return cur_steps, all_rounds

    if not conflicts_by_pair:
        print("  No conflicting pairs -> skip negotiation entirely.")
        return cur_steps, all_rounds

    n_total_pairs = len(agent_ids) * (len(agent_ids) - 1) // 2
    print(f"  {len(conflicts_by_pair)}/{n_total_pairs} pair(s) will negotiate:")
    for (a, b), cl in conflicts_by_pair.items():
        print(f"    {a} <-> {b}  ({len(cl)} conflicts)")

    # 충돌 쌍끼리만 순차 negotiation. 같은 agent가 여러 쌍에 걸쳐 있으면, 직전 협상
    # 결과가 반영된 최신 plan을 다음 협상에 넘겨 전체 일관성을 유지한다.
    for (a, b), pair_conflicts in conflicts_by_pair.items():
        plans[a] = _p2._dicts_to_localplan(cur_steps[a], offers[a])
        plans[b] = _p2._dicts_to_localplan(cur_steps[b], offers[b])
        new_a, new_b, rounds = _negotiate_pair(
            a, b, plans, offers, images_map, pair_conflicts, task, verbose)
        cur_steps[a], cur_steps[b] = new_a, new_b
        all_rounds[(a, b)] = rounds

    return cur_steps, all_rounds


def _convergence_check_global(
    cur_steps, offers, agent_ids, all_conflicts,
):
    """PASS 매칭/관측가능성 검사는 본질적으로 전역(全 agent) 속성이라 쌍(pair) 단위로
    쪼개면 안 된다 — B->D PASS를 (A,B) 쌍만 놓고 보면 D가 그 쌍에 없으니 항상
    '못 받았다'고 오판된다. 그래서 전체 agent의 step을 한 번에 놓고 검사한다."""
    all_steps = [s for aid in agent_ids for s in cur_steps[aid]]
    no_cycle = not _p2._has_cycle(all_steps)

    _DERIVED_OBS = {
        "water": {"water", "drink", "beverage", "glass", "cup", "bottle"},
        "drink": {"drink", "beverage", "cup", "glass", "water", "bottle"},
        "beverage": {"beverage", "drink", "cup", "glass", "water", "bottle"},
        "coffee": {"coffee", "cup", "mug", "drink", "beverage"},
        "tea": {"tea", "cup", "mug", "drink", "beverage"},
        "kettle": {"kettle", "water", "drink", "beverage", "cup"},
        "snack": {"snack", "plate", "tray", "food", "fruit", "bread"},
        "food": {"food", "snack", "plate", "tray", "meal", "fruit"},
        "fruit": {"fruit", "snack", "plate", "tray", "food"},
        "bread": {"bread", "snack", "plate", "tray", "food"},
        "tray": {"tray", "plate", "bowl", "snack", "food", "drink"},
        "refrigerator": {"refrigerator", "drink", "beverage", "water", "food", "snack"},
        "stove": {"stove", "pot", "pan", "water", "food"},
        "sink": {"sink", "water", "glass", "cup", "beverage"},
        "counter": {"counter", "countertop", "countertops", "tray", "plate", "snack"},
        "countertop": {"counter", "countertop", "countertops", "tray", "plate", "snack"},
        "cabinet": {"cabinet", "plate", "bowl", "cup", "glass", "tray"},
    }

    def _obs_pool(offer):
        scope_kw = set(re.findall(r"\w+", offer.obs_scope.lower()))
        derived = set()
        for w in scope_kw:
            if w in _DERIVED_OBS:
                derived |= _DERIVED_OBS[w]
        can_do_kw = set()
        for cd in offer.can_do:
            can_do_kw |= _kw(cd)
        return scope_kw | derived | can_do_kw

    pools = {aid: _obs_pool(offers[aid]) for aid in agent_ids}
    _SKIP = {"receive", "accept", "notify", "inform", "wait", "pick", "take", "get", "collect", "gather"}
    obs_ok = True
    for s in all_steps:
        if s.get("handoff_type") in ("PASS", "INFORM"):
            continue
        if "auto-added" in s.get("notes", "").lower():
            continue
        fw = s.get("action", "").lower().split()[0] if s.get("action", "").strip() else ""
        if fw in _SKIP:
            continue
        pool = pools.get(s.get("agent_id"), set())
        kw = _kw(s.get("action", ""))
        if kw and pool and not (kw & pool):
            obs_ok = False
            break

    _RECV = {"receive", "accept", "pick", "collect", "get", "take"}
    truly_unmatched = set()
    for s in all_steps:
        if s.get("handoff_type") != "PASS":
            continue
        sid = s["step_id"]
        target = s.get("target_agent")
        target_steps = [t for t in all_steps if t.get("agent_id") == target]
        linked = any(sid in t.get("depends_on", []) for t in target_steps)
        if linked:
            continue
        if any(t["time_min"] > s["time_min"] for t in target_steps):
            continue
        if any(
            t.get("action", "").lower().split()[0] in _RECV
            for t in target_steps if t.get("action", "").strip()
        ):
            continue
        truly_unmatched.add(sid)
    no_missing = len(truly_unmatched) == 0

    all_ids = {s["step_id"] for s in all_steps}
    unresolved = [
        c for c in all_conflicts
        if c.conflict_type in (ConflictType.REDUNDANCY, ConflictType.CANNOT_DO)
        and all(sid in all_ids for sid in c.step_ids)
    ]
    converged = no_cycle and obs_ok and no_missing
    return ConvergenceResult(converged, no_cycle, obs_ok, no_missing, unresolved)




def phase5_convergence_check_n(
    cur_steps: Dict[str, List[Dict]], offers: Dict[str, Offer], agent_ids: List[str],
    all_conflicts: Optional[List[ConflictEntry]] = None,
) -> ConvergenceResult:
    _banner("PHASE 5 — CONVERGENCE CHECK (N-agent, global)")

    conv = _convergence_check_global(cur_steps, offers, agent_ids, all_conflicts or [])

    print(f"  No dep cycle : {'OK' if conv.no_dep_cycle else 'FAIL'}")
    print(f"  Observability: {'OK' if conv.observability_ok else 'FAIL'}")
    print(f"  PASS matched : {'OK' if conv.no_missing_deps else 'FAIL'}")
    print(f"  -> Converged : {'YES' if conv.converged else 'NO'}")
    return conv


# ══════════════════════════════════════════════════════════════════════════
# PHASE 6: DEFERRED HUMAN QUERY (N-agent)
# ══════════════════════════════════════════════════════════════════════════
#
# Negotiation + Convergence Check까지 거쳤는데도 안 풀린 문제(dep cycle, PASS
# 짝 안 맞음, observability 위반, 미해결 conflict, 아무도 못 주는 need)가 있으면
# 사람에게 질문해서 답을 받는다. 원본 phase6_human_query()의 트리거 판정 규칙과
# 질문 생성 방식을 그대로 재사용하되, agent_A/agent_B 2명 전제를 N명으로 일반화한다.

_HQ_TEMPLATES_N = {
    "DEP_CYCLE": "There's a circular dependency between agents' plans. How should it be resolved?",
    "DEPENDENCY": "A handoff step has no matching receive step. Should it be kept or removed?",
    "OBSERVABILITY": "A step references an object that may not be visible to that agent. Should it be kept?",
    "UNMATCHED": "No agent can provide a needed item. How should this be handled?",
}


def _generate_hq_question_n(
    trigger_type: str, detail: str, offers: Dict[str, Offer], agent_ids: List[str], image: str,
) -> str:
    template = _HQ_TEMPLATES_N.get(trigger_type, "How should the agents handle this?")
    agent_desc = "; ".join(f"{aid} is in the {offers[aid].room_type}" for aid in agent_ids)
    prompt = f"""You are coordinating {len(agent_ids)} home agents.
{agent_desc}.
Issue ({trigger_type}): {detail[:200]}

Write ONE clear question for the human operator that:
- Names which agent and step is involved
- Asks for a concrete decision
One sentence only. No preamble."""
    try:
        q, _ = run_vlm(image, prompt)
        q = q.strip().strip('"').strip("'")
        if 10 < len(q) < 400:
            return q
    except Exception as e:
        print(f"  [HQ VLM error] {e}")
    return f"{template}\nContext: {detail[:100]}"


def phase6_human_query_n(
    offers: Dict[str, Offer], agent_ids: List[str], images_map: Dict[str, str],
    convergence: ConvergenceResult, use_human_query: bool = True, verbose: str = "full",
) -> Tuple[Dict[str, str], List[str], List[str]]:
    _banner("PHASE 6 — DEFERRED HUMAN QUERY (N-agent)")

    if not use_human_query:
        print("  [ABLATION] disabled.")
        return {}, [], []

    if convergence.converged and not convergence.unresolved_conflicts:
        print("  Plan converged -> no query needed.")
        return {}, [], []

    raw_triggers: List[Tuple[str, str, float]] = []
    triggered: List[str] = []

    if not convergence.no_dep_cycle:
        d = "Dependency cycle detected across agents' plans."
        triggered.append(f"[DEP_CYCLE] {d}")
        raw_triggers.append(("DEP_CYCLE", d, 0.90))

    if not convergence.no_missing_deps:
        d = "A PASS step has no matching receive step on the target agent's side."
        triggered.append(f"[DEPENDENCY] {d}")
        raw_triggers.append(("DEPENDENCY", d, 0.85))

    if not convergence.observability_ok:
        d = "A step references objects outside that agent's visible scope."
        triggered.append(f"[OBSERVABILITY] {d}")
        raw_triggers.append(("OBSERVABILITY", d, 0.75))

    for c in convergence.unresolved_conflicts:
        triggered.append(f"[{c.conflict_type}] {c.description}")
        raw_triggers.append((str(c.conflict_type), c.description, 0.80))

    _INFO_KW = {
        "confirmation", "confirm", "ready", "status", "notify",
        "check", "verified", "done", "complete", "that", "whether",
    }
    all_provides = [p for aid in agent_ids for p in offers[aid].can_provide]
    for aid in agent_ids:
        for need in offers[aid].need_from_other:
            if _kw(need) & _INFO_KW:
                continue
            if not any(_fuzzy_match_soft(need, p) for p in all_provides):
                d = f"No agent can provide what {aid} needs: '{need}'"
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
    asked: List[str] = []
    rep_image = images_map[agent_ids[0]]  # 대표 이미지 (원본도 img_a 하나만 사용)

    for i, (ttype, detail, pri) in enumerate(raw_triggers[:HQ_TOP_K], 1):
        print(f"\n  Generating Q{i} [{ttype}]...", end=" ", flush=True)
        q = _generate_hq_question_n(ttype, detail, offers, agent_ids, rep_image)
        print("done")
        print(f"  Q{i}: {q}")
        asked.append(q)

        try:
            ans = input("  A: ").strip()
        except EOFError:
            ans = ""

        if ans:
            answers[q] = ans

    return answers, triggered, asked


# ══════════════════════════════════════════════════════════════════════════
# FINALIZE: RULE-BASED MERGE (N-agent)
# ══════════════════════════════════════════════════════════════════════════

def phase_finalize_n(
    cur_steps: Dict[str, List[Dict]], agent_ids: List[str],
    human_answers: Optional[Dict[str, str]] = None, verbose: str = "full",
) -> List[Dict]:
    _banner("FINALIZE - RULE-BASED MERGE (N-agent)")
    human_answers = human_answers or {}

    _DELETE_HINTS = {"delete", "remove", "skip", "drop", "ignore", "no"}
    for q, ans in human_answers.items():
        a_words = set(ans.lower().split())
        if a_words & _DELETE_HINTS:
            sids = [int(x) for x in re.findall(r"step[\s_]?(\d+)", q, re.IGNORECASE)]
            for sid in sids:
                for aid in agent_ids:
                    before = len(cur_steps[aid])
                    cur_steps[aid][:] = [s for s in cur_steps[aid] if s.get("step_id") != sid]
                    if len(cur_steps[aid]) < before:
                        print(f"  [FINALIZE] step{sid} removed per human answer: '{ans[:40]}'")

    merged = [s for aid in agent_ids for s in cur_steps[aid]]
    merged.sort(key=lambda s: (s.get("time_min", 0), s.get("step_id", 0)))

    old_to_new = {s["step_id"]: i for i, s in enumerate(merged, start=1)}
    for s in merged:
        s["step_id"] = old_to_new[s["step_id"]]
        s["depends_on"] = [old_to_new[d] for d in s.get("depends_on", []) if d in old_to_new]
        new_pre = []
        for p in s.get("preconditions", []):
            m = re.match(r"step (\d+) completed", p)
            if m and int(m.group(1)) in old_to_new:
                new_pre.append(f"step {old_to_new[int(m.group(1))]} completed")
            else:
                new_pre.append(p)
        s["preconditions"] = new_pre

    if verbose in ("full", "summary"):
        counts = ", ".join(f"{aid}={len(cur_steps[aid])}" for aid in agent_ids)
        print(f"  {counts}  ->  merged total = {len(merged)}")
    return merged


# ══════════════════════════════════════════════════════════════════════════
# OUTPUT FORMAT (N-agent)
# ══════════════════════════════════════════════════════════════════════════

def format_joint_plan_n(plan: List[Dict], agent_ids: List[str], task: str = "") -> str:
    if not plan:
        return "  (empty)"
    SEP = "-" * 68
    n_pass = sum(1 for s in plan if s.get("handoff_type") == "PASS")
    n_inform = sum(1 for s in plan if s.get("handoff_type") == "INFORM")
    max_t = max((s.get("time_min", 0) for s in plan), default=0)

    rooms = {}
    for aid in agent_ids:
        st = next((s for s in plan if s.get("agent_id") == aid), None)
        rooms[aid] = st.get("room", aid) if st else aid

    lines = [SEP]
    if task:
        t = task[:60] + "..." if len(task) > 60 else task
        lines.append(f'  "{t}"')
    header = " + ".join(f"{rooms[aid].upper()} ({aid})" for aid in agent_ids)
    lines.append(f"  {header}  |  {len(plan)} steps  |  {n_pass} handoff  |  "
                 f"{n_inform} notify  |  ~{max_t} min")
    lines.append(SEP)
    lines.append("")

    id_to_step = {s["step_id"]: s for s in plan}
    all_times = sorted({s.get("time_min", 0) for s in plan})
    for t in all_times:
        slot = sorted([s for s in plan if s.get("time_min") == t], key=lambda s: s.get("agent_id", ""))
        lines.append(f"  T = {t:>2} min")
        for s in slot:
            room, action = s.get("room", "?"), s.get("action", "")
            ht, tgt = s.get("handoff_type"), s.get("target_agent", "")
            deps = s.get("depends_on", [])
            cross = [id_to_step[d] for d in deps
                     if d in id_to_step and id_to_step[d].get("agent_id") != s.get("agent_id")]

            if ht == "PASS":
                tag = f"[PASS->{tgt}] "
            elif ht == "INFORM":
                tag = f"[NOTIFY->{tgt}] "
            elif cross and any(id_to_step[d].get("handoff_type") == "PASS" for d in deps if d in id_to_step):
                # PASS를 받는 쪽(별도 handoff_type 없이 depends_on으로만 연결된 receive step)
                sender = next(
                    id_to_step[d].get("agent_id") for d in deps
                    if d in id_to_step and id_to_step[d].get("handoff_type") == "PASS"
                )
                tag = f"[RECEIVE<-{sender}] "
            else:
                tag = ""

            dep_note = ""
            if cross:
                acts = ", ".join(f'"{x["action"][:35]}"' for x in cross[:2])
                dep_note = f"\n       (waits for: {acts})"
            lines.append(f"  [{s.get('agent_id')}|{room}] {tag}{action}{dep_note}")
        lines.append("")
    lines.append(SEP)
    return "\n".join(lines)
