# ══════════════════════════════════════════════════════════════════════════════
# verifier.py
# Joint Plan 검증 및 휴리스틱 점수 계산
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import Dict, List

from config import VALID_AGENTS
from models import CannotEntry, Offer, VerifyResult
from utils import _fuzzy_match, _fuzzy_match_soft, safe_int


# ──────────────────────────────────────────────────────────────────────────────
# 내부 검사 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _violates_cannot(action: str, cannot: List[CannotEntry]) -> bool:
    """
    cannot_do 위반 검사.
    1) exact match 먼저 체크
    2) fuzzy match는 min_overlap=3 (오탐 방지)
    3) auto-generated / INFORM / receive 액션은 스킵
    """
    action_lower = action.lower().strip()
    skip_prefixes = ("auto-generated", "inform", "receive and", "receive snack", "receive item")
    if any(action_lower.startswith(p) for p in skip_prefixes):
        return False
    for c in cannot:
        if c.action.lower().strip() == action_lower:
            return True
        if _fuzzy_match(action, c.action, min_overlap=3):
            return True
    return False


def _relay_sender_in_cando(action: str, can_do: List[str]) -> bool:
    """
    RELAY sender의 액션이 해당 에이전트의 can_do에 있는지 확인.
    fuzzy match soft 사용 (0.4 이상 overlap).
    """
    return any(_fuzzy_match_soft(action, cd) for cd in can_do)


# ──────────────────────────────────────────────────────────────────────────────
# 메인 검증 함수
# ──────────────────────────────────────────────────────────────────────────────

def verify(plan: List[Dict], offer_a: Offer, offer_b: Offer) -> VerifyResult:
    """
    Joint plan 리스트를 받아 오류·경고를 탐지하고 각 차원의 점수를 반환한다.
    """
    errors:   List[str] = []
    warnings: List[str] = []

    if not plan:
        return VerifyResult(False, ["EMPTY_PLAN"], [])

    offers      = {"agent_A": offer_a, "agent_B": offer_b}
    step_id_set = {s.get("step_id") for s in plan}

    # ── 중복 step_id 검사 ────────────────────────────────────────────────────
    seen: set = set()
    for s in plan:
        sid = s.get("step_id")
        if sid in seen:
            errors.append(f"DUPLICATE_STEP_ID:{sid}")
        seen.add(sid)

    # ── 부하 균형 검사 ────────────────────────────────────────────────────────
    finish: Dict[str, int] = {}
    for s in plan:
        aid = s.get("agent_id", "")
        finish[aid] = max(finish.get(aid, 0), safe_int(s.get("time_min", 0), 0))
    if len(finish) == 2:
        vals = list(finish.values())
        if abs(vals[0] - vals[1]) > 5:
            warnings.append(f"LOAD_IMBALANCE: finish={finish}")

    # ── RELAY 수신 미완성 검사 ────────────────────────────────────────────────
    relay_senders   = {s["step_id"]: s for s in plan if s.get("handoff_type") == "RELAY"}
    relay_received: set = set()
    for s in plan:
        for dep in s.get("depends_on", []):
            if dep in relay_senders:
                relay_received.add(dep)
    for sid in relay_senders:
        if sid not in relay_received:
            warnings.append(f"UNRESOLVED_RELAY:step_{sid}")

    # ── RELAY 방향 검증 ───────────────────────────────────────────────────────
    # sender와 target이 같은 에이전트인지 (self-loop)
    # sender의 액션이 자신의 can_do에 있는지
    for sid, relay_s in relay_senders.items():
        sender = relay_s.get("agent_id", "")
        target = relay_s.get("target_agent", "")

        # self-loop 검사
        if sender == target:
            errors.append(f"RELAY_SELF_LOOP:step_{sid} (sender==target=={sender})")

        # sender가 valid agent인지
        if sender in offers:
            offer = offers[sender]
            action = relay_s.get("action", "")
            if not _relay_sender_in_cando(action, offer.can_do):
                warnings.append(
                    f"RELAY_NOT_IN_CANDO:step_{sid}:{sender}:'{action}' "
                    f"not found in can_do"
                )

        # receiver가 반대 에이전트인지 확인
        expected_receiver = "agent_B" if sender == "agent_A" else "agent_A"
        if target and target != expected_receiver:
            warnings.append(
                f"RELAY_WRONG_TARGET:step_{sid} sender={sender} target={target} "
                f"(expected {expected_receiver})"
            )

    exec_violations = obs_violations = dep_errors = handoff_count = 0

    # ── 스텝별 검사 ───────────────────────────────────────────────────────────
    for idx, s in enumerate(plan, 1):
        agent_id = s.get("agent_id", "")
        room     = s.get("room", "").lower().replace("_", " ")
        action   = s.get("action", "")
        deps     = s.get("depends_on", [])

        if "|" in agent_id or "|" in room:
            errors.append(f"PIPE_SEPARATOR:step_{idx}")
            continue
        if agent_id not in VALID_AGENTS:
            errors.append(f"UNKNOWN_AGENT:step_{idx}:{agent_id}")
            continue
        if not action:
            errors.append(f"MISSING_ACTION:step_{idx}")
            continue

        offer = offers[agent_id]

        # cannot_do 위반
        if _violates_cannot(action, offer.cannot_do):
            errors.append(f"CANNOT_DO_VIOLATION:step_{idx}:{agent_id}:'{action}'")
            exec_violations += 1

        # depends_on 유효성
        for dep in (deps if isinstance(deps, list) else []):
            if dep not in step_id_set:
                errors.append(f"MISSING_DEP:step_{idx} dep={dep}")
                dep_errors += 1

        # 시간 범위
        t = safe_int(s.get("time_min", 0), 0)
        if t < 0 or t > 30:
            errors.append(f"TIME_RANGE:step_{idx}:t={t}")

        if s.get("handoff_type"):
            handoff_count += 1

    total = max(len(plan), 1)
    return VerifyResult(
        is_valid            = len(errors) == 0,
        errors              = errors,
        warnings            = warnings,
        completeness_score  = round(min(1.0, len(plan) / 8), 3),
        executability_score = round(max(0.0, 1.0 - exec_violations / total), 3),
        observability_score = round(max(0.0, 1.0 - obs_violations / total), 3),
        handoff_score       = round(min(1.0, handoff_count / 2), 3),
        sequential_score    = round(
            max(0.0, 1.0 - dep_errors / max(
                sum(len(s.get("depends_on", [])) for s in plan), 1
            )), 3
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 출력 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def format_final_plan(plan: List[Dict]) -> str:
    if not plan:
        return "  (empty)"
    lines = []
    for s in plan:
        dep  = f" deps={s['depends_on']}" if s.get("depends_on") else ""
        hoff = f" [{s['handoff_type']}→{s['target_agent']}]" if s.get("handoff_type") else ""
        note = f" ({s['notes']})" if s.get("notes") else ""
        lines.append(
            f"  {s['step_id']:>2}. [T={s['time_min']:>2}m] [{s['room']:<12}] [{s['agent_id']}]  "
            f"{s['action']}{hoff}{dep}{note}"
        )
    return "\n".join(lines)


def print_scores(vr: VerifyResult, label: str = ""):
    title = "EVALUATION SCORES" + (f" — {label}" if label else "")
    print(f"\n  {title}")
    print(f"  {'─'*40}")
    for name, val in [
        ("Completeness",        vr.completeness_score),
        ("Executability",       vr.executability_score),
        ("Observability",       vr.observability_score),
        ("Handoff Quality",     vr.handoff_score),
        ("Sequential Consist.", vr.sequential_score),
    ]:
        print(f"  {name:<25} {val:>6.3f}")
    print(f"  {'─'*40}")
    print(f"  {'TOTAL (weighted)':<25} {vr.total_score:>6.3f}")
