# verifier.py — Joint Plan 최종 검증 및 점수 계산

from __future__ import annotations
from typing import Dict, List

from p2p_config import VALID_AGENTS
from p2p_models import CannotEntry, Offer, VerifyResult
from p2p_utils import _fuzzy_match, safe_int


def _violates_cannot(action: str, cannot: List[CannotEntry]) -> bool:
    al = action.lower().strip()
    if any(al.startswith(p) for p in ("receive", "confirm")):
        return False
    for c in cannot:
        if _fuzzy_match(action, c.action, min_overlap=3):
            return True
    return False


def verify(plan: List[Dict], offer_a: Offer, offer_b: Offer) -> VerifyResult:
    errors:   List[str] = []
    warnings: List[str] = []

    if not plan:
        return VerifyResult(False, ["EMPTY_PLAN"])

    offers      = {"agent_A": offer_a, "agent_B": offer_b}
    step_id_set = {s.get("step_id") for s in plan}

    # 중복 step_id
    seen: set = set()
    for s in plan:
        sid = s.get("step_id")
        if sid in seen:
            errors.append(f"DUPLICATE_STEP_ID:{sid}")
        seen.add(sid)

    # 부하 균형
    finish: Dict[str, int] = {}
    for s in plan:
        aid = s.get("agent_id", "")
        finish[aid] = max(finish.get(aid, 0), safe_int(s.get("time_min", 0), 0))
    if len(finish) == 2:
        vals = list(finish.values())
        if abs(vals[0] - vals[1]) > 5:
            warnings.append(f"LOAD_IMBALANCE: finish={finish}")

    exec_violations = dep_errors = 0

    for idx, s in enumerate(plan, 1):
        agent_id = s.get("agent_id", "")
        action   = s.get("action", "")
        deps     = s.get("depends_on", [])

        if agent_id not in VALID_AGENTS:
            errors.append(f"UNKNOWN_AGENT:step{idx}:{agent_id}")
            continue
        if not action:
            errors.append(f"MISSING_ACTION:step{idx}")
            continue

        offer = offers[agent_id]
        if _violates_cannot(action, offer.cannot_do):
            errors.append(f"CANNOT_DO:step{idx}:'{action}'")
            exec_violations += 1

        for dep in (deps if isinstance(deps, list) else []):
            if dep not in step_id_set:
                errors.append(f"MISSING_DEP:step{idx}:dep={dep}")
                dep_errors += 1

        t = safe_int(s.get("time_min", 0), 0)
        if not (0 <= t <= 30):
            errors.append(f"TIME_RANGE:step{idx}:t={t}")

    total = max(len(plan), 1)

    # cross-agent depends_on (coordination 지표)
    cross_deps = sum(
        1 for s in plan
        for dep in s.get("depends_on", [])
        if any(s2["step_id"] == dep and s2["agent_id"] != s["agent_id"] for s2 in plan)
    )
    coord_score = round(min(1.0, cross_deps / max(1, len(plan) // 3)), 3)

    return VerifyResult(
        is_valid            = len(errors) == 0,
        errors              = errors,
        warnings            = warnings,
        completeness_score  = round(min(1.0, len(plan) / 8), 3),
        executability_score = round(max(0.0, 1.0 - exec_violations / total), 3),
        observability_score = 1.0,
        handoff_score       = coord_score,
        sequential_score    = round(max(0.0, 1.0 - dep_errors / max(
            sum(len(s.get("depends_on", [])) for s in plan), 1
        )), 3),
    )


def format_final_plan(plan: List[Dict]) -> str:
    if not plan:
        return "  (empty)"
    lines = []
    for s in sorted(plan, key=lambda x: (x.get("time_min", 0), x.get("step_id", 0))):
        dep  = f" deps={s['depends_on']}" if s.get("depends_on") else ""
        note = f" ({s['notes']})"          if s.get("notes")     else ""
        lines.append(
            f"  {s['step_id']:>2}. [T={s['time_min']:>2}m] "
            f"[{s.get('room','?'):<12}] [{s.get('agent_id','?')}]  "
            f"{s['action']}{dep}{note}"
        )
    return "\n".join(lines)


def print_scores(vr: VerifyResult, label: str = "") -> None:
    title = "EVALUATION SCORES" + (f" — {label}" if label else "")
    print(f"\n  {title}")
    print(f"  {'─'*40}")
    for name, val in [
        ("Completeness",        vr.completeness_score),
        ("Executability",       vr.executability_score),
        ("Coordination (deps)", vr.handoff_score),
        ("Sequential consist.", vr.sequential_score),
    ]:
        print(f"  {name:<26} {val:>6.3f}")
    print(f"  {'─'*40}")
    print(f"  {'TOTAL':<26} {vr.total_score:>6.3f}")
