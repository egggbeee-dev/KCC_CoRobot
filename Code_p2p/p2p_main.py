# main.py
# 진입점: run() — 단일 실험 실행
#
# 파이프라인:
#   Phase 1 : Observation & Offer Generation
#   Phase 2 : Independent Local Planning
#   Phase 3 : Conflict Detection
#   Phase 4 : P2P Negotiation (구조화 제안, hard limit 3라운드, step lock)
#   Phase 5 : Convergence Check (rule-based)
#   Phase 6 : Deferred Human Query (필요 시에만)
#   Finalize: Rule-based merge (LLM 호출 없음)
#
# 사용 예:
#   from p2p_main import run, list_tasks
#   result = run(task_id="task_003", img_a="...", img_b="...")
#
#   from p2p_ablation import run_ablation
#   results = run_ablation(task_id="task_003", img_a=..., img_b=...)

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from p2p_phases import (
    format_joint_plan,
    local_plan_to_dict,
    offer_to_dict,
    phase1_offer,
    phase2_local_plan,
    phase3_conflict_detection,
    phase4_negotiation,
    phase5_convergence_check,
    phase6_human_query,
    phase_finalize,
    plan_steps_to_dicts,
    _kw,
)
from p2p_utils import compute_joint_uncertainty, _banner, jdump

_BASE      = Path(__file__).parent.parent
TASKS_PATH = _BASE / "Data" / "Task" / "tasks.json"


def _load_tasks() -> List[Dict]:
    if not TASKS_PATH.exists():
        raise FileNotFoundError(f"tasks.json not found: {TASKS_PATH}")
    with open(TASKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def list_tasks() -> None:
    tasks = _load_tasks()
    print(f"\n{'─'*60}")
    print(f"  {'ID':<12} Description")
    print(f"{'─'*60}")
    for t in tasks:
        desc = t["description"].replace("\n", " ")
        print(f"  {t['id']:<12} {desc[:55]}{'...' if len(desc) > 55 else ''}")
    print(f"{'─'*60}\n")


def get_task(task_id: str) -> str:
    tasks = _load_tasks()
    for t in tasks:
        if t["id"] == task_id:
            return t["description"]
    available = [t["id"] for t in tasks]
    raise KeyError(f"task_id '{task_id}' not found. Available: {available}")


def run(
    task_id:         Optional[str] = None,
    img_a:           Optional[str] = None,
    img_b:           Optional[str] = None,
    use_offer:       bool = True,
    use_negotiation: bool = True,
    use_human_query: bool = True,
    use_handoff:     bool = True,   # 하위 호환성 유지 (내부적으로 무시됨)
    label:           Optional[str] = None,
    verbose:         str  = "full",
) -> Dict:
    """
    P2P Collaborative VLM Planning 전체 파이프라인.

    Args:
        task_id        : tasks.json의 ID
        img_a          : agent_A 이미지 경로
        img_b          : agent_B 이미지 경로
        use_offer      : False → offer 없이 방 타입만으로 플래닝 (ablation)
        use_handoff    : False → handoff 선언 없이 플래닝 (ablation)
        use_human_query: False → Phase 6 skip (ablation)
        use_negotiation: False → Phase 4 skip (ablation)
        label          : 실험 레이블
        verbose        : "full" | "summary" | "minimal"

    Returns:
        실험 결과 dict
    """
    if task_id is None:
        list_tasks()
        raise ValueError("task_id를 지정해주세요.")

    task  = get_task(task_id)
    label = label or task_id

    if not img_a or not img_b:
        raise ValueError("img_a와 img_b 경로를 모두 지정해주세요.")
    if not Path(img_a).exists():
        raise FileNotFoundError(f"img_a not found: {img_a}")
    if not Path(img_b).exists():
        raise FileNotFoundError(f"img_b not found: {img_b}")

    print("\n" + "█" * 68)
    print(f"  P2P COLLABORATIVE VLM PLANNING — {label}")
    print("█" * 68)
    print(f"  Task    : {task[:80]}{'...' if len(task) > 80 else ''}")
    print(f"  Flags   : offer={use_offer} | handoff={use_handoff} | "
          f"negotiation={use_negotiation} | hq={use_human_query}")

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    offer_a, offer_b = phase1_offer(img_a, img_b, task, verbose=verbose)

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    plan_a, plan_b = phase2_local_plan(
        offer_a, offer_b, img_a, img_b, task,
        use_offer=use_offer, verbose=verbose,
    )

    # conflicts before negotiation (논문 메트릭용)
    conflicts = phase3_conflict_detection(plan_a, plan_b, offer_a, offer_b, verbose=verbose)
    n_conflicts = len(conflicts)

    # ── Phase 4 ──────────────────────────────────────────────────────────────
    if use_negotiation:
        neg_steps_a, neg_steps_b, neg_rounds = phase4_negotiation(
            plan_a, plan_b, offer_a, offer_b, conflicts,
            img_a, img_b, task, verbose=verbose,
        )
    else:
        _banner("PHASE 4 — P2P NEGOTIATION")
        print("  [ABLATION] Negotiation disabled.")
        neg_steps_a = plan_steps_to_dicts(plan_a.steps)
        neg_steps_b = plan_steps_to_dicts(plan_b.steps)
        neg_rounds  = []

    # ── Phase 5 ──────────────────────────────────────────────────────────────
    # Phase 5용 conflict list: 협상 후 재탐지가 이상적이지만
    # 비용 절감을 위해 원래 conflicts를 넘기고 수렴 조건만 rule-based로 판단
    convergence = phase5_convergence_check(
        neg_steps_a, neg_steps_b, offer_a, offer_b, conflicts,
    )

    # ── Phase 6 ──────────────────────────────────────────────────────────────
    human_answers, hq_triggers, hq_asked = phase6_human_query(
        plan_a, plan_b, offer_a, offer_b,
        img_a, img_b,
        task=task,
        use_human_query=use_human_query,
    )

    # ── Finalize (rule-based merge, LLM 없음) ─────────────────────────────────
    joint = phase_finalize(
        neg_steps_a, neg_steps_b,
        offer_a, offer_b,
        human_answers,
        verbose=verbose,
    )

    # ── 논문 메트릭 계산 ──────────────────────────────────────────────────────
    # conflict_reduction: 협상 전후 conflict 수 감소율
    conflict_reduction = (
        (n_conflicts - convergence.unresolved_conflicts.__len__()) / max(n_conflicts, 1)
    )

    # observability: 각 step이 자기 obs_scope 내에 있는 비율
    scope_a = set(re.findall(r"\w+", offer_a.obs_scope.lower()))
    scope_b = set(re.findall(r"\w+", offer_b.obs_scope.lower()))
    can_kw_a: set = set()
    can_kw_b: set = set()
    for cd in offer_a.can_do: can_kw_a |= _kw(cd)
    for cd in offer_b.can_do: can_kw_b |= _kw(cd)

    obs_violations = 0
    for s in joint:
        if s.get("handoff_type") == "PASS":
            continue
        pool = (can_kw_a | scope_a) if s.get("agent_id") == "agent_A" else (can_kw_b | scope_b)
        kw   = _kw(s.get("action", ""))
        if kw and pool and not (kw & pool):
            obs_violations += 1
    observability_rate = round(1.0 - obs_violations / max(len(joint), 1), 3)


    # cross-agent depends_on 수
    id_to_agent = {s["step_id"]: s.get("agent_id") for s in joint}
    cross_deps  = sum(
        1 for s in joint
        for d in s.get("depends_on", [])
        if id_to_agent.get(d) and id_to_agent[d] != s.get("agent_id")
    )

    # PASS 매칭률
    pass_steps    = {s["step_id"] for s in joint if s.get("handoff_type") == "PASS"}
    all_deps      = {d for s in joint for d in s.get("depends_on", [])}
    matched_pass  = len(pass_steps & all_deps)
    handoff_match = matched_pass / max(len(pass_steps), 1) if pass_steps else 1.0

    U_joint = compute_joint_uncertainty(joint)

    metrics = {
        # 플랜 품질 지표 (핵심)
        "handoff_match_rate":  round(handoff_match, 3),
        "cross_agent_deps":    cross_deps,
        "conflict_reduction":  round(conflict_reduction, 3),
        "observability_rate":  round(observability_rate, 3),
        "U_joint":             U_joint,
        # 협상/HQ 지표
        "negotiation_rounds":  len(neg_rounds),
        "hq_triggered":        len(hq_triggers),
        "hq_asked":            len(hq_asked),
        # 참고용
        "conflicts":           n_conflicts,
        "conflicts_after":     len(convergence.unresolved_conflicts),
    }

    # ── 최종 출력 ─────────────────────────────────────────────────────────────
    print("\n" + "█" * 68)
    print(f"  FINAL JOINT PLAN — {label}")
    print("█" * 68)
    print(format_joint_plan(joint))

    print(f"\n  METRICS")
    print(f"  {'─'*40}")
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")


    return {
        "label":   label,
        "task_id": task_id,
        "task":    task,
        "flags": {
            "use_offer":       use_offer,
            "use_handoff":     use_handoff,
            "use_negotiation": use_negotiation,
            "use_human_query": use_human_query,
        },
        "offers": {
            "agent_A": offer_to_dict(offer_a),
            "agent_B": offer_to_dict(offer_b),
        },
        "local_plans": {
            "agent_A": local_plan_to_dict(plan_a),
            "agent_B": local_plan_to_dict(plan_b),
        },
        "conflicts": [asdict(c) for c in conflicts],
        "negotiation": {
            "rounds": len(neg_rounds),
            "history": [
                {
                    "round_num":       r.round_num,
                    "proposals_a":     [asdict(p) for p in r.proposals_a],
                    "proposals_b":     [asdict(p) for p in r.proposals_b],
                    "locked_step_ids": r.locked_step_ids,
                }
                for r in neg_rounds
            ],
        },
        "convergence": {
            "converged":        convergence.converged,
            "no_missing_deps":  convergence.no_missing_deps,
            "no_dep_cycle":     convergence.no_dep_cycle,
            "observability_ok": convergence.observability_ok,
            "unresolved":       len(convergence.unresolved_conflicts),
        },
        "human_answers": human_answers,
        "hq_triggers":   hq_triggers,
        "hq_asked":      hq_asked,
        "joint_plan":    joint,
        "metrics":       metrics,
    }
