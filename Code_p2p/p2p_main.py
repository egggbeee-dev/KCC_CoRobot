# ══════════════════════════════════════════════════════════════════════════════
# main.py
# 진입점: run() — 단일 실험 실행
#
# 새 파이프라인 구조:
#   Phase 1 : Observation & Offer Generation
#   Phase 2 : Independent Local Planning (각 에이전트 독립)
#   Phase 3 : Conflict Detection (temporal / dependency / redundancy / ...)
#   Phase 4 : P2P Negotiation (최대 MAX_NEGOTIATION_ROUNDS 라운드, 합의 step lock)
#   Phase 5 : Convergence Check (rule-based: PASS매칭, cycle없음, observability)
#   Phase 6 : Deferred Human Query (수렴 실패 or 미해결 충돌 시에만)
#   Finalize: Joint Plan 확정
#
# 코랩 사용 예:
#   from p2p_main import run, get_task, list_tasks
#
#   list_tasks()   # 사용 가능한 task_id 확인
#
#   result = run(
#       task_id = "task_003",
#       img_a   = "/content/repo/Data/Room/Kitchens/real/k1.jpg",
#       img_b   = "/content/repo/Data/Room/Livingrooms/real/l1.jpg",
#       verbose = "summary",
#   )
#
#   # ablation:
#   from p2p_ablation import run_ablation
#   results = run_ablation(task_id="task_003", img_a=..., img_b=...)
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from p2p_phases import (
    local_plan_to_dict,
    offer_to_dict,
    phase1_offer,
    phase2_local_plan,
    phase3_conflict_detection,
    phase4_negotiation,
    phase5_convergence_check,
    phase6_human_query,
    phase_finalize,
)
from p2p_utils import _banner, jdump
from p2p_verifier import format_final_plan, print_scores, verify

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
_BASE      = Path(__file__).parent.parent
TASKS_PATH = _BASE / "Data" / "Task" / "tasks.json"
ROOMS_PATH = _BASE / "Data" / "Room"


# ── Task 로딩 ─────────────────────────────────────────────────────────────────

def _load_tasks() -> List[Dict]:
    if not TASKS_PATH.exists():
        raise FileNotFoundError(
            f"tasks.json을 찾을 수 없습니다: {TASKS_PATH}\n"
            f"  경로를 확인하거나 TASKS_PATH를 수정하세요."
        )
    with open(TASKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def list_tasks() -> None:
    """사용 가능한 task_id 목록을 출력한다. 코랩에서 확인용으로 사용."""
    tasks = _load_tasks()
    print(f"\n{'─'*60}")
    print(f"  {'ID':<12} Description")
    print(f"{'─'*60}")
    for t in tasks:
        desc = t["description"].replace("\n", " ")
        print(f"  {t['id']:<12} {desc[:55]}{'...' if len(desc) > 55 else ''}")
    print(f"{'─'*60}\n")


def get_task(task_id: str) -> str:
    """
    tasks.json에서 task_id에 해당하는 description을 반환한다.

    Args:
        task_id: "task_001" 형식의 ID

    Returns:
        태스크 description 문자열

    Raises:
        KeyError: task_id가 존재하지 않을 때
    """
    tasks = _load_tasks()
    for t in tasks:
        if t["id"] == task_id:
            return t["description"]
    available = [t["id"] for t in tasks]
    raise KeyError(
        f"task_id '{task_id}'를 찾을 수 없습니다.\n"
        f"  사용 가능한 ID: {available}\n"
        f"  list_tasks()로 목록을 확인하세요."
    )


# ── 메인 파이프라인 ────────────────────────────────────────────────────────────

def run(
    task_id:           Optional[str]  = None,
    img_a:             Optional[str]  = None,
    img_b:             Optional[str]  = None,
    use_offer:         bool           = True,
    use_handoff:       bool           = True,
    use_human_query:   bool           = True,
    use_negotiation:   bool           = True,
    label:             Optional[str]  = None,
    verbose:           str            = "full",
) -> Dict:
    """
    P2P Collaborative VLM Planning 전체 파이프라인을 실행한다.

    Args:
        task_id          : tasks.json의 ID (예: "task_003").
                           None이면 list_tasks() 출력 후 ValueError 발생.
        img_a            : agent_A 이미지 경로
        img_b            : agent_B 이미지 경로
        use_offer        : False → offer 없이 방 타입만으로 플래닝 (ablation)
        use_handoff      : False → handoff 선언 없이 플래닝 (ablation)
        use_human_query  : False → Phase 6 skip (ablation)
        use_negotiation  : False → Phase 4 skip, 바로 finalize (ablation)
        label            : 실험 레이블 (None이면 task_id 사용)
        verbose          : "full" | "summary" | "minimal"

    Returns:
        실험 결과 전체를 담은 dict
    """
    # ── task 로딩 ────────────────────────────────────────────────────────────
    if task_id is None:
        list_tasks()
        raise ValueError(
            "task_id를 지정해주세요.\n"
            "  예: run(task_id='task_003', img_a='...', img_b='...')"
        )
    task  = get_task(task_id)
    label = label or task_id

    # ── 이미지 경로 확인 ─────────────────────────────────────────────────────
    if not img_a or not img_b:
        raise ValueError(
            "img_a와 img_b 경로를 모두 지정해주세요.\n"
            f"  예: run(task_id='{task_id}', "
            "img_a='Data/Room/Kitchens/real/k1.jpg', "
            "img_b='Data/Room/Livingrooms/real/l1.jpg')"
        )
    if not Path(img_a).exists():
        raise FileNotFoundError(f"img_a 파일이 없습니다: {img_a}")
    if not Path(img_b).exists():
        raise FileNotFoundError(f"img_b 파일이 없습니다: {img_b}")

    # ── 헤더 출력 ────────────────────────────────────────────────────────────
    print("\n" + "█" * 68)
    print(f"  P2P COLLABORATIVE VLM PLANNING — {label}")
    print("█" * 68)
    print(f"  Task ID : {task_id}")
    print(f"  Task    : {task[:80]}{'...' if len(task) > 80 else ''}")
    print(f"  Img A   : {Path(img_a).name}")
    print(f"  Img B   : {Path(img_b).name}")
    print(f"  Flags   : offer={use_offer} | handoff={use_handoff} | "
          f"negotiation={use_negotiation} | hq={use_human_query} | verbose={verbose}")

    # ── Phase 1: Observation & Offer ─────────────────────────────────────────
    offer_a, offer_b = phase1_offer(img_a, img_b, task, verbose=verbose)

    # ── Phase 2: Independent Local Planning ──────────────────────────────────
    plan_a, plan_b = phase2_local_plan(
        offer_a, offer_b, img_a, img_b, task,
        use_offer=use_offer, use_handoff=use_handoff, verbose=verbose,
    )

    # ── Phase 3: Conflict Detection ───────────────────────────────────────────
    conflicts = phase3_conflict_detection(plan_a, plan_b, offer_a, offer_b, verbose=verbose)

    # ── Phase 4: P2P Negotiation ──────────────────────────────────────────────
    if use_negotiation:
        neg_steps_a, neg_steps_b, neg_rounds = phase4_negotiation(
            plan_a, plan_b, offer_a, offer_b, conflicts,
            img_a, img_b, task,
            use_handoff=use_handoff, verbose=verbose,
        )
    else:
        _banner("PHASE 4 — P2P NEGOTIATION")
        print("  [ABLATION] Negotiation disabled. Using raw local plans.")
        neg_steps_a = [asdict(s) for s in plan_a.steps]
        neg_steps_b = [asdict(s) for s in plan_b.steps]
        neg_rounds  = []

    # ── Phase 5: Convergence Check ────────────────────────────────────────────
    convergence = phase5_convergence_check(
        neg_steps_a, neg_steps_b, offer_a, offer_b, conflicts
    )

    # ── Phase 6: Deferred Human Query ────────────────────────────────────────
    human_answers, hq_triggers, hq_asked = phase6_human_query(
        plan_a, plan_b, offer_a, offer_b,
        convergence, img_a, img_b,
        use_human_query=use_human_query,
    )

    # ── Finalize: Joint Plan 확정 ─────────────────────────────────────────────
    joint = phase_finalize(
        neg_steps_a, neg_steps_b,
        offer_a, offer_b,
        human_answers, convergence,
        task, img_a, img_b,
        verbose=verbose,
    )

    # ── 최종 검증 (경량 sanity check) ─────────────────────────────────────────
    vr = verify(joint, offer_a, offer_b)

    _banner("FINAL VERIFICATION")
    print(f"  is_valid : {vr.is_valid}")
    if vr.errors:
        for e in vr.errors: print(f"    ✗ {e}")
    if vr.warnings:
        for w in vr.warnings: print(f"    ⚠ {w}")

    # ── Final Output ──────────────────────────────────────────────────────────
    print("\n" + "█" * 68)
    print(f"  FINAL JOINT PLAN — {label}")
    print("█" * 68)
    print(format_final_plan(joint))
    print_scores(vr, label)

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
        "offers":      {"agent_A": offer_to_dict(offer_a), "agent_B": offer_to_dict(offer_b)},
        "local_plans": {"agent_A": local_plan_to_dict(plan_a), "agent_B": local_plan_to_dict(plan_b)},
        "conflicts":   [asdict(c) for c in conflicts],
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
            "pass_matched":     convergence.pass_matched,
            "no_dep_cycle":     convergence.no_dep_cycle,
            "observability_ok": convergence.observability_ok,
            "unresolved":       len(convergence.unresolved_conflicts),
        },
        "human_answers": human_answers,
        "hq_triggers":   hq_triggers,
        "hq_asked":      hq_asked,
        "joint_plan":    joint,
        "verification":  asdict(vr),
        "scores": {
            "completeness":  vr.completeness_score,
            "executability": vr.executability_score,
            "observability": vr.observability_score,
            "handoff":       vr.handoff_score,
            "sequential":    vr.sequential_score,
            "total":         vr.total_score,
        },
    }
