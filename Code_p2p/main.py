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
# 사용 예:
#   from main import run
#   result = run()
#
#   # ablation:
#   from ablation import run_ablation
#   results = run_ablation()
#
#   # LLM-as-Judge:
#   from evaluator import evaluate, print_evaluation
#   eval_result = evaluate(result, api_key="sk-...")
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from phases import (
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
from utils import _banner, jdump
from verifier import format_final_plan, print_scores, verify

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
_BASE      = Path(__file__).parent.parent
TASKS_PATH = _BASE / "Data" / "Task" / "tasks.json"
ROOMS_PATH = _BASE / "Data" / "Room"


# ── Task 선택 ─────────────────────────────────────────────────────────────────

def _load_tasks() -> List[Dict]:
    if not TASKS_PATH.exists():
        return []
    with open(TASKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _select_task() -> str:
    tasks = _load_tasks()
    if tasks:
        print("\n사용 가능한 태스크:")
        for i, t in enumerate(tasks, 1):
            print(f"  [{i}] {t['id']} — {t['description']}")
        print(f"  [{len(tasks) + 1}] 직접 입력")
        while True:
            choice = input("\n번호 선택: ").strip()
            if not choice.isdigit():
                print("  숫자를 입력해주세요.")
                continue
            idx = int(choice) - 1
            if 0 <= idx < len(tasks):
                return tasks[idx]["description"]
            if idx == len(tasks):
                break
            print("  범위를 벗어난 번호입니다.")
    task = input("태스크를 입력하세요: ").strip()
    if not task:
        raise ValueError("태스크를 입력해야 합니다.")
    return task


# ── Image 선택 ────────────────────────────────────────────────────────────────

def _list_images() -> List[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    images: List[Path] = []
    for ext in exts:
        images.extend(sorted(ROOMS_PATH.rglob(ext)))
    return images


def _select_image(label: str) -> str:
    images = _list_images()
    if not images:
        path = input(f"{label} 이미지 경로를 직접 입력하세요: ").strip()
        if not path:
            raise ValueError(f"{label} 이미지 경로를 입력해야 합니다.")
        return path
    print(f"\n{label} 이미지 선택 (Data/Room/):")
    for i, p in enumerate(images, 1):
        print(f"  [{i}] {p.relative_to(_BASE)}")
    while True:
        choice = input("번호 선택: ").strip()
        if not choice.isdigit():
            print("  숫자를 입력해주세요.")
            continue
        idx = int(choice) - 1
        if 0 <= idx < len(images):
            return str(images[idx])
        print("  범위를 벗어난 번호입니다.")


# ── 메인 파이프라인 ────────────────────────────────────────────────────────────

def run(
    task:              Optional[str]  = None,
    img_a:             Optional[str]  = None,
    img_b:             Optional[str]  = None,
    use_offer:         bool           = True,
    use_handoff:       bool           = True,
    use_human_query:   bool           = True,
    use_negotiation:   bool           = True,   # False → Phase 4 skip (ablation)
    label:             str            = "FULL",
    verbose:           str            = "full",
) -> Dict:
    """
    P2P Collaborative VLM Planning 전체 파이프라인을 실행한다.

    Args:
        task             : 글로벌 태스크 (None이면 대화형 선택)
        img_a            : agent_A 이미지 경로
        img_b            : agent_B 이미지 경로
        use_offer        : False → offer 없이 방 타입만으로 플래닝 (ablation)
        use_handoff      : False → handoff 선언 없이 플래닝 (ablation)
        use_human_query  : False → Phase 6 skip (ablation)
        use_negotiation  : False → Phase 4 skip, 바로 finalize (ablation)
        label            : 실험 레이블 (출력용)
        verbose          : "full" | "summary" | "minimal"

    Returns:
        실험 결과 전체를 담은 dict
    """
    # ── 입력 수집 ────────────────────────────────────────────────────────────
    if not task:
        task = _select_task()
    if not img_a:
        img_a = _select_image("Agent A")
    if not img_b:
        img_b = _select_image("Agent B")

    # ── 헤더 출력 ────────────────────────────────────────────────────────────
    print("\n" + "█" * 68)
    print(f"  P2P COLLABORATIVE VLM PLANNING — {label}")
    print("█" * 68)
    print(f"  Task   : {task}")
    print(f"  Img A  : {Path(img_a).name}")
    print(f"  Img B  : {Path(img_b).name}")
    print(f"  Flags  : offer={use_offer} | handoff={use_handoff} | "
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
        from dataclasses import asdict
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
        for e in vr.errors:
            print(f"    ✗ {e}")
    if vr.warnings:
        for w in vr.warnings:
            print(f"    ⚠ {w}")

    # ── Final Output ──────────────────────────────────────────────────────────
    print("\n" + "█" * 68)
    print(f"  FINAL JOINT PLAN — {label}")
    print("█" * 68)
    print(format_final_plan(joint))
    print_scores(vr, label)

    return {
        "label":        label,
        "task":         task,
        "flags": {
            "use_offer":       use_offer,
            "use_handoff":     use_handoff,
            "use_negotiation": use_negotiation,
            "use_human_query": use_human_query,
        },
        "offers":         {"agent_A": offer_to_dict(offer_a), "agent_B": offer_to_dict(offer_b)},
        "local_plans":    {"agent_A": local_plan_to_dict(plan_a), "agent_B": local_plan_to_dict(plan_b)},
        "conflicts":      [asdict(c) for c in conflicts],
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
            "converged":         convergence.converged,
            "pass_matched":      convergence.pass_matched,
            "no_dep_cycle":      convergence.no_dep_cycle,
            "observability_ok":  convergence.observability_ok,
            "unresolved":        len(convergence.unresolved_conflicts),
        },
        "human_answers":  human_answers,
        "hq_triggers":    hq_triggers,
        "hq_asked":       hq_asked,
        "joint_plan":     joint,
        "verification":   asdict(vr),
        "scores": {
            "completeness":  vr.completeness_score,
            "executability": vr.executability_score,
            "observability": vr.observability_score,
            "handoff":       vr.handoff_score,
            "sequential":    vr.sequential_score,
            "total":         vr.total_score,
        },
    }


# ── CLI 진입점 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run()
