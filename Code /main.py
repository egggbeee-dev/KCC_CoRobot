# ══════════════════════════════════════════════════════════════════════════════
# main.py
# 진입점: run() — 단일 실험 실행
#
# 사용 예 (Colab / 스크립트):
#   from main import run
#   result = run()
#
#   # ablation 실험:
#   from ablation import run_ablation
#   results = run_ablation()
#
#   # LLM-as-Judge 평가:
#   from evaluator import evaluate, print_evaluation
#   eval_result = evaluate(result, api_key="sk-...")
#   print_evaluation(eval_result, label=result["label"])
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import asdict
from typing import Dict

from config import IMAGE_A_PATH, IMAGE_B_PATH, TASK
from phases import (
    local_plan_to_dict,
    offer_to_dict,
    phase1_offer,
    phase2_leader,
    phase3_local_plan,
    phase4a_human_query,
    phase4b_joint_plan,
)
from utils import _banner, _log, jdump
from verifier import format_final_plan, print_scores, verify


def run(
    task:                str  = TASK,
    img_a:               str  = IMAGE_A_PATH,
    img_b:               str  = IMAGE_B_PATH,
    use_offer:           bool = True,
    use_leader_election: bool = True,
    use_handoff:         bool = True,
    use_human_query:     bool = True,
    label:               str  = "FULL",
    verbose:             str  = "full",
) -> Dict:
    """
    Collaborative VLM Planning 전체 파이프라인을 실행한다.

    Args:
        task                : 글로벌 태스크 설명 문자열
        img_a               : agent_A 가 관찰하는 방의 이미지 경로
        img_b               : agent_B 가 관찰하는 방의 이미지 경로
        use_offer           : False → offer 없이 방 타입만으로 플래닝 (ablation)
        use_leader_election : False → agent_A 고정 리더 (ablation)
        use_handoff         : False → handoff 선언 없이 플래닝 (ablation)
        use_human_query     : False → Phase 4a 스킵 (ablation)
        label               : 실험 레이블 (출력용)
        verbose             : "full" | "summary" | "minimal"
                              full    — raw JSON + parsed 전부 출력 (디버깅용)
                              summary — parsed 결과만 출력, raw 생략
                              minimal — 각 phase 헤더 + 최종 플랜만 출력

    Returns:
        실험 결과 전체를 담은 dict
    """
    print("\n" + "█" * 68)
    print(f"  COLLABORATIVE VLM PLANNING — {label}")
    print("█" * 68)
    print(f"  Task  : {task}")
    print(f"  Flags : offer={use_offer} | leader={use_leader_election} | handoff={use_handoff} | hq={use_human_query} | verbose={verbose}")

    # ── Phase 1: Observation & Offer ─────────────────────────────────────────
    offer_a, offer_b = phase1_offer(img_a, img_b, task, verbose=verbose)

    # ── Phase 2: Leader Election ──────────────────────────────────────────────
    leader = phase2_leader(offer_a, offer_b, use_leader_election)

    # ── Phase 3: Local Planning ───────────────────────────────────────────────
    plan_a, plan_b = phase3_local_plan(
        offer_a, offer_b, leader, img_a, img_b, task,
        use_offer, use_handoff, verbose=verbose,
    )

    # ── Phase 4a: Human Query ─────────────────────────────────────────────────
    human_answers, hq_triggers, hq_asked = phase4a_human_query(
        plan_a, plan_b, offer_a, offer_b, leader.leader_id, use_human_query
    )

    # ── Phase 4b: Joint Planning ──────────────────────────────────────────────
    joint, validity = phase4b_joint_plan(
        plan_a, plan_b, offer_a, offer_b, leader,
        human_answers, task, img_a, img_b, verbose=verbose,
    )

    # ── Verification ──────────────────────────────────────────────────────────
    vr = verify(joint, offer_a, offer_b)

    _banner("VERIFICATION")
    print(f"  is_valid : {vr.is_valid}")
    print(f"  errors   : {vr.errors}")
    print(f"  warnings : {vr.warnings}")

    # ── Final Output ──────────────────────────────────────────────────────────
    print("\n" + "█" * 68)
    print(f"  FINAL JOINT PLAN — {label}")
    print("█" * 68)
    print(format_final_plan(joint))
    print(f"\n  Valid={vr.is_valid} | Errors={len(vr.errors)} | Warnings={len(vr.warnings)}")
    for e in vr.errors:   print(f"    ✗ {e}")
    for w in vr.warnings: print(f"    ⚠ {w}")
    print_scores(vr, label)

    return {
        "label": label,
        "task":  task,
        "flags": {
            "use_offer":           use_offer,
            "use_leader_election": use_leader_election,
            "use_handoff":         use_handoff,
            "use_human_query":     use_human_query,
        },
        "leader":       asdict(leader),
        "offers":       {"agent_A": offer_to_dict(offer_a), "agent_B": offer_to_dict(offer_b)},
        "local_plans":  {"agent_A": local_plan_to_dict(plan_a), "agent_B": local_plan_to_dict(plan_b)},
        "human_answers": human_answers,
        "hq_triggers":  hq_triggers,
        "hq_asked":     hq_asked,
        "joint_plan":   joint,
        "validity":     validity,
        "verification": asdict(vr),
        "scores": {
            "completeness": vr.completeness_score,
            "executability": vr.executability_score,
            "observability": vr.observability_score,
            "handoff":       vr.handoff_score,
            "sequential":    vr.sequential_score,
            "total":         vr.total_score,
        },
    }

if __name__ == "__main__":
    result = run()
