# ══════════════════════════════════════════════════════════════════════════════
# ablation.py
# Ablation 실험: 컴포넌트별 기여도 측정 (P2P 파이프라인용)
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import Dict, List, Optional


def run_ablation(
    task_id: str,
    img_a:   str,
    img_b:   str,
) -> List[Dict]:
    """
    5가지 ablation 설정을 순서대로 실행하고 결과를 비교한다.

    설정:
        Full (Ours)       : 모든 컴포넌트 활성화
        w/o Negotiation   : Phase 4 P2P 협상 비활성화
        w/o Human Query   : Phase 6 비활성화
        w/o Handoff       : handoff 선언 비활성화
        w/o Offer         : offer 없이 방 타입 정보만 전달

    Args:
        task_id : tasks.json의 ID (예: "task_003")
        img_a   : agent_A 이미지 경로
        img_b   : agent_B 이미지 경로

    Returns:
        각 설정별 run() 결과 dict 리스트

    사용 예:
        from p2p_ablation import run_ablation
        results = run_ablation(
            task_id = "task_003",
            img_a   = "/content/repo/Data/Room/Kitchens/real/k1.jpg",
            img_b   = "/content/repo/Data/Room/Livingrooms/real/l1.jpg",
        )
    """
    from p2p_main import run

    configs = [
        # (label,              use_offer, use_handoff, use_negotiation, use_hq)
        ("Full (Ours)",        True,  True,  True,  True),
        ("w/o Negotiation",    True,  True,  False, True),
        ("w/o Human Query",    True,  True,  True,  False),
        ("w/o Handoff",        True,  False, True,  True),
        ("w/o Offer",          False, True,  True,  True),
    ]

    results = []
    for label, use_offer, use_handoff, use_neg, use_hq in configs:
        result = run(
            task_id         = task_id,
            img_a           = img_a,
            img_b           = img_b,
            use_offer       = use_offer,
            use_handoff     = use_handoff,
            use_negotiation = use_neg,
            use_human_query = use_hq,
            label           = label,
            verbose         = "minimal",
        )
        results.append(result)

    _print_summary(results)
    return results


def _print_summary(results: List[Dict]):
    print("\n" + "█" * 80)
    print("  ABLATION SUMMARY")
    print("█" * 80)
    print(f"  {'Method':<22} {'Comp':>5} {'Exec':>5} {'Obs':>5} {'Hand':>5} {'Seq':>5} {'Total':>6} {'Conv':>5} {'Rnd':>4}")
    print(f"  {'─'*66}")
    for r in results:
        sc   = r["scores"]
        conv = r["convergence"]
        print(
            f"  {r['label']:<22} "
            f"{sc['completeness']:>5.2f} {sc['executability']:>5.2f} "
            f"{sc['observability']:>5.2f} {sc['handoff']:>5.2f} {sc['sequential']:>5.2f} "
            f"{sc['total']:>6.3f} "
            f"{'Y' if conv['converged'] else 'N':>5} "
            f"{r['negotiation']['rounds']:>4}"
        )
    print()
