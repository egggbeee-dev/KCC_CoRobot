# ablation.py
# Ablation 실험: 컴포넌트별 기여도 측정

from __future__ import annotations

from typing import Dict, List


def run_ablation(
    task_id: str,
    img_a:   str,
    img_b:   str,
) -> List[Dict]:
    """
    5가지 ablation 설정을 순서대로 실행하고 결과를 비교한다.

    설정:
        Full (Ours)     : 모든 컴포넌트 활성화
        w/o Negotiation : Phase 4 협상 비활성화 → 협상의 conflict 감소 기여 측정
        w/o Human Query : Phase 6 비활성화 → deferred HQ 기여 측정
        w/o Handoff     : handoff 선언 비활성화 → PASS/INFORM 기여 측정
        w/o Offer       : offer 없이 방 타입 정보만 전달 → offer exchange 기여 측정

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
        # (label,              use_offer, use_negotiation, use_hq)
        ("Full (Ours)",        True,  True,  True),
        ("w/o Negotiation",    True,  False, True),
        ("w/o Human Query",    True,  True,  False),
        ("w/o Offer",          False, True,  True),
    ]

    results = []
    for label, use_offer, use_neg, use_hq in configs:
        result = run(
            task_id         = task_id,
            img_a           = img_a,
            img_b           = img_b,
            use_offer       = use_offer,
            use_negotiation = use_neg,
            use_human_query = use_hq,
            label           = label,
            verbose         = "minimal",
        )
        results.append(result)

    _print_summary(results)
    return results


def _print_summary(results: List[Dict]):
    print("\n" + "█" * 90)
    print("  ABLATION SUMMARY")
    print("█" * 90)
    print(
        f"  {'Method':<22} "
        f"{'Conf↓':>6} {'Conf_R':>6} {'Conv':>5} "
        f"{'Rnd':>4} {'Obs':>5} {'Hand':>5} {'HQ':>4}"
    )
    print(f"  {'─'*72}")
    for r in results:
        m    = r["metrics"]
        conv = r["convergence"]
        print(
            f"  {r['label']:<22} "
            f"{m['conflicts_before']:>6} "
            f"{m['conflict_reduction']:>6.2f} "
            f"{'Y' if conv['converged'] else 'N':>5} "
            f"{m['negotiation_rounds']:>4} "
            f"{m['observability_rate']:>5.2f} "
            f"{m['handoff_match_rate']:>5.2f} "
            f"{m['hq_asked']:>4}"
        )
    print()
    print("  Columns:")
    print("    Conf↓   : conflicts detected before negotiation")
    print("    Conf_R  : conflict reduction rate after negotiation")
    print("    Conv    : Phase 5 convergence (Y/N)")
    print("    Rnd     : negotiation rounds used")
    print("    Obs     : observability compliance rate")
    print("    Hand    : PASS handoff match rate")
    print("    HQ      : human queries asked")
    print()
