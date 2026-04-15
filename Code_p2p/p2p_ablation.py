# ablation.py

from __future__ import annotations
from typing import Dict, List


def run_ablation(task_id: str, img_a: str, img_b: str) -> List[Dict]:
    """
    4가지 ablation 설정 순서대로 실행.

    Full (Ours)     : 모든 컴포넌트 활성화
    w/o Negotiation : Phase 4 협상 비활성화
    w/o Human Query : Phase 6 비활성화
    w/o Offer       : offer 없이 방 타입만으로 플래닝
    """
    from p2p_main import run

    configs = [
        # (label,           use_offer, use_negotiation, use_hq)
        ("Full (Ours)",     True,  True,  True),
        ("w/o Negotiation", True,  False, True),
        ("w/o Human Query", True,  True,  False),
        ("w/o Offer",       False, True,  True),
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


def _print_summary(results: List[Dict]) -> None:
    print("\n" + "█" * 80)
    print("  ABLATION SUMMARY")
    print("█" * 80)
    print(
        f"  {'Method':<22} "
        f"{'Conf↓':>6} {'Conf_R':>7} {'Conv':>5} "
        f"{'Rnd':>4} {'XDep':>5} {'HQ':>4}"
    )
    print(f"  {'─'*60}")
    for r in results:
        m    = r.get("metrics", {})
        conv = r.get("convergence", {})
        print(
            f"  {r['label']:<22} "
            f"{m.get('conflicts_before', 0):>6} "
            f"{m.get('conflict_reduction', 0.0):>7.3f} "
            f"{'Y' if conv.get('converged') else 'N':>5} "
            f"{m.get('negotiation_rounds', 0):>4} "
            f"{m.get('cross_agent_deps', 0):>5} "
            f"{m.get('hq_asked', 0):>4}"
        )
    print()
    print("  Columns:")
    print("    Conf↓  : conflicts before negotiation")
    print("    Conf_R : conflict reduction rate")
    print("    Conv   : Phase 5 convergence (Y/N)")
    print("    Rnd    : negotiation rounds used")
    print("    XDep   : cross-agent depends_on count")
    print("    HQ     : human queries asked")
