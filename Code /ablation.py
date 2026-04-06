# ══════════════════════════════════════════════════════════════════════════════
# ablation.py
# Ablation 실험: 컴포넌트별 기여도 측정
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import Dict, List

from config import IMAGE_A_PATH, IMAGE_B_PATH, TASK


def run_ablation(
    task:  str = TASK,
    img_a: str = IMAGE_A_PATH,
    img_b: str = IMAGE_B_PATH,
) -> List[Dict]:
    """
    5가지 ablation 설정을 순서대로 실행하고 결과를 비교한다.

    설정:
        Full (Ours)          : 모든 컴포넌트 활성화
        w/o Human Query      : Phase 4a 비활성화
        w/o Handoff          : handoff 선언 비활성화
        w/o Leader Election  : agent_A 고정 리더
        w/o Offer            : offer 없이 방 타입 정보만 전달

    Returns:
        각 설정별 run() 결과 dict 리스트
    """
    # 순환 임포트 방지를 위해 함수 내부에서 임포트
    from main import run

    configs = [
        # (label,                use_offer, use_leader, use_handoff, use_hq)
        ("Full (Ours)",          True,  True,  True,  True),
        ("w/o Human Query",      True,  True,  True,  False),
        ("w/o Handoff",          True,  True,  False, True),
        ("w/o Leader Election",  True,  False, True,  True),
        ("w/o Offer",            False, True,  True,  True),
    ]

    results = []
    for label, use_offer, use_leader, use_handoff, use_hq in configs:
        result = run(
            task=task, img_a=img_a, img_b=img_b,
            use_offer=use_offer, use_leader_election=use_leader,
            use_handoff=use_handoff, use_human_query=use_hq,
            label=label,
            verbose="minimal",
        )
        results.append(result)

    _print_summary(results)
    return results


def _print_summary(results: List[Dict]):
    print("\n" + "█" * 68)
    print("  ABLATION SUMMARY")
    print("█" * 68)
    print(f"  {'Method':<25} {'Comp':>5} {'Exec':>5} {'Obs':>5} {'HQ':>5} {'Seq':>5} {'Total':>6}")
    print(f"  {'─'*58}")
    for r in results:
        sc = r["scores"]
        print(
            f"  {r['label']:<25} {sc['completeness']:>5.2f} {sc['executability']:>5.2f} "
            f"{sc['observability']:>5.2f} {sc['handoff']:>5.2f} {sc['sequential']:>5.2f} {sc['total']:>6.3f}"
        )
