# p2p_main_nagent.py
# N-agent 파이프라인 진입점. 기존 p2p_main.get_task()(task 로더)는 그대로 재사용한다.

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import p2p_phase as _p2
from p2p_config_nagent import make_agent_ids
from p2p_main import get_task, list_tasks
from p2p_phase_nagent import (
    format_joint_plan_n,
    phase1_offer_n,
    phase2_local_plan_n,
    phase3_conflict_detection_n,
    phase4_negotiation_n,
    phase5_convergence_check_n,
    phase_finalize_n,
)


def run_n(
    task_id: Optional[str] = None,
    image: Optional[str] = None,
    n_agents: int = 2,
    room_labels: Optional[List[str]] = None,
    use_offer: bool = True,          # 현재 offer 생성은 항상 수행 (ablation 옵션은 추후 필요시 확장)
    use_negotiation: bool = True,
    label: Optional[str] = None,
    verbose: str = "full",
) -> Dict:
    """
    N-agent P2P 협업 플래닝 파이프라인. N=2, N=4 둘 다 이 함수 하나로 지원한다.

    image       : 단일 이미지 경로. 모든 agent가 이 이미지를 공유해서 입력받고,
                  프롬프트 상에서 담당 구역(room_labels)만 다르게 지시받는다.
    n_agents    : agent 수. 2 또는 4 (혹은 그 외 임의의 N)를 그대로 지정하면 된다.
    room_labels : 길이 n_agents인 구역 이름 리스트. 생략하면 "Zone 1".."Zone N"으로 자동 생성.
                  예: n_agents=2 -> ["kitchen","living room"], n_agents=4 -> ["kitchen","bedroom","bathroom","living room"]
    """
    if task_id is None:
        list_tasks()
        raise ValueError("task_id를 지정해주세요.")
    if not image:
        raise ValueError("image 경로를 지정해주세요. 예: image='room.jpg'")
    if not Path(image).exists():
        raise FileNotFoundError(f"image not found: {image}")
    if n_agents < 2:
        raise ValueError("n_agents는 2 이상이어야 합니다.")
    if room_labels is not None and len(room_labels) != n_agents:
        raise ValueError(f"room_labels 개수({len(room_labels)})가 n_agents({n_agents})와 일치해야 합니다.")

    agent_ids = make_agent_ids(n_agents)
    images_map = {aid: image for aid in agent_ids}  # 전부 같은 이미지 공유
    task = get_task(task_id)
    label = label or task_id

    print("\n" + "#" * 68)
    print(f"  N-AGENT P2P PLANNING (N={n_agents}) - {label}")
    print("#" * 68)
    print(f"  Task    : {task[:80]}{'...' if len(task) > 80 else ''}")
    print(f"  Agents  : {agent_ids}")
    print(f"  Image   : {image} (shared by all {n_agents} agents)")
    print(f"  Flags   : negotiation={use_negotiation}")

    # ── PHASE 1: OFFER (broadcast, shared image) ─────────────────────────
    offers = phase1_offer_n(image, agent_ids, task, room_labels=room_labels, verbose=verbose)

    # ── PHASE 2: LOCAL PLAN (peer-aware) + rule-based PASS sync ──────────
    plans = phase2_local_plan_n(offers, image, agent_ids, task, verbose=verbose)

    # ── PHASE 3: DECENTRALIZED CONFLICT DETECTION (all pairs) ───────────
    all_conflicts, conflicts_by_pair = phase3_conflict_detection_n(
        plans, offers, agent_ids, verbose=verbose)

    # ── PHASE 4: P2P NEGOTIATION (conflict-pair scoped) ──────────────────
    cur_steps, neg_rounds_by_pair = phase4_negotiation_n(
        plans, offers, images_map, conflicts_by_pair, agent_ids, task,
        use_negotiation=use_negotiation, verbose=verbose,
    )

    # ── PHASE 5: CONVERGENCE CHECK ────────────────────────────────────────
    convergence = phase5_convergence_check_n(cur_steps, offers, agent_ids)

    # ── FINALIZE: MERGE ──────────────────────────────────────────────────
    joint = phase_finalize_n(cur_steps, agent_ids, human_answers={}, verbose=verbose)

    print("\n" + "#" * 68)
    print(f"  FINAL JOINT PLAN - {label}")
    print("#" * 68)
    print(format_joint_plan_n(joint, agent_ids, task))

    n_total_pairs = n_agents * (n_agents - 1) // 2
    metrics = {
        "n_agents": n_agents,
        "total_pairs": n_total_pairs,
        "conflicting_pairs": len(conflicts_by_pair),
        "total_conflicts": len(all_conflicts),
        "conflicts_after": len(convergence.unresolved_conflicts),
        "negotiation_rounds_total": sum(len(r) for r in neg_rounds_by_pair.values()),
        "converged": convergence.converged,
    }
    print(f"\n  METRICS")
    print(f"  {'-'*40}")
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")

    return {
        "label": label,
        "task_id": task_id,
        "task": task,
        "n_agents": n_agents,
        "agent_ids": agent_ids,
        "flags": {"use_negotiation": use_negotiation},
        "offers": {aid: _p2.offer_to_dict(offers[aid]) for aid in agent_ids},
        "conflicts": [asdict(c) for c in all_conflicts],
        "conflicting_pairs": [f"{a}-{b}" for (a, b) in conflicts_by_pair],
        "negotiation": {
            f"{a}-{b}": [
                {
                    "round_num": r.round_num,
                    "proposals_x": [asdict(p) for p in r.proposals_a],
                    "proposals_y": [asdict(p) for p in r.proposals_b],
                    "locked_step_ids": r.locked_step_ids,
                }
                for r in rounds
            ]
            for (a, b), rounds in neg_rounds_by_pair.items()
        },
        "convergence": {
            "converged": convergence.converged,
            "no_dep_cycle": convergence.no_dep_cycle,
            "observability_ok": convergence.observability_ok,
            "no_missing_deps": convergence.no_missing_deps,
            "unresolved": len(convergence.unresolved_conflicts),
        },
        "joint_plan": joint,
        "metrics": metrics,
    }
