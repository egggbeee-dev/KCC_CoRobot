# p2p_main_nagent.py
# N-agent 파이프라인 진입점. 기존 p2p_main.get_task()(task 로더)는 그대로 재사용한다.

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import p2p_phases as _p2
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
    task: Optional[str] = None,
    images: Optional[List[str]] = None,
    n_agents: Optional[int] = None,
    use_offer: bool = True,          # 현재 offer 생성은 항상 수행 (ablation 옵션은 추후 필요시 확장)
    use_negotiation: bool = True,
    label: Optional[str] = None,
    verbose: str = "full",
) -> Dict:
    """
    N-agent P2P 협업 플래닝 파이프라인. N=2, N=4 둘 다 이 함수 하나로 지원한다.

    task     : task 설명을 문자열로 직접 전달. 지정하면 tasks.json 조회 없이 바로 이 텍스트를 쓴다.
               예: run_n(task="Prepare the kitchen and bedroom for a movie night.", images=[...])
    task_id  : tasks.json에 등록된 task를 쓰고 싶을 때만 사용 (task를 직접 넘기면 무시됨).
    images   : agent 수만큼의 이미지 경로 리스트. 순서대로 agent_A, agent_B, ... 에 매핑된다.
               N=2면 이미지 2장, N=4면 이미지 4장을 넣으면 된다.
    n_agents : 생략하면 len(images)로 자동 결정. 명시하면 images 개수와 일치해야 함.
    """
    if not images:
        raise ValueError("images 리스트를 지정해주세요. 예: images=['room1.jpg','room2.jpg', ...]")

    if task:
        # task 텍스트를 직접 받은 경우: tasks.json 조회 없이 바로 사용
        task_id = task_id or "custom_task"
    else:
        if task_id is None:
            list_tasks()
            raise ValueError("task 또는 task_id 중 하나는 지정해주세요.")
        task = get_task(task_id)

    n_agents = n_agents or len(images)
    if len(images) != n_agents:
        raise ValueError(f"images 개수({len(images)})가 n_agents({n_agents})와 일치해야 합니다.")
    for img in images:
        if not Path(img).exists():
            raise FileNotFoundError(f"image not found: {img}")

    agent_ids = make_agent_ids(n_agents)
    images_map = dict(zip(agent_ids, images))
    label = label or task_id

    print("\n" + "#" * 68)
    print(f"  N-AGENT P2P PLANNING (N={n_agents}) - {label}")
    print("#" * 68)
    print(f"  Task    : {task[:80]}{'...' if len(task) > 80 else ''}")
    print(f"  Agents  : {agent_ids}")
    print(f"  Images  : {images}")
    print(f"  Flags   : negotiation={use_negotiation}")

    # ── PHASE 1: OFFER (broadcast, agent별 이미지) ────────────────────────
    offers = phase1_offer_n(images, agent_ids, task, verbose=verbose)

    # ── PHASE 2: LOCAL PLAN (peer-aware) + rule-based PASS sync ──────────
    plans = phase2_local_plan_n(offers, images, agent_ids, task, verbose=verbose)

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
