# main.py

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from p2p_phases import (
    local_plan_to_dict, offer_to_dict,
    phase1_offer, phase2_local_plan,
    phase3_conflict_detection, phase4_negotiation,
    phase5_convergence_check, phase6_human_query,
    phase_finalize, plan_steps_to_dicts,
)
from p2p_utils import _banner, format_joint_plan, jdump

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
        print(f"  {t['id']:<12} {desc[:55]}{'...' if len(desc)>55 else ''}")
    print(f"{'─'*60}\n")


def get_task(task_id: str) -> str:
    for t in _load_tasks():
        if t["id"] == task_id:
            return t["description"]
    raise KeyError(f"task_id '{task_id}' not found. Run list_tasks().")


def run(
    task_id:         Optional[str] = None,
    img_a:           Optional[str] = None,
    img_b:           Optional[str] = None,
    use_offer:       bool = True,
    use_negotiation: bool = True,
    use_human_query: bool = True,
    label:           Optional[str] = None,
    verbose:         str  = "full",
) -> Dict:
    if task_id is None:
        list_tasks()
        raise ValueError("task_id를 지정해주세요.")
    if not img_a or not img_b:
        raise ValueError("img_a, img_b 경로를 지정해주세요.")
    if not Path(img_a).exists():
        raise FileNotFoundError(f"img_a not found: {img_a}")
    if not Path(img_b).exists():
        raise FileNotFoundError(f"img_b not found: {img_b}")

    task  = get_task(task_id)
    label = label or task_id

    print("\n" + "█" * 68)
    print(f"  P2P COLLABORATIVE VLM PLANNING — {label}")
    print("█" * 68)
    print(f"  Task  : {task[:80]}{'...' if len(task)>80 else ''}")
    print(f"  Flags : offer={use_offer} | negotiation={use_negotiation} | hq={use_human_query}")

    # Phase 1
    offer_a, offer_b = phase1_offer(img_a, img_b, task, verbose=verbose)

    # Phase 2 + 2b
    plan_a, plan_b = phase2_local_plan(
        offer_a, offer_b, img_a, img_b, task,
        use_offer=use_offer, verbose=verbose,
    )

    # Phase 3
    conflicts          = phase3_conflict_detection(plan_a, plan_b, offer_a, offer_b, verbose=verbose)
    n_conflicts_before = len(conflicts)

    # Phase 4
    if use_negotiation and conflicts:
        neg_a, neg_b, neg_rounds = phase4_negotiation(
            plan_a, plan_b, offer_a, offer_b, conflicts,
            img_a, img_b, task, verbose=verbose,
        )
    else:
        if not conflicts:
            _banner("PHASE 4 — P2P NEGOTIATION")
            print("  No conflicts → skip.")
        else:
            _banner("PHASE 4 — P2P NEGOTIATION")
            print("  [ABLATION] skipped.")
        neg_a      = plan_steps_to_dicts(plan_a.steps)
        neg_b      = plan_steps_to_dicts(plan_b.steps)
        neg_rounds = []

    # Phase 5
    convergence = phase5_convergence_check(neg_a, neg_b, offer_a, offer_b, conflicts)

    # Phase 6
    human_answers, hq_triggers, hq_asked = phase6_human_query(
        plan_a, plan_b, offer_a, offer_b,
        convergence, img_a, img_b,
        use_human_query=use_human_query,
    )

    # Finalize
    joint = phase_finalize(
        neg_a, neg_b, offer_a, offer_b,
        human_answers, convergence, verbose=verbose,
    )

    # 출력
    print("\n" + "█" * 68)
    print(f"  FINAL JOINT PLAN — {label}")
    print("█" * 68)
    print(format_joint_plan(joint))

    # 메트릭
    n_after   = len(convergence.unresolved_conflicts)
    cross_deps = sum(
        1 for s in joint
        for dep in s.get("depends_on", [])
        if any(s2["step_id"] == dep and s2["agent_id"] != s["agent_id"] for s2 in joint)
    )
    metrics = {
        "conflicts_before":   n_conflicts_before,
        "conflicts_after":    n_after,
        "conflict_reduction": round(1 - n_after / max(n_conflicts_before, 1), 3),
        "convergence_rate":   1.0 if convergence.converged else 0.0,
        "negotiation_rounds": len(neg_rounds),
        "cross_agent_deps":   cross_deps,
        "hq_triggered":       len(hq_triggers),
        "hq_asked":           len(hq_asked),
    }

    print(f"\n  METRICS")
    print(f"  {'─'*40}")
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")
    print(f"  Converged: {'YES ✓' if convergence.converged else 'NO ✗'}")

    return {
        "label":    label,
        "task_id":  task_id,
        "task":     task,
        "flags": {
            "use_offer":       use_offer,
            "use_negotiation": use_negotiation,
            "use_human_query": use_human_query,
        },
        "offers":      {"agent_A": offer_to_dict(offer_a), "agent_B": offer_to_dict(offer_b)},
        "local_plans": {"agent_A": local_plan_to_dict(plan_a), "agent_B": local_plan_to_dict(plan_b)},
        "conflicts":   [asdict(c) for c in conflicts],
        "negotiation": {
            "rounds":  len(neg_rounds),
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
            "no_dep_cycle":     convergence.no_dep_cycle,
            "observability_ok": convergence.observability_ok,
            "no_missing_deps":  convergence.no_missing_deps,
            "unresolved":       len(convergence.unresolved_conflicts),
        },
        "human_answers": human_answers,
        "joint_plan":    joint,
        "metrics":       metrics,
    }
