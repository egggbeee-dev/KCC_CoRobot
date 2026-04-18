# p2p_main.py
# Pipeline:
#   OBSERVATION   : offer + draft plan (VLM × 2)
#   COORDINATION  : mutual awareness final plan (VLM × 2)
#   HANDOFF SYNC  : rule-based PASS coordination
#   CONFLICT CHECK: conflict detection
#   NEGOTIATION   : P2P negotiation (VLM × up to 6)
#   PLAN QUALITY  : quality check
#   HUMAN QUERY   : human clarification + VLM polish (VLM × 1 + 2, rare)
#   MERGE         : final joint plan

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from p2p_phases import (
    format_joint_plan,
    local_plan_to_dict,
    offer_to_dict,
    observe_and_draft,
    coordinate,
    phase3_conflict_detection,
    phase4_negotiation,
    phase5_convergence_check,
    phase6_human_query,
    phase_finalize,
    plan_steps_to_dicts,
    _kw,
)
from p2p_utils import compute_joint_uncertainty, _banner, jdump

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
        print(f"  {t['id']:<12} {desc[:55]}{'...' if len(desc) > 55 else ''}")
    print(f"{'─'*60}\n")


def get_task(task_id: str) -> str:
    tasks = _load_tasks()
    for t in tasks:
        if t["id"] == task_id:
            return t["description"]
    available = [t["id"] for t in tasks]
    raise KeyError(f"task_id '{task_id}' not found. Available: {available}")


def run(
    task_id:         Optional[str] = None,
    img_a:           Optional[str] = None,
    img_b:           Optional[str] = None,
    use_offer:       bool = True,
    use_negotiation: bool = True,
    use_human_query: bool = True,
    use_handoff:     bool = True,
    label:           Optional[str] = None,
    verbose:         str  = "full",
) -> Dict:
    if task_id is None:
        list_tasks()
        raise ValueError("task_id를 지정해주세요.")

    task  = get_task(task_id)
    label = label or task_id

    if not img_a or not img_b:
        raise ValueError("img_a와 img_b 경로를 모두 지정해주세요.")
    if not Path(img_a).exists():
        raise FileNotFoundError(f"img_a not found: {img_a}")
    if not Path(img_b).exists():
        raise FileNotFoundError(f"img_b not found: {img_b}")

    print("\n" + "█" * 68)
    print(f"  P2P COLLABORATIVE VLM PLANNING — {label}")
    print("█" * 68)
    print(f"  Task    : {task[:80]}{'...' if len(task) > 80 else ''}")
    print(f"  Flags   : offer={use_offer} | negotiation={use_negotiation} | hq={use_human_query}")

    # ── OBSERVATION: offer + draft (VLM × 2) ─────────────────────────────────
    offer_a, offer_b, draft_a, draft_b = observe_and_draft(
        img_a, img_b, task, verbose=verbose,
    )

    # ── COORDINATION: mutual awareness final plan (VLM × 2) ──────────────────
    plan_a, plan_b = coordinate(
        offer_a, offer_b, draft_a, draft_b,
        img_a, img_b, task,
        use_offer=use_offer, verbose=verbose,
    )

    # ── CONFLICT CHECK ────────────────────────────────────────────────────────
    conflicts   = phase3_conflict_detection(plan_a, plan_b, offer_a, offer_b, verbose=verbose)
    n_conflicts = len(conflicts)

    # ── NEGOTIATION: P2P (VLM × up to 6) ─────────────────────────────────────
    if use_negotiation:
        neg_steps_a, neg_steps_b, neg_rounds = phase4_negotiation(
            plan_a, plan_b, offer_a, offer_b, conflicts,
            img_a, img_b, task, verbose=verbose,
        )
    else:
        _banner("NEGOTIATION — P2P")
        print("  [ABLATION] Negotiation disabled.")
        neg_steps_a = plan_steps_to_dicts(plan_a.steps)
        neg_steps_b = plan_steps_to_dicts(plan_b.steps)
        neg_rounds  = []

    # ── PLAN QUALITY CHECK ────────────────────────────────────────────────────
    convergence = phase5_convergence_check(
        neg_steps_a, neg_steps_b, offer_a, offer_b, conflicts,
    )

    # ── HUMAN QUERY (rare) + VLM polish ──────────────────────────────────────
    human_answers, hq_triggers, hq_asked = phase6_human_query(
        plan_a, plan_b, offer_a, offer_b,
        img_a, img_b,
        task=task,
        use_human_query=use_human_query,
        unresolved_conflicts=convergence.unresolved_conflicts,
    )

    # ── MERGE ─────────────────────────────────────────────────────────────────
    joint = phase_finalize(
        neg_steps_a, neg_steps_b,
        offer_a, offer_b,
        human_answers,
        verbose=verbose,
    )

    # ── 메트릭 ────────────────────────────────────────────────────────────────
    conflict_reduction = (
        (n_conflicts - len(convergence.unresolved_conflicts)) / max(n_conflicts, 1)
    )

    scope_a  = set(re.findall(r"\w+", offer_a.obs_scope.lower()))
    scope_b  = set(re.findall(r"\w+", offer_b.obs_scope.lower()))
    can_kw_a: set = set()
    can_kw_b: set = set()
    for cd in offer_a.can_do: can_kw_a |= _kw(cd)
    for cd in offer_b.can_do: can_kw_b |= _kw(cd)

    obs_violations = 0
    for s in joint:
        if s.get("handoff_type") == "PASS":
            continue
        pool = (can_kw_a | scope_a) if s.get("agent_id") == "agent_A" else (can_kw_b | scope_b)
        kw   = _kw(s.get("action", ""))
        if kw and pool and not (kw & pool):
            obs_violations += 1
    observability_rate = round(1.0 - obs_violations / max(len(joint), 1), 3)

    id_to_agent = {s["step_id"]: s.get("agent_id") for s in joint}
    cross_deps  = sum(
        1 for s in joint
        for d in s.get("depends_on", [])
        if id_to_agent.get(d) and id_to_agent[d] != s.get("agent_id")
    )

    pass_steps   = {s["step_id"] for s in joint if s.get("handoff_type") == "PASS"}
    all_deps     = {d for s in joint for d in s.get("depends_on", [])}
    matched_pass = len(pass_steps & all_deps)
    handoff_match = matched_pass / max(len(pass_steps), 1) if pass_steps else 1.0

    U_joint = compute_joint_uncertainty(joint)

    metrics = {
        "handoff_match_rate":  round(handoff_match, 3),
        "cross_agent_deps":    cross_deps,
        "conflict_reduction":  round(conflict_reduction, 3),
        "observability_rate":  observability_rate,
        "U_joint":             U_joint,
        "negotiation_rounds":  len(neg_rounds),
        "hq_triggered":        len(hq_triggers),
        "hq_asked":            len(hq_asked),
        "conflicts":           n_conflicts,
        "conflicts_after":     len(convergence.unresolved_conflicts),
    }

    print("\n" + "█" * 68)
    print(f"  FINAL JOINT PLAN — {label}")
    print("█" * 68)
    print(format_joint_plan(joint, task))
    print(f"\n  METRICS")
    print(f"  {'─'*40}")
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")

    return {
        "label":   label,
        "task_id": task_id,
        "task":    task,
        "flags": {
            "use_offer":       use_offer,
            "use_negotiation": use_negotiation,
            "use_human_query": use_human_query,
        },
        "offers": {
            "agent_A": offer_to_dict(offer_a),
            "agent_B": offer_to_dict(offer_b),
        },
        "local_plans": {
            "agent_A": local_plan_to_dict(plan_a),
            "agent_B": local_plan_to_dict(plan_b),
        },
        "conflicts":    [asdict(c) for c in conflicts],
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
            "no_missing_deps":  convergence.no_missing_deps,
            "no_dep_cycle":     convergence.no_dep_cycle,
            "observability_ok": convergence.observability_ok,
            "unresolved":       len(convergence.unresolved_conflicts),
        },
        "human_answers": human_answers,
        "hq_triggers":   hq_triggers,
        "joint_plan":    joint,
        "metrics":       metrics,
    }
