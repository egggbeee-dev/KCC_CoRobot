# p2p_ablation.py
#
# Ablation Study: offer / negotiate / hq 각 컴포넌트 기여도 측정
#
# 조건:
#   Full P2P        : offer + negotiate + HQ  (우리 시스템 전체)
#   w/o Offer       : offer 정보 미사용 (use_offer=False) + negotiate + HQ
#   w/o Negotiate   : offer + negotiate 스킵 + HQ
#   w/o HQ          : offer + negotiate + HQ 스킵 (수렴 실패 시 강제 확정)
#
# 출력 스타일: p2p_phases.py 와 동일 (_banner / _log / format_joint_plan)
# 측정:        PT (elapsed time), TC (token cost) — 자동
#
# 실행 (Colab):
#   %run p2p_ablation.py
#   또는
#   from p2p_ablation import run_ablation_study
#   run_ablation_study("task_003", img_a, img_b, n_runs=3)

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from IPython.display import display

os.chdir("/content/KCC_CoRobot")
if "Code_p2p" not in sys.path:
    sys.path.insert(0, "Code_p2p")

os.environ["VLM_BACKEND"] = "openai"
try:
    from google.colab import userdata
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
except Exception:
    from dotenv import load_dotenv
    load_dotenv()

from p2p_phases import (
    _banner, _log, _run_parallel,
    _build_phase1_prompt, _parse_offer,
    _build_phase2_prompt, _parse_local_plan,
    _ensure_pass, plan_steps_to_dicts,
    phase1_offer, phase2_local_plan,
    phase3_conflict_detection, phase4_negotiation,
    phase5_convergence_check, phase6_human_query,
    phase_finalize, format_joint_plan, offer_to_dict,
)
from p2p_config import AGENT_B_STEP_OFFSET
from p2p_main import get_task, run
from p2p_tracker import tracker
from p2p_utils import extract_json, jdump
from p2p_models import LocalPlan, PlanStep
from p2p_utils import compute_plan_uncertainty


# ── 헬퍼: Dict list → LocalPlan ──────────────────────────────────────────────

def _dicts_to_localplan(steps: List[Dict], offer) -> LocalPlan:
    ps = [PlanStep(
        step_id      = s.get("step_id", 0),
        time_min     = s.get("time_min", 0),
        room         = s.get("room", offer.room_type),
        agent_id     = s.get("agent_id", offer.agent_id),
        action       = s.get("action", ""),
        preconditions= s.get("preconditions", []),
        depends_on   = s.get("depends_on", []),
        handoff_type = s.get("handoff_type"),
        target_agent = s.get("target_agent"),
        uncertainty  = s.get("uncertainty", 0.1),
        notes        = s.get("notes", ""),
    ) for s in steps]
    unc = compute_plan_uncertainty([p.uncertainty for p in ps]) if ps else 0.0
    return LocalPlan(offer.agent_id, ps, unc, [], [])


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION 조건별 실행 함수
# ══════════════════════════════════════════════════════════════════════════════

def run_full_p2p(task_id: str, img_a: str, img_b: str) -> Dict:
    """Full P2P: offer + negotiate + HQ (우리 시스템 전체)."""
    task_str   = get_task(task_id)
    result     = run(task_id, img_a, img_b, verbose="full")
    joint_plan = result.get("joint_plan", [])

    print("\n" + format_joint_plan(joint_plan, task_str))

    return {
        "condition":  "Full P2P",
        "task_id":    task_id,
        "task_description": task_str,
        "img_a": img_a, "img_b": img_b,
        "joint_plan": joint_plan,
        "neg_rounds": result.get("metrics", {}).get("negotiation_rounds", 0),
        "hq_used":    result.get("human_query_used", False),
    }


def run_without_offer(task_id: str, img_a: str, img_b: str) -> Dict:
    """
    w/o Offer:
    - Phase 1: offer 생성은 하되 상대방 offer 정보를 local plan에 전달하지 않음
    - use_offer=False → 상대방 can_provide / need_from_other 미제공
    - Phase 3~6: negotiate + HQ 정상 실행
    """
    task_str = get_task(task_id)

    # Phase 1 — offer 생성 (관측용)
    offer_a, offer_b = phase1_offer(img_a, img_b, task_str, verbose="full")

    # Phase 2 — use_offer=False: 상대 offer 정보 미제공
    _banner("w/o Offer — PHASE 2: LOCAL PLANNING (상대방 offer 정보 미사용)")
    plan_a, plan_b = phase2_local_plan(
        offer_a, offer_b, img_a, img_b, task_str,
        use_offer=False,
        verbose="full",
    )

    # Phase 3
    conflicts = phase3_conflict_detection(plan_a, plan_b, offer_a, offer_b, verbose="full")

    # Phase 4 — negotiate (있음)
    steps_a, steps_b, rounds = phase4_negotiation(
        plan_a, plan_b, offer_a, offer_b, conflicts,
        img_a, img_b, task_str, verbose="full",
    )

    # Phase 5
    convergence = phase5_convergence_check(steps_a, steps_b, offer_a, offer_b, conflicts)

    # Phase 6 — HQ (있음)
    lp_a = _dicts_to_localplan(steps_a, offer_a)
    lp_b = _dicts_to_localplan(steps_b, offer_b)
    human_answers, _, _ = phase6_human_query(
        lp_a, lp_b, offer_a, offer_b,
        convergence, img_a, img_b, use_human_query=True,
    )

    # Finalize
    joint_plan = phase_finalize(
        steps_a, steps_b, offer_a, offer_b,
        human_answers, convergence, verbose="full",
    )

    print("\n" + format_joint_plan(joint_plan, task_str))

    return {
        "condition":  "w/o Offer",
        "task_id":    task_id,
        "task_description": task_str,
        "img_a": img_a, "img_b": img_b,
        "joint_plan": joint_plan,
        "neg_rounds": len(rounds),
        "hq_used":    bool(human_answers),
    }


def run_without_negotiate(task_id: str, img_a: str, img_b: str) -> Dict:
    """
    w/o Negotiate:
    - Phase 1: offer 정상
    - Phase 2: local plan 정상 (use_offer=True)
    - Phase 3: conflict detection 정상
    - Phase 4: SKIP — negotiate 없이 바로 Phase 5로
    - Phase 5~6: convergence + HQ 정상
    """
    task_str = get_task(task_id)

    # Phase 1
    offer_a, offer_b = phase1_offer(img_a, img_b, task_str, verbose="full")

    # Phase 2
    plan_a, plan_b = phase2_local_plan(
        offer_a, offer_b, img_a, img_b, task_str,
        use_offer=True, verbose="full",
    )

    # Phase 3
    conflicts = phase3_conflict_detection(plan_a, plan_b, offer_a, offer_b, verbose="full")

    # Phase 4 — SKIP
    _banner("w/o Negotiate — PHASE 4: NEGOTIATION SKIPPED")
    print("  [ABLATION] Negotiation 스킵 — conflict 있어도 그대로 진행.")
    steps_a = plan_steps_to_dicts(plan_a.steps)
    steps_b = plan_steps_to_dicts(plan_b.steps)

    # Phase 5
    convergence = phase5_convergence_check(steps_a, steps_b, offer_a, offer_b, conflicts)

    # Phase 6 — HQ (있음)
    human_answers, _, _ = phase6_human_query(
        plan_a, plan_b, offer_a, offer_b,
        convergence, img_a, img_b, use_human_query=True,
    )

    # Finalize
    joint_plan = phase_finalize(
        steps_a, steps_b, offer_a, offer_b,
        human_answers, convergence, verbose="full",
    )

    print("\n" + format_joint_plan(joint_plan, task_str))

    return {
        "condition":  "w/o Negotiate",
        "task_id":    task_id,
        "task_description": task_str,
        "img_a": img_a, "img_b": img_b,
        "joint_plan": joint_plan,
        "neg_rounds": 0,
        "hq_used":    bool(human_answers),
    }


def run_without_hq(task_id: str, img_a: str, img_b: str) -> Dict:
    """
    w/o HQ:
    - Phase 1~4: 정상 실행
    - Phase 5: convergence check 정상
    - Phase 6: SKIP — 수렴 실패해도 강제 finalize
    """
    task_str = get_task(task_id)

    # Phase 1
    offer_a, offer_b = phase1_offer(img_a, img_b, task_str, verbose="full")

    # Phase 2
    plan_a, plan_b = phase2_local_plan(
        offer_a, offer_b, img_a, img_b, task_str,
        use_offer=True, verbose="full",
    )

    # Phase 3
    conflicts = phase3_conflict_detection(plan_a, plan_b, offer_a, offer_b, verbose="full")

    # Phase 4 — negotiate (있음)
    steps_a, steps_b, rounds = phase4_negotiation(
        plan_a, plan_b, offer_a, offer_b, conflicts,
        img_a, img_b, task_str, verbose="full",
    )

    # Phase 5
    convergence = phase5_convergence_check(steps_a, steps_b, offer_a, offer_b, conflicts)

    # Phase 6 — SKIP
    _banner("w/o HQ — PHASE 6: HUMAN QUERY SKIPPED")
    print("  [ABLATION] Human Query 스킵 — 수렴 실패해도 강제 확정.")
    human_answers = {}

    # Finalize
    joint_plan = phase_finalize(
        steps_a, steps_b, offer_a, offer_b,
        human_answers, convergence, verbose="full",
    )

    print("\n" + format_joint_plan(joint_plan, task_str))

    return {
        "condition":  "w/o HQ",
        "task_id":    task_id,
        "task_description": task_str,
        "img_a": img_a, "img_b": img_b,
        "joint_plan": joint_plan,
        "neg_rounds": len(rounds),
        "hq_used":    False,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RESULT SAVE
# ══════════════════════════════════════════════════════════════════════════════

def _save_result(result: Dict, pt: float, tc: int, run_idx: int):
    Path("results").mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    cond = result["condition"].replace(" ", "_").replace("/", "")
    fname = f"results/ablation_{result['task_id']}_{cond}_run{run_idx}_{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump({**result, "pt": pt, "tc": tc}, f, ensure_ascii=False, indent=2)
    print(f"  → 저장: {fname}")
    return fname


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Ablation Study
# ══════════════════════════════════════════════════════════════════════════════

ABLATION_CONDITIONS = [
    ("Full P2P",      run_full_p2p),
    ("w/o Offer",     run_without_offer),
    ("w/o Negotiate", run_without_negotiate),
    ("w/o HQ",        run_without_hq),
]


def run_ablation_study(
    task_id: str,
    img_a: str,
    img_b: str,
    n_runs: int = 3,
) -> pd.DataFrame:
    """
    Ablation Study 실행.

    Args:
        task_id : 실험 태스크 ID
        img_a   : Agent A 이미지 경로
        img_b   : Agent B 이미지 경로
        n_runs  : 반복 횟수 (평균 산출)

    Returns:
        결과 DataFrame (PT, TC, NR 포함)
    """
    SEP = "═" * 65
    print(SEP)
    print(f"  ABLATION STUDY  |  task={task_id}  |  N={n_runs}")
    print(SEP)

    all_rows: Dict[str, list] = {cond: [] for cond, _ in ABLATION_CONDITIONS}

    for run_idx in range(1, n_runs + 1):
        print(f"\n{'━'*65}")
        print(f"  [Run {run_idx}/{n_runs}]")
        print(f"{'━'*65}")

        for condition, run_fn in ABLATION_CONDITIONS:
            print(f"\n▶ {condition}")
            tracker.start()
            try:
                result = run_fn(task_id, img_a, img_b)
            except Exception as e:
                print(f"  [ERROR] {condition}: {e}")
                tracker.stop()
                all_rows[condition].append({"pt": 0.0, "tc": 0, "neg_rounds": 0})
                continue
            tracker.stop()

            pt = tracker.elapsed
            tc = tracker.total_tokens
            nr = result.get("neg_rounds", 0)
            print(tracker.summary(condition))
            _save_result(result, pt, tc, run_idx)
            all_rows[condition].append({"pt": pt, "tc": tc, "neg_rounds": nr})

    # ── 결과 테이블 ──────────────────────────────────────────────────────────
    final_rows = []
    for condition, _ in ABLATION_CONDITIONS:
        rows = all_rows[condition]
        final_rows.append({
            "Condition": condition,
            "PT(s)":     round(float(np.mean([r["pt"] for r in rows])), 2),
            "TC":        int(np.mean([r["tc"] for r in rows])),
            "NR":        round(float(np.mean([r["neg_rounds"] for r in rows])), 1),
        })

    df = pd.DataFrame(final_rows)[["Condition", "PT(s)", "TC", "NR"]]

    print("\n" + "█" * 65)
    print("  Table. Ablation — PT / TC / NR")
    print("█" * 65)
    display(
        df.style
          .hide(axis="index")
          .format({"PT(s)": "{:.2f}", "TC": "{:,}", "NR": "{:.1f}"})
          .set_properties(**{"text-align": "center"})
    )
    print("\n[Markdown]")
    print(df.to_markdown(index=False))
    print("\n※ 플랜 품질은 위 자연어 출력을 통해 확인하세요.")
    print("※ NR = Negotiation Rounds (w/o Negotiate는 항상 0)")

    return df


# ── 직접 실행 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TASK_ID = "task_003"
    IMG_A   = "Data/Room/Livingrooms/simul/livingroom_12.png"
    IMG_B   = "Data/Room/Kitchens/simul/kitchen_14.png"
    N_RUNS  = 3

    run_ablation_study(TASK_ID, IMG_A, IMG_B, n_runs=N_RUNS)
