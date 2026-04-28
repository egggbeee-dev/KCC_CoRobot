# p2p_baseline.py
#
# Baseline Comparison: Centralized / Independent
# (P2P Ours 제외 — Ablation의 Full P2P로 대체)
#
# Centralized:
#   Step 1. phase1 프롬프트(우리 것) → 두 방 observation 추출
#   Step 2. observation 합쳐서 단일 플래너 → joint plan (PASS/handoff 없음)
#
# Independent:
#   Step 1. phase1 프롬프트(우리 것) → 각자 observation 추출
#   Step 2. 각자 자기 방 plan 생성 (상대방 모름, PASS/handoff 없음)
#   Step 3. rule-based merge
#
# 측정: PT (elapsed time), TC (token cost) 만 측정
#
# 실행:
#   from p2p_baseline import run_baseline_comparison
#   run_baseline_comparison(
#       task_id     = "task_003",
#       image_pairs = [(img_a, img_b), ...],
#   )

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from IPython.display import display

import p2p_vlm
from p2p_config import AGENT_B_STEP_OFFSET
from p2p_phases import (
    _banner, _log, _run_parallel,
    _build_phase1_prompt, _parse_offer,
    format_joint_plan, offer_to_dict,
)
from p2p_main import get_task
from p2p_tracker import tracker
from p2p_utils import extract_json, jdump


# ══════════════════════════════════════════════════════════════════════════════
# CENTRALIZED
# ══════════════════════════════════════════════════════════════════════════════

def _build_centralized_prompt(task: str, offer_a, offer_b) -> str:
    return f"""Task: "{task}"

Room A (agent_A):
- Observation: {offer_a.observation}
- Visible objects: {offer_a.obs_scope}

Room B (agent_B):
- Observation: {offer_b.observation}
- Visible objects: {offer_b.obs_scope}

Generate a plan for both agents. Each agent works ONLY in their own room.
No handoff or transfer between agents.

Return ONLY valid JSON inside <JSON> tags:
<JSON>
{{
  "agent_A": [
    {{"step_id": 1, "time_min": 0, "action": "verb + specific object"}}
  ],
  "agent_B": [
    {{"step_id": 101, "time_min": 0, "action": "verb + specific object"}}
  ]
}}
</JSON>"""


def run_centralized(task_id: str, img_a: str, img_b: str) -> Dict:
    task_str = get_task(task_id)

    _banner("CENTRALIZED — STEP 1: OBSERVATION")
    prompt_obs = _build_phase1_prompt(task_str)
    results    = _run_parallel([
        (img_a, prompt_obs, False),
        (img_b, prompt_obs, False),
    ])
    raw_a, _ = results[0]
    raw_b, _ = results[1]
    _log("A RAW OFFER", raw_a)
    _log("B RAW OFFER", raw_b)

    offer_a = _parse_offer(raw_a, "agent_A")
    offer_b = _parse_offer(raw_b, "agent_B")
    _log("OFFER A", jdump(offer_to_dict(offer_a)))
    _log("OFFER B", jdump(offer_to_dict(offer_b)))
    print(f"\n  A: room={offer_a.room_type} | can_do={len(offer_a.can_do)}")
    print(f"  B: room={offer_b.room_type} | can_do={len(offer_b.can_do)}")

    _banner("CENTRALIZED — STEP 2: JOINT PLAN (단일 플래너, handoff 없음)")
    prompt = _build_centralized_prompt(task_str, offer_a, offer_b)
    raw, _ = p2p_vlm.run_vlm(img_a, prompt)
    _log("CENTRALIZED RAW PLAN", raw)

    data = extract_json(raw)
    if not isinstance(data, dict):
        data = {}

    def _parse(steps_raw, agent_id, room, offset):
        if not isinstance(steps_raw, list):
            return []
        out = []
        for s in steps_raw:
            if not isinstance(s, dict) or "action" not in s:
                continue
            sid = s.get("step_id", len(out) + 1)
            out.append({
                "step_id":      sid if sid >= offset else sid + offset,
                "time_min":     s.get("time_min", 0),
                "agent_id":     agent_id,
                "room":         room,
                "action":       s.get("action", ""),
                "depends_on":   [],
                "handoff_type": None,
                "target_agent": None,
            })
        return out

    steps_a    = _parse(data.get("agent_A", []), "agent_A", offer_a.room_type, 0)
    steps_b    = _parse(data.get("agent_B", []), "agent_B", offer_b.room_type, AGENT_B_STEP_OFFSET)
    joint_plan = sorted(steps_a + steps_b,
                        key=lambda s: (s.get("time_min", 0), s.get("step_id", 0)))

    print(f"\n  A: {len(steps_a)} steps | B: {len(steps_b)} steps | total: {len(joint_plan)}")
    print("\n" + "█" * 68)
    print("  FINAL JOINT PLAN — Centralized")
    print("█" * 68)
    print(format_joint_plan(joint_plan, task_str))

    return {
        "method":     "Centralized",
        "task_id":    task_id,
        "task":       task_str,
        "offers":     {"agent_A": offer_to_dict(offer_a), "agent_B": offer_to_dict(offer_b)},
        "joint_plan": joint_plan,
    }


# ══════════════════════════════════════════════════════════════════════════════
# INDEPENDENT
# ══════════════════════════════════════════════════════════════════════════════

def _build_independent_prompt(task: str, offer) -> str:
    return f"""Task: "{task}"

Your room:
- Observation: {offer.observation}
- Visible objects: {offer.obs_scope}

Generate a plan for YOUR room only. Work independently.
No handoff or transfer to other agents.

Return ONLY valid JSON inside <JSON> tags:
<JSON>
{{
  "plan_steps": [
    {{"step_id": 1, "time_min": 0, "action": "verb + specific object"}}
  ]
}}
</JSON>"""


def _rule_based_merge(steps_a: List[Dict], steps_b: List[Dict]) -> List[Dict]:
    for s in steps_b:
        if s.get("step_id", 0) < AGENT_B_STEP_OFFSET:
            s["step_id"] = s["step_id"] + AGENT_B_STEP_OFFSET
    merged = list(steps_a) + list(steps_b)
    merged.sort(key=lambda s: (s.get("time_min", 0), s.get("step_id", 0)))
    return merged


def run_independent(task_id: str, img_a: str, img_b: str) -> Dict:
    task_str = get_task(task_id)

    _banner("INDEPENDENT — STEP 1: OBSERVATION")
    prompt_obs = _build_phase1_prompt(task_str)
    results    = _run_parallel([
        (img_a, prompt_obs, False),
        (img_b, prompt_obs, False),
    ])
    raw_a, _ = results[0]
    raw_b, _ = results[1]
    _log("A RAW OFFER", raw_a)
    _log("B RAW OFFER", raw_b)

    offer_a = _parse_offer(raw_a, "agent_A")
    offer_b = _parse_offer(raw_b, "agent_B")
    _log("OFFER A", jdump(offer_to_dict(offer_a)))
    _log("OFFER B", jdump(offer_to_dict(offer_b)))
    print(f"\n  A: room={offer_a.room_type} | can_do={len(offer_a.can_do)}")
    print(f"  B: room={offer_b.room_type} | can_do={len(offer_b.can_do)}")

    _banner("INDEPENDENT — STEP 2: LOCAL PLANNING (상대방 정보 없음)")
    prompt_a = _build_independent_prompt(task_str, offer_a)
    prompt_b = _build_independent_prompt(task_str, offer_b)
    results       = _run_parallel([
        (img_a, prompt_a, False),
        (img_b, prompt_b, False),
    ])
    raw_pa, _ = results[0]
    raw_pb, _ = results[1]
    _log("A RAW PLAN", raw_pa)
    _log("B RAW PLAN", raw_pb)

    def _parse(raw, agent_id, room, offset):
        data = extract_json(raw)
        if isinstance(data, list):
            data = {"plan_steps": data}
        if not isinstance(data, dict):
            return []
        out = []
        for s in data.get("plan_steps", []):
            if not isinstance(s, dict) or "action" not in s:
                continue
            sid = s.get("step_id", len(out) + 1)
            out.append({
                "step_id":      sid if sid >= offset else sid + offset,
                "time_min":     s.get("time_min", 0),
                "agent_id":     agent_id,
                "room":         room,
                "action":       s.get("action", ""),
                "depends_on":   [],
                "handoff_type": None,
                "target_agent": None,
            })
        return out

    steps_a = _parse(raw_pa, "agent_A", offer_a.room_type, 0)
    steps_b = _parse(raw_pb, "agent_B", offer_b.room_type, 0)
    print(f"\n  A: {len(steps_a)} steps | B: {len(steps_b)} steps")

    _banner("INDEPENDENT — STEP 3: RULE-BASED MERGE")
    joint_plan = _rule_based_merge(steps_a, steps_b)
    print(f"  Merged: {len(joint_plan)} steps total")
    print("\n" + "█" * 68)
    print("  FINAL JOINT PLAN — Independent")
    print("█" * 68)
    print(format_joint_plan(joint_plan, task_str))

    return {
        "method":     "Independent",
        "task_id":    task_id,
        "task":       task_str,
        "offers":     {"agent_A": offer_to_dict(offer_a), "agent_B": offer_to_dict(offer_b)},
        "joint_plan": joint_plan,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════

def _save_result(result: Dict, pt: float, tc: int, run_idx: int):
    save_dir = Path("/content/KCC_CoRobot/results")
    save_dir.mkdir(exist_ok=True)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    method = result["method"].replace(" ", "_")
    fname  = save_dir / f"baseline_{result['task_id']}_{method}_run{run_idx}_{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump({**result, "pt": pt, "tc": tc}, f, ensure_ascii=False, indent=2)
    print(f"  → 저장: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline_comparison(
    task_id:     str,
    image_pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    """
    Args:
        task_id     : 실험 태스크 ID
        image_pairs : [(img_a, img_b), ...] 이미지 페어 리스트
    """
    conditions = [
        ("Centralized", run_centralized),
        ("Independent", run_independent),
    ]

    SEP = "═" * 68
    print(SEP)
    print(f"  BASELINE COMPARISON  |  task={task_id}  |  N={len(image_pairs)}")
    print(SEP)

    all_rows: Dict[str, List[Dict]] = {name: [] for name, _ in conditions}

    for run_idx, (img_a, img_b) in enumerate(image_pairs, 1):
        print(f"\n{'━'*68}")
        print(f"  [Run {run_idx}/{len(image_pairs)}]")
        print(f"  img_a: {img_a}")
        print(f"  img_b: {img_b}")
        print(f"{'━'*68}")

        for method_name, run_fn in conditions:
            _banner(f"BASELINE — {method_name}")
            tracker.start()
            try:
                result = run_fn(task_id, img_a, img_b)
            except Exception as e:
                print(f"  [ERROR] {method_name}: {e}")
                tracker.stop()
                all_rows[method_name].append({"pt": 0.0, "tc": 0})
                continue
            tracker.stop()

            pt = tracker.elapsed
            tc = tracker.total_tokens
            print(tracker.summary(method_name))
            _save_result(result, pt, tc, run_idx)
            all_rows[method_name].append({"pt": pt, "tc": tc})

    # ── 결과 테이블 ──────────────────────────────────────────────────────────
    final_rows = []
    for method_name, _ in conditions:
        rows = all_rows[method_name]
        final_rows.append({
            "Method": method_name,
            "PT(s)":  round(float(np.mean([r["pt"] for r in rows])), 2),
            "TC":     int(np.mean([r["tc"] for r in rows])),
        })

    df = pd.DataFrame(final_rows)[["Method", "PT(s)", "TC"]]

    print("\n" + "█" * 68)
    print("  Table. Baseline Comparison — PT / TC")
    print("█" * 68)
    display(
        df.style
          .hide(axis="index")
          .format({"PT(s)": "{:.2f}", "TC": "{:,}"})
          .set_properties(**{"text-align": "center"})
    )
    print("\n[Markdown]")
    print(df.to_markdown(index=False))
    print("\n※ 플랜 품질은 위 자연어 출력을 통해 확인하세요.")

    return df
