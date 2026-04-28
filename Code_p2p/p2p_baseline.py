# p2p_baseline.py
#
# Baseline Comparison: Centralized / Independent
#
# Centralized:
#   Step 1. VLM(img_a) → Room A 자연어 묘사
#   Step 2. VLM(img_b) → Room B 자연어 묘사
#   Step 3. VLM(img_a) + 두 묘사 + task → joint plan (few-shot 출력 포맷만)
#
# Independent:
#   Step 1. VLM(img_a) → Room A 자연어 묘사
#   Step 2. VLM(img_b) → Room B 자연어 묘사
#   Step 3. 각자 묘사 + task → 각자 plan (few-shot 출력 포맷만)
#   Step 4. rule-based merge
#
# 설계 원칙:
#   - can_do/cannot_do/can_provide/need_from_other 등 우리 방법론 구조 없음
#   - observation은 자연어로만 (이미지를 실제로 봄)
#   - few-shot은 출력 JSON 포맷만 가이드
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
from p2p_phases import _banner, _log, _run_parallel, format_joint_plan
from p2p_main import get_task
from p2p_tracker import tracker
from p2p_utils import extract_json


# ══════════════════════════════════════════════════════════════════════════════
# OBSERVATION PROMPT (공통)
# can_do/cannot_do 등 우리 방법론 구조 없이 자연어로만
# ══════════════════════════════════════════════════════════════════════════════

def _build_observation_prompt(task: str) -> str:
    return f"""Look at this image carefully.

Task: "{task}"

Describe the following in natural language:
1. What room is this?
2. What objects and areas do you see?
3. What actions can be done in this room to help with the task?

Be specific about visible objects. Keep it concise."""


# ══════════════════════════════════════════════════════════════════════════════
# CENTRALIZED
# ══════════════════════════════════════════════════════════════════════════════

_CENTRALIZED_FEW_SHOT = """
EXAMPLE OUTPUT FORMAT:
<JSON>
{
  "agent_A": [
    {"step_id": 1, "time_min": 0,  "action": "take water bottle from refrigerator"},
    {"step_id": 2, "time_min": 5,  "action": "prepare sandwich on countertop"},
    {"step_id": 3, "time_min": 10, "action": "place food items on kitchen counter"}
  ],
  "agent_B": [
    {"step_id": 101, "time_min": 0,  "action": "take laptop from dresser"},
    {"step_id": 102, "time_min": 5,  "action": "place laptop on side table"},
    {"step_id": 103, "time_min": 10, "action": "tidy bed for cleaner environment"}
  ]
}
</JSON>"""


def _build_centralized_plan_prompt(task: str, obs_a: str, obs_b: str) -> str:
    return f"""You are coordinating two home agents to complete a task.

Task: "{task}"

Room A (agent_A):
{obs_a}

Room B (agent_B):
{obs_b}

{_CENTRALIZED_FEW_SHOT}

Generate a plan for BOTH agents. Each agent works ONLY in their own room.
- agent_A steps: only actions possible in Room A
- agent_B steps: only actions possible in Room B
- step_id for agent_A: 1–99, for agent_B: 101–199
- Generate 4–6 steps per agent
- No handoff or transfer between agents

Return ONLY valid JSON inside <JSON> tags."""


def run_centralized(task_id: str, img_a: str, img_b: str) -> Dict:
    task_str = get_task(task_id)

    # Step 1+2: 각 이미지 자연어 묘사
    _banner("CENTRALIZED — STEP 1+2: OBSERVATION (자연어, 구조화 없음)")
    prompt_obs = _build_observation_prompt(task_str)
    results    = _run_parallel([
        (img_a, prompt_obs, False),
        (img_b, prompt_obs, False),
    ])
    obs_a, _ = results[0]
    obs_b, _ = results[1]
    _log("A OBSERVATION", obs_a)
    _log("B OBSERVATION", obs_b)

    # Step 3: 단일 플래너 → joint plan
    _banner("CENTRALIZED — STEP 3: JOINT PLAN (단일 플래너, handoff 없음)")
    prompt = _build_centralized_plan_prompt(task_str, obs_a, obs_b)
    raw, _ = p2p_vlm.run_vlm(img_a, prompt)
    _log("CENTRALIZED RAW PLAN", raw)

    data = extract_json(raw)
    if not isinstance(data, dict):
        data = {}

    def _parse(steps_raw, agent_id, offset):
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
                "room":         "kitchen" if agent_id == "agent_A" else "bedroom",
                "action":       s.get("action", ""),
                "depends_on":   [],
                "handoff_type": None,
                "target_agent": None,
            })
        return out

    steps_a    = _parse(data.get("agent_A", []), "agent_A", 0)
    steps_b    = _parse(data.get("agent_B", []), "agent_B", AGENT_B_STEP_OFFSET)
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
        "joint_plan": joint_plan,
    }


# ══════════════════════════════════════════════════════════════════════════════
# INDEPENDENT
# ══════════════════════════════════════════════════════════════════════════════

_INDEPENDENT_FEW_SHOT = """
EXAMPLE OUTPUT FORMAT:
<JSON>
{
  "plan_steps": [
    {"step_id": 1, "time_min": 0,  "action": "take water bottle from refrigerator"},
    {"step_id": 2, "time_min": 5,  "action": "prepare sandwich on countertop"},
    {"step_id": 3, "time_min": 10, "action": "place food on kitchen counter"}
  ]
}
</JSON>"""


def _build_independent_plan_prompt(task: str, obs: str) -> str:
    return f"""You are a home agent working independently.

Task: "{task}"

Your room:
{obs}

{_INDEPENDENT_FEW_SHOT}

Generate a plan for YOUR room only.
- Only actions possible in your room
- Generate 4–6 steps
- No handoff or transfer to other agents

Return ONLY valid JSON inside <JSON> tags."""


def _rule_based_merge(steps_a: List[Dict], steps_b: List[Dict]) -> List[Dict]:
    for s in steps_b:
        if s.get("step_id", 0) < AGENT_B_STEP_OFFSET:
            s["step_id"] = s["step_id"] + AGENT_B_STEP_OFFSET
    merged = list(steps_a) + list(steps_b)
    merged.sort(key=lambda s: (s.get("time_min", 0), s.get("step_id", 0)))
    return merged


def run_independent(task_id: str, img_a: str, img_b: str) -> Dict:
    task_str = get_task(task_id)

    # Step 1+2: 각 이미지 자연어 묘사
    _banner("INDEPENDENT — STEP 1+2: OBSERVATION (자연어, 구조화 없음)")
    prompt_obs = _build_observation_prompt(task_str)
    results    = _run_parallel([
        (img_a, prompt_obs, False),
        (img_b, prompt_obs, False),
    ])
    obs_a, _ = results[0]
    obs_b, _ = results[1]
    _log("A OBSERVATION", obs_a)
    _log("B OBSERVATION", obs_b)

    # Step 3: 각자 독립 plan
    _banner("INDEPENDENT — STEP 3: LOCAL PLANNING (상대방 모름)")
    prompt_a = _build_independent_plan_prompt(task_str, obs_a)
    prompt_b = _build_independent_plan_prompt(task_str, obs_b)
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

    steps_a = _parse(raw_pa, "agent_A", "kitchen", 0)
    steps_b = _parse(raw_pb, "agent_B", "bedroom", 0)
    print(f"\n  A: {len(steps_a)} steps | B: {len(steps_b)} steps")

    # Step 4: rule-based merge
    _banner("INDEPENDENT — STEP 4: RULE-BASED MERGE")
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
