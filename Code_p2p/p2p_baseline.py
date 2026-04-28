# p2p_baseline.py
#
# Baseline Comparison: Centralized / Independent / P2P (Ours)
#
# ─ Centralized  : phase1으로 두 방 observation 추출 → 단일 플래너가 joint plan
# ─ Independent  : 각자 이미지만 보고 개별 plan → rule-based merge
# ─ P2P (Ours)   : p2p_main.run() 그대로
#
# 출력 스타일: p2p_phases.py 와 동일 (_banner / _log / format_joint_plan)
# 측정:        PT (elapsed time), TC (token cost) — 자동
#
# 실행 (Colab):
#   %run p2p_baseline.py
#   또는
#   from p2p_baseline import run_baseline_comparison
#   run_baseline_comparison("task_003", img_a, img_b, n_runs=3)

from __future__ import annotations

import json
import os
import sys
import time
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
    _normalize_pass, _ensure_pass,
    plan_steps_to_dicts, format_joint_plan,
    offer_to_dict,
)
from p2p_config import AGENT_B_STEP_OFFSET
from p2p_main import get_task, run
from p2p_tracker import tracker
from p2p_utils import extract_json, jdump


# ══════════════════════════════════════════════════════════════════════════════
# CENTRALIZED BASELINE
# ══════════════════════════════════════════════════════════════════════════════
#
# 설계:
#   Step 1. phase1_offer 프롬프트로 두 에이전트의 observation / can_do 추출
#           (우리 시스템과 동일한 프롬프트 구조)
#   Step 2. 두 observation을 합쳐 단일 플래너에게 joint plan 요청
#           (협상 없이 한 번에 생성 — centralized의 핵심)

_CENTRALIZED_JOINT_EXAMPLE = """
EXAMPLE — task "prepare movie night":
<JSON>
{
  "agent_A": [
    {"step_id": 1,   "time_min": 0,  "action": "arrange sofa cushions for comfortable seating",
     "depends_on": [], "handoff_type": null, "target_agent": null},
    {"step_id": 2,   "time_min": 5,  "action": "dim floor lamp to create movie atmosphere",
     "depends_on": [], "handoff_type": null, "target_agent": null},
    {"step_id": 3,   "time_min": 16, "action": "receive snack tray from kitchen doorway and place on table",
     "depends_on": [103], "handoff_type": null, "target_agent": null}
  ],
  "agent_B": [
    {"step_id": 101, "time_min": 0,  "action": "place apple and orange from island onto serving tray",
     "depends_on": [], "handoff_type": null, "target_agent": null},
    {"step_id": 102, "time_min": 5,  "action": "fill water glass from kitchen tap",
     "depends_on": [], "handoff_type": null, "target_agent": null},
    {"step_id": 103, "time_min": 15, "action": "carry snack tray to kitchen doorway for agent_A pickup",
     "depends_on": [101, 102], "handoff_type": "PASS", "target_agent": "agent_A"}
  ]
}
</JSON>
""".strip()


def _build_centralized_joint_prompt(
    task: str, offer_a, offer_b,
) -> str:
    """
    두 방의 observation 을 합쳐 단일 플래너에게 joint plan 요청.
    우리 phase2 프롬프트 구조를 베이스로 하되, 협상 없이 한 번에 생성.
    """
    return f"""You are a centralized planner coordinating two embodied home agents.

Global task: "{task}"

Room A (agent_A):
- Observation: {offer_a.observation}
- Visible objects: {offer_a.obs_scope}
- Can do: {json.dumps(offer_a.can_do, ensure_ascii=False)}
- Cannot do: {json.dumps([c.action for c in offer_a.cannot_do], ensure_ascii=False)}

Room B (agent_B):
- Observation: {offer_b.observation}
- Visible objects: {offer_b.obs_scope}
- Can do: {json.dumps(offer_b.can_do, ensure_ascii=False)}
- Cannot do: {json.dumps([c.action for c in offer_b.cannot_do], ensure_ascii=False)}

{_CENTRALIZED_JOINT_EXAMPLE}

Generate a joint plan for BOTH agents to achieve the task together.

RULES:
1. agent_A steps: ONLY use objects visible in Room A.
2. agent_B steps: ONLY use objects visible in Room B.
3. step_id for agent_A: 1–99. step_id for agent_B: 101–199.
4. Format: "verb + specific visible object + purpose"
5. Generate 4–6 steps per agent (8–12 total).
6. HANDOFF (PASS): if agent_B physically carries an item to the doorway:
   - action starts with "carry" or "bring"
   - depends_on=[prep step ids], handoff_type="PASS", target_agent="agent_A"
   - agent_A must have a receive step with depends_on=[PASS step_id]
7. No negotiation — produce the best plan in one shot.
8. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "agent_A": [
    {{"step_id": 1, "time_min": 0, "action": "verb + specific object",
      "depends_on": [], "handoff_type": null, "target_agent": null}}
  ],
  "agent_B": [
    {{"step_id": 101, "time_min": 0, "action": "verb + specific object",
      "depends_on": [], "handoff_type": null, "target_agent": null}}
  ]
}}
</JSON>"""


def run_centralized(task_id: str, img_a: str, img_b: str) -> Dict:
    """Centralized baseline 실행."""
    import p2p_vlm

    task_str = get_task(task_id)

    _banner("CENTRALIZED — STEP 1: OBSERVATION (phase1 프롬프트 동일)")
    prompt_obs = _build_phase1_prompt(task_str)
    results        = _run_parallel([
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
    print(f"\n  A: room={offer_a.room_type} | obs={offer_a.observation[:60]}")
    print(f"  B: room={offer_b.room_type} | obs={offer_b.observation[:60]}")

    _banner("CENTRALIZED — STEP 2: JOINT PLAN (단일 플래너, 협상 없음)")
    prompt_joint = _build_centralized_joint_prompt(task_str, offer_a, offer_b)
    # img_a 기준 호출 (observation은 텍스트로 이미 포함)
    raw_joint, _ = p2p_vlm.run_vlm(img_a, prompt_joint)
    _log("CENTRALIZED RAW JOINT PLAN", raw_joint)

    data = extract_json(raw_joint)
    if not isinstance(data, dict):
        data = {}

    def _parse(steps_raw, agent_id, offset):
        if not isinstance(steps_raw, list):
            return []
        result = []
        for s in steps_raw:
            if not isinstance(s, dict) or "action" not in s:
                continue
            s["agent_id"] = agent_id
            s["room"]     = offer_a.room_type if agent_id == "agent_A" else offer_b.room_type
            sid = s.get("step_id", len(result) + 1)
            s["step_id"] = sid if sid >= offset else sid + offset
            result.append(s)
        return result

    steps_a    = _parse(data.get("agent_A", []), "agent_A", 0)
    steps_b    = _parse(data.get("agent_B", []), "agent_B", AGENT_B_STEP_OFFSET)
    joint_plan = sorted(steps_a + steps_b, key=lambda s: (s.get("time_min", 0), s.get("step_id", 0)))

    print(f"\n  A: {len(steps_a)} steps | B: {len(steps_b)} steps | total: {len(joint_plan)}")
    print("\n" + format_joint_plan(joint_plan, task_str))

    return {
        "method":     "Centralized",
        "task_id":    task_id,
        "task_description": task_str,
        "img_a":      img_a,
        "img_b":      img_b,
        "offer_a":    offer_to_dict(offer_a),
        "offer_b":    offer_to_dict(offer_b),
        "joint_plan": joint_plan,
    }


# ══════════════════════════════════════════════════════════════════════════════
# INDEPENDENT BASELINE
# ══════════════════════════════════════════════════════════════════════════════
#
# 설계:
#   - 우리 phase1 + phase2 프롬프트 구조 동일
#   - use_offer=False → 상대방 정보 없이 각자 계획
#   - rule-based merge: time_min 정렬 + step_id 충돌 방지
#   - 의미론적 충돌 해결 없음 (Independent의 약점 그대로 반영)

def _build_independent_prompt(task: str, my_offer, other_room: str) -> str:
    """
    우리 phase2 프롬프트 구조를 베이스로.
    상대방 offer 정보 없이 자기 방만 계획.
    """
    from p2p_phases import _P2_EXAMPLE, _P2_HANDOFF_RULES
    return f"""You are the {my_offer.room_type} agent ({my_offer.agent_id}).
Global task: "{task}"

YOUR ROOM ONLY:
- Observation: {my_offer.observation}
- Visible objects: {my_offer.obs_scope}
- Can do: {json.dumps(my_offer.can_do, ensure_ascii=False)}
- Cannot do: {json.dumps([c.action for c in my_offer.cannot_do], ensure_ascii=False)}

Other agent is in: {other_room} (you do NOT know what they are doing)

{_P2_EXAMPLE}

Generate YOUR local plan independently. You are working without coordination.

PLANNING RULES:
1. Steps ONLY in your room ({my_offer.room_type}), using ONLY visible objects.
2. Generate 4–6 steps over 0–25 minutes. NO repeated actions.
3. Prioritize actions that DIRECTLY contribute to the global task.
4. Do NOT add handoff steps — you are working independently.
5. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "plan_steps": [
    {{"step_id": 1, "time_min": 0, "action": "verb + specific object",
      "preconditions": [], "depends_on": [], "handoff_type": null,
      "target_agent": null, "uncertainty": 0.1, "notes": ""}}
  ]
}}
</JSON>"""


def _rule_based_merge(steps_a: List[Dict], steps_b: List[Dict]) -> List[Dict]:
    """
    Rule-based merge:
    - step_id 충돌 방지 (agent_B offset 적용)
    - time_min 기준 정렬
    - 의미론적 충돌 해결 없음 (Independent의 약점)
    """
    for s in steps_b:
        if s.get("step_id", 0) < AGENT_B_STEP_OFFSET:
            s["step_id"] = s["step_id"] + AGENT_B_STEP_OFFSET

    merged = list(steps_a) + list(steps_b)
    merged.sort(key=lambda s: (s.get("time_min", 0), s.get("step_id", 0)))
    return merged


def run_independent(task_id: str, img_a: str, img_b: str) -> Dict:
    """Independent baseline 실행."""
    task_str = get_task(task_id)

    # Step 1: phase1 프롬프트로 observation 추출 (우리 시스템과 동일)
    _banner("INDEPENDENT — STEP 1: OBSERVATION (phase1 프롬프트 동일)")
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
    print(f"\n  A: room={offer_a.room_type} | obs={offer_a.observation[:60]}")
    print(f"  B: room={offer_b.room_type} | obs={offer_b.observation[:60]}")

    # Step 2: 각자 독립 계획 (상대방 정보 없음)
    _banner("INDEPENDENT — STEP 2: LOCAL PLANNING (상대방 정보 없음)")
    prompt_a = _build_independent_prompt(task_str, offer_a, offer_b.room_type)
    prompt_b = _build_independent_prompt(task_str, offer_b, offer_a.room_type)

    results        = _run_parallel([
        (img_a, prompt_a, True),
        (img_b, prompt_b, True),
    ])
    raw_pa, logp_a = results[0]
    raw_pb, logp_b = results[1]
    _log("A RAW PLAN", raw_pa)
    _log("B RAW PLAN", raw_pb)

    plan_a = _parse_local_plan(raw_pa, logp_a, offer_a, step_offset=0)
    plan_b = _parse_local_plan(raw_pb, logp_b, offer_b, step_offset=AGENT_B_STEP_OFFSET)

    steps_a = plan_steps_to_dicts(plan_a.steps)
    steps_b = plan_steps_to_dicts(plan_b.steps)
    print(f"\n  A: {len(steps_a)} steps | B: {len(steps_b)} steps")

    # Step 3: Rule-based merge
    _banner("INDEPENDENT — STEP 3: RULE-BASED MERGE (의미론적 충돌 해결 없음)")
    joint_plan = _rule_based_merge(steps_a, steps_b)
    print(f"  Merged: {len(joint_plan)} steps total")
    print("\n" + format_joint_plan(joint_plan, task_str))

    return {
        "method":     "Independent",
        "task_id":    task_id,
        "task_description": task_str,
        "img_a":      img_a,
        "img_b":      img_b,
        "offer_a":    offer_to_dict(offer_a),
        "offer_b":    offer_to_dict(offer_b),
        "joint_plan": joint_plan,
    }


# ══════════════════════════════════════════════════════════════════════════════
# P2P FULL (OURS)
# ══════════════════════════════════════════════════════════════════════════════

def run_ours(task_id: str, img_a: str, img_b: str) -> Dict:
    """P2P Full (우리 시스템) — p2p_main.run() 그대로."""
    task_str = get_task(task_id)
    result   = run(task_id, img_a, img_b, verbose="full")

    joint_plan = result.get("joint_plan", [])
    print("\n" + format_joint_plan(joint_plan, task_str))

    return {
        "method":     "P2P (Ours)",
        "task_id":    task_id,
        "task_description": task_str,
        "img_a":      img_a,
        "img_b":      img_b,
        "joint_plan": joint_plan,
        "raw_result": result,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RESULT SAVE
# ══════════════════════════════════════════════════════════════════════════════

def _save_result(result: Dict, pt: float, tc: int, run_idx: int):
    Path("results").mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_tag = result["method"].replace(" ", "_").replace("(", "").replace(")", "")
    fname = f"results/baseline_{result['task_id']}_{method_tag}_run{run_idx}_{ts}.json"
    payload = {**result, "pt": pt, "tc": tc}
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  → 저장: {fname}")
    return fname


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Baseline Comparison
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline_comparison(
    task_id: str,
    img_a: str,
    img_b: str,
    n_runs: int = 1,
) -> pd.DataFrame:
    """
    Baseline Comparison 실행.

    Args:
        task_id : 실험 태스크 ID
        img_a   : Agent A 이미지 경로
        img_b   : Agent B 이미지 경로
        n_runs  : 반복 횟수 (평균 산출)

    Returns:
        결과 DataFrame (PT, TC 포함)
    """
    SEP = "═" * 65
    print(SEP)
    print(f"  BASELINE COMPARISON  |  task={task_id}  |  N={n_runs}")
    print(SEP)

    conditions = [
        ("P2P (Ours)",  run_ours),
        ("Centralized", run_centralized),
        ("Independent", run_independent),
    ]

    all_rows: Dict[str, list] = {name: [] for name, _ in conditions}

    for run_idx in range(1, n_runs + 1):
        print(f"\n{'━'*65}")
        print(f"  [Run {run_idx}/{n_runs}]")
        print(f"{'━'*65}")

        for method_name, run_fn in conditions:
            print(f"\n▶ {method_name}")
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

    print("\n" + "█" * 65)
    print("  Table. Baseline — PT / TC")
    print("█" * 65)
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


# ── 직접 실행 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TASK_ID = "task_003"
    IMG_A   = "Data/Room/Livingrooms/simul/livingroom_12.png"
    IMG_B   = "Data/Room/Kitchens/simul/kitchen_14.png"
    N_RUNS  = 3

    run_baseline_comparison(TASK_ID, IMG_A, IMG_B, n_runs=N_RUNS)
