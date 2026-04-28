# p2p_baseline.py
#
# Baseline Comparison: Centralized / Independent / P2P (Ours)
#
# ─ P2P (Ours)   : p2p_main.run() 그대로
# ─ Centralized  : phase1으로 두 방 observation 추출
#                  → 단일 플래너가 joint plan 생성 (협상 없음)
# ─ Independent  : phase1(우리 프롬프트 그대로) + phase2(use_offer=False)
#                  → rule-based merge (협상 없음, 상대방 정보 없음)
#
# 실행 (Colab):
#   from p2p_baseline import run_baseline_comparison
#   run_baseline_comparison(
#       task_id     = "task_003",
#       image_pairs = [
#           (img_a_path, img_b_path),
#           ...
#       ],
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
    _banner, _log,
    _build_phase1_prompt, _parse_offer,
    _build_phase2_prompt, _parse_local_plan,
    _run_parallel,
    plan_steps_to_dicts,
    format_joint_plan, offer_to_dict,
)
from p2p_main import get_task, run
from p2p_tracker import tracker
from p2p_utils import extract_json, jdump


# ══════════════════════════════════════════════════════════════════════════════
# CENTRALIZED BASELINE
# ══════════════════════════════════════════════════════════════════════════════
#
# Step 1. phase1 프롬프트(우리 것 그대로)로 두 방 observation 추출
# Step 2. 두 observation 합쳐서 단일 플래너에게 joint plan 요청 (협상 없음)

_CENTRALIZED_EXAMPLE = """
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


def _build_centralized_prompt(task: str, offer_a, offer_b) -> str:
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

{_CENTRALIZED_EXAMPLE}

Generate a joint plan for BOTH agents to achieve the task together.

RULES:
1. agent_A steps: ONLY use objects visible in Room A.
2. agent_B steps: ONLY use objects visible in Room B.
3. step_id for agent_A: 1–99. step_id for agent_B: 101–199.
4. Format: "verb + specific visible object + purpose"
5. Generate 4–6 steps per agent (8–12 total).
6. HANDOFF (PASS): if an agent physically carries an item to the doorway:
   - action starts with "carry" or "bring"
   - depends_on=[prep step ids], handoff_type="PASS", target_agent="other_agent"
   - receiving agent must have a receive step with depends_on=[PASS step_id]
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
    task_str = get_task(task_id)

    # Step 1: 우리 phase1 프롬프트 그대로 — observation 추출
    _banner("CENTRALIZED — STEP 1: OBSERVATION (우리 phase1 프롬프트 동일)")
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

    # Step 2: 두 observation 합쳐서 단일 플래너 → joint plan (협상 없음)
    _banner("CENTRALIZED — STEP 2: JOINT PLAN (단일 플래너, 협상 없음)")
    prompt_joint = _build_centralized_prompt(task_str, offer_a, offer_b)
    raw_joint, _ = p2p_vlm.run_vlm(img_a, prompt_joint)
    _log("CENTRALIZED RAW JOINT PLAN", raw_joint)

    data = extract_json(raw_joint)
    if not isinstance(data, dict):
        data = {}

    def _parse(steps_raw, agent_id, room, offset):
        if not isinstance(steps_raw, list):
            return []
        out = []
        for s in steps_raw:
            if not isinstance(s, dict) or "action" not in s:
                continue
            s["agent_id"] = agent_id
            s["room"]     = room
            sid = s.get("step_id", len(out) + 1)
            s["step_id"]  = sid if sid >= offset else sid + offset
            out.append(s)
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
# INDEPENDENT BASELINE
# ══════════════════════════════════════════════════════════════════════════════
#
# Step 1. phase1 프롬프트(우리 것 그대로)로 각자 observation 추출
# Step 2. phase2_local_plan(use_offer=False) — 상대방 정보 없이 각자 계획
#         (coordinate() 사용 안 함 — 상대 draft를 보여주는 구조라 부적합)
# Step 3. rule-based merge — 의미론적 충돌 해결 없음

def _rule_based_merge(steps_a: List[Dict], steps_b: List[Dict]) -> List[Dict]:
    """
    rule-based merge:
    - step_id 충돌 방지 (agent_B offset 적용)
    - time_min 기준 정렬
    - 의미론적 충돌 해결 없음 (Independent의 핵심 약점)
    """
    for s in steps_b:
        if s.get("step_id", 0) < AGENT_B_STEP_OFFSET:
            s["step_id"] = s["step_id"] + AGENT_B_STEP_OFFSET
    merged = list(steps_a) + list(steps_b)
    merged.sort(key=lambda s: (s.get("time_min", 0), s.get("step_id", 0)))
    return merged


def run_independent(task_id: str, img_a: str, img_b: str) -> Dict:
    task_str = get_task(task_id)

    # Step 1: 우리 phase1 프롬프트 그대로 — 각자 observation 추출
    _banner("INDEPENDENT — STEP 1: OBSERVATION (우리 phase1 프롬프트 동일)")
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

    # Step 2: phase2_local_plan(use_offer=False)
    # 상대방 can_provide / need_from_other 미제공 → 완전 독립 계획
    _banner("INDEPENDENT — STEP 2: LOCAL PLANNING (우리 phase2 프롬프트, 상대방 정보 없음)")
    prompt_a = _build_phase2_prompt(offer_a, offer_b, task_str, use_offer=False)
    prompt_b = _build_phase2_prompt(offer_b, offer_a, task_str, use_offer=False)
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
# P2P FULL (OURS)
# ══════════════════════════════════════════════════════════════════════════════

def run_ours(task_id: str, img_a: str, img_b: str) -> Dict:
    """P2P Full (우리 시스템) — p2p_main.run() 그대로."""
    return run(
        task_id = task_id,
        img_a   = img_a,
        img_b   = img_b,
        label   = "P2P (Ours)",
        verbose = "full",
    )


# ══════════════════════════════════════════════════════════════════════════════
# RESULT SAVE
# ══════════════════════════════════════════════════════════════════════════════

def _save_result(result: Dict, pt: float, tc: int, run_idx: int):
    save_dir = Path("/content/KCC_CoRobot/results")
    save_dir.mkdir(exist_ok=True)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    method = result.get("method", result.get("label", "ours"))
    method = method.replace(" ", "_").replace("(", "").replace(")", "")
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
    Baseline Comparison 실행.

    Args:
        task_id     : 실험 태스크 ID (예: "task_003")
        image_pairs : [(img_a, img_b), ...] 이미지 페어 리스트

    Returns:
        PT / TC 요약 DataFrame
    """
    conditions = [
        ("P2P (Ours)",  run_ours),
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
