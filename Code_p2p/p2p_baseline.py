# p2p_baseline.py
# Baseline 비교 실험용
#
# 방법:
#   independent  : 각자 독립 플랜, 교환/협상 없음
#   sequential   : A 독립 플랜 → B가 A 플랜 보고 플랜 생성, 협상 없음
#   centralized  : 단일 VLM이 두 이미지 모두 보고 joint plan 직접 생성
#   single_vlm   : centralized와 동일 (이름 별칭)
#
# 사용법:
#   from p2p_baseline import run_baseline
#   result = run_baseline("independent", task_id="task_002",
#                         img_a="...", img_b="...")

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from p2p_phases import (
    _banner, _build_phase2_prompt, _build_phase2b_prompt,
    _ensure_pass, _kw, _parse_local_plan, _parse_offer,
    _run_parallel, format_joint_plan, jdump,
    local_plan_to_dict, merge_plans, offer_to_dict,
    plan_steps_to_dicts,
    AGENT_B_STEP_OFFSET,
)
from p2p_models import LocalPlan, Offer
from p2p_vlm import run_vlm
from p2p_utils import compute_joint_uncertainty, _banner as banner
from p2p_main import get_task, list_tasks


def _run_vlm_two_images(img_a: str, img_b: str, prompt: str) -> str:
    """두 이미지를 동시에 VLM에 전달 (Centralized용)."""
    import base64, os
    from p2p_vlm import _load_openai, _openai_client, MAX_NEW_TOKENS

    _load_openai()

    def _encode(path: str):
        ext  = path.rsplit(".", 1)[-1].lower()
        mime = {"jpg":"image/jpeg","jpeg":"image/jpeg",
                "png":"image/png","webp":"image/webp"}.get(ext,"image/jpeg")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return mime, b64

    mime_a, b64_a = _encode(img_a)
    mime_b, b64_b = _encode(img_b)

    response = _openai_client.chat.completions.create(
        model="gpt-4o",
        max_tokens=MAX_NEW_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {"type":"text","text":"Image 1 (kitchen / agent_A):"},
                {"type":"image_url","image_url":{"url":f"data:{mime_a};base64,{b64_a}"}},
                {"type":"text","text":"Image 2 (other room / agent_B):"},
                {"type":"image_url","image_url":{"url":f"data:{mime_b};base64,{b64_b}"}},
                {"type":"text","text":prompt},
            ],
        }],
    )
    return response.choices[0].message.content or ""

# ──────────────────────────────────────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────────────────────────────────────

def _compute_metrics(joint: List[Dict], offer_a: Offer, offer_b: Offer,
                     n_conflicts: int = 0) -> Dict:
    scope_a = set(re.findall(r"\w+", offer_a.obs_scope.lower()))
    scope_b = set(re.findall(r"\w+", offer_b.obs_scope.lower()))
    can_kw_a: set = set()
    can_kw_b: set = set()
    for cd in offer_a.can_do: can_kw_a |= _kw(cd)
    for cd in offer_b.can_do: can_kw_b |= _kw(cd)

    obs_violations = 0
    for s in joint:
        if s.get("handoff_type") == "PASS": continue
        pool = (can_kw_a | scope_a) if s.get("agent_id") == "agent_A" else (can_kw_b | scope_b)
        kw = _kw(s.get("action", ""))
        if kw and pool and not (kw & pool):
            obs_violations += 1

    id_to_agent = {s["step_id"]: s.get("agent_id") for s in joint}
    cross_deps = sum(
        1 for s in joint
        for d in s.get("depends_on", [])
        if id_to_agent.get(d) and id_to_agent[d] != s.get("agent_id")
    )
    pass_steps = {s["step_id"] for s in joint if s.get("handoff_type") == "PASS"}
    all_deps   = {d for s in joint for d in s.get("depends_on", [])}
    matched    = len(pass_steps & all_deps)
    hmr        = matched / max(len(pass_steps), 1) if pass_steps else 1.0

    return {
        "handoff_match_rate": round(hmr, 3),
        "cross_agent_deps":   cross_deps,
        "conflict_reduction": 0.0,
        "observability_rate": round(1.0 - obs_violations / max(len(joint), 1), 3),
        "U_joint":            compute_joint_uncertainty(joint),
        "n_steps":            len(joint),
        "n_pass":             len(pass_steps),
    }


def _empty_offer(agent_id: str, room_type: str) -> Offer:
    from p2p_models import Offer as O
    return O(agent_id=agent_id, room_type=room_type, observation="",
             obs_scope="", can_do=[], cannot_do=[], conf={},
             can_provide=[], need_from_other=[], uncertain_count=0)


# ──────────────────────────────────────────────────────────────────────────────
# 1. INDEPENDENT
#    각자 이미지만 보고 독립 플랜, 교환/협상 없음
# ──────────────────────────────────────────────────────────────────────────────

def _run_independent(task: str, img_a: str, img_b: str,
                     verbose: str = "full") -> Dict:
    _banner("INDEPENDENT — no exchange, no negotiation")

    _INDEP_PROMPT = """\
You are a home agent. Look at the room image carefully.
Task: "{task}"

STRICT VISIBILITY RULE:
- Use ONLY objects literally visible in the image.
- Generate a plan for YOUR ROOM ONLY.
- 3-5 steps, each step uses a visible object.
- No handoffs, no coordination with other agents.
- Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{"room_type": "kitchen or bedroom or living room",
  "obs_scope": "comma-separated visible objects",
  "plan_steps": [
    {{"step_id":1,"time_min":1,"action":"verb + visible object",
      "preconditions":[],"depends_on":[],"handoff_type":null,
      "target_agent":null,"uncertainty":0.1,"notes":""}}
  ]
}}
</JSON>"""

    prompt = _INDEP_PROMPT.format(task=task)
    results = _run_parallel([(img_a, prompt, False), (img_b, prompt, False)])
    raw_a, lp_a = results[0]
    raw_b, lp_b = results[1]

    if verbose == "full":
        print(f"\n[A RAW]\n{raw_a[:400]}")
        print(f"\n[B RAW]\n{raw_b[:400]}")

    # 파싱 — room_type 포함된 JSON 처리
    def _parse_indep(raw: str, agent_id: str, offset: int) -> tuple:
        from p2p_phases import extract_json
        data = extract_json(raw)
        if not isinstance(data, dict): data = {}
        room = str(data.get("room_type", "")).strip() or (
            "kitchen" if agent_id == "agent_A" else "bedroom")
        obs  = str(data.get("obs_scope", "")).strip()
        offer = _empty_offer(agent_id, room)
        offer.obs_scope = obs
        plan_raw = json.dumps({"plan_steps": data.get("plan_steps", [])})
        plan = _parse_local_plan(plan_raw, lp_a if agent_id=="agent_A" else lp_b,
                                 offer, step_offset=offset)
        return offer, plan

    offer_a, plan_a = _parse_indep(raw_a, "agent_A", 0)
    offer_b, plan_b = _parse_indep(raw_b, "agent_B", AGENT_B_STEP_OFFSET)

    print(f"  A: {len(plan_a.steps)} steps | B: {len(plan_b.steps)} steps")

    steps_a = plan_steps_to_dicts(plan_a.steps)
    steps_b = plan_steps_to_dicts(plan_b.steps)
    joint   = merge_plans(steps_a, steps_b, offer_a, offer_b, {}, verbose="minimal")
    return offer_a, offer_b, joint


# ──────────────────────────────────────────────────────────────────────────────
# 2. SEQUENTIAL
#    A 독립 플랜 → B가 A 플랜 보고 자기 플랜 생성, 협상 없음
# ──────────────────────────────────────────────────────────────────────────────

def _run_sequential(task: str, img_a: str, img_b: str,
                    verbose: str = "full") -> Dict:
    _banner("SEQUENTIAL — A first, B sees A's plan")

    # A: 독립 플랜
    _A_PROMPT = """\
You are agent_A (kitchen). Look at the kitchen image carefully.
Task: "{task}"

STRICT VISIBILITY RULE: Use ONLY visible objects.

Generate a plan for YOUR ROOM ONLY. 3-5 steps.
If you can provide something useful to the other room (bedroom/living room),
add a PASS step: carry [item] to doorway.
Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{"room_type":"kitchen","obs_scope":"...","can_provide":["item if any"],
  "plan_steps":[
    {{"step_id":1,"time_min":1,"action":"...","preconditions":[],
      "depends_on":[],"handoff_type":null,"target_agent":null,
      "uncertainty":0.1,"notes":""}}
  ]
}}
</JSON>"""

    raw_a, lp_a = run_vlm(img_a, _A_PROMPT.format(task=task))
    if verbose == "full":
        print(f"\n[A RAW]\n{raw_a[:400]}")

    from p2p_phases import extract_json
    data_a = extract_json(raw_a) if isinstance(extract_json(raw_a), dict) else {}
    offer_a = _empty_offer("agent_A", str(data_a.get("room_type","kitchen")))
    offer_a.obs_scope   = str(data_a.get("obs_scope",""))
    offer_a.can_provide = [str(x) for x in data_a.get("can_provide",[]) if x]
    plan_a_raw = json.dumps({"plan_steps": data_a.get("plan_steps",[])})
    plan_a = _parse_local_plan(plan_a_raw, lp_a, offer_a, step_offset=0)

    # A 플랜 요약 (B에게 전달할 정보)
    a_summary = "\n".join(
        f"  Step {i+1}: {s.action}" +
        (f" [→ PASS to agent_B]" if s.handoff_type=="PASS" else "")
        for i, s in enumerate(plan_a.steps)
    )

    # B: A 플랜 보고 자기 플랜 생성
    _B_PROMPT = f"""\
You are agent_B. Look at your room image carefully.
Task: "{task}"

agent_A's plan (already decided):
{a_summary}

STRICT VISIBILITY RULE: Use ONLY objects visible in YOUR room image.

Generate a plan for YOUR ROOM ONLY.
- If agent_A has a PASS step, add a RECEIVE step.
- Prepare your room for the task goal.
- 3-5 steps total.
Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{"room_type":"bedroom or living room","obs_scope":"...",
  "plan_steps":[
    {{"step_id":101,"time_min":1,"action":"...","preconditions":[],
      "depends_on":[],"handoff_type":null,"target_agent":null,
      "uncertainty":0.1,"notes":""}}
  ]
}}
</JSON>"""

    raw_b, lp_b = run_vlm(img_b, _B_PROMPT)
    if verbose == "full":
        print(f"\n[B RAW]\n{raw_b[:400]}")

    data_b  = extract_json(raw_b) if isinstance(extract_json(raw_b), dict) else {}
    offer_b = _empty_offer("agent_B", str(data_b.get("room_type","bedroom")))
    offer_b.obs_scope = str(data_b.get("obs_scope",""))
    plan_b_raw = json.dumps({"plan_steps": data_b.get("plan_steps",[])})
    plan_b = _parse_local_plan(plan_b_raw, lp_b, offer_b, step_offset=AGENT_B_STEP_OFFSET)

    print(f"  A: {len(plan_a.steps)} steps | B: {len(plan_b.steps)} steps")

    # rule-based PASS 보완
    plan_a, plan_b = _ensure_pass(plan_a, plan_b, offer_a, offer_b)
    steps_a = plan_steps_to_dicts(plan_a.steps)
    steps_b = plan_steps_to_dicts(plan_b.steps)
    joint   = merge_plans(steps_a, steps_b, offer_a, offer_b, {}, verbose="minimal")
    return offer_a, offer_b, joint


# ──────────────────────────────────────────────────────────────────────────────
# 3. CENTRALIZED (= single_vlm)
#    단일 VLM이 두 이미지 모두 보고 joint plan 직접 생성
# ──────────────────────────────────────────────────────────────────────────────

def _run_centralized(task: str, img_a: str, img_b: str,
                     verbose: str = "full") -> Dict:
    _banner("CENTRALIZED — single VLM sees both images")

    _C_PROMPT = f"""\
You are a central planner. You can see TWO room images.
Image 1 is the kitchen (agent_A). Image 2 is the other room (agent_B).
Task: "{task}"

Generate a complete joint plan for BOTH agents.
- agent_A steps: use ONLY objects visible in Image 1 (kitchen)
- agent_B steps: use ONLY objects visible in Image 2 (other room)
- If kitchen prepares something for the other room, include a PASS step.
- The other room should include a RECEIVE step for incoming items.
- 3-5 steps per agent.
Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "agent_A": {{
    "room_type": "kitchen",
    "obs_scope": "...",
    "plan_steps": [
      {{"step_id":1,"time_min":1,"action":"...","preconditions":[],
        "depends_on":[],"handoff_type":null,"target_agent":null,
        "uncertainty":0.1,"notes":""}}
    ]
  }},
  "agent_B": {{
    "room_type": "bedroom or living room",
    "obs_scope": "...",
    "plan_steps": [
      {{"step_id":101,"time_min":1,"action":"...","preconditions":[],
        "depends_on":[],"handoff_type":null,"target_agent":null,
        "uncertainty":0.1,"notes":""}}
    ]
  }}
}}
</JSON>"""

    # 두 이미지를 하나의 VLM 호출로 (image_a + image_b 함께)
    raw    = _run_vlm_two_images(img_a, img_b, _C_PROMPT)
    lp     = []

    if verbose == "full":
        print(f"\n[CENTRALIZED RAW]\n{raw[:600]}")

    from p2p_phases import extract_json
    data = extract_json(raw)
    if not isinstance(data, dict): data = {}

    data_a = data.get("agent_A", {})
    data_b = data.get("agent_B", {})

    offer_a = _empty_offer("agent_A", str(data_a.get("room_type","kitchen")))
    offer_a.obs_scope = str(data_a.get("obs_scope",""))
    offer_b = _empty_offer("agent_B", str(data_b.get("room_type","bedroom")))
    offer_b.obs_scope = str(data_b.get("obs_scope",""))

    plan_a_raw = json.dumps({"plan_steps": data_a.get("plan_steps",[])})
    plan_b_raw = json.dumps({"plan_steps": data_b.get("plan_steps",[])})
    plan_a = _parse_local_plan(plan_a_raw, lp, offer_a, step_offset=0)
    plan_b = _parse_local_plan(plan_b_raw, lp, offer_b, step_offset=AGENT_B_STEP_OFFSET)

    print(f"  A: {len(plan_a.steps)} steps | B: {len(plan_b.steps)} steps")

    plan_a, plan_b = _ensure_pass(plan_a, plan_b, offer_a, offer_b)
    steps_a = plan_steps_to_dicts(plan_a.steps)
    steps_b = plan_steps_to_dicts(plan_b.steps)
    joint   = merge_plans(steps_a, steps_b, offer_a, offer_b, {}, verbose="minimal")
    return offer_a, offer_b, joint


# ──────────────────────────────────────────────────────────────────────────────
# 메인 진입점
# ──────────────────────────────────────────────────────────────────────────────

def run_baseline(
    method:          str,
    task_id:         Optional[str] = None,
    img_a:           Optional[str] = None,
    img_b:           Optional[str] = None,
    label:           Optional[str] = None,
    verbose:         str = "full",
) -> Dict:
    """
    Baseline 실험 실행.

    Args:
        method   : "independent" | "sequential" | "centralized" | "single_vlm"
        task_id  : tasks.json의 ID
        img_a    : agent_A 이미지 경로 (kitchen)
        img_b    : agent_B 이미지 경로 (bedroom/living room)
        label    : 출력 레이블
        verbose  : "full" | "summary" | "minimal"

    Returns:
        결과 dict (metrics 포함)

    Example:
        result = run_baseline("independent", task_id="task_002",
                              img_a="kitchen.png", img_b="bedroom.png")
        result = run_baseline("sequential",  task_id="task_002", ...)
        result = run_baseline("centralized", task_id="task_002", ...)
    """
    if task_id is None:
        list_tasks(); raise ValueError("task_id를 지정해주세요.")
    if not img_a or not img_b:
        raise ValueError("img_a와 img_b 경로를 지정해주세요.")
    if not Path(img_a).exists(): raise FileNotFoundError(f"img_a not found: {img_a}")
    if not Path(img_b).exists(): raise FileNotFoundError(f"img_b not found: {img_b}")

    # single_vlm → centralized 별칭
    if method == "single_vlm": method = "centralized"
    if method not in ("independent", "sequential", "centralized"):
        raise ValueError(f"method must be one of: independent, sequential, centralized, single_vlm")

    task  = get_task(task_id)
    label = label or f"{task_id}_{method}"

    print("\n" + "█" * 68)
    print(f"  BASELINE — {method.upper()} — {label}")
    print("█" * 68)
    print(f"  Task: {task[:80]}{'...' if len(task)>80 else ''}")

    if method == "independent":
        offer_a, offer_b, joint = _run_independent(task, img_a, img_b, verbose)
    elif method == "sequential":
        offer_a, offer_b, joint = _run_sequential(task, img_a, img_b, verbose)
    else:
        offer_a, offer_b, joint = _run_centralized(task, img_a, img_b, verbose)

    metrics = _compute_metrics(joint, offer_a, offer_b)

    print("\n" + "█" * 68)
    print(f"  JOINT PLAN — {label}")
    print("█" * 68)
    print(format_joint_plan(joint, task))

    print(f"\n  METRICS [{method}]")
    print(f"  {'─'*40}")
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")

    return {
        "label":   label,
        "method":  method,
        "task_id": task_id,
        "task":    task,
        "joint_plan": joint,
        "metrics":    metrics,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 배치 실험: 여러 태스크 × 여러 방법 한번에 실행
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(
    task_ids:  List[str],
    img_pairs: List[tuple],
    methods:   List[str] = ("independent", "sequential", "centralized"),
    verbose:   str = "summary",
) -> List[Dict]:
    """
    여러 태스크 × 여러 방법을 순차 실행하고 결과를 반환.

    Args:
        task_ids  : ["task_001", "task_002", ...]
        img_pairs : [(img_a1, img_b1), (img_a2, img_b2), ...]
                    task_ids와 1:1 대응
        methods   : 실행할 방법 목록
        verbose   : "full" | "summary" | "minimal"

    Returns:
        결과 dict 리스트

    Example:
        results = run_experiment(
            task_ids  = ["task_002", "task_003"],
            img_pairs = [("kitchen_1.png","bedroom_1.png"),
                         ("kitchen_2.png","living_1.png")],
            methods   = ["independent", "sequential"],
        )
    """
    results = []
    total = len(task_ids) * len(methods)
    done  = 0

    for (task_id, (img_a, img_b)) in zip(task_ids, img_pairs):
        for method in methods:
            done += 1
            print(f"\n{'='*68}")
            print(f"  [{done}/{total}] {task_id} / {method}")
            print(f"{'='*68}")
            try:
                r = run_baseline(method, task_id=task_id,
                                 img_a=img_a, img_b=img_b, verbose=verbose)
                results.append(r)
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({"task_id": task_id, "method": method,
                                 "error": str(e)})

    # 요약 출력
    print("\n" + "═"*68)
    print("  EXPERIMENT SUMMARY")
    print("═"*68)
    header = f"  {'Task':<12} {'Method':<14} {'HMR':>6} {'CAD':>5} {'OR':>6} {'Steps':>6}"
    print(header)
    print("  " + "─"*60)
    for r in results:
        if "error" in r:
            print(f"  {r['task_id']:<12} {r['method']:<14} ERROR: {r['error'][:30]}")
            continue
        m = r["metrics"]
        print(f"  {r['task_id']:<12} {r['method']:<14} "
              f"{m['handoff_match_rate']:>6.3f} {m['cross_agent_deps']:>5} "
              f"{m['observability_rate']:>6.3f} {m['n_steps']:>6}")

    return results
