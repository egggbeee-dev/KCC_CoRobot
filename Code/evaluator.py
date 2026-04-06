# ══════════════════════════════════════════════════════════════════════════════
# evaluator.py
# LLM-as-Judge: GPT-4o를 사용한 다차원 플랜 평가
#
# 사용법:
#   from evaluator import evaluate, print_evaluation
#   result = run(...)
#   eval_result = evaluate(result, api_key="sk-...")
#   print_evaluation(eval_result)
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import json
import time
from typing import Dict, List, Tuple

from openai import OpenAI

from config import EVAL_METRIC_NAMES, EVAL_WEIGHTS


# ──────────────────────────────────────────────────────────────────────────────
# Judge 프롬프트 템플릿
# ──────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPTS: Dict[str, str] = {

    "TS": """
You are evaluating a multi-agent home planning system.

Task Success (TS): Did the plan successfully address the task
within the OBSERVABLE scope of each agent?

IMPORTANT: Each agent can only see their own room through a single camera.
Agents CANNOT be penalized for not addressing hazards outside their visible scope.
Cannot-do declarations are CORRECT behavior, not failures.

Score 1 if:
- All observable objectives are addressed in the plan
- Agents correctly declared what they cannot do (Cannot-do)
Score 0 if:
- Observable objectives were missed despite being in can_do
- Agents acted on information they couldn't have

Respond ONLY with valid JSON: {"score": 0 or 1, "reason": "one sentence"}
""",

    "PE": """
You are evaluating a multi-agent home planning system.

Plan Executability (PE): Is every action physically executable?

- Does each action use only objects realistically available in that room?
- Are actions specific and actionable (not vague like "secure the area")?
- Could a real robot or person actually perform each step?
- Penalize abstract or ungrounded actions heavily.

Score: 1.0 = all concrete and executable, 0.5 = some vague, 0.0 = mostly abstract
Respond ONLY with valid JSON: {"score": float 0-1, "reason": "one sentence"}
""",

    "OC": """
You are evaluating a multi-agent home planning system.

Observability Consistency (OC): Did agents stay within their observation scope?

Each agent can only observe objects in their own room.
The observation scope of each agent is listed in their offer (obs_scope field) above.

- Did any agent perform actions involving objects outside their obs_scope?
- Did any agent assume knowledge they couldn't have from their single camera view?
- Hallucination = acting on information outside the agent's visible scope.

Score: 1.0 = no violations, 0.5 = minor violations, 0.0 = severe hallucination
Respond ONLY with valid JSON: {"score": float 0-1, "reason": "one sentence"}
""",

    "SC": """
You are evaluating a multi-agent home planning system.

Sequential Coherence (SC): Does the plan flow logically?

- Are preconditions met before dependent steps execute?
- Is the ordering sensible? (e.g., clean before arrange)
- Are there logical contradictions in the sequence?
- Do parallel steps (same time_min) make sense simultaneously?

Score: 1.0 = perfect flow, 0.5 = minor issues, 0.0 = major contradictions
Respond ONLY with valid JSON: {"score": float 0-1, "reason": "one sentence"}
""",

    "CQ": """
You are evaluating a multi-agent home planning system.

Collaboration Quality (CQ): How well do the two agents collaborate?

- Are handoffs (RELAY/INFORM) meaningful and necessary?
- Does information flow appropriately between agents?
- Is workload distributed reasonably?
- Would collaboration actually help more than working alone?

Score: 1.0 = excellent collaboration, 0.5 = some but weak, 0.0 = no real collaboration
Respond ONLY with valid JSON: {"score": float 0-1, "reason": "one sentence"}
""",

    "HQE": """
You are evaluating whether human queries were appropriate in a multi-agent planning system.

{hqe_context}

For each question asked, judge:
1. Was this question genuinely necessary given the trigger condition?
2. Did the human's answer get reflected in the final joint plan?
3. Could the system have resolved this without asking?

If no queries were asked: judge whether queries were actually needed.

Score: 1.0 = all questions necessary and impactful
       0.7 = reasonable but limited impact
       0.4 = some unnecessary or answers not reflected
       0.0 = wasteful or critical uncertainties missed

Respond ONLY with valid JSON: {"score": float 0-1, "reason": "brief judgment per question, then overall"}
""",
}


# ──────────────────────────────────────────────────────────────────────────────
# 컨텍스트 빌더
# ──────────────────────────────────────────────────────────────────────────────

def _format_plan_for_judge(plan: List[Dict]) -> str:
    if not plan:
        return "  (empty)"
    lines = []
    for s in plan:
        dep  = f" deps={s['depends_on']}" if s.get("depends_on") else ""
        hoff = f" [{s['handoff_type']}→{s['target_agent']}]" if s.get("handoff_type") else ""
        lines.append(
            f"  step {s['step_id']:>2} [T={s['time_min']:>2}m] "
            f"[{s['room']:<12}] [{s['agent_id']}]  {s['action']}{hoff}{dep}"
        )
    return "\n".join(lines)


def _build_judge_context(result: Dict) -> str:
    task       = result.get("task", "")
    offer_a    = result.get("offers", {}).get("agent_A", {})
    offer_b    = result.get("offers", {}).get("agent_B", {})
    plan_a     = result.get("local_plans", {}).get("agent_A", {})
    plan_b     = result.get("local_plans", {}).get("agent_B", {})
    joint_plan = result.get("joint_plan", [])
    hq         = result.get("human_answers", {})
    hq_str     = "\n".join(f"  Q: {q}\n  A: {a}" for q, a in hq.items()) if hq else "  (no queries asked)"

    return f"""=== TASK ===
{task}

=== AGENT A — {offer_a.get('room_type', 'unknown room')} ===
Observation: {offer_a.get('observation', '')}
Obs scope: {offer_a.get('obs_scope', '')}
Can-do: {json.dumps(offer_a.get('can_do', []), ensure_ascii=False)}
Cannot-do: {json.dumps([c['action'] for c in offer_a.get('cannot_do', [])], ensure_ascii=False)}

=== AGENT B — {offer_b.get('room_type', 'unknown room')} ===
Observation: {offer_b.get('observation', '')}
Obs scope: {offer_b.get('obs_scope', '')}
Can-do: {json.dumps(offer_b.get('can_do', []), ensure_ascii=False)}
Cannot-do: {json.dumps([c['action'] for c in offer_b.get('cannot_do', [])], ensure_ascii=False)}

=== AGENT A LOCAL PLAN ===
{_format_plan_for_judge(plan_a.get('steps', []))}

=== AGENT B LOCAL PLAN ===
{_format_plan_for_judge(plan_b.get('steps', []))}

=== JOINT PLAN (final output) ===
{_format_plan_for_judge(joint_plan)}

=== HUMAN QUERIES & ANSWERS ===
{hq_str}"""


def _build_hqe_context(result: Dict) -> str:
    triggers   = result.get("hq_triggers", [])
    hq_asked   = result.get("hq_asked", [])
    hq         = result.get("human_answers", {})
    joint_plan = result.get("joint_plan", [])

    trigger_str  = "\n".join(f"  {t}" for t in triggers) if triggers else "  (no triggers)"
    asked_str    = "\n".join(f"  - {q}" for q in hq_asked) if hq_asked else "  (no questions asked)"
    answered_str = "\n".join(f"  Q: {q}\n  A: {a}" for q, a in hq.items()) if hq else "  (no answers given — user skipped)"

    return (
        f"WHY the system decided to ask (trigger conditions):\n{trigger_str}\n\n"
        f"QUESTIONS PRESENTED TO HUMAN:\n{asked_str}\n\n"
        f"ANSWERS RECEIVED:\n{answered_str}\n\n"
        f"NOTE: If questions were asked but no answers given, the system still proceeded with joint planning.\n\n"
        f"FINAL JOINT PLAN:\n{_format_plan_for_judge(joint_plan)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# GPT-4o 호출
# ──────────────────────────────────────────────────────────────────────────────

def _call_gpt_judge(
    client: OpenAI,
    context: str,
    metric_prompt: str,
    metric: str,
    max_retries: int = 3,
) -> Tuple[float, str]:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=300,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert evaluator for multi-agent planning systems. "
                            "Always respond with valid JSON only, no extra text."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{context}\n\n{'='*40}\n{metric_prompt}"
                            if context
                            else metric_prompt
                        ),
                    },
                ],
            )
            raw  = response.choices[0].message.content.strip()
            raw  = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            score = max(0.0, min(1.0, float(data.get("score", 0.0))))
            return score, str(data.get("reason", ""))

        except json.JSONDecodeError:
            print(f"  [WARN] {metric} JSON parse failed (attempt {attempt+1}), retrying...")
            time.sleep(1)
        except Exception as e:
            print(f"  [WARN] {metric} API error: {e} (attempt {attempt+1})")
            time.sleep(2)

    return 0.5, "evaluation failed"


# ──────────────────────────────────────────────────────────────────────────────
# 메인 평가 함수
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(result: Dict, api_key: str, verbose: bool = True) -> Dict:
    """
    run()의 결과를 받아 GPT-4o로 다차원 채점을 수행한다.

    Args:
        result  : main.run()의 반환값
        api_key : OpenAI API key ("sk-...")
        verbose : 진행 상황 출력 여부

    Returns:
        scores, reasons, total, weights, num_asked, num_answered 포함 dict
    """
    client      = OpenAI(api_key=api_key)
    context     = _build_judge_context(result)
    hqe_context = _build_hqe_context(result)

    scores:  Dict[str, float] = {}
    reasons: Dict[str, str]   = {}

    # DC — 대화 비용 (제시된 쿼리 수 기반, 자동 계산)
    num_asked    = len(result.get("hq_asked", []))
    num_answered = len(result.get("human_answers", {}))
    scores["DC"]  = max(0.0, round(1.0 - num_asked * 0.3, 3))
    reasons["DC"] = f"{num_asked} queries asked, {num_answered} answered"
    if verbose:
        print(f"  DC (Dialogue Cost) → auto: {scores['DC']:.3f} ({reasons['DC']})")

    # 나머지 메트릭 — GPT-4o judge
    for metric, prompt in JUDGE_PROMPTS.items():
        if verbose:
            print(f"  Judging {metric} ({EVAL_METRIC_NAMES[metric]})...", end=" ", flush=True)

        if metric == "HQE":
            filled_prompt = prompt.replace("{hqe_context}", hqe_context)
            score, reason = _call_gpt_judge(client, "", filled_prompt, metric)
        else:
            score, reason = _call_gpt_judge(client, context, prompt, metric)

        scores[metric]  = round(score, 3)
        reasons[metric] = reason

        if verbose:
            print(f"→ {score:.3f}")

    total = round(sum(scores.get(m, 0) * w for m, w in EVAL_WEIGHTS.items()), 3)

    return {
        "scores":       scores,
        "reasons":      reasons,
        "total":        total,
        "weights":      EVAL_WEIGHTS,
        "num_asked":    num_asked,
        "num_answered": num_answered,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────────────────────────────────────────

def print_evaluation(eval_result: Dict, label: str = ""):
    title = "LLM-JUDGE EVALUATION (GPT-4o)" + (f" — {label}" if label else "")
    print(f"\n{'█'*68}")
    print(f"  {title}")
    print(f"{'█'*68}")
    print(f"  {'Metric':<32} {'Score':>6}  {'Weight':>6}  {'Contrib':>7}")
    print(f"  {'─'*56}")
    for metric, weight in EVAL_WEIGHTS.items():
        score   = eval_result["scores"].get(metric, 0.0)
        contrib = round(score * weight, 4)
        name    = f"{metric} ({EVAL_METRIC_NAMES[metric]})"
        print(f"  {name:<32} {score:>6.3f}  {weight:>6.2f}  {contrib:>7.4f}")
    print(f"  {'─'*56}")
    print(f"  {'TOTAL (weighted)':<32} {eval_result['total']:>6.3f}")
    print(f"\n  REASONS:")
    for metric, reason in eval_result["reasons"].items():
        print(f"    [{metric}] {reason}")
