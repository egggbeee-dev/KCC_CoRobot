# ══════════════════════════════════════════════════════════════════════════════
# p2p_judge_simple.py
# LLM-as-a-Judge: 7개 지표를 단일 GPT-4o 호출로 평가
#
# 평가 지표 (p2p_config.EVAL_WEIGHTS 기준):
#   TS  Task Success            0.25  ← ground_truth 비교
#   PE  Plan Executability      0.20
#   OC  Observability Consist.  0.20  ← 이미지 직접 참조
#   SC  Sequential Coherence    0.15
#   CQ  Collaboration Quality   0.10
#   HQE Human Query Efficiency  0.05
#   DC  Dialogue Cost           0.05  ← 라운드 수로 자동 고정
#
# 호환: p2p_main.py (metrics 키 구조, rule-based finalize)
#
# 사용법:
#   from p2p_judge_simple import judge, print_report, save_report
#   result = run(task_id="task_001", img_a="...", img_b="...")
#   report = judge(result, img_a="...", img_b="...")
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

from p2p_config import EVAL_WEIGHTS, EVAL_METRIC_NAMES


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 모델
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MetricScore:
    key:      str
    name:     str
    score:    float
    evidence: str
    weight:   float


@dataclass
class JudgeReport:
    task_id:        str
    task:           str
    metrics:        List[MetricScore]
    final_weighted: float
    verdict:        str
    top_issue:      str
    raw_response:   str
    ts_coverage:    Optional[Dict] = None   # embedding 기반 TS 커버리지 상세


# ══════════════════════════════════════════════════════════════════════════════
# tasks.json 로더 (ground_truth 조회)
# ══════════════════════════════════════════════════════════════════════════════

_tasks_cache: Optional[List[Dict]] = None

def _load_ground_truth(task_id: str) -> Dict:
    global _tasks_cache
    if _tasks_cache is None:
        base = Path(__file__).parent.parent
        path = base / "Data" / "Task" / "tasks.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                _tasks_cache = json.load(f)
        else:
            _tasks_cache = []
    for t in _tasks_cache:
        if t.get("id") == task_id:
            return t.get("ground_truth", {})
    return {}



# ══════════════════════════════════════════════════════════════════════════════
# Embedding 기반 TS 계산
# OpenAI text-embedding-3-small 사용
# ══════════════════════════════════════════════════════════════════════════════

_emb_cache: Dict[str, List[float]] = {}   # 텍스트 → embedding 캐시


def _get_embeddings(texts: List[str]) -> List[List[float]]:
    """텍스트 리스트를 embedding으로 변환. 캐시 적용."""
    client = _get_client()
    to_fetch = [t for t in texts if t not in _emb_cache]
    if to_fetch:
        resp = client.embeddings.create(
            model = "text-embedding-3-small",
            input = to_fetch,
        )
        for text, emb_obj in zip(to_fetch, resp.data):
            _emb_cache[text] = emb_obj.embedding
    return [_emb_cache[t] for t in texts]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_ts_coverage(
    ground_truth: Dict,
    joint_plan:   List[Dict],
    full_thresh:  float = 0.80,
    partial_thresh: float = 0.60,
    verbose: bool = False,
) -> Dict:
    """
    Embedding cosine similarity로 GT 대비 joint_plan 커버리지를 계산한다.

    반환값:
        {
          "ts_score"         : float (0–10),
          "coverage_rate"    : float (0–1),
          "full_match"       : int,
          "partial_match"    : int,
          "no_match"         : int,
          "total_gt"         : int,
          "matched_pairs"    : [(gt_item, plan_action, similarity), ...],
          "unmatched_gt"     : [str, ...],
        }
    """
    # GT 항목 전체 펼치기
    gt_items: List[str] = []
    for steps in ground_truth.values():
        gt_items.extend(steps)

    if not gt_items or not joint_plan:
        return {
            "ts_score": 0.0, "coverage_rate": 0.0,
            "full_match": 0, "partial_match": 0, "no_match": len(gt_items),
            "total_gt": len(gt_items), "matched_pairs": [], "unmatched_gt": gt_items,
        }

    plan_actions = [s["action"] for s in joint_plan]

    # embedding 계산
    gt_embs   = _get_embeddings(gt_items)
    plan_embs = _get_embeddings(plan_actions)

    matched_pairs = []
    unmatched_gt  = []
    full_count    = 0
    partial_count = 0
    no_count      = 0

    for gt_text, gt_emb in zip(gt_items, gt_embs):
        sims = [_cosine_similarity(gt_emb, pe) for pe in plan_embs]
        best_idx  = max(range(len(sims)), key=lambda i: sims[i])
        best_sim  = sims[best_idx]
        best_action = plan_actions[best_idx]

        if best_sim >= full_thresh:
            full_count += 1
            matched_pairs.append((gt_text, best_action, round(best_sim, 3)))
        elif best_sim >= partial_thresh:
            partial_count += 1
            matched_pairs.append((gt_text, best_action, round(best_sim, 3)))
        else:
            no_count += 1
            unmatched_gt.append(gt_text)

        if verbose:
            tag = "FULL" if best_sim >= full_thresh else ("PART" if best_sim >= partial_thresh else "MISS")
            print(f"  [{tag} {best_sim:.2f}] GT: {gt_text[:50]}")
            if best_sim >= partial_thresh:
                print(f"           Plan: {best_action[:50]}")

    total   = len(gt_items)
    # 가중 커버리지: full=1.0, partial=0.5
    coverage = (full_count * 1.0 + partial_count * 0.5) / total if total > 0 else 0.0
    ts_score = round(coverage * 10.0, 2)

    return {
        "ts_score"      : ts_score,
        "coverage_rate" : round(coverage, 3),
        "full_match"    : full_count,
        "partial_match" : partial_count,
        "no_match"      : no_count,
        "total_gt"      : total,
        "matched_pairs" : matched_pairs,
        "unmatched_gt"  : unmatched_gt,
    }


# ══════════════════════════════════════════════════════════════════════════════
# OpenAI 클라이언트
# ══════════════════════════════════════════════════════════════════════════════

_client = None

def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY 환경변수가 설정되지 않았습니다.\n"
                "  os.environ['OPENAI_API_KEY'] = 'sk-...' 로 설정하세요."
            )
        _client = OpenAI(api_key=api_key)
        return _client
    except ImportError:
        raise ImportError("openai 패키지가 필요합니다: pip install openai")


def _encode_image(path: str) -> Tuple[str, str]:
    ext  = path.rsplit(".", 1)[-1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png",  "webp": "image/webp"}.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


# ══════════════════════════════════════════════════════════════════════════════
# 프롬프트 생성
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM = """\
You are an expert evaluator for multi-agent collaborative planning.
You will be given two room images, a task description with ground truth,
and a planning session result.
Evaluate the final joint plan across 7 metrics.
Respond ONLY with valid JSON — no prose before or after.\
"""


def _build_prompt(result: Dict, ground_truth: Dict, ts_cov: Optional[Dict] = None) -> str:
    task        = result["task"]
    offers      = result["offers"]
    joint_plan  = result["joint_plan"]
    convergence = result["convergence"]
    n_rounds    = result["negotiation"]["rounds"]
    hq_asked       = result.get("hq_asked", [])
    human_answers  = result.get("human_answers", {})

    # ground_truth 포맷
    gt_lines = []
    for room, steps in ground_truth.items():
        gt_lines.append(f"  [{room}]")
        for s in steps:
            gt_lines.append(f"    - {s}")
    gt_str = "\n".join(gt_lines) if gt_lines else "  (not available)"

    # 협상 히스토리 요약 (proposal 구조 버전 무관하게 처리)
    def _summarize_proposal(p: Dict) -> str:
        # 구버전: proposed_change 필드
        if "proposed_change" in p:
            return p["proposed_change"]
        # 신버전: field + new_value 구조
        if "field" in p and "new_value" in p:
            return f"step{p.get('step_id','?')}.{p['field']}={p['new_value']}"
        return str(p)

    neg_lines = []
    for r in result["negotiation"]["history"]:
        pa = [_summarize_proposal(p) for p in r["proposals_a"]]
        pb = [_summarize_proposal(p) for p in r["proposals_b"]]
        neg_lines.append(
            f"  Round {r['round_num']}: "
            f"A proposed {pa} | B proposed {pb} | locked={r['locked_step_ids']}"
        )
    neg_str = "\n".join(neg_lines) if neg_lines else "  (no negotiation)"

    # human query Q&A 포맷
    hq_lines = []
    for i, q in enumerate(hq_asked, 1):
        ans = human_answers.get(q, "(no answer)")
        hq_lines.append(f"  Q{i}: {q}")
        hq_lines.append(f"  A{i}: {ans}")
    hq_str = "\n".join(hq_lines) if hq_lines else "  (none)"

    # joint plan 요약
    plan_lines = []
    for s in joint_plan:
        hoff = f" [{s['handoff_type']}→{s['target_agent']}]" if s.get("handoff_type") else ""
        dep  = f" deps={s['depends_on']}" if s.get("depends_on") else ""
        plan_lines.append(
            f"  Step {s['step_id']} [T={s['time_min']}m] "
            f"[{s['agent_id']}] {s['action']}{hoff}{dep}"
        )
    plan_str = "\n".join(plan_lines)

    dc_score = round(max(2.5, 10.0 - n_rounds * 2.5), 1)

    # TS coverage 섹션 생성
    if ts_cov:
        ts_lines = [
            f"  Score (embedding): {ts_cov['ts_score']:.1f}/10  "
            f"(coverage={ts_cov['coverage_rate']:.2f})",
            f"  Full match : {ts_cov['full_match']} / {ts_cov['total_gt']}  "
            f"(cosine ≥ 0.80)",
            f"  Partial match: {ts_cov['partial_match']} / {ts_cov['total_gt']}  "
            f"(cosine 0.60–0.79)",
            f"  No match   : {ts_cov['no_match']} / {ts_cov['total_gt']}  "
            f"(cosine < 0.60)",
        ]
        if ts_cov["matched_pairs"]:
            ts_lines.append("  Top matched pairs:")
            for gt_t, plan_t, sim in ts_cov["matched_pairs"][:5]:
                ts_lines.append(f"    [{sim:.2f}] GT: {gt_t[:45]}")
                ts_lines.append(f"           Plan: {plan_t[:45]}")
        if ts_cov["unmatched_gt"]:
            ts_lines.append("  Unmatched GT items (plan lacks these):")
            for u in ts_cov["unmatched_gt"][:5]:
                ts_lines.append(f"    - {u}")
        ts_coverage_str = "\n".join(ts_lines)
    else:
        ts_coverage_str = "  (ground truth not available)"

    return f"""\
[IMAGE A is Agent A's room — {offers['agent_A']['room_type']}]
[IMAGE B is Agent B's room — {offers['agent_B']['room_type']}]

## Task
{task}

## Ground Truth (expected actions per room)
{gt_str}

## TS Coverage (pre-computed via embedding similarity)
{ts_coverage_str}

## Agent A Offer
can_do   : {json.dumps(offers['agent_A']['can_do'], ensure_ascii=False)}
cannot_do: {json.dumps([c['action'] for c in offers['agent_A']['cannot_do']], ensure_ascii=False)}
obs_scope: {offers['agent_A']['obs_scope']}

## Agent B Offer
can_do   : {json.dumps(offers['agent_B']['can_do'], ensure_ascii=False)}
cannot_do: {json.dumps([c['action'] for c in offers['agent_B']['cannot_do']], ensure_ascii=False)}
obs_scope: {offers['agent_B']['obs_scope']}

## Negotiation ({n_rounds} rounds)
{neg_str}
converged={convergence['converged']} | pass_matched={convergence['pass_matched']} | unresolved={convergence['unresolved']}
human_queries ({len(hq_asked)} asked):
{hq_str}

## Final Joint Plan
{plan_str}

## Evaluation Instructions
Score each metric 0.0–10.0 and provide one sentence of evidence.

TS  (Task Success)           : An embedding-based coverage score has been pre-computed.
                               Use the TS Coverage section above as the primary basis.
                               Adjust by ±1.0 only if the embedding score seems clearly wrong
                               (e.g. high similarity but semantically unrelated actions).
                               Do NOT ignore the pre-computed score — anchor your judgment to it.

PE  (Plan Executability)     : Are all steps within each agent's can_do? No cannot_do violations?
                               10=all executable, 7=1-2 borderline, 4=multiple violations, 1=mostly impossible.

OC  (Observability Consist.) : Look at the images. Are all objects/locations in the plan
                               actually visible in the correct agent's image?
                               10=all grounded, 7=1-2 ambiguous, 4=several ungrounded, 1=severe hallucination.

SC  (Sequential Coherence)   : Are depends_on references valid? No temporal conflicts or cycles?
                               10=fully coherent, 7=minor issues, 4=multiple dep errors, 1=structural breakdown.

CQ  (Collaboration Quality)  : Did negotiation improve the plan? Are PASS/INFORM handoffs logical?
                               10=clear improvement, 7=some improvement, 4=unclear effect, 1=worsened plan.

HQE (Human Query Efficiency) : Strictly judge whether each human query was truly NECESSARY.
                               UNNECESSARY = agent could resolve it autonomously, or asked human
                               to make a decision the agent should make, or answer did not improve
                               the plan, or re-confirms something already in the plan.
                               NECESSARY = asks for info that cannot be inferred from images/offers
                               and is genuinely blocking plan execution.
                               If no queries: score 5.0 exactly.
                               10=all necessary and improved plan, 7=mostly necessary,
                               5=none asked, 3=most unnecessary, 1=all unnecessary/harmful.

DC  (Dialogue Cost)          : FIXED. Use exactly {dc_score:.1f} (formula: 10 - {n_rounds}*2.5).
                               Evidence must state: "{n_rounds} negotiation round(s)."

Respond with EXACTLY this JSON:
{{
  "TS":  {{"score": 0.0, "evidence": "one sentence"}},
  "PE":  {{"score": 0.0, "evidence": "one sentence"}},
  "OC":  {{"score": 0.0, "evidence": "one sentence"}},
  "SC":  {{"score": 0.0, "evidence": "one sentence"}},
  "CQ":  {{"score": 0.0, "evidence": "one sentence"}},
  "HQE": {{"score": 0.0, "evidence": "one sentence"}},
  "DC":  {{"score": {dc_score:.1f}, "evidence": "{n_rounds} negotiation round(s)."}}
}}\
"""


# ══════════════════════════════════════════════════════════════════════════════
# LLM 호출
# ══════════════════════════════════════════════════════════════════════════════

def _call_llm(prompt: str, img_a: str, img_b: str) -> str:
    client  = _get_client()
    content = []

    b64, mime = _encode_image(img_a)
    content.append({"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"}})
    content.append({"type": "text", "text": "[IMAGE A — Agent A's room]"})

    b64, mime = _encode_image(img_b)
    content.append({"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"}})
    content.append({"type": "text", "text": "[IMAGE B — Agent B's room]"})

    content.append({"type": "text", "text": prompt})

    resp = client.chat.completions.create(
        model      = "gpt-4o",
        max_tokens = 1200,
        messages   = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": content},
        ],
    )
    return resp.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# JSON 파싱
# ══════════════════════════════════════════════════════════════════════════════

def _parse_response(text: str) -> Dict:
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start = text.find("{")
    if start == -1:
        return {}
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            esc = (not esc) if ch == "\\" else False
            if not esc and ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except Exception:
                        return {}
    return {}


def _safe_score(v: Any) -> float:
    try:
        return max(0.0, min(10.0, float(v)))
    except Exception:
        return 5.0


# ══════════════════════════════════════════════════════════════════════════════
# 메인 함수
# ══════════════════════════════════════════════════════════════════════════════

def judge(
    result:  Dict,
    img_a:   str,
    img_b:   str,
    verbose: bool = True,
) -> JudgeReport:
    """
    run() 결과를 받아 7개 지표를 단일 LLM 호출로 평가한다.

    Args:
        result  : p2p_main.run() 반환값
        img_a   : agent_A 이미지 경로
        img_b   : agent_B 이미지 경로
        verbose : True면 결과 출력

    Returns:
        JudgeReport
    """
    task_id      = result["task_id"]
    ground_truth = _load_ground_truth(task_id)

    if verbose:
        print("\n" + "═" * 60)
        print("  LLM-AS-A-JUDGE")
        print(f"  Task  : {task_id}")
        print(f"  GT    : {list(ground_truth.keys()) if ground_truth else 'not found'}")
        print(f"  Plan  : {len(result['joint_plan'])} steps | "
              f"Rounds: {result['negotiation']['rounds']}")
        print("═" * 60)

    # ── Embedding 기반 TS 사전 계산 ──────────────────────────────────────────
    ts_cov = None
    if ground_truth:
        if verbose:
            print("  [TS] Computing embedding similarity...")
        ts_cov = compute_ts_coverage(
            ground_truth = ground_truth,
            joint_plan   = result["joint_plan"],
            verbose      = False,
        )
        if verbose:
            print(f"  [TS] full={ts_cov['full_match']} partial={ts_cov['partial_match']} "
                  f"miss={ts_cov['no_match']} / total={ts_cov['total_gt']} "
                  f"→ score={ts_cov['ts_score']:.1f}")

    # ── LLM 호출 (TS 수치를 프롬프트에 포함) ─────────────────────────────────
    prompt = _build_prompt(result, ground_truth, ts_cov)
    raw    = _call_llm(prompt, img_a, img_b)
    parsed = _parse_response(raw)

    # DC는 라운드 수로 강제 고정 (LLM 판단 배제)
    n_rounds = result["negotiation"]["rounds"]
    dc_fixed = round(max(2.5, 10.0 - n_rounds * 2.5), 1)

    metrics: List[MetricScore] = []
    for key in EVAL_WEIGHTS:
        entry = parsed.get(key, {})
        if key == "DC":
            score = dc_fixed
        elif key == "TS" and ts_cov is not None:
            # embedding 점수를 강제 고정 (LLM 자의적 판단 배제)
            score = ts_cov["ts_score"]
        else:
            score = _safe_score(entry.get("score", 5.0))
        ev = str(entry.get("evidence", "")).strip() or "(no evidence)"
        if key == "TS" and ts_cov is not None:
            ev = (f"Embedding coverage: {ts_cov['full_match']} full + "
                  f"{ts_cov['partial_match']} partial / {ts_cov['total_gt']} GT items "
                  f"(rate={ts_cov['coverage_rate']:.2f})")
        metrics.append(MetricScore(
            key      = key,
            name     = EVAL_METRIC_NAMES.get(key, key),
            score    = score,
            evidence = ev,
            weight   = EVAL_WEIGHTS[key],
        ))

    weighted  = sum(m.score * m.weight for m in metrics)
    verdict   = "accept" if weighted >= 7.5 else ("revise" if weighted >= 5.0 else "reject")
    worst     = min(metrics, key=lambda m: m.score)
    top_issue = f"{worst.name} ({worst.key}): {worst.score:.1f}/10"

    report = JudgeReport(
        task_id        = task_id,
        task           = result["task"],
        metrics        = metrics,
        final_weighted = round(weighted, 3),
        verdict        = verdict,
        top_issue      = top_issue,
        raw_response   = raw,
        ts_coverage    = ts_cov,
    )

    if verbose:
        print_report(report)

    return report


# ══════════════════════════════════════════════════════════════════════════════
# 출력 / 저장
# ══════════════════════════════════════════════════════════════════════════════

def print_report(report: JudgeReport):
    print(f"\n  {'─'*56}")
    print(f"  {'Key':<5} {'Metric':<28} {'Score':>5}   Evidence")
    print(f"  {'─'*56}")
    for m in report.metrics:
        bar = "█" * int(m.score) + "░" * (10 - int(m.score))
        print(f"  {m.key:<5} {m.name:<28} {m.score:>5.1f}   {bar}")
        print(f"        ↳ {m.evidence[:68]}")
    print(f"  {'─'*56}")
    print(f"  {'WEIGHTED TOTAL':<34} {report.final_weighted:>6.3f} / 10")
    print(f"  {'VERDICT':<34} {report.verdict.upper()}")
    print(f"  {'TOP ISSUE':<34} {report.top_issue}")
    print(f"  {'═'*56}")

    # TS coverage 상세 출력
    if report.ts_coverage:
        cov = report.ts_coverage
        print(f"\n  TS Coverage Detail (embedding cosine similarity)")
        print(f"  {'─'*56}")
        print(f"  Full match  (≥0.80): {cov['full_match']:>3} / {cov['total_gt']}")
        print(f"  Partial     (≥0.60): {cov['partial_match']:>3} / {cov['total_gt']}")
        print(f"  No match    (<0.60): {cov['no_match']:>3} / {cov['total_gt']}")
        print(f"  Coverage rate      : {cov['coverage_rate']:.3f}")
        if cov["unmatched_gt"]:
            print(f"  Unmatched GT items:")
            for u in cov["unmatched_gt"]:
                print(f"    ✗ {u}")
        print(f"  {'─'*56}")


def save_report(report: JudgeReport, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Ablation 배치 평가
# ══════════════════════════════════════════════════════════════════════════════

def judge_batch(
    results: List[Dict],
    img_a:   str,
    img_b:   str,
) -> List[JudgeReport]:
    """
    run_ablation() 결과 리스트 전체를 평가한다.

    사용 예:
        from p2p_ablation import run_ablation
        from p2p_judge_simple import judge_batch, print_batch_summary

        results = run_ablation("task_001", img_a=IMG_A, img_b=IMG_B)
        reports = judge_batch(results, img_a=IMG_A, img_b=IMG_B)
        print_batch_summary(reports)
    """
    reports = []
    for r in results:
        label = r.get("label", r.get("task_id", "?"))
        print(f"\n  Judging: {label}")
        reports.append(judge(r, img_a=img_a, img_b=img_b, verbose=False))
    print_batch_summary(reports)
    return reports


def print_batch_summary(reports: List[JudgeReport]):
    keys = list(EVAL_WEIGHTS.keys())
    print("\n" + "█" * 80)
    print("  ABLATION — LLM JUDGE SUMMARY")
    print("█" * 80)
    header = f"  {'Method':<22} " + "  ".join(f"{k:>5}" for k in keys) + f"  {'Total':>6}  Verdict"
    print(header)
    print(f"  {'─'*74}")
    for r in reports:
        s_str = "  ".join(
            f"{next((m.score for m in r.metrics if m.key == k), 0.0):>5.1f}"
            for k in keys
        )
        print(f"  {r.task_id:<22} {s_str}  {r.final_weighted:>6.3f}  {r.verdict}")
    print()
