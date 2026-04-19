# ══════════════════════════════════════════════════════════════════════════════
# p2p_judge.py
# 7개 지표 통합 Judge
#
# 지표별 방식:
#   TS  (0.25) : Embedding + Hungarian 1:1 매칭 → F-score (β=1.5)
#   PE  (0.20) : Embedding 후보 필터 → LLM 위반 확인 → can_do 범위 체크
#   OC  (0.20) : step별 Vision LLM (detail: high)
#   SC  (0.15) : Rule-based (참조오류/사이클/시간충돌) + LLM 역순 탐지
#   CQ  (0.10) : Rule-based (invalid handoff/no change) + LLM 협상 품질
#   HQE (0.05) : query 없으면 VLM으로 판단 / query 있으면 개별 NECESSARY 판단
#   DC  (0.05) : Rule-based. 라운드/메시지 수 정규화
#
# 사용법:
#   from p2p_judge import judge, print_report, save_report
#   result = run(task_id="task_001", img_a="...", img_b="...")
#   report = judge(result, img_a="...", img_b="...")
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import base64
import json
import math
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ── 가중치 / 지표 이름 ────────────────────────────────────────────────────────
EVAL_WEIGHTS = {
    "TS" : 0.25,
    "PE" : 0.20,
    "OC" : 0.20,
    "SC" : 0.15,
    "CQ" : 0.10,
    "HQE": 0.05,
    "DC" : 0.05,
}
EVAL_METRIC_NAMES = {
    "TS" : "Task Success",
    "PE" : "Plan Executability",
    "OC" : "Observability Consistency",
    "SC" : "Sequential Coherence",
    "CQ" : "Collaboration Quality",
    "HQE": "Human Query Efficiency",
    "DC" : "Dialogue Cost",
}

# ── DC 설정 ───────────────────────────────────────────────────────────────────
DC_MAX_ROUNDS   = 4
DC_MAX_MESSAGES = 20
DC_ALPHA        = 0.6
DC_BETA         = 0.4

# ── TS 설정 ───────────────────────────────────────────────────────────────────
TS_FULL_THRESH    = 0.75
TS_PARTIAL_THRESH = 0.55
TS_BETA           = 1.5

# ── PE 설정 ───────────────────────────────────────────────────────────────────
PE_CANDIDATE_THRESH = 0.35
PE_SCOPE_THRESH     = 0.50

# ── SC 설정 ───────────────────────────────────────────────────────────────────
# (rule-based + LLM, 설정값 없음)


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 모델
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MetricScore:
    key      : str
    name     : str
    score    : float
    evidence : str
    weight   : float
    detail   : Optional[Dict] = None


@dataclass
class JudgeReport:
    task_id        : str
    task           : str
    metrics        : List[MetricScore]
    final_weighted : float
    verdict        : str
    top_issue      : str


# ══════════════════════════════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════════════════════════════

_client = None
_emb_cache: Dict[str, List[float]] = {}
_tasks_cache: Optional[List[Dict]] = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경변수를 설정하세요.")
    _client = OpenAI(api_key=api_key)
    return _client


def _get_embeddings(texts: List[str]) -> List[List[float]]:
    client   = _get_client()
    to_fetch = [t for t in texts if t not in _emb_cache]
    if to_fetch:
        resp = client.embeddings.create(model="text-embedding-3-large", input=to_fetch)
        for text, obj in zip(to_fetch, resp.data):
            _emb_cache[text] = obj.embedding
    return [_emb_cache[t] for t in texts]


def _cosine(a: List[float], b: List[float]) -> float:
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _encode_image(path: str) -> Tuple[str, str]:
    ext  = path.rsplit(".", 1)[-1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png",  "webp": "image/webp"}.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def _image_block(img_path: str, label: str) -> List[Dict]:
    b64, mime = _encode_image(img_path)
    return [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
        {"type": "text", "text": f"[{label}]"},
    ]


def _llm(system: str, user_content: Any, max_tokens: int = 400) -> str:
    client = _get_client()
    resp   = client.chat.completions.create(
        model      = "gpt-4o",
        max_tokens = max_tokens,
        messages   = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
    )
    return resp.choices[0].message.content.strip()


def _parse_json(raw: str) -> Dict:
    for pat, flags in [
        (r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE),
        (r"```\s*(.*?)\s*```",     re.DOTALL),
    ]:
        m = re.search(pat, raw, flags)
        if m:
            try: return json.loads(m.group(1))
            except: pass
    start = raw.find("{") if "{" in raw else raw.find("[")
    if start == -1:
        return {}
    try: return json.loads(raw[start:])
    except: return {}


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
# TS — Task Success
# ══════════════════════════════════════════════════════════════════════════════

def _compute_ts(ground_truth: Dict, joint_plan: List[Dict]) -> Tuple[float, str, Dict]:
    gt_items     = [s for steps in ground_truth.values() for s in steps]
    plan_actions = [s["action"] for s in joint_plan]

    if not gt_items or not plan_actions:
        return 0.0, "No GT or plan", {}

    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        _scipy = True
    except ImportError:
        _scipy = False

    gt_embs   = _get_embeddings(gt_items)
    plan_embs = _get_embeddings(plan_actions)

    # 헝가리안 1:1 매칭
    if _scipy:
        import numpy as np
        sim_matrix = np.array([[_cosine(g, p) for p in plan_embs] for g in gt_embs])
        if len(gt_items) <= len(plan_actions):
            row_ind, col_ind = linear_sum_assignment(-sim_matrix)
            pairs = [(r, c, float(sim_matrix[r, c])) for r, c in zip(row_ind, col_ind)]
        else:
            col_ind, row_ind = linear_sum_assignment(-sim_matrix.T)
            matched = set(row_ind)
            pairs   = [(r, c, float(sim_matrix[r, c])) for r, c in zip(row_ind, col_ind)]
            for i in range(len(gt_items)):
                if i not in matched:
                    pairs.append((i, -1, 0.0))
    else:
        pairs, used = [], set()
        for gi, ge in enumerate(gt_embs):
            best_sim, best_pi = -1.0, -1
            for pi, pe in enumerate(plan_embs):
                if pi in used: continue
                s = _cosine(ge, pe)
                if s > best_sim: best_sim, best_pi = s, pi
            if best_pi != -1: used.add(best_pi)
            pairs.append((gi, best_pi, best_sim))

    full_count = partial_count = 0
    weighted_sum = 0.0
    matched_plan_set: Set[int] = set()
    span = TS_FULL_THRESH - TS_PARTIAL_THRESH

    for gt_i, plan_i, sim in pairs:
        if sim >= TS_FULL_THRESH:
            full_count += 1
            weighted_sum += 1.0
            if plan_i >= 0: matched_plan_set.add(plan_i)
        elif sim >= TS_PARTIAL_THRESH:
            partial_count += 1
            w = 0.3 + 0.4 * (sim - TS_PARTIAL_THRESH) / span
            weighted_sum += w
            if plan_i >= 0: matched_plan_set.add(plan_i)

    total_gt   = len(gt_items)
    total_plan = len(plan_actions)
    recall     = weighted_sum / total_gt if total_gt > 0 else 0.0
    precision  = len(matched_plan_set) / total_plan if total_plan > 0 else 0.0

    b2 = TS_BETA ** 2
    denom = b2 * precision + recall
    fs = (1 + b2) * precision * recall / denom if denom > 0 else 0.0
    score = round(fs * 10.0, 2)

    no_match = total_gt - full_count - partial_count
    ev = (f"Hungarian matching: {full_count} full + {partial_count} partial / {total_gt} GT "
          f"| recall={recall:.3f} precision={precision:.3f} F(β={TS_BETA})={fs:.3f}")
    detail = {
        "full_match": full_count, "partial_match": partial_count,
        "no_match": no_match, "total_gt": total_gt,
        "recall": round(recall, 4), "precision": round(precision, 4), "f_score": round(fs, 4),
    }
    return score, ev, detail


# ══════════════════════════════════════════════════════════════════════════════
# PE — Plan Executability
# ══════════════════════════════════════════════════════════════════════════════

def _compute_pe(offers: Dict, joint_plan: List[Dict]) -> Tuple[float, str, Dict]:
    agent_can_do:    Dict[str, List[str]] = {}
    agent_cannot_do: Dict[str, List[str]] = {}
    for ak, offer in offers.items():
        agent_can_do[ak]    = list(offer.get("can_do", []))
        agent_cannot_do[ak] = [c["action"] if isinstance(c, dict) else c
                                for c in offer.get("cannot_do", [])]

    all_texts = set()
    for ak in agent_can_do:
        all_texts.update(agent_can_do[ak])
        all_texts.update(agent_cannot_do[ak])
    for s in joint_plan:
        all_texts.add(s["action"])
    if all_texts:
        _get_embeddings(list(all_texts))

    violation_count = out_of_scope_count = llm_calls = 0
    step_tags: Dict[int, str] = {}

    for step in joint_plan:
        action    = step["action"]
        agent_key = step.get("agent_id", "")
        cannot_list = agent_cannot_do.get(agent_key, [])
        can_list    = agent_can_do.get(agent_key, [])

        # 1단계: cannot_do 후보 필터
        violation = False
        if cannot_list:
            action_emb  = _get_embeddings([action])[0]
            cannot_embs = _get_embeddings(cannot_list)
            candidates  = [(cannot_list[i], _cosine(action_emb, cannot_embs[i]))
                           for i in range(len(cannot_list))
                           if _cosine(action_emb, cannot_embs[i]) >= PE_CANDIDATE_THRESH]
            candidates.sort(key=lambda x: x[1], reverse=True)

            if candidates:
                llm_calls += 1
                cannot_str = "\n".join(f"  - {item} (sim={sim:.3f})"
                                        for item, sim in candidates)
                prompt = (f'Agent: {agent_key}\nAction: "{action}"\n'
                          f'Cannot-do rules:\n{cannot_str}\n'
                          f'Does this action violate any cannot-do rule?\n'
                          f'{{"violation": true/false, "matched_rule": "...", "reason": "..."}}')
                raw    = _llm("You are a strict robot task evaluator. "
                              "Respond ONLY with valid JSON.", prompt, max_tokens=150)
                parsed = _parse_json(raw)
                if parsed.get("violation"):
                    violation = True
                    violation_count += 1

        # 2단계: can_do 범위 체크
        out_of_scope = False
        if not violation and can_list:
            action_emb = _get_embeddings([action])[0]
            can_embs   = _get_embeddings(can_list)
            best_sim   = max(_cosine(action_emb, e) for e in can_embs)
            if best_sim < PE_SCOPE_THRESH:
                out_of_scope = True
                out_of_scope_count += 1

        if violation:
            step_tags[step["step_id"]] = "VIOLATION"
        elif out_of_scope:
            step_tags[step["step_id"]] = "OUT_OF_SCOPE"
        else:
            step_tags[step["step_id"]] = "OK"

    total        = len(joint_plan)
    problem_rate = (violation_count + out_of_scope_count) / total if total > 0 else 0.0
    score        = round(10.0 * (1 - problem_rate) ** 2, 2)
    ev = (f"Violations={violation_count} OutOfScope={out_of_scope_count} / {total} steps "
          f"(LLM calls={llm_calls})")
    detail = {"violation_count": violation_count, "out_of_scope_count": out_of_scope_count,
              "total_steps": total, "problem_rate": round(problem_rate, 4),
              "step_tags": step_tags}
    return score, ev, detail


# ══════════════════════════════════════════════════════════════════════════════
# OC — Observability Consistency
# ══════════════════════════════════════════════════════════════════════════════

def _compute_oc(joint_plan: List[Dict], img_map: Dict[str, str],
                room_map: Dict[str, str]) -> Tuple[float, str, Dict]:
    _SYS = ("You are a precise robot vision evaluator. "
            "Respond ONLY with valid JSON. No prose before or after.")
    grounded = ambiguous = hallucinated = 0
    step_details = []

    for step in joint_plan:
        action    = step["action"]
        agent_key = step.get("agent_id", "")
        img_path  = img_map.get(agent_key, "")
        room      = room_map.get(agent_key, agent_key)

        if not img_path or not os.path.exists(img_path):
            verdict, reason = "ambiguous", "Image not found"
        else:
            content = _image_block(img_path, f"{agent_key} — {room}")
            content.append({"type": "text", "text":
                f'Agent: {agent_key}\nRoom: {room}\nAction: "{action}"\n'
                f'Are the objects/locations in this action visible in the image?\n'
                f'{{"verdict": "yes"/"ambiguous"/"no", "reason": "one sentence"}}'})
            raw    = _llm(_SYS, content, max_tokens=150)
            parsed = _parse_json(raw)
            verdict = str(parsed.get("verdict", "ambiguous")).lower()
            reason  = str(parsed.get("reason", "")).strip()
            if verdict not in ("yes", "ambiguous", "no"):
                verdict = "ambiguous"

        if verdict == "yes":       grounded    += 1
        elif verdict == "ambiguous": ambiguous  += 1
        else:                        hallucinated += 1
        step_details.append({"step_id": step["step_id"], "verdict": verdict, "reason": reason})

    total = len(joint_plan)
    score = round(10.0 * (grounded + 0.5 * ambiguous) / total, 2) if total > 0 else 0.0
    ev    = f"Grounded={grounded} Ambiguous={ambiguous} Hallucinated={hallucinated} / {total}"
    detail = {"grounded": grounded, "ambiguous": ambiguous,
              "hallucinated": hallucinated, "total_steps": total,
              "step_details": step_details}
    return score, ev, detail


# ══════════════════════════════════════════════════════════════════════════════
# SC — Sequential Coherence
# ══════════════════════════════════════════════════════════════════════════════

def _dfs_cycles(joint_plan: List[Dict], valid_ids: Set[int]) -> List[int]:
    graph = {s["step_id"]: [] for s in joint_plan}
    for step in joint_plan:
        for d in (step.get("depends_on") or []):
            if d in valid_ids:
                graph[step["step_id"]].append(d)
    color: Dict[int, str] = {sid: "white" for sid in graph}
    in_cycle: Set[int]    = set()

    def dfs(node: int, path: List[int]):
        color[node] = "gray"
        path.append(node)
        for nb in graph.get(node, []):
            if color[nb] == "gray":
                for n in path[path.index(nb):]:
                    in_cycle.add(n)
            elif color[nb] == "white":
                dfs(nb, path)
        path.pop()
        color[node] = "black"

    for sid in list(graph.keys()):
        if color[sid] == "white":
            dfs(sid, [])
    return sorted(in_cycle)


def _compute_sc(joint_plan: List[Dict]) -> Tuple[float, str, Dict]:
    valid_ids = {s["step_id"] for s in joint_plan}
    step_map  = {s["step_id"]: s for s in joint_plan}

    # 체크 1: 잘못된 참조
    invalid_ref_steps = []
    for step in joint_plan:
        deps    = step.get("depends_on") or []
        invalid = [d for d in deps if d not in valid_ids]
        if invalid:
            invalid_ref_steps.append(step["step_id"])

    # 체크 2: 순환 의존성
    cycle_steps = _dfs_cycles(joint_plan, valid_ids)

    # 체크 3: 시간 충돌
    time_conflict_steps = []
    for step in joint_plan:
        step_start = step.get("time_min")
        if step_start is None: continue
        for dep_id in (step.get("depends_on") or []):
            if dep_id not in valid_ids: continue
            dep       = step_map[dep_id]
            dep_start = dep.get("time_min")
            dep_dur   = dep.get("duration_min")
            if dep_start is None: continue
            dep_end = dep_start + dep_dur if dep_dur is not None else dep_start
            if step_start < dep_end:
                time_conflict_steps.append(step["step_id"])
                break

    rule_problems = set(invalid_ref_steps + cycle_steps + time_conflict_steps)

    # 체크 4: LLM 의미적 역순 탐지
    pairs = []
    for step in joint_plan:
        if step["step_id"] in rule_problems: continue
        for dep_id in (step.get("depends_on") or []):
            if dep_id in valid_ids and dep_id not in rule_problems:
                pairs.append({
                    "step_id"   : step["step_id"],
                    "dep_id"    : dep_id,
                    "action"    : step["action"],
                    "dep_action": step_map[dep_id]["action"],
                })

    semantic_error_steps: List[int] = []
    if pairs:
        pairs_str = "\n".join(
            f'  {i+1}. "{p["action"]}" cannot start until "{p["dep_action"]}" is finished'
            for i, p in enumerate(pairs)
        )
        prompt = (f"{pairs_str}\n\nJudge whether each order is correct or clearly reversed.\n"
                  f"Mark INVALID only if obviously backwards. Default VALID.\n"
                  f'[{{"pair_index":1,"valid":true/false,"reason":"..."}}...]')
        raw    = _llm("You are a robot task planner evaluator. Respond ONLY with valid JSON.",
                      prompt, max_tokens=600)
        parsed = _parse_json(raw)
        if isinstance(parsed, list):
            for item in parsed:
                idx = item.get("pair_index", 0) - 1
                if 0 <= idx < len(pairs) and not item.get("valid", True):
                    semantic_error_steps.append(pairs[idx]["step_id"])

    problem_steps = sorted(rule_problems | set(semantic_error_steps))
    total         = len(joint_plan)
    error_rate    = len(problem_steps) / total if total > 0 else 0.0
    score         = round(10.0 * (1 - error_rate) ** 2, 2)
    ev = (f"InvalidRef={len(invalid_ref_steps)} Cycles={len(cycle_steps)} "
          f"TimeConflict={len(time_conflict_steps)} Semantic={len(semantic_error_steps)} "
          f"/ {total} steps")
    detail = {
        "invalid_ref_steps": invalid_ref_steps, "cycle_steps": cycle_steps,
        "time_conflict_steps": time_conflict_steps,
        "semantic_error_steps": semantic_error_steps,
        "problem_steps": problem_steps, "error_rate": round(error_rate, 4),
    }
    return score, ev, detail


# ══════════════════════════════════════════════════════════════════════════════
# CQ — Collaboration Quality
# ══════════════════════════════════════════════════════════════════════════════

def _compute_cq(offers: Dict, negotiation: Dict,
                joint_plan: List[Dict]) -> Tuple[float, str, Dict]:
    # 1단계: invalid handoff
    invalid_steps: Set[int] = set()
    for step in joint_plan:
        ht = step.get("handoff_type")
        ta = step.get("target_agent")
        if not ht or not ta: continue
        if ht in ("PASS", "INFORM"): continue
        can_do = offers.get(ta, {}).get("can_do", [])
        action = step.get("action", "")
        matched = any(action.lower() in c.lower() or c.lower() in action.lower()
                      for c in can_do)
        if not matched:
            invalid_steps.add(step["step_id"])

    # 1단계: no change
    initial_actions: Set[str] = set()
    for ak, offer in offers.items():
        for s in offer.get("initial_plan", []):
            initial_actions.add(s.get("action", "").strip().lower())
    final_actions = {s.get("action", "").strip().lower() for s in joint_plan}
    # rounds=0이면 협상 자체가 없었던 경우 → no_change 체크 스킵
    no_change = (negotiation.get("rounds", 0) > 0) and bool(initial_actions) and (initial_actions == final_actions)

    total        = len(joint_plan)
    # no_change 패널티 제거 — 협상 품질은 LLM이 직접 평가
    problem_cnt  = len(invalid_steps)
    problem_rate = min(1.0, problem_cnt / total) if total > 0 else 0.0

    # 2단계: LLM 협상 품질
    initial_str = "\n".join(
        f"  [{ak}] step{s['step_id']}: {s['action']}"
        for ak, offer in offers.items()
        for s in offer.get("initial_plan", [])
    ) or "  (not available)"

    neg_str = "\n".join(
        f"  Round {r['round_num']}: A={[str(p) for p in r.get('proposals_a',[])]}"
        f" B={[str(p) for p in r.get('proposals_b',[])]} locked={r.get('locked_step_ids',[])}"
        for r in negotiation.get("history", [])
    ) or "  (no negotiation)"

    final_str = "\n".join(
        f"  step{s['step_id']} [{s.get('agent_id')}] {s['action']}"
        + (f" [{s.get('handoff_type')}→{s.get('target_agent')}]" if s.get("handoff_type") else "")
        for s in joint_plan
    )

    invalid_str = "\n".join(
        f"  step{sid}: invalid handoff" for sid in invalid_steps
    ) or "  (none)"

    prompt = (f"## Initial Plans\n{initial_str}\n\n"
              f"## Negotiation ({negotiation.get('rounds',0)} rounds)\n{neg_str}\n\n"
              f"## Final Plan\n{final_str}\n\n"
              f"## Invalid Handoffs\n{invalid_str}\n\n"
              f"Evaluate the QUALITY of the negotiation process (0.0-1.0).\n"
              f"Consider:\n"
              f"  - Did agents actively propose meaningful changes?\n"
              f"  - Were conflicts resolved through negotiation?\n"
              f"  - Did the final plan improve over the initial plan?\n"
              f"  - Were handoffs properly coordinated?\n"
              f"Score: 1.0=excellent with clear improvement, 0.7=good with some improvement,\n"
              f"0.5=negotiation happened but little improvement, 0.3=mostly ineffective, 0.0=no meaningful negotiation.\n"
              f'{{"quality_score":0.0,"reason":"one sentence"}}')

    raw    = _llm("You are a robot task planner evaluator. Respond ONLY with valid JSON.",
                  prompt, max_tokens=200)
    parsed = _parse_json(raw)
    quality = max(0.0, min(1.0, float(parsed.get("quality_score", 0.5))))
    reason  = str(parsed.get("reason", "")).strip()

    score = round(10.0 * (1 - problem_rate) * quality, 2)
    ev    = (f"InvalidHandoffs={len(invalid_steps)} NoChange={no_change} "
             f"quality={quality:.2f} — {reason[:50]}")
    detail = {"invalid_handoff_count": len(invalid_steps), "no_change": no_change,
              "problem_rate": round(problem_rate, 4), "negotiation_quality": round(quality, 4)}
    return score, ev, detail


# ══════════════════════════════════════════════════════════════════════════════
# HQE — Human Query Efficiency
# ══════════════════════════════════════════════════════════════════════════════

def _compute_hqe(offers: Dict, joint_plan: List[Dict],
                 hq_asked: List[str], human_answers: Dict[str, str],
                 img_map: Dict[str, str], room_map: Dict[str, str]) -> Tuple[float, str, Dict]:
    _SYS = "You are a robot task planner evaluator. Respond ONLY with valid JSON."

    offer_str = "\n".join(
        f"  [{ak}] can_do={offer.get('can_do',[])} | "
        f"cannot_do={[c['action'] if isinstance(c,dict) else c for c in offer.get('cannot_do',[])]}"
        for ak, offer in offers.items()
    )
    plan_str = "\n".join(
        f"  step{s['step_id']} [{s.get('agent_id')}] {s['action']}" for s in joint_plan
    )

    if not hq_asked:
        # query 없음: 물어봤어야 했는지 판단
        content: List[Dict] = []
        for ak, img_path in img_map.items():
            if img_path and os.path.exists(img_path):
                content.extend(_image_block(img_path, f"{ak} — {room_map.get(ak, ak)}"))
        content.append({"type": "text", "text":
            f"## Agent Offers\n{offer_str}\n\n## Final Plan\n{plan_str}\n\n"
            f"Was there anything the agents could NOT determine from images/offers alone "
            f"and SHOULD have asked the human?\n"
            f'{{"should_have_asked":true/false,"reason":"one sentence"}}'})
        raw    = _llm(_SYS, content, max_tokens=150)
        parsed = _parse_json(raw)
        should = bool(parsed.get("should_have_asked", False))
        reason = str(parsed.get("reason", "")).strip()
        score  = 5.0 if should else 9.0
        ev     = f"No queries. Should have asked={should} — {reason[:55]}"
        detail = {"mode": "no_query", "should_have_asked": should}
        return score, ev, detail

    # query 있음: 개별 NECESSARY 판단
    qa_str = "\n".join(
        f"  Q{i+1}: {q}\n  A{i+1}: {human_answers.get(q,'(no answer)')}"
        for i, q in enumerate(hq_asked)
    )
    content = []
    for ak, img_path in img_map.items():
        if img_path and os.path.exists(img_path):
            content.extend(_image_block(img_path, f"{ak} — {room_map.get(ak, ak)}"))
    content.append({"type": "text", "text":
        f"## Offers\n{offer_str}\n\n## Q&A\n{qa_str}\n\n## Plan\n{plan_str}\n\n"
        f"For each query: NECESSARY if agent couldn't determine from images/offers alone, "
        f"UNNECESSARY otherwise.\n"
        f'[{{"query_index":1,"verdict":"necessary"/"unnecessary","reason":"..."}}...]'})
    raw    = _llm(_SYS, content, max_tokens=600)
    parsed = _parse_json(raw)

    results = parsed if isinstance(parsed, list) else []
    unnecessary = sum(1 for r in results if str(r.get("verdict","")).lower() == "unnecessary")
    necessary   = len(hq_asked) - unnecessary
    rate        = unnecessary / len(hq_asked) if hq_asked else 0.0
    score       = round(10.0 * (1 - rate), 2)
    ev          = f"Necessary={necessary} Unnecessary={unnecessary} / {len(hq_asked)} queries"
    detail      = {"mode": "with_query", "necessary": necessary,
                   "unnecessary": unnecessary, "total": len(hq_asked),
                   "query_details": results}
    return score, ev, detail


# ══════════════════════════════════════════════════════════════════════════════
# DC — Dialogue Cost
# ══════════════════════════════════════════════════════════════════════════════

def _compute_dc(negotiation: Dict) -> Tuple[float, str, Dict]:
    rounds   = negotiation.get("rounds", 0)
    messages = sum(
        len(r.get("proposals_a", [])) + len(r.get("proposals_b", []))
        for r in negotiation.get("history", [])
    )
    # rounds=0이면 협상 자체가 없었던 경우 → 만점
    if rounds == 0:
        return 10.0, "No negotiation needed (0 rounds, 0 messages)", {
            "rounds": 0, "messages": 0, "round_score": 1.0, "message_score": 1.0
        }
    round_score   = max(0.0, 1.0 - (rounds - 1) / DC_MAX_ROUNDS)
    message_score = max(0.0, 1.0 - (max(1, messages) - 1) / DC_MAX_MESSAGES)
    score = round(10.0 * (DC_ALPHA * round_score + DC_BETA * message_score), 2)
    ev    = f"Rounds={rounds} Messages={messages} → round_s={round_score:.3f} msg_s={message_score:.3f}"
    detail = {"rounds": rounds, "messages": messages,
              "round_score": round(round_score, 4),
              "message_score": round(message_score, 4)}
    return score, ev, detail


# ══════════════════════════════════════════════════════════════════════════════
# 메인 judge 함수
# ══════════════════════════════════════════════════════════════════════════════

def judge(
    result:  Dict,
    img_a:   str,
    img_b:   str,
    verbose: bool = True,
) -> JudgeReport:
    """
    p2p_main.run() 결과를 받아 7개 지표를 평가한다.

    Args:
        result  : p2p_main.run() 반환값
        img_a   : agent_A 이미지 경로
        img_b   : agent_B 이미지 경로
        verbose : True면 결과 출력

    Returns:
        JudgeReport
    """
    task_id      = result["task_id"]
    task         = result["task"]
    joint_plan   = result["joint_plan"]
    offers_raw   = result["offers"]
    negotiation  = result["negotiation"]
    hq_asked     = result.get("hq_asked", [])
    human_answers= result.get("human_answers", {})
    ground_truth = _load_ground_truth(task_id)

    img_map  = {"agent_A": img_a, "agent_B": img_b}
    room_map = {
        "agent_A": offers_raw.get("agent_A", {}).get("room_type", "agent_A"),
        "agent_B": offers_raw.get("agent_B", {}).get("room_type", "agent_B"),
    }

    # offers에 initial_plan 주입 (local_plans에서 추출)
    offers = {}
    for ak in ("agent_A", "agent_B"):
        offer_data = dict(offers_raw.get(ak, {}))
        local_plan = result.get("local_plans", {}).get(ak, {})
        offer_data["initial_plan"] = local_plan.get("steps", [])
        offers[ak] = offer_data

    if verbose:
        print("\n" + "═" * 65)
        print("  P2P JUDGE — 7 METRICS")
        print(f"  Task  : {task_id}")
        print(f"  Plan  : {len(joint_plan)} steps | "
              f"Rounds: {negotiation['rounds']} | "
              f"HQ: {len(hq_asked)}")
        print("═" * 65)

    # ── 각 지표 계산 ─────────────────────────────────────────────────────────
    def _run(key: str, fn, *args):
        if verbose: print(f"  [{key}] computing...")
        score, ev, detail = fn(*args)
        if verbose: print(f"  [{key}] score={score:.2f}")
        return score, ev, detail

    ts_score,  ts_ev,  ts_det  = _run("TS",  _compute_ts,  ground_truth, joint_plan)
    pe_score,  pe_ev,  pe_det  = _run("PE",  _compute_pe,  offers, joint_plan)
    oc_score,  oc_ev,  oc_det  = _run("OC",  _compute_oc,  joint_plan, img_map, room_map)
    sc_score,  sc_ev,  sc_det  = _run("SC",  _compute_sc,  joint_plan)
    cq_score,  cq_ev,  cq_det  = _run("CQ",  _compute_cq,  offers, negotiation, joint_plan)
    hqe_score, hqe_ev, hqe_det = _run("HQE", _compute_hqe, offers, joint_plan,
                                       hq_asked, human_answers, img_map, room_map)
    dc_score,  dc_ev,  dc_det  = _run("DC",  _compute_dc,  negotiation)

    score_map = {"TS": ts_score, "PE": pe_score, "OC": oc_score, "SC": sc_score,
                 "CQ": cq_score, "HQE": hqe_score, "DC": dc_score}
    ev_map    = {"TS": ts_ev,    "PE": pe_ev,    "OC": oc_ev,    "SC": sc_ev,
                 "CQ": cq_ev,    "HQE": hqe_ev,  "DC": dc_ev}
    det_map   = {"TS": ts_det,   "PE": pe_det,   "OC": oc_det,   "SC": sc_det,
                 "CQ": cq_det,   "HQE": hqe_det, "DC": dc_det}

    metrics = [
        MetricScore(
            key     = k,
            name    = EVAL_METRIC_NAMES[k],
            score   = score_map[k],
            evidence= ev_map[k],
            weight  = EVAL_WEIGHTS[k],
            detail  = det_map[k],
        )
        for k in EVAL_WEIGHTS
    ]

    weighted = sum(m.score * m.weight for m in metrics)
    verdict  = "accept" if weighted >= 7.5 else ("revise" if weighted >= 5.0 else "reject")
    worst    = min(metrics, key=lambda m: m.score)
    top_issue= f"{worst.name} ({worst.key}): {worst.score:.1f}/10"

    report = JudgeReport(
        task_id        = task_id,
        task           = task,
        metrics        = metrics,
        final_weighted = round(weighted, 3),
        verdict        = verdict,
        top_issue      = top_issue,
    )

    if verbose:
        print_report(report)

    return report


# ══════════════════════════════════════════════════════════════════════════════
# 출력 / 저장
# ══════════════════════════════════════════════════════════════════════════════

def print_report(report: JudgeReport):
    print(f"\n  {'─'*62}")
    print(f"  {'Key':<5} {'Metric':<30} {'W':>4}  {'Score':>5}  Bar")
    print(f"  {'─'*62}")
    for m in report.metrics:
        bar = "█" * int(m.score) + "░" * (10 - int(m.score))
        print(f"  {m.key:<5} {m.name:<30} {m.weight:>4.2f}  {m.score:>5.1f}  {bar}")
        print(f"        ↳ {m.evidence[:65]}")
    print(f"  {'─'*62}")
    print(f"  {'WEIGHTED TOTAL':<36} {report.final_weighted:>6.3f} / 10")
    print(f"  {'VERDICT':<36} {report.verdict.upper()}")
    print(f"  {'TOP ISSUE':<36} {report.top_issue}")
    print(f"  {'═'*62}")


def save_report(report: JudgeReport, path: str):
    data = asdict(report)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Ablation 배치 평가
# ══════════════════════════════════════════════════════════════════════════════

def judge_batch(
    results: List[Dict],
    img_a:   str,
    img_b:   str,
) -> List[JudgeReport]:
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
    print("  ABLATION — JUDGE SUMMARY")
    print("█" * 80)
    header = f"  {'Method':<22} " + "  ".join(f"{k:>5}" for k in keys) + f"  {'Total':>6}  Verdict"
    print(header)
    print(f"  {'─'*76}")
    for r in reports:
        s_str = "  ".join(
            f"{next((m.score for m in r.metrics if m.key == k), 0.0):>5.1f}"
            for k in keys
        )
        print(f"  {r.task_id:<22} {s_str}  {r.final_weighted:>6.3f}  {r.verdict}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# run_and_judge
# ══════════════════════════════════════════════════════════════════════════════

def _extract_img_info(img_path):
    from pathlib import Path
    p = Path(img_path)
    return p.parent.parent.name, p.parent.name, p.name

def _format_joint_plan_table(joint_plan):
    lines = []
    for i, s in enumerate(joint_plan, 1):
        ht  = s.get("handoff_type")
        dep = s.get("depends_on", [])
        ht_str  = " [PASS→]" if ht == "PASS" else (" [INFORM→]" if ht == "INFORM" else "")
        dep_str = f"\n      → waits for: step{dep}" if dep else ""
        lines.append(f"   {i}. [{s.get('room','')}] [{s.get('agent_id','')}]{ht_str}")
        lines.append(f"      {s.get('action','')}{dep_str}")
    return "\n".join(lines)

def print_experiment_table(task_id, task_desc, vlm_model, img_a, img_b, result, report):
    room_a, mode_a, name_a = _extract_img_info(img_a)
    room_b, mode_b, name_b = _extract_img_info(img_b)
    print("\n" + "═"*72)
    print("  EXPERIMENT RESULT")
    print("═"*72)
    print(f"  Task ID   : {task_id}")
    print(f"  Task      : {task_desc[:80]}{'...' if len(task_desc)>80 else ''}")
    print(f"  VLM Model : {vlm_model}")
    print(f"  Agent A   : {room_a}/{mode_a}/{name_a}")
    print(f"  Agent B   : {room_b}/{mode_b}/{name_b}")
    print(f"  {'─'*70}")
    print(f"  FINAL JOINT PLAN ({len(result['joint_plan'])} steps)")
    print(f"  {'─'*70}")
    print(_format_joint_plan_table(result["joint_plan"]))
    print(f"  {'─'*70}")
    print("  LLM-AS-A-JUDGE")
    print(f"  {'─'*70}")
    print(f"  {'Key':<5} {'Metric':<30} {'W':>4}  {'Score':>5}  Bar")
    print(f"  {'─'*70}")
    for m in report.metrics:
        bar = "█"*int(m.score) + "░"*(10-int(m.score))
        print(f"  {m.key:<5} {m.name:<30} {m.weight:>4.2f}  {m.score:>5.1f}  {bar}")
        print(f"        ↳ {m.evidence[:65]}")
    print(f"  {'─'*70}")
    print(f"  WEIGHTED TOTAL : {report.final_weighted:.3f} / 10")
    print(f"  VERDICT        : {report.verdict.upper()}")
    print(f"  TOP ISSUE      : {report.top_issue}")
    print("═"*72)

def run_and_judge(task_id, img_a, img_b, vlm_model="gpt-4o", verbose="summary", save_path=None):
    from p2p_main import run
    import json as _json
    result = run(task_id=task_id, img_a=img_a, img_b=img_b, verbose=verbose)
    report = judge(result, img_a=img_a, img_b=img_b, verbose=False)
    print_experiment_table(task_id, result["task"], vlm_model, img_a, img_b, result, report)
    record = {
        "task_id": task_id, "task": result["task"], "vlm_model": vlm_model,
        "img_a": img_a, "img_b": img_b,
        "joint_plan": result["joint_plan"],
        "judge": {
            "scores": {m.key: m.score for m in report.metrics},
            "total": report.final_weighted,
            "verdict": report.verdict,
            "top_issue": report.top_issue,
        },
        "planning_metrics": result.get("metrics", {}),
    }
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            _json.dump(record, f, ensure_ascii=False, indent=2, default=str)
        print(f"  Saved: {save_path}")
    return result, report, record
