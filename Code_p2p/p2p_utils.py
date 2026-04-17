# utils.py
# 공통 유틸리티: 타입 변환, JSON 추출, 퍼지 매칭, 불확실성 계산, 로깅

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional

from p2p_config import FUZZY_STOPWORDS, VALID_AGENTS, VALID_HANDOFFS, VALID_REASONS


# ── 타입 변환 헬퍼 ─────────────────────────────────────────────────────────────

def clamp01(x: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


# ── 정규화 헬퍼 ────────────────────────────────────────────────────────────────

def _norm_reason(r: Any) -> str:
    s = str(r).strip().upper().replace(" ", "_")
    return s if s in VALID_REASONS else "UNCERTAIN"


def _norm_handoff(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    return s if s in VALID_HANDOFFS else None


def _norm_agent(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if s.lower() in {"", "none", "null", "unknown"} or "|" in s:
        return None
    if s in VALID_AGENTS:
        return s
    # room 이름으로 들어온 경우 — 맥락상 상대방 agent_B로 해석
    _ROOM_KW = {"kitchen", "bedroom", "living", "bathroom", "room"}
    if any(kw in s.lower() for kw in _ROOM_KW):
        return "agent_B"  # 기본값: 상대방
    # "agent_a" 같은 소문자 변형
    sl = s.lower().replace("-","_").replace(" ","_")
    if sl == "agent_a": return "agent_A"
    if sl == "agent_b": return "agent_B"
    return None


def _norm_depends(v: Any) -> List[int]:
    if isinstance(v, list):
        out = []
        for x in v:
            if isinstance(x, int):
                out.append(x)
            elif isinstance(x, str):
                m = re.search(r"(\d+)", x)
                if m:
                    out.append(int(m.group(1)))
        return out
    if isinstance(v, str):
        return [int(n) for n in re.findall(r"\d+", v)]
    return []


# ── 로깅 ──────────────────────────────────────────────────────────────────────

def _banner(title: str):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def _log(label: str, content: str):
    print(f"\n[{label}]\n{content}")


# ── 퍼지 매칭 ─────────────────────────────────────────────────────────────────

def _keywords(text: str) -> set:
    return set(re.findall(r"\w+", text.lower())) - FUZZY_STOPWORDS


def _fuzzy_match(a: str, b: str, min_overlap: int = 2) -> bool:
    ka, kb = _keywords(a), _keywords(b)
    return bool(ka and kb and len(ka & kb) >= min_overlap)


def _fuzzy_match_soft(a: str, b: str) -> bool:
    ka, kb = _keywords(a), _keywords(b)
    if not ka or not kb:
        return False
    return len(ka & kb) / min(len(ka), len(kb)) >= 0.4


def _match_conf(conf_raw: Dict[str, float], can_do: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for action in can_do:
        if action in conf_raw:
            out[action] = clamp01(conf_raw[action])
            continue
        best_score, best_val = 0.0, 0.7
        for key, val in conf_raw.items():
            ka, kk = _keywords(action), _keywords(key)
            if ka and kk:
                score = len(ka & kk) / min(len(ka), len(kk))
                if score > best_score:
                    best_score, best_val = score, clamp01(val)
        out[action] = best_val if best_score >= 0.4 else 0.7
    return out


# ── JSON 추출 ─────────────────────────────────────────────────────────────────

def _try_parse(raw: str) -> Dict:
    raw = raw.strip()
    raw = raw.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", raw))
    except Exception:
        return {}


def _balanced_json(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return ""


def extract_json(text: str) -> Dict:
    for pattern, flags in [
        (r"<JSON>(.*?)</JSON>", re.DOTALL | re.IGNORECASE),
        (r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE),
        (r"```\s*(.*?)\s*```", re.DOTALL),
    ]:
        m = re.search(pattern, text, flags)
        if m:
            d = _try_parse(m.group(1))
            if d:
                return d
    return _try_parse(_balanced_json(text))


# ── 불확실성 계산 ─────────────────────────────────────────────────────────────

def compute_token_uncertainty(log_probs: List[float], vocab_size: int = 32000) -> float:
    if not log_probs:
        return 0.5
    log_V = math.log(vocab_size)
    return sum(min(1.0, max(0.0, -lp / log_V)) for lp in log_probs) / len(log_probs)


def compute_plan_uncertainty(step_uncertainties: List[float]) -> float:
    return (
        sum(step_uncertainties) / len(step_uncertainties)
        if step_uncertainties
        else 0.0
    )


def compute_joint_uncertainty(joint: List[Any]) -> float:
    """Joint plan 전체 불확실성 (메트릭용).
    cross-agent PASS/receive step에 가중치 1.5 부여."""
    if not joint:
        return 0.0
    id_to_agent = {s["step_id"]: s.get("agent_id") for s in joint}
    total, count = 0.0, 0
    for s in joint:
        u = float(s.get("uncertainty", 0.2))
        # cross-agent dep 있으면 가중치
        has_cross = any(
            id_to_agent.get(d) and id_to_agent[d] != s.get("agent_id")
            for d in s.get("depends_on", [])
        )
        w = 1.5 if (has_cross or s.get("handoff_type") == "PASS") else 1.0
        total += u * w
        count += w
    return round(total / count, 3) if count > 0 else 0.0


# ── 출력 헬퍼 (verifier 제거 후 여기로 이동) ──────────────────────────────────

def format_joint_plan(plan: List[Dict]) -> str:
    """Joint plan을 읽기 좋은 형태로 출력."""
    if not plan:
        return "  (empty)"
    lines = []
    for s in plan:
        dep  = f" deps={s['depends_on']}" if s.get("depends_on") else ""
        hoff = f" [{s['handoff_type']}→{s['target_agent']}]" if s.get("handoff_type") else ""
        note = f" ({s['notes']})" if s.get("notes") else ""
        lines.append(
            f"  {s['step_id']:>2}. [T={s['time_min']:>2}m] [{s['room']:<12}] [{s['agent_id']}]  "
            f"{s['action']}{hoff}{dep}{note}"
        )
    return "\n".join(lines)
