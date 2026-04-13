# config.py
# 전역 상수 및 하이퍼파라미터

# ── VLM 생성 설정 ──────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 2048

# ── Offer 파싱 제한 ────────────────────────────────────────────────────────────
MAX_CAN_DO    = 8
MAX_CANNOT_DO = 5

# ── 불확실성 임계값 ────────────────────────────────────────────────────────────
UNCERTAINTY_THRESH = 0.50
HQ_TOP_K           = 3

# ── 검증 허용 값 ───────────────────────────────────────────────────────────────
VALID_REASONS  = {"NO_OBJECT", "NO_CAPABILITY", "UNCERTAIN"}

# PASS  : 물리적 물건 전달 (sender → receiver, 다른 방으로 이동)
# INFORM: 상태/완료 정보 공유 (물건 이동 없음)
VALID_HANDOFFS = {"PASS", "INFORM"}

VALID_AGENTS   = {"agent_A", "agent_B"}

# ── 협상 제안 필드 ─────────────────────────────────────────────────────────────
# NegotiationProposal.field 허용값
VALID_PROPOSAL_FIELDS = {"time_min", "action", "handoff_type", "depends_on", "delete"}

# ── 퍼지 매칭 불용어 ───────────────────────────────────────────────────────────
FUZZY_STOPWORDS = {
    "the", "a", "an", "and", "or", "with", "on", "in", "at", "to",
    "clean", "arrange", "set", "up", "get", "put", "make", "do",
    "place", "move", "check", "use", "take", "open", "close",
}

# ── P2P 협상 설정 ──────────────────────────────────────────────────────────────
MAX_NEGOTIATION_ROUNDS = 3   # 협상 최대 라운드 수 (hard limit)

# ── Step ID 오프셋 ─────────────────────────────────────────────────────────────
# Agent B의 step_id를 오프셋해서 A와 충돌 방지 (100번대 = B)
AGENT_B_STEP_OFFSET = 100

# ── Human Query 자동 응답 (batch 실험용) ───────────────────────────────────────
# None이면 실제 input() 호출, 문자열이면 모든 질문에 해당 값으로 자동 응답
AUTO_HQ_ANSWER: str | None = None

# ── LLM-as-Judge 평가 가중치 ───────────────────────────────────────────────────
EVAL_WEIGHTS = {
    "conflict_reduction": 0.35,   # 협상 전후 conflict 수 감소율
    "convergence_rate":   0.25,   # Phase 5 수렴 조건 충족 비율
    "negotiation_rounds": 0.15,   # 사용된 협상 라운드 수 (적을수록 좋음)
    "observability":      0.15,   # obs_scope 준수율
    "handoff_match":      0.10,   # PASS sender-receiver 매칭률
}
