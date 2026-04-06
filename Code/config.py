# ══════════════════════════════════════════════════════════════════════════════
# config.py
# 전역 상수 및 하이퍼파라미터
# ══════════════════════════════════════════════════════════════════════════════

# ── 태스크 기본값 ──────────────────────────────────────────────────────────────
TASK         = None
IMAGE_A_PATH = None
IMAGE_B_PATH = None

# ── VLM 생성 설정 ──────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 2048

# ── Offer 파싱 제한 ────────────────────────────────────────────────────────────
MAX_CAN_DO    = 8
MAX_CANNOT_DO = 5

# ── 불확실성 임계값 ────────────────────────────────────────────────────────────
UNCERTAINTY_THRESH = 0.50
HQ_TOP_K           = 3

# ── 리더 선출 가중치 ───────────────────────────────────────────────────────────
ALPHA = 0.40   # coverage
BETA  = 0.30   # mean_conf
GAMMA = 0.20   # unc_ratio (패널티)
DELTA = 0.10   # self_suff

# ── 검증 허용 값 ───────────────────────────────────────────────────────────────
VALID_REASONS  = {"NO_OBJECT", "NO_CAPABILITY", "UNCERTAIN"}
VALID_HANDOFFS = {"DELEGATE", "RELAY", "SYNC", "INFORM"}
VALID_AGENTS   = {"agent_A", "agent_B"}

# ── 퍼지 매칭 불용어 ───────────────────────────────────────────────────────────
FUZZY_STOPWORDS = {
    "the", "a", "an", "and", "or", "with", "on", "in", "at", "to",
    "clean", "arrange", "set", "up", "get", "put", "make", "do",
    "place", "move", "check", "use", "take", "open", "close",
}

# ── LLM-as-Judge 평가 가중치 ───────────────────────────────────────────────────
EVAL_WEIGHTS = {
    "TS":  0.25,   # Task Success
    "PE":  0.20,   # Plan Executability
    "OC":  0.20,   # Observability Consistency
    "SC":  0.15,   # Sequential Coherence
    "CQ":  0.10,   # Collaboration Quality
    "HQE": 0.05,   # Human Query Efficiency
    "DC":  0.05,   # Dialogue Cost
}

EVAL_METRIC_NAMES = {
    "TS":  "Task Success",
    "PE":  "Plan Executability",
    "OC":  "Observability Consistency",
    "SC":  "Sequential Coherence",
    "CQ":  "Collaboration Quality",
    "HQE": "Human Query Efficiency",
    "DC":  "Dialogue Cost",
}
