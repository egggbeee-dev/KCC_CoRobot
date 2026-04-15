# config.py

MAX_NEW_TOKENS         = 2048
MAX_CAN_DO             = 8
MAX_CANNOT_DO          = 5
UNCERTAINTY_THRESH     = 0.50
HQ_TOP_K               = 3
MAX_NEGOTIATION_ROUNDS = 3
AGENT_B_STEP_OFFSET    = 100
AUTO_HQ_ANSWER: str | None = None

VALID_REASONS  = {"NO_OBJECT", "NO_CAPABILITY", "UNCERTAIN"}
VALID_AGENTS   = {"agent_A", "agent_B"}

# handoff 관련 필드 제거 — 협상은 depends_on으로만
VALID_PROPOSAL_FIELDS = {"time_min", "action", "depends_on", "delete"}

FUZZY_STOPWORDS = {
    "the", "a", "an", "and", "or", "with", "on", "in", "at", "to",
    "set", "up", "get", "put", "make", "do", "move", "check", "use",
    "take", "open", "close", "place", "arrange", "clean",
}
