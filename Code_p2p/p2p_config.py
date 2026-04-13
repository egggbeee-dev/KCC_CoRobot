# config.py

MAX_NEW_TOKENS         = 2048
MAX_CAN_DO             = 8
MAX_CANNOT_DO          = 5
UNCERTAINTY_THRESH     = 0.50
HQ_TOP_K               = 3
MAX_NEGOTIATION_ROUNDS = 3
AGENT_B_STEP_OFFSET    = 100
AUTO_HQ_ANSWER: str | None = None   # None=실제 input(), 문자열=자동응답(batch용)

VALID_REASONS  = {"NO_OBJECT", "NO_CAPABILITY", "UNCERTAIN"}
VALID_AGENTS   = {"agent_A", "agent_B"}

# 협상 제안 field 허용값 (PASS 관련 handoff_type 제거)
VALID_PROPOSAL_FIELDS = {"time_min", "action", "depends_on", "delete"}

FUZZY_STOPWORDS = {
    "the", "a", "an", "and", "or", "with", "on", "in", "at", "to",
    "set", "up", "get", "put", "make", "do", "move", "check", "use",
    "take", "open", "close", "place", "arrange", "clean",
}
