# p2p_utils_nagent.py
# 기존 p2p_utils._norm_agent()는 VALID_AGENTS={"agent_A","agent_B"} 고정 집합과
# "모르면 agent_B" fallback을 전제로 하드코딩되어 있어 N-agent에서 그대로 못 씀.
# 여기서는 agent_id 리스트를 받아 클로저를 만드는 팩토리로 일반화한다.

from __future__ import annotations
from typing import Callable, Dict, List, Optional


def make_norm_agent(
    agent_ids: List[str],
    room_to_agent: Dict[str, str],
) -> Callable[[object], Optional[str]]:
    """
    N-agent용 target_agent 정규화 함수 팩토리.

    agent_ids     : 유효한 agent_id 리스트 (예: ["agent_A","agent_B","agent_C","agent_D"])
    room_to_agent : room_type(소문자) -> agent_id 매핑.
                    VLM이 target_agent에 agent_id 대신 room 이름을 적었을 때 역matching용.

    기존 2-agent _norm_agent()와 달리, "모르면 상대방으로 간주" 같은 fallback을 두지 않는다.
    N=4 이상에서는 "모르면 무조건 B"라는 추측이 틀릴 확률이 훨씬 높기 때문에,
    매칭 안 되면 None을 반환해 PASS/INFORM이 안전하게 폐기되도록 한다
    (p2p_phase._normalize_pass 계열 로직이 target_agent=None인 PASS를 걸러냄).
    """
    valid = set(agent_ids)

    def _norm(x: object) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip()
        if s.lower() in {"", "none", "null", "unknown", "other", "others"} or "|" in s:
            return None
        if s in valid:
            return s
        sl = s.lower().replace("-", "_").replace(" ", "_")
        for aid in agent_ids:
            if sl == aid.lower():
                return aid
        # room 이름으로 들어온 경우 (exact match 우선, 그 다음 부분 문자열 매칭)
        if s.lower() in room_to_agent:
            return room_to_agent[s.lower()]
        for room, aid in room_to_agent.items():
            if room and room in s.lower():
                return aid
        return None

    return _norm
