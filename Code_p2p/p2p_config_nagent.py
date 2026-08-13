# p2p_config_nagent.py
# N-agent 확장 설정. 기존 p2p_config.py는 그대로 두고(2-agent KCC 재현용),
# 여기서는 N-agent 관련 상수/유틸만 추가로 정의한다.

from __future__ import annotations
import string


def make_agent_ids(n: int) -> list[str]:
    """N개의 agent_id 생성: agent_A, agent_B, agent_C, agent_D, ..."""
    if n < 2:
        raise ValueError("최소 2개 이상의 agent가 필요합니다.")
    if n > 26:
        raise ValueError("26개 이상의 agent는 지원하지 않습니다 (A-Z 범위).")
    return [f"agent_{string.ascii_uppercase[i]}" for i in range(n)]


# agent별 step_id 오프셋 단위. agent_index * STEP_OFFSET_UNIT 만큼 띄워서
# step_id 충돌 없이 최대 STEP_OFFSET_UNIT-1 개의 스텝을 각 agent가 가질 수 있음.
# 기존 2-agent 코드의 AGENT_B_STEP_OFFSET=100 을 N-agent로 일반화한 버전.
STEP_OFFSET_UNIT = 1000


def step_offset(agent_index: int) -> int:
    return agent_index * STEP_OFFSET_UNIT


def agent_index_from_step_id(step_id: int) -> int:
    return step_id // STEP_OFFSET_UNIT
