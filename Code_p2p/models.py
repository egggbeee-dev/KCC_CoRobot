# models.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CannotEntry:
    action: str
    reason: str


@dataclass
class Offer:
    agent_id:        str
    room_type:       str
    observation:     str
    obs_scope:       str
    can_do:          List[str]
    cannot_do:       List[CannotEntry]
    conf:            Dict[str, float]
    can_provide:     List[str]
    need_from_other: List[str]
    uncertain_count: int = 0


@dataclass
class HQEntry:
    step_id:     int
    question_nl: str
    u_step:      float


@dataclass
class Handoff:
    step_id:      int
    action:       str
    handoff_type: str
    target_agent: Optional[str]
    payload:      str = ""


@dataclass
class PlanStep:
    step_id:       int
    time_min:      int
    room:          str
    agent_id:      str
    action:        str
    preconditions: List[str]     = field(default_factory=list)
    depends_on:    List[int]     = field(default_factory=list)
    handoff_type:  Optional[str] = None
    target_agent:  Optional[str] = None
    uncertainty:   float         = 0.0
    notes:         str           = ""


@dataclass
class LocalPlan:
    agent_id: str
    steps:    List[PlanStep]
    U_plan:   float
    hq_list:  List[HQEntry]
    handoffs: List[Handoff]


@dataclass
class LeaderResult:
    leader_id:   str
    follower_id: str
    score_a:     float
    score_b:     float
    reason:      str


# ── 충돌 분류 ──────────────────────────────────────────────────────────────────

class ConflictType:
    TEMPORAL    = "TEMPORAL"       # 같은 시간에 같은 자원 사용
    DEPENDENCY  = "DEPENDENCY"     # 한 에이전트가 다른 에이전트의 결과에 의존하지만 handoff 없음
    REDUNDANCY  = "REDUNDANCY"     # 두 에이전트가 같은 작업을 중복으로 수행
    CANNOT_DO   = "CANNOT_DO"      # 에이전트가 cannot_do 위반 액션 실행 시도
    OBSERV      = "OBSERVABILITY"  # 에이전트가 자기 obs_scope 밖 액션 시도
    HANDOFF     = "HANDOFF"        # sender-receiver 매칭 불일치


@dataclass
class ConflictEntry:
    conflict_type: str          # ConflictType 상수 중 하나
    step_ids:      List[int]    # 충돌에 관련된 step_id 목록
    agent_ids:     List[str]    # 관련 에이전트
    description:   str          # 사람이 읽을 수 있는 충돌 설명


@dataclass
class NegotiationProposal:
    """한 에이전트가 제안하는 단일 수정 제안."""
    step_id:         int
    proposed_change: str   # 변경 내용 (자연어)
    reason:          str   # 변경 이유


@dataclass
class NegotiationRound:
    round_num:       int
    proposals_a:     List[NegotiationProposal]
    proposals_b:     List[NegotiationProposal]
    locked_step_ids: List[int]   # 이 라운드 후 합의된(잠긴) step_id


@dataclass
class VerifyResult:
    is_valid:            bool
    errors:              List[str]
    warnings:            List[str]
    completeness_score:  float = 0.0
    executability_score: float = 0.0
    observability_score: float = 0.0
    handoff_score:       float = 0.0
    sequential_score:    float = 0.0

    @property
    def total_score(self) -> float:
        return round(
            self.completeness_score  * 0.25
            + self.executability_score * 0.25
            + self.observability_score * 0.20
            + self.handoff_score       * 0.20
            + self.sequential_score    * 0.10, 3
        )


@dataclass
class ConvergenceResult:
    """Rule-based 수렴 판단 결과."""
    converged:              bool
    pass_matched:           bool   # 모든 PASS sender-receiver가 매칭됨
    no_dep_cycle:           bool   # temporal dependency cycle 없음
    observability_ok:       bool   # 각 에이전트 액션이 자기 obs_scope 내에 있음
    unresolved_conflicts:   List[ConflictEntry]
