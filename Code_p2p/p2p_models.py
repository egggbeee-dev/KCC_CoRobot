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
    can_provide:     List[str]   # 다른 에이전트에게 제공 가능한 아이템
    need_from_other: List[str]   # 다른 에이전트로부터 필요한 아이템
    uncertain_count: int = 0


@dataclass
class HQEntry:
    step_id:     int
    question_nl: str
    u_step:      float


@dataclass
class PlanStep:
    step_id:       int
    time_min:      int
    room:          str
    agent_id:      str
    action:        str
    preconditions: List[str] = field(default_factory=list)
    depends_on:    List[int] = field(default_factory=list)  # 선행 step_id (cross-agent 포함)
    uncertainty:   float     = 0.0
    notes:         str       = ""


@dataclass
class LocalPlan:
    agent_id: str
    steps:    List[PlanStep]
    U_plan:   float
    hq_list:  List[HQEntry]


class ConflictType:
    TEMPORAL    = "TEMPORAL"     # 같은 시간·같은 방에서 자원 충돌
    DEPENDENCY  = "DEPENDENCY"   # A 준비물을 B가 쓰는데 depends_on 연결 없음
    REDUNDANCY  = "REDUNDANCY"   # 두 에이전트가 같은 작업 중복
    CANNOT_DO   = "CANNOT_DO"    # cannot_do 위반
    OBSERV      = "OBSERVABILITY"# can_do 범위 밖 액션


@dataclass
class ConflictEntry:
    conflict_type: str
    step_ids:      List[int]
    agent_ids:     List[str]
    description:   str
    # DEPENDENCY용: 어떤 step에 depends_on을 추가해야 하는지 힌트
    fix_hint:      str = ""


@dataclass
class NegotiationProposal:
    step_id:   int
    agent_id:  str
    field:     str   # "time_min" | "action" | "depends_on" | "delete"
    new_value: str
    reason:    str


@dataclass
class NegotiationRound:
    round_num:       int
    proposals_a:     List[NegotiationProposal]
    proposals_b:     List[NegotiationProposal]
    locked_step_ids: List[int]


@dataclass
class ConvergenceResult:
    converged:            bool
    no_dep_cycle:         bool
    observability_ok:     bool
    no_missing_deps:      bool   # cross-agent depends_on 누락 없음
    unresolved_conflicts: List[ConflictEntry]


@dataclass
class VerifyResult:
    is_valid:             bool
    errors:               List[str]       = field(default_factory=list)
    warnings:             List[str]       = field(default_factory=list)
    completeness_score:   float           = 0.0
    executability_score:  float           = 0.0
    observability_score:  float           = 0.0
    handoff_score:        float           = 0.0
    sequential_score:     float           = 0.0

    @property
    def total_score(self) -> float:
        return round(
            self.completeness_score  * 0.25 +
            self.executability_score * 0.30 +
            self.observability_score * 0.20 +
            self.sequential_score    * 0.25,
            3,
        )
