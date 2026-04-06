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
