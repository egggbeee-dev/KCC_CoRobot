# models.py
# PASS/INFORM/handoff 완전 제거.
# 에이전트 간 협력은 depends_on으로만 표현한다.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CannotEntry:
    action: str
    reason: str   # NO_OBJECT | NO_CAPABILITY | UNCERTAIN


@dataclass
class Offer:
    agent_id:        str
    room_type:       str
    observation:     str
    obs_scope:       str
    can_do:          List[str]
    cannot_do:       List[CannotEntry]
    conf:            Dict[str, float]
    can_provide:     List[str]   # 상대에게 제공 가능한 결과물/정보
    need_from_other: List[str]   # 상대로부터 필요한 것
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
    depends_on:    List[int] = field(default_factory=list)
    uncertainty:   float     = 0.0
    notes:         str       = ""


@dataclass
class LocalPlan:
    agent_id: str
    steps:    List[PlanStep]
    U_plan:   float
    hq_list:  List[HQEntry]


class ConflictType:
    TEMPORAL    = "TEMPORAL"
    DEPENDENCY  = "DEPENDENCY"
    REDUNDANCY  = "REDUNDANCY"
    CANNOT_DO   = "CANNOT_DO"
    OBSERV      = "OBSERVABILITY"


@dataclass
class ConflictEntry:
    conflict_type: str
    step_ids:      List[int]
    agent_ids:     List[str]
    description:   str
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
    no_missing_deps:      bool
    unresolved_conflicts: List[ConflictEntry]
