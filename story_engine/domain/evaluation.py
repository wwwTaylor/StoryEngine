"""Immutable candidate, observation, and selection records."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from story_engine.domain.common import FrozenModel
from story_engine.storage import ArtifactRef


class TechnicalStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


class CriterionKind(StrEnum):
    REQUIREMENT = "requirement"
    PREFERENCE = "preference"


class CriterionStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class SelectionOutcome(StrEnum):
    SELECTED_COMPLIANT = "selected_compliant"
    SELECTED_DEGRADED = "selected_degraded"


class TechnicalFinding(FrozenModel):
    code: str
    passed: bool
    detail: str


class ProviderUsage(FrozenModel):
    call_count: int = Field(default=0, ge=0)
    transport_retries: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0)
    known_cost_usd: float | None = Field(default=None, ge=0)

    @classmethod
    def combine(cls, usages: tuple[ProviderUsage, ...]) -> ProviderUsage:
        known_costs = tuple(
            item.known_cost_usd for item in usages if item.known_cost_usd is not None
        )
        return cls(
            call_count=sum(item.call_count for item in usages),
            transport_retries=sum(item.transport_retries for item in usages),
            elapsed_seconds=sum(item.elapsed_seconds for item in usages),
            known_cost_usd=(sum(known_costs) if known_costs else None),
        )


class CandidateMedia(FrozenModel):
    """One named image participating in candidate evaluation or downstream binding."""

    role: str = Field(min_length=1, max_length=80)
    artifact_ref: ArtifactRef


class CandidateRecord(FrozenModel):
    candidate_key: str
    operation_key: str
    artifact_ref: ArtifactRef | None
    media: tuple[CandidateMedia, ...] = ()
    technical_status: TechnicalStatus
    technical_findings: tuple[TechnicalFinding, ...] = ()
    technical_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    provider_call_ref: str
    provider_call_refs: tuple[str, ...] = ()
    provider_usage: ProviderUsage = ProviderUsage(call_count=1)
    logical_attempt: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_artifact_status(self) -> CandidateRecord:
        if self.technical_status == TechnicalStatus.VALID and self.artifact_ref is None:
            raise ValueError("technically valid candidate requires an artifact")
        roles = [item.role for item in self.media]
        if len(roles) != len(set(roles)):
            raise ValueError("candidate media roles must be unique")
        if (
            self.media
            and self.artifact_ref is not None
            and all(item.artifact_ref != self.artifact_ref for item in self.media)
        ):
            raise ValueError("candidate media must contain its primary artifact")
        return self


class CriterionResult(FrozenModel):
    criterion_id: str
    kind: CriterionKind
    priority: int = Field(default=50, ge=0, le=100)
    status: CriterionStatus
    evidence: str = Field(min_length=1, max_length=2_000)
    category: str = "general"
    owner_shot_key: str | None = None


class EvaluationReport(FrozenModel):
    report_key: str
    candidate_key: str
    criterion_results: tuple[CriterionResult, ...]
    evaluator_ref: str
    evaluator_usage: ProviderUsage = ProviderUsage(call_count=1)

    @model_validator(mode="after")
    def validate_criteria(self) -> EvaluationReport:
        identifiers = [item.criterion_id for item in self.criterion_results]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("criterion results must be unique")
        return self


class RankVector(FrozenModel):
    requirement_failures: int = Field(ge=0)
    requirement_unknowns: int = Field(ge=0)
    priority_loss: int = Field(ge=0)
    continuity_loss: int = Field(ge=0)
    preference_loss: int = Field(ge=0)
    technical_quality: float = Field(ge=0.0, le=1.0)

    def sort_key(self, candidate_key: str) -> tuple[int, int, int, int, int, float, str]:
        return (
            self.requirement_failures,
            self.requirement_unknowns,
            self.priority_loss,
            self.continuity_loss,
            self.preference_loss,
            -self.technical_quality,
            candidate_key,
        )


class CandidateRank(FrozenModel):
    candidate_key: str
    vector: RankVector


class SelectionDecision(FrozenModel):
    decision_key: str
    operation_key: str
    eligible_candidates: tuple[str, ...]
    ranks: tuple[CandidateRank, ...]
    selected_candidate: str
    selected_report: str
    outcome: SelectionOutcome

    @model_validator(mode="after")
    def validate_selected_candidate(self) -> SelectionDecision:
        if self.selected_candidate not in self.eligible_candidates:
            raise ValueError("selected candidate must be eligible")
        rank_keys = {rank.candidate_key for rank in self.ranks}
        if rank_keys != set(self.eligible_candidates):
            raise ValueError("rank vector coverage must equal eligible candidates")
        return self
