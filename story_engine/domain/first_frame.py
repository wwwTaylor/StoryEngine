"""Minimal records for deciding how a previous tail participates in a first frame."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from story_engine.domain.common import FrozenModel
from story_engine.domain.evaluation import CriterionStatus, ProviderUsage
from story_engine.storage import ArtifactRef


class FirstFrameMode(StrEnum):
    REUSE = "reuse"
    REFERENCE = "reference"
    FRESH = "fresh"


class TailFrameStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class GuidanceAction(StrEnum):
    PRESERVE = "preserve"
    DO_NOT_COPY = "do_not_copy"


class TailFrameInput(FrozenModel):
    status: TailFrameStatus
    source_video: ArtifactRef | None = None
    frame: ArtifactRef | None = None
    reason: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_shape(self) -> TailFrameInput:
        if self.status == TailFrameStatus.AVAILABLE:
            if self.source_video is None or self.frame is None or self.reason is not None:
                raise ValueError("available tail requires video and frame only")
        elif self.status == TailFrameStatus.UNAVAILABLE:
            if self.source_video is None or self.frame is not None or not self.reason:
                raise ValueError("unavailable tail requires source video and reason")
        elif self.source_video is not None or self.frame is not None or not self.reason:
            raise ValueError("not-required tail accepts only a reason")
        return self


class FirstFrameHardCondition(FrozenModel):
    criterion_id: str
    status: CriterionStatus
    evidence: str = Field(min_length=1, max_length=2_000)


class FirstFrameGuidance(FrozenModel):
    action: GuidanceAction
    canonical_keys: tuple[str, ...] = Field(min_length=1)
    criterion_ids: tuple[str, ...] = Field(min_length=1)
    fact: str = Field(min_length=1, max_length=1_000)


class FirstFrameParticipationAssessment(FrozenModel):
    conditions: tuple[FirstFrameHardCondition, ...]
    guidance: tuple[FirstFrameGuidance, ...] = ()
    rationale: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_unique_rows(self) -> FirstFrameParticipationAssessment:
        criterion_ids = [item.criterion_id for item in self.conditions]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("participation hard conditions must be unique")
        guidance_rows = [
            (item.action, item.canonical_keys, item.criterion_ids, item.fact)
            for item in self.guidance
        ]
        if len(guidance_rows) != len(set(guidance_rows)):
            raise ValueError("participation guidance rows must be unique")
        return self

    @property
    def hard_failure_count(self) -> int:
        return sum(item.status != CriterionStatus.PASS for item in self.conditions)

    @property
    def preserve(self) -> tuple[FirstFrameGuidance, ...]:
        return tuple(item for item in self.guidance if item.action == GuidanceAction.PRESERVE)


class FirstFrameParticipationResult(FrozenModel):
    tail: TailFrameInput
    assessment: FirstFrameParticipationAssessment | None = None
    initial_mode: FirstFrameMode
    hard_failure_count: int | None = Field(default=None, ge=0)
    unavailable_reason: str | None = Field(default=None, max_length=1_000)
    planner_call_ref: str | None = None
    planner_usage: ProviderUsage = ProviderUsage()

    @model_validator(mode="after")
    def validate_resolution(self) -> FirstFrameParticipationResult:
        if self.assessment is None:
            if (
                self.initial_mode != FirstFrameMode.FRESH
                or self.hard_failure_count is not None
                or not self.unavailable_reason
            ):
                raise ValueError("unavailable assessment must resolve to fresh with a reason")
            return self
        if self.tail.status != TailFrameStatus.AVAILABLE:
            raise ValueError("an assessment requires an available tail")
        if self.hard_failure_count != self.assessment.hard_failure_count:
            raise ValueError("hard failure count does not match the assessment")
        if self.unavailable_reason is not None:
            raise ValueError("available assessment cannot have an unavailable reason")
        if self.initial_mode == FirstFrameMode.REUSE and self.hard_failure_count > 1:
            raise ValueError("reuse permits at most one failed or unknown hard condition")
        if self.initial_mode == FirstFrameMode.REFERENCE and (
            self.hard_failure_count < 2 or not self.assessment.preserve
        ):
            raise ValueError("reference requires two failures and preserve guidance")
        if self.initial_mode == FirstFrameMode.FRESH and self.hard_failure_count <= 1:
            raise ValueError("an available assessment with at most one failure must reuse")
        return self
