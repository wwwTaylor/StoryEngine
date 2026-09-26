"""Persisted fixed-workflow operation results."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from story_engine.domain.common import FrozenModel
from story_engine.domain.evaluation import (
    CandidateRecord,
    EvaluationReport,
    SelectionDecision,
)
from story_engine.domain.first_frame import FirstFrameMode, FirstFrameParticipationResult
from story_engine.domain.reference import ReferenceRecipe, StationSpatialEvidence
from story_engine.domain.spatial import ResolvedSpatialPlan
from story_engine.domain.trace import ProviderPrompt
from story_engine.run_state import AttemptRecord
from story_engine.spatial.camera import CameraConflictKind
from story_engine.spatial.grounding import SceneAnchorMap
from story_engine.spatial.probes import ProbeSet


class MediaOperationResult(FrozenModel):
    operation_key: str
    candidates: tuple[CandidateRecord, ...]
    reports: tuple[EvaluationReport, ...]
    decision: SelectionDecision
    attempts: tuple[AttemptRecord, ...]
    prompts: tuple[ProviderPrompt, ...] = ()


class MediaOperationProgress(FrozenModel):
    operation_key: str
    candidates: tuple[CandidateRecord, ...] = ()
    reports: tuple[EvaluationReport, ...] = ()
    attempts: tuple[AttemptRecord, ...] = ()
    prompts: tuple[ProviderPrompt, ...] = ()


class FirstFrameOperationResult(MediaOperationResult):
    participation: FirstFrameParticipationResult
    effective_mode: FirstFrameMode
    reuse_evaluation: EvaluationReport | None = None

    @model_validator(mode="after")
    def validate_mode_transition(self) -> FirstFrameOperationResult:
        initial = self.participation.initial_mode
        allowed = {
            FirstFrameMode.REUSE: {
                FirstFrameMode.REUSE,
                FirstFrameMode.REFERENCE,
                FirstFrameMode.FRESH,
            },
            FirstFrameMode.REFERENCE: {FirstFrameMode.REFERENCE},
            FirstFrameMode.FRESH: {FirstFrameMode.FRESH},
        }
        if self.effective_mode not in allowed[initial]:
            raise ValueError("first-frame mode transition is invalid")
        if (initial == FirstFrameMode.REUSE) != (self.reuse_evaluation is not None):
            raise ValueError("initial reuse requires exactly one formal reuse evaluation")
        if self.effective_mode == FirstFrameMode.REUSE:
            selected = next(
                (
                    item
                    for item in self.candidates
                    if item.candidate_key == self.decision.selected_candidate
                ),
                None,
            )
            if (
                selected is None
                or selected.artifact_ref != self.participation.tail.frame
                or self.reuse_evaluation is None
                or self.decision.selected_report != self.reuse_evaluation.report_key
            ):
                raise ValueError("effective reuse must select the exact evaluated tail")
        return self


class ReferenceOperationResult(MediaOperationResult):
    recipe: ReferenceRecipe


class SceneAnchorOperationResult(FrozenModel):
    probe_set: ProbeSet
    anchor_map: SceneAnchorMap


class SpatialPreflightShotResult(FrozenModel):
    shot_key: str
    status: str
    conflict_kind: CameraConflictKind | None = None
    detail: str = ""


class PanoramaPreflightResult(FrozenModel):
    kind: Literal["success"] = "success"
    candidate_key: str
    scene_key: str
    station_key: str
    anchor_map_hash: str
    executable: bool
    shots: tuple[SpatialPreflightShotResult, ...]
    correction: str | None = None


class CandidatePreflightFailure(FrozenModel):
    kind: Literal["grounding_failure"] = "grounding_failure"
    candidate_key: str
    scene_key: str
    station_key: str
    reference_hash: str
    executable: Literal[False] = False
    failure_code: str
    failed_target_aliases: tuple[str, ...]
    logical_attempt_refs: tuple[str, ...]
    provider_call_refs: tuple[str, ...]
    response_refs: tuple[str, ...] = ()
    validation_refs: tuple[str, ...] = ()
    correction: str


type PanoramaPreflightValue = Annotated[
    PanoramaPreflightResult | CandidatePreflightFailure,
    Field(discriminator="kind"),
]


class PanoramaPreflightOutcome(FrozenModel):
    outcome: PanoramaPreflightValue


class SpatialOperationResult(FrozenModel):
    evidence: StationSpatialEvidence
    resolution: ResolvedSpatialPlan
