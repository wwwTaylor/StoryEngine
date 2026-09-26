"""Closed projections consumed by individual model operations."""

from __future__ import annotations

from pydantic import model_validator

from story_engine.domain.common import FrozenModel
from story_engine.domain.evaluation import CriterionKind
from story_engine.domain.render import RequirementPhase, ResolvedCriterion
from story_engine.domain.request import PixelSize
from story_engine.storage import ArtifactRef


class StoryPlanningView(FrozenModel):
    idea: str
    shot_target: int
    visual_style: str
    output_language: str
    required_entities: tuple[str, ...]
    forbidden_content: tuple[str, ...]
    supported_durations: tuple[int, ...]
    resolution: PixelSize
    provided_asset_summaries: tuple[str, ...]
    provided_asset_binding_summaries: tuple[str, ...]
    image_input_limit: int
    video_reference_limit: int
    max_beats_per_shot: int
    max_visible_entities_per_shot: int


class VisibleSubject(FrozenModel):
    alias: str
    t0_facts: tuple[str, ...]
    semantic_description: str | None = None


class SubjectReferenceImage(FrozenModel):
    role: str
    artifact_ref: ArtifactRef


class SubjectReference(FrozenModel):
    alias: str
    images: tuple[SubjectReferenceImage, ...]

    @model_validator(mode="after")
    def validate_images(self) -> SubjectReference:
        if not self.images:
            raise ValueError("subject reference images cannot be empty")
        roles = [item.role for item in self.images]
        if len(roles) != len(set(roles)):
            raise ValueError("subject reference image roles must be unique")
        return self


class FirstFrameView(FrozenModel):
    scene_view: ArtifactRef
    scene_panorama: ArtifactRef
    subject_references: tuple[SubjectReference, ...]
    scene_description: str
    visible_subjects: tuple[VisibleSubject, ...]
    ordered_beats: tuple[str, ...]
    shot_constraints: tuple[str, ...]
    composition: str
    camera: str
    visual_style: str
    avoid: tuple[str, ...]

    @property
    def reference_images(self) -> tuple[ArtifactRef, ...]:
        return tuple(
            image.artifact_ref for subject in self.subject_references for image in subject.images
        )


class EvaluationCriterionView(FrozenModel):
    criterion_id: str
    kind: CriterionKind
    priority: int = 50
    statement: str
    phase: RequirementPhase
    category: str = "general"
    owner_shot_key: str | None = None


class ParticipationSubjectView(FrozenModel):
    entity_key: str
    alias: str
    reference_images: tuple[SubjectReferenceImage, ...] = ()


class FirstFrameParticipationView(FrozenModel):
    previous_tail: ArtifactRef
    previous_scene_key: str
    current_scene_key: str
    current_scene_view: ArtifactRef
    current_scene_description: str
    previous_subjects: tuple[ParticipationSubjectView, ...]
    current_subjects: tuple[ParticipationSubjectView, ...]
    current_criteria: tuple[EvaluationCriterionView, ...]

    @model_validator(mode="after")
    def validate_hard_criteria(self) -> FirstFrameParticipationView:
        invalid = tuple(
            item.criterion_id
            for item in self.current_criteria
            if item.kind != CriterionKind.REQUIREMENT
            or (
                item.phase != RequirementPhase.FIRST_FRAME
                and not (item.phase == RequirementPhase.ALWAYS and item.category == "safety")
            )
        )
        if invalid:
            raise ValueError(
                "participation view accepts only first-frame and always-safety requirements: "
                + ", ".join(invalid)
            )
        return self

    @property
    def attachments(self) -> tuple[ArtifactRef, ...]:
        return (
            self.previous_tail,
            self.current_scene_view,
            *(
                image.artifact_ref
                for item in self.current_subjects
                for image in item.reference_images
            ),
        )


class VideoMotionView(FrozenModel):
    start_image: ArtifactRef
    reference_images: tuple[ArtifactRef, ...]
    ordered_beats: tuple[str, ...]
    camera_behavior: str
    reveal_content: tuple[str, ...] = ()
    end_facts: tuple[str, ...]
    avoid: tuple[str, ...]
    duration: int
    resolution: PixelSize
    fps: int


class NovelStationView(FrozenModel):
    panorama: ArtifactRef
    probes: tuple[ArtifactRef, ...]
    scene_description: str
    station_description: str
    composition: str
    landmarks: tuple[str, ...]
    required_content: tuple[str, ...] = ()
    semantic_layout: tuple[str, ...]
    visual_style: str
    avoid: tuple[str, ...]


class EvaluationView(FrozenModel):
    operation: str
    criteria: tuple[EvaluationCriterionView, ...]
    owner_shot_key: str | None = None
    media_instructions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_criterion_owners(self) -> EvaluationView:
        mismatched = tuple(
            item.criterion_id
            for item in self.criteria
            if item.owner_shot_key is not None and item.owner_shot_key != self.owner_shot_key
        )
        if mismatched:
            raise ValueError("evaluation criteria belong to another shot: " + ", ".join(mismatched))
        return self

    @classmethod
    def from_resolved(
        cls,
        operation: str,
        criteria: tuple[ResolvedCriterion, ...],
        *,
        owner_shot_key: str | None = None,
    ) -> EvaluationView:
        return cls(
            operation=operation,
            owner_shot_key=owner_shot_key,
            criteria=tuple(
                EvaluationCriterionView(
                    criterion_id=item.criterion_id,
                    kind=item.kind,
                    priority=item.priority,
                    statement=item.statement,
                    phase=item.phase,
                    category=item.category,
                    owner_shot_key=item.owner_shot_key,
                )
                for item in criteria
            ),
        )


class RenderedPrompt(FrozenModel):
    text: str
    attachments: tuple[ArtifactRef, ...]
