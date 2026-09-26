"""Executable RenderPlan and identity input bindings."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from story_engine.domain.common import FrozenModel
from story_engine.domain.evaluation import CriterionKind
from story_engine.domain.reference import CharacterReferenceView
from story_engine.domain.spatial import SpatialResolutionStatus
from story_engine.domain.state import WorldState
from story_engine.ids import canonical_hash
from story_engine.spatial.camera import NovelStationRecipe, SourceStationRecipe
from story_engine.storage import ArtifactRef


class RequirementPhase(StrEnum):
    ALWAYS = "always"
    FIRST_FRAME = "first_frame"
    MOTION = "motion"
    END = "end"


class ResolvedCriterion(FrozenModel):
    criterion_id: str
    kind: CriterionKind
    priority: int = Field(ge=0, le=100)
    phase: RequirementPhase
    statement: str = Field(min_length=1, max_length=1_000)
    category: str = "general"
    owner_shot_key: str | None = None


class InputBindingBase(FrozenModel):
    kind: str
    entity_key: str


class VisibleInStartFrame(InputBindingBase):
    kind: Literal["visible_in_start_frame"] = "visible_in_start_frame"
    reference_asset: ArtifactRef
    reference_views: tuple[CharacterReferenceView, ...] = ()

    @model_validator(mode="after")
    def validate_reference_views(self) -> VisibleInStartFrame:
        if self.reference_views:
            roles = tuple(item.role for item in self.reference_views)
            if roles != ("front", "side", "back"):
                raise ValueError(
                    "visible character references must be ordered front, side, and back"
                )
            if self.reference_views[0].artifact_ref != self.reference_asset:
                raise ValueError("visible character primary reference must be its front view")
        return self


class AdditionalReference(InputBindingBase):
    kind: Literal["additional_reference"] = "additional_reference"
    reference_asset: ArtifactRef
    provider_channel: str


class SemanticOnly(InputBindingBase):
    kind: Literal["semantic_only"] = "semantic_only"
    description: str


type InputBinding = Annotated[
    VisibleInStartFrame | AdditionalReference | SemanticOnly,
    Field(discriminator="kind"),
]
type RenderCameraRecipe = Annotated[
    SourceStationRecipe | NovelStationRecipe,
    Field(discriminator="kind"),
]


class RenderShot(FrozenModel):
    shot_key: str
    planned_start_boundary: WorldState
    planned_end_boundary: WorldState
    selected_scene_view: ArtifactRef
    selected_scene_panorama: ArtifactRef
    camera_recipe: RenderCameraRecipe
    spatial_resolution_hash: str
    spatial_status: SpatialResolutionStatus
    spatial_failure_codes: tuple[str, ...] = ()
    dropped_spatial_content: tuple[str, ...] = ()
    reveal_anchors: tuple[str, ...] = ()
    camera_motion: str
    input_bindings: tuple[InputBinding, ...]
    start_visible_entities: tuple[str, ...]
    end_visible_entities: tuple[str, ...]
    resolved_requirements: tuple[ResolvedCriterion, ...]


class RenderPlan(FrozenModel):
    render_plan_key: str
    version: int = Field(ge=1)
    story_plan_hash: str
    reference_library_hash: str
    provider_profile_hash: str
    ordered_render_shots: tuple[RenderShot, ...]
    render_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_shot_keys(self) -> RenderPlan:
        keys = [shot.shot_key for shot in self.ordered_render_shots]
        if len(keys) != len(set(keys)):
            raise ValueError("RenderPlan shot keys must be unique")
        return self

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(exclude={"render_plan_hash"}))

    def assert_hash(self) -> None:
        if self.content_hash() != self.render_plan_hash:
            raise ValueError("RenderPlan hash mismatch")
