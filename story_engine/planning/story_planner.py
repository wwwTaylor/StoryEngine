"""Minimal wire models for creative StoryDraft output."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from story_engine.domain.common import WireModel
from story_engine.domain.state import AttributeValueType, EntityKind, Scalar
from story_engine.domain.story import (
    CameraSide,
    FrameEdge,
    FramingScale,
    RequirementKind,
    ScreenDirection,
    SemanticRelationKind,
    SpatialPresentation,
    SpatialStrength,
    ViewpointIntentKind,
)


class DraftAttributeDefinition(WireModel):
    key: str
    value_type: AttributeValueType
    allowed_values: list[Scalar] = Field(default_factory=list)
    initial_value: Scalar | None = None
    is_visual: bool = False


class DraftPlacement(WireModel):
    kind: Literal[
        "in_scene_zone",
        "on_surface",
        "in_container",
        "held_by",
        "attached_to",
        "offscreen",
    ]
    scene_alias: str | None = Field(
        default=None,
        description="When used, an alias from StoryDraft.scenes.",
    )
    zone_alias: str | None = Field(
        default=None,
        description="When used, a zone declared in scene_alias.",
    )
    target_entity_alias: str | None = Field(
        default=None,
        description="When used, an alias from StoryDraft.entities.",
    )


class DraftEntity(WireModel):
    alias: str
    kind: EntityKind
    visual_identity: str
    initial_placement: DraftPlacement
    attributes: list[DraftAttributeDefinition] = Field(default_factory=list)
    freeze_appearance: bool = True
    provided_asset_id: str | None = None


class DraftZone(WireModel):
    alias: str
    description: str


class DraftAnchorLandmark(WireModel):
    alias: str
    description: str
    zone_alias: str = Field(description="A zone alias declared in this same scene.")


class DraftSceneRegion(WireModel):
    alias: str
    description: str
    zone_alias: str = Field(description="A zone alias declared in this same scene.")
    representative_anchor_aliases: list[str] = Field(
        default_factory=list,
        description="Local unique anchor aliases declared in this same scene.",
    )


class DraftSemanticRelation(WireModel):
    subject_alias: str = Field(
        description="A zone, scene-region, or anchor alias declared in this same scene."
    )
    relation: SemanticRelationKind
    object_alias: str = Field(
        description="A zone, scene-region, or anchor alias declared in this same scene."
    )


class DraftScene(WireModel):
    alias: str
    visual_identity: str
    zones: list[DraftZone]
    scene_regions: list[DraftSceneRegion] = Field(default_factory=list)
    anchor_landmarks: list[DraftAnchorLandmark] = Field(default_factory=list)
    semantic_relations: list[DraftSemanticRelation] = Field(default_factory=list)
    default_axis: str | None = None
    lighting: str
    style: str
    provided_asset_id: str | None = None


class DraftTransition(WireModel):
    kind: Literal["set_placement", "set_attribute"]
    entity_alias: str = Field(description="An alias from StoryDraft.entities.")
    placement: DraftPlacement | None = None
    attribute_key: str | None = None
    value: Scalar | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> DraftTransition:
        if self.kind == "set_placement":
            if self.placement is None or self.attribute_key is not None or self.value is not None:
                raise ValueError("set_placement requires only placement")
        elif self.placement is not None or self.attribute_key is None or self.value is None:
            raise ValueError("set_attribute requires attribute_key and value")
        return self


class DraftBeat(WireModel):
    action: str
    transition: DraftTransition | None = None
    scene_alias: str | None = Field(
        default=None,
        description=(
            "An alias from StoryDraft.scenes, set only for a mid-shot scene change: "
            "this beat and all later beats take place in that scene until the next "
            "change or the shot end. The shot's scene_alias is the opening scene."
        ),
    )


class DraftSpatialContent(WireModel):
    subject_alias: str = Field(
        description="A scene-region or anchor alias declared in the shot's scene."
    )
    presentation: SpatialPresentation
    strength: SpatialStrength


class DraftViewpointIntent(WireModel):
    kind: ViewpointIntentKind = ViewpointIntentKind.SCENE_LOCATION
    description: str
    translation_expected: bool = False


class DraftShotSpatialIntent(WireModel):
    story_target_aliases: list[str] = Field(
        description="Declared entity aliases that carry the shot's narrative action."
    )
    action_zone_alias: str = Field(description="A zone declared in the shot's scene.")
    spatial_content: list[DraftSpatialContent] = Field(default_factory=list)
    viewpoint_intent: DraftViewpointIntent
    axis_key: str | None = None
    camera_side: CameraSide | None = None
    screen_direction: ScreenDirection | None = None
    entry_edge: FrameEdge | None = None
    exit_edge: FrameEdge | None = None
    framing: str
    framing_scale: FramingScale = Field(
        description="Use tight, medium, or wide to define the intended field-of-view scale."
    )
    camera_motion_intent: str


class DraftShot(WireModel):
    alias: str
    scene_alias: str = Field(description="An alias from StoryDraft.scenes.")
    purpose: str
    duration: int
    visible_entity_aliases: list[str] = Field(description="Aliases from StoryDraft.entities only.")
    beats: list[DraftBeat]
    spatial_intent: DraftShotSpatialIntent


class DraftRequirement(WireModel):
    alias: str
    kind: RequirementKind
    description: str
    priority: int = Field(default=50, ge=0, le=100)
    shot_alias: str = Field(
        description=(
            "Exactly one alias from StoryDraft.shots. The requirement must be independently "
            "pixel-observable in that shot's complete video."
        ),
    )


class DraftPlanInvariant(WireModel):
    alias: str
    kind: Literal["shot_order"]
    description: str
    before_shot_alias: str = Field(description="An alias from StoryDraft.shots.")
    after_shot_alias: str = Field(description="A later alias from StoryDraft.shots.")

    @model_validator(mode="after")
    def validate_distinct_shots(self) -> DraftPlanInvariant:
        if self.before_shot_alias == self.after_shot_alias:
            raise ValueError("shot_order requires two distinct shot aliases")
        return self


class StoryDraft(WireModel):
    entities: list[DraftEntity]
    scenes: list[DraftScene]
    shots: list[DraftShot]
    requirements: list[DraftRequirement] = Field(default_factory=list)
    plan_invariants: list[DraftPlanInvariant] = Field(default_factory=list)
