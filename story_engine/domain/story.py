"""Immutable StoryPlan domain."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from story_engine.domain.common import FrozenModel
from story_engine.domain.state import (
    AttributeDefinition,
    EntityKind,
    Transition,
    WorldRules,
    WorldState,
)
from story_engine.ids import canonical_hash


class SemanticRelationKind(StrEnum):
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    INSIDE = "inside"
    ADJACENT_TO = "adjacent_to"
    CONNECTS_TO = "connects_to"


class RequirementKind(StrEnum):
    REQUIREMENT = "requirement"
    PREFERENCE = "preference"


class RequirementCategory(StrEnum):
    SHOT_LOCAL = "shot_local"
    SAFETY = "safety"


class PlanInvariantKind(StrEnum):
    SHOT_ORDER = "shot_order"


class FramingScale(StrEnum):
    TIGHT = "tight"
    MEDIUM = "medium"
    WIDE = "wide"


class SpatialPresentation(StrEnum):
    SIMULTANEOUS = "simultaneous"
    SEQUENTIAL = "sequential"


class SpatialStrength(StrEnum):
    MUST = "must"
    PREFERRED = "preferred"


class ViewpointIntentKind(StrEnum):
    SCENE_LOCATION = "scene_location"


class CameraSide(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class ScreenDirection(StrEnum):
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"


class FrameEdge(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class ReferenceNeedKind(StrEnum):
    CHARACTER = "character"
    PROP = "prop"
    PROP_STATE = "prop_state"
    SCENE_PANORAMA = "scene_panorama"
    PROVIDED = "provided"


class EntitySpec(FrozenModel):
    entity_key: str
    alias: str
    kind: EntityKind
    visual_identity: str = Field(min_length=1, max_length=2_000)
    attribute_definitions: tuple[AttributeDefinition, ...] = ()
    freeze_appearance: bool = True


class ZoneSpec(FrozenModel):
    zone_key: str
    alias: str
    description: str = Field(min_length=1, max_length=1_000)


class AnchorLandmark(FrozenModel):
    anchor_key: str
    alias: str
    description: str = Field(min_length=1, max_length=1_000)
    zone_key: str


class SceneRegion(FrozenModel):
    region_key: str
    alias: str
    description: str = Field(min_length=1, max_length=1_000)
    zone_key: str
    representative_anchor_keys: tuple[str, ...] = ()


class SemanticRelation(FrozenModel):
    subject_key: str
    relation: SemanticRelationKind
    object_key: str


class SceneSpec(FrozenModel):
    scene_key: str
    alias: str
    visual_identity: str = Field(min_length=1, max_length=4_000)
    zones: tuple[ZoneSpec, ...]
    scene_regions: tuple[SceneRegion, ...] = ()
    anchor_landmarks: tuple[AnchorLandmark, ...] = ()
    semantic_relations: tuple[SemanticRelation, ...] = ()
    default_axis: str | None = None
    lighting: str = Field(min_length=1, max_length=1_000)
    style: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_scene_references(self) -> SceneSpec:
        zone_keys = {zone.zone_key for zone in self.zones}
        if not zone_keys:
            raise ValueError("scene requires at least one zone")
        if len(zone_keys) != len(self.zones):
            raise ValueError("scene zone keys must be unique")
        anchor_keys = {anchor.anchor_key for anchor in self.anchor_landmarks}
        if len(anchor_keys) != len(self.anchor_landmarks):
            raise ValueError("anchor keys must be unique")
        region_keys = {region.region_key for region in self.scene_regions}
        if len(region_keys) != len(self.scene_regions):
            raise ValueError("region keys must be unique")
        if anchor_keys & region_keys:
            raise ValueError("anchor and region key domains must be disjoint")
        anchors_by_key = {anchor.anchor_key: anchor for anchor in self.anchor_landmarks}
        for anchor in self.anchor_landmarks:
            if anchor.zone_key not in zone_keys:
                raise ValueError(f"anchor references unknown zone {anchor.zone_key}")
        for region in self.scene_regions:
            if region.zone_key not in zone_keys:
                raise ValueError(f"region references unknown zone {region.zone_key}")
            if len(region.representative_anchor_keys) != len(
                set(region.representative_anchor_keys)
            ):
                raise ValueError("region representative anchors must be unique")
            for anchor_key in region.representative_anchor_keys:
                representative = anchors_by_key.get(anchor_key)
                if representative is None:
                    raise ValueError("region references an unknown representative anchor")
                if representative.zone_key != region.zone_key:
                    raise ValueError("region representative anchor must belong to the same zone")
        spatial_keys = zone_keys | anchor_keys | region_keys
        for relation in self.semantic_relations:
            if relation.subject_key not in spatial_keys or relation.object_key not in spatial_keys:
                raise ValueError("semantic relation references an unknown spatial key")
            if relation.subject_key == relation.object_key:
                raise ValueError("semantic relation cannot reference itself")
        return self


class SpatialContent(FrozenModel):
    subject_key: str
    presentation: SpatialPresentation
    strength: SpatialStrength


class ViewpointIntent(FrozenModel):
    intent_key: str
    kind: ViewpointIntentKind = ViewpointIntentKind.SCENE_LOCATION
    description: str = Field(min_length=1, max_length=500)
    translation_expected: bool = False


class ShotSpatialIntent(FrozenModel):
    story_targets: tuple[str, ...]
    action_zone: str
    spatial_content: tuple[SpatialContent, ...] = ()
    viewpoint_intent: ViewpointIntent
    axis_key: str | None = None
    camera_side: CameraSide | None = None
    screen_direction: ScreenDirection | None = None
    entry_edge: FrameEdge | None = None
    exit_edge: FrameEdge | None = None
    framing: str = Field(min_length=1, max_length=200)
    framing_scale: FramingScale = FramingScale.MEDIUM
    camera_motion_intent: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def validate_content(self) -> ShotSpatialIntent:
        if not self.story_targets:
            raise ValueError("shot spatial intent requires at least one story target")
        if len(self.story_targets) != len(set(self.story_targets)):
            raise ValueError("story targets must be unique")
        content_keys = [item.subject_key for item in self.spatial_content]
        if len(content_keys) != len(set(content_keys)):
            raise ValueError("spatial content subjects must be unique per shot")
        return self


class Beat(FrozenModel):
    beat_key: str
    action: str = Field(min_length=1, max_length=1_000)
    transition: Transition | None = None


class StoryShot(FrozenModel):
    shot_key: str
    alias: str
    scene_key: str
    purpose: str = Field(min_length=1, max_length=1_000)
    duration: int = Field(gt=0)
    visible_entities: tuple[str, ...]
    beats: tuple[Beat, ...]
    spatial_intent: ShotSpatialIntent
    requirement_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_shot_shape(self) -> StoryShot:
        if not self.beats:
            raise ValueError("shot requires at least one beat")
        if len(self.visible_entities) != len(set(self.visible_entities)):
            raise ValueError("visible_entities must be unique")
        return self


class Requirement(FrozenModel):
    requirement_key: str
    kind: RequirementKind
    description: str = Field(min_length=1, max_length=2_000)
    priority: int = Field(default=50, ge=0, le=100)
    owner_shot_key: str | None = None
    category: RequirementCategory

    @model_validator(mode="after")
    def validate_owner(self) -> Requirement:
        if self.category == RequirementCategory.SHOT_LOCAL and self.owner_shot_key is None:
            raise ValueError("shot-local requirement requires one owner shot")
        if self.category == RequirementCategory.SAFETY and self.owner_shot_key is not None:
            raise ValueError("safety requirement is projected globally and cannot have an owner")
        return self


class PlanInvariant(FrozenModel):
    invariant_key: str
    kind: PlanInvariantKind
    description: str = Field(min_length=1, max_length=2_000)
    before_shot_key: str
    after_shot_key: str

    @model_validator(mode="after")
    def validate_distinct_shots(self) -> PlanInvariant:
        if self.before_shot_key == self.after_shot_key:
            raise ValueError("shot-order invariant requires two distinct shots")
        return self


class ReferenceNeedSpec(FrozenModel):
    need_key: str
    kind: ReferenceNeedKind
    subject_key: str
    visible_state: str | None = None
    provided_asset_id: str | None = None


class StoryPlan(FrozenModel):
    plan_key: str
    version: int = Field(ge=1)
    request_ref: str
    entity_catalog: tuple[EntitySpec, ...]
    scene_catalog: tuple[SceneSpec, ...]
    initial_world: WorldState
    ordered_shots: tuple[StoryShot, ...]
    requirements: tuple[Requirement, ...]
    plan_invariants: tuple[PlanInvariant, ...] = ()
    reference_needs: tuple[ReferenceNeedSpec, ...]
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_requirement_ownership(self) -> StoryPlan:
        shot_keys = [shot.shot_key for shot in self.ordered_shots]
        if len(shot_keys) != len(set(shot_keys)):
            raise ValueError("StoryPlan shot keys must be unique")
        shot_positions = {shot_key: index for index, shot_key in enumerate(shot_keys)}
        requirements = {item.requirement_key: item for item in self.requirements}
        if len(requirements) != len(self.requirements):
            raise ValueError("StoryPlan requirement keys must be unique")
        occurrences: dict[str, list[str]] = {
            requirement_key: [] for requirement_key in requirements
        }
        for shot in self.ordered_shots:
            if len(shot.requirement_refs) != len(set(shot.requirement_refs)):
                raise ValueError("StoryShot requirement refs must be unique")
            for requirement_key in shot.requirement_refs:
                if requirement_key not in requirements:
                    raise ValueError("StoryShot references an unknown requirement")
                occurrences[requirement_key].append(shot.shot_key)
        for requirement_key, requirement in requirements.items():
            owners = occurrences[requirement_key]
            if requirement.category == RequirementCategory.SHOT_LOCAL:
                if owners != [requirement.owner_shot_key]:
                    raise ValueError("shot-local requirement must appear only on its owner shot")
            elif set(owners) != set(shot_keys):
                raise ValueError("safety requirement must be projected to every shot")

        invariant_keys = [item.invariant_key for item in self.plan_invariants]
        if len(invariant_keys) != len(set(invariant_keys)):
            raise ValueError("StoryPlan invariant keys must be unique")
        for invariant in self.plan_invariants:
            before = shot_positions.get(invariant.before_shot_key)
            after = shot_positions.get(invariant.after_shot_key)
            if before is None or after is None:
                raise ValueError("plan invariant references an unknown shot")
            if before >= after:
                raise ValueError("shot-order invariant contradicts canonical shot order")
        return self

    def world_rules(self) -> WorldRules:
        from story_engine.domain.state import EntityRule, SceneRule

        return WorldRules(
            entities=tuple(
                EntityRule(
                    entity_key=entity.entity_key,
                    kind=entity.kind,
                    attribute_definitions=entity.attribute_definitions,
                )
                for entity in self.entity_catalog
            ),
            scenes=tuple(
                SceneRule(
                    scene_key=scene.scene_key,
                    zone_keys=tuple(zone.zone_key for zone in scene.zones),
                )
                for scene in self.scene_catalog
            ),
        )

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(exclude={"plan_hash"}))

    def assert_hash(self) -> None:
        actual = self.content_hash()
        if actual != self.plan_hash:
            raise ValueError(f"StoryPlan hash mismatch: expected {self.plan_hash}, got {actual}")
