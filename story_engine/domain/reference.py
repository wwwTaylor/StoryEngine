"""Reference recipes and immutable selected reference library."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from story_engine.domain.common import FrozenModel
from story_engine.domain.state import WorldState
from story_engine.domain.story import ReferenceNeedKind
from story_engine.ids import canonical_hash, canonical_json
from story_engine.spatial.grounding import SceneAnchorMap
from story_engine.storage import ArtifactRef


class ReferenceRecipeBase(FrozenModel):
    kind: str
    need_key: str
    subject_key: str


class CharacterReferenceRecipe(ReferenceRecipeBase):
    kind: Literal["character"] = "character"
    appearance: str
    clothing: str = ""
    neutral_pose: str = "neutral full-body pose"
    style: str
    forbidden_co_subjects: tuple[str, ...] = ()


class GuidedCharacterReferenceRecipe(CharacterReferenceRecipe):
    kind: Literal["guided_character"] = "guided_character"
    provided_asset_id: str


class PropReferenceRecipe(ReferenceRecipeBase):
    kind: Literal["prop"] = "prop"
    appearance: str
    visible_state: str | None = None
    scale_cues: str = "clear isolated product-scale view"
    style: str
    forbidden_co_subjects: tuple[str, ...] = ()


class SourceStationSpec(FrozenModel):
    station_key: str
    zone_key: str
    description: str = Field(min_length=1, max_length=1_000)
    height_relation: str = Field(default="eye_level", min_length=1, max_length=100)
    supported_viewpoint_intents: tuple[str, ...] = ()


class ScenePanoramaRecipe(ReferenceRecipeBase):
    kind: Literal["scene_panorama"] = "scene_panorama"
    scene_visual_identity: str
    zones_and_landmarks: tuple[str, ...]
    semantic_layout: tuple[str, ...]
    lighting: str
    style: str
    allowed_population: tuple[str, ...]
    source_station: SourceStationSpec


class GuidedScenePanoramaRecipe(ScenePanoramaRecipe):
    kind: Literal["guided_scene_panorama"] = "guided_scene_panorama"
    provided_asset_id: str


class ProvidedReferenceRecipe(ReferenceRecipeBase):
    kind: Literal["provided"] = "provided"
    provided_asset_id: str
    reference_kind: ReferenceNeedKind
    require_panorama: bool = False
    source_station: SourceStationSpec | None = None

    @model_validator(mode="after")
    def validate_source_station(self) -> ProvidedReferenceRecipe:
        if self.require_panorama != (self.source_station is not None):
            raise ValueError("provided panoramas require an explicit source station")
        return self


type ReferenceRecipe = Annotated[
    CharacterReferenceRecipe
    | GuidedCharacterReferenceRecipe
    | PropReferenceRecipe
    | ScenePanoramaRecipe
    | GuidedScenePanoramaRecipe
    | ProvidedReferenceRecipe,
    Field(discriminator="kind"),
]


type CharacterViewRole = Literal["front", "side", "back"]
CHARACTER_VIEW_ROLES: tuple[CharacterViewRole, ...] = ("front", "side", "back")


class CharacterReferenceView(FrozenModel):
    role: CharacterViewRole
    artifact_ref: ArtifactRef


class SelectedReference(FrozenModel):
    need_key: str
    subject_key: str
    kind: ReferenceNeedKind
    candidate_key: str
    artifact_ref: ArtifactRef
    character_views: tuple[CharacterReferenceView, ...] = ()
    evaluation_ref: str
    degraded: bool
    visible_state: str | None = None

    @model_validator(mode="after")
    def validate_character_views(self) -> SelectedReference:
        if self.kind == ReferenceNeedKind.CHARACTER:
            roles = tuple(item.role for item in self.character_views)
            if roles != ("front", "side", "back"):
                raise ValueError("character references require ordered front, side, and back views")
            if self.character_views[0].artifact_ref != self.artifact_ref:
                raise ValueError("character primary artifact must be the front view")
        elif self.character_views:
            raise ValueError("only character references may carry character views")
        return self


class ScenePanoramaBinding(FrozenModel):
    scene_key: str
    selected_reference: str
    artifact_ref: ArtifactRef
    source_station: SourceStationSpec


class AnchorMapEvidence(FrozenModel):
    kind: Literal["anchor_map"] = "anchor_map"
    anchor_map: SceneAnchorMap
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def from_anchor_map(cls, anchor_map: SceneAnchorMap) -> AnchorMapEvidence:
        payload = {"kind": "anchor_map", "anchor_map": anchor_map}
        return cls(anchor_map=anchor_map, evidence_hash=canonical_hash(payload))

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(exclude={"evidence_hash"}))

    def assert_hash(self) -> None:
        if self.content_hash() != self.evidence_hash:
            raise ValueError("AnchorMapEvidence hash mismatch")


class GroundingUnavailableEvidence(FrozenModel):
    kind: Literal["grounding_unavailable"] = "grounding_unavailable"
    scene_key: str
    station_key: str
    reference_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    failed_target_aliases: tuple[str, ...]
    failure_code: Literal["GROUNDING_VALIDATION_EXHAUSTED"] = "GROUNDING_VALIDATION_EXHAUSTED"
    logical_attempt_refs: tuple[str, ...]
    provider_call_refs: tuple[str, ...]
    response_refs: tuple[str, ...] = ()
    validation_refs: tuple[str, ...] = ()
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        scene_key: str,
        station_key: str,
        reference_hash: str,
        failed_target_aliases: tuple[str, ...],
        logical_attempt_refs: tuple[str, ...],
        provider_call_refs: tuple[str, ...],
        response_refs: tuple[str, ...] = (),
        validation_refs: tuple[str, ...] = (),
    ) -> GroundingUnavailableEvidence:
        payload = {
            "kind": "grounding_unavailable",
            "scene_key": scene_key,
            "station_key": station_key,
            "reference_hash": reference_hash,
            "failed_target_aliases": failed_target_aliases,
            "failure_code": "GROUNDING_VALIDATION_EXHAUSTED",
            "logical_attempt_refs": logical_attempt_refs,
            "provider_call_refs": provider_call_refs,
            "response_refs": response_refs,
            "validation_refs": validation_refs,
        }
        return cls(
            scene_key=scene_key,
            station_key=station_key,
            reference_hash=reference_hash,
            failed_target_aliases=failed_target_aliases,
            logical_attempt_refs=logical_attempt_refs,
            provider_call_refs=provider_call_refs,
            response_refs=response_refs,
            validation_refs=validation_refs,
            evidence_hash=canonical_hash(payload),
        )

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(exclude={"evidence_hash"}))

    def assert_hash(self) -> None:
        if self.content_hash() != self.evidence_hash:
            raise ValueError("GroundingUnavailableEvidence hash mismatch")


type StationSpatialEvidence = Annotated[
    AnchorMapEvidence | GroundingUnavailableEvidence,
    Field(discriminator="kind"),
]


class StationSpatialReference(FrozenModel):
    kind: Literal["source_station", "novel_station"]
    station_key: str
    description: str
    station_panorama: ArtifactRef
    scene_view: ArtifactRef
    evidence: StationSpatialEvidence
    selected_candidate: str | None = None

    @model_validator(mode="after")
    def validate_station_reference(self) -> StationSpatialReference:
        if self.kind == "novel_station" and self.selected_candidate is None:
            raise ValueError("novel station spatial reference requires a selected candidate")
        if self.kind == "source_station" and self.selected_candidate is not None:
            raise ValueError("source station spatial reference cannot select a novel candidate")
        if isinstance(self.evidence, AnchorMapEvidence):
            self.evidence.assert_hash()
            if self.evidence.anchor_map.station_key != self.station_key:
                raise ValueError("station spatial reference and Anchor Map keys differ")
            if self.evidence.anchor_map.reference_hash != self.station_panorama.sha256:
                raise ValueError("station panorama and Anchor Map reference hashes differ")
        else:
            self.evidence.assert_hash()
            if self.evidence.station_key != self.station_key:
                raise ValueError("station spatial reference and unavailable evidence keys differ")
            if self.evidence.reference_hash != self.station_panorama.sha256:
                raise ValueError("station panorama and unavailable evidence hashes differ")
        return self


class SceneSpatialReference(FrozenModel):
    scene_key: str
    canonical_panorama: ArtifactRef
    source_station: StationSpatialReference
    novel_stations: tuple[StationSpatialReference, ...] = ()

    @model_validator(mode="after")
    def validate_scene_spatial_reference(self) -> SceneSpatialReference:
        if self.source_station.kind != "source_station":
            raise ValueError("scene spatial reference requires one source station")
        source_scene_key = (
            self.source_station.evidence.anchor_map.scene_key
            if isinstance(self.source_station.evidence, AnchorMapEvidence)
            else self.source_station.evidence.scene_key
        )
        if source_scene_key != self.scene_key:
            raise ValueError("source station evidence belongs to another scene")
        for station in self.novel_stations:
            station_scene_key = (
                station.evidence.anchor_map.scene_key
                if isinstance(station.evidence, AnchorMapEvidence)
                else station.evidence.scene_key
            )
            if station_scene_key != self.scene_key:
                raise ValueError("novel station evidence belongs to another scene")
        station_keys = [self.source_station.station_key] + [
            item.station_key for item in self.novel_stations
        ]
        if len(station_keys) != len(set(station_keys)):
            raise ValueError("scene station keys must be unique")
        return self


class ReferenceLibrary(FrozenModel):
    story_plan_hash: str
    selected_assets: tuple[SelectedReference, ...]
    panorama_by_scene: tuple[ScenePanoramaBinding, ...]
    spatial_by_scene: tuple[SceneSpatialReference, ...] = ()
    evaluation_refs: tuple[str, ...]
    library_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_uniqueness(self) -> ReferenceLibrary:
        need_keys = [item.need_key for item in self.selected_assets]
        if len(need_keys) != len(set(need_keys)):
            raise ValueError("reference needs may only be selected once")
        scene_keys = [item.scene_key for item in self.panorama_by_scene]
        if len(scene_keys) != len(set(scene_keys)):
            raise ValueError("each scene may have only one canonical panorama")
        spatial_scene_keys = [item.scene_key for item in self.spatial_by_scene]
        if len(spatial_scene_keys) != len(set(spatial_scene_keys)):
            raise ValueError("each scene may have only one spatial reference")
        if spatial_scene_keys and set(spatial_scene_keys) != set(scene_keys):
            raise ValueError("spatial references must cover every canonical panorama scene")
        bindings = {item.scene_key: item for item in self.panorama_by_scene}
        for spatial in self.spatial_by_scene:
            binding = bindings[spatial.scene_key]
            if spatial.canonical_panorama != binding.artifact_ref:
                raise ValueError("spatial reference canonical panorama differs from its binding")
            if (
                spatial.source_station.station_key != binding.source_station.station_key
                or spatial.source_station.station_panorama != binding.artifact_ref
            ):
                raise ValueError("spatial reference source station differs from its binding")
        return self

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(exclude={"library_hash"}))

    def assert_hash(self) -> None:
        if self.content_hash() != self.library_hash:
            raise ValueError("ReferenceLibrary hash mismatch")

    def reference_for_subject(
        self,
        subject_key: str,
        *,
        visible_state: str | None = None,
    ) -> SelectedReference:
        matches = [item for item in self.selected_assets if item.subject_key == subject_key]
        if visible_state is not None:
            matches = [item for item in matches if item.visible_state == visible_state]
        if len(matches) != 1:
            raise KeyError(subject_key)
        return matches[0]

    def panorama_for_scene(self, scene_key: str) -> ScenePanoramaBinding:
        for binding in self.panorama_by_scene:
            if binding.scene_key == scene_key:
                return binding
        raise KeyError(scene_key)


def visible_state_for(
    entity_key: str,
    state: WorldState,
    *,
    attribute_keys: tuple[str, ...],
) -> str | None:
    facts = tuple(
        (item.attribute_key, item.value)
        for item in state.attributes_by_entity
        if item.entity_key == entity_key and item.attribute_key in attribute_keys
    )
    if not facts:
        return None
    return "; ".join(f"{key}={canonical_json(value)}" for key, value in facts)
