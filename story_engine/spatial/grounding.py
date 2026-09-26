"""Station-level anchor evidence and deterministic coverage validation."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from story_engine.domain.common import FrozenModel, WireModel
from story_engine.errors import SpatialError
from story_engine.ids import canonical_hash, stable_key
from story_engine.spatial.probes import ProbeRole, ProbeSet
from story_engine.version import GROUNDING_VALIDATOR_VERSION


class GroundingTargetKind(StrEnum):
    ANCHOR = "anchor"
    REGION = "region"


class Visibility(StrEnum):
    VISIBLE = "visible"
    PARTIALLY_VISIBLE = "partially_visible"
    NOT_VISIBLE = "not_visible"
    UNKNOWN = "unknown"


LOCALIZED_VISIBILITIES = frozenset({Visibility.VISIBLE, Visibility.PARTIALLY_VISIBLE})


class NormalizedBox(WireModel):
    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_order(self) -> NormalizedBox:
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("normalized box must have positive area")
        return self


class GroundingTarget(FrozenModel):
    alias: str
    target_key: str
    kind: GroundingTargetKind


class GroundingView(FrozenModel):
    scene_key: str
    scene_alias: str
    station_key: str
    targets: tuple[GroundingTarget, ...]
    probe_set: ProbeSet

    @model_validator(mode="after")
    def validate_targets(self) -> GroundingView:
        aliases = [target.alias for target in self.targets]
        keys = [target.target_key for target in self.targets]
        if len(aliases) != len(set(aliases)):
            raise ValueError("grounding target aliases must be unique")
        if len(keys) != len(set(keys)):
            raise ValueError("grounding target keys must be unique")
        if self.probe_set.scene_key != self.scene_key:
            raise ValueError("grounding probes belong to another scene")
        return self


class GroundingWireRowBase(WireModel):
    target_alias: str
    observation_index: int = Field(ge=1)
    probe_role: ProbeRole
    confidence: float = Field(ge=0.0, le=1.0)
    is_unique: bool | None


class LocalizedGroundingWireRow(GroundingWireRowBase):
    normalized_box: NormalizedBox
    visibility: Literal[Visibility.VISIBLE, Visibility.PARTIALLY_VISIBLE]


class UnlocalizedGroundingWireRow(GroundingWireRowBase):
    normalized_box: None
    visibility: Literal[Visibility.NOT_VISIBLE, Visibility.UNKNOWN]


type GroundingWireRow = LocalizedGroundingWireRow | UnlocalizedGroundingWireRow


class GroundingWireResponse(WireModel):
    items: list[GroundingWireRow]


class GroundingValidationCode(StrEnum):
    UNKNOWN_TARGET_ALIAS = "unknown_target_alias"
    DUPLICATE_OBSERVATION = "duplicate_observation"
    VISIBILITY_BOX_MISMATCH = "visibility_box_mismatch"
    ANCHOR_UNIQUENESS_REQUIRED = "anchor_uniqueness_required"
    REGION_UNIQUENESS_FORBIDDEN = "region_uniqueness_forbidden"
    MISSING_TARGETS = "missing_targets"
    ANCHOR_CARDINALITY = "anchor_cardinality"


class GroundingValidationError(SpatialError):
    """A deterministic local Grounding validation failure."""

    def __init__(
        self,
        code: GroundingValidationCode,
        message: str,
        *,
        target_aliases: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.target_aliases = target_aliases


class GroundingObservation(FrozenModel):
    observation_key: str
    target_key: str
    target_kind: GroundingTargetKind
    observation_index: int
    probe_role: ProbeRole
    probe_hash: str
    orientation_matrix: tuple[float, ...]
    probe_hfov_degrees: float
    normalized_box: NormalizedBox | None
    visibility: Visibility
    confidence: float
    is_unique: bool | None

    @property
    def is_visible_unique_anchor(self) -> bool:
        return (
            self.target_kind == GroundingTargetKind.ANCHOR
            and self.visibility in LOCALIZED_VISIBILITIES
            and self.normalized_box is not None
            and self.is_unique is True
        )


class SceneAnchorMap(FrozenModel):
    map_key: str
    scene_key: str
    station_key: str
    reference_hash: str
    probe_set_hash: str
    anchor_schema_version: str
    grounding_implementation_version: str
    observations: tuple[GroundingObservation, ...]
    map_hash: str

    def observations_for(self, target_key: str) -> tuple[GroundingObservation, ...]:
        return tuple(item for item in self.observations if item.target_key == target_key)

    def visible_unique_anchors(self) -> tuple[GroundingObservation, ...]:
        return tuple(item for item in self.observations if item.is_visible_unique_anchor)

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(exclude={"map_hash"}))

    def assert_hash(self) -> None:
        if self.content_hash() != self.map_hash:
            raise ValueError("SceneAnchorMap hash mismatch")


class GroundingValidator:
    """Validate wire coverage; semantic invisibility is evidence, not a retry error."""

    ANCHOR_SCHEMA_VERSION = "scene-anchor-map-v2-localized-visibility"

    def validate(self, view: GroundingView, response: GroundingWireResponse) -> SceneAnchorMap:
        expected = {target.alias: target for target in view.targets}
        rows_by_alias: dict[str, list[GroundingWireRow]] = {}
        coordinates: set[tuple[str, int]] = set()
        for row in response.items:
            target = expected.get(row.target_alias)
            if target is None:
                raise GroundingValidationError(
                    GroundingValidationCode.UNKNOWN_TARGET_ALIAS,
                    f"unknown grounding target alias {row.target_alias}",
                    target_aliases=(row.target_alias,),
                )
            coordinate = (row.target_alias, row.observation_index)
            if coordinate in coordinates:
                raise GroundingValidationError(
                    GroundingValidationCode.DUPLICATE_OBSERVATION,
                    f"duplicate grounding observation {row.target_alias}#{row.observation_index}",
                    target_aliases=(row.target_alias,),
                )
            coordinates.add(coordinate)
            localized = row.visibility in LOCALIZED_VISIBILITIES
            if localized != (row.normalized_box is not None):
                raise GroundingValidationError(
                    GroundingValidationCode.VISIBILITY_BOX_MISMATCH,
                    (
                        f"localized target {row.target_alias} requires a box"
                        if localized
                        else f"unlocalized target {row.target_alias} cannot have a box"
                    ),
                    target_aliases=(row.target_alias,),
                )
            if target.kind == GroundingTargetKind.ANCHOR and row.is_unique is None:
                raise GroundingValidationError(
                    GroundingValidationCode.ANCHOR_UNIQUENESS_REQUIRED,
                    f"anchor target {row.target_alias} requires is_unique",
                    target_aliases=(row.target_alias,),
                )
            if target.kind == GroundingTargetKind.REGION and row.is_unique is not None:
                raise GroundingValidationError(
                    GroundingValidationCode.REGION_UNIQUENESS_FORBIDDEN,
                    f"region target {row.target_alias} must use null is_unique",
                    target_aliases=(row.target_alias,),
                )
            rows_by_alias.setdefault(row.target_alias, []).append(row)
        missing = set(expected) - set(rows_by_alias)
        if missing:
            aliases = tuple(sorted(missing))
            raise GroundingValidationError(
                GroundingValidationCode.MISSING_TARGETS,
                f"grounding response missing targets: {list(aliases)}",
                target_aliases=aliases,
            )
        for alias, target in expected.items():
            if target.kind == GroundingTargetKind.ANCHOR and len(rows_by_alias[alias]) != 1:
                raise GroundingValidationError(
                    GroundingValidationCode.ANCHOR_CARDINALITY,
                    f"anchor target {alias} requires exactly one observation",
                    target_aliases=(alias,),
                )

        compiled: list[GroundingObservation] = []
        for alias, target in expected.items():
            for row in sorted(rows_by_alias[alias], key=lambda item: item.observation_index):
                probe = view.probe_set.by_role(row.probe_role)
                compiled.append(
                    GroundingObservation(
                        observation_key=stable_key(
                            "grounding_observation",
                            view.station_key,
                            f"{target.target_key}:{row.observation_index}",
                        ),
                        target_key=target.target_key,
                        target_kind=target.kind,
                        observation_index=row.observation_index,
                        probe_role=row.probe_role,
                        probe_hash=probe.artifact_ref.sha256,
                        orientation_matrix=probe.orientation_matrix,
                        probe_hfov_degrees=probe.hfov_degrees,
                        normalized_box=row.normalized_box,
                        visibility=row.visibility,
                        confidence=row.confidence,
                        is_unique=row.is_unique,
                    )
                )
        observations = tuple(
            sorted(compiled, key=lambda item: (item.target_key, item.observation_index))
        )
        map_key = stable_key(
            "scene_anchor_map",
            view.scene_key,
            f"{view.probe_set.source_panorama_hash}:{view.station_key}",
        )
        payload = {
            "map_key": map_key,
            "scene_key": view.scene_key,
            "station_key": view.station_key,
            "reference_hash": view.probe_set.source_panorama_hash,
            "probe_set_hash": view.probe_set.probe_set_hash,
            "anchor_schema_version": self.ANCHOR_SCHEMA_VERSION,
            "grounding_implementation_version": GROUNDING_VALIDATOR_VERSION,
            "observations": observations,
        }
        result = SceneAnchorMap(
            map_key=map_key,
            scene_key=view.scene_key,
            station_key=view.station_key,
            reference_hash=view.probe_set.source_panorama_hash,
            probe_set_hash=view.probe_set.probe_set_hash,
            anchor_schema_version=self.ANCHOR_SCHEMA_VERSION,
            grounding_implementation_version=GROUNDING_VALIDATOR_VERSION,
            observations=observations,
            map_hash=canonical_hash(payload),
        )
        result.assert_hash()
        return result
