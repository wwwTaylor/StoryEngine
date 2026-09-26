"""Resolved shot-level spatial decisions and their audit trail."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from story_engine.domain.common import FrozenModel
from story_engine.domain.story import SpatialPresentation, SpatialStrength
from story_engine.ids import canonical_hash
from story_engine.spatial.camera import CameraConflictKind, CameraRecipe
from story_engine.spatial.fov import CommonFovSolution


class SpatialResolutionStatus(StrEnum):
    SUCCESS = "SUCCESS"
    REPAIRED = "REPAIRED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class ResolvedStationKind(StrEnum):
    SOURCE = "source_station"
    NOVEL = "novel_station"


class ResolvedStation(FrozenModel):
    kind: ResolvedStationKind
    station_key: str
    description: str


class ContentResolution(FrozenModel):
    subject_key: str
    resolved_anchor: str | None = None
    presentation: SpatialPresentation
    strength: SpatialStrength
    retained: bool = True
    detail: str = ""


class SpatialAuditEntry(FrozenModel):
    sequence: int = Field(ge=1)
    code: str
    detail: str
    station_key: str | None = None
    target_keys: tuple[str, ...] = ()
    conflict_kind: CameraConflictKind | None = None
    fov_solution: CommonFovSolution | None = None


class ResolvedSpatialPlan(FrozenModel):
    shot_key: str
    scene_key: str
    original_story_targets: tuple[str, ...]
    original_spatial_content: tuple[ContentResolution, ...]
    original_viewpoint_intent: str
    station: ResolvedStation
    aim_anchor: str | None
    must_include_anchors: tuple[str, ...]
    preferred_include_anchors: tuple[str, ...]
    reveal_anchors: tuple[str, ...]
    content_resolution: tuple[ContentResolution, ...]
    camera_recipe: CameraRecipe
    fov_solution: CommonFovSolution | None = None
    result_status: SpatialResolutionStatus
    audit: tuple[SpatialAuditEntry, ...]
    resolution_hash: str

    @model_validator(mode="after")
    def validate_resolution(self) -> ResolvedSpatialPlan:
        if self.station.kind.value != self.camera_recipe.kind:
            raise ValueError("resolved station kind and CameraRecipe kind differ")
        if self.station.station_key != self.camera_recipe.station_key:
            raise ValueError("resolved station and CameraRecipe station keys differ")
        if self.aim_anchor != self.camera_recipe.aim_anchor:
            raise ValueError("resolved aim anchor and CameraRecipe aim anchor differ")
        for label, values in (
            ("must include anchors", self.must_include_anchors),
            ("preferred include anchors", self.preferred_include_anchors),
            ("reveal anchors", self.reveal_anchors),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        if self.result_status != SpatialResolutionStatus.DEGRADED and self.aim_anchor is None:
            raise ValueError("non-degraded spatial plans require one aim anchor")
        if self.result_status == SpatialResolutionStatus.DEGRADED and (
            self.must_include_anchors or self.preferred_include_anchors or self.reveal_anchors
        ):
            raise ValueError("degraded safe shots cannot retain include or reveal constraints")
        if self.result_status == SpatialResolutionStatus.DEGRADED and (
            self.camera_recipe.hfov_degrees > 90
            or self.fov_solution is not None
            or any(item.retained for item in self.content_resolution)
        ):
            raise ValueError("degraded safe shots must use a bounded unconstrained camera")
        if self.result_status != SpatialResolutionStatus.DEGRADED and self.fov_solution is None:
            raise ValueError("non-degraded spatial plans require a verified FOV solution")
        if self.fov_solution is not None and (
            self.camera_recipe.yaw_degrees,
            self.camera_recipe.pitch_degrees,
            self.camera_recipe.hfov_degrees,
            self.camera_recipe.vfov_degrees,
        ) != (
            self.fov_solution.yaw_degrees,
            self.fov_solution.pitch_degrees,
            self.fov_solution.hfov_degrees,
            self.fov_solution.vfov_degrees,
        ):
            raise ValueError("CameraRecipe does not match the verified FOV solution")
        if [item.sequence for item in self.audit] != list(range(1, len(self.audit) + 1)):
            raise ValueError("spatial audit entries must have contiguous sequence numbers")
        return self

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(exclude={"resolution_hash"}))

    def assert_hash(self) -> None:
        if self.content_hash() != self.resolution_hash:
            raise ValueError("ResolvedSpatialPlan hash mismatch")
