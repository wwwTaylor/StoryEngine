"""Camera materialization over already-resolved station-level evidence."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from story_engine.domain.common import FrozenModel
from story_engine.domain.evaluation import CandidateRecord, SelectionDecision, TechnicalStatus
from story_engine.domain.story import FramingScale
from story_engine.errors import SpatialError
from story_engine.spatial.fov import CommonFovSolution, solve_common_fov
from story_engine.spatial.grounding import GroundingObservation
from story_engine.spatial.panorama import PanoramaSource
from story_engine.spatial.probes import PanoramaProjector
from story_engine.storage import ArtifactRef


class CameraConflictKind(StrEnum):
    ANCHOR_NOT_VISIBLE = "anchor_not_visible"
    ANCHOR_NOT_UNIQUE = "anchor_not_unique"
    NO_COMMON_FOV = "no_common_fov"
    STATION_INCOMPATIBLE = "station_incompatible"
    REVEAL_NOT_EXECUTABLE = "reveal_not_executable"
    INVALID_EVIDENCE = "invalid_evidence"


class CameraConflict(FrozenModel):
    kind: CameraConflictKind
    shot_key: str
    target_keys: tuple[str, ...]
    detail: str
    fov_solution: CommonFovSolution | None = None


class SourceStationRecipe(FrozenModel):
    kind: Literal["source_station"] = "source_station"
    source_panorama_hash: str
    station_key: str
    aim_anchor: str | None
    yaw_degrees: float
    pitch_degrees: float
    hfov_degrees: float = Field(gt=0, lt=180)
    vfov_degrees: float = Field(gt=0, lt=180)
    scene_view: ArtifactRef


class NovelStationRecipe(FrozenModel):
    kind: Literal["novel_station"] = "novel_station"
    source_panorama_hash: str
    station_panorama: ArtifactRef
    station_key: str
    station_description: str
    aim_anchor: str | None
    selected_candidate: str
    yaw_degrees: float
    pitch_degrees: float
    hfov_degrees: float = Field(gt=0, lt=180)
    vfov_degrees: float = Field(gt=0, lt=180)
    scene_view: ArtifactRef


type CameraRecipe = SourceStationRecipe | NovelStationRecipe


class CameraSolveResult(FrozenModel):
    recipe: CameraRecipe | None = None
    conflict: CameraConflict | None = None
    fov_solution: CommonFovSolution | None = None


class CameraConstraints(FrozenModel):
    min_hfov_degrees: float = Field(default=25.0, gt=0, lt=180)
    max_hfov_degrees: float = Field(default=110.0, gt=0, lt=180)
    safe_margin_degrees: float = Field(default=5.0, ge=0, le=30)
    tight_hfov_degrees: float = Field(default=35.0, gt=0, lt=180)
    medium_hfov_degrees: float = Field(default=60.0, gt=0, lt=180)
    wide_hfov_degrees: float = Field(default=90.0, gt=0, lt=180)
    output_width: int = Field(default=1280, gt=0)
    output_height: int = Field(default=720, gt=0)


class CameraSolver:
    def __init__(self, projector: PanoramaProjector, constraints: CameraConstraints) -> None:
        if constraints.min_hfov_degrees >= constraints.max_hfov_degrees:
            raise ValueError("minimum camera FOV must be lower than maximum")
        if any(
            value > constraints.max_hfov_degrees
            for value in (
                constraints.tight_hfov_degrees,
                constraints.medium_hfov_degrees,
                constraints.wide_hfov_degrees,
            )
        ):
            raise ValueError("framing FOV must not exceed the maximum camera FOV")
        self.projector = projector
        self.constraints = constraints

    def solve_source_station(
        self,
        panorama: PanoramaSource,
        observations: tuple[GroundingObservation, ...],
        *,
        shot_key: str,
        station_key: str,
        aim_anchor: str | None,
        framing_scale: FramingScale = FramingScale.MEDIUM,
    ) -> CameraSolveResult:
        if not observations:
            return CameraSolveResult(
                conflict=CameraConflict(
                    kind=CameraConflictKind.INVALID_EVIDENCE,
                    shot_key=shot_key,
                    target_keys=(),
                    detail="camera solve requires at least one resolved visible anchor",
                )
            )
        framing_hfov = {
            FramingScale.TIGHT: self.constraints.tight_hfov_degrees,
            FramingScale.MEDIUM: self.constraints.medium_hfov_degrees,
            FramingScale.WIDE: self.constraints.wide_hfov_degrees,
        }[framing_scale]
        try:
            solution = solve_common_fov(
                observations,
                output_width=self.constraints.output_width,
                output_height=self.constraints.output_height,
                framing_hfov_degrees=framing_hfov,
                min_hfov_degrees=self.constraints.min_hfov_degrees,
                max_hfov_degrees=self.constraints.max_hfov_degrees,
                safe_margin_degrees=self.constraints.safe_margin_degrees,
            )
        except SpatialError as exc:
            return CameraSolveResult(
                conflict=CameraConflict(
                    kind=CameraConflictKind.NO_COMMON_FOV,
                    shot_key=shot_key,
                    target_keys=tuple(dict.fromkeys(item.target_key for item in observations)),
                    detail=str(exc),
                )
            )
        if not solution.feasible:
            return CameraSolveResult(
                conflict=CameraConflict(
                    kind=CameraConflictKind.NO_COMMON_FOV,
                    shot_key=shot_key,
                    target_keys=tuple(dict.fromkeys(item.target_key for item in observations)),
                    detail=(
                        f"required HFOV {solution.minimum_hfov_degrees:.2f} and VFOV "
                        f"{solution.minimum_vfov_degrees:.2f} exceed the configured "
                        f"HFOV limit {self.constraints.max_hfov_degrees:.2f}"
                    ),
                    fov_solution=solution,
                ),
                fov_solution=solution,
            )
        scene_view = self.projector.project_view(
            panorama,
            yaw_degrees=solution.yaw_degrees,
            pitch_degrees=solution.pitch_degrees,
            hfov_degrees=solution.hfov_degrees,
            width=self.constraints.output_width,
            height=self.constraints.output_height,
        )
        return CameraSolveResult(
            recipe=SourceStationRecipe(
                source_panorama_hash=panorama.artifact_ref.sha256,
                station_key=station_key,
                aim_anchor=aim_anchor,
                yaw_degrees=solution.yaw_degrees,
                pitch_degrees=solution.pitch_degrees,
                hfov_degrees=solution.hfov_degrees,
                vfov_degrees=solution.vfov_degrees,
                scene_view=scene_view,
            ),
            fov_solution=solution,
        )

    @staticmethod
    def materialize_novel_station(
        *,
        source_panorama_hash: str,
        station_panorama: ArtifactRef,
        station_key: str,
        station_description: str,
        aim_anchor: str | None,
        yaw_degrees: float,
        pitch_degrees: float,
        hfov_degrees: float,
        vfov_degrees: float,
        scene_view: ArtifactRef,
        decision: SelectionDecision,
        candidates: tuple[CandidateRecord, ...],
    ) -> NovelStationRecipe:
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate.candidate_key == decision.selected_candidate
            ),
            None,
        )
        if (
            selected is None
            or selected.technical_status != TechnicalStatus.VALID
            or selected.artifact_ref is None
            or selected.artifact_ref != station_panorama
        ):
            raise SpatialError("novel station has no selected technically valid artifact")
        return NovelStationRecipe(
            source_panorama_hash=source_panorama_hash,
            station_panorama=station_panorama,
            station_key=station_key,
            station_description=station_description,
            aim_anchor=aim_anchor,
            selected_candidate=selected.candidate_key,
            yaw_degrees=yaw_degrees,
            pitch_degrees=pitch_degrees,
            hfov_degrees=hfov_degrees,
            vfov_degrees=vfov_degrees,
            scene_view=scene_view,
        )
