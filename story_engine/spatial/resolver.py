"""Deterministic resolution from semantic shot intent to executable camera evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from story_engine.config import SpatialFailurePolicy
from story_engine.domain.reference import SourceStationSpec
from story_engine.domain.spatial import (
    ContentResolution,
    ResolvedSpatialPlan,
    ResolvedStation,
    ResolvedStationKind,
    SpatialAuditEntry,
    SpatialResolutionStatus,
)
from story_engine.domain.story import (
    FramingScale,
    SceneSpec,
    SpatialContent,
    SpatialPresentation,
    SpatialStrength,
    StoryShot,
)
from story_engine.errors import SpatialError
from story_engine.ids import canonical_hash
from story_engine.spatial.camera import (
    CameraConflict,
    CameraConflictKind,
    CameraSolver,
    NovelStationRecipe,
    SourceStationRecipe,
)
from story_engine.spatial.fov import CommonFovSolution, solve_common_fov
from story_engine.spatial.grounding import (
    LOCALIZED_VISIBILITIES,
    GroundingObservation,
    GroundingTargetKind,
    SceneAnchorMap,
    Visibility,
)
from story_engine.spatial.panorama import PanoramaSource


@dataclass(frozen=True, slots=True)
class SpatialResolutionResult:
    plan: ResolvedSpatialPlan | None = None
    conflict: CameraConflict | None = None
    novel_station_needed: bool = False
    audit: tuple[SpatialAuditEntry, ...] = ()


class SpatialResolver:
    def __init__(
        self,
        camera_solver: CameraSolver,
        *,
        failure_policy: SpatialFailurePolicy,
        fallback_hfov_min: float,
        fallback_hfov_max: float,
        supports_camera_motion: bool,
        repair_attempts: int = 2,
    ) -> None:
        if repair_attempts < 0:
            raise ValueError("spatial repair attempts cannot be negative")
        self.camera_solver = camera_solver
        self.failure_policy = failure_policy
        self.fallback_hfov_min = fallback_hfov_min
        self.fallback_hfov_max = fallback_hfov_max
        self.supports_camera_motion = supports_camera_motion
        self.repair_attempts = repair_attempts

    def resolve_source(
        self,
        *,
        shot: StoryShot,
        scene: SceneSpec,
        panorama: PanoramaSource,
        station: SourceStationSpec,
        anchor_map: SceneAnchorMap,
        allow_novel_station: bool = True,
        translated_station: bool = False,
        prior_audit: tuple[SpatialAuditEntry, ...] = (),
    ) -> SpatialResolutionResult:
        self._validate_inputs(shot, scene, panorama, station, anchor_map)
        audit = [
            item.model_copy(update={"sequence": index})
            for index, item in enumerate(prior_audit, start=1)
        ]
        viewpoint = shot.spatial_intent.viewpoint_intent
        if (viewpoint.translation_expected and not translated_station) or (
            station.supported_viewpoint_intents
            and viewpoint.intent_key not in station.supported_viewpoint_intents
        ):
            conflict = CameraConflict(
                kind=CameraConflictKind.STATION_INCOMPATIBLE,
                shot_key=shot.shot_key,
                target_keys=(),
                detail=(
                    "viewpoint intent requires camera translation"
                    if viewpoint.translation_expected and not translated_station
                    else "source station does not declare support for the viewpoint intent"
                ),
            )
            audit.append(self._conflict_entry(audit, conflict, station.station_key))
            return self._unresolved(
                shot=shot,
                scene=scene,
                panorama=panorama,
                station=station,
                anchor_map=anchor_map,
                conflict=conflict,
                audit=audit,
                allow_novel_station=allow_novel_station,
            )

        anchors = self._visible_anchor_observations(anchor_map)
        aim_candidates = self._aim_candidates(scene, shot, anchors)[: self.repair_attempts + 1]
        if not aim_candidates:
            non_unique = self._non_unique_anchor_keys(anchor_map)
            conflict_kind = (
                CameraConflictKind.ANCHOR_NOT_UNIQUE
                if non_unique
                else CameraConflictKind.ANCHOR_NOT_VISIBLE
            )
            conflict = CameraConflict(
                kind=conflict_kind,
                shot_key=shot.shot_key,
                target_keys=(
                    non_unique
                    if non_unique
                    else tuple(item.anchor_key for item in scene.anchor_landmarks)
                ),
                detail=(
                    "visible scene anchors are not unique local instances"
                    if non_unique
                    else "scene has no unique, local, visible anchor at the source station"
                ),
            )
            audit.append(self._conflict_entry(audit, conflict, station.station_key))
            return self._unresolved(
                shot=shot,
                scene=scene,
                panorama=panorama,
                station=station,
                anchor_map=anchor_map,
                conflict=conflict,
                audit=audit,
                allow_novel_station=allow_novel_station,
            )

        resolutions: list[ContentResolution] = []
        must_observations: list[GroundingObservation] = []
        preferred_items: list[tuple[SpatialContent, GroundingObservation, str | None]] = []
        reveal_anchors: list[str] = []
        hard_conflict: CameraConflict | None = None
        for content in shot.spatial_intent.spatial_content:
            observation, resolved_anchor = self._resolve_content(
                scene,
                anchor_map,
                content,
            )
            if content.presentation == SpatialPresentation.SEQUENTIAL:
                executable = self.supports_camera_motion and resolved_anchor is not None
                if executable:
                    assert resolved_anchor is not None
                    reveal_anchors.append(resolved_anchor)
                    resolutions.append(
                        ContentResolution(
                            subject_key=content.subject_key,
                            resolved_anchor=resolved_anchor,
                            presentation=content.presentation,
                            strength=content.strength,
                        )
                    )
                    continue
                if content.strength == SpatialStrength.PREFERRED:
                    resolutions.append(
                        ContentResolution(
                            subject_key=content.subject_key,
                            resolved_anchor=resolved_anchor,
                            presentation=content.presentation,
                            strength=content.strength,
                            retained=False,
                            detail="preferred reveal is not executable",
                        )
                    )
                    audit.append(
                        self._entry(
                            audit,
                            "PREFERRED_REMOVED",
                            "Removed preferred sequential content because provider motion or "
                            "a visible representative anchor is unavailable.",
                            station_key=station.station_key,
                            target_keys=(content.subject_key,),
                        )
                    )
                    continue
                hard_conflict = CameraConflict(
                    kind=CameraConflictKind.REVEAL_NOT_EXECUTABLE,
                    shot_key=shot.shot_key,
                    target_keys=(content.subject_key,),
                    detail="must sequential content has no executable reveal",
                )
                break
            if observation is None:
                if content.strength == SpatialStrength.PREFERRED:
                    resolutions.append(
                        ContentResolution(
                            subject_key=content.subject_key,
                            resolved_anchor=resolved_anchor,
                            presentation=content.presentation,
                            strength=content.strength,
                            retained=False,
                            detail="preferred content is not visible from this station",
                        )
                    )
                    audit.append(
                        self._entry(
                            audit,
                            "PREFERRED_REMOVED",
                            "Removed preferred simultaneous content that is not visible.",
                            station_key=station.station_key,
                            target_keys=(content.subject_key,),
                        )
                    )
                    continue
                content_anchor_keys = self._content_anchor_keys(scene, content)
                non_unique = self._non_unique_anchor_keys(
                    anchor_map,
                    allowed_keys=content_anchor_keys,
                )
                hard_conflict = CameraConflict(
                    kind=(
                        CameraConflictKind.ANCHOR_NOT_UNIQUE
                        if non_unique
                        else CameraConflictKind.ANCHOR_NOT_VISIBLE
                    ),
                    shot_key=shot.shot_key,
                    target_keys=(content.subject_key,),
                    detail=(
                        "must simultaneous content has no unique representative anchor"
                        if non_unique
                        else "must simultaneous content is not visible from this station"
                    ),
                )
                break
            resolutions.append(
                ContentResolution(
                    subject_key=content.subject_key,
                    resolved_anchor=resolved_anchor,
                    presentation=content.presentation,
                    strength=content.strength,
                )
            )
            if content.strength == SpatialStrength.MUST:
                must_observations.append(observation)
            else:
                preferred_items.append((content, observation, resolved_anchor))

        if hard_conflict is not None:
            audit.append(self._conflict_entry(audit, hard_conflict, station.station_key))
            return self._unresolved(
                shot=shot,
                scene=scene,
                panorama=panorama,
                station=station,
                anchor_map=anchor_map,
                conflict=hard_conflict,
                audit=audit,
                allow_novel_station=allow_novel_station,
            )

        hard_observations = self._deduplicate_observations(must_observations)
        aim: GroundingObservation | None = None
        selected_must: tuple[GroundingObservation, ...] = ()
        failures: list[
            tuple[
                GroundingObservation,
                tuple[GroundingObservation, ...],
                CommonFovSolution | None,
                str,
            ]
        ] = []
        selected_aim_index = 0
        for aim_index, candidate_aim in enumerate(aim_candidates):
            candidate_observations = self._deduplicate_observations(
                (candidate_aim, *hard_observations)
            )
            try:
                candidate_solution = self._solve_math(
                    candidate_observations,
                    shot.spatial_intent.framing_scale,
                )
            except SpatialError as exc:
                failures.append((candidate_aim, candidate_observations, None, str(exc)))
                continue
            if candidate_solution.feasible:
                aim = candidate_aim
                selected_must = candidate_observations
                selected_aim_index = aim_index
                break
            failures.append(
                (
                    candidate_aim,
                    candidate_observations,
                    candidate_solution,
                    (
                        f"required HFOV {candidate_solution.minimum_hfov_degrees:.2f} and "
                        f"VFOV {candidate_solution.minimum_vfov_degrees:.2f} exceed the "
                        "configured limit"
                    ),
                )
            )
        if aim is None:
            failure = min(
                failures,
                key=lambda item: (
                    item[2].minimum_hfov_degrees if item[2] is not None else math.inf,
                    item[0].target_key,
                ),
            )
            conflict = CameraConflict(
                kind=CameraConflictKind.NO_COMMON_FOV,
                shot_key=shot.shot_key,
                target_keys=tuple(item.target_key for item in failure[1]),
                detail=failure[3],
                fov_solution=failure[2],
            )
            audit.append(self._conflict_entry(audit, conflict, station.station_key))
            return self._unresolved(
                shot=shot,
                scene=scene,
                panorama=panorama,
                station=station,
                anchor_map=anchor_map,
                conflict=conflict,
                audit=audit,
                allow_novel_station=allow_novel_station,
            )
        if selected_aim_index > 0:
            audit.append(
                self._entry(
                    audit,
                    "AIM_ANCHOR_CHANGED",
                    "Selected another visible anchor in the action zone after FOV preflight.",
                    station_key=station.station_key,
                    target_keys=(aim_candidates[0].target_key, aim.target_key),
                )
            )

        selected_observations = list(selected_must)
        preferred_anchors: list[str] = []
        for content, observation, resolved_anchor in preferred_items:
            candidate_observations = self._deduplicate_observations(
                [*selected_observations, observation]
            )
            preferred_solution: CommonFovSolution | None
            try:
                preferred_solution = self._solve_math(
                    candidate_observations,
                    shot.spatial_intent.framing_scale,
                )
            except SpatialError:
                preferred_solution = None
            if preferred_solution is not None and preferred_solution.feasible:
                selected_observations = list(candidate_observations)
                if resolved_anchor is not None:
                    preferred_anchors.append(resolved_anchor)
                continue
            resolutions = [
                item.model_copy(
                    update={
                        "retained": False,
                        "detail": "preferred content exceeded the common FOV",
                    }
                )
                if item.subject_key == content.subject_key
                else item
                for item in resolutions
            ]
            audit.append(
                self._entry(
                    audit,
                    "PREFERRED_REMOVED",
                    "Removed preferred simultaneous content after shared FOV preflight.",
                    station_key=station.station_key,
                    target_keys=(content.subject_key,),
                )
            )

        for observation in sorted(
            (
                item
                for item in selected_observations
                if item.visibility == Visibility.PARTIALLY_VISIBLE
            ),
            key=lambda item: item.target_key,
        ):
            audit.append(
                self._entry(
                    audit,
                    "PARTIALLY_VISIBLE_EVIDENCE",
                    "Used only the bounded, actually visible pixels of this target.",
                    station_key=station.station_key,
                    target_keys=(observation.target_key,),
                )
            )
        solved = self.camera_solver.solve_source_station(
            panorama,
            tuple(selected_observations),
            shot_key=shot.shot_key,
            station_key=station.station_key,
            aim_anchor=aim.target_key,
            framing_scale=shot.spatial_intent.framing_scale,
        )
        if solved.recipe is None or not isinstance(solved.recipe, SourceStationRecipe):
            conflict = solved.conflict or CameraConflict(
                kind=CameraConflictKind.INVALID_EVIDENCE,
                shot_key=shot.shot_key,
                target_keys=(),
                detail="formal CameraSolver rejected preflight evidence",
            )
            audit.append(self._conflict_entry(audit, conflict, station.station_key))
            return self._unresolved(
                shot=shot,
                scene=scene,
                panorama=panorama,
                station=station,
                anchor_map=anchor_map,
                conflict=conflict,
                audit=audit,
                allow_novel_station=allow_novel_station,
            )
        status = (
            SpatialResolutionStatus.REPAIRED
            if any(item.code in {"PREFERRED_REMOVED", "AIM_ANCHOR_CHANGED"} for item in audit)
            else SpatialResolutionStatus.SUCCESS
        )
        audit.append(
            self._entry(
                audit,
                "SOURCE_STATION_RESOLVED",
                "Resolved one unique aim anchor and verified all retained hard target corners.",
                station_key=station.station_key,
                target_keys=tuple(item.target_key for item in selected_observations),
                fov_solution=solved.fov_solution,
            )
        )
        plan = self._plan(
            shot=shot,
            station=ResolvedStation(
                kind=ResolvedStationKind.SOURCE,
                station_key=station.station_key,
                description=station.description,
            ),
            aim_anchor=aim.target_key,
            must_include=tuple(
                item.resolved_anchor
                for item in resolutions
                if item.retained
                and item.presentation == SpatialPresentation.SIMULTANEOUS
                and item.strength == SpatialStrength.MUST
                and item.resolved_anchor is not None
            ),
            preferred_include=tuple(preferred_anchors),
            reveal=tuple(reveal_anchors),
            content_resolution=tuple(resolutions),
            camera_recipe=solved.recipe,
            fov_solution=solved.fov_solution,
            status=status,
            audit=tuple(audit),
        )
        return SpatialResolutionResult(plan=plan, audit=tuple(audit))

    def promote_novel(
        self,
        *,
        shot: StoryShot,
        recipe: NovelStationRecipe,
        novel_resolution: ResolvedSpatialPlan,
        prior_audit: tuple[SpatialAuditEntry, ...],
    ) -> ResolvedSpatialPlan:
        source_recipe = novel_resolution.camera_recipe
        if not isinstance(source_recipe, SourceStationRecipe):
            raise SpatialError("novel station preflight did not produce a source camera recipe")
        if novel_resolution.result_status == SpatialResolutionStatus.DEGRADED:
            raise SpatialError("novel station preflight cannot be promoted from degraded evidence")
        if (
            source_recipe.station_key != recipe.station_key
            or source_recipe.source_panorama_hash != recipe.station_panorama.sha256
            or source_recipe.scene_view != recipe.scene_view
            or source_recipe.aim_anchor != recipe.aim_anchor
        ):
            raise SpatialError("novel station recipe does not match its grounded preflight")
        audit = list(prior_audit)
        for item in novel_resolution.audit:
            code = (
                "NOVEL_STATION_PREFLIGHT_RESOLVED"
                if item.code == "SOURCE_STATION_RESOLVED"
                else item.code
            )
            audit.append(
                item.model_copy(
                    update={
                        "sequence": len(audit) + 1,
                        "code": code,
                    }
                )
            )
        aim_targets = (
            (novel_resolution.aim_anchor,) if novel_resolution.aim_anchor is not None else ()
        )
        resolved_targets = tuple(
            dict.fromkeys(
                (
                    *aim_targets,
                    *novel_resolution.must_include_anchors,
                    *novel_resolution.preferred_include_anchors,
                    *novel_resolution.reveal_anchors,
                )
            )
        )
        audit.append(
            self._entry(
                audit,
                "NOVEL_STATION_RESOLVED",
                (
                    "Generated a station-translated panorama, built its independent Anchor Map, "
                    "and projected a corner-verified scene view."
                ),
                station_key=recipe.station_key,
                target_keys=resolved_targets,
                fov_solution=novel_resolution.fov_solution,
            )
        )
        return self._plan(
            shot=shot,
            station=ResolvedStation(
                kind=ResolvedStationKind.NOVEL,
                station_key=recipe.station_key,
                description=recipe.station_description,
            ),
            aim_anchor=novel_resolution.aim_anchor,
            must_include=novel_resolution.must_include_anchors,
            preferred_include=novel_resolution.preferred_include_anchors,
            reveal=novel_resolution.reveal_anchors,
            content_resolution=novel_resolution.content_resolution,
            camera_recipe=recipe,
            fov_solution=novel_resolution.fov_solution,
            status=SpatialResolutionStatus.REPAIRED,
            audit=tuple(audit),
        )

    def degrade(
        self,
        *,
        shot: StoryShot,
        scene: SceneSpec,
        panorama: PanoramaSource,
        station: SourceStationSpec,
        anchor_map: SceneAnchorMap,
        conflict: CameraConflict,
        prior_audit: tuple[SpatialAuditEntry, ...],
    ) -> ResolvedSpatialPlan:
        anchors = self._visible_anchor_observations(anchor_map)
        aim = self._select_aim(scene, shot, anchors)
        yaw, pitch = self._observation_center(aim) if aim is not None else (0.0, 0.0)
        configured = {
            FramingScale.TIGHT: 35.0,
            FramingScale.MEDIUM: 60.0,
            FramingScale.WIDE: 90.0,
        }[shot.spatial_intent.framing_scale]
        hfov = min(self.fallback_hfov_max, max(self.fallback_hfov_min, configured))
        aspect = (
            self.camera_solver.constraints.output_width
            / self.camera_solver.constraints.output_height
        )
        vfov = math.degrees(2 * math.atan(math.tan(math.radians(hfov) / 2) / aspect))
        scene_view = self.camera_solver.projector.project_view(
            panorama,
            yaw_degrees=yaw,
            pitch_degrees=pitch,
            hfov_degrees=hfov,
            width=self.camera_solver.constraints.output_width,
            height=self.camera_solver.constraints.output_height,
        )
        recipe = SourceStationRecipe(
            source_panorama_hash=panorama.artifact_ref.sha256,
            station_key=station.station_key,
            aim_anchor=aim.target_key if aim is not None else None,
            yaw_degrees=yaw,
            pitch_degrees=pitch,
            hfov_degrees=hfov,
            vfov_degrees=vfov,
            scene_view=scene_view,
        )
        audit = list(prior_audit)
        for item in shot.spatial_intent.spatial_content:
            audit.append(
                self._entry(
                    audit,
                    (
                        "MUST_DEGRADED"
                        if item.strength == SpatialStrength.MUST
                        else "PREFERRED_REMOVED_BY_DEGRADE"
                    ),
                    "Removed the spatial content constraint for the deterministic safe shot.",
                    station_key=station.station_key,
                    target_keys=(item.subject_key,),
                    conflict_kind=conflict.kind,
                )
            )
        audit.append(
            self._entry(
                audit,
                "DETERMINISTIC_SAFE_SHOT",
                (
                    "Used the best visible action-zone anchor."
                    if aim is not None
                    else "No legal anchor exists; used canonical panorama front."
                ),
                station_key=station.station_key,
                target_keys=((aim.target_key,) if aim is not None else ()),
                conflict_kind=conflict.kind,
            )
        )
        content = tuple(
            ContentResolution(
                subject_key=item.subject_key,
                presentation=item.presentation,
                strength=item.strength,
                retained=False,
                detail="removed by deterministic degraded safe shot",
            )
            for item in shot.spatial_intent.spatial_content
        )
        return self._plan(
            shot=shot,
            station=ResolvedStation(
                kind=ResolvedStationKind.SOURCE,
                station_key=station.station_key,
                description=station.description,
            ),
            aim_anchor=aim.target_key if aim is not None else None,
            must_include=(),
            preferred_include=(),
            reveal=(),
            content_resolution=content,
            camera_recipe=recipe,
            fov_solution=None,
            status=SpatialResolutionStatus.DEGRADED,
            audit=tuple(audit),
        )

    def degrade_grounding_unavailable(
        self,
        *,
        shot: StoryShot,
        panorama: PanoramaSource,
        station: SourceStationSpec,
        failed_target_aliases: tuple[str, ...],
    ) -> ResolvedSpatialPlan:
        """Create a front-facing safe shot without inventing any Grounding geometry."""

        configured = {
            FramingScale.TIGHT: 35.0,
            FramingScale.MEDIUM: 60.0,
            FramingScale.WIDE: 90.0,
        }[shot.spatial_intent.framing_scale]
        hfov = min(self.fallback_hfov_max, max(self.fallback_hfov_min, configured), 90.0)
        aspect = (
            self.camera_solver.constraints.output_width
            / self.camera_solver.constraints.output_height
        )
        vfov = math.degrees(2 * math.atan(math.tan(math.radians(hfov) / 2) / aspect))
        scene_view = self.camera_solver.projector.project_view(
            panorama,
            yaw_degrees=0.0,
            pitch_degrees=0.0,
            hfov_degrees=hfov,
            width=self.camera_solver.constraints.output_width,
            height=self.camera_solver.constraints.output_height,
        )
        recipe = SourceStationRecipe(
            source_panorama_hash=panorama.artifact_ref.sha256,
            station_key=station.station_key,
            aim_anchor=None,
            yaw_degrees=0.0,
            pitch_degrees=0.0,
            hfov_degrees=hfov,
            vfov_degrees=vfov,
            scene_view=scene_view,
        )
        audit: list[SpatialAuditEntry] = [
            SpatialAuditEntry(
                sequence=1,
                code="GROUNDING_UNAVAILABLE",
                detail=(
                    "All configured complete Grounding attempts failed validation; no bbox or "
                    "anchor geometry was fabricated."
                ),
                station_key=station.station_key,
                target_keys=failed_target_aliases,
                conflict_kind=CameraConflictKind.INVALID_EVIDENCE,
            )
        ]
        for item in shot.spatial_intent.spatial_content:
            audit.append(
                self._entry(
                    audit,
                    (
                        "MUST_DEGRADED"
                        if item.strength == SpatialStrength.MUST
                        else "PREFERRED_REMOVED_BY_DEGRADE"
                    ),
                    "Removed the spatial content constraint because Grounding is unavailable.",
                    station_key=station.station_key,
                    target_keys=(item.subject_key,),
                    conflict_kind=CameraConflictKind.INVALID_EVIDENCE,
                )
            )
        audit.append(
            self._entry(
                audit,
                "DETERMINISTIC_SAFE_SHOT",
                "Used the canonical panorama front with no include or reveal constraints.",
                station_key=station.station_key,
                conflict_kind=CameraConflictKind.INVALID_EVIDENCE,
            )
        )
        content = tuple(
            ContentResolution(
                subject_key=item.subject_key,
                presentation=item.presentation,
                strength=item.strength,
                retained=False,
                detail="removed because Grounding evidence is unavailable",
            )
            for item in shot.spatial_intent.spatial_content
        )
        return self._plan(
            shot=shot,
            station=ResolvedStation(
                kind=ResolvedStationKind.SOURCE,
                station_key=station.station_key,
                description=station.description,
            ),
            aim_anchor=None,
            must_include=(),
            preferred_include=(),
            reveal=(),
            content_resolution=content,
            camera_recipe=recipe,
            fov_solution=None,
            status=SpatialResolutionStatus.DEGRADED,
            audit=tuple(audit),
        )

    def _unresolved(
        self,
        *,
        shot: StoryShot,
        scene: SceneSpec,
        panorama: PanoramaSource,
        station: SourceStationSpec,
        anchor_map: SceneAnchorMap,
        conflict: CameraConflict,
        audit: list[SpatialAuditEntry],
        allow_novel_station: bool,
    ) -> SpatialResolutionResult:
        if allow_novel_station:
            return SpatialResolutionResult(
                conflict=conflict,
                novel_station_needed=True,
                audit=tuple(audit),
            )
        if self.failure_policy == SpatialFailurePolicy.DEGRADE:
            plan = self.degrade(
                shot=shot,
                scene=scene,
                panorama=panorama,
                station=station,
                anchor_map=anchor_map,
                conflict=conflict,
                prior_audit=tuple(audit),
            )
            return SpatialResolutionResult(plan=plan, conflict=conflict, audit=plan.audit)
        return SpatialResolutionResult(conflict=conflict, audit=tuple(audit))

    def _solve_math(
        self,
        observations: tuple[GroundingObservation, ...],
        framing: FramingScale,
    ) -> CommonFovSolution:
        constraints = self.camera_solver.constraints
        framing_hfov = {
            FramingScale.TIGHT: constraints.tight_hfov_degrees,
            FramingScale.MEDIUM: constraints.medium_hfov_degrees,
            FramingScale.WIDE: constraints.wide_hfov_degrees,
        }[framing]
        return solve_common_fov(
            observations,
            output_width=constraints.output_width,
            output_height=constraints.output_height,
            framing_hfov_degrees=framing_hfov,
            min_hfov_degrees=constraints.min_hfov_degrees,
            max_hfov_degrees=constraints.max_hfov_degrees,
            safe_margin_degrees=constraints.safe_margin_degrees,
        )

    @staticmethod
    def _visible_anchor_observations(
        anchor_map: SceneAnchorMap,
    ) -> dict[str, GroundingObservation]:
        return {
            item.target_key: item
            for item in anchor_map.observations
            if item.is_visible_unique_anchor
        }

    @staticmethod
    def _select_aim(
        scene: SceneSpec,
        shot: StoryShot,
        anchors: dict[str, GroundingObservation],
    ) -> GroundingObservation | None:
        candidates = SpatialResolver._aim_candidates(scene, shot, anchors)
        return candidates[0] if candidates else None

    @staticmethod
    def _aim_candidates(
        scene: SceneSpec,
        shot: StoryShot,
        anchors: dict[str, GroundingObservation],
    ) -> tuple[GroundingObservation, ...]:
        action_zone = shot.spatial_intent.action_zone
        local = [
            anchors[item.anchor_key]
            for item in scene.anchor_landmarks
            if item.zone_key == action_zone and item.anchor_key in anchors
        ]
        candidates = local or list(anchors.values())
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    0 if item.visibility == Visibility.VISIBLE else 1,
                    -item.confidence,
                    item.target_key,
                ),
            )
        )

    @staticmethod
    def _resolve_content(
        scene: SceneSpec,
        anchor_map: SceneAnchorMap,
        content: SpatialContent,
    ) -> tuple[GroundingObservation | None, str | None]:
        anchor = next(
            (item for item in scene.anchor_landmarks if item.anchor_key == content.subject_key),
            None,
        )
        if anchor is not None:
            observation = next(
                (
                    item
                    for item in anchor_map.observations_for(anchor.anchor_key)
                    if item.is_visible_unique_anchor
                ),
                None,
            )
            return observation, anchor.anchor_key if observation is not None else None
        region = next(
            (item for item in scene.scene_regions if item.region_key == content.subject_key),
            None,
        )
        if region is None:
            return None, None
        visible_anchors = {item.target_key: item for item in anchor_map.visible_unique_anchors()}
        representatives = tuple(
            visible_anchors[key]
            for key in region.representative_anchor_keys
            if key in visible_anchors
        )
        if not representatives:
            return None, None
        observation = sorted(
            representatives,
            key=lambda item: (
                0 if item.visibility == Visibility.VISIBLE else 1,
                -item.confidence,
                item.target_key,
            ),
        )[0]
        return observation, observation.target_key

    @staticmethod
    def _content_anchor_keys(
        scene: SceneSpec,
        content: SpatialContent,
    ) -> tuple[str, ...]:
        if any(item.anchor_key == content.subject_key for item in scene.anchor_landmarks):
            return (content.subject_key,)
        region = next(
            (item for item in scene.scene_regions if item.region_key == content.subject_key),
            None,
        )
        return region.representative_anchor_keys if region is not None else ()

    @staticmethod
    def _non_unique_anchor_keys(
        anchor_map: SceneAnchorMap,
        *,
        allowed_keys: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        allowed = set(allowed_keys) if allowed_keys is not None else None
        return tuple(
            sorted(
                {
                    item.target_key
                    for item in anchor_map.observations
                    if item.target_kind == GroundingTargetKind.ANCHOR
                    and item.visibility in LOCALIZED_VISIBILITIES
                    and item.normalized_box is not None
                    and item.is_unique is False
                    and (allowed is None or item.target_key in allowed)
                }
            )
        )

    @staticmethod
    def _deduplicate_observations(
        observations: list[GroundingObservation] | tuple[GroundingObservation, ...],
    ) -> tuple[GroundingObservation, ...]:
        return tuple({item.observation_key: item for item in observations}.values())

    @staticmethod
    def _validate_inputs(
        shot: StoryShot,
        scene: SceneSpec,
        panorama: PanoramaSource,
        station: SourceStationSpec,
        anchor_map: SceneAnchorMap,
    ) -> None:
        if len({shot.scene_key, scene.scene_key, panorama.scene_key, anchor_map.scene_key}) != 1:
            raise SpatialError("spatial resolver inputs belong to different scenes")
        if anchor_map.reference_hash != panorama.artifact_ref.sha256:
            raise SpatialError("Scene Anchor Map belongs to another reference")
        if anchor_map.station_key != station.station_key:
            raise SpatialError("Scene Anchor Map belongs to another station")

    @staticmethod
    def _observation_center(
        observation: GroundingObservation | None,
    ) -> tuple[float, float]:
        if observation is None or observation.normalized_box is None:
            return 0.0, 0.0
        box = observation.normalized_box
        x = (box.x_min + box.x_max) / 2
        y = (box.y_min + box.y_max) / 2
        matrix = np.asarray(observation.orientation_matrix, dtype=np.float64).reshape(3, 3)
        tangent = math.tan(math.radians(observation.probe_hfov_degrees) / 2)
        local = np.asarray(((x - 0.5) * 2 * tangent, (0.5 - y) * 2 * tangent, 1.0))
        local /= np.linalg.norm(local)
        world = matrix @ local
        world /= np.linalg.norm(world)
        return (
            math.degrees(math.atan2(float(world[0]), float(world[2]))),
            math.degrees(math.asin(float(np.clip(world[1], -1.0, 1.0)))),
        )

    @staticmethod
    def _entry(
        audit: list[SpatialAuditEntry],
        code: str,
        detail: str,
        *,
        station_key: str | None = None,
        target_keys: tuple[str, ...] = (),
        conflict_kind: CameraConflictKind | None = None,
        fov_solution: CommonFovSolution | None = None,
    ) -> SpatialAuditEntry:
        return SpatialAuditEntry(
            sequence=len(audit) + 1,
            code=code,
            detail=detail,
            station_key=station_key,
            target_keys=target_keys,
            conflict_kind=conflict_kind,
            fov_solution=fov_solution,
        )

    def _conflict_entry(
        self,
        audit: list[SpatialAuditEntry],
        conflict: CameraConflict,
        station_key: str,
    ) -> SpatialAuditEntry:
        return self._entry(
            audit,
            conflict.kind.value.upper(),
            conflict.detail,
            station_key=station_key,
            target_keys=conflict.target_keys,
            conflict_kind=conflict.kind,
            fov_solution=conflict.fov_solution,
        )

    @staticmethod
    def _plan(
        *,
        shot: StoryShot,
        station: ResolvedStation,
        aim_anchor: str | None,
        must_include: tuple[str, ...],
        preferred_include: tuple[str, ...],
        reveal: tuple[str, ...],
        content_resolution: tuple[ContentResolution, ...],
        camera_recipe: SourceStationRecipe | NovelStationRecipe,
        fov_solution: CommonFovSolution | None,
        status: SpatialResolutionStatus,
        audit: tuple[SpatialAuditEntry, ...],
    ) -> ResolvedSpatialPlan:
        original = tuple(
            ContentResolution(
                subject_key=item.subject_key,
                presentation=item.presentation,
                strength=item.strength,
            )
            for item in shot.spatial_intent.spatial_content
        )
        must_include_anchors = tuple(dict.fromkeys(must_include))
        preferred_include_anchors = tuple(dict.fromkeys(preferred_include))
        reveal_anchors = tuple(dict.fromkeys(reveal))
        payload = {
            "shot_key": shot.shot_key,
            "scene_key": shot.scene_key,
            "original_story_targets": shot.spatial_intent.story_targets,
            "original_spatial_content": original,
            "original_viewpoint_intent": shot.spatial_intent.viewpoint_intent.description,
            "station": station,
            "aim_anchor": aim_anchor,
            "must_include_anchors": must_include_anchors,
            "preferred_include_anchors": preferred_include_anchors,
            "reveal_anchors": reveal_anchors,
            "content_resolution": content_resolution,
            "camera_recipe": camera_recipe,
            "fov_solution": fov_solution,
            "result_status": status,
            "audit": audit,
        }
        plan = ResolvedSpatialPlan(
            shot_key=shot.shot_key,
            scene_key=shot.scene_key,
            original_story_targets=shot.spatial_intent.story_targets,
            original_spatial_content=original,
            original_viewpoint_intent=shot.spatial_intent.viewpoint_intent.description,
            station=station,
            aim_anchor=aim_anchor,
            must_include_anchors=must_include_anchors,
            preferred_include_anchors=preferred_include_anchors,
            reveal_anchors=reveal_anchors,
            content_resolution=content_resolution,
            camera_recipe=camera_recipe,
            fov_solution=fov_solution,
            result_status=status,
            audit=audit,
            resolution_hash=canonical_hash(payload),
        )
        plan.assert_hash()
        return plan
