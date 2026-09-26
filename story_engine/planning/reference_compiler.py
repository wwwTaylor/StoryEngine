"""Derive local reference recipes and compile selected references."""

from __future__ import annotations

from story_engine.domain.evaluation import (
    CandidateRecord,
    EvaluationReport,
    SelectionDecision,
    SelectionOutcome,
    TechnicalStatus,
)
from story_engine.domain.reference import (
    CHARACTER_VIEW_ROLES,
    CharacterReferenceRecipe,
    CharacterReferenceView,
    GuidedCharacterReferenceRecipe,
    GuidedScenePanoramaRecipe,
    PropReferenceRecipe,
    ProvidedReferenceRecipe,
    ReferenceLibrary,
    ReferenceRecipe,
    ScenePanoramaBinding,
    ScenePanoramaRecipe,
    SceneSpatialReference,
    SelectedReference,
    SourceStationSpec,
)
from story_engine.domain.request import ProjectRequest
from story_engine.domain.story import ReferenceNeedKind, SceneSpec, StoryPlan
from story_engine.errors import ReferenceError
from story_engine.ids import canonical_hash, stable_key


class ReferenceCompiler:
    def recipes(
        self, request: ProjectRequest, story_plan: StoryPlan
    ) -> tuple[ReferenceRecipe, ...]:
        entities = {item.entity_key: item for item in story_plan.entity_catalog}
        scenes = {item.scene_key: item for item in story_plan.scene_catalog}
        assets = {item.asset_id: item for item in request.provided_assets}
        recipes: list[ReferenceRecipe] = []
        entity_aliases = tuple(item.alias for item in story_plan.entity_catalog)
        for need in story_plan.reference_needs:
            source_station = (
                self._source_station(story_plan, scenes[need.subject_key])
                if need.kind == ReferenceNeedKind.SCENE_PANORAMA
                else None
            )
            if need.provided_asset_id is not None:
                asset = assets.get(need.provided_asset_id)
                if asset is None:
                    raise ReferenceError(
                        f"provided asset is missing: {need.provided_asset_id}"
                    )
                if need.kind == ReferenceNeedKind.CHARACTER:
                    entity = entities[need.subject_key]
                    recipes.append(
                        GuidedCharacterReferenceRecipe(
                            need_key=need.need_key,
                            subject_key=need.subject_key,
                            appearance=entity.visual_identity,
                            style=request.visual_style,
                            forbidden_co_subjects=tuple(
                                alias for alias in entity_aliases if alias != entity.alias
                            ),
                            provided_asset_id=asset.asset_id,
                        )
                    )
                elif (
                    need.kind == ReferenceNeedKind.SCENE_PANORAMA
                    and asset.kind != "panorama"
                ):
                    scene = scenes[need.subject_key]
                    if source_station is None:
                        raise ReferenceError("scene panorama has no source station")
                    recipes.append(
                        self._scene_recipe(
                            request,
                            scene,
                            source_station,
                            need_key=need.need_key,
                            provided_asset_id=asset.asset_id,
                        )
                    )
                else:
                    recipes.append(
                        ProvidedReferenceRecipe(
                            need_key=need.need_key,
                            subject_key=need.subject_key,
                            provided_asset_id=need.provided_asset_id,
                            reference_kind=need.kind,
                            require_panorama=(need.kind == ReferenceNeedKind.SCENE_PANORAMA),
                            source_station=source_station,
                        )
                    )
            elif need.kind == ReferenceNeedKind.CHARACTER:
                entity = entities[need.subject_key]
                recipes.append(
                    CharacterReferenceRecipe(
                        need_key=need.need_key,
                        subject_key=need.subject_key,
                        appearance=entity.visual_identity,
                        style=request.visual_style,
                        forbidden_co_subjects=tuple(
                            alias for alias in entity_aliases if alias != entity.alias
                        ),
                    )
                )
            elif need.kind in {ReferenceNeedKind.PROP, ReferenceNeedKind.PROP_STATE}:
                entity = entities[need.subject_key]
                recipes.append(
                    PropReferenceRecipe(
                        need_key=need.need_key,
                        subject_key=need.subject_key,
                        appearance=entity.visual_identity,
                        visible_state=need.visible_state,
                        style=request.visual_style,
                        forbidden_co_subjects=tuple(
                            alias for alias in entity_aliases if alias != entity.alias
                        ),
                    )
                )
            elif need.kind == ReferenceNeedKind.SCENE_PANORAMA:
                scene = scenes[need.subject_key]
                if source_station is None:
                    raise ReferenceError("scene panorama has no source station")
                recipes.append(
                    self._scene_recipe(
                        request,
                        scene,
                        source_station,
                        need_key=need.need_key,
                    )
                )
            elif need.kind == ReferenceNeedKind.PROVIDED:
                raise ReferenceError("provided reference need has no asset id")
            else:
                raise ReferenceError(f"unsupported reference need kind {need.kind}")
        return tuple(sorted(recipes, key=lambda item: item.need_key))

    @staticmethod
    def _scene_recipe(
        request: ProjectRequest,
        scene: SceneSpec,
        source_station: SourceStationSpec,
        *,
        need_key: str,
        provided_asset_id: str | None = None,
    ) -> ScenePanoramaRecipe:
        spatial_alias = (
            {zone.zone_key: zone.alias for zone in scene.zones}
            | {region.region_key: region.alias for region in scene.scene_regions}
            | {anchor.anchor_key: anchor.alias for anchor in scene.anchor_landmarks}
        )
        values = dict(
            need_key=need_key,
            subject_key=scene.scene_key,
            scene_visual_identity=scene.visual_identity,
            zones_and_landmarks=tuple(
                [f"zone {zone.alias}: {zone.description}" for zone in scene.zones]
                + [
                    f"region {region.alias}: {region.description}"
                    for region in scene.scene_regions
                ]
                + [
                    f"anchor {anchor.alias}: {anchor.description}"
                    for anchor in scene.anchor_landmarks
                ]
            ),
            semantic_layout=tuple(
                f"{spatial_alias[relation.subject_key]} "
                f"{relation.relation.value} "
                f"{spatial_alias[relation.object_key]}"
                for relation in scene.semantic_relations
            ),
            lighting=scene.lighting,
            style=scene.style or request.visual_style,
            allowed_population=(),
            source_station=source_station,
        )
        if provided_asset_id is not None:
            return GuidedScenePanoramaRecipe(
                **values,
                provided_asset_id=provided_asset_id,
            )
        return ScenePanoramaRecipe(**values)

    def build_library(
        self,
        story_plan: StoryPlan,
        decisions: tuple[SelectionDecision, ...],
        candidates: tuple[CandidateRecord, ...],
        reports: tuple[EvaluationReport, ...],
    ) -> ReferenceLibrary:
        decisions_by_operation = {item.operation_key: item for item in decisions}
        candidates_by_key = {item.candidate_key: item for item in candidates}
        reports_by_key = {item.report_key: item for item in reports}
        if len(decisions_by_operation) != len(decisions):
            raise ReferenceError("multiple reference decisions for one operation")

        selected: list[SelectedReference] = []
        panoramas: list[ScenePanoramaBinding] = []
        for need in story_plan.reference_needs:
            decision = decisions_by_operation.get(need.need_key)
            if decision is None:
                raise ReferenceError(f"reference need was not selected: {need.need_key}")
            candidate = candidates_by_key.get(decision.selected_candidate)
            report = reports_by_key.get(decision.selected_report)
            if candidate is None or report is None:
                raise ReferenceError("selection references missing candidate or report")
            if candidate.operation_key != need.need_key:
                raise ReferenceError("selected candidate belongs to another operation")
            if (
                candidate.technical_status != TechnicalStatus.VALID
                or candidate.artifact_ref is None
            ):
                raise ReferenceError("selected reference is not technically valid")
            if report.candidate_key != candidate.candidate_key:
                raise ReferenceError("evaluation report candidate mismatch")
            character_views: tuple[CharacterReferenceView, ...] = ()
            if need.kind == ReferenceNeedKind.CHARACTER:
                roles = tuple(item.role for item in candidate.media)
                if roles != ("front", "side", "back"):
                    raise ReferenceError(
                        "selected character candidate lacks ordered front, side, and back media"
                    )
                character_views = tuple(
                    CharacterReferenceView(
                        role=role,
                        artifact_ref=item.artifact_ref,
                    )
                    for role, item in zip(
                        CHARACTER_VIEW_ROLES,
                        candidate.media,
                        strict=True,
                    )
                )
                if character_views[0].artifact_ref != candidate.artifact_ref:
                    raise ReferenceError(
                        "selected character primary artifact is not its front view"
                    )
            item = SelectedReference(
                need_key=need.need_key,
                subject_key=need.subject_key,
                kind=need.kind,
                candidate_key=candidate.candidate_key,
                artifact_ref=candidate.artifact_ref,
                character_views=character_views,
                evaluation_ref=report.report_key,
                degraded=decision.outcome == SelectionOutcome.SELECTED_DEGRADED,
                visible_state=need.visible_state,
            )
            selected.append(item)
            if need.kind == ReferenceNeedKind.SCENE_PANORAMA:
                panoramas.append(
                    ScenePanoramaBinding(
                        scene_key=need.subject_key,
                        selected_reference=need.need_key,
                        artifact_ref=candidate.artifact_ref,
                        source_station=self._source_station(
                            story_plan,
                            next(
                                scene
                                for scene in story_plan.scene_catalog
                                if scene.scene_key == need.subject_key
                            ),
                        ),
                    )
                )
        selected_assets = tuple(sorted(selected, key=lambda item: item.need_key))
        panorama_bindings = tuple(sorted(panoramas, key=lambda item: item.scene_key))
        evaluation_refs = tuple(sorted(item.evaluation_ref for item in selected))
        payload = {
            "story_plan_hash": story_plan.plan_hash,
            "selected_assets": selected_assets,
            "panorama_by_scene": panorama_bindings,
            "spatial_by_scene": (),
            "evaluation_refs": evaluation_refs,
        }
        library = ReferenceLibrary(
            story_plan_hash=story_plan.plan_hash,
            selected_assets=selected_assets,
            panorama_by_scene=panorama_bindings,
            spatial_by_scene=(),
            evaluation_refs=evaluation_refs,
            library_hash=canonical_hash(payload),
        )
        library.assert_hash()
        return library

    @staticmethod
    def attach_spatial_references(
        library: ReferenceLibrary,
        spatial_references: tuple[SceneSpatialReference, ...],
    ) -> ReferenceLibrary:
        payload = library.model_dump(exclude={"library_hash", "spatial_by_scene"})
        payload["spatial_by_scene"] = tuple(
            sorted(spatial_references, key=lambda item: item.scene_key)
        )
        updated = ReferenceLibrary(**payload, library_hash=canonical_hash(payload))
        updated.assert_hash()
        return updated

    @staticmethod
    def _source_station(story_plan: StoryPlan, scene: SceneSpec) -> SourceStationSpec:
        shots = tuple(
            shot for shot in story_plan.ordered_shots if shot.scene_key == scene.scene_key
        )
        preferred = (
            tuple(
                shot
                for shot in shots
                if not shot.spatial_intent.viewpoint_intent.translation_expected
            )
            or shots
        )
        zone_counts: dict[str, int] = {}
        for shot in preferred:
            action_zone_key = shot.spatial_intent.action_zone
            zone_counts[action_zone_key] = zone_counts.get(action_zone_key, 0) + 1
        zone_key = (
            sorted(zone_counts, key=lambda key: (-zone_counts[key], key))[0]
            if zone_counts
            else scene.zones[0].zone_key
        )
        zone = next(item for item in scene.zones if item.zone_key == zone_key)
        supported = tuple(
            sorted(
                shot.spatial_intent.viewpoint_intent.intent_key
                for shot in shots
                if not shot.spatial_intent.viewpoint_intent.translation_expected
            )
        )
        description = (
            f"Canonical panorama source station inside {zone.alias}: {zone.description} "
            "at standing eye height."
        )
        return SourceStationSpec(
            station_key=stable_key("source_station", scene.scene_key, zone_key),
            zone_key=zone_key,
            description=description,
            supported_viewpoint_intents=supported,
        )
