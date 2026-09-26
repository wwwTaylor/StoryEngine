"""Freeze an executable RenderPlan only after real spatial evidence exists."""

from __future__ import annotations

from story_engine.domain.evaluation import CriterionKind
from story_engine.domain.reference import (
    ReferenceLibrary,
    SelectedReference,
    visible_state_for,
)
from story_engine.domain.render import (
    AdditionalReference,
    InputBinding,
    RenderPlan,
    RenderShot,
    RequirementPhase,
    ResolvedCriterion,
    SemanticOnly,
    VisibleInStartFrame,
)
from story_engine.domain.request import ProjectRequest
from story_engine.domain.spatial import ResolvedSpatialPlan, SpatialResolutionStatus
from story_engine.domain.state import (
    AttachedTo,
    HeldBy,
    InContainer,
    InSceneZone,
    Offscreen,
    OnSurface,
    Placement,
    WorldReducer,
    WorldState,
)
from story_engine.domain.story import (
    EntitySpec,
    RequirementCategory,
    RequirementKind,
    StoryPlan,
    StoryShot,
)
from story_engine.errors import ArtifactError, RenderCompilationError
from story_engine.ids import canonical_hash, stable_key
from story_engine.providers.ports import RuntimeCapabilities
from story_engine.spatial.camera import CameraConflictKind, NovelStationRecipe
from story_engine.spatial.continuity import validate_spatial_continuity
from story_engine.storage import ArtifactStore


class RenderCompiler:
    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    def compile(
        self,
        request: ProjectRequest,
        story_plan: StoryPlan,
        reference_library: ReferenceLibrary,
        spatial_evidence: tuple[ResolvedSpatialPlan, ...],
        capabilities: RuntimeCapabilities,
        *,
        version: int = 3,
    ) -> RenderPlan:
        story_plan.assert_hash()
        reference_library.assert_hash()
        if reference_library.story_plan_hash != story_plan.plan_hash:
            raise RenderCompilationError("ReferenceLibrary belongs to another StoryPlan")
        image_capability = capabilities.image.image
        video_capability = capabilities.video.video
        if image_capability is None or video_capability is None:
            raise RenderCompilationError("image and video capabilities are required")
        if not video_capability.supports_start_image:
            raise RenderCompilationError("VideoProvider must support a start image")
        if not image_capability.supports_resolution(request.resolution):
            raise RenderCompilationError("ImageProvider does not support requested resolution")
        if not video_capability.supports_resolution(request.resolution):
            raise RenderCompilationError("VideoProvider does not support requested resolution")
        delivery = request.generation_requirements.delivery_requirements
        if delivery.video_fps not in video_capability.supported_fps:
            raise RenderCompilationError("VideoProvider does not support requested FPS")

        continuity_issues = validate_spatial_continuity(story_plan)
        if continuity_issues:
            raise RenderCompilationError(
                "spatial continuity conflicts: "
                + ", ".join(issue.code for issue in continuity_issues)
            )
        evidence_by_shot = {item.shot_key: item for item in spatial_evidence}
        if len(evidence_by_shot) != len(spatial_evidence):
            raise RenderCompilationError("duplicate spatial evidence for one shot")
        expected_shots = {shot.shot_key for shot in story_plan.ordered_shots}
        if set(evidence_by_shot) != expected_shots:
            raise RenderCompilationError("spatial evidence must exactly cover all shots")

        reducer = WorldReducer(story_plan.world_rules())
        boundaries = reducer.boundaries(
            story_plan.initial_world,
            tuple(
                (
                    shot.shot_key,
                    tuple(beat.transition for beat in shot.beats if beat.transition is not None),
                )
                for shot in story_plan.ordered_shots
            ),
        )
        boundaries_by_shot = {item.shot_key: item for item in boundaries}
        entities = {item.entity_key: item for item in story_plan.entity_catalog}
        references = self._reference_map(reference_library)
        panoramas = {
            item.scene_key: item.artifact_ref for item in reference_library.panorama_by_scene
        }
        render_shots: list[RenderShot] = []
        for shot in story_plan.ordered_shots:
            if shot.duration not in video_capability.supported_durations:
                raise RenderCompilationError(
                    f"shot {shot.shot_key} uses unsupported duration {shot.duration}"
                )
            evidence = evidence_by_shot[shot.shot_key]
            evidence.assert_hash()
            if evidence.scene_key != shot.scene_key:
                raise RenderCompilationError("spatial evidence scene mismatch")
            panorama = panoramas.get(shot.scene_key)
            if panorama is None:
                raise RenderCompilationError(f"scene has no panorama: {shot.scene_key}")
            if evidence.camera_recipe.source_panorama_hash != panorama.sha256:
                raise RenderCompilationError("camera recipe uses another panorama")
            selected_scene_panorama = (
                evidence.camera_recipe.station_panorama
                if isinstance(evidence.camera_recipe, NovelStationRecipe)
                else panorama
            )
            try:
                self.store.verify(evidence.camera_recipe.scene_view)
                self.store.verify(panorama)
                self.store.verify(selected_scene_panorama)
            except ArtifactError as exc:
                raise RenderCompilationError(str(exc)) from exc

            boundary = boundaries_by_shot[shot.shot_key]
            start_visible = self._visible_entities(shot.visible_entities, boundary.planned_start)
            end_visible = self._visible_entities(shot.visible_entities, boundary.planned_end)
            bindings = self._bindings(
                shot,
                boundary.planned_start,
                start_visible,
                entities,
                references,
                reducer,
                image_capability.max_input_images,
                video_capability.max_reference_images,
            )
            criteria = self._criteria(
                story_plan,
                shot,
                boundary.planned_start,
                boundary.planned_end,
                entities,
                start_visible,
                end_visible,
            )
            render_shots.append(
                RenderShot(
                    shot_key=shot.shot_key,
                    planned_start_boundary=boundary.planned_start,
                    planned_end_boundary=boundary.planned_end,
                    selected_scene_view=evidence.camera_recipe.scene_view,
                    selected_scene_panorama=selected_scene_panorama,
                    camera_recipe=evidence.camera_recipe,
                    spatial_resolution_hash=evidence.resolution_hash,
                    spatial_status=evidence.result_status,
                    spatial_failure_codes=_spatial_failure_codes(evidence),
                    dropped_spatial_content=tuple(
                        dict.fromkeys(
                            item.subject_key
                            for item in evidence.content_resolution
                            if not item.retained
                        )
                    ),
                    reveal_anchors=evidence.reveal_anchors,
                    camera_motion=(
                        shot.spatial_intent.camera_motion_intent
                        if video_capability.supports_camera_motion
                        and evidence.result_status != SpatialResolutionStatus.DEGRADED
                        else "static locked camera"
                    ),
                    input_bindings=bindings,
                    start_visible_entities=start_visible,
                    end_visible_entities=end_visible,
                    resolved_requirements=criteria,
                )
            )

        plan_key = stable_key(
            "render_plan",
            story_plan.plan_hash,
            f"{reference_library.library_hash}:v{version}",
        )
        payload = {
            "render_plan_key": plan_key,
            "version": version,
            "story_plan_hash": story_plan.plan_hash,
            "reference_library_hash": reference_library.library_hash,
            "provider_profile_hash": capabilities.fingerprint,
            "ordered_render_shots": tuple(render_shots),
        }
        plan = RenderPlan(
            render_plan_key=plan_key,
            version=version,
            story_plan_hash=story_plan.plan_hash,
            reference_library_hash=reference_library.library_hash,
            provider_profile_hash=capabilities.fingerprint,
            ordered_render_shots=tuple(render_shots),
            render_plan_hash=canonical_hash(payload),
        )
        plan.assert_hash()
        return plan

    @staticmethod
    def _reference_map(
        library: ReferenceLibrary,
    ) -> dict[str, tuple[SelectedReference, ...]]:
        grouped: dict[str, list[SelectedReference]] = {}
        for reference in library.selected_assets:
            grouped.setdefault(reference.subject_key, []).append(reference)
        return {
            key: tuple(sorted(values, key=lambda item: item.need_key))
            for key, values in grouped.items()
        }

    @staticmethod
    def _visible_entities(
        participating_entities: tuple[str, ...], state: WorldState
    ) -> tuple[str, ...]:
        result: list[str] = []
        for entity_key in participating_entities:
            placement = state.placement_for(entity_key)
            if not isinstance(placement, (Offscreen, InContainer)):
                result.append(entity_key)
        return tuple(result)

    def _bindings(
        self,
        shot: StoryShot,
        planned_start: WorldState,
        start_visible: tuple[str, ...],
        entities: dict[str, EntitySpec],
        references: dict[str, tuple[SelectedReference, ...]],
        reducer: WorldReducer,
        image_input_limit: int,
        video_reference_limit: int,
    ) -> tuple[InputBinding, ...]:
        bindings: list[InputBinding] = []
        start_reference_count = 0
        later_reference_count = 0
        for entity_key in shot.visible_entities:
            entity = entities[entity_key]
            if not entity.freeze_appearance:
                bindings.append(
                    SemanticOnly(entity_key=entity_key, description=entity.visual_identity)
                )
                continue
            subject_references = references.get(entity_key)
            if subject_references is None:
                raise RenderCompilationError(
                    f"frozen entity has no selected reference: {entity.alias}"
                )
            states = [planned_start]
            current = planned_start
            for beat in shot.beats:
                if beat.transition is not None:
                    current = reducer.apply(current, beat.transition)
                    states.append(current)
            required_references: list[SelectedReference] = []
            for state in states:
                visible_state = visible_state_for(
                    entity_key,
                    state,
                    attribute_keys=tuple(
                        definition.key
                        for definition in entity.attribute_definitions
                        if definition.is_visual
                    ),
                )
                reference = self._reference_for_state(
                    entity,
                    subject_references,
                    visible_state,
                )
                if reference.need_key not in {item.need_key for item in required_references}:
                    required_references.append(reference)
            reference = required_references[0]
            try:
                for item in required_references:
                    self.store.verify(item.artifact_ref)
                    for view in item.character_views:
                        self.store.verify(view.artifact_ref)
            except ArtifactError as exc:
                raise RenderCompilationError(str(exc)) from exc
            if entity_key in start_visible:
                start_reference_count += len(reference.character_views) or 1
                bindings.append(
                    VisibleInStartFrame(
                        entity_key=entity_key,
                        reference_asset=reference.artifact_ref,
                        reference_views=reference.character_views,
                    )
                )
            else:
                later_reference_count += 1
                bindings.append(
                    AdditionalReference(
                        entity_key=entity_key,
                        reference_asset=reference.artifact_ref,
                        provider_channel="video.reference_images",
                    )
                )
            for state_reference in required_references[1:]:
                later_reference_count += 1
                bindings.append(
                    AdditionalReference(
                        entity_key=entity_key,
                        reference_asset=state_reference.artifact_ref,
                        provider_channel="video.reference_images",
                    )
                )
        # Two input slots belong to the selected scene view and its matching panorama.
        if 2 + start_reference_count > image_input_limit:
            raise RenderCompilationError(
                f"shot {shot.shot_key} needs {2 + start_reference_count} image inputs, "
                f"provider supports {image_input_limit}"
            )
        if later_reference_count > video_reference_limit:
            raise RenderCompilationError(
                f"shot {shot.shot_key} needs {later_reference_count} video references, "
                f"provider supports {video_reference_limit}; split the shot upstream"
            )
        return tuple(bindings)

    @staticmethod
    def _reference_for_state(
        entity: EntitySpec,
        references: tuple[SelectedReference, ...],
        visible_state: str | None,
    ) -> SelectedReference:
        state_references = tuple(item for item in references if item.visible_state is not None)
        if state_references:
            matches = tuple(
                item for item in state_references if item.visible_state == visible_state
            )
        else:
            matches = references
        if len(matches) != 1:
            raise RenderCompilationError(
                f"no unique reference for {entity.alias} state {visible_state!r}"
            )
        return matches[0]

    def _criteria(
        self,
        plan: StoryPlan,
        shot: StoryShot,
        start: WorldState,
        end: WorldState,
        entities: dict[str, EntitySpec],
        start_visible: tuple[str, ...],
        end_visible: tuple[str, ...],
    ) -> tuple[ResolvedCriterion, ...]:
        criteria: list[ResolvedCriterion] = []
        entity_aliases = {key: item.alias for key, item in entities.items()}
        zone_aliases = {
            zone.zone_key: zone.alias for scene in plan.scene_catalog for zone in scene.zones
        }
        scene_aliases = {scene.scene_key: scene.alias for scene in plan.scene_catalog}
        requirements = {
            item.requirement_key: item
            for item in plan.requirements
            if item.requirement_key in shot.requirement_refs
        }
        for requirement in requirements.values():
            if (
                requirement.category == RequirementCategory.SHOT_LOCAL
                and requirement.owner_shot_key != shot.shot_key
            ):
                raise RenderCompilationError(
                    "shot-local requirement is attached to a non-owner shot"
                )
            criteria.append(
                ResolvedCriterion(
                    criterion_id=requirement.requirement_key,
                    kind=(
                        CriterionKind.REQUIREMENT
                        if requirement.kind == RequirementKind.REQUIREMENT
                        else CriterionKind.PREFERENCE
                    ),
                    priority=requirement.priority,
                    phase=RequirementPhase.ALWAYS,
                    statement=requirement.description,
                    category=requirement.category.value,
                    owner_shot_key=requirement.owner_shot_key,
                )
            )
            if requirement.category == RequirementCategory.SHOT_LOCAL:
                criteria.append(
                    ResolvedCriterion(
                        criterion_id=stable_key(
                            "criterion",
                            shot.shot_key,
                            f"opening_requirement:{requirement.requirement_key}",
                        ),
                        kind=CriterionKind.REQUIREMENT,
                        priority=max(85, requirement.priority),
                        phase=RequirementPhase.FIRST_FRAME,
                        statement=(
                            "Opening state has no visible setup that conflicts with this "
                            f"shot-local constraint: {requirement.description}"
                        ),
                        category="action_setup",
                        owner_shot_key=shot.shot_key,
                    )
                )
        for entity_key in start_visible:
            criteria.extend(
                self._state_criteria(
                    shot.shot_key,
                    entities[entity_key],
                    start,
                    RequirementPhase.FIRST_FRAME,
                    entity_aliases,
                    zone_aliases,
                    scene_aliases,
                )
            )
        state_before = start
        reducer = WorldReducer(plan.world_rules())
        for index, beat in enumerate(shot.beats):
            criteria.append(
                ResolvedCriterion(
                    criterion_id=stable_key(
                        "criterion",
                        shot.shot_key,
                        f"opening_beat:{index}:{beat.beat_key}",
                    ),
                    kind=CriterionKind.REQUIREMENT,
                    priority=85,
                    phase=RequirementPhase.FIRST_FRAME,
                    statement=(
                        "Opening state has no visible prop or action target that conflicts "
                        f"with Beat {index + 1}: {beat.action}"
                    ),
                    category="action_setup",
                    owner_shot_key=shot.shot_key,
                )
            )
            criteria.append(
                ResolvedCriterion(
                    criterion_id=stable_key(
                        "criterion", shot.shot_key, f"beat:{index}:{beat.beat_key}"
                    ),
                    kind=CriterionKind.REQUIREMENT,
                    priority=80,
                    phase=RequirementPhase.MOTION,
                    statement=f"Beat {index + 1}: {beat.action}",
                    category="action",
                    owner_shot_key=shot.shot_key,
                )
            )
            if beat.transition is not None:
                state_before = reducer.apply(state_before, beat.transition)
        if state_before != end:
            raise RenderCompilationError("beat transitions do not reproduce planned end")
        for entity_key in end_visible:
            criteria.extend(
                self._state_criteria(
                    shot.shot_key,
                    entities[entity_key],
                    end,
                    RequirementPhase.END,
                    entity_aliases,
                    zone_aliases,
                    scene_aliases,
                )
            )
        deduplicated = {item.criterion_id: item for item in criteria}
        return tuple(sorted(deduplicated.values(), key=lambda item: item.criterion_id))

    @staticmethod
    def _state_criteria(
        shot_key: str,
        entity: EntitySpec,
        state: WorldState,
        phase: RequirementPhase,
        entity_aliases: dict[str, str],
        zone_aliases: dict[str, str],
        scene_aliases: dict[str, str],
    ) -> tuple[ResolvedCriterion, ...]:
        facts: list[ResolvedCriterion] = []
        placement = state.placement_for(entity.entity_key)
        facts.append(
            ResolvedCriterion(
                criterion_id=stable_key(
                    "criterion",
                    shot_key,
                    f"{phase.value}:placement:{entity.entity_key}",
                ),
                kind=CriterionKind.REQUIREMENT,
                priority=85,
                phase=phase,
                statement=(
                    f"{entity.alias} placement is "
                    f"{_describe_placement(placement, entity_aliases, zone_aliases, scene_aliases)}"
                ),
                category="key_state",
                owner_shot_key=shot_key,
            )
        )
        for fact in state.attributes_by_entity:
            if fact.entity_key != entity.entity_key:
                continue
            facts.append(
                ResolvedCriterion(
                    criterion_id=stable_key(
                        "criterion",
                        shot_key,
                        f"{phase.value}:attribute:{entity.entity_key}:{fact.attribute_key}",
                    ),
                    kind=CriterionKind.REQUIREMENT,
                    priority=85,
                    phase=phase,
                    statement=f"{entity.alias} {fact.attribute_key} is {fact.value}",
                    category="key_state",
                    owner_shot_key=shot_key,
                )
            )
        return tuple(facts)


def _spatial_failure_codes(evidence: ResolvedSpatialPlan) -> tuple[str, ...]:
    known = {
        *(item.value.upper() for item in CameraConflictKind),
        "GROUNDING_UNAVAILABLE",
        "NOVEL_STATION_FAILED",
    }
    return tuple(dict.fromkeys(item.code for item in evidence.audit if item.code in known))


def _describe_placement(
    placement: Placement,
    entity_aliases: dict[str, str],
    zone_aliases: dict[str, str],
    scene_aliases: dict[str, str],
) -> str:
    if isinstance(placement, OnSurface):
        return f"on {_required_alias(entity_aliases, placement.surface_entity, 'surface entity')}"
    if isinstance(placement, InContainer):
        return f"inside {_required_alias(entity_aliases, placement.container_entity, 'container')}"
    if isinstance(placement, HeldBy):
        return f"held by {_required_alias(entity_aliases, placement.character_entity, 'character')}"
    if isinstance(placement, AttachedTo):
        return f"attached to {_required_alias(entity_aliases, placement.entity, 'entity')}"
    if isinstance(placement, InSceneZone):
        scene = _required_alias(scene_aliases, placement.scene_key, "scene")
        zone = _required_alias(zone_aliases, placement.zone_key, "zone")
        return f"in {scene} zone {zone}"
    if isinstance(placement, Offscreen):
        if placement.scene_key is None:
            return "offscreen"
        return f"offscreen in {_required_alias(scene_aliases, placement.scene_key, 'scene')}"
    raise RenderCompilationError(f"unsupported placement kind {placement.kind!r}")


def _required_alias(mapping: dict[str, str], key: str, kind: str) -> str:
    try:
        return mapping[key]
    except KeyError as exc:
        raise RenderCompilationError(f"placement references unknown {kind}") from exc
