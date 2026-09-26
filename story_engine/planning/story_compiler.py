"""Compile creative aliases into a deterministic immutable StoryPlan."""

from __future__ import annotations

from dataclasses import dataclass

from story_engine.domain.request import ProjectRequest
from story_engine.domain.state import (
    AttachedTo,
    AttributeDefinition,
    EntityKind,
    EntityRule,
    HeldBy,
    InContainer,
    InSceneZone,
    Offscreen,
    OnSurface,
    Placement,
    Scalar,
    SceneRule,
    SetAttribute,
    SetPlacement,
    ShotBoundary,
    Transition,
    WorldReducer,
    WorldRules,
    WorldState,
)
from story_engine.domain.story import (
    AnchorLandmark,
    Beat,
    EntitySpec,
    PlanInvariant,
    PlanInvariantKind,
    ReferenceNeedKind,
    ReferenceNeedSpec,
    Requirement,
    RequirementCategory,
    RequirementKind,
    SceneRegion,
    SceneSpec,
    SemanticRelation,
    ShotSpatialIntent,
    SpatialContent,
    StoryPlan,
    StoryShot,
    ViewpointIntent,
    ZoneSpec,
)
from story_engine.errors import StoryCompilationError, WorldStateError
from story_engine.ids import canonical_hash, canonical_json, stable_key
from story_engine.planning.story_planner import (
    DraftBeat,
    DraftPlacement,
    DraftScene,
    DraftShot,
    DraftShotSpatialIntent,
    DraftTransition,
    StoryDraft,
)
from story_engine.spatial.continuity import validate_spatial_continuity


@dataclass(frozen=True, slots=True)
class StoryCompileConstraints:
    supported_durations: tuple[int, ...]
    max_beats_per_shot: int = 6
    max_visible_entities_per_shot: int = 8
    max_image_input_images: int = 16
    max_video_reference_images: int = 16


class StoryCompiler:
    def __init__(self, constraints: StoryCompileConstraints) -> None:
        if not constraints.supported_durations:
            raise ValueError("supported durations cannot be empty")
        self.constraints = constraints

    def compile(
        self,
        request: ProjectRequest,
        draft: StoryDraft,
        *,
        version: int = 2,
    ) -> StoryPlan:
        namespace = request.request_hash
        self._validate_aliases(draft)
        self._validate_placement_targets(draft)
        self._validate_transition_visibility(draft)
        self._validate_required_entities(request, draft)
        self._validate_asset_binding_contract(request, draft)
        if len(draft.shots) != request.shot_target:
            raise StoryCompilationError(
                f"expected exactly {request.shot_target} shots, got {len(draft.shots)}"
            )

        entity_keys = {
            entity.alias: stable_key("entity", namespace, entity.alias) for entity in draft.entities
        }
        scene_keys = {
            scene.alias: stable_key("scene", namespace, scene.alias) for scene in draft.scenes
        }
        zone_keys = {
            (scene.alias, zone.alias): stable_key("zone", scene_keys[scene.alias], zone.alias)
            for scene in draft.scenes
            for zone in scene.zones
        }
        anchor_keys = {
            (scene.alias, anchor.alias): stable_key("anchor", scene_keys[scene.alias], anchor.alias)
            for scene in draft.scenes
            for anchor in scene.anchor_landmarks
        }
        region_keys = {
            (scene.alias, region.alias): stable_key("region", scene_keys[scene.alias], region.alias)
            for scene in draft.scenes
            for region in scene.scene_regions
        }
        provided_bindings = self._provided_bindings(
            request,
            draft,
            entity_keys,
            scene_keys,
        )

        entities = tuple(
            EntitySpec(
                entity_key=entity_keys[entity.alias],
                alias=entity.alias,
                kind=entity.kind,
                visual_identity=entity.visual_identity,
                attribute_definitions=tuple(
                    AttributeDefinition(
                        key=attribute.key,
                        value_type=attribute.value_type,
                        allowed_values=tuple(attribute.allowed_values),
                        is_visual=attribute.is_visual,
                    )
                    for attribute in entity.attributes
                ),
                freeze_appearance=entity.freeze_appearance,
            )
            for entity in draft.entities
        )
        scenes = tuple(
            self._compile_scene(scene, scene_keys, zone_keys, region_keys, anchor_keys)
            for scene in draft.scenes
        )
        initial_world = self._compile_initial_world(draft, entity_keys, scene_keys, zone_keys)

        draft = self._split_scene_segments(
            draft,
            entity_keys,
            scene_keys,
            zone_keys,
            initial_world,
        )
        shot_keys = {
            shot.alias: stable_key("shot", namespace, f"{index}:{shot.alias}")
            for index, shot in enumerate(draft.shots)
        }

        requirements = self._compile_requirements(request, draft, namespace, shot_keys)
        plan_invariants = self._compile_plan_invariants(
            draft,
            namespace,
            shot_keys,
        )
        requirements_by_shot: dict[str, list[str]] = {
            shot_key: [
                requirement.requirement_key
                for requirement in requirements
                if requirement.category == RequirementCategory.SAFETY
            ]
            for shot_key in shot_keys.values()
        }
        for requirement in requirements:
            if requirement.category != RequirementCategory.SHOT_LOCAL:
                continue
            if requirement.owner_shot_key is None:
                raise StoryCompilationError("shot-local requirement has no owner")
            requirements_by_shot[requirement.owner_shot_key].append(requirement.requirement_key)

        shots: list[StoryShot] = []
        for shot in draft.shots:
            if shot.duration not in self.constraints.supported_durations:
                raise StoryCompilationError(
                    f"shot {shot.alias} duration {shot.duration} is unsupported"
                )
            if len(shot.beats) > self.constraints.max_beats_per_shot:
                raise StoryCompilationError(
                    f"shot {shot.alias} has too many beats; split it upstream"
                )
            if len(shot.visible_entity_aliases) > self.constraints.max_visible_entities_per_shot:
                raise StoryCompilationError(
                    f"shot {shot.alias} has too many visible entities; split it upstream"
                )
            scene_key = self._required(scene_keys, shot.scene_alias, "scene")
            visible = tuple(
                self._required(entity_keys, alias, "visible entity")
                for alias in shot.visible_entity_aliases
            )
            intent = shot.spatial_intent
            action_zone = self._required(
                {
                    alias: key
                    for (scene_alias, alias), key in zone_keys.items()
                    if scene_alias == shot.scene_alias
                },
                intent.action_zone_alias,
                "action zone",
            )
            story_targets = tuple(
                self._required(entity_keys, alias, "story target")
                for alias in intent.story_target_aliases
            )
            if not set(story_targets).issubset(visible):
                raise StoryCompilationError(
                    f"shot {shot.alias} story targets must be visible entities"
                )
            local_content = {
                alias: key
                for (scene_alias, alias), key in {**region_keys, **anchor_keys}.items()
                if scene_alias == shot.scene_alias
            }
            spatial_content = tuple(
                SpatialContent(
                    subject_key=self._required(
                        local_content,
                        item.subject_alias,
                        "spatial content",
                    ),
                    presentation=item.presentation,
                    strength=item.strength,
                )
                for item in intent.spatial_content
            )
            compiled_beats = tuple(
                Beat(
                    beat_key=stable_key(
                        "beat", shot_keys[shot.alias], f"{beat_index}:{beat.action}"
                    ),
                    action=beat.action,
                    transition=(
                        self._compile_transition(
                            beat.transition,
                            entity_keys,
                            scene_keys,
                            zone_keys,
                        )
                        if beat.transition
                        else None
                    ),
                )
                for beat_index, beat in enumerate(shot.beats)
            )
            shot_key = shot_keys[shot.alias]
            shots.append(
                StoryShot(
                    shot_key=shot_key,
                    alias=shot.alias,
                    scene_key=scene_key,
                    purpose=shot.purpose,
                    duration=shot.duration,
                    visible_entities=visible,
                    beats=compiled_beats,
                    spatial_intent=ShotSpatialIntent(
                        story_targets=story_targets,
                        action_zone=action_zone,
                        spatial_content=spatial_content,
                        viewpoint_intent=ViewpointIntent(
                            intent_key=stable_key(
                                "viewpoint_intent",
                                shot_key,
                                intent.viewpoint_intent.description,
                            ),
                            kind=intent.viewpoint_intent.kind,
                            description=intent.viewpoint_intent.description,
                            translation_expected=(intent.viewpoint_intent.translation_expected),
                        ),
                        axis_key=intent.axis_key,
                        camera_side=intent.camera_side,
                        screen_direction=intent.screen_direction,
                        entry_edge=intent.entry_edge,
                        exit_edge=intent.exit_edge,
                        framing=intent.framing,
                        framing_scale=intent.framing_scale,
                        camera_motion_intent=intent.camera_motion_intent,
                    ),
                    requirement_refs=tuple(sorted(requirements_by_shot[shot_key])),
                )
            )

        reference_needs = self._derive_reference_needs(
            request,
            entities,
            scenes,
            namespace,
            initial_world,
            tuple(shots),
            provided_bindings,
        )
        plan_key = stable_key("story_plan", namespace, f"v{version}")
        payload = {
            "plan_key": plan_key,
            "version": version,
            "request_ref": request.request_hash,
            "entity_catalog": entities,
            "scene_catalog": scenes,
            "initial_world": initial_world,
            "ordered_shots": tuple(shots),
            "requirements": requirements,
            "plan_invariants": plan_invariants,
            "reference_needs": reference_needs,
        }
        plan = StoryPlan(
            plan_key=plan_key,
            version=version,
            request_ref=request.request_hash,
            entity_catalog=entities,
            scene_catalog=scenes,
            initial_world=initial_world,
            ordered_shots=tuple(shots),
            requirements=requirements,
            plan_invariants=plan_invariants,
            reference_needs=reference_needs,
            plan_hash=canonical_hash(payload),
        )
        plan.assert_hash()
        continuity_issues = validate_spatial_continuity(plan)
        if continuity_issues:
            shot_aliases = {item.shot_key: item.alias for item in plan.ordered_shots}
            details = ", ".join(
                f"{issue.code} between "
                f"{shot_aliases[issue.previous_shot]} and {shot_aliases[issue.current_shot]}"
                for issue in continuity_issues
            )
            raise StoryCompilationError(f"spatial continuity conflicts: {details}")
        try:
            reducer = WorldReducer(plan.world_rules())
            boundaries = reducer.boundaries(
                plan.initial_world,
                tuple(
                    (
                        shot.shot_key,
                        tuple(
                            beat.transition for beat in shot.beats if beat.transition is not None
                        ),
                    )
                    for shot in plan.ordered_shots
                ),
            )
        except WorldStateError as exc:
            raise StoryCompilationError(str(exc)) from exc
        self._validate_provider_channels(plan, boundaries)
        return plan

    def _split_scene_segments(
        self,
        draft: StoryDraft,
        entity_keys: dict[str, str],
        scene_keys: dict[str, str],
        zone_keys: dict[tuple[str, str], str],
        initial_world: WorldState,
    ) -> StoryDraft:
        declared_scene_aliases = {scene.alias for scene in draft.scenes}
        declared_shot_aliases = {shot.alias for shot in draft.shots}
        zones_by_scene = {
            scene.alias: [zone.alias for zone in scene.zones] for scene in draft.scenes
        }
        zone_alias_by_key = {key: alias for (_scene, alias), key in zone_keys.items()}
        local_subjects = {
            scene.alias: {region.alias for region in scene.scene_regions}
            | {anchor.alias for anchor in scene.anchor_landmarks}
            for scene in draft.scenes
        }
        rules = WorldRules(
            entities=tuple(
                EntityRule(
                    entity_key=entity_keys[entity.alias],
                    kind=entity.kind,
                    attribute_definitions=tuple(
                        AttributeDefinition(
                            key=attribute.key,
                            value_type=attribute.value_type,
                            allowed_values=tuple(attribute.allowed_values),
                            is_visual=attribute.is_visual,
                        )
                        for attribute in entity.attributes
                    ),
                )
                for entity in draft.entities
            ),
            scenes=tuple(
                SceneRule(
                    scene_key=scene_keys[scene.alias],
                    zone_keys=tuple(zone_keys[(scene.alias, zone.alias)] for zone in scene.zones),
                )
                for scene in draft.scenes
            ),
        )
        reducer = WorldReducer(rules)
        state = initial_world
        new_shots: list[DraftShot] = []
        parts_by_parent: dict[str, list[str]] = {}
        for shot in draft.shots:
            if shot.duration not in self.constraints.supported_durations:
                raise StoryCompilationError(
                    f"shot {shot.alias} duration {shot.duration} is unsupported"
                )
            if len(shot.beats) > self.constraints.max_beats_per_shot:
                raise StoryCompilationError(
                    f"shot {shot.alias} has too many beats; split it upstream"
                )
            if len(shot.visible_entity_aliases) > self.constraints.max_visible_entities_per_shot:
                raise StoryCompilationError(
                    f"shot {shot.alias} has too many visible entities; split it upstream"
                )
            segments = self._segment_beats(shot, declared_scene_aliases)
            parts: list[DraftShot] = []
            if len(segments) == 1:
                parts.append(shot)
            else:
                for index, (scene_alias, beats) in enumerate(segments, start=1):
                    part = self._build_part_shot(
                        shot,
                        f"{shot.alias}_s{index}",
                        scene_alias,
                        beats,
                        state,
                        entity_keys,
                        scene_keys,
                        zones_by_scene,
                        zone_alias_by_key,
                        local_subjects,
                    )
                    if part.alias in declared_shot_aliases:
                        raise StoryCompilationError(
                            f"split part alias {part.alias!r} collides with a declared "
                            "shot alias; rename the declared shot"
                        )
                    if len(part.alias) > 80:
                        raise StoryCompilationError(
                            f"shot {shot.alias} alias is too long to split into scene "
                            "parts; shorten it to at most 77 characters"
                        )
                    parts.append(part)
                    state = self._advance_world_state(
                        state, part, reducer, entity_keys, scene_keys, zone_keys
                    )
            parts_by_parent[shot.alias] = [part.alias for part in parts]
            if len(segments) == 1:
                state = self._advance_world_state(
                    state, shot, reducer, entity_keys, scene_keys, zone_keys
                )
            new_shots.extend(parts)
        remapped_requirements = [
            item.model_copy(
                update={
                    "shot_alias": parts_by_parent.get(item.shot_alias, [item.shot_alias])[-1]
                }
            )
            for item in draft.requirements
        ]
        remapped_invariants = [
            item.model_copy(
                update={
                    "before_shot_alias": parts_by_parent.get(
                        item.before_shot_alias, [item.before_shot_alias]
                    )[0],
                    "after_shot_alias": parts_by_parent.get(
                        item.after_shot_alias, [item.after_shot_alias]
                    )[-1],
                }
            )
            for item in draft.plan_invariants
        ]
        return draft.model_copy(
            update={
                "shots": new_shots,
                "requirements": remapped_requirements,
                "plan_invariants": remapped_invariants,
            }
        )

    @staticmethod
    def _segment_beats(
        shot: DraftShot,
        declared_scene_aliases: set[str],
    ) -> list[tuple[str, list[DraftBeat]]]:
        segments: list[tuple[str, list[DraftBeat]]] = []
        current = shot.scene_alias
        segment_beats: list[DraftBeat] = []
        changes = 0
        for index, beat in enumerate(shot.beats):
            override = beat.scene_alias
            if override is not None:
                if override not in declared_scene_aliases:
                    raise StoryCompilationError(
                        f"shot {shot.alias} beat {index} scene override names undeclared "
                        f"scene {override!r}"
                    )
                if override == current:
                    raise StoryCompilationError(
                        f"shot {shot.alias} beat {index} scene override {override!r} "
                        "repeats the current scene; omit the override unless the scene "
                        "changes"
                    )
                changes += 1
                if changes > 2:
                    raise StoryCompilationError(
                        f"shot {shot.alias} has more than 2 mid-shot scene changes; "
                        "split the scene change into separate shots"
                    )
                segments.append((current, segment_beats))
                current = override
                segment_beats = []
            segment_beats.append(beat)
        segments.append((current, segment_beats))
        if not segments[0][1]:
            raise StoryCompilationError(
                f"shot {shot.alias} first beat changes scene; the shot's scene_alias is "
                "the opening scene, so the change must start on a later beat"
            )
        return segments

    def _build_part_shot(
        self,
        parent: DraftShot,
        part_alias: str,
        scene_alias: str,
        beats: list[DraftBeat],
        planned_start: WorldState,
        entity_keys: dict[str, str],
        scene_keys: dict[str, str],
        zones_by_scene: dict[str, list[str]],
        zone_alias_by_key: dict[str, str],
        local_subjects: dict[str, set[str]],
    ) -> DraftShot:
        scene_key = scene_keys[scene_alias]
        present_visible: list[str] = []
        for alias in parent.visible_entity_aliases:
            entity_key = entity_keys.get(alias)
            if entity_key is None:
                continue
            if self._placement_scene(planned_start, entity_key) == scene_key:
                present_visible.append(alias)
        beat_entities: set[str] = set()
        for beat in beats:
            transition = beat.transition
            if transition is None:
                continue
            beat_entities.add(transition.entity_alias)
            if (
                transition.kind == "set_placement"
                and transition.placement is not None
                and transition.placement.target_entity_alias is not None
            ):
                beat_entities.add(transition.placement.target_entity_alias)
        part_story_targets = [
            alias
            for alias in parent.spatial_intent.story_target_aliases
            if (entity_key := entity_keys.get(alias)) is not None
            and self._placement_scene(planned_start, entity_key) == scene_key
        ]
        if not part_story_targets:
            raise StoryCompilationError(
                f"shot {parent.alias} scene segment in {scene_alias!r} has no story "
                "target in that scene; list a story target that is present in every "
                "scene segment"
            )
        visible_aliases: list[str] = []
        for alias in [*present_visible, *part_story_targets]:
            if alias not in visible_aliases:
                visible_aliases.append(alias)
        for alias in sorted(beat_entities):
            if alias not in visible_aliases:
                visible_aliases.append(alias)
        zone_alias: str | None = None
        for alias in part_story_targets:
            try:
                placement = planned_start.placement_for(entity_keys[alias])
            except KeyError:
                continue
            if isinstance(placement, InSceneZone) and placement.scene_key == scene_key:
                zone_alias = zone_alias_by_key.get(placement.zone_key)
                if zone_alias is not None:
                    break
        if zone_alias is None:
            zones = zones_by_scene.get(scene_alias) or []
            if not zones:
                raise StoryCompilationError(
                    f"scene {scene_alias!r} has no zones; every scene needs at least "
                    "one zone"
                )
            zone_alias = zones[0]
        intent = parent.spatial_intent
        return DraftShot(
            alias=part_alias,
            scene_alias=scene_alias,
            purpose=parent.purpose,
            duration=parent.duration,
            visible_entity_aliases=visible_aliases,
            beats=[
                DraftBeat(action=beat.action, transition=beat.transition) for beat in beats
            ],
            spatial_intent=DraftShotSpatialIntent(
                story_target_aliases=part_story_targets,
                action_zone_alias=zone_alias,
                spatial_content=[
                    item
                    for item in intent.spatial_content
                    if item.subject_alias in local_subjects.get(scene_alias, set())
                ],
                viewpoint_intent=intent.viewpoint_intent,
                axis_key=intent.axis_key,
                camera_side=intent.camera_side,
                screen_direction=intent.screen_direction,
                entry_edge=intent.entry_edge,
                exit_edge=intent.exit_edge,
                framing=intent.framing,
                framing_scale=intent.framing_scale,
                camera_motion_intent=intent.camera_motion_intent,
            ),
        )

    def _advance_world_state(
        self,
        state: WorldState,
        shot: DraftShot,
        reducer: WorldReducer,
        entity_keys: dict[str, str],
        scene_keys: dict[str, str],
        zone_keys: dict[tuple[str, str], str],
    ) -> WorldState:
        for beat in shot.beats:
            if beat.transition is None:
                continue
            try:
                state = reducer.apply(
                    state,
                    self._compile_transition(
                        beat.transition, entity_keys, scene_keys, zone_keys
                    ),
                )
            except WorldStateError as exc:
                raise StoryCompilationError(str(exc)) from exc
        return state

    @staticmethod
    def _placement_scene(
        state: WorldState,
        entity_key: str,
        visited: frozenset[str] = frozenset(),
    ) -> str | None:
        if entity_key in visited:
            return None
        try:
            placement = state.placement_for(entity_key)
        except KeyError:
            return None
        if isinstance(placement, InSceneZone):
            return placement.scene_key
        if isinstance(placement, Offscreen):
            return None
        if isinstance(placement, OnSurface):
            target_key = placement.surface_entity
        elif isinstance(placement, InContainer):
            target_key = placement.container_entity
        elif isinstance(placement, HeldBy):
            target_key = placement.character_entity
        elif isinstance(placement, AttachedTo):
            target_key = placement.entity
        else:  # pragma: no cover - sealed by the type union
            return None
        return StoryCompiler._placement_scene(
            state, target_key, visited | {entity_key}
        )

    def _validate_provider_channels(
        self,
        plan: StoryPlan,
        boundaries: tuple[ShotBoundary, ...],
    ) -> None:
        entities = {item.entity_key: item for item in plan.entity_catalog}
        reducer = WorldReducer(plan.world_rules())
        for shot, boundary in zip(plan.ordered_shots, boundaries, strict=True):
            planned_start = boundary.planned_start
            start_visible = {
                entity_key
                for entity_key in shot.visible_entities
                if not isinstance(
                    planned_start.placement_for(entity_key),
                    (Offscreen, InContainer),
                )
            }
            frozen_start = {
                entity_key for entity_key in start_visible if entities[entity_key].freeze_appearance
            }
            image_inputs = 2 + sum(
                3 if entities[entity_key].kind == EntityKind.CHARACTER else 1
                for entity_key in frozen_start
            )
            if image_inputs > self.constraints.max_image_input_images:
                aliases = ", ".join(sorted(entities[key].alias for key in frozen_start))
                raise StoryCompilationError(
                    f"shot {shot.alias} needs {image_inputs} first-frame image inputs but "
                    f"provider supports {self.constraints.max_image_input_images}: {aliases}"
                )

            later_references: list[str] = []
            current = planned_start
            states = [current]
            for beat in shot.beats:
                if beat.transition is not None:
                    current = reducer.apply(current, beat.transition)
                    states.append(current)
            for entity_key in shot.visible_entities:
                entity = entities[entity_key]
                if not entity.freeze_appearance:
                    continue
                if entity_key not in start_visible:
                    later_references.append(f"{entity.alias} first appears after t0")
                visible_states = {self._visible_state_at(entity, state) for state in states}
                for visible_state in sorted(item for item in visible_states if item is not None)[
                    1:
                ]:
                    later_references.append(f"{entity.alias} state {visible_state}")
            if len(later_references) > self.constraints.max_video_reference_images:
                detail = "; ".join(later_references)
                raise StoryCompilationError(
                    f"shot {shot.alias} needs {len(later_references)} additional video "
                    f"references but provider supports "
                    f"{self.constraints.max_video_reference_images}; make the frozen "
                    f"entity/state visible at t0 or move the change across a cut: {detail}"
                )

    @staticmethod
    def _validate_aliases(draft: StoryDraft) -> None:
        groups = {
            "entity": [item.alias for item in draft.entities],
            "scene": [item.alias for item in draft.scenes],
            "shot": [item.alias for item in draft.shots],
            "requirement": [item.alias for item in draft.requirements],
            "plan invariant": [item.alias for item in draft.plan_invariants],
        }
        for name, aliases in groups.items():
            if not aliases:
                if name in {"entity", "scene", "shot"}:
                    raise StoryCompilationError(f"draft requires at least one {name}")
                continue
            if len(aliases) != len(set(aliases)):
                raise StoryCompilationError(f"duplicate {name} alias")
            if any(not alias.strip() or len(alias) > 80 for alias in aliases):
                raise StoryCompilationError(f"invalid {name} alias")
        for scene in draft.scenes:
            for label, aliases in {
                "zone": [zone.alias for zone in scene.zones],
                "region": [region.alias for region in scene.scene_regions],
                "anchor": [anchor.alias for anchor in scene.anchor_landmarks],
            }.items():
                if len(aliases) != len(set(aliases)):
                    raise StoryCompilationError(f"duplicate {label} alias in scene {scene.alias}")
            local_domains = (
                [zone.alias for zone in scene.zones]
                + [region.alias for region in scene.scene_regions]
                + [anchor.alias for anchor in scene.anchor_landmarks]
            )
            if len(local_domains) != len(set(local_domains)):
                raise StoryCompilationError(
                    f"zone, region, and anchor aliases must be disjoint in scene {scene.alias}"
                )

    @staticmethod
    def _validate_required_entities(
        request: ProjectRequest,
        draft: StoryDraft,
    ) -> None:
        aliases = {entity.alias for entity in draft.entities}
        for required in request.generation_requirements.required_entities:
            if required not in aliases:
                raise StoryCompilationError(
                    f"required entity must be used as an exact alias: {required!r}"
                )
            if not any(required in shot.visible_entity_aliases for shot in draft.shots):
                raise StoryCompilationError(
                    f"required entity is never visible in a shot: {required!r}"
                )

    @staticmethod
    def _validate_asset_binding_contract(
        request: ProjectRequest,
        draft: StoryDraft,
    ) -> None:
        entities = {item.alias: item for item in draft.entities}
        scenes = {item.alias: item for item in draft.scenes}
        for binding in request.provided_asset_bindings:
            if binding.kind == "character":
                entity = entities.get(binding.canonical_alias)
                if entity is None:
                    raise StoryCompilationError(
                        f"confirmed character asset requires exact alias: "
                        f"{binding.canonical_alias!r}"
                    )
                if entity.kind.value != "character":
                    raise StoryCompilationError(
                        f"confirmed character asset alias has wrong entity kind: "
                        f"{binding.canonical_alias!r}"
                    )
                if entity.provided_asset_id != binding.asset_id:
                    raise StoryCompilationError(
                        f"confirmed character alias must bind asset {binding.asset_id!r}: "
                        f"{binding.canonical_alias!r}"
                    )
                if not any(
                    binding.canonical_alias in shot.visible_entity_aliases
                    for shot in draft.shots
                ):
                    raise StoryCompilationError(
                        f"confirmed character asset is never visible: "
                        f"{binding.canonical_alias!r}"
                    )
            else:
                scene = scenes.get(binding.canonical_alias)
                if scene is None:
                    raise StoryCompilationError(
                        f"confirmed scene asset requires exact alias: "
                        f"{binding.canonical_alias!r}"
                    )
                if scene.provided_asset_id != binding.asset_id:
                    raise StoryCompilationError(
                        f"confirmed scene alias must bind asset {binding.asset_id!r}: "
                        f"{binding.canonical_alias!r}"
                    )
                if not StoryCompiler._scene_referenced(binding.canonical_alias, draft):
                    raise StoryCompilationError(
                        f"confirmed scene asset is never used: {binding.canonical_alias!r}"
                    )

    @staticmethod
    def _scene_referenced(scene_alias: str, draft: StoryDraft) -> bool:
        for shot in draft.shots:
            if shot.scene_alias == scene_alias:
                return True
            for beat in getattr(shot, "beats", ()):
                if getattr(beat, "scene_alias", None) == scene_alias:
                    return True
        return False

    @staticmethod
    def _validate_placement_targets(draft: StoryDraft) -> None:
        entities = {item.alias: item for item in draft.entities}
        placements = [
            (entity.alias, "initial placement", entity.initial_placement)
            for entity in draft.entities
        ]
        placements.extend(
            (
                transition.entity_alias,
                f"transition in shot {shot.alias}",
                transition.placement,
            )
            for shot in draft.shots
            for beat in shot.beats
            if (transition := beat.transition) is not None
            and transition.kind == "set_placement"
            and transition.placement is not None
        )
        required_kinds = {
            "on_surface": "surface",
            "in_container": "container",
            "held_by": "character",
        }
        for owner_alias, context, placement in placements:
            if placement.kind in {"in_scene_zone", "offscreen"}:
                continue
            target_alias = placement.target_entity_alias
            if target_alias is None:
                raise StoryCompilationError(
                    f"{context} for {owner_alias!r}: {placement.kind} needs target_entity_alias"
                )
            target = entities.get(target_alias)
            if target is None:
                raise StoryCompilationError(
                    f"{context} for {owner_alias!r} references unknown target entity "
                    f"{target_alias!r}"
                )
            if target_alias == owner_alias:
                raise StoryCompilationError(
                    f"{context} for {owner_alias!r} cannot target the same entity"
                )
            required_kind = required_kinds.get(placement.kind)
            if required_kind is not None and target.kind.value != required_kind:
                raise StoryCompilationError(
                    f"{context} for {owner_alias!r}: {placement.kind} target "
                    f"{target_alias!r} must be {required_kind}, got {target.kind.value}"
                )

    @staticmethod
    def _validate_transition_visibility(draft: StoryDraft) -> None:
        for shot in draft.shots:
            visible = set(shot.visible_entity_aliases)
            transitioned: set[str] = set()
            targets: set[str] = set()
            for beat in shot.beats:
                transition = beat.transition
                if transition is None:
                    continue
                transitioned.add(transition.entity_alias)
                if (
                    transition.kind == "set_placement"
                    and transition.placement is not None
                    and transition.placement.target_entity_alias is not None
                ):
                    targets.add(transition.placement.target_entity_alias)
            missing_transitioned = sorted(transitioned - visible)
            missing_targets = sorted(targets - visible)
            if not missing_transitioned and not missing_targets:
                continue
            details: list[str] = []
            if missing_transitioned:
                details.append("transitioned=" + ", ".join(missing_transitioned))
            if missing_targets:
                details.append("targets=" + ", ".join(missing_targets))
            raise StoryCompilationError(
                f"shot {shot.alias} beat transitions reference entities not listed in "
                f"visible_entity_aliases ({'; '.join(details)}); add them to this shot or "
                "move the transition to a shot where they are visible"
            )

    @staticmethod
    def _required(mapping: dict[str, str], alias: str, kind: str) -> str:
        try:
            return mapping[alias]
        except KeyError as exc:
            raise StoryCompilationError(f"unknown {kind} alias {alias!r}") from exc

    def _compile_scene(
        self,
        scene: DraftScene,
        scene_keys: dict[str, str],
        zone_keys: dict[tuple[str, str], str],
        region_keys: dict[tuple[str, str], str],
        anchor_keys: dict[tuple[str, str], str],
    ) -> SceneSpec:
        scene_key = scene_keys[scene.alias]
        local_spatial = {
            **{
                alias: key
                for (scene_alias, alias), key in zone_keys.items()
                if scene_alias == scene.alias
            },
            **{
                alias: key
                for (scene_alias, alias), key in region_keys.items()
                if scene_alias == scene.alias
            },
            **{
                alias: key
                for (scene_alias, alias), key in anchor_keys.items()
                if scene_alias == scene.alias
            },
        }
        local_zones = {
            alias: key
            for (scene_alias, alias), key in zone_keys.items()
            if scene_alias == scene.alias
        }
        local_anchors = {
            alias: key
            for (scene_alias, alias), key in anchor_keys.items()
            if scene_alias == scene.alias
        }
        return SceneSpec(
            scene_key=scene_key,
            alias=scene.alias,
            visual_identity=scene.visual_identity,
            zones=tuple(
                ZoneSpec(
                    zone_key=zone_keys[(scene.alias, zone.alias)],
                    alias=zone.alias,
                    description=zone.description,
                )
                for zone in scene.zones
            ),
            scene_regions=tuple(
                SceneRegion(
                    region_key=region_keys[(scene.alias, region.alias)],
                    alias=region.alias,
                    description=region.description,
                    zone_key=self._required(
                        local_zones,
                        region.zone_alias,
                        "region zone",
                    ),
                    representative_anchor_keys=tuple(
                        self._required(
                            local_anchors,
                            alias,
                            "region representative anchor",
                        )
                        for alias in region.representative_anchor_aliases
                    ),
                )
                for region in scene.scene_regions
            ),
            anchor_landmarks=tuple(
                AnchorLandmark(
                    anchor_key=anchor_keys[(scene.alias, anchor.alias)],
                    alias=anchor.alias,
                    description=anchor.description,
                    zone_key=self._required(
                        local_zones,
                        anchor.zone_alias,
                        "anchor zone",
                    ),
                )
                for anchor in scene.anchor_landmarks
            ),
            semantic_relations=tuple(
                SemanticRelation(
                    subject_key=self._required(
                        local_spatial, relation.subject_alias, "relation subject"
                    ),
                    relation=relation.relation,
                    object_key=self._required(
                        local_spatial, relation.object_alias, "relation object"
                    ),
                )
                for relation in scene.semantic_relations
            ),
            default_axis=scene.default_axis,
            lighting=scene.lighting,
            style=scene.style,
        )

    def _compile_initial_world(
        self,
        draft: StoryDraft,
        entity_keys: dict[str, str],
        scene_keys: dict[str, str],
        zone_keys: dict[tuple[str, str], str],
    ) -> WorldState:
        placements: dict[str, Placement] = {}
        attributes: dict[tuple[str, str], Scalar] = {}
        for entity in draft.entities:
            entity_key = entity_keys[entity.alias]
            placements[entity_key] = self._compile_placement(
                entity.initial_placement, entity_keys, scene_keys, zone_keys
            )
            for definition in entity.attributes:
                if definition.initial_value is None:
                    raise StoryCompilationError(
                        f"attribute {entity.alias}.{definition.key} needs an initial value"
                    )
                attributes[(entity_key, definition.key)] = definition.initial_value
        return WorldState.build(placements, attributes)

    def _compile_placement(
        self,
        placement: DraftPlacement,
        entity_keys: dict[str, str],
        scene_keys: dict[str, str],
        zone_keys: dict[tuple[str, str], str],
    ) -> Placement:
        if placement.kind == "in_scene_zone":
            if placement.scene_alias is None or placement.zone_alias is None:
                raise StoryCompilationError("in_scene_zone needs scene_alias and zone_alias")
            scene_key = self._required(scene_keys, placement.scene_alias, "placement scene")
            zone_key = zone_keys.get((placement.scene_alias, placement.zone_alias))
            if zone_key is None:
                raise StoryCompilationError(
                    f"unknown zone {placement.scene_alias}/{placement.zone_alias}"
                )
            return InSceneZone(scene_key=scene_key, zone_key=zone_key)
        if placement.kind == "offscreen":
            return Offscreen(
                scene_key=(
                    self._required(scene_keys, placement.scene_alias, "offscreen scene")
                    if placement.scene_alias
                    else None
                )
            )
        if placement.target_entity_alias is None:
            raise StoryCompilationError(f"{placement.kind} needs target_entity_alias")
        target = self._required(entity_keys, placement.target_entity_alias, "placement target")
        if placement.kind == "on_surface":
            return OnSurface(surface_entity=target)
        if placement.kind == "in_container":
            return InContainer(container_entity=target)
        if placement.kind == "held_by":
            return HeldBy(character_entity=target)
        if placement.kind == "attached_to":
            return AttachedTo(entity=target)
        raise StoryCompilationError(f"unsupported placement kind {placement.kind}")

    def _compile_transition(
        self,
        transition: DraftTransition,
        entity_keys: dict[str, str],
        scene_keys: dict[str, str],
        zone_keys: dict[tuple[str, str], str],
    ) -> Transition:
        entity_key = self._required(entity_keys, transition.entity_alias, "transition entity")
        if transition.kind == "set_placement":
            if transition.placement is None:
                raise StoryCompilationError("set_placement transition is incomplete")
            return SetPlacement(
                entity_key=entity_key,
                placement=self._compile_placement(
                    transition.placement, entity_keys, scene_keys, zone_keys
                ),
            )
        if transition.attribute_key is None or transition.value is None:
            raise StoryCompilationError("set_attribute transition is incomplete")
        return SetAttribute(
            entity_key=entity_key,
            attribute_key=transition.attribute_key,
            value=transition.value,
        )

    def _compile_requirements(
        self,
        request: ProjectRequest,
        draft: StoryDraft,
        namespace: str,
        shot_keys: dict[str, str],
    ) -> tuple[Requirement, ...]:
        requirements: list[Requirement] = []
        for item in draft.requirements:
            requirements.append(
                Requirement(
                    requirement_key=stable_key("requirement", namespace, item.alias),
                    kind=item.kind,
                    description=item.description,
                    priority=item.priority,
                    owner_shot_key=self._required(
                        shot_keys,
                        item.shot_alias,
                        "requirement shot",
                    ),
                    category=RequirementCategory.SHOT_LOCAL,
                )
            )
        for index, description in enumerate(request.generation_requirements.forbidden_content):
            requirements.append(
                Requirement(
                    requirement_key=stable_key(
                        "requirement", namespace, f"forbidden:{index}:{description}"
                    ),
                    kind=RequirementKind.REQUIREMENT,
                    description=f"Must not contain: {description}",
                    priority=100,
                    category=RequirementCategory.SAFETY,
                )
            )
        return tuple(sorted(requirements, key=lambda item: item.requirement_key))

    def _compile_plan_invariants(
        self,
        draft: StoryDraft,
        namespace: str,
        shot_keys: dict[str, str],
    ) -> tuple[PlanInvariant, ...]:
        shot_positions = {shot.alias: index for index, shot in enumerate(draft.shots)}
        invariants: list[PlanInvariant] = []
        for item in draft.plan_invariants:
            before = self._required(
                shot_keys,
                item.before_shot_alias,
                "plan invariant before shot",
            )
            after = self._required(
                shot_keys,
                item.after_shot_alias,
                "plan invariant after shot",
            )
            if shot_positions[item.before_shot_alias] >= shot_positions[item.after_shot_alias]:
                raise StoryCompilationError(
                    f"shot-order invariant {item.alias!r} contradicts canonical shot order: "
                    f"{item.before_shot_alias!r} must precede {item.after_shot_alias!r}"
                )
            invariants.append(
                PlanInvariant(
                    invariant_key=stable_key(
                        "plan_invariant",
                        namespace,
                        item.alias,
                    ),
                    kind=PlanInvariantKind.SHOT_ORDER,
                    description=item.description,
                    before_shot_key=before,
                    after_shot_key=after,
                )
            )
        return tuple(sorted(invariants, key=lambda item: item.invariant_key))

    @staticmethod
    def _derive_reference_needs(
        request: ProjectRequest,
        entities: tuple[EntitySpec, ...],
        scenes: tuple[SceneSpec, ...],
        namespace: str,
        initial_world: WorldState,
        shots: tuple[StoryShot, ...],
        provided_bindings: dict[str, str],
    ) -> tuple[ReferenceNeedSpec, ...]:
        needs: list[ReferenceNeedSpec] = []
        visible_states = StoryCompiler._visible_prop_states(
            entities,
            initial_world,
            shots,
        )
        for entity in entities:
            if not entity.freeze_appearance:
                continue
            if entity.kind.value == "character":
                kind = ReferenceNeedKind.CHARACTER
            elif entity.kind.value in {
                "prop",
                "container",
                "surface",
                "landmark",
            }:
                states = visible_states.get(entity.entity_key, ())
                if states:
                    initial_state = StoryCompiler._visible_state_at(
                        entity,
                        initial_world,
                    )
                    for state in states:
                        needs.append(
                            ReferenceNeedSpec(
                                need_key=stable_key(
                                    "reference_need",
                                    namespace,
                                    f"{entity.entity_key}:state:{state}",
                                ),
                                kind=ReferenceNeedKind.PROP_STATE,
                                subject_key=entity.entity_key,
                                visible_state=state,
                                provided_asset_id=(
                                    provided_bindings.get(entity.entity_key)
                                    if state == initial_state
                                    else None
                                ),
                            )
                        )
                    continue
                kind = ReferenceNeedKind.PROP
            else:
                continue
            needs.append(
                ReferenceNeedSpec(
                    need_key=stable_key("reference_need", namespace, entity.entity_key),
                    kind=kind,
                    subject_key=entity.entity_key,
                    provided_asset_id=provided_bindings.get(entity.entity_key),
                )
            )
        for scene in scenes:
            needs.append(
                ReferenceNeedSpec(
                    need_key=stable_key("reference_need", namespace, f"panorama:{scene.scene_key}"),
                    kind=ReferenceNeedKind.SCENE_PANORAMA,
                    subject_key=scene.scene_key,
                    provided_asset_id=provided_bindings.get(scene.scene_key),
                )
            )
        bound_asset_ids = set(provided_bindings.values())
        for asset in request.provided_assets:
            if asset.asset_id in bound_asset_ids:
                continue
            needs.append(
                ReferenceNeedSpec(
                    need_key=stable_key("reference_need", namespace, f"provided:{asset.asset_id}"),
                    kind=ReferenceNeedKind.PROVIDED,
                    subject_key=f"provided:{asset.asset_id}",
                    provided_asset_id=asset.asset_id,
                )
            )
        return tuple(sorted(needs, key=lambda item: item.need_key))

    @staticmethod
    def _provided_bindings(
        request: ProjectRequest,
        draft: StoryDraft,
        entity_keys: dict[str, str],
        scene_keys: dict[str, str],
    ) -> dict[str, str]:
        assets = {asset.asset_id: asset for asset in request.provided_assets}
        result: dict[str, str] = {}
        claimed: set[str] = set()
        for entity in draft.entities:
            asset_id = entity.provided_asset_id
            if asset_id is None:
                continue
            asset = assets.get(asset_id)
            if asset is None:
                raise StoryCompilationError(
                    f"entity {entity.alias} binds unknown provided asset {asset_id!r}"
                )
            allowed = (
                {"character", "image"} if entity.kind.value == "character" else {"prop", "image"}
            )
            if asset.kind not in allowed:
                raise StoryCompilationError(
                    f"provided asset {asset_id!r} cannot bind entity {entity.alias}"
                )
            if not entity.freeze_appearance:
                raise StoryCompilationError(
                    f"entity {entity.alias} cannot bind an identity asset "
                    "when freeze_appearance is false"
                )
            if asset_id in claimed:
                raise StoryCompilationError(f"provided asset {asset_id!r} is bound more than once")
            claimed.add(asset_id)
            result[entity_keys[entity.alias]] = asset_id
        for scene in draft.scenes:
            asset_id = scene.provided_asset_id
            if asset_id is None:
                continue
            asset = assets.get(asset_id)
            if asset is None:
                raise StoryCompilationError(
                    f"scene {scene.alias} binds unknown provided asset {asset_id!r}"
                )
            if asset.kind not in {"scene", "panorama", "image"}:
                raise StoryCompilationError(
                    f"provided asset {asset_id!r} cannot bind scene {scene.alias}"
                )
            if asset_id in claimed:
                raise StoryCompilationError(f"provided asset {asset_id!r} is bound more than once")
            claimed.add(asset_id)
            result[scene_keys[scene.alias]] = asset_id
        return result

    @staticmethod
    def _visible_state_at(
        entity: EntitySpec,
        state: WorldState,
    ) -> str | None:
        visual_keys = {
            definition.key for definition in entity.attribute_definitions if definition.is_visual
        }
        facts = tuple(
            (item.attribute_key, item.value)
            for item in state.attributes_by_entity
            if item.entity_key == entity.entity_key and item.attribute_key in visual_keys
        )
        if not facts:
            return None
        return "; ".join(
            f"{attribute_key}={canonical_json(value)}" for attribute_key, value in facts
        )

    @staticmethod
    def _visible_prop_states(
        entities: tuple[EntitySpec, ...],
        initial_world: WorldState,
        shots: tuple[StoryShot, ...],
    ) -> dict[str, tuple[str, ...]]:
        tracked = {
            entity.entity_key: tuple(
                definition.key
                for definition in entity.attribute_definitions
                if definition.is_visual
            )
            for entity in entities
            if entity.kind.value in {"prop", "container", "surface", "landmark"}
            and any(definition.is_visual for definition in entity.attribute_definitions)
        }
        attributes = {
            (fact.entity_key, fact.attribute_key): fact.value
            for fact in initial_world.attributes_by_entity
        }
        states: dict[str, set[str]] = {entity_key: set() for entity_key in tracked}

        def remember() -> None:
            for entity_key, visual_keys in tracked.items():
                facts = tuple(
                    (attribute_key, value)
                    for (owner, attribute_key), value in sorted(attributes.items())
                    if owner == entity_key and attribute_key in visual_keys
                )
                if facts:
                    states[entity_key].add(
                        "; ".join(f"{key}={canonical_json(value)}" for key, value in facts)
                    )

        remember()
        for shot in shots:
            for beat in shot.beats:
                if isinstance(beat.transition, SetAttribute):
                    attributes[
                        (
                            beat.transition.entity_key,
                            beat.transition.attribute_key,
                        )
                    ] = beat.transition.value
                    remember()
        return {
            entity_key: tuple(sorted(values)) for entity_key, values in states.items() if values
        }
