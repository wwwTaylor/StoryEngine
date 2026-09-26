"""Deterministically project canonical plans into operation-local views."""

from __future__ import annotations

from story_engine.domain.evaluation import CriterionKind
from story_engine.domain.render import (
    AdditionalReference,
    RenderPlan,
    RenderShot,
    RequirementPhase,
    SemanticOnly,
    VisibleInStartFrame,
)
from story_engine.domain.request import ProjectRequest
from story_engine.domain.story import StoryPlan, StoryShot
from story_engine.errors import ContractError
from story_engine.planning.prompt_views import (
    EvaluationView,
    FirstFrameParticipationView,
    FirstFrameView,
    NovelStationView,
    ParticipationSubjectView,
    StoryPlanningView,
    SubjectReference,
    SubjectReferenceImage,
    VideoMotionView,
    VisibleSubject,
)
from story_engine.planning.story_compiler import StoryCompileConstraints
from story_engine.providers.ports import RuntimeCapabilities
from story_engine.spatial.grounding import (
    GroundingTarget,
    GroundingTargetKind,
    GroundingView,
    SceneAnchorMap,
)
from story_engine.spatial.probes import ProbeSet
from story_engine.storage import ArtifactRef


class PromptViewBuilder:
    def story_planning(
        self,
        request: ProjectRequest,
        capabilities: RuntimeCapabilities,
        constraints: StoryCompileConstraints,
    ) -> StoryPlanningView:
        video = capabilities.video.video
        image = capabilities.image.image
        if video is None or image is None:
            raise ContractError("image and video capabilities are required for story planning")
        requested_durations = (
            request.generation_requirements.delivery_requirements.allowed_video_duration_seconds
        )
        supported_durations = constraints.supported_durations
        if not supported_durations:
            raise ContractError("request and VideoProvider have no common shot duration")
        unsupported_durations = tuple(
            value
            for value in supported_durations
            if value not in video.supported_durations or value not in requested_durations
        )
        if unsupported_durations:
            raise ContractError(
                "story compile durations exceed request or VideoProvider capabilities"
            )
        if constraints.max_image_input_images > image.max_input_images:
            raise ContractError("story compile image-input limit exceeds provider capability")
        if constraints.max_video_reference_images > video.max_reference_images:
            raise ContractError("story compile video-reference limit exceeds provider capability")
        return StoryPlanningView(
            idea=request.idea,
            shot_target=request.shot_target,
            visual_style=request.visual_style,
            output_language=request.generation_requirements.output_language,
            required_entities=request.generation_requirements.required_entities,
            forbidden_content=request.generation_requirements.forbidden_content,
            supported_durations=supported_durations,
            resolution=request.resolution,
            provided_asset_summaries=tuple(
                f"{asset.asset_id} ({asset.kind}; display name={asset.name or asset.asset_id}): "
                f"{asset.description}"
                for asset in request.provided_assets
            ),
            provided_asset_binding_summaries=tuple(
                f"asset_id={binding.asset_id}; kind={binding.kind}; exact alias="
                f"{binding.canonical_alias}; semantic role={binding.semantic_role}; idea mentions="
                f"{', '.join(binding.idea_mentions) or 'none'}"
                for binding in request.provided_asset_bindings
            ),
            image_input_limit=constraints.max_image_input_images,
            video_reference_limit=constraints.max_video_reference_images,
            max_beats_per_shot=constraints.max_beats_per_shot,
            max_visible_entities_per_shot=constraints.max_visible_entities_per_shot,
        )

    def first_frame(
        self,
        request: ProjectRequest,
        story_plan: StoryPlan,
        render_plan: RenderPlan,
        *,
        shot_key: str,
    ) -> FirstFrameView:
        story_shot, render_shot = self._shots(story_plan, render_plan, shot_key)
        entities = {item.entity_key: item for item in story_plan.entity_catalog}
        scene = next(
            item for item in story_plan.scene_catalog if item.scene_key == story_shot.scene_key
        )
        visible_subjects: list[VisibleSubject] = []
        references: list[SubjectReference] = []
        for entity_key in render_shot.start_visible_entities:
            entity = entities[entity_key]
            binding = next(
                item
                for item in render_shot.input_bindings
                if item.entity_key == entity_key
                and isinstance(item, (VisibleInStartFrame, SemanticOnly))
            )
            description = binding.description if isinstance(binding, SemanticOnly) else None
            if isinstance(binding, VisibleInStartFrame):
                references.append(
                    SubjectReference(
                        alias=entity.alias,
                        images=self._binding_reference_images(binding),
                    )
                )
            facts = tuple(
                criterion.statement
                for criterion in render_shot.resolved_requirements
                if criterion.phase == RequirementPhase.FIRST_FRAME
                and criterion.statement.startswith(entity.alias + " ")
            )
            visible_subjects.append(
                VisibleSubject(
                    alias=entity.alias,
                    t0_facts=facts,
                    semantic_description=description,
                )
            )
        return FirstFrameView(
            scene_view=render_shot.selected_scene_view,
            scene_panorama=render_shot.selected_scene_panorama,
            subject_references=tuple(references),
            scene_description=scene.visual_identity,
            visible_subjects=tuple(visible_subjects),
            ordered_beats=tuple(beat.action for beat in story_shot.beats),
            shot_constraints=tuple(
                criterion.statement
                for criterion in render_shot.resolved_requirements
                if criterion.category == "shot_local"
            ),
            composition=story_shot.spatial_intent.framing,
            camera="Preserve the selected scene view as the exact opening viewpoint",
            visual_style=request.visual_style,
            avoid=request.generation_requirements.forbidden_content,
        )

    def first_frame_participation(
        self,
        story_plan: StoryPlan,
        render_plan: RenderPlan,
        *,
        shot_key: str,
        previous_tail: ArtifactRef,
    ) -> FirstFrameParticipationView:
        current_index = next(
            (
                index
                for index, shot in enumerate(story_plan.ordered_shots)
                if shot.shot_key == shot_key
            ),
            None,
        )
        if current_index is None:
            raise ContractError(f"unknown story shot {shot_key}")
        if current_index == 0:
            raise ContractError("the first shot has no previous tail to assess")
        current_shot, current_render = self._shots(story_plan, render_plan, shot_key)
        previous_shot = story_plan.ordered_shots[current_index - 1]
        previous_render = render_plan.ordered_render_shots[current_index - 1]
        if previous_render.shot_key != previous_shot.shot_key:
            raise ContractError("story and render shot order differ")
        scene = next(
            item for item in story_plan.scene_catalog if item.scene_key == current_shot.scene_key
        )
        entities = {item.entity_key: item for item in story_plan.entity_catalog}
        subjects: list[ParticipationSubjectView] = []
        for entity_key in current_render.start_visible_entities:
            binding = next(
                item
                for item in current_render.input_bindings
                if item.entity_key == entity_key
                and isinstance(item, (VisibleInStartFrame, SemanticOnly))
            )
            subjects.append(
                ParticipationSubjectView(
                    entity_key=entity_key,
                    alias=entities[entity_key].alias,
                    reference_images=(
                        self._binding_reference_images(binding)
                        if isinstance(binding, VisibleInStartFrame)
                        else ()
                    ),
                )
            )
        criteria = self.evaluation(
            render_plan,
            shot_key=shot_key,
            operation="first_frame",
            phases=(RequirementPhase.FIRST_FRAME,),
        )
        return FirstFrameParticipationView(
            previous_tail=previous_tail,
            previous_scene_key=previous_shot.scene_key,
            current_scene_key=current_shot.scene_key,
            current_scene_view=current_render.selected_scene_view,
            current_scene_description=scene.visual_identity,
            previous_subjects=tuple(
                ParticipationSubjectView(
                    entity_key=entity_key,
                    alias=entities[entity_key].alias,
                )
                for entity_key in previous_render.end_visible_entities
            ),
            current_subjects=tuple(subjects),
            current_criteria=tuple(
                item for item in criteria.criteria if item.kind == CriterionKind.REQUIREMENT
            ),
        )

    def grounding(
        self,
        story_plan: StoryPlan,
        probe_set: ProbeSet,
        *,
        station_key: str,
    ) -> GroundingView:
        scene = next(
            (item for item in story_plan.scene_catalog if item.scene_key == probe_set.scene_key),
            None,
        )
        if scene is None:
            raise ContractError(f"unknown probe scene {probe_set.scene_key}")
        targets = tuple(
            [
                GroundingTarget(
                    alias=anchor.alias,
                    target_key=anchor.anchor_key,
                    kind=GroundingTargetKind.ANCHOR,
                )
                for anchor in scene.anchor_landmarks
            ]
            + [
                GroundingTarget(
                    alias=region.alias,
                    target_key=region.region_key,
                    kind=GroundingTargetKind.REGION,
                )
                for region in scene.scene_regions
            ]
        )
        return GroundingView(
            scene_key=scene.scene_key,
            scene_alias=scene.alias,
            station_key=station_key,
            targets=targets,
            probe_set=probe_set,
        )

    def video_motion(
        self,
        request: ProjectRequest,
        story_plan: StoryPlan,
        render_plan: RenderPlan,
        *,
        shot_key: str,
        start_image: ArtifactRef,
    ) -> VideoMotionView:
        story_shot, render_shot = self._shots(story_plan, render_plan, shot_key)
        references = tuple(
            binding.reference_asset
            for binding in render_shot.input_bindings
            if isinstance(binding, AdditionalReference)
        )
        end_facts = tuple(
            item.statement
            for item in render_shot.resolved_requirements
            if item.phase == RequirementPhase.END
        )
        scene = next(
            item for item in story_plan.scene_catalog if item.scene_key == story_shot.scene_key
        )
        spatial_aliases = {item.anchor_key: item.alias for item in scene.anchor_landmarks} | {
            item.region_key: item.alias for item in scene.scene_regions
        }
        return VideoMotionView(
            start_image=start_image,
            reference_images=references,
            ordered_beats=tuple(beat.action for beat in story_shot.beats),
            camera_behavior=render_shot.camera_motion,
            reveal_content=tuple(
                spatial_aliases.get(item, item) for item in render_shot.reveal_anchors
            ),
            end_facts=end_facts,
            avoid=request.generation_requirements.forbidden_content,
            duration=story_shot.duration,
            resolution=request.resolution,
            fps=request.generation_requirements.delivery_requirements.video_fps,
        )

    @staticmethod
    def _binding_reference_images(
        binding: VisibleInStartFrame,
    ) -> tuple[SubjectReferenceImage, ...]:
        if binding.reference_views:
            return tuple(
                SubjectReferenceImage(
                    role=item.role,
                    artifact_ref=item.artifact_ref,
                )
                for item in binding.reference_views
            )
        return (
            SubjectReferenceImage(
                role="primary",
                artifact_ref=binding.reference_asset,
            ),
        )

    def novel_station(
        self,
        request: ProjectRequest,
        story_plan: StoryPlan,
        probe_set: ProbeSet,
        grounding: SceneAnchorMap,
        *,
        shot_key: str,
        panorama: ArtifactRef,
    ) -> NovelStationView:
        shot = next(
            (item for item in story_plan.ordered_shots if item.shot_key == shot_key),
            None,
        )
        if shot is None:
            raise ContractError(f"unknown story shot {shot_key}")
        scene = next(item for item in story_plan.scene_catalog if item.scene_key == shot.scene_key)
        roles = tuple(dict.fromkeys(item.probe_role for item in grounding.observations))
        spatial_aliases = (
            {item.zone_key: item.alias for item in scene.zones}
            | {item.region_key: item.alias for item in scene.scene_regions}
            | {item.anchor_key: item.alias for item in scene.anchor_landmarks}
        )
        return NovelStationView(
            panorama=panorama,
            probes=tuple(probe_set.by_role(role).artifact_ref for role in roles),
            scene_description=scene.visual_identity,
            station_description=shot.spatial_intent.viewpoint_intent.description,
            composition=shot.spatial_intent.framing,
            landmarks=tuple(
                f"unique anchor {item.alias}: {item.description}" for item in scene.anchor_landmarks
            ),
            required_content=tuple(
                spatial_aliases[item.subject_key]
                for item in shot.spatial_intent.spatial_content
                if item.strength.value == "must"
            ),
            semantic_layout=tuple(
                f"{spatial_aliases[item.subject_key]} {item.relation.value} "
                f"{spatial_aliases[item.object_key]}"
                for item in scene.semantic_relations
            ),
            visual_style=request.visual_style,
            avoid=request.generation_requirements.forbidden_content,
        )

    def evaluation(
        self,
        render_plan: RenderPlan,
        *,
        shot_key: str,
        operation: str,
        phases: tuple[RequirementPhase, ...],
    ) -> EvaluationView:
        render_shot = next(
            (item for item in render_plan.ordered_render_shots if item.shot_key == shot_key),
            None,
        )
        if render_shot is None:
            raise ContractError(f"unknown render shot {shot_key}")
        return EvaluationView.from_resolved(
            operation,
            tuple(
                item
                for item in render_shot.resolved_requirements
                if item.phase in phases
                or (
                    item.phase == RequirementPhase.ALWAYS
                    and (operation != "first_frame" or item.category == "safety")
                )
            ),
            owner_shot_key=shot_key,
        )

    @staticmethod
    def _shots(
        story_plan: StoryPlan, render_plan: RenderPlan, shot_key: str
    ) -> tuple[StoryShot, RenderShot]:
        story_shot = next(
            (item for item in story_plan.ordered_shots if item.shot_key == shot_key),
            None,
        )
        render_shot = next(
            (item for item in render_plan.ordered_render_shots if item.shot_key == shot_key),
            None,
        )
        if story_shot is None or render_shot is None:
            raise ContractError(f"shot does not exist in both plans: {shot_key}")
        return story_shot, render_shot
