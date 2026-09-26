"""Deterministic fixed workflow; agents never call one another."""

from __future__ import annotations

import asyncio
import math
import mimetypes
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from pydantic import BaseModel

from story_engine.agents.asset import AssetAgent
from story_engine.agents.evaluation import EvaluationAgent
from story_engine.agents.execution import ExecutionAgent, GroundingAttemptEvent
from story_engine.config import AppConfig, SpatialFailurePolicy
from story_engine.domain.evaluation import (
    CandidateMedia,
    CandidateRecord,
    CriterionKind,
    CriterionStatus,
    EvaluationReport,
    ProviderUsage,
    SelectionDecision,
    SelectionOutcome,
    TechnicalFinding,
    TechnicalStatus,
)
from story_engine.domain.first_frame import (
    FirstFrameGuidance,
    FirstFrameMode,
    FirstFrameParticipationResult,
    TailFrameInput,
    TailFrameStatus,
)
from story_engine.domain.manifest import (
    CriterionSummary,
    DeliveryManifest,
    DeliveryStatus,
    FinalMediaRecord,
    ProvenanceRecord,
    ProviderRecord,
    ShotDeliveryRecord,
    UsageRecord,
)
from story_engine.domain.reference import (
    CHARACTER_VIEW_ROLES,
    AnchorMapEvidence,
    CharacterReferenceRecipe,
    CharacterViewRole,
    GroundingUnavailableEvidence,
    GuidedCharacterReferenceRecipe,
    GuidedScenePanoramaRecipe,
    ProvidedReferenceRecipe,
    ReferenceLibrary,
    ReferenceRecipe,
    ScenePanoramaRecipe,
    SceneSpatialReference,
    SourceStationSpec,
    StationSpatialEvidence,
    StationSpatialReference,
)
from story_engine.domain.render import RenderPlan, RequirementPhase
from story_engine.domain.request import PixelSize, ProjectRequest
from story_engine.domain.spatial import (
    ResolvedSpatialPlan,
    SpatialAuditEntry,
    SpatialResolutionStatus,
)
from story_engine.domain.story import StoryPlan
from story_engine.domain.trace import (
    PromptAttachment,
    PromptPurpose,
    ProviderPrompt,
)
from story_engine.errors import (
    ArtifactError,
    CandidatePreflightExhausted,
    ContractError,
    GroundingValidationExhausted,
    PlanComplexityError,
    ProviderError,
    ReferenceError,
    SpatialError,
    StoryEngineError,
    WorkflowError,
)
from story_engine.ids import canonical_bytes, canonical_hash, stable_key
from story_engine.media.assemble import AssemblyResult, VideoAssembler
from story_engine.media.tail import VideoTailExtractor
from story_engine.media.validate import ImageTechnicalValidator, VideoTechnicalValidator
from story_engine.planning.participation import (
    has_preserve_guidance,
    reference_guidance,
    resolve_first_frame_participation,
)
from story_engine.planning.prompt_views import EvaluationCriterionView, EvaluationView
from story_engine.planning.render_compiler import RenderCompiler
from story_engine.planning.revision import IssueLedger
from story_engine.prompts.image import (
    render_character_reference,
    render_first_frame,
    render_novel_station,
    render_reference_first_frame,
    render_reference_image,
)
from story_engine.prompts.video import render_video
from story_engine.providers.ports import (
    Attachment,
    ImageRequest,
    ProviderCallMetrics,
    ResponseContract,
    RuntimeCapabilities,
    VideoRequest,
)
from story_engine.providers.registry import ProviderSet
from story_engine.run_state import (
    AttemptRecord,
    ProjectStatus,
    RunState,
    RunStore,
    ShotRunState,
    ShotStatus,
    StepRecord,
    StepStatus,
    step_key,
    utc_now,
)
from story_engine.spatial.camera import CameraConflictKind, CameraConstraints, CameraSolver
from story_engine.spatial.grounding import SceneAnchorMap
from story_engine.spatial.panorama import PanoramaSource, validate_panorama_source
from story_engine.spatial.probes import PanoramaProjector, ProbeSet
from story_engine.spatial.resolver import SpatialResolver
from story_engine.stage_output import StageOutputPublisher
from story_engine.storage import ArtifactRef
from story_engine.task_pool import TaskPool
from story_engine.version import (
    CAMERA_SOLVER_VERSION,
    FOV_SOLVER_VERSION,
    GROUNDING_VALIDATOR_VERSION,
    MANIFEST_VERSION,
    PANORAMA_PROJECTOR_VERSION,
    PROMPT_RENDERER_VERSION,
    REFERENCE_COMPILER_VERSION,
    RENDER_COMPILER_VERSION,
    SPATIAL_RESOLVER_VERSION,
    STORY_COMPILER_VERSION,
    WORKFLOW_VERSION,
)
from story_engine.workflow_records import (
    CandidatePreflightFailure,
    FirstFrameOperationResult,
    MediaOperationProgress,
    MediaOperationResult,
    PanoramaPreflightOutcome,
    PanoramaPreflightResult,
    PanoramaPreflightValue,
    ReferenceOperationResult,
    SceneAnchorOperationResult,
    SpatialOperationResult,
    SpatialPreflightShotResult,
)

ModelT = TypeVar("ModelT", bound=BaseModel)
ImageRequestFactory = Callable[[tuple[str, ...]], ImageRequest]
ImageCandidateAcceptance = Callable[[CandidateRecord], Awaitable[PanoramaPreflightValue]]


class Workflow:
    """Run the one allowed phase order with content-addressed resume."""

    def __init__(
        self,
        *,
        request: ProjectRequest,
        config: AppConfig,
        run_store: RunStore,
        providers: ProviderSet,
    ) -> None:
        self.request = request
        self.config = config
        self.run_store = run_store
        self.providers = providers
        self.capabilities: RuntimeCapabilities = providers.capabilities
        self.task_pool = TaskPool(config.generation.max_concurrency)
        self.execution = ExecutionAgent(
            planner=providers.planner,
            capabilities=self.capabilities,
            generation_policy=config.generation,
            task_pool=self.task_pool,
            render_compiler=RenderCompiler(run_store.artifacts),
        )
        self.assets = AssetAgent(
            image_provider=providers.image,
            video_provider=providers.video,
            store=run_store.artifacts,
            task_pool=self.task_pool,
        )
        self.evaluation = EvaluationAgent(providers.judge, self.task_pool)
        self.projector = PanoramaProjector(run_store.artifacts)
        self.camera_solver = CameraSolver(
            self.projector,
            CameraConstraints(
                output_width=request.resolution.width,
                output_height=request.resolution.height,
                max_hfov_degrees=config.generation.max_hfov_degrees,
            ),
        )
        video_capability = self.capabilities.video.video
        self.spatial_resolver = SpatialResolver(
            self.camera_solver,
            failure_policy=config.generation.spatial_failure_policy,
            fallback_hfov_min=config.generation.safe_fallback_hfov_min,
            fallback_hfov_max=config.generation.safe_fallback_hfov_max,
            supports_camera_motion=(
                video_capability.supports_camera_motion if video_capability is not None else False
            ),
            repair_attempts=config.generation.spatial_repair_attempts,
        )
        self.strict_spatial_resolver = SpatialResolver(
            self.camera_solver,
            failure_policy=SpatialFailurePolicy.STRICT,
            fallback_hfov_min=config.generation.safe_fallback_hfov_min,
            fallback_hfov_max=config.generation.safe_fallback_hfov_max,
            supports_camera_motion=(
                video_capability.supports_camera_motion if video_capability is not None else False
            ),
            repair_attempts=config.generation.spatial_repair_attempts,
        )
        self.assembler = VideoAssembler()
        self.tail_extractor = VideoTailExtractor()
        self.stage_output = StageOutputPublisher(run_store)
        self._state = run_store.load()
        self._started = time.monotonic()

    async def execute(self) -> RunState:
        try:
            return await self._execute()
        finally:
            await self.providers.aclose()

    async def _execute(self) -> RunState:
        self.stage_output.publish_input(self.request, self.config.redacted_dict())
        self.stage_output.publish_audit(self._state)
        if self._state.status in {
            ProjectStatus.DELIVERED,
            ProjectStatus.DELIVERED_DEGRADED,
        }:
            self._verify_delivered()
            return self._state
        if self._state.status == ProjectStatus.PROCESS_FAILED:
            raise WorkflowError("process_failed runs are terminal; start a new run")
        try:
            self._resume_interrupted()
            await self._preflight()
            story_plan = await self._story_plan()
            references, reference_results = await self._references(story_plan)
            references, spatial = await self._spatial(story_plan, references)
            render_plan = await self._render_plan(story_plan, references, spatial)
            frame_results, video_results = await self._render_shots(
                story_plan,
                render_plan,
            )
            assembly = await self._assemble(story_plan, video_results)
            await self._deliver(
                story_plan,
                references,
                render_plan,
                reference_results,
                frame_results,
                video_results,
                assembly,
            )
            return self._state
        except ProviderError as exc:
            if exc.retryable:
                self._interrupt(str(exc))
            else:
                self._fail(str(exc))
            raise
        except asyncio.CancelledError:
            self._interrupt("workflow cancelled")
            raise
        except StoryEngineError as exc:
            self._fail(str(exc))
            raise
        except Exception as exc:
            self._fail(f"unexpected workflow failure: {type(exc).__name__}: {exc}")
            raise WorkflowError(f"unexpected workflow failure: {type(exc).__name__}") from exc

    async def _preflight(self) -> None:
        self._validate_declared_capabilities()
        key = self._key(
            "preflight",
            (self.capabilities.fingerprint,),
            provider=self.capabilities.fingerprint,
        )
        reusable = self.run_store.reusable_step(
            self._state,
            "preflight",
            key,
        )
        if reusable is None:
            step = self._begin_step(
                "preflight",
                key,
                (self.capabilities.fingerprint,),
            )
            for provider in (
                self.providers.planner,
                self.providers.image,
                self.providers.video,
                self.providers.judge,
            ):
                async with self.task_pool.slot():
                    await provider.preflight()
            self._complete_step(step)
        if self._state.status == ProjectStatus.CREATED:
            self._state = self.run_store.transition(
                self._state,
                ProjectStatus.PREFLIGHTED,
            )

    def _validate_declared_capabilities(self) -> None:
        planner = self.capabilities.planner.text
        image = self.capabilities.image.image
        video = self.capabilities.video.video
        judge = self.capabilities.judge.judge
        if planner is None or image is None or video is None or judge is None:
            raise WorkflowError("all four provider roles require declared capabilities")
        if planner.max_attachments < 6:
            raise WorkflowError("planner must accept six grounding probe attachments")
        if not image.supports_resolution(self.request.resolution):
            raise WorkflowError("image provider cannot produce delivery resolution")
        if not video.supports_resolution(self.request.resolution):
            raise WorkflowError("video provider cannot produce delivery resolution")
        delivery = self.request.generation_requirements.delivery_requirements
        if delivery.video_fps not in video.supported_fps:
            raise WorkflowError("video provider cannot produce delivery FPS")
        if not set(delivery.allowed_video_duration_seconds).intersection(video.supported_durations):
            raise WorkflowError("request and video provider have no common duration")
        if "image/png" not in judge.supported_media_types:
            raise WorkflowError("judge must support generated PNG evaluation")
        if "video/mp4" not in judge.supported_media_types:
            raise WorkflowError("judge must support final shot video evaluation")
        if self.request.provided_asset_bindings and image.max_input_images < 1:
            raise WorkflowError("guided provided assets require one image input slot")
        if self.request.provided_asset_bindings and judge.max_media < 2:
            raise WorkflowError("guided provided assets require two-media judge evaluation")
        if delivery.audio:
            raise WorkflowError("audio delivery is not implemented by this workflow")
        self._reference_image_size(panorama=True)

    async def _story_plan(self) -> StoryPlan:
        name = "story_plan"
        key = self._key(
            name,
            (self.request.request_hash, self.capabilities.fingerprint),
            implementation=STORY_COMPILER_VERSION,
            renderer=PROMPT_RENDERER_VERSION,
            provider=self.capabilities.planner.fingerprint,
        )
        reusable = self.run_store.reusable_step(self._state, name, key)
        if reusable and reusable.selected_output:
            plan = self._load(reusable.selected_output, StoryPlan)
            plan.assert_hash()
            prompts = tuple(
                prompt for attempt in reusable.attempt_records for prompt in attempt.prompts
            )
        else:
            step = self._begin_step(name, key, (self.request.request_hash,))
            started = utc_now()
            plan, calls, prompts = await self.execution.create_story_plan(
                self.request,
                on_prompt=self.stage_output.publish_story_prompt,
            )
            artifact = self._put(plan)
            attempts = tuple(
                AttemptRecord(
                    logical_attempt=index,
                    provider_call_refs=(metrics.call_ref,),
                    provider_usage=_provider_usage(metrics),
                    prompts=(prompts[index - 1],),
                    started_at=started,
                    completed_at=utc_now(),
                )
                for index, metrics in enumerate(calls, start=1)
            )
            step = step.model_copy(update={"attempt_records": attempts})
            self._complete_step(step, output=artifact, selected_key=plan.plan_key)
        if self._state.status == ProjectStatus.PREFLIGHTED:
            active_artifact = self._find_step(name).selected_output
            if active_artifact is None:
                raise WorkflowError("completed story plan step has no output")
            self._state = self.run_store.transition(
                self._state,
                ProjectStatus.STORY_PLANNED,
                active_story_plan=active_artifact,
            )
        else:
            active_artifact = self._find_step(name).selected_output
            if active_artifact is not None and self._state.active_story_plan != active_artifact:
                self._save_state(active_story_plan=active_artifact)
        self.stage_output.publish_story(plan, prompts)
        self.stage_output.publish_audit(self._state)
        return plan

    async def _references(
        self,
        story_plan: StoryPlan,
    ) -> tuple[ReferenceLibrary, tuple[ReferenceOperationResult, ...]]:
        results = tuple(
            [
                await self._reference_operation(story_plan, recipe)
                for recipe in self.execution.reference_recipes(self.request, story_plan)
            ]
        )
        name = "reference_library"
        operation_refs = tuple(
            self._find_step(f"reference:{item.recipe.need_key}").step_key for item in results
        )
        key = self._key(
            name,
            (story_plan.plan_hash, *operation_refs),
            implementation=REFERENCE_COMPILER_VERSION,
        )
        reusable = self.run_store.reusable_step(self._state, name, key)
        if reusable is not None and reusable.selected_output is not None:
            library = self._load(reusable.selected_output, ReferenceLibrary)
            try:
                library.assert_hash()
                for item in library.selected_assets:
                    self.run_store.artifacts.verify(item.artifact_ref)
                    for view in item.character_views:
                        self.run_store.artifacts.verify(view.artifact_ref)
            except (ArtifactError, ValueError):
                reusable = None
        else:
            reusable = None
        if reusable is None:
            step = self._begin_step(name, key, (story_plan.plan_hash, *operation_refs))
            library = self.execution.build_reference_library(
                story_plan,
                tuple(item.decision for item in results),
                tuple(candidate for item in results for candidate in item.candidates),
                tuple(report for item in results for report in item.reports),
            )
            artifact = self._put(library)
            self._complete_step(step, output=artifact, selected_key=library.library_hash)
        if self._state.status == ProjectStatus.STORY_PLANNED:
            active_artifact = self._find_step(name).selected_output
            if active_artifact is None:
                raise WorkflowError("completed reference library step has no output")
            self._state = self.run_store.transition(
                self._state,
                ProjectStatus.REFERENCES_READY,
                active_reference_library=active_artifact,
            )
        else:
            active_artifact = self._find_step(name).selected_output
            if (
                active_artifact is not None
                and self._state.active_reference_library != active_artifact
            ):
                self._save_state(active_reference_library=active_artifact)
        self.stage_output.publish_reference_library(library)
        self.stage_output.publish_audit(self._state)
        return library, results

    async def _reference_operation(
        self,
        story_plan: StoryPlan,
        recipe: ReferenceRecipe,
    ) -> ReferenceOperationResult:
        name = f"reference:{recipe.need_key}"
        provider_fingerprint = (
            None
            if isinstance(recipe, ProvidedReferenceRecipe)
            else self.capabilities.image.fingerprint
        )
        key = self._key(
            "reference",
            (canonical_hash(recipe),),
            renderer=PROMPT_RENDERER_VERSION,
            provider=provider_fingerprint,
        )
        reusable = self.run_store.reusable_step(self._state, name, key)
        require_panorama = _recipe_requires_panorama(recipe)
        if reusable and reusable.selected_output:
            result = self._load(reusable.selected_output, ReferenceOperationResult)
            expected_size: tuple[int, int] | None = None
            if not isinstance(recipe, ProvidedReferenceRecipe):
                size = self._reference_image_size(panorama=isinstance(recipe, ScenePanoramaRecipe))
                expected_size = (size.width, size.height)
            if self._valid_image_result(
                result,
                expected_size=expected_size,
                require_panorama=require_panorama,
                validate_media_bundle=not isinstance(
                    recipe,
                    GuidedScenePanoramaRecipe,
                ),
            ) and (
                not isinstance(recipe, CharacterReferenceRecipe)
                or self._valid_character_reference_bundle(result)
            ):
                self.stage_output.publish_reference(story_plan, result)
                return result
        if isinstance(recipe, CharacterReferenceRecipe):
            return await self._character_reference_operation(
                story_plan,
                recipe,
                name=name,
                key=key,
            )
        step = self._begin_step(name, key, (canonical_hash(recipe),))
        view = self.execution.reference_evaluation_view(recipe)
        if isinstance(recipe, ProvidedReferenceRecipe):
            candidate = await self._provided_candidate(recipe)
            evaluation_prompt = self.evaluation.render_prompt(view)
            prompt_record = _judge_prompt_record(
                candidate,
                evaluation_prompt,
                logical_attempt=1,
                candidate_index=1,
            )
            self.stage_output.publish_reference_prompt(
                story_plan,
                recipe,
                prompt_record,
            )
            report = await self._evaluate(candidate, view, prompt=evaluation_prompt)
            decision = self.execution.select(recipe.need_key, (candidate,), (report,))
            attempts = (
                AttemptRecord(
                    logical_attempt=1,
                    provider_call_refs=(candidate.provider_call_ref, report.evaluator_ref),
                    provider_usage=ProviderUsage.combine(
                        (candidate.provider_usage, report.evaluator_usage)
                    ),
                    candidate_refs=(candidate.candidate_key,),
                    evaluation_refs=(report.report_key,),
                    prompts=(prompt_record,),
                    started_at=utc_now(),
                    completed_at=utc_now(),
                ),
            )
            media = MediaOperationResult(
                operation_key=recipe.need_key,
                candidates=(candidate,),
                reports=(report,),
                decision=decision,
                attempts=attempts,
                prompts=(prompt_record,),
            )
        else:
            base_prompt = render_reference_image(recipe)
            size = self._reference_image_size(panorama=isinstance(recipe, ScenePanoramaRecipe))
            source_artifact: ArtifactRef | None = None
            if isinstance(recipe, GuidedScenePanoramaRecipe):
                source_candidate = await self._provided_candidate(recipe)
                if (
                    source_candidate.technical_status != TechnicalStatus.VALID
                    or source_candidate.artifact_ref is None
                ):
                    raise ReferenceError(
                        "uploaded scene reference is not a valid supported image"
                    )
                source_artifact = source_candidate.artifact_ref

            def request_factory(corrections: tuple[str, ...]) -> ImageRequest:
                prompt = base_prompt
                if corrections:
                    prompt += "\nCorrect: " + "; ".join(corrections)
                return ImageRequest(
                    prompt=prompt,
                    input_images=(
                        (
                            Attachment(
                                name="uploaded_scene_reference",
                                artifact_ref=source_artifact,
                            ),
                        )
                        if source_artifact is not None
                        else ()
                    ),
                    aspect_ratio=_aspect_ratio(size),
                    resolution=size,
                )

            candidate_acceptance: ImageCandidateAcceptance | None = None
            if isinstance(recipe, ScenePanoramaRecipe):

                async def accept_panorama(
                    candidate: CandidateRecord,
                ) -> PanoramaPreflightValue:
                    return await self._preflight_panorama_candidate(
                        story_plan,
                        recipe,
                        candidate,
                    )

                candidate_acceptance = accept_panorama

            media, step = await self._image_operation(
                step,
                operation_key=recipe.need_key,
                request_factory=request_factory,
                evaluation_view=view,
                evaluation_reference_media=(
                    (
                        CandidateMedia(
                            role="source_reference",
                            artifact_ref=source_artifact,
                        ),
                    )
                    if source_artifact is not None
                    else ()
                ),
                candidate_media_role=(
                    "scene_panorama" if source_artifact is not None else None
                ),
                require_panorama=require_panorama,
                candidate_acceptance=candidate_acceptance,
                on_prompt=lambda prompt: self.stage_output.publish_reference_prompt(
                    story_plan,
                    recipe,
                    prompt,
                ),
            )
        result = ReferenceOperationResult(
            recipe=recipe,
            **media.model_dump(),
        )
        output = self._put(result)
        self._complete_step(
            step.model_copy(update={"attempt_records": result.attempts}),
            output=output,
            selected_key=result.decision.selected_candidate,
        )
        self.stage_output.publish_reference(story_plan, result)
        return result

    async def _preflight_panorama_candidate(
        self,
        story_plan: StoryPlan,
        recipe: ScenePanoramaRecipe,
        candidate: CandidateRecord,
    ) -> PanoramaPreflightValue:
        if candidate.artifact_ref is None:
            raise SpatialError("panorama preflight candidate has no artifact")
        name = f"spatial_preflight:{candidate.candidate_key}"
        key = self._key(
            "panorama_spatial_preflight",
            (
                story_plan.plan_hash,
                canonical_hash(recipe.source_station),
                candidate.artifact_ref.sha256,
            ),
            implementation=(
                f"{GROUNDING_VALIDATOR_VERSION}+{FOV_SOLVER_VERSION}+{SPATIAL_RESOLVER_VERSION}"
            ),
            renderer=PROMPT_RENDERER_VERSION,
            provider=self.capabilities.planner.fingerprint,
            behavior={
                "grounding_attempts": self.config.generation.grounding_attempts,
                "max_hfov_degrees": self.config.generation.max_hfov_degrees,
                "supports_camera_motion": self.strict_spatial_resolver.supports_camera_motion,
            },
        )
        reusable = self.run_store.reusable_step(self._state, name, key)
        if reusable and reusable.selected_output:
            result = self._load(reusable.selected_output, PanoramaPreflightOutcome).outcome
            self.stage_output.publish_spatial_preflight(story_plan, result)
            return result
        step = self._begin_step(
            name,
            key,
            (story_plan.plan_hash, candidate.artifact_ref.sha256),
        )
        panorama = validate_panorama_source(
            self.run_store.artifacts,
            candidate.artifact_ref,
            scene_key=recipe.subject_key,
            story_plan_hash=story_plan.plan_hash,
            reference_library_hash="preflight",
        )
        scene = next(
            item for item in story_plan.scene_catalog if item.scene_key == recipe.subject_key
        )
        probes = await self._probe_set(story_plan, panorama)
        try:
            anchor_map = (
                await self._scene_anchor_map(
                    story_plan,
                    probes,
                    station_key=recipe.source_station.station_key,
                )
            ).anchor_map
        except GroundingValidationExhausted as exc:
            failure = _candidate_preflight_failure(
                candidate_key=candidate.candidate_key,
                scene_key=scene.scene_key,
                station_key=recipe.source_station.station_key,
                reference_hash=candidate.artifact_ref.sha256,
                error=exc,
            )
            output = self._put(PanoramaPreflightOutcome(outcome=failure))
            self._complete_step(
                step,
                output=output,
                selected_key="grounding_validation_exhausted",
            )
            self.stage_output.publish_spatial_preflight(story_plan, failure)
            return failure
        shot_results: list[SpatialPreflightShotResult] = []
        for shot in story_plan.ordered_shots:
            if shot.scene_key != scene.scene_key:
                continue
            resolved = await asyncio.to_thread(
                self.strict_spatial_resolver.resolve_source,
                shot=shot,
                scene=scene,
                panorama=panorama,
                station=recipe.source_station,
                anchor_map=anchor_map,
                allow_novel_station=True,
            )
            if resolved.plan is not None:
                shot_results.append(
                    SpatialPreflightShotResult(
                        shot_key=shot.shot_key,
                        status=resolved.plan.result_status.value,
                    )
                )
                continue
            conflict = resolved.conflict
            if (
                conflict is not None
                and conflict.kind
                in {
                    CameraConflictKind.STATION_INCOMPATIBLE,
                    CameraConflictKind.NO_COMMON_FOV,
                }
                and (
                    self.config.generation.spatial_repair_attempts > 0
                    and self.config.generation.novel_station_attempts > 0
                )
            ):
                shot_results.append(
                    SpatialPreflightShotResult(
                        shot_key=shot.shot_key,
                        status="NEEDS_NOVEL_STATION",
                        conflict_kind=conflict.kind,
                        detail=conflict.detail,
                    )
                )
                continue
            shot_results.append(
                SpatialPreflightShotResult(
                    shot_key=shot.shot_key,
                    status="FAILED",
                    conflict_kind=(conflict.kind if conflict is not None else None),
                    detail=(conflict.detail if conflict is not None else "unknown conflict"),
                )
            )
        failures = tuple(item for item in shot_results if item.status == "FAILED")
        correction = (
            "; ".join(
                f"{item.conflict_kind.value if item.conflict_kind else 'spatial_failed'} "
                f"for shot {item.shot_key}: {item.detail}"
                for item in failures
            )[:2_000]
            if failures
            else None
        )
        result = PanoramaPreflightResult(
            candidate_key=candidate.candidate_key,
            scene_key=scene.scene_key,
            station_key=recipe.source_station.station_key,
            anchor_map_hash=anchor_map.map_hash,
            executable=not failures,
            shots=tuple(shot_results),
            correction=correction,
        )
        output = self._put(PanoramaPreflightOutcome(outcome=result))
        self._complete_step(
            step,
            output=output,
            selected_key=("executable" if result.executable else "failed"),
        )
        self.stage_output.publish_spatial_preflight(story_plan, result)
        return result

    async def _preflight_novel_station_candidate(
        self,
        story_plan: StoryPlan,
        source_panorama: PanoramaSource,
        *,
        shot_key: str,
        station_key: str,
        candidate: CandidateRecord,
    ) -> PanoramaPreflightValue:
        if candidate.artifact_ref is None:
            raise SpatialError("novel station preflight candidate has no artifact")
        name = f"novel_station_preflight:{candidate.candidate_key}"
        key = self._key(
            "novel_station_spatial_preflight",
            (
                story_plan.plan_hash,
                source_panorama.artifact_ref.sha256,
                station_key,
                candidate.artifact_ref.sha256,
                shot_key,
            ),
            implementation=(
                f"{GROUNDING_VALIDATOR_VERSION}+{FOV_SOLVER_VERSION}+{SPATIAL_RESOLVER_VERSION}"
            ),
            renderer=PROMPT_RENDERER_VERSION,
            provider=self.capabilities.planner.fingerprint,
            behavior={
                "grounding_attempts": self.config.generation.grounding_attempts,
                "max_hfov_degrees": self.config.generation.max_hfov_degrees,
                "supports_camera_motion": self.strict_spatial_resolver.supports_camera_motion,
            },
        )
        reusable = self.run_store.reusable_step(self._state, name, key)
        if reusable and reusable.selected_output:
            result = self._load(reusable.selected_output, PanoramaPreflightOutcome).outcome
            self.stage_output.publish_spatial_preflight(story_plan, result)
            return result
        step = self._begin_step(
            name,
            key,
            (story_plan.plan_hash, candidate.artifact_ref.sha256, shot_key),
        )
        panorama = validate_panorama_source(
            self.run_store.artifacts,
            candidate.artifact_ref,
            scene_key=source_panorama.scene_key,
            story_plan_hash=story_plan.plan_hash,
            reference_library_hash=source_panorama.reference_library_hash,
        )
        shot = next(item for item in story_plan.ordered_shots if item.shot_key == shot_key)
        scene = next(item for item in story_plan.scene_catalog if item.scene_key == shot.scene_key)
        probes = await self._probe_set(story_plan, panorama)
        try:
            anchor_map = (
                await self._scene_anchor_map(
                    story_plan,
                    probes,
                    station_key=station_key,
                )
            ).anchor_map
        except GroundingValidationExhausted as exc:
            failure = _candidate_preflight_failure(
                candidate_key=candidate.candidate_key,
                scene_key=scene.scene_key,
                station_key=station_key,
                reference_hash=candidate.artifact_ref.sha256,
                error=exc,
            )
            output = self._put(PanoramaPreflightOutcome(outcome=failure))
            self._complete_step(
                step,
                output=output,
                selected_key="grounding_validation_exhausted",
            )
            self.stage_output.publish_spatial_preflight(story_plan, failure)
            return failure
        station = SourceStationSpec(
            station_key=station_key,
            zone_key=shot.spatial_intent.action_zone,
            description=shot.spatial_intent.viewpoint_intent.description,
            supported_viewpoint_intents=(shot.spatial_intent.viewpoint_intent.intent_key,),
        )
        resolved = await asyncio.to_thread(
            self.strict_spatial_resolver.resolve_source,
            shot=shot,
            scene=scene,
            panorama=panorama,
            station=station,
            anchor_map=anchor_map,
            allow_novel_station=False,
            translated_station=True,
        )
        if resolved.plan is not None:
            shot_result = SpatialPreflightShotResult(
                shot_key=shot_key,
                status=resolved.plan.result_status.value,
            )
            correction = None
        else:
            conflict = resolved.conflict
            shot_result = SpatialPreflightShotResult(
                shot_key=shot_key,
                status="FAILED",
                conflict_kind=(conflict.kind if conflict is not None else None),
                detail=(conflict.detail if conflict is not None else "unknown spatial conflict"),
            )
            correction = (
                f"{conflict.kind.value if conflict is not None else 'spatial_failed'} "
                f"at translated station for shot {shot_key}: {shot_result.detail}"
            )[:2_000]
        result = PanoramaPreflightResult(
            candidate_key=candidate.candidate_key,
            scene_key=shot.scene_key,
            station_key=station_key,
            anchor_map_hash=anchor_map.map_hash,
            executable=resolved.plan is not None,
            shots=(shot_result,),
            correction=correction,
        )
        output = self._put(PanoramaPreflightOutcome(outcome=result))
        self._complete_step(
            step,
            output=output,
            selected_key=("executable" if result.executable else "failed"),
        )
        self.stage_output.publish_spatial_preflight(story_plan, result)
        return result

    async def _character_reference_operation(
        self,
        story_plan: StoryPlan,
        recipe: CharacterReferenceRecipe,
        *,
        name: str,
        key: str,
    ) -> ReferenceOperationResult:
        judge = self.capabilities.judge.judge
        if judge is None or judge.max_media < 2:
            raise WorkflowError(
                "judge must accept a front reference and one derived character view"
            )
        source: ArtifactRef | None = None
        if isinstance(recipe, GuidedCharacterReferenceRecipe):
            source_candidate = await self._provided_candidate(recipe)
            if (
                source_candidate.technical_status != TechnicalStatus.VALID
                or source_candidate.artifact_ref is None
            ):
                raise ReferenceError(
                    "uploaded character reference is not a valid supported image"
                )
            source = source_candidate.artifact_ref
        front = await self._character_reference_view_operation(
            story_plan,
            recipe,
            role="front",
            source=source,
        )
        front_candidate = _selected_candidate(front)
        front_artifact = _required_artifact(front_candidate)
        side = await self._character_reference_view_operation(
            story_plan,
            recipe,
            role="side",
            front=front_artifact,
        )
        back = await self._character_reference_view_operation(
            story_plan,
            recipe,
            role="back",
            front=front_artifact,
        )
        selected_candidates = tuple(_selected_candidate(result) for result in (front, side, back))
        selected_reports = tuple(
            _selected_report(result.decision, result.reports) for result in (front, side, back)
        )
        artifacts = tuple(_required_artifact(item) for item in selected_candidates)
        media = tuple(
            CandidateMedia(role=role, artifact_ref=artifact)
            for role, artifact in zip(
                ("front", "side", "back"),
                artifacts,
                strict=True,
            )
        )
        provider_call_refs = tuple(
            reference
            for candidate in selected_candidates
            for reference in (
                candidate.provider_call_refs
                if candidate.provider_call_refs
                else (candidate.provider_call_ref,)
            )
        )
        bundle_key = stable_key(
            "candidate",
            recipe.need_key,
            canonical_hash(tuple(item.artifact_ref.sha256 for item in media)),
        )
        bundle_candidate = CandidateRecord(
            candidate_key=bundle_key,
            operation_key=recipe.need_key,
            artifact_ref=artifacts[0],
            media=media,
            technical_status=TechnicalStatus.VALID,
            technical_findings=tuple(
                TechnicalFinding(
                    code=f"character_{role}_{finding.code}",
                    passed=finding.passed,
                    detail=finding.detail,
                )
                for role, candidate in zip(
                    ("front", "side", "back"),
                    selected_candidates,
                    strict=True,
                )
                for finding in candidate.technical_findings
            ),
            technical_quality=min(item.technical_quality for item in selected_candidates),
            provider_call_ref=(
                provider_call_refs[-1]
                if provider_call_refs
                else stable_key("provider_bundle", recipe.need_key, bundle_key)
            ),
            provider_call_refs=provider_call_refs,
            provider_usage=ProviderUsage.combine(
                tuple(item.provider_usage for item in selected_candidates)
            ),
            logical_attempt=max(item.logical_attempt for item in selected_candidates),
        )
        criterion_results = tuple(
            criterion for report in selected_reports for criterion in report.criterion_results
        )
        evaluator_refs = tuple(item.evaluator_ref for item in selected_reports)
        evaluator_ref = stable_key(
            "evaluator_bundle",
            recipe.need_key,
            canonical_hash(evaluator_refs),
        )
        report_payload = {
            "candidate_key": bundle_key,
            "criterion_results": criterion_results,
            "evaluator_ref": evaluator_ref,
        }
        bundle_report = EvaluationReport(
            report_key=stable_key(
                "evaluation",
                bundle_key,
                canonical_hash(report_payload),
            ),
            candidate_key=bundle_key,
            criterion_results=criterion_results,
            evaluator_ref=evaluator_ref,
            evaluator_usage=ProviderUsage.combine(
                tuple(item.evaluator_usage for item in selected_reports)
            ),
        )
        decision = self.execution.select(
            recipe.need_key,
            (bundle_candidate,),
            (bundle_report,),
        )
        result = ReferenceOperationResult(
            recipe=recipe,
            operation_key=recipe.need_key,
            candidates=(bundle_candidate,),
            reports=(bundle_report,),
            decision=decision,
            attempts=(),
            prompts=tuple(
                prompt for view_result in (front, side, back) for prompt in view_result.prompts
            ),
        )
        step = self._begin_step(
            name,
            key,
            (
                canonical_hash(recipe),
                *(
                    self._find_step(
                        self._character_reference_view_step_name(
                            recipe.need_key,
                            role,
                        )
                    ).step_key
                    for role in CHARACTER_VIEW_ROLES
                ),
            ),
        )
        output = self._put(result)
        self._complete_step(
            step,
            output=output,
            selected_key=result.decision.selected_candidate,
        )
        self.stage_output.publish_reference(story_plan, result)
        return result

    async def _character_reference_view_operation(
        self,
        story_plan: StoryPlan,
        recipe: CharacterReferenceRecipe,
        *,
        role: CharacterViewRole,
        front: ArtifactRef | None = None,
        source: ArtifactRef | None = None,
    ) -> MediaOperationResult:
        if (
            (role == "front") != (front is None)
            or (source is not None and role != "front")
            or (
                role == "front"
                and isinstance(recipe, GuidedCharacterReferenceRecipe)
                != (source is not None)
            )
        ):
            raise WorkflowError("character view front-reference contract is invalid")
        name = self._character_reference_view_step_name(recipe.need_key, role)
        operation_key = stable_key(
            "operation",
            recipe.need_key,
            f"character_view:{role}",
        )
        inputs = (
            canonical_hash(recipe),
            role,
            front.sha256 if front is not None else "no_front_reference",
            source.sha256 if source is not None else "no_uploaded_source",
        )
        key = self._key(
            "character_reference_view",
            inputs,
            implementation=REFERENCE_COMPILER_VERSION,
            renderer=PROMPT_RENDERER_VERSION,
            provider=self.capabilities.image.fingerprint,
        )
        reusable = self.run_store.reusable_step(self._state, name, key)
        size = self._reference_image_size(panorama=False)
        if reusable is not None and reusable.selected_output is not None:
            result = self._load(reusable.selected_output, MediaOperationResult)
            if self._valid_image_result(
                result,
                expected_size=(size.width, size.height),
                validate_media_bundle=source is None,
            ) and self._valid_character_view_evaluation_media(
                result,
                role=role,
                front=front,
                source=source,
            ):
                return result
        step = self._begin_step(name, key, inputs)
        view = self.execution.character_view_evaluation_view(recipe, role)

        def request_factory(corrections: tuple[str, ...]) -> ImageRequest:
            return ImageRequest(
                prompt=render_character_reference(
                    recipe,
                    role,
                    guided_source=source is not None,
                    corrections=corrections,
                ),
                input_images=(
                    (
                        Attachment(
                            name=(
                                "uploaded_character_reference"
                                if source is not None
                                else "front_reference"
                            ),
                            artifact_ref=(source if source is not None else front),
                        ),
                    )
                    if front is not None or source is not None
                    else ()
                ),
                aspect_ratio=_aspect_ratio(size),
                resolution=size,
            )

        result, step = await self._image_operation(
            step,
            operation_key=operation_key,
            request_factory=request_factory,
            evaluation_view=view,
            evaluation_reference_media=(
                (
                    CandidateMedia(
                        role=(
                            "source_reference"
                            if source is not None
                            else "front_reference"
                        ),
                        artifact_ref=(source if source is not None else front),
                    ),
                )
                if front is not None or source is not None
                else ()
            ),
            candidate_media_role=(
                role if front is not None or source is not None else None
            ),
            on_prompt=lambda prompt: self.stage_output.publish_reference_prompt(
                story_plan,
                recipe,
                prompt,
            ),
        )
        output = self._put(result)
        self._complete_step(
            step.model_copy(update={"attempt_records": result.attempts}),
            output=output,
            selected_key=result.decision.selected_candidate,
        )
        return result

    @staticmethod
    def _character_reference_view_step_name(
        need_key: str,
        role: CharacterViewRole,
    ) -> str:
        return f"reference_view:{need_key}:{role}"

    @staticmethod
    def _valid_character_view_evaluation_media(
        result: MediaOperationResult,
        *,
        role: CharacterViewRole,
        front: ArtifactRef | None,
        source: ArtifactRef | None,
    ) -> bool:
        try:
            candidate = _selected_candidate(result)
        except WorkflowError:
            return False
        if role == "front":
            if source is None:
                return not candidate.media and front is None
            return (
                front is None
                and tuple(item.role for item in candidate.media)
                == ("source_reference", "front")
                and candidate.media[0].artifact_ref == source
                and candidate.media[1].artifact_ref == candidate.artifact_ref
            )
        return (
            front is not None
            and tuple(item.role for item in candidate.media) == ("front_reference", role)
            and candidate.media[0].artifact_ref == front
            and candidate.media[1].artifact_ref == candidate.artifact_ref
        )

    @staticmethod
    def _valid_character_reference_bundle(
        result: MediaOperationResult,
    ) -> bool:
        try:
            candidate = _selected_candidate(result)
        except WorkflowError:
            return False
        return (
            tuple(item.role for item in candidate.media) == ("front", "side", "back")
            and candidate.artifact_ref == candidate.media[0].artifact_ref
        )

    async def _provided_candidate(
        self,
        recipe: ProvidedReferenceRecipe
        | GuidedCharacterReferenceRecipe
        | GuidedScenePanoramaRecipe,
    ) -> CandidateRecord:
        asset = next(
            (
                item
                for item in self.request.provided_assets
                if item.asset_id == recipe.provided_asset_id
            ),
            None,
        )
        if asset is None:
            raise ReferenceError(f"provided asset is missing: {recipe.provided_asset_id}")
        if asset.path is not None:
            media_type = mimetypes.guess_type(asset.path.name)[0] or "application/octet-stream"
            artifact = self.run_store.artifacts.put_file(asset.path, media_type)
        elif asset.uri is not None:
            async with self.task_pool.slot():
                try:
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(120.0),
                        follow_redirects=True,
                    ) as client:
                        response = await client.get(asset.uri)
                        response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise ReferenceError(
                        f"cannot retrieve provided asset: HTTP {exc.response.status_code}"
                    ) from None
                except httpx.HTTPError:
                    raise ReferenceError(
                        "cannot retrieve provided asset: network failure"
                    ) from None
            maximum = (
                50 * 1024 * 1024
                if asset.kind in {"character", "scene", "panorama", "image"}
                else 250 * 1024 * 1024
            )
            if len(response.content) > maximum:
                raise ReferenceError(
                    f"provided asset exceeds {maximum // (1024 * 1024)} MiB"
                )
            media_type = response.headers.get(
                "content-type",
                "application/octet-stream",
            ).split(";", 1)[0]
            artifact = self.run_store.artifacts.put_bytes(response.content, media_type)
        else:
            raise ReferenceError("provided asset has no source")
        if asset.content_sha256 is not None and artifact.sha256 != asset.content_sha256:
            raise ReferenceError(
                f"provided asset content changed after validation: {asset.asset_id}"
            )
        if asset.kind == "video":
            video_validation = VideoTechnicalValidator().validate(
                self.run_store.artifacts,
                artifact,
            )
            return CandidateRecord(
                candidate_key=stable_key("candidate", recipe.need_key, "1:1"),
                operation_key=recipe.need_key,
                artifact_ref=artifact,
                technical_status=video_validation.status,
                technical_findings=video_validation.findings,
                technical_quality=video_validation.quality_score,
                provider_call_ref=f"provided:{asset.asset_id}",
                provider_usage=ProviderUsage(),
                logical_attempt=1,
            )
        validator = ImageTechnicalValidator()
        image_validation = validator.validate(
            self.run_store.artifacts,
            artifact,
            require_panorama=(
                recipe.require_panorama
                if isinstance(recipe, ProvidedReferenceRecipe)
                else False
            ),
        )
        if image_validation.info is None:
            raise ReferenceError("provided reference asset is not a decodable image")
        normalized_media_type = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
        }.get(image_validation.info.format)
        if normalized_media_type is None:
            raise ReferenceError("provided reference image format is unsupported")
        if artifact.media_type != normalized_media_type:
            artifact = self.run_store.artifacts.put_bytes(
                self.run_store.artifacts.verify(artifact).read_bytes(),
                normalized_media_type,
            )
            image_validation = validator.validate(
                self.run_store.artifacts,
                artifact,
                require_panorama=(
                    recipe.require_panorama
                    if isinstance(recipe, ProvidedReferenceRecipe)
                    else False
                ),
            )
        return CandidateRecord(
            candidate_key=stable_key("candidate", recipe.need_key, "1:1"),
            operation_key=recipe.need_key,
            artifact_ref=artifact,
            technical_status=image_validation.status,
            technical_findings=image_validation.findings,
            technical_quality=(1.0 if image_validation.status == TechnicalStatus.VALID else 0.0),
            provider_call_ref=f"provided:{asset.asset_id}",
            provider_usage=ProviderUsage(),
            logical_attempt=1,
        )

    def _reference_image_size(self, *, panorama: bool) -> PixelSize:
        capability = self.capabilities.image.image
        if capability is None:
            raise WorkflowError("image capability is missing")
        ratio = "2:1" if panorama else _aspect_ratio(self.request.resolution)
        if ratio not in capability.supported_aspect_ratios:
            raise WorkflowError(f"image provider does not support required aspect ratio {ratio}")
        if not capability.supported_resolutions:
            return PixelSize(width=1024, height=512) if panorama else self.request.resolution
        candidates = tuple(
            item for item in capability.supported_resolutions if _aspect_ratio(item) == ratio
        )
        if not candidates:
            raise WorkflowError(f"image provider has no declared {ratio} resolution")
        if not panorama and self.request.resolution in candidates:
            return self.request.resolution
        return max(candidates, key=lambda item: item.width * item.height)

    async def _spatial(
        self,
        story_plan: StoryPlan,
        library: ReferenceLibrary,
    ) -> tuple[ReferenceLibrary, tuple[ResolvedSpatialPlan, ...]]:
        panoramas: dict[str, PanoramaSource] = {}
        probes: dict[str, ProbeSet] = {}
        source_evidence: dict[str, StationSpatialEvidence] = {}
        bindings = {item.scene_key: item for item in library.panorama_by_scene}
        selected_by_need = {item.need_key: item for item in library.selected_assets}
        for binding in library.panorama_by_scene:
            source = validate_panorama_source(
                self.run_store.artifacts,
                binding.artifact_ref,
                scene_key=binding.scene_key,
                story_plan_hash=story_plan.plan_hash,
                reference_library_hash=library.library_hash,
            )
            panoramas[binding.scene_key] = source
            probes[binding.scene_key] = await self._probe_set(story_plan, source)
            selected = selected_by_need[binding.selected_reference]
            preflight = self._persisted_preflight(selected.candidate_key)
            if isinstance(preflight, CandidatePreflightFailure):
                if self.config.generation.spatial_failure_policy == SpatialFailurePolicy.STRICT:
                    raise CandidatePreflightExhausted(
                        operation_key=binding.selected_reference,
                        candidate_keys=(selected.candidate_key,),
                        failure_codes=(preflight.failure_code,),
                        detail=preflight.correction,
                    )
                source_evidence[binding.scene_key] = _grounding_unavailable_from_failure(preflight)
                continue
            try:
                anchor_map = (
                    await self._scene_anchor_map(
                        story_plan,
                        probes[binding.scene_key],
                        station_key=binding.source_station.station_key,
                    )
                ).anchor_map
            except GroundingValidationExhausted as exc:
                if self.config.generation.spatial_failure_policy == SpatialFailurePolicy.STRICT:
                    raise
                source_evidence[binding.scene_key] = _grounding_unavailable_from_error(exc)
            else:
                source_evidence[binding.scene_key] = AnchorMapEvidence.from_anchor_map(anchor_map)
        result_rows: list[SpatialOperationResult] = []
        for shot in story_plan.ordered_shots:
            evidence = source_evidence[shot.scene_key]
            if isinstance(evidence, AnchorMapEvidence):
                result_rows.append(
                    await self._spatial_shot(
                        story_plan,
                        panoramas[shot.scene_key],
                        probes[shot.scene_key],
                        bindings[shot.scene_key].source_station,
                        evidence.anchor_map,
                        shot.shot_key,
                    )
                )
            else:
                result_rows.append(
                    await self._spatial_shot_grounding_unavailable(
                        story_plan,
                        panoramas[shot.scene_key],
                        bindings[shot.scene_key].source_station,
                        evidence,
                        shot.shot_key,
                    )
                )
        results = tuple(result_rows)
        resolutions = tuple(item.resolution for item in results)
        spatial_references: list[SceneSpatialReference] = []
        for scene_key, binding in sorted(bindings.items()):
            novel_stations: list[StationSpatialReference] = []
            for result in results:
                if (
                    result.resolution.scene_key != scene_key
                    or result.resolution.camera_recipe.kind != "novel_station"
                ):
                    continue
                if not isinstance(result.evidence, AnchorMapEvidence):
                    raise WorkflowError("resolved novel station lacks Anchor Map evidence")
                novel_stations.append(
                    StationSpatialReference(
                        kind="novel_station",
                        station_key=result.resolution.station.station_key,
                        description=result.resolution.station.description,
                        station_panorama=result.resolution.camera_recipe.station_panorama,
                        scene_view=result.resolution.camera_recipe.scene_view,
                        evidence=result.evidence,
                        selected_candidate=(result.resolution.camera_recipe.selected_candidate),
                    )
                )
            spatial_references.append(
                SceneSpatialReference(
                    scene_key=scene_key,
                    canonical_panorama=binding.artifact_ref,
                    source_station=StationSpatialReference(
                        kind="source_station",
                        station_key=binding.source_station.station_key,
                        description=binding.source_station.description,
                        station_panorama=binding.artifact_ref,
                        scene_view=binding.artifact_ref,
                        evidence=source_evidence[scene_key],
                    ),
                    novel_stations=tuple(novel_stations),
                )
            )
        final_library = self.execution.reference_compiler.attach_spatial_references(
            library,
            tuple(spatial_references),
        )
        name = "reference_library_spatial"
        key = self._key(
            name,
            (
                library.library_hash,
                *(item.evidence_hash for item in source_evidence.values()),
                *(item.evidence.evidence_hash for item in results),
                *(item.resolution_hash for item in resolutions),
            ),
            implementation=REFERENCE_COMPILER_VERSION,
        )
        reusable = self.run_store.reusable_step(self._state, name, key)
        if reusable is None or reusable.selected_output is None:
            step = self._begin_step(name, key, (library.library_hash,))
            artifact = self._put(final_library)
            self._complete_step(
                step,
                output=artifact,
                selected_key=final_library.library_hash,
            )
        else:
            final_library = self._load(reusable.selected_output, ReferenceLibrary)
            final_library.assert_hash()
        active = self._find_step(name).selected_output
        if active is None:
            raise WorkflowError("spatial reference library step has no output")
        self._save_state(active_reference_library=active)
        self.stage_output.publish_reference_library(final_library)
        self.stage_output.publish_audit(self._state)
        return final_library, resolutions

    async def _probe_set(
        self,
        story_plan: StoryPlan,
        panorama: PanoramaSource,
    ) -> ProbeSet:
        name = f"probes:{panorama.scene_key}:{panorama.artifact_ref.sha256[:16]}"
        key = self._key(
            "probe_set",
            (panorama.artifact_ref.sha256,),
            implementation=PANORAMA_PROJECTOR_VERSION,
        )
        reusable = self.run_store.reusable_step(self._state, name, key)
        if reusable and reusable.selected_output:
            result = self._load(reusable.selected_output, ProbeSet)
            if all(
                ImageTechnicalValidator()
                .validate(
                    self.run_store.artifacts,
                    probe.artifact_ref,
                    expected_size=(probe.width, probe.height),
                )
                .status
                == TechnicalStatus.VALID
                for probe in result.probes
            ):
                self.stage_output.publish_probes(
                    story_plan,
                    panorama.scene_key,
                    result,
                )
                return result
        step = self._begin_step(name, key, (panorama.artifact_ref.sha256,))
        async with self.task_pool.slot():
            result = await asyncio.to_thread(
                self.projector.generate_six_probes,
                panorama,
            )
        output = self._put(result)
        self._complete_step(step, output=output, selected_key=result.probe_set_hash)
        self.stage_output.publish_probes(story_plan, panorama.scene_key, result)
        return result

    async def _scene_anchor_map(
        self,
        story_plan: StoryPlan,
        probes: ProbeSet,
        *,
        station_key: str,
    ) -> SceneAnchorOperationResult:
        name = f"anchor_map:{probes.scene_key}:{station_key}:{probes.source_panorama_hash[:16]}"
        key = self._key(
            "scene_anchor_map",
            (story_plan.plan_hash, probes.probe_set_hash, station_key),
            implementation=GROUNDING_VALIDATOR_VERSION,
            renderer=PROMPT_RENDERER_VERSION,
            provider=self.capabilities.planner.fingerprint,
        )
        reusable = self.run_store.reusable_step(self._state, name, key)
        if reusable and reusable.selected_output:
            result = self._load(reusable.selected_output, SceneAnchorOperationResult)
            result.anchor_map.assert_hash()
            self.stage_output.publish_anchor_map(story_plan, result.anchor_map)
            return result
        step = self._begin_step(
            name,
            key,
            (story_plan.plan_hash, probes.probe_set_hash, station_key),
        )

        def persist_attempt(event: GroundingAttemptEvent) -> None:
            nonlocal step
            response_payload = {
                "logical_attempt": event.logical_attempt,
                "scene_key": probes.scene_key,
                "station_key": station_key,
                "reference_hash": probes.source_panorama_hash,
                "target_aliases": event.target_aliases,
                "raw_response": event.raw_response,
                "provider_call_ref": event.metrics.call_ref,
            }
            validation_payload = {
                "logical_attempt": event.logical_attempt,
                "status": event.validation_status.value,
                "code": event.validation_code,
                "error": event.validation_error,
                "next_correction": event.next_correction,
                "scene_key": probes.scene_key,
                "station_key": station_key,
                "reference_hash": probes.source_panorama_hash,
                "target_aliases": event.target_aliases,
            }
            response_ref = self.run_store.artifacts.put_bytes(
                canonical_bytes(response_payload) + b"\n",
                "application/json",
            )
            validation_ref = self.run_store.artifacts.put_bytes(
                canonical_bytes(validation_payload) + b"\n",
                "application/json",
            )
            record = AttemptRecord(
                logical_attempt=event.logical_attempt,
                attempt_ref=stable_key(
                    "grounding_attempt",
                    probes.probe_set_hash,
                    f"{station_key}:{event.logical_attempt}",
                ),
                provider_call_refs=(event.metrics.call_ref,),
                target_aliases=event.target_aliases,
                response_refs=(response_ref,),
                validation_refs=(validation_ref,),
                validation_status=event.validation_status.value,
                validation_code=event.validation_code,
                provider_usage=_provider_usage(event.metrics),
                prompts=(event.prompt,),
                started_at=event.started_at,
                completed_at=event.completed_at,
            )
            records = {item.logical_attempt: item for item in step.attempt_records}
            records[event.logical_attempt] = record
            step = step.model_copy(
                update={"attempt_records": tuple(records[index] for index in sorted(records))}
            )
            self._state = self.run_store.upsert_step(self._state, step)
            self.stage_output.publish_grounding_attempt(
                story_plan,
                probes.scene_key,
                record,
            )

        try:
            anchor_map, _calls, _prompts = await self.execution.ground(
                story_plan,
                probes,
                station_key=station_key,
                on_prompt=lambda prompt: self.stage_output.publish_anchor_prompt(
                    story_plan,
                    probes.scene_key,
                    prompt,
                ),
                on_attempt=persist_attempt,
            )
        except GroundingValidationExhausted as exc:
            exc.logical_attempt_refs = tuple(
                item.attempt_ref for item in step.attempt_records if item.attempt_ref is not None
            )
            exc.response_refs = tuple(
                reference.sha256
                for item in step.attempt_records
                for reference in item.response_refs
            )
            exc.validation_refs = tuple(
                reference.sha256
                for item in step.attempt_records
                for reference in item.validation_refs
            )
            failed = step.model_copy(
                update={"status": StepStatus.FAILED, "error": str(exc)[:2_000]}
            )
            self._state = self.run_store.upsert_step(self._state, failed)
            self.run_store.append_event(
                "step_failed",
                {"step": step.step_name, "step_key": step.step_key, "error": str(exc)},
            )
            self.stage_output.publish_audit(self._state)
            raise
        result = SceneAnchorOperationResult(probe_set=probes, anchor_map=anchor_map)
        output = self._put(result)
        self._complete_step(
            step,
            output=output,
            selected_key=anchor_map.map_key,
        )
        self.stage_output.publish_anchor_map(story_plan, anchor_map)
        return result

    def _persisted_preflight(
        self,
        candidate_key: str,
        *,
        novel_station: bool = False,
    ) -> PanoramaPreflightValue | None:
        prefix = "novel_station_preflight" if novel_station else "spatial_preflight"
        name = f"{prefix}:{candidate_key}"
        step = next(
            (
                item
                for item in self._state.steps
                if item.step_name == name
                and item.status == StepStatus.COMPLETED
                and item.selected_output is not None
            ),
            None,
        )
        if step is None or step.selected_output is None:
            return None
        return self._load(step.selected_output, PanoramaPreflightOutcome).outcome

    async def _spatial_shot_grounding_unavailable(
        self,
        story_plan: StoryPlan,
        panorama: PanoramaSource,
        source_station: SourceStationSpec,
        evidence: GroundingUnavailableEvidence,
        shot_key: str,
    ) -> SpatialOperationResult:
        if self.config.generation.spatial_failure_policy != SpatialFailurePolicy.DEGRADE:
            raise SpatialError("Grounding unavailable safe shots require degrade policy")
        story_shot = next(item for item in story_plan.ordered_shots if item.shot_key == shot_key)
        name = f"spatial:{shot_key}"
        key = self._key(
            "spatial_grounding_unavailable",
            (story_plan.plan_hash, evidence.evidence_hash, shot_key),
            implementation=(
                f"{GROUNDING_VALIDATOR_VERSION}+{SPATIAL_RESOLVER_VERSION}+{CAMERA_SOLVER_VERSION}"
            ),
            behavior={
                "spatial_failure_policy": self.config.generation.spatial_failure_policy,
                "safe_fallback_hfov_min": self.config.generation.safe_fallback_hfov_min,
                "safe_fallback_hfov_max": self.config.generation.safe_fallback_hfov_max,
            },
        )
        reusable = self.run_store.reusable_step(self._state, name, key)
        if reusable and reusable.selected_output:
            result = self._load(reusable.selected_output, SpatialOperationResult)
            try:
                if not isinstance(result.evidence, GroundingUnavailableEvidence):
                    raise ValueError("safe spatial result has another evidence kind")
                result.evidence.assert_hash()
                result.resolution.assert_hash()
                self.run_store.artifacts.verify(result.resolution.camera_recipe.scene_view)
                self.stage_output.publish_spatial_resolution(story_plan, result.resolution)
                return result
            except (ArtifactError, ValueError):
                pass
        step = self._begin_step(
            name,
            key,
            (story_plan.plan_hash, evidence.evidence_hash, shot_key),
        )
        plan = await asyncio.to_thread(
            self.spatial_resolver.degrade_grounding_unavailable,
            shot=story_shot,
            panorama=panorama,
            station=source_station,
            failed_target_aliases=evidence.failed_target_aliases,
        )
        result = SpatialOperationResult(evidence=evidence, resolution=plan)
        output = self._put(result)
        self._complete_step(step, output=output, selected_key=plan.resolution_hash)
        self.stage_output.publish_spatial_resolution(story_plan, plan)
        return result

    async def _spatial_shot(
        self,
        story_plan: StoryPlan,
        panorama: PanoramaSource,
        probes: ProbeSet,
        source_station: SourceStationSpec,
        anchor_map: SceneAnchorMap,
        shot_key: str,
    ) -> SpatialOperationResult:
        name = f"spatial:{shot_key}"
        story_shot = next(item for item in story_plan.ordered_shots if item.shot_key == shot_key)
        key = self._key(
            "spatial",
            (story_plan.plan_hash, anchor_map.map_hash, shot_key),
            implementation=(
                f"{GROUNDING_VALIDATOR_VERSION}+{FOV_SOLVER_VERSION}+"
                f"{SPATIAL_RESOLVER_VERSION}+{CAMERA_SOLVER_VERSION}"
            ),
            renderer=PROMPT_RENDERER_VERSION,
            provider=canonical_hash(
                (
                    self.capabilities.planner.fingerprint,
                    self.capabilities.image.fingerprint,
                )
            ),
            behavior={
                "spatial_failure_policy": self.config.generation.spatial_failure_policy,
                "spatial_repair_attempts": self.config.generation.spatial_repair_attempts,
                "novel_station_attempts": self.config.generation.novel_station_attempts,
                "max_hfov_degrees": self.config.generation.max_hfov_degrees,
                "safe_fallback_hfov_min": self.config.generation.safe_fallback_hfov_min,
                "safe_fallback_hfov_max": self.config.generation.safe_fallback_hfov_max,
            },
        )
        reusable = self.run_store.reusable_step(self._state, name, key)
        if reusable and reusable.selected_output:
            result = self._load(reusable.selected_output, SpatialOperationResult)
            try:
                result.resolution.assert_hash()
                if not isinstance(result.evidence, AnchorMapEvidence):
                    raise ValueError("grounded spatial result lacks Anchor Map evidence")
                result.evidence.assert_hash()
                result_anchor_map = result.evidence.anchor_map
                recipe = result.resolution.camera_recipe
                reference_hash = (
                    recipe.station_panorama.sha256
                    if recipe.kind == "novel_station"
                    else recipe.source_panorama_hash
                )
                if (
                    result_anchor_map.station_key != recipe.station_key
                    or result_anchor_map.reference_hash != reference_hash
                ):
                    raise ValueError("spatial result Anchor Map does not match its camera station")
                self.run_store.artifacts.verify(result.resolution.camera_recipe.scene_view)
                if recipe.kind == "novel_station":
                    self.run_store.artifacts.verify(recipe.station_panorama)
                self.stage_output.publish_spatial_resolution(story_plan, result.resolution)
                return result
            except (ArtifactError, ValueError):
                pass
        step = self._begin_step(
            name,
            key,
            (story_plan.plan_hash, anchor_map.map_hash, shot_key),
        )
        scene = next(
            item for item in story_plan.scene_catalog if item.scene_key == story_shot.scene_key
        )
        resolution = await asyncio.to_thread(
            self.spatial_resolver.resolve_source,
            shot=story_shot,
            scene=scene,
            panorama=panorama,
            station=source_station,
            anchor_map=anchor_map,
            allow_novel_station=(
                self.config.generation.spatial_repair_attempts > 0
                and self.config.generation.novel_station_attempts > 0
            ),
        )
        plan = resolution.plan
        selected_anchor_map = anchor_map
        novel_failure_detail: str | None = None
        novel_failure_kind: CameraConflictKind | None = None
        novel_station_key: str | None = None
        if plan is None and resolution.novel_station_needed:
            station_key = stable_key(
                "novel_station",
                shot_key,
                story_shot.spatial_intent.viewpoint_intent.description,
            )
            novel_station_key = station_key
            novel = await self._novel_station(
                story_plan,
                panorama,
                probes,
                anchor_map,
                shot_key,
                station_key=station_key,
            )
            if novel.decision.outcome == SelectionOutcome.SELECTED_COMPLIANT:
                try:
                    selected_novel_candidate = _selected_candidate(novel)
                    persisted_preflight = self._persisted_preflight(
                        selected_novel_candidate.candidate_key,
                        novel_station=True,
                    )
                    if isinstance(persisted_preflight, CandidatePreflightFailure):
                        novel_failure_kind = CameraConflictKind.INVALID_EVIDENCE
                        raise SpatialError(
                            f"{persisted_preflight.failure_code}: {persisted_preflight.correction}"
                        )
                    station_panorama_ref = _required_artifact(selected_novel_candidate)
                    station_panorama = validate_panorama_source(
                        self.run_store.artifacts,
                        station_panorama_ref,
                        scene_key=story_shot.scene_key,
                        story_plan_hash=story_plan.plan_hash,
                        reference_library_hash=panorama.reference_library_hash,
                    )
                    station_probes = await self._probe_set(story_plan, station_panorama)
                    station_anchor_map = (
                        await self._scene_anchor_map(
                            story_plan,
                            station_probes,
                            station_key=station_key,
                        )
                    ).anchor_map
                    translated_station = SourceStationSpec(
                        station_key=station_key,
                        zone_key=story_shot.spatial_intent.action_zone,
                        description=story_shot.spatial_intent.viewpoint_intent.description,
                        supported_viewpoint_intents=(
                            story_shot.spatial_intent.viewpoint_intent.intent_key,
                        ),
                    )
                    grounded_novel = await asyncio.to_thread(
                        self.strict_spatial_resolver.resolve_source,
                        shot=story_shot,
                        scene=scene,
                        panorama=station_panorama,
                        station=translated_station,
                        anchor_map=station_anchor_map,
                        allow_novel_station=False,
                        translated_station=True,
                    )
                    if grounded_novel.plan is None:
                        novel_failure_kind = (
                            grounded_novel.conflict.kind
                            if grounded_novel.conflict is not None
                            else None
                        )
                        raise SpatialError(
                            grounded_novel.conflict.detail
                            if grounded_novel.conflict is not None
                            else "novel station grounding did not produce an executable plan"
                        )
                    grounded_recipe = grounded_novel.plan.camera_recipe
                    if grounded_recipe.kind != "source_station":
                        raise SpatialError("novel station preflight produced an invalid recipe")
                    recipe = CameraSolver.materialize_novel_station(
                        source_panorama_hash=panorama.artifact_ref.sha256,
                        station_panorama=station_panorama_ref,
                        station_key=station_key,
                        station_description=(
                            story_shot.spatial_intent.viewpoint_intent.description
                        ),
                        aim_anchor=grounded_recipe.aim_anchor,
                        yaw_degrees=grounded_recipe.yaw_degrees,
                        pitch_degrees=grounded_recipe.pitch_degrees,
                        hfov_degrees=grounded_recipe.hfov_degrees,
                        vfov_degrees=grounded_recipe.vfov_degrees,
                        scene_view=grounded_recipe.scene_view,
                        decision=novel.decision,
                        candidates=novel.candidates,
                    )
                    plan = self.spatial_resolver.promote_novel(
                        shot=story_shot,
                        recipe=recipe,
                        novel_resolution=grounded_novel.plan,
                        prior_audit=resolution.audit,
                    )
                    selected_anchor_map = station_anchor_map
                except (ArtifactError, SpatialError, ValueError) as exc:
                    novel_failure_detail = str(exc)
                    plan = None
            else:
                novel_failure_detail = (
                    "novel station generation did not select a fully compliant candidate: "
                    f"{novel.decision.outcome.value}"
                )
        if plan is None:
            retry_audit = list(resolution.audit) if novel_station_key is not None else []
            if novel_failure_detail is not None:
                retry_audit.append(
                    SpatialAuditEntry(
                        sequence=len(retry_audit) + 1,
                        code="NOVEL_STATION_FAILED",
                        detail=novel_failure_detail[:2_000],
                        station_key=novel_station_key,
                        conflict_kind=novel_failure_kind,
                    )
                )
            final_source = await asyncio.to_thread(
                self.spatial_resolver.resolve_source,
                shot=story_shot,
                scene=scene,
                panorama=panorama,
                station=source_station,
                anchor_map=anchor_map,
                allow_novel_station=False,
                prior_audit=tuple(retry_audit),
            )
            plan = final_source.plan
            resolution = final_source
        if plan is None:
            conflict = resolution.conflict
            detail = conflict.detail if conflict is not None else "unknown spatial conflict"
            kind = conflict.kind.value if conflict is not None else "spatial_failed"
            raise SpatialError(f"{kind} for {shot_key}: {detail}")
        result = SpatialOperationResult(
            evidence=AnchorMapEvidence.from_anchor_map(selected_anchor_map),
            resolution=plan,
        )
        output = self._put(result)
        self._complete_step(
            step,
            output=output,
            selected_key=plan.resolution_hash,
        )
        self.stage_output.publish_spatial_resolution(story_plan, plan)
        return result

    async def _novel_station(
        self,
        story_plan: StoryPlan,
        panorama: PanoramaSource,
        probes: ProbeSet,
        grounding: SceneAnchorMap,
        shot_key: str,
        *,
        station_key: str,
    ) -> MediaOperationResult:
        view = self.execution.view_builder.novel_station(
            self.request,
            story_plan,
            probes,
            grounding,
            shot_key=shot_key,
            panorama=panorama.artifact_ref,
        )
        operation_key = stable_key("operation", shot_key, "novel_station")
        name = f"novel_station:{shot_key}"
        size = self._reference_image_size(panorama=True)
        key = self._key(
            "novel_station",
            (
                story_plan.plan_hash,
                probes.probe_set_hash,
                grounding.map_hash,
                shot_key,
                f"{size.width}x{size.height}",
            ),
            implementation=(
                f"{GROUNDING_VALIDATOR_VERSION}+{FOV_SOLVER_VERSION}+{SPATIAL_RESOLVER_VERSION}"
            ),
            renderer=PROMPT_RENDERER_VERSION,
            provider=self.capabilities.image.fingerprint,
        )
        reusable = self.run_store.reusable_step(self._state, name, key)
        if reusable and reusable.selected_output:
            result = self._load(reusable.selected_output, MediaOperationResult)
            if self._valid_image_result(
                result,
                expected_size=(size.width, size.height),
                require_panorama=True,
                validate_media_bundle=False,
            ):
                preflight = await self._preflight_novel_station_candidate(
                    story_plan,
                    panorama,
                    shot_key=shot_key,
                    station_key=station_key,
                    candidate=_selected_candidate(result),
                )
                if preflight.executable or (
                    self.config.generation.spatial_failure_policy == SpatialFailurePolicy.DEGRADE
                ):
                    self.stage_output.publish_novel_station(story_plan, shot_key, result)
                    return result
        step = self._begin_step(
            name,
            key,
            (
                story_plan.plan_hash,
                probes.probe_set_hash,
                grounding.map_hash,
                shot_key,
            ),
        )
        image_capability = self.capabilities.image.image
        if image_capability is None:
            raise WorkflowError("image capability is missing")
        attachment_count = 1 + len(view.probes)
        if attachment_count > image_capability.max_input_images:
            raise SpatialError(
                f"novel station needs {attachment_count} image inputs; "
                f"provider supports {image_capability.max_input_images}"
            )
        evaluation_view = EvaluationView(
            operation="novel_station",
            owner_shot_key=shot_key,
            media_instructions=(
                "Image 1 is the frozen canonical source-station panorama.",
                "Image 2 is the translated-station panorama candidate.",
            ),
            criteria=(
                EvaluationCriterionView(
                    criterion_id=stable_key("criterion", shot_key, "novel:scene"),
                    kind=CriterionKind.REQUIREMENT,
                    priority=95,
                    statement="The panorama preserves the canonical scene identity.",
                    phase=RequirementPhase.ALWAYS,
                    category="continuity",
                    owner_shot_key=shot_key,
                ),
                EvaluationCriterionView(
                    criterion_id=stable_key("criterion", shot_key, "novel:layout"),
                    kind=CriterionKind.REQUIREMENT,
                    priority=90,
                    statement=(
                        "The translated-station panorama keeps fixed landmarks unique and "
                        "the semantic layout coherent."
                    ),
                    phase=RequirementPhase.ALWAYS,
                    category="continuity",
                    owner_shot_key=shot_key,
                ),
                EvaluationCriterionView(
                    criterion_id=stable_key("criterion", shot_key, "novel:composition"),
                    kind=CriterionKind.REQUIREMENT,
                    priority=85,
                    statement=(
                        "The panorama provides enough coverage to later project this "
                        f"composition: {view.composition}."
                    ),
                    phase=RequirementPhase.ALWAYS,
                    category="composition",
                    owner_shot_key=shot_key,
                ),
                *(
                    EvaluationCriterionView(
                        criterion_id=stable_key(
                            "criterion",
                            shot_key,
                            f"novel:content:{index}",
                        ),
                        kind=CriterionKind.REQUIREMENT,
                        priority=92,
                        statement=(
                            "The panorama makes this required content visibly and uniquely "
                            f"locatable: {content}."
                        ),
                        phase=RequirementPhase.ALWAYS,
                        category="spatial_content",
                        owner_shot_key=shot_key,
                    )
                    for index, content in enumerate(view.required_content, start=1)
                ),
            ),
        )

        def request_factory(corrections: tuple[str, ...]) -> ImageRequest:
            rendered = render_novel_station(
                view,
                max_prompt_characters=image_capability.max_prompt_characters,
                corrections=corrections,
            )
            return ImageRequest(
                prompt=rendered.text,
                input_images=tuple(
                    Attachment(name=f"spatial_{index}", artifact_ref=reference)
                    for index, reference in enumerate(rendered.attachments)
                ),
                aspect_ratio=_aspect_ratio(size),
                resolution=size,
            )

        async def accept_candidate(
            candidate: CandidateRecord,
        ) -> PanoramaPreflightValue:
            return await self._preflight_novel_station_candidate(
                story_plan,
                panorama,
                shot_key=shot_key,
                station_key=station_key,
                candidate=candidate,
            )

        result, step = await self._image_operation(
            step,
            operation_key=operation_key,
            request_factory=request_factory,
            evaluation_view=evaluation_view,
            require_panorama=True,
            evaluation_reference_media=(
                CandidateMedia(
                    role="canonical_source_panorama",
                    artifact_ref=panorama.artifact_ref,
                ),
            ),
            candidate_media_role="translated_station_panorama",
            max_attempts=self.config.generation.novel_station_attempts,
            candidate_acceptance=accept_candidate,
            on_prompt=lambda prompt: self.stage_output.publish_spatial_prompt(
                story_plan,
                shot_key,
                prompt,
            ),
        )
        output = self._put(result)
        self._complete_step(
            step.model_copy(update={"attempt_records": result.attempts}),
            output=output,
            selected_key=result.decision.selected_candidate,
        )
        self.stage_output.publish_novel_station(story_plan, shot_key, result)
        return result

    async def _render_plan(
        self,
        story_plan: StoryPlan,
        library: ReferenceLibrary,
        spatial: tuple[ResolvedSpatialPlan, ...],
    ) -> RenderPlan:
        name = "render_plan"
        spatial_hash = canonical_hash(spatial)
        key = self._key(
            name,
            (
                story_plan.plan_hash,
                library.library_hash,
                spatial_hash,
                self.capabilities.fingerprint,
            ),
            implementation=RENDER_COMPILER_VERSION,
            provider=self.capabilities.fingerprint,
        )
        reusable = self.run_store.reusable_step(self._state, name, key)
        if reusable is not None and reusable.selected_output is not None:
            plan = self._load(reusable.selected_output, RenderPlan)
            try:
                plan.assert_hash()
                for item in plan.ordered_render_shots:
                    self.run_store.artifacts.verify(item.selected_scene_view)
                    self.run_store.artifacts.verify(item.selected_scene_panorama)
            except (ArtifactError, ValueError):
                reusable = None
        else:
            reusable = None
        if reusable is None:
            step = self._begin_step(
                name,
                key,
                (story_plan.plan_hash, library.library_hash, spatial_hash),
            )
            plan = self.execution.create_render_plan(
                self.request,
                story_plan,
                library,
                spatial,
            )
            output = self._put(plan)
            self._complete_step(step, output=output, selected_key=plan.render_plan_key)
        if self._state.status == ProjectStatus.REFERENCES_READY:
            artifact = self._find_step(name).selected_output
            if artifact is None:
                raise WorkflowError("completed render plan step has no output")
            self._state = self.run_store.transition(
                self._state,
                ProjectStatus.RENDER_PLANNED,
                active_render_plan=artifact,
                shots=tuple(
                    ShotRunState(shot_key=shot.shot_key) for shot in story_plan.ordered_shots
                ),
            )
        else:
            active_artifact = self._find_step(name).selected_output
            if active_artifact is not None and self._state.active_render_plan != active_artifact:
                self._save_state(active_render_plan=active_artifact)
        self.stage_output.publish_render_plan(story_plan, plan)
        self.stage_output.publish_audit(self._state)
        return plan

    async def _render_shots(
        self,
        story_plan: StoryPlan,
        render_plan: RenderPlan,
    ) -> tuple[tuple[MediaOperationResult, ...], tuple[MediaOperationResult, ...]]:
        frame_results: list[FirstFrameOperationResult] = []
        video_results: list[MediaOperationResult] = []
        previous_end_allowed: bool | None = None
        previous_tail = TailFrameInput(
            status=TailFrameStatus.NOT_REQUIRED,
            reason="the first shot has no previous video",
        )
        for shot_index, shot in enumerate(story_plan.ordered_shots):
            frame = await self._first_frame(
                story_plan,
                render_plan,
                shot.shot_key,
                previous_tail,
            )
            frame_results.append(frame)
            self._update_shot(
                shot.shot_key,
                status=ShotStatus.FRAME_SELECTED,
                selected_frame_candidate=frame.decision.selected_candidate,
                previous_end_frame_allowed=previous_end_allowed,
            )
            frame_candidate = _selected_candidate(frame)
            if frame_candidate.artifact_ref is None:
                raise ContractError("selected first frame has no artifact")
            video = await self._video_shot(
                story_plan,
                render_plan,
                shot.shot_key,
                frame_candidate.artifact_ref,
            )
            video_results.append(video)
            if shot_index + 1 < len(story_plan.ordered_shots):
                selected_video = _required_artifact(_selected_candidate(video))
                previous_tail = await asyncio.to_thread(
                    self.tail_extractor.extract,
                    self.run_store.artifacts,
                    selected_video,
                    expected_size=(
                        self.request.resolution.width,
                        self.request.resolution.height,
                    ),
                )
            conflicts = self._boundary_conflicts(render_plan, shot.shot_key, video)
            degraded = (
                frame.decision.outcome == SelectionOutcome.SELECTED_DEGRADED
                or video.decision.outcome == SelectionOutcome.SELECTED_DEGRADED
                or render_plan.ordered_render_shots[shot_index].spatial_status
                == SpatialResolutionStatus.DEGRADED
            )
            self._update_shot(
                shot.shot_key,
                status=(ShotStatus.COMPLETED_DEGRADED if degraded else ShotStatus.COMPLETED),
                selected_frame_candidate=frame.decision.selected_candidate,
                selected_video_candidate=video.decision.selected_candidate,
                degraded=degraded,
                boundary_conflicts=conflicts,
                previous_end_frame_allowed=previous_end_allowed,
            )
            previous_end_allowed = not conflicts
        if self._state.status == ProjectStatus.RENDER_PLANNED:
            self._state = self.run_store.transition(
                self._state,
                ProjectStatus.SHOTS_RENDERED,
            )
            self.stage_output.publish_audit(self._state)
        return tuple(frame_results), tuple(video_results)

    async def _first_frame(
        self,
        story_plan: StoryPlan,
        render_plan: RenderPlan,
        shot_key: str,
        previous_tail: TailFrameInput,
    ) -> FirstFrameOperationResult:
        view = self.execution.view_builder.first_frame(
            self.request,
            story_plan,
            render_plan,
            shot_key=shot_key,
        )
        evaluation_view = self.execution.view_builder.evaluation(
            render_plan,
            shot_key=shot_key,
            operation="first_frame",
            phases=(RequirementPhase.FIRST_FRAME,),
        )
        operation_key = stable_key("operation", shot_key, "first_frame")
        name = f"first_frame:{shot_key}"
        tail_dependency = (
            previous_tail.frame.sha256
            if previous_tail.frame is not None
            else stable_key(
                "tail_input",
                previous_tail.status.value,
                (
                    previous_tail.source_video.sha256
                    if previous_tail.source_video is not None
                    else "none"
                ),
            )
        )
        inputs = (
            render_plan.render_plan_hash,
            shot_key,
            tail_dependency,
            (
                self.capabilities.planner.fingerprint
                if previous_tail.status == TailFrameStatus.AVAILABLE
                else "planner_not_used"
            ),
        )
        key = self._key(
            "first_frame",
            inputs,
            renderer=PROMPT_RENDERER_VERSION,
            provider=self.capabilities.image.fingerprint,
        )
        reusable = self.run_store.reusable_step(self._state, name, key)
        if reusable and reusable.selected_output:
            result = self._load(reusable.selected_output, FirstFrameOperationResult)
            if self._valid_image_result(
                result,
                expected_size=(
                    self.request.resolution.width,
                    self.request.resolution.height,
                ),
            ):
                self.stage_output.publish_first_frame(story_plan, shot_key, result)
                return result
        step = self._begin_step(name, key, inputs)
        image_capability = self.capabilities.image.image
        if image_capability is None:
            raise WorkflowError("image capability is missing")
        participation, planner_prompt = await self._first_frame_participation(
            story_plan,
            render_plan,
            shot_key,
            previous_tail,
            reference_executable=(
                3 + len(view.reference_images) <= image_capability.max_input_images
            ),
        )
        effective_mode = participation.initial_mode
        reuse_evaluation: EvaluationReport | None = None
        reuse_prompt: ProviderPrompt | None = None
        base_result: MediaOperationResult | None = None
        guidance: tuple[FirstFrameGuidance, ...] = ()

        if effective_mode == FirstFrameMode.REUSE:
            tail_frame = participation.tail.frame
            if tail_frame is None:
                raise WorkflowError("reuse resolution has no available tail frame")
            base_result, reuse_prompt = await self._reuse_first_frame(
                operation_key,
                tail_frame,
                evaluation_view,
            )
            reuse_evaluation = _selected_report(
                base_result.decision,
                base_result.reports,
            )
            if base_result.decision.outcome != SelectionOutcome.SELECTED_COMPLIANT:
                assessment = participation.assessment
                if assessment is None:
                    raise WorkflowError("reuse resolution has no participation assessment")
                guidance = reference_guidance(assessment, reuse_evaluation)
                effective_mode = (
                    FirstFrameMode.REFERENCE
                    if has_preserve_guidance(guidance)
                    and 3 + len(view.reference_images) <= image_capability.max_input_images
                    else FirstFrameMode.FRESH
                )
                base_result = None

        if base_result is None:
            if effective_mode == FirstFrameMode.REFERENCE:
                assessment = participation.assessment
                tail_frame = participation.tail.frame
                if assessment is None or tail_frame is None:
                    raise WorkflowError("reference resolution lacks assessment or tail")
                if not guidance:
                    guidance = reference_guidance(assessment)

            def request_factory(corrections: tuple[str, ...]) -> ImageRequest:
                if effective_mode == FirstFrameMode.REFERENCE:
                    tail_frame = participation.tail.frame
                    if tail_frame is None:
                        raise WorkflowError("reference request lacks a tail frame")
                    rendered = render_reference_first_frame(
                        view,
                        previous_tail=tail_frame,
                        guidance=guidance,
                        max_prompt_characters=image_capability.max_prompt_characters,
                        corrections=corrections,
                    )
                else:
                    rendered = render_first_frame(
                        view,
                        max_prompt_characters=image_capability.max_prompt_characters,
                        corrections=corrections,
                    )
                attachment_names = ["scene_view", "scene_panorama"]
                if effective_mode == FirstFrameMode.REFERENCE:
                    attachment_names.append("previous_tail")
                attachment_names.extend(
                    f"subject_{subject_index}_{image.role}"
                    for subject_index, subject in enumerate(
                        view.subject_references,
                        start=1,
                    )
                    for image in subject.images
                )
                if len(attachment_names) != len(rendered.attachments):
                    raise WorkflowError("first-frame attachment naming is inconsistent")
                attachments = tuple(
                    Attachment(name=name, artifact_ref=reference)
                    for name, reference in zip(
                        attachment_names,
                        rendered.attachments,
                        strict=True,
                    )
                )
                return ImageRequest(
                    prompt=rendered.text,
                    input_images=attachments,
                    aspect_ratio=_aspect_ratio(self.request.resolution),
                    resolution=self.request.resolution,
                )

            base_result, step = await self._image_operation(
                step,
                operation_key=operation_key,
                request_factory=request_factory,
                evaluation_view=evaluation_view,
                on_prompt=lambda prompt: self.stage_output.publish_first_frame_prompt(
                    story_plan,
                    shot_key,
                    prompt,
                ),
            )

        result = _first_frame_result(
            base_result,
            participation=participation,
            effective_mode=effective_mode,
            reuse_evaluation=reuse_evaluation,
            planner_prompt=planner_prompt,
            reuse_prompt=reuse_prompt,
        )
        output = self._put(result)
        self._complete_step(
            step.model_copy(update={"attempt_records": result.attempts}),
            output=output,
            selected_key=result.decision.selected_candidate,
        )
        self.stage_output.publish_first_frame(story_plan, shot_key, result)
        return result

    async def _first_frame_participation(
        self,
        story_plan: StoryPlan,
        render_plan: RenderPlan,
        shot_key: str,
        previous_tail: TailFrameInput,
        *,
        reference_executable: bool,
    ) -> tuple[FirstFrameParticipationResult, ProviderPrompt | None]:
        if previous_tail.status != TailFrameStatus.AVAILABLE:
            return (
                resolve_first_frame_participation(
                    tail=previous_tail,
                    assessment=None,
                    reference_executable=False,
                ),
                None,
            )
        if previous_tail.frame is None:
            raise WorkflowError("available tail has no frame")
        try:
            view = self.execution.view_builder.first_frame_participation(
                story_plan,
                render_plan,
                shot_key=shot_key,
                previous_tail=previous_tail.frame,
            )
            (
                assessment,
                metrics,
                prompt,
                error,
            ) = await self.execution.assess_first_frame_participation(view)
        except (ArtifactError, ContractError, ProviderError, ValueError) as exc:
            call_ref = (
                stable_key("planner_error", shot_key, exc.kind.value)
                if isinstance(exc, ProviderError)
                else None
            )
            return (
                resolve_first_frame_participation(
                    tail=previous_tail,
                    assessment=None,
                    reference_executable=reference_executable,
                    unavailable_reason=f"planner assessment unavailable: {exc}",
                    planner_call_ref=call_ref,
                    planner_usage=ProviderUsage(
                        call_count=1 if isinstance(exc, ProviderError) else 0
                    ),
                ),
                None,
            )
        usage = _provider_usage(metrics)
        return (
            resolve_first_frame_participation(
                tail=previous_tail,
                assessment=assessment,
                reference_executable=reference_executable,
                unavailable_reason=(
                    f"planner assessment invalid: {error}" if assessment is None else None
                ),
                planner_call_ref=metrics.call_ref,
                planner_usage=usage,
            ),
            prompt,
        )

    async def _reuse_first_frame(
        self,
        operation_key: str,
        tail_frame: ArtifactRef,
        evaluation_view: EvaluationView,
    ) -> tuple[MediaOperationResult, ProviderPrompt]:
        candidate = CandidateRecord(
            candidate_key=stable_key(
                "candidate",
                operation_key,
                f"reuse:{tail_frame.sha256}",
            ),
            operation_key=operation_key,
            artifact_ref=tail_frame,
            technical_status=TechnicalStatus.VALID,
            technical_findings=(
                TechnicalFinding(
                    code="reused_previous_tail",
                    passed=True,
                    detail=tail_frame.sha256,
                ),
            ),
            technical_quality=1.0,
            provider_call_ref=stable_key("artifact_source", tail_frame.sha256, "reuse"),
            provider_usage=ProviderUsage(call_count=0),
            logical_attempt=1,
        )
        evaluation_prompt = self.evaluation.render_prompt(evaluation_view)
        prompt = _judge_prompt_record(
            candidate,
            evaluation_prompt,
            logical_attempt=1,
            candidate_index=1,
        )
        started = utc_now()
        report = await self._evaluate(
            candidate,
            evaluation_view,
            prompt=evaluation_prompt,
        )
        decision = self.execution.select(
            operation_key,
            (candidate,),
            (report,),
        )
        return (
            MediaOperationResult(
                operation_key=operation_key,
                candidates=(candidate,),
                reports=(report,),
                decision=decision,
                attempts=(
                    _attempt_record(
                        1,
                        started,
                        (candidate,),
                        (report,),
                    ),
                ),
                prompts=(prompt,),
            ),
            prompt,
        )

    async def _image_operation(
        self,
        step: StepRecord,
        *,
        operation_key: str,
        request_factory: ImageRequestFactory,
        evaluation_view: EvaluationView,
        require_panorama: bool = False,
        evaluation_reference_media: tuple[CandidateMedia, ...] = (),
        candidate_media_role: str | None = None,
        on_prompt: Callable[[ProviderPrompt], None] | None = None,
        max_attempts: int | None = None,
        candidate_acceptance: ImageCandidateAcceptance | None = None,
    ) -> tuple[MediaOperationResult, StepRecord]:
        if bool(evaluation_reference_media) != (candidate_media_role is not None):
            raise WorkflowError(
                "evaluation reference media and candidate media role must be configured together"
            )
        progress = self._load_progress(step, operation_key)
        candidates = list({item.candidate_key: item for item in progress.candidates}.values())
        reports = list({item.candidate_key: item for item in progress.reports}.values())
        attempts = list(progress.attempts)
        prompts = list(progress.prompts)
        ledger = self._ledger(reports, evaluation_view)
        accepted_candidate_keys: set[str] = set()
        preflighted_candidate_keys: set[str] = set()
        preflight_failures: dict[str, CandidatePreflightFailure] = {}
        spatial_corrections: list[str] = []
        if candidate_acceptance is not None:
            reports_by_candidate = {item.candidate_key: item for item in reports}
            for candidate in sorted(candidates, key=lambda item: item.candidate_key):
                report = reports_by_candidate.get(candidate.candidate_key)
                if report is None or not self.execution.should_stop_early((report,)):
                    continue
                outcome = await candidate_acceptance(candidate)
                preflighted_candidate_keys.add(candidate.candidate_key)
                if outcome.executable:
                    accepted_candidate_keys.add(candidate.candidate_key)
                elif outcome.correction:
                    spatial_corrections.append(outcome.correction)
                if isinstance(outcome, CandidatePreflightFailure):
                    preflight_failures[candidate.candidate_key] = outcome
        completed_attempts = {item.logical_attempt for item in attempts}
        attempt_limit = self.config.generation.attempts if max_attempts is None else max_attempts
        for attempt in range(1, attempt_limit + 1):
            if accepted_candidate_keys:
                break
            if attempt in completed_attempts:
                continue
            started = utc_now()
            request = request_factory(
                tuple(dict.fromkeys((*ledger.corrections(), *spatial_corrections)))
            )
            image_capability = self.capabilities.image.image
            if image_capability is None:
                raise WorkflowError("image capability is missing")
            if len(request.prompt) > image_capability.max_prompt_characters:
                raise PlanComplexityError("canonical image prompt exceeds provider context")
            if len(request.input_images) > image_capability.max_input_images:
                raise WorkflowError("image operation exceeds declared input channels")
            candidate_keys: list[str] = []
            missing_candidates: list[tuple[int, str]] = []
            for index in range(1, self.config.generation.candidates_per_attempt + 1):
                candidate_key = stable_key(
                    "candidate",
                    operation_key,
                    f"{attempt}:{index}",
                )
                prompt_record = _image_prompt_record(
                    request,
                    logical_attempt=attempt,
                    candidate_index=index,
                    candidate_key=candidate_key,
                )
                candidate_keys.append(candidate_key)
                if _record_prompt(prompts, prompt_record) and on_prompt is not None:
                    on_prompt(prompt_record)
                if not any(item.candidate_key == candidate_key for item in candidates):
                    missing_candidates.append((index, candidate_key))
            progress = MediaOperationProgress(
                operation_key=operation_key,
                candidates=tuple(candidates),
                reports=tuple(reports),
                attempts=tuple(attempts),
                prompts=tuple(prompts),
            )
            step = self._save_progress(step, progress)
            if missing_candidates:
                produced = await asyncio.gather(
                    *(
                        self._produce_image_safe(
                            operation_key,
                            request,
                            attempt,
                            index,
                            require_panorama,
                        )
                        for index, _candidate_key in missing_candidates
                    )
                )
                generated = tuple(
                    _candidate_with_evaluation_media(
                        candidate,
                        reference_media=evaluation_reference_media,
                        candidate_role=candidate_media_role,
                    )
                    for candidate in produced
                )
                candidates.extend(generated)
                progress = MediaOperationProgress(
                    operation_key=operation_key,
                    candidates=tuple(candidates),
                    reports=tuple(reports),
                    attempts=tuple(attempts),
                    prompts=tuple(prompts),
                )
                step = self._save_progress(step, progress)
            candidates_by_key = {item.candidate_key: item for item in candidates}
            attempt_candidates = tuple(
                candidates_by_key[candidate_key] for candidate_key in candidate_keys
            )
            valid = tuple(
                item
                for item in attempt_candidates
                if item.technical_status == TechnicalStatus.VALID
            )
            positions = {
                candidate_key: index for index, candidate_key in enumerate(candidate_keys, start=1)
            }
            evaluation_prompt = self.evaluation.render_prompt(evaluation_view)
            reports_by_candidate = {item.candidate_key: item for item in reports}
            missing_reports: list[CandidateRecord] = []
            for candidate in valid:
                if candidate.candidate_key in reports_by_candidate:
                    continue
                prompt_record = _judge_prompt_record(
                    candidate,
                    evaluation_prompt,
                    logical_attempt=attempt,
                    candidate_index=positions[candidate.candidate_key],
                )
                if _record_prompt(prompts, prompt_record) and on_prompt is not None:
                    on_prompt(prompt_record)
                missing_reports.append(candidate)
            progress = MediaOperationProgress(
                operation_key=operation_key,
                candidates=tuple(candidates),
                reports=tuple(reports),
                attempts=tuple(attempts),
                prompts=tuple(prompts),
            )
            step = self._save_progress(step, progress)
            if missing_reports:
                reports.extend(
                    await asyncio.gather(
                        *(
                            self._evaluate(
                                candidate,
                                evaluation_view,
                                prompt=evaluation_prompt,
                            )
                            for candidate in missing_reports
                        )
                    )
                )
                progress = MediaOperationProgress(
                    operation_key=operation_key,
                    candidates=tuple(candidates),
                    reports=tuple(reports),
                    attempts=tuple(attempts),
                    prompts=tuple(prompts),
                )
                step = self._save_progress(step, progress)
            reports_by_candidate = {item.candidate_key: item for item in reports}
            attempt_reports = tuple(
                reports_by_candidate[candidate.candidate_key]
                for candidate in valid
                if candidate.candidate_key in reports_by_candidate
            )
            attempt_record = _attempt_record(
                attempt,
                started,
                attempt_candidates,
                attempt_reports,
            )
            attempts.append(attempt_record)
            progress = MediaOperationProgress(
                operation_key=operation_key,
                candidates=tuple(candidates),
                reports=tuple(reports),
                attempts=tuple(attempts),
                prompts=tuple(prompts),
            )
            step = self._save_progress(step, progress)
            if attempt_reports and self.execution.should_stop_early(attempt_reports):
                if candidate_acceptance is None:
                    break
                reports_by_key = {item.candidate_key: item for item in attempt_reports}
                for candidate in valid:
                    report = reports_by_key.get(candidate.candidate_key)
                    if report is None or not self.execution.should_stop_early((report,)):
                        continue
                    outcome = await candidate_acceptance(candidate)
                    preflighted_candidate_keys.add(candidate.candidate_key)
                    if outcome.executable:
                        accepted_candidate_keys.add(candidate.candidate_key)
                    elif outcome.correction:
                        spatial_corrections.append(outcome.correction)
                    if isinstance(outcome, CandidatePreflightFailure):
                        preflight_failures[candidate.candidate_key] = outcome
                if accepted_candidate_keys:
                    break
            if attempt_reports:
                current = self.execution.select(
                    operation_key,
                    attempt_candidates,
                    attempt_reports,
                )
                ledger = ledger.replace_from_report(
                    _selected_report(current, attempt_reports),
                    evaluation_view,
                )
        if (
            candidate_acceptance is not None
            and not accepted_candidate_keys
            and spatial_corrections
            and self.config.generation.spatial_failure_policy == SpatialFailurePolicy.STRICT
        ):
            detail = "; ".join(dict.fromkeys(spatial_corrections))
            raise CandidatePreflightExhausted(
                operation_key=operation_key,
                candidate_keys=tuple(sorted(preflighted_candidate_keys)),
                failure_codes=tuple(
                    dict.fromkeys(
                        item.failure_code
                        for item in sorted(
                            preflight_failures.values(),
                            key=lambda item: item.candidate_key,
                        )
                    )
                ),
                detail=detail[:2_000],
            )
        eligible_candidate_keys = accepted_candidate_keys or preflighted_candidate_keys
        selectable_candidates = tuple(
            item
            for item in candidates
            if not eligible_candidate_keys or item.candidate_key in eligible_candidate_keys
        )
        selectable_keys = {item.candidate_key for item in selectable_candidates}
        selectable_reports = tuple(
            item for item in reports if item.candidate_key in selectable_keys
        )
        decision = self.execution.select(
            operation_key,
            selectable_candidates,
            selectable_reports,
        )
        return (
            MediaOperationResult(
                operation_key=operation_key,
                candidates=tuple(candidates),
                reports=tuple(reports),
                decision=decision,
                attempts=tuple(attempts),
                prompts=tuple(prompts),
            ),
            step,
        )

    async def _video_shot(
        self,
        story_plan: StoryPlan,
        render_plan: RenderPlan,
        shot_key: str,
        start_image: ArtifactRef,
    ) -> MediaOperationResult:
        view = self.execution.view_builder.video_motion(
            self.request,
            story_plan,
            render_plan,
            shot_key=shot_key,
            start_image=start_image,
        )
        evaluation_view = self.execution.view_builder.evaluation(
            render_plan,
            shot_key=shot_key,
            operation="video",
            phases=(RequirementPhase.MOTION, RequirementPhase.END),
        )
        operation_key = stable_key("operation", shot_key, "video")
        name = f"video:{shot_key}"
        key = self._key(
            "video",
            (render_plan.render_plan_hash, shot_key, start_image.sha256),
            renderer=PROMPT_RENDERER_VERSION,
            provider=self.capabilities.video.fingerprint,
        )
        reusable = self.run_store.reusable_step(self._state, name, key)
        if reusable and reusable.selected_output:
            result = self._load(reusable.selected_output, MediaOperationResult)
            if self._valid_video_result(
                result,
                view.duration,
                view.fps,
                start_image,
            ):
                self.stage_output.publish_video(
                    story_plan,
                    shot_key,
                    start_image,
                    result,
                )
                return result
        step = self._begin_step(
            name,
            key,
            (render_plan.render_plan_hash, shot_key, start_image.sha256),
        )
        video_capability = self.capabilities.video.video
        if video_capability is None:
            raise WorkflowError("video capability is missing")
        progress = self._load_progress(step, operation_key)
        candidates = list(progress.candidates)
        reports = list(progress.reports)
        attempts = list(progress.attempts)
        prompts = list(progress.prompts)
        ledger = self._ledger(reports, evaluation_view)
        completed_attempts = {item.logical_attempt for item in attempts}
        for attempt in range(1, self.config.generation.attempts + 1):
            if attempt in completed_attempts:
                continue
            started = utc_now()
            rendered = render_video(
                view,
                max_prompt_characters=video_capability.max_prompt_characters,
                corrections=ledger.corrections(),
            )
            request = VideoRequest(
                prompt=rendered.text,
                start_image=Attachment(name="start_image", artifact_ref=start_image),
                reference_images=tuple(
                    Attachment(name=f"reference_{index}", artifact_ref=reference)
                    for index, reference in enumerate(view.reference_images)
                ),
                duration=view.duration,
                resolution=view.resolution,
                fps=view.fps,
            )
            attempt_candidates: list[CandidateRecord] = []
            attempt_reports: list[EvaluationReport] = []
            for index in range(1, self.config.generation.candidates_per_attempt + 1):
                candidate_key = stable_key(
                    "candidate",
                    operation_key,
                    f"{attempt}:{index}",
                )
                existing = next(
                    (item for item in candidates if item.candidate_key == candidate_key),
                    None,
                )
                generation_prompt = _video_prompt_record(
                    request,
                    logical_attempt=attempt,
                    candidate_index=index,
                    candidate_key=candidate_key,
                )
                if _record_prompt(prompts, generation_prompt):
                    self.stage_output.publish_video_prompt(
                        story_plan,
                        shot_key,
                        generation_prompt,
                    )
                    progress = MediaOperationProgress(
                        operation_key=operation_key,
                        candidates=tuple(candidates),
                        reports=tuple(reports),
                        attempts=tuple(attempts),
                        prompts=tuple(prompts),
                    )
                    step = self._save_progress(step, progress)
                if existing is None:
                    resume_job = (
                        step.pending_provider_job
                        if step.pending_logical_attempt == attempt
                        and step.pending_candidate_index == index
                        else None
                    )

                    def save_job(
                        job_id: str,
                        usage: ProviderUsage,
                        call_refs: tuple[str, ...],
                        *,
                        logical_attempt: int = attempt,
                        candidate_index: int = index,
                    ) -> None:
                        nonlocal step
                        step = step.model_copy(
                            update={
                                "pending_provider_job": job_id,
                                "pending_logical_attempt": logical_attempt,
                                "pending_candidate_index": candidate_index,
                                "pending_provider_usage": usage,
                                "pending_provider_call_refs": call_refs,
                            }
                        )
                        self._state = self.run_store.upsert_step(self._state, step)

                    try:
                        existing = await self.assets.produce_video(
                            operation_key=operation_key,
                            request=request,
                            logical_attempt=attempt,
                            candidate_index=index,
                            on_job_progress=save_job,
                            resume_job_id=resume_job,
                            prior_usage=(
                                step.pending_provider_usage
                                if resume_job is not None
                                and step.pending_provider_usage is not None
                                else ProviderUsage()
                            ),
                            prior_call_refs=(
                                step.pending_provider_call_refs if resume_job is not None else ()
                            ),
                        )
                    except ProviderError as exc:
                        if exc.retryable:
                            raise
                        prior_usage = step.pending_provider_usage or ProviderUsage()
                        error_usage = ProviderUsage.combine(
                            (prior_usage, ProviderUsage(call_count=1))
                        )
                        existing = _invalid_candidate(
                            operation_key,
                            attempt,
                            index,
                            str(exc),
                            provider_usage=error_usage,
                            provider_call_refs=step.pending_provider_call_refs,
                        )
                    step = step.model_copy(
                        update={
                            "pending_provider_job": None,
                            "pending_logical_attempt": None,
                            "pending_candidate_index": None,
                            "pending_provider_usage": None,
                            "pending_provider_call_refs": (),
                        }
                    )
                    self._state = self.run_store.upsert_step(self._state, step)
                    candidates.append(existing)
                    progress = MediaOperationProgress(
                        operation_key=operation_key,
                        candidates=tuple(candidates),
                        reports=tuple(reports),
                        attempts=tuple(attempts),
                        prompts=tuple(prompts),
                    )
                    step = self._save_progress(step, progress)
                if existing is None:
                    raise WorkflowError("video candidate was not persisted")
                found_report = next(
                    (item for item in reports if item.candidate_key == existing.candidate_key),
                    None,
                )
                if existing.technical_status == TechnicalStatus.VALID and found_report is None:
                    evaluation_prompt = self.evaluation.render_prompt(evaluation_view)
                    evaluation_record = _judge_prompt_record(
                        existing,
                        evaluation_prompt,
                        logical_attempt=attempt,
                        candidate_index=index,
                    )
                    if _record_prompt(prompts, evaluation_record):
                        self.stage_output.publish_video_prompt(
                            story_plan,
                            shot_key,
                            evaluation_record,
                        )
                        progress = MediaOperationProgress(
                            operation_key=operation_key,
                            candidates=tuple(candidates),
                            reports=tuple(reports),
                            attempts=tuple(attempts),
                            prompts=tuple(prompts),
                        )
                        step = self._save_progress(step, progress)
                    found_report = await self._evaluate(
                        existing,
                        evaluation_view,
                        prompt=evaluation_prompt,
                    )
                    reports.append(found_report)
                    progress = MediaOperationProgress(
                        operation_key=operation_key,
                        candidates=tuple(candidates),
                        reports=tuple(reports),
                        attempts=tuple(attempts),
                        prompts=tuple(prompts),
                    )
                    step = self._save_progress(step, progress)
                attempt_candidates.append(existing)
                if found_report is not None:
                    attempt_reports.append(found_report)
            record = _attempt_record(
                attempt,
                started,
                tuple(attempt_candidates),
                tuple(attempt_reports),
            )
            attempts.append(record)
            progress = MediaOperationProgress(
                operation_key=operation_key,
                candidates=tuple(candidates),
                reports=tuple(reports),
                attempts=tuple(attempts),
                prompts=tuple(prompts),
            )
            step = self._save_progress(step, progress)
            if attempt_reports and self.execution.should_stop_early(tuple(attempt_reports)):
                break
            if attempt_reports:
                current = self.execution.select(
                    operation_key,
                    tuple(attempt_candidates),
                    tuple(attempt_reports),
                )
                ledger = ledger.replace_from_report(
                    _selected_report(current, tuple(attempt_reports)),
                    evaluation_view,
                )
        decision = self.execution.select(
            operation_key,
            tuple(candidates),
            tuple(reports),
        )
        result = MediaOperationResult(
            operation_key=operation_key,
            candidates=tuple(candidates),
            reports=tuple(reports),
            decision=decision,
            attempts=tuple(attempts),
            prompts=tuple(prompts),
        )
        output = self._put(result)
        self._complete_step(
            step.model_copy(update={"attempt_records": result.attempts}),
            output=output,
            selected_key=decision.selected_candidate,
        )
        self.stage_output.publish_video(
            story_plan,
            shot_key,
            start_image,
            result,
        )
        return result

    async def _produce_image_safe(
        self,
        operation_key: str,
        request: ImageRequest,
        attempt: int,
        index: int,
        require_panorama: bool,
    ) -> CandidateRecord:
        try:
            return await self.assets.produce_image(
                operation_key=operation_key,
                request=request,
                logical_attempt=attempt,
                candidate_index=index,
                require_panorama=require_panorama,
            )
        except ProviderError as exc:
            if exc.retryable:
                raise
            return _invalid_candidate(operation_key, attempt, index, str(exc))

    async def _evaluate(
        self,
        candidate: CandidateRecord,
        view: EvaluationView,
        *,
        prompt: str | None = None,
    ) -> EvaluationReport:
        try:
            return await self.evaluation.evaluate(candidate, view, prompt=prompt)
        except ProviderError as exc:
            if exc.retryable:
                raise
            return self.evaluation.unknown_report(
                candidate,
                view,
                evaluator_ref=stable_key(
                    "evaluator_error",
                    candidate.candidate_key,
                    exc.kind.value,
                ),
                evidence=_evaluation_error_evidence(exc),
                evaluator_usage=ProviderUsage(call_count=1),
            )
        except ContractError as exc:
            return self.evaluation.unknown_report(
                candidate,
                view,
                evaluator_ref=stable_key(
                    "evaluator_error",
                    candidate.candidate_key,
                    type(exc).__name__,
                ),
                evidence=_evaluation_error_evidence(exc),
            )

    async def _assemble(
        self,
        story_plan: StoryPlan,
        videos: tuple[MediaOperationResult, ...],
    ) -> AssemblyResult:
        selected = tuple(_required_artifact(_selected_candidate(item)) for item in videos)
        duration = float(sum(shot.duration for shot in story_plan.ordered_shots))
        fps = self.request.generation_requirements.delivery_requirements.video_fps
        name = "assembly"
        key = self._key(
            name,
            tuple(item.sha256 for item in selected),
            implementation=WORKFLOW_VERSION,
            behavior={
                "resolution": self.request.resolution,
                "fps": fps,
                "duration": duration,
            },
        )
        reusable = self.run_store.reusable_step(self._state, name, key)
        if reusable and reusable.selected_output:
            validation = VideoTechnicalValidator().validate(
                self.run_store.artifacts,
                reusable.selected_output,
                expected_duration=duration,
                expected_resolution=(
                    self.request.resolution.width,
                    self.request.resolution.height,
                ),
                expected_fps=fps,
            )
            if validation.info is not None and validation.status == TechnicalStatus.VALID:
                result = AssemblyResult(
                    artifact_ref=reusable.selected_output,
                    technical_info=validation.info,
                )
            else:
                reusable = None
        if reusable is None:
            step = self._begin_step(
                name,
                key,
                tuple(item.sha256 for item in selected),
            )
            async with self.task_pool.slot():
                result = await asyncio.to_thread(
                    self.assembler.assemble,
                    self.run_store.artifacts,
                    selected,
                    resolution=self.request.resolution,
                    fps=fps,
                    expected_duration=duration,
                )
            self._complete_step(
                step,
                output=result.artifact_ref,
                selected_key=result.artifact_ref.sha256,
            )
        if self._state.status == ProjectStatus.SHOTS_RENDERED:
            self._state = self.run_store.transition(
                self._state,
                ProjectStatus.ASSEMBLED,
                final_video=result.artifact_ref,
            )
        return result

    async def _deliver(
        self,
        story_plan: StoryPlan,
        references: ReferenceLibrary,
        render_plan: RenderPlan,
        reference_results: tuple[ReferenceOperationResult, ...],
        frames: tuple[MediaOperationResult, ...],
        videos: tuple[MediaOperationResult, ...],
        assembly: AssemblyResult,
    ) -> None:
        if self._state.status not in {
            ProjectStatus.ASSEMBLED,
            ProjectStatus.DELIVERED,
            ProjectStatus.DELIVERED_DEGRADED,
        }:
            raise WorkflowError("delivery requires assembled state")
        shot_records: list[ShotDeliveryRecord] = []
        degraded = any(
            item.decision.outcome == SelectionOutcome.SELECTED_DEGRADED
            for item in (*reference_results, *frames, *videos)
        ) or any(
            item.spatial_status == SpatialResolutionStatus.DEGRADED
            for item in render_plan.ordered_render_shots
        )
        previous_end_allowed: bool | None = None
        for shot, frame, video in zip(
            story_plan.ordered_shots,
            frames,
            videos,
            strict=True,
        ):
            render_shot = next(
                item for item in render_plan.ordered_render_shots if item.shot_key == shot.shot_key
            )
            report = _selected_report(video.decision, video.reports)
            frame_report = _selected_report(frame.decision, frame.reports)
            summary = _requirement_summary((frame_report, report))
            conflicts = self._boundary_conflicts(render_plan, shot.shot_key, video)
            shot_records.append(
                ShotDeliveryRecord(
                    shot_key=shot.shot_key,
                    selected_frame_candidate=frame.decision.selected_candidate,
                    selected_video_candidate=video.decision.selected_candidate,
                    selection_outcome=(
                        SelectionOutcome.SELECTED_DEGRADED
                        if frame.decision.outcome == SelectionOutcome.SELECTED_DEGRADED
                        or video.decision.outcome == SelectionOutcome.SELECTED_DEGRADED
                        or render_shot.spatial_status == SpatialResolutionStatus.DEGRADED
                        else SelectionOutcome.SELECTED_COMPLIANT
                    ),
                    spatial_status=render_shot.spatial_status,
                    spatial_resolution_hash=render_shot.spatial_resolution_hash,
                    spatial_failure_codes=render_shot.spatial_failure_codes,
                    dropped_spatial_content=render_shot.dropped_spatial_content,
                    fallback_camera_recipe=(
                        render_shot.camera_recipe
                        if render_shot.spatial_status == SpatialResolutionStatus.DEGRADED
                        else None
                    ),
                    evaluation_refs=(
                        frame.decision.selected_report,
                        video.decision.selected_report,
                    ),
                    criterion_summary=summary,
                    video_technical_findings=_selected_candidate(video).technical_findings,
                    boundary_conflicts=conflicts,
                    previous_end_frame_allowed=previous_end_allowed,
                )
            )
            previous_end_allowed = not conflicts
        delivery_status = (
            DeliveryStatus.DELIVERED_DEGRADED if degraded else DeliveryStatus.DELIVERED
        )
        providers = tuple(
            ProviderRecord(
                role=role,
                adapter=profile.identity.adapter,
                model=profile.identity.model,
                endpoint=profile.identity.endpoint,
                capability_fingerprint=profile.fingerprint,
            )
            for role, profile in (
                ("planner", self.capabilities.planner),
                ("image", self.capabilities.image),
                ("video", self.capabilities.video),
                ("judge", self.capabilities.judge),
            )
        )
        persisted_attempts = tuple(
            attempt for step in self._state.steps for attempt in step.attempt_records
        )
        usage = UsageRecord(
            logical_attempts=len(persisted_attempts),
            provider_calls=4
            + sum(attempt.provider_usage.call_count for attempt in persisted_attempts),
            transport_retries=sum(
                attempt.provider_usage.transport_retries for attempt in persisted_attempts
            ),
            elapsed_seconds=max(0.0, time.monotonic() - self._started),
            known_cost_usd=_known_cost(persisted_attempts),
        )
        provenance = tuple(
            ProvenanceRecord(
                record_key=step.step_name,
                artifact_ref=step.selected_output,
                parents=step.input_refs,
            )
            for step in self._state.steps
            if step.status == StepStatus.COMPLETED and step.selected_output is not None
        )
        payload = {
            "manifest_version": MANIFEST_VERSION,
            "run_id": self._state.run_id,
            "delivery_status": delivery_status,
            "story_plan_hash": story_plan.plan_hash,
            "reference_library_hash": references.library_hash,
            "render_plan_hash": render_plan.render_plan_hash,
            "final_media": FinalMediaRecord(
                artifact_ref=assembly.artifact_ref,
                duration_seconds=assembly.technical_info.duration_seconds,
                resolution=PixelSize(
                    width=assembly.technical_info.width,
                    height=assembly.technical_info.height,
                ),
                fps=assembly.technical_info.fps,
            ),
            "shots": tuple(shot_records),
            "requirement_summary": _requirement_summary(
                tuple(
                    _selected_report(item.decision, item.reports)
                    for item in (*reference_results, *frames, *videos)
                )
            ),
            "providers": providers,
            "usage": usage,
            "plan_revisions": (),
            "provenance": provenance,
            "created_at": utc_now(),
        }
        manifest = DeliveryManifest.model_validate(
            {
                **payload,
                "manifest_hash": canonical_hash(payload),
            }
        )
        manifest.assert_hash()
        manifest_ref = self.run_store.artifacts.put_bytes(
            canonical_bytes(manifest) + b"\n",
            "application/json",
        )
        self.stage_output.publish_final(self._state, story_plan, manifest)
        if self._state.status == ProjectStatus.ASSEMBLED:
            self._state = self.run_store.transition(
                self._state,
                (ProjectStatus.DELIVERED_DEGRADED if degraded else ProjectStatus.DELIVERED),
                manifest=manifest_ref,
            )
        self.stage_output.publish_audit(self._state, manifest)

    def _boundary_conflicts(
        self,
        render_plan: RenderPlan,
        shot_key: str,
        video: MediaOperationResult,
    ) -> tuple[str, ...]:
        render_shot = next(
            item for item in render_plan.ordered_render_shots if item.shot_key == shot_key
        )
        end_criteria = {
            item.criterion_id
            for item in render_shot.resolved_requirements
            if item.phase == RequirementPhase.END
        }
        report = _selected_report(video.decision, video.reports)
        return tuple(
            sorted(
                item.criterion_id
                for item in report.criterion_results
                if item.criterion_id in end_criteria and item.status != CriterionStatus.PASS
            )
        )

    def _ledger(
        self,
        reports: list[EvaluationReport],
        view: EvaluationView,
    ) -> IssueLedger:
        ledger = IssueLedger()
        for report in reports:
            ledger = ledger.replace_from_report(report, view)
        return ledger

    def _load_progress(
        self,
        step: StepRecord,
        operation_key: str,
    ) -> MediaOperationProgress:
        if step.status == StepStatus.RUNNING and step.selected_output is not None:
            progress = self._load(step.selected_output, MediaOperationProgress)
            if progress.operation_key != operation_key:
                raise WorkflowError("persisted media progress belongs to another operation")
            return progress
        return MediaOperationProgress(operation_key=operation_key)

    def _save_progress(
        self,
        step: StepRecord,
        progress: MediaOperationProgress,
    ) -> StepRecord:
        output = self._put(progress)
        changed = step.model_copy(
            update={
                "selected_output": output,
                "attempt_records": progress.attempts,
            }
        )
        self._state = self.run_store.upsert_step(self._state, changed)
        return changed

    def _update_shot(self, shot_key: str, **updates: object) -> None:
        existing = next(
            (item for item in self._state.shots if item.shot_key == shot_key),
            ShotRunState(shot_key=shot_key),
        )
        changed = existing.model_copy(update=updates)
        self._state = self.run_store.upsert_shot(self._state, changed)
        self.stage_output.publish_audit(self._state)

    def _valid_image_result(
        self,
        result: MediaOperationResult,
        *,
        expected_size: tuple[int, int] | None,
        require_panorama: bool = False,
        validate_media_bundle: bool = True,
    ) -> bool:
        try:
            candidate = _selected_candidate(result)
        except WorkflowError:
            return False
        if candidate.artifact_ref is None:
            return False
        artifacts = (
            tuple(item.artifact_ref for item in candidate.media)
            if candidate.media and validate_media_bundle
            else (candidate.artifact_ref,)
        )
        return all(
            ImageTechnicalValidator()
            .validate(
                self.run_store.artifacts,
                artifact,
                expected_size=expected_size,
                require_panorama=require_panorama,
            )
            .status
            == TechnicalStatus.VALID
            for artifact in artifacts
        )

    def _valid_video_result(
        self,
        result: MediaOperationResult,
        duration: int,
        fps: int,
        start_image: ArtifactRef,
    ) -> bool:
        try:
            candidate = _selected_candidate(result)
        except WorkflowError:
            return False
        if candidate.artifact_ref is None:
            return False
        validation = VideoTechnicalValidator().validate(
            self.run_store.artifacts,
            candidate.artifact_ref,
            expected_duration=duration,
            expected_resolution=(
                self.request.resolution.width,
                self.request.resolution.height,
            ),
            expected_fps=fps,
            expected_start_image=start_image,
        )
        return validation.status == TechnicalStatus.VALID

    def _save_state(self, **updates: object) -> None:
        self._state = self._state.model_copy(update={"updated_at": utc_now(), **updates})
        self.run_store.save(self._state)

    def _begin_step(
        self,
        name: str,
        key: str,
        inputs: tuple[str, ...],
    ) -> StepRecord:
        existing = next(
            (item for item in self._state.steps if item.step_name == name),
            None,
        )
        if (
            existing is not None
            and existing.step_key == key
            and existing.status == StepStatus.RUNNING
        ):
            return existing
        step = StepRecord(
            step_name=name,
            step_key=key,
            status=StepStatus.RUNNING,
            input_refs=inputs,
        )
        self._state = self.run_store.upsert_step(self._state, step)
        self.run_store.append_event("step_started", {"step": name, "step_key": key})
        return step

    def _complete_step(
        self,
        step: StepRecord,
        *,
        output: ArtifactRef | None = None,
        selected_key: str | None = None,
    ) -> StepRecord:
        completed = step.model_copy(
            update={
                "status": StepStatus.COMPLETED,
                "selected_output": output,
                "selected_key": selected_key,
                "pending_provider_job": None,
                "pending_logical_attempt": None,
                "pending_candidate_index": None,
                "pending_provider_usage": None,
                "pending_provider_call_refs": (),
                "error": None,
            }
        )
        self._state = self.run_store.upsert_step(self._state, completed)
        self.run_store.append_event(
            "step_completed",
            {"step": step.step_name, "step_key": step.step_key},
        )
        return completed

    def _find_step(self, name: str) -> StepRecord:
        try:
            return next(item for item in self._state.steps if item.step_name == name)
        except StopIteration as exc:
            raise WorkflowError(f"missing workflow step {name}") from exc

    def _put(self, value: BaseModel) -> ArtifactRef:
        return self.run_store.artifacts.put_bytes(
            canonical_bytes(value) + b"\n",
            "application/json",
        )

    def _load(self, reference: ArtifactRef, model: type[ModelT]) -> ModelT:
        path = self.run_store.artifacts.verify(reference)
        try:
            return model.model_validate_json(path.read_bytes())
        except ValueError as exc:
            raise WorkflowError(f"invalid persisted {model.__name__}: {exc}") from exc

    def _key(
        self,
        kind: str,
        inputs: tuple[str, ...],
        *,
        implementation: str = WORKFLOW_VERSION,
        renderer: str | None = None,
        provider: str | None = None,
        behavior: object | None = None,
    ) -> str:
        return step_key(
            kind=kind,
            input_hashes=inputs,
            implementation_version=implementation,
            renderer_version=renderer,
            provider_fingerprint=provider,
            behavior_config=(
                behavior
                if behavior is not None
                else {
                    "attempts": self.config.generation.attempts,
                    "story_planning_attempts": (self.config.generation.story_planning_attempts),
                    "grounding_attempts": self.config.generation.grounding_attempts,
                    "candidates_per_attempt": self.config.generation.candidates_per_attempt,
                    "spatial_failure_policy": (self.config.generation.spatial_failure_policy),
                    "spatial_repair_attempts": (self.config.generation.spatial_repair_attempts),
                    "novel_station_attempts": (self.config.generation.novel_station_attempts),
                    "max_hfov_degrees": self.config.generation.max_hfov_degrees,
                    "safe_fallback_hfov_min": (self.config.generation.safe_fallback_hfov_min),
                    "safe_fallback_hfov_max": (self.config.generation.safe_fallback_hfov_max),
                }
            ),
        )

    def _resume_interrupted(self) -> None:
        if self._state.status != ProjectStatus.INTERRUPTED:
            return
        if self._state.final_video is not None:
            target = ProjectStatus.ASSEMBLED
        elif self._state.active_render_plan is not None:
            target = ProjectStatus.RENDER_PLANNED
        elif self._state.active_reference_library is not None:
            target = ProjectStatus.REFERENCES_READY
        elif self._state.active_story_plan is not None:
            target = ProjectStatus.STORY_PLANNED
        else:
            target = ProjectStatus.PREFLIGHTED
        self._state = self.run_store.transition(self._state, target, error=None)

    def _interrupt(self, error: str) -> None:
        if self._state.status in {
            ProjectStatus.DELIVERED,
            ProjectStatus.DELIVERED_DEGRADED,
            ProjectStatus.PROCESS_FAILED,
            ProjectStatus.INTERRUPTED,
        }:
            return
        self._state = self.run_store.transition(
            self._state,
            ProjectStatus.INTERRUPTED,
            error=error[:2_000],
        )
        self.stage_output.publish_audit(self._state)

    def _fail(self, error: str) -> None:
        for step in self._state.steps:
            if step.status == StepStatus.RUNNING:
                failed = step.model_copy(
                    update={
                        "status": StepStatus.FAILED,
                        "error": error[:2_000],
                    }
                )
                self._state = self.run_store.upsert_step(self._state, failed)
        if self._state.status not in {
            ProjectStatus.DELIVERED,
            ProjectStatus.DELIVERED_DEGRADED,
            ProjectStatus.PROCESS_FAILED,
        }:
            self._state = self.run_store.transition(
                self._state,
                ProjectStatus.PROCESS_FAILED,
                error=error[:2_000],
            )
        self.stage_output.publish_audit(self._state)

    def _verify_delivered(self) -> None:
        if self._state.final_video is None or self._state.manifest is None:
            raise WorkflowError("delivered state lacks final artifacts")
        self.run_store.artifacts.verify(self._state.final_video)
        manifest = self._load(self._state.manifest, DeliveryManifest)
        manifest.assert_hash()


def _aspect_ratio(size: PixelSize) -> str:
    divisor = math.gcd(size.width, size.height)
    return f"{size.width // divisor}:{size.height // divisor}"


def _evaluation_error_evidence(error: ContractError | ProviderError) -> str:
    if isinstance(error, ProviderError):
        detail = f"{error.kind.value}: {error.message}"
    else:
        detail = f"{type(error).__name__}: {error}"
    compact = " ".join(detail.split())
    return f"Evaluator unavailable ({compact[:1_950]})"


def _image_prompt_record(
    request: ImageRequest,
    *,
    logical_attempt: int,
    candidate_index: int,
    candidate_key: str,
) -> ProviderPrompt:
    return ProviderPrompt(
        purpose=PromptPurpose.GENERATION,
        provider_role="image",
        logical_attempt=logical_attempt,
        candidate_index=candidate_index,
        candidate_key=candidate_key,
        prompt=request.prompt,
        attachments=tuple(
            PromptAttachment(name=item.name, artifact_ref=item.artifact_ref)
            for item in request.input_images
        ),
        aspect_ratio=request.aspect_ratio,
        resolution=request.resolution,
    )


def _video_prompt_record(
    request: VideoRequest,
    *,
    logical_attempt: int,
    candidate_index: int,
    candidate_key: str,
) -> ProviderPrompt:
    attachments = (request.start_image, *request.reference_images)
    return ProviderPrompt(
        purpose=PromptPurpose.GENERATION,
        provider_role="video",
        logical_attempt=logical_attempt,
        candidate_index=candidate_index,
        candidate_key=candidate_key,
        prompt=request.prompt,
        attachments=tuple(
            PromptAttachment(name=item.name, artifact_ref=item.artifact_ref) for item in attachments
        ),
        resolution=request.resolution,
        duration_seconds=request.duration,
        fps=request.fps,
    )


def _judge_prompt_record(
    candidate: CandidateRecord,
    prompt: str,
    *,
    logical_attempt: int,
    candidate_index: int,
) -> ProviderPrompt:
    artifact = _required_artifact(candidate)
    media = candidate.media or (CandidateMedia(role="candidate", artifact_ref=artifact),)
    return ProviderPrompt(
        purpose=PromptPurpose.EVALUATION,
        provider_role="judge",
        logical_attempt=logical_attempt,
        candidate_index=candidate_index,
        candidate_key=candidate.candidate_key,
        prompt=prompt,
        attachments=tuple(
            PromptAttachment(name=item.role, artifact_ref=item.artifact_ref) for item in media
        ),
        response_contract=ResponseContract.EVALUATION_ROWS.value,
    )


def _record_prompt(
    prompts: list[ProviderPrompt],
    prompt: ProviderPrompt,
) -> bool:
    identity = (
        prompt.purpose,
        prompt.logical_attempt,
        prompt.candidate_index,
        prompt.candidate_key,
    )
    if any(
        (
            item.purpose,
            item.logical_attempt,
            item.candidate_index,
            item.candidate_key,
        )
        == identity
        for item in prompts
    ):
        return False
    prompts.append(prompt)
    return True


def _first_frame_result(
    base: MediaOperationResult,
    *,
    participation: FirstFrameParticipationResult,
    effective_mode: FirstFrameMode,
    reuse_evaluation: EvaluationReport | None,
    planner_prompt: ProviderPrompt | None,
    reuse_prompt: ProviderPrompt | None,
) -> FirstFrameOperationResult:
    extra_prompts: list[ProviderPrompt] = []
    if planner_prompt is not None:
        extra_prompts.append(planner_prompt)
    if reuse_prompt is not None and effective_mode != FirstFrameMode.REUSE:
        extra_prompts.append(reuse_prompt)
    prompts = tuple(
        [
            *extra_prompts,
            *(item for item in base.prompts if item not in extra_prompts),
        ]
    )
    extra_usages: list[ProviderUsage] = []
    extra_call_refs: list[str] = []
    if participation.planner_usage.call_count:
        extra_usages.append(participation.planner_usage)
        if participation.planner_call_ref is not None:
            extra_call_refs.append(participation.planner_call_ref)
    if reuse_evaluation is not None and effective_mode != FirstFrameMode.REUSE:
        extra_usages.append(reuse_evaluation.evaluator_usage)
        extra_call_refs.append(reuse_evaluation.evaluator_ref)
    attempts = list(base.attempts)
    if extra_usages:
        if not attempts:
            raise WorkflowError("first-frame provider usage has no logical attempt")
        first = attempts[0]
        attempts[0] = first.model_copy(
            update={
                "provider_call_refs": tuple(
                    dict.fromkeys((*extra_call_refs, *first.provider_call_refs))
                ),
                "provider_usage": ProviderUsage.combine(
                    (*tuple(extra_usages), first.provider_usage)
                ),
                "prompts": tuple(
                    [
                        *extra_prompts,
                        *(item for item in first.prompts if item not in extra_prompts),
                    ]
                ),
            }
        )
    return FirstFrameOperationResult(
        operation_key=base.operation_key,
        candidates=base.candidates,
        reports=base.reports,
        decision=base.decision,
        attempts=tuple(attempts),
        prompts=prompts,
        participation=participation,
        effective_mode=effective_mode,
        reuse_evaluation=reuse_evaluation,
    )


def _recipe_requires_panorama(recipe: ReferenceRecipe) -> bool:
    return isinstance(recipe, ScenePanoramaRecipe) or (
        isinstance(recipe, ProvidedReferenceRecipe) and recipe.require_panorama
    )


def _attempt_record(
    attempt: int,
    started_at: str,
    candidates: tuple[CandidateRecord, ...],
    reports: tuple[EvaluationReport, ...],
) -> AttemptRecord:
    usages = tuple(
        [item.provider_usage for item in candidates] + [item.evaluator_usage for item in reports]
    )
    return AttemptRecord(
        logical_attempt=attempt,
        provider_call_refs=tuple(
            [
                reference
                for item in candidates
                for reference in (
                    item.provider_call_refs
                    if item.provider_call_refs
                    else ((item.provider_call_ref,) if item.provider_usage.call_count else ())
                )
            ]
            + [item.evaluator_ref for item in reports]
        ),
        candidate_refs=tuple(item.candidate_key for item in candidates),
        evaluation_refs=tuple(item.report_key for item in reports),
        provider_usage=ProviderUsage.combine(usages),
        started_at=started_at,
        completed_at=utc_now(),
    )


def _candidate_preflight_failure(
    *,
    candidate_key: str,
    scene_key: str,
    station_key: str,
    reference_hash: str,
    error: GroundingValidationExhausted,
) -> CandidatePreflightFailure:
    aliases = ", ".join(error.failed_target_aliases) or "the requested targets"
    correction = (
        f"Grounding validation exhausted for {aliases}. Regenerate a panorama where every "
        "requested anchor or region is clearly locatable, then obey the visibility/box truth "
        "table exactly."
    )[:2_000]
    return CandidatePreflightFailure(
        candidate_key=candidate_key,
        scene_key=scene_key,
        station_key=station_key,
        reference_hash=reference_hash,
        failure_code=error.failure_code,
        failed_target_aliases=error.failed_target_aliases,
        logical_attempt_refs=error.logical_attempt_refs,
        provider_call_refs=error.provider_call_refs,
        response_refs=error.response_refs,
        validation_refs=error.validation_refs,
        correction=correction,
    )


def _grounding_unavailable_from_failure(
    failure: CandidatePreflightFailure,
) -> GroundingUnavailableEvidence:
    return GroundingUnavailableEvidence.create(
        scene_key=failure.scene_key,
        station_key=failure.station_key,
        reference_hash=failure.reference_hash,
        failed_target_aliases=failure.failed_target_aliases,
        logical_attempt_refs=failure.logical_attempt_refs,
        provider_call_refs=failure.provider_call_refs,
        response_refs=failure.response_refs,
        validation_refs=failure.validation_refs,
    )


def _grounding_unavailable_from_error(
    error: GroundingValidationExhausted,
) -> GroundingUnavailableEvidence:
    return GroundingUnavailableEvidence.create(
        scene_key=error.scene_key,
        station_key=error.station_key,
        reference_hash=error.reference_hash,
        failed_target_aliases=error.failed_target_aliases,
        logical_attempt_refs=error.logical_attempt_refs,
        provider_call_refs=error.provider_call_refs,
        response_refs=error.response_refs,
        validation_refs=error.validation_refs,
    )


def _invalid_candidate(
    operation_key: str,
    attempt: int,
    index: int,
    detail: str,
    *,
    provider_usage: ProviderUsage | None = None,
    provider_call_refs: tuple[str, ...] = (),
) -> CandidateRecord:
    error_ref = stable_key(
        "provider_error",
        operation_key,
        f"{attempt}:{index}",
    )
    return CandidateRecord(
        candidate_key=stable_key(
            "candidate",
            operation_key,
            f"{attempt}:{index}",
        ),
        operation_key=operation_key,
        artifact_ref=None,
        technical_status=TechnicalStatus.INVALID,
        technical_findings=(
            TechnicalFinding(
                code="provider_call",
                passed=False,
                detail=detail[:500],
            ),
        ),
        provider_call_ref=error_ref,
        provider_call_refs=(*provider_call_refs, error_ref),
        provider_usage=provider_usage or ProviderUsage(call_count=1),
        logical_attempt=attempt,
    )


def _candidate_with_evaluation_media(
    candidate: CandidateRecord,
    *,
    reference_media: tuple[CandidateMedia, ...],
    candidate_role: str | None,
) -> CandidateRecord:
    if candidate_role is None or candidate.artifact_ref is None:
        return candidate
    return CandidateRecord.model_validate(
        {
            **candidate.model_dump(mode="python"),
            "media": (
                *reference_media,
                CandidateMedia(
                    role=candidate_role,
                    artifact_ref=candidate.artifact_ref,
                ),
            ),
        }
    )


def _provider_usage(metrics: ProviderCallMetrics) -> ProviderUsage:
    return ProviderUsage(
        call_count=1,
        transport_retries=metrics.transport_retries,
        elapsed_seconds=metrics.elapsed_seconds,
        known_cost_usd=metrics.cost_usd,
    )


def _known_cost(attempts: tuple[AttemptRecord, ...]) -> float | None:
    values = tuple(
        attempt.provider_usage.known_cost_usd
        for attempt in attempts
        if attempt.provider_usage.known_cost_usd is not None
    )
    return sum(values) if values else None


def _selected_candidate(result: MediaOperationResult) -> CandidateRecord:
    try:
        return next(
            item
            for item in result.candidates
            if item.candidate_key == result.decision.selected_candidate
        )
    except StopIteration as exc:
        raise WorkflowError("selection references a missing candidate") from exc


def _selected_report(
    decision: SelectionDecision,
    reports: tuple[EvaluationReport, ...],
) -> EvaluationReport:
    try:
        return next(item for item in reports if item.report_key == decision.selected_report)
    except StopIteration as exc:
        raise WorkflowError("selection references a missing evaluation report") from exc


def _required_artifact(candidate: CandidateRecord) -> ArtifactRef:
    if candidate.artifact_ref is None:
        raise WorkflowError("selected candidate has no media artifact")
    return candidate.artifact_ref


def _requirement_summary(
    reports: tuple[EvaluationReport, ...],
) -> CriterionSummary:
    status_by_criterion: dict[str, CriterionStatus] = {}
    severity = {
        CriterionStatus.PASS: 0,
        CriterionStatus.UNKNOWN: 1,
        CriterionStatus.FAIL: 2,
    }
    for report in reports:
        for item in report.criterion_results:
            if item.kind != CriterionKind.REQUIREMENT:
                continue
            current = status_by_criterion.get(item.criterion_id)
            if current is None or severity[item.status] > severity[current]:
                status_by_criterion[item.criterion_id] = item.status
    return CriterionSummary(
        passed=sum(item == CriterionStatus.PASS for item in status_by_criterion.values()),
        failed=sum(item == CriterionStatus.FAIL for item in status_by_criterion.values()),
        unknown=sum(item == CriterionStatus.UNKNOWN for item in status_by_criterion.values()),
    )
