"""ExecutionAgent is the sole plan and logical-attempt decision owner."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import ValidationError

from story_engine.config import GenerationPolicy
from story_engine.domain.evaluation import (
    CandidateRecord,
    CriterionKind,
    CriterionStatus,
    EvaluationReport,
    SelectionDecision,
)
from story_engine.domain.first_frame import FirstFrameParticipationAssessment
from story_engine.domain.reference import (
    CharacterReferenceRecipe,
    CharacterViewRole,
    GuidedCharacterReferenceRecipe,
    GuidedScenePanoramaRecipe,
    ReferenceLibrary,
    ReferenceRecipe,
)
from story_engine.domain.render import RenderPlan, RequirementPhase
from story_engine.domain.request import ProjectRequest
from story_engine.domain.spatial import ResolvedSpatialPlan
from story_engine.domain.story import StoryPlan
from story_engine.domain.trace import (
    PromptAttachment,
    PromptPurpose,
    ProviderPrompt,
)
from story_engine.errors import (
    ContractError,
    GroundingValidationExhausted,
    PlanComplexityError,
    ProviderError,
    ProviderErrorKind,
    StoryCompilationError,
)
from story_engine.planning.participation import (
    FirstFrameParticipationValidator,
    FirstFrameParticipationWireResponse,
)
from story_engine.planning.prompt_views import (
    EvaluationCriterionView,
    EvaluationView,
    FirstFrameParticipationView,
)
from story_engine.planning.reference_compiler import ReferenceCompiler
from story_engine.planning.render_compiler import RenderCompiler
from story_engine.planning.story_compiler import StoryCompileConstraints, StoryCompiler
from story_engine.planning.story_planner import StoryDraft
from story_engine.planning.view_builder import PromptViewBuilder
from story_engine.prompts.grounding import render_grounding
from story_engine.prompts.participation import render_first_frame_participation
from story_engine.prompts.planning import render_story_planning
from story_engine.providers.ports import (
    Attachment,
    ProviderCallMetrics,
    ResponseContract,
    RuntimeCapabilities,
    TextProvider,
    TextRequest,
)
from story_engine.selection import SelectionPolicy
from story_engine.spatial.grounding import (
    GroundingValidationError,
    GroundingValidator,
    GroundingView,
    GroundingWireResponse,
    SceneAnchorMap,
)
from story_engine.spatial.probes import ProbeSet
from story_engine.task_pool import TaskPool


class GroundingAttemptValidationStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class GroundingAttemptEvent:
    logical_attempt: int
    target_aliases: tuple[str, ...]
    prompt: ProviderPrompt
    raw_response: str
    metrics: ProviderCallMetrics
    validation_status: GroundingAttemptValidationStatus
    validation_code: str
    validation_error: str | None
    next_correction: str | None
    started_at: str
    completed_at: str


class ExecutionAgent:
    def __init__(
        self,
        *,
        planner: TextProvider,
        capabilities: RuntimeCapabilities,
        generation_policy: GenerationPolicy,
        task_pool: TaskPool,
        render_compiler: RenderCompiler,
    ) -> None:
        self.planner = planner
        self.capabilities = capabilities
        self.generation_policy = generation_policy
        self.task_pool = task_pool
        self.render_compiler = render_compiler
        self.reference_compiler = ReferenceCompiler()
        self.selection_policy = SelectionPolicy()
        self.view_builder = PromptViewBuilder()
        self.grounding_validator = GroundingValidator()

    async def create_story_plan(
        self,
        request: ProjectRequest,
        *,
        on_prompt: Callable[[ProviderPrompt], None] | None = None,
    ) -> tuple[
        StoryPlan,
        tuple[ProviderCallMetrics, ...],
        tuple[ProviderPrompt, ...],
    ]:
        video = self.capabilities.video.video
        image = self.capabilities.image.image
        planner_capability = self.capabilities.planner.text
        if video is None or image is None or planner_capability is None:
            raise StoryCompilationError("planner, image, and video capabilities are required")
        delivery = request.generation_requirements.delivery_requirements
        requested_durations = delivery.allowed_video_duration_seconds
        allowed = tuple(
            duration for duration in video.supported_durations if duration in requested_durations
        )
        if not allowed:
            raise StoryCompilationError("request and VideoProvider have no common shot duration")
        compile_constraints = StoryCompileConstraints(
            supported_durations=allowed,
            max_image_input_images=image.max_input_images,
            max_video_reference_images=video.max_reference_images,
        )
        view = self.view_builder.story_planning(
            request,
            self.capabilities,
            compile_constraints,
        )
        last_error: Exception | None = None
        correction: str | None = None
        calls: list[ProviderCallMetrics] = []
        prompts: list[ProviderPrompt] = []
        for attempt in range(1, self.generation_policy.story_planning_attempts + 1):
            prompt = render_story_planning(view, correction=correction)
            if len(prompt) > planner_capability.max_prompt_characters:
                raise PlanComplexityError("story planning view exceeds provider context")
            provider_request = TextRequest(
                prompt=prompt,
                response_contract=ResponseContract.STORY_DRAFT,
            )
            prompt_record = _text_prompt_record(
                provider_request,
                purpose=PromptPurpose.PLANNING,
                logical_attempt=attempt,
            )
            prompts.append(prompt_record)
            if on_prompt is not None:
                on_prompt(prompt_record)
            async with self.task_pool.slot():
                result = await self.planner.generate_text(provider_request)
            calls.append(result.metrics)
            try:
                draft = StoryDraft.model_validate(json.loads(result.text))
                plan = StoryCompiler(compile_constraints).compile(request, draft)
                return plan, tuple(calls), tuple(prompts)
            except (json.JSONDecodeError, ValueError, StoryCompilationError) as exc:
                last_error = exc
                correction = _story_planning_correction(exc)
        raise StoryCompilationError(f"story planning exhausted logical attempts: {last_error}")

    def reference_recipes(
        self, request: ProjectRequest, plan: StoryPlan
    ) -> tuple[ReferenceRecipe, ...]:
        return self.reference_compiler.recipes(request, plan)

    async def ground(
        self,
        story_plan: StoryPlan,
        probe_set: ProbeSet,
        *,
        station_key: str,
        on_prompt: Callable[[ProviderPrompt], None] | None = None,
        on_attempt: Callable[[GroundingAttemptEvent], None] | None = None,
    ) -> tuple[
        SceneAnchorMap,
        tuple[ProviderCallMetrics, ...],
        tuple[ProviderPrompt, ...],
    ]:
        capability = self.capabilities.planner.text
        if capability is None:
            raise StoryCompilationError("planner text capability is required for grounding")
        view = self.view_builder.grounding(story_plan, probe_set, station_key=station_key)
        if not view.targets:
            grounding = self.grounding_validator.validate(
                view,
                GroundingWireResponse(items=[]),
            )
            return grounding, (), ()
        last_error: Exception | None = None
        last_validation_code = "unknown"
        last_failed_aliases = tuple(target.alias for target in view.targets)
        correction: str | None = None
        calls: list[ProviderCallMetrics] = []
        prompts: list[ProviderPrompt] = []
        for attempt in range(1, self.generation_policy.grounding_attempts + 1):
            started_at = _utc_now()
            rendered = render_grounding(
                view,
                max_prompt_characters=capability.max_prompt_characters,
                correction=correction,
            )
            if len(rendered.attachments) > capability.max_attachments:
                raise StoryCompilationError(
                    "planner attachment capability cannot carry the six grounding probes"
                )
            request = TextRequest(
                prompt=rendered.text,
                attachments=tuple(
                    Attachment(name=probe.role.value, artifact_ref=probe.artifact_ref)
                    for probe in probe_set.probes
                ),
                response_contract=ResponseContract.GROUNDING_ROWS,
            )
            prompt_record = _text_prompt_record(
                request,
                purpose=PromptPurpose.GROUNDING,
                logical_attempt=attempt,
            )
            prompts.append(prompt_record)
            if on_prompt is not None:
                on_prompt(prompt_record)
            async with self.task_pool.slot():
                result = await self.planner.generate_text(request)
            calls.append(result.metrics)
            try:
                response = GroundingWireResponse.model_validate(json.loads(result.text))
                grounding = self.grounding_validator.validate(view, response)
                if on_attempt is not None:
                    on_attempt(
                        GroundingAttemptEvent(
                            logical_attempt=attempt,
                            target_aliases=tuple(target.alias for target in view.targets),
                            prompt=prompt_record,
                            raw_response=result.text,
                            metrics=result.metrics,
                            validation_status=GroundingAttemptValidationStatus.VALID,
                            validation_code="valid",
                            validation_error=None,
                            next_correction=None,
                            started_at=started_at,
                            completed_at=_utc_now(),
                        )
                    )
                return grounding, tuple(calls), tuple(prompts)
            except (json.JSONDecodeError, ValidationError, GroundingValidationError) as exc:
                last_error = exc
                last_validation_code = _grounding_validation_code(exc)
                last_failed_aliases = _grounding_failed_aliases(view, result.text, exc)
                correction = _grounding_correction(exc)
                if on_attempt is not None:
                    on_attempt(
                        GroundingAttemptEvent(
                            logical_attempt=attempt,
                            target_aliases=tuple(target.alias for target in view.targets),
                            prompt=prompt_record,
                            raw_response=result.text,
                            metrics=result.metrics,
                            validation_status=GroundingAttemptValidationStatus.INVALID,
                            validation_code=last_validation_code,
                            validation_error=" ".join(str(exc).split())[:2_000],
                            next_correction=correction,
                            started_at=started_at,
                            completed_at=_utc_now(),
                        )
                    )
        raise GroundingValidationExhausted(
            scene_key=view.scene_key,
            station_key=station_key,
            reference_hash=probe_set.source_panorama_hash,
            attempt_count=self.generation_policy.grounding_attempts,
            failed_target_aliases=last_failed_aliases,
            last_validation_code=last_validation_code,
            last_validation_error=(
                " ".join(str(last_error).split())[:2_000]
                if last_error is not None
                else "grounding validation did not return a valid result"
            ),
            provider_call_refs=tuple(metrics.call_ref for metrics in calls),
        )

    def reference_evaluation_view(self, recipe: ReferenceRecipe) -> EvaluationView:
        statements: tuple[tuple[str, str, int], ...]
        if isinstance(recipe, GuidedScenePanoramaRecipe) or recipe.kind == "scene_panorama" or (
            recipe.kind == "provided" and recipe.reference_kind.value == "scene_panorama"
        ):
            statements = tuple(
                [
                    (
                        "source_fidelity",
                        "Image 2 preserves all visible fixed architecture, layout, materials, "
                        "lighting character, and visual style from Image 1 without contradiction.",
                        100,
                    )
                ]
                if isinstance(recipe, GuidedScenePanoramaRecipe)
                else []
            ) + (
                ("scene_identity", "The panorama matches the specified scene.", 95),
                ("layout", "Fixed landmarks and semantic layout are coherent.", 90),
                (
                    "station",
                    "The panorama is captured from the explicitly declared source station.",
                    95,
                ),
                (
                    "anchor_uniqueness",
                    "Named camera anchors are single, local, clearly bounded structures.",
                    92,
                ),
                ("seam", "The equirectangular panorama has no broken seam.", 90),
                ("style", "Lighting and visual style match the recipe.", 50),
            )
        else:
            statements = (
                ("identity", "The subject matches its specified appearance and state.", 95),
                ("isolation", "No forbidden co-subject is present.", 90),
                ("style", "The visual style matches the recipe.", 50),
            )
        return EvaluationView(
            operation=f"reference:{recipe.kind}",
            media_instructions=(
                (
                    "Image 1 is the uploaded scene reference.",
                    "Image 2 is the candidate completed equirectangular panorama.",
                )
                if isinstance(recipe, GuidedScenePanoramaRecipe)
                else ()
            ),
            criteria=tuple(
                EvaluationCriterionView(
                    criterion_id=f"{recipe.need_key}:{suffix}",
                    kind=(
                        CriterionKind.PREFERENCE if suffix == "style" else CriterionKind.REQUIREMENT
                    ),
                    priority=_priority,
                    statement=statement,
                    phase=RequirementPhase.ALWAYS,
                    category=("continuity" if suffix in {"identity", "layout"} else suffix),
                )
                for suffix, statement, _priority in statements
            ),
        )

    def character_view_evaluation_view(
        self,
        recipe: CharacterReferenceRecipe,
        role: CharacterViewRole,
    ) -> EvaluationView:
        if role == "front":
            if isinstance(recipe, GuidedCharacterReferenceRecipe):
                return EvaluationView(
                    operation="reference:guided_character:front",
                    media_instructions=(
                        "Image 1 is the authoritative uploaded character reference.",
                        "Image 2 is the normalized candidate front view.",
                    ),
                    criteria=(
                        EvaluationCriterionView(
                            criterion_id=f"{recipe.need_key}:front:identity",
                            kind=CriterionKind.REQUIREMENT,
                            priority=100,
                            statement=(
                                "Image 2 preserves the exact identity, face, hair, body "
                                "proportions, clothing, colors, materials, and distinctive "
                                "attachments visible in Image 1 without redesign."
                            ),
                            phase=RequirementPhase.ALWAYS,
                            category="identity",
                        ),
                        EvaluationCriterionView(
                            criterion_id=f"{recipe.need_key}:front:reference_contract",
                            kind=CriterionKind.REQUIREMENT,
                            priority=90,
                            statement=(
                                "Image 2 contains exactly one isolated, uncropped, full-body "
                                "front view in a neutral pose with no grid, collage, text, or "
                                "co-subject."
                            ),
                            phase=RequirementPhase.ALWAYS,
                            category="isolation",
                        ),
                    ),
                )
            return self.reference_evaluation_view(recipe)
        orientation = (
            "Image 2 is a true left-facing side view, not a front, back, or "
            "three-quarter duplicate."
            if role == "side"
            else "Image 2 is a true back view with no visible face, not a front, "
            "side, or three-quarter duplicate."
        )
        statements = (
            (
                "identity",
                "Image 2 depicts the exact same character as the canonical front "
                "view in Image 1, preserving silhouette, proportions, colors, "
                "materials, clothing, and attachments.",
                100,
                "identity",
            ),
            ("orientation", orientation, 95, "continuity"),
            (
                "reference_contract",
                "Image 2 contains exactly one isolated, uncropped, full-body character "
                "in a neutral pose with no grid, collage, text, or co-subject.",
                90,
                "isolation",
            ),
        )
        return EvaluationView(
            operation=f"reference:character:{role}",
            media_instructions=(
                "Image 1 is the frozen canonical front view.",
                f"Image 2 is the candidate {role} view.",
            ),
            criteria=tuple(
                EvaluationCriterionView(
                    criterion_id=f"{recipe.need_key}:{role}:{suffix}",
                    kind=CriterionKind.REQUIREMENT,
                    priority=priority,
                    statement=statement,
                    phase=RequirementPhase.ALWAYS,
                    category=category,
                )
                for suffix, statement, priority, category in statements
            ),
        )

    async def assess_first_frame_participation(
        self,
        view: FirstFrameParticipationView,
    ) -> tuple[
        FirstFrameParticipationAssessment | None,
        ProviderCallMetrics,
        ProviderPrompt,
        str | None,
    ]:
        capability = self.capabilities.planner.text
        if capability is None:
            raise StoryCompilationError("planner text capability is required for participation")
        rendered = render_first_frame_participation(
            view,
            max_prompt_characters=capability.max_prompt_characters,
        )
        if len(rendered.attachments) > capability.max_attachments:
            raise PlanComplexityError(
                "planner cannot carry every first-frame participation attachment"
            )
        request = TextRequest(
            prompt=rendered.text,
            attachments=tuple(
                Attachment(name=f"image_{index}", artifact_ref=artifact)
                for index, artifact in enumerate(rendered.attachments, start=1)
            ),
            response_contract=ResponseContract.FIRST_FRAME_PARTICIPATION,
        )
        prompt = _text_prompt_record(
            request,
            purpose=PromptPurpose.PLANNING,
            logical_attempt=1,
        )
        async with self.task_pool.slot():
            result = await self.planner.generate_text(request)
        try:
            response = FirstFrameParticipationWireResponse.model_validate(json.loads(result.text))
            assessment = FirstFrameParticipationValidator().validate(view, response)
        except (json.JSONDecodeError, ValueError, ContractError) as exc:
            detail = " ".join(str(exc).split()) or type(exc).__name__
            return None, result.metrics, prompt, detail[:1_000]
        return assessment, result.metrics, prompt, None

    def select(
        self,
        operation_key: str,
        candidates: tuple[CandidateRecord, ...],
        reports: tuple[EvaluationReport, ...],
    ) -> SelectionDecision:
        return self.selection_policy.select(operation_key, candidates, reports)

    def should_stop_early(self, reports: tuple[EvaluationReport, ...]) -> bool:
        return any(
            all(
                result.status == CriterionStatus.PASS
                for result in report.criterion_results
                if result.kind == CriterionKind.REQUIREMENT
            )
            for report in reports
        )

    def build_reference_library(
        self,
        story_plan: StoryPlan,
        decisions: tuple[SelectionDecision, ...],
        candidates: tuple[CandidateRecord, ...],
        reports: tuple[EvaluationReport, ...],
    ) -> ReferenceLibrary:
        return self.reference_compiler.build_library(story_plan, decisions, candidates, reports)

    def create_render_plan(
        self,
        request: ProjectRequest,
        story_plan: StoryPlan,
        library: ReferenceLibrary,
        evidence: tuple[ResolvedSpatialPlan, ...],
    ) -> RenderPlan:
        return self.render_compiler.compile(
            request, story_plan, library, evidence, self.capabilities
        )

    @staticmethod
    def response_contract_error(operation: str) -> ProviderError:
        return ProviderError(
            kind=ProviderErrorKind.RESPONSE_CONTRACT,
            message=f"{operation} returned an invalid stable wire response",
            retryable=False,
        )


def _story_planning_correction(error: Exception) -> str:
    if isinstance(error, json.JSONDecodeError):
        return "Return one syntactically valid JSON object matching the StoryDraft schema."
    if isinstance(error, ValidationError):
        details = error.errors(include_input=False, include_url=False)
        rows = [
            f"{'.'.join(str(item) for item in detail['loc'])}: {detail['msg']}"
            for detail in details[:8]
        ]
        return "Fix StoryDraft schema validation: " + "; ".join(rows)
    message = " ".join(str(error).split())
    if not message:
        return "Return a complete StoryDraft that passes deterministic validation."
    return f"Fix deterministic StoryDraft validation: {message[:1_000]}"


def _grounding_correction(error: Exception) -> str:
    truth_table = (
        " For every item: visible or partially_visible requires a normalized_box object; "
        "not_visible or unknown requires normalized_box=null; never emit occluded."
    )
    if isinstance(error, json.JSONDecodeError):
        return (
            "Return one syntactically valid JSON object matching the grounding schema."
            + truth_table
        )
    if isinstance(error, ValidationError):
        details = error.errors(include_input=False, include_url=False)
        rows = [
            f"{'.'.join(str(item) for item in detail['loc'])}: {detail['msg']}"
            for detail in details[:8]
        ]
        return "Fix grounding schema validation: " + "; ".join(rows) + truth_table
    message = " ".join(str(error).split())
    if not message:
        return "Return complete grounding rows that pass deterministic validation." + truth_table
    return f"Fix deterministic grounding validation: {message[:1_000]}" + truth_table


def _grounding_validation_code(error: Exception) -> str:
    if isinstance(error, json.JSONDecodeError):
        return "invalid_json"
    if isinstance(error, ValidationError):
        return "wire_schema_invalid"
    if isinstance(error, GroundingValidationError):
        return error.code.value
    return "grounding_validation_failed"


def _grounding_failed_aliases(
    view: GroundingView,
    raw_response: str,
    error: Exception,
) -> tuple[str, ...]:
    if isinstance(error, GroundingValidationError) and error.target_aliases:
        return error.target_aliases
    expected = tuple(target.alias for target in view.targets)
    try:
        payload = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError):
        return expected
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        return expected
    rows = payload["items"]
    invalid_indices: set[int] = set()
    if isinstance(error, ValidationError):
        for detail in error.errors(include_input=False, include_url=False):
            location = detail.get("loc", ())
            if len(location) >= 2 and location[0] == "items" and isinstance(location[1], int):
                invalid_indices.add(location[1])
    aliases = tuple(
        dict.fromkeys(
            str(row.get("target_alias"))
            for index, row in enumerate(rows)
            if isinstance(row, dict)
            and isinstance(row.get("target_alias"), str)
            and (not invalid_indices or index in invalid_indices)
        )
    )
    return aliases or expected


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _text_prompt_record(
    request: TextRequest,
    *,
    purpose: PromptPurpose,
    logical_attempt: int,
) -> ProviderPrompt:
    return ProviderPrompt(
        purpose=purpose,
        provider_role="planner",
        logical_attempt=logical_attempt,
        prompt=request.prompt,
        attachments=tuple(
            PromptAttachment(name=item.name, artifact_ref=item.artifact_ref)
            for item in request.attachments
        ),
        response_contract=(
            request.response_contract.value if request.response_contract is not None else None
        ),
    )
