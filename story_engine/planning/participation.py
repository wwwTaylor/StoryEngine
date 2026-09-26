"""Validate planner observations and resolve first-frame participation locally."""

from __future__ import annotations

from pydantic import Field

from story_engine.domain.common import WireModel
from story_engine.domain.evaluation import CriterionStatus, EvaluationReport, ProviderUsage
from story_engine.domain.first_frame import (
    FirstFrameGuidance,
    FirstFrameHardCondition,
    FirstFrameMode,
    FirstFrameParticipationAssessment,
    FirstFrameParticipationResult,
    GuidanceAction,
    TailFrameInput,
    TailFrameStatus,
)
from story_engine.errors import ContractError
from story_engine.planning.prompt_views import FirstFrameParticipationView


class ParticipationConditionWire(WireModel):
    criterion_id: str
    status: CriterionStatus
    evidence: str = Field(min_length=1, max_length=2_000)


class ParticipationGuidanceWire(WireModel):
    action: GuidanceAction
    canonical_keys: list[str] = Field(min_length=1)
    criterion_ids: list[str] = Field(min_length=1)
    fact: str = Field(min_length=1, max_length=1_000)


class FirstFrameParticipationWireResponse(WireModel):
    conditions: list[ParticipationConditionWire]
    guidance: list[ParticipationGuidanceWire]
    rationale: str = Field(min_length=1, max_length=2_000)


class FirstFrameParticipationValidator:
    def validate(
        self,
        view: FirstFrameParticipationView,
        response: FirstFrameParticipationWireResponse,
    ) -> FirstFrameParticipationAssessment:
        expected_ids = tuple(item.criterion_id for item in view.current_criteria)
        rows = {item.criterion_id: item for item in response.conditions}
        if len(rows) != len(response.conditions) or set(rows) != set(expected_ids):
            raise ContractError("participation conditions must exactly cover current hard criteria")
        statuses = {criterion_id: rows[criterion_id].status for criterion_id in expected_ids}
        current_entity_keys = {item.entity_key for item in view.current_subjects}
        entity_keys = current_entity_keys | {item.entity_key for item in view.previous_subjects}
        allowed_keys = entity_keys | {
            view.previous_scene_key,
            view.current_scene_key,
        }
        guidance: list[FirstFrameGuidance] = []
        for item in response.guidance:
            canonical_keys = tuple(item.canonical_keys)
            criterion_ids = tuple(item.criterion_ids)
            if len(canonical_keys) != len(set(canonical_keys)):
                raise ContractError("participation guidance canonical keys must be unique")
            if len(criterion_ids) != len(set(criterion_ids)):
                raise ContractError("participation guidance criterion IDs must be unique")
            if not set(canonical_keys) <= allowed_keys:
                raise ContractError("participation guidance references unknown canonical keys")
            if not set(criterion_ids) <= set(expected_ids):
                raise ContractError("participation guidance references unknown criteria")
            if item.action == GuidanceAction.PRESERVE:
                if not set(canonical_keys) & current_entity_keys:
                    raise ContractError("preserve guidance must bind a current visible entity")
                if any(
                    statuses[criterion_id] != CriterionStatus.PASS for criterion_id in criterion_ids
                ):
                    raise ContractError(
                        "preserve guidance cannot bind failed or unknown hard conditions"
                    )
            guidance.append(
                FirstFrameGuidance(
                    action=item.action,
                    canonical_keys=canonical_keys,
                    criterion_ids=criterion_ids,
                    fact=item.fact,
                )
            )
        if view.previous_scene_key != view.current_scene_key and expected_ids:
            scene_exclusion = FirstFrameGuidance(
                action=GuidanceAction.DO_NOT_COPY,
                canonical_keys=(view.previous_scene_key, view.current_scene_key),
                criterion_ids=expected_ids,
                fact=(
                    "Do not copy the previous scene background or viewpoint; "
                    "the current scene view is authoritative."
                ),
            )
            if scene_exclusion not in guidance:
                guidance.append(scene_exclusion)
        return FirstFrameParticipationAssessment(
            conditions=tuple(
                FirstFrameHardCondition(
                    criterion_id=criterion_id,
                    status=rows[criterion_id].status,
                    evidence=rows[criterion_id].evidence,
                )
                for criterion_id in expected_ids
            ),
            guidance=tuple(guidance),
            rationale=response.rationale,
        )


def resolve_first_frame_participation(
    *,
    tail: TailFrameInput,
    assessment: FirstFrameParticipationAssessment | None,
    reference_executable: bool,
    unavailable_reason: str | None = None,
    planner_call_ref: str | None = None,
    planner_usage: ProviderUsage | None = None,
) -> FirstFrameParticipationResult:
    if tail.status != TailFrameStatus.AVAILABLE or assessment is None:
        reason = " ".join(
            (unavailable_reason or tail.reason or "participation assessment unavailable").split()
        )[:1_000]
        return FirstFrameParticipationResult(
            tail=tail,
            initial_mode=FirstFrameMode.FRESH,
            unavailable_reason=reason,
            planner_call_ref=planner_call_ref,
            planner_usage=planner_usage or ProviderUsage(),
        )
    failures = assessment.hard_failure_count
    if failures <= 1:
        mode = FirstFrameMode.REUSE
    elif assessment.preserve and reference_executable:
        mode = FirstFrameMode.REFERENCE
    else:
        mode = FirstFrameMode.FRESH
    return FirstFrameParticipationResult(
        tail=tail,
        assessment=assessment,
        initial_mode=mode,
        hard_failure_count=failures,
        planner_call_ref=planner_call_ref,
        planner_usage=planner_usage or ProviderUsage(),
    )


def reference_guidance(
    assessment: FirstFrameParticipationAssessment,
    rejected_reuse: EvaluationReport | None = None,
) -> tuple[FirstFrameGuidance, ...]:
    if rejected_reuse is None:
        return assessment.guidance
    statuses = {item.criterion_id: item.status for item in rejected_reuse.criterion_results}
    return tuple(
        item
        for item in assessment.guidance
        if item.action != GuidanceAction.PRESERVE
        or all(
            statuses.get(criterion_id) == CriterionStatus.PASS
            for criterion_id in item.criterion_ids
        )
    )


def has_preserve_guidance(guidance: tuple[FirstFrameGuidance, ...]) -> bool:
    return any(item.action == GuidanceAction.PRESERVE for item in guidance)
