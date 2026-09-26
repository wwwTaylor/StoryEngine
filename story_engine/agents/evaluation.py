"""EvaluationAgent reports evidence without selecting or retrying."""

from __future__ import annotations

import json

from pydantic import Field

from story_engine.domain.common import WireModel
from story_engine.domain.evaluation import (
    CandidateMedia,
    CandidateRecord,
    CriterionResult,
    CriterionStatus,
    EvaluationReport,
    ProviderUsage,
    TechnicalStatus,
)
from story_engine.errors import ContractError, ProviderError, ProviderErrorKind
from story_engine.ids import canonical_hash, stable_key
from story_engine.planning.prompt_views import EvaluationView
from story_engine.prompts.judge import render_evaluation
from story_engine.providers.ports import (
    Attachment,
    JudgeProvider,
    JudgeRequest,
    ResponseContract,
)
from story_engine.task_pool import TaskPool


class EvaluationWireRow(WireModel):
    criterion_id: str
    status: CriterionStatus
    evidence: str = Field(min_length=1, max_length=2_000)


class EvaluationWireResponse(WireModel):
    items: list[EvaluationWireRow]


class EvaluationAgent:
    def __init__(self, provider: JudgeProvider, task_pool: TaskPool) -> None:
        self.provider = provider
        self.task_pool = task_pool

    async def evaluate(
        self,
        candidate: CandidateRecord,
        view: EvaluationView,
        *,
        prompt: str | None = None,
    ) -> EvaluationReport:
        if candidate.technical_status != TechnicalStatus.VALID or candidate.artifact_ref is None:
            raise ContractError("technical invalid candidates cannot be evaluated")
        capability = self.provider.capability_profile.judge
        if capability is None:
            raise ContractError("JudgeProvider exposes no judge capability")
        rendered_prompt = prompt if prompt is not None else self.render_prompt(view)
        media = candidate.media or (
            CandidateMedia(role="candidate", artifact_ref=candidate.artifact_ref),
        )
        if len(media) > capability.max_media:
            raise ContractError(
                f"candidate evaluation needs {len(media)} media inputs; "
                f"judge supports {capability.max_media}"
            )
        request = JudgeRequest(
            prompt=rendered_prompt,
            media=tuple(
                Attachment(name=item.role, artifact_ref=item.artifact_ref) for item in media
            ),
            response_contract=ResponseContract.EVALUATION_ROWS,
        )
        async with self.task_pool.slot():
            result = await self.provider.judge(request)
        wire = self._parse(result.text)
        expected = {criterion.criterion_id: criterion for criterion in view.criteria}
        rows = {row.criterion_id: row for row in wire.items}
        if len(rows) != len(wire.items) or set(rows) != set(expected):
            raise ProviderError(
                kind=ProviderErrorKind.RESPONSE_CONTRACT,
                message="evaluation rows do not exactly cover current criteria",
                retryable=False,
            )
        criteria = tuple(
            CriterionResult(
                criterion_id=criterion_id,
                kind=expected[criterion_id].kind,
                priority=expected[criterion_id].priority,
                status=rows[criterion_id].status,
                evidence=rows[criterion_id].evidence,
                category=expected[criterion_id].category,
                owner_shot_key=expected[criterion_id].owner_shot_key,
            )
            for criterion_id in sorted(expected)
        )
        payload = {
            "candidate_key": candidate.candidate_key,
            "criterion_results": criteria,
            "evaluator_ref": result.metrics.call_ref,
        }
        return EvaluationReport(
            report_key=stable_key("evaluation", candidate.candidate_key, canonical_hash(payload)),
            candidate_key=candidate.candidate_key,
            criterion_results=criteria,
            evaluator_ref=result.metrics.call_ref,
            evaluator_usage=ProviderUsage(
                call_count=1,
                transport_retries=result.metrics.transport_retries,
                elapsed_seconds=result.metrics.elapsed_seconds,
                known_cost_usd=result.metrics.cost_usd,
            ),
        )

    def render_prompt(self, view: EvaluationView) -> str:
        capability = self.provider.capability_profile.judge
        if capability is None:
            raise ContractError("JudgeProvider exposes no judge capability")
        return render_evaluation(
            view,
            max_prompt_characters=capability.max_prompt_characters,
        )

    @staticmethod
    def unknown_report(
        candidate: CandidateRecord,
        view: EvaluationView,
        *,
        evaluator_ref: str,
        evidence: str,
        evaluator_usage: ProviderUsage | None = None,
    ) -> EvaluationReport:
        criteria = tuple(
            CriterionResult(
                criterion_id=item.criterion_id,
                kind=item.kind,
                priority=item.priority,
                status=CriterionStatus.UNKNOWN,
                evidence=evidence,
                category=item.category,
                owner_shot_key=item.owner_shot_key,
            )
            for item in view.criteria
        )
        payload = {
            "candidate_key": candidate.candidate_key,
            "criterion_results": criteria,
            "evaluator_ref": evaluator_ref,
        }
        return EvaluationReport(
            report_key=stable_key("evaluation", candidate.candidate_key, canonical_hash(payload)),
            candidate_key=candidate.candidate_key,
            criterion_results=criteria,
            evaluator_ref=evaluator_ref,
            evaluator_usage=evaluator_usage or ProviderUsage(),
        )

    @staticmethod
    def _parse(text: str) -> EvaluationWireResponse:
        try:
            raw = json.loads(text)
            return EvaluationWireResponse.model_validate(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderError(
                kind=ProviderErrorKind.RESPONSE_CONTRACT,
                message="judge returned invalid evaluation rows",
                retryable=False,
            ) from exc
