"""Deterministic selection over immutable observations."""

from __future__ import annotations

from story_engine.domain.evaluation import (
    CandidateRank,
    CandidateRecord,
    CriterionKind,
    CriterionStatus,
    EvaluationReport,
    RankVector,
    SelectionDecision,
    SelectionOutcome,
    TechnicalStatus,
)
from story_engine.errors import ContractError
from story_engine.ids import canonical_hash, stable_key


class SelectionPolicy:
    """Apply the fixed lexicographic policy without mutating reports."""

    def select(
        self,
        operation_key: str,
        candidates: tuple[CandidateRecord, ...],
        reports: tuple[EvaluationReport, ...],
    ) -> SelectionDecision:
        eligible = tuple(
            sorted(
                (
                    candidate
                    for candidate in candidates
                    if candidate.technical_status == TechnicalStatus.VALID
                    and candidate.artifact_ref is not None
                ),
                key=lambda item: item.candidate_key,
            )
        )
        if not eligible:
            raise ContractError(f"operation {operation_key} has no technically valid candidate")
        candidate_keys = {candidate.candidate_key for candidate in eligible}
        reports_by_candidate = {report.candidate_key: report for report in reports}
        if len(reports_by_candidate) != len(reports):
            raise ContractError("multiple evaluation reports exist for one candidate")
        missing = candidate_keys - set(reports_by_candidate)
        if missing:
            raise ContractError(f"eligible candidates lack evaluation reports: {sorted(missing)}")

        ranks: list[CandidateRank] = []
        for candidate in eligible:
            report = reports_by_candidate[candidate.candidate_key]
            vector = self._rank(candidate, report)
            ranks.append(CandidateRank(candidate_key=candidate.candidate_key, vector=vector))
        ranks.sort(key=lambda item: item.vector.sort_key(item.candidate_key))
        selected_rank = ranks[0]
        selected_report = reports_by_candidate[selected_rank.candidate_key]
        compliant = all(
            result.status == CriterionStatus.PASS
            for result in selected_report.criterion_results
            if result.kind == CriterionKind.REQUIREMENT
        )
        outcome = (
            SelectionOutcome.SELECTED_COMPLIANT if compliant else SelectionOutcome.SELECTED_DEGRADED
        )
        eligible_keys = tuple(item.candidate_key for item in eligible)
        rank_records = tuple(ranks)
        payload = {
            "operation_key": operation_key,
            "eligible_candidates": eligible_keys,
            "ranks": rank_records,
            "selected_candidate": selected_rank.candidate_key,
            "selected_report": selected_report.report_key,
            "outcome": outcome,
        }
        return SelectionDecision(
            decision_key=stable_key("selection", operation_key, canonical_hash(payload)),
            operation_key=operation_key,
            eligible_candidates=eligible_keys,
            ranks=rank_records,
            selected_candidate=selected_rank.candidate_key,
            selected_report=selected_report.report_key,
            outcome=outcome,
        )

    @staticmethod
    def _rank(candidate: CandidateRecord, report: EvaluationReport) -> RankVector:
        requirements = tuple(
            result
            for result in report.criterion_results
            if result.kind == CriterionKind.REQUIREMENT
        )
        preferences = tuple(
            result for result in report.criterion_results if result.kind == CriterionKind.PREFERENCE
        )
        requirement_failures = sum(result.status == CriterionStatus.FAIL for result in requirements)
        requirement_unknowns = sum(
            result.status == CriterionStatus.UNKNOWN for result in requirements
        )
        priority_loss = sum(
            result.priority for result in requirements if result.status != CriterionStatus.PASS
        )
        continuity_loss = sum(
            result.priority
            for result in requirements
            if result.category in {"continuity", "identity", "key_state"}
            and result.status != CriterionStatus.PASS
        )
        preference_loss = sum(
            result.priority for result in preferences if result.status != CriterionStatus.PASS
        )
        return RankVector(
            requirement_failures=requirement_failures,
            requirement_unknowns=requirement_unknowns,
            priority_loss=priority_loss,
            continuity_loss=continuity_loss,
            preference_loss=preference_loss,
            technical_quality=candidate.technical_quality,
        )
