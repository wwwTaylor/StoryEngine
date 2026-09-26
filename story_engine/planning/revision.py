"""Criterion-indexed correction ledger rebuilt from canonical input."""

from __future__ import annotations

from story_engine.domain.common import FrozenModel
from story_engine.domain.evaluation import CriterionStatus, EvaluationReport
from story_engine.errors import ContractError
from story_engine.planning.prompt_views import EvaluationView


class IssueCorrection(FrozenModel):
    criterion_id: str
    correction: str


class IssueLedger(FrozenModel):
    entries: tuple[IssueCorrection, ...] = ()

    def corrections(self) -> tuple[str, ...]:
        return tuple(item.correction for item in self.entries)

    def replace_from_report(
        self, report: EvaluationReport, evaluation_view: EvaluationView
    ) -> IssueLedger:
        criteria = {criterion.criterion_id: criterion for criterion in evaluation_view.criteria}
        current = {entry.criterion_id: entry for entry in self.entries}
        for result in report.criterion_results:
            criterion = criteria.get(result.criterion_id)
            if criterion is None:
                continue
            if result.owner_shot_key != criterion.owner_shot_key:
                raise ContractError(f"criterion ownership mismatch for {result.criterion_id}")
            if result.status == CriterionStatus.PASS:
                current.pop(result.criterion_id, None)
            else:
                # Use the canonical criterion, not evaluator prose or issue keywords.
                current[result.criterion_id] = IssueCorrection(
                    criterion_id=result.criterion_id,
                    correction=criterion.statement,
                )
        # Keep unresolved entries that this report did not cover. Exact wire
        # coverage validation normally makes this branch empty.
        return IssueLedger(
            entries=tuple(sorted(current.values(), key=lambda item: item.criterion_id))
        )
