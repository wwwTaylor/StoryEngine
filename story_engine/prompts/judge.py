"""Stable evaluation prompt; response schema travels in the API contract."""

from __future__ import annotations

from story_engine.errors import PlanComplexityError
from story_engine.planning.prompt_views import EvaluationView


def render_evaluation(view: EvaluationView, *, max_prompt_characters: int) -> str:
    rows = "\n".join(
        f"- {item.criterion_id}: [{item.phase.value}] {item.statement}" for item in view.criteria
    )
    prompt = "\n".join(
        (
            f"Evaluate only the attached media for operation {view.operation}.",
            *view.media_instructions,
            f"Return exactly {len(view.criteria)} rows: copy every criterion ID below "
            "exactly once, with no omitted, duplicate, renamed, or extra IDs.",
            "Each row needs status pass, fail, or unknown and short pixel-grounded evidence.",
            rows,
            "Do not choose a candidate and do not infer facts from the plan "
            "when pixels are unclear.",
        )
    )
    if len(prompt) > max_prompt_characters:
        raise PlanComplexityError(
            f"canonical evaluation prompt has {len(prompt)} characters; "
            f"provider limit is {max_prompt_characters}"
        )
    return prompt
