"""Planner prompt for hard-condition-only first-frame participation assessment."""

from __future__ import annotations

import json

from story_engine.errors import PlanComplexityError
from story_engine.planning.prompt_views import (
    FirstFrameParticipationView,
    RenderedPrompt,
)


def render_first_frame_participation(
    view: FirstFrameParticipationView,
    *,
    max_prompt_characters: int,
) -> RenderedPrompt:
    subject_lines: list[str] = []
    reference_index = 3
    for subject in view.current_subjects:
        image_count = len(subject.reference_images)
        if image_count == 0:
            reference_label = "none"
        elif image_count == 1:
            reference_label = str(reference_index)
        else:
            roles = ",".join(item.role for item in subject.reference_images)
            reference_label = f"{reference_index}-{reference_index + image_count - 1} ({roles})"
        reference_index += image_count
        subject_lines.append(
            "- entity_key="
            + json.dumps(subject.entity_key)
            + "; alias="
            + json.dumps(subject.alias)
            + "; reference_images="
            + reference_label
        )
    criterion_lines = [
        "- criterion_id="
        + json.dumps(item.criterion_id)
        + "; phase="
        + json.dumps(item.phase.value)
        + "; category="
        + json.dumps(item.category)
        + "; statement="
        + json.dumps(item.statement)
        for item in view.current_criteria
    ]
    sections = (
        "Compare the previous real video tail with the current shot's canonical opening truth.",
        "Image 1 is the previous shot's real tail.",
        "Image 2 is the current shot's authoritative scene view.",
        "Image 3+ are compact canonical identity references indexed below.",
        f"Previous scene key: {json.dumps(view.previous_scene_key)}",
        f"Current scene key: {json.dumps(view.current_scene_key)}",
        f"Current scene: {view.current_scene_description}",
        "Previous tail subject keys:",
        *(
            [
                "- entity_key="
                + json.dumps(subject.entity_key)
                + "; alias="
                + json.dumps(subject.alias)
                for subject in view.previous_subjects
            ]
            or ["- none"]
        ),
        "Current visible subjects:",
        *(subject_lines or ["- none"]),
        "Hard conditions to assess:",
        *(criterion_lines or ["- none"]),
        "Return exactly one condition row per listed criterion. Use pass only when Image 1 "
        "visibly satisfies it, fail when Image 1 visibly contradicts it, and unknown when "
        "the valid images do not provide enough evidence.",
        "Do not assess style, lighting, color, composition, or any other soft preference "
        "unless it appears verbatim as a listed hard condition.",
        "Guidance must use exact listed canonical keys and criterion IDs. Preserve guidance "
        "may bind only a condition marked pass. For a cross-scene reference, preserve only "
        "safe subject continuity and mark the previous background and viewpoint do_not_copy.",
        "Do not output or recommend reuse, reference, fresh, or any participation mode. "
        "The local resolver owns that decision and ignores mode language in rationale.",
    )
    prompt = "\n".join(sections)
    if len(prompt) > max_prompt_characters:
        raise PlanComplexityError("first-frame participation prompt exceeds planner context")
    return RenderedPrompt(text=prompt, attachments=view.attachments)
