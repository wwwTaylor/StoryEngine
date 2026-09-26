"""Video motion prompt from an already selected first frame."""

from __future__ import annotations

from story_engine.errors import PlanComplexityError
from story_engine.planning.prompt_views import RenderedPrompt, VideoMotionView


def render_video(
    view: VideoMotionView,
    *,
    max_prompt_characters: int,
    corrections: tuple[str, ...] = (),
) -> RenderedPrompt:
    beats = "\n".join(f"{index}. {beat}" for index, beat in enumerate(view.ordered_beats, start=1))
    end = "; ".join(_deduplicate(view.end_facts))
    avoid = ", ".join(_deduplicate(view.avoid))
    sections = [
        "Animate from the attached start frame.",
        beats,
        f"Camera: {view.camera_behavior}",
        f"Ordered spatial reveal: {'; '.join(view.reveal_content) or 'none'}",
        f"End: {end or 'hold the planned final composition'}",
        f"Avoid: {avoid or 'cuts, text, watermark'}",
    ]
    canonical_prompt = "\n".join(sections)
    additions = tuple(
        correction for correction in _deduplicate(corrections) if correction not in canonical_prompt
    )
    if additions:
        sections.append(f"Correct: {'; '.join(additions)}")
    prompt = "\n".join(sections)
    if len(prompt) > max_prompt_characters:
        raise PlanComplexityError(
            f"canonical video prompt has {len(prompt)} characters; "
            f"provider limit is {max_prompt_characters}"
        )
    return RenderedPrompt(
        text=prompt,
        attachments=(view.start_image, *view.reference_images),
    )


def _deduplicate(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))
