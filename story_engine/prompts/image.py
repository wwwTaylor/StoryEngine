"""Reference and first-frame image prompts."""

from __future__ import annotations

from story_engine.domain.first_frame import FirstFrameGuidance, GuidanceAction
from story_engine.domain.reference import (
    CharacterReferenceRecipe,
    CharacterViewRole,
    GuidedCharacterReferenceRecipe,
    GuidedScenePanoramaRecipe,
    PropReferenceRecipe,
    ReferenceRecipe,
    ScenePanoramaRecipe,
)
from story_engine.errors import PlanComplexityError
from story_engine.planning.prompt_views import (
    FirstFrameView,
    NovelStationView,
    RenderedPrompt,
)
from story_engine.storage import ArtifactRef


def render_reference_image(recipe: ReferenceRecipe) -> str:
    if isinstance(recipe, CharacterReferenceRecipe):
        return render_character_reference(
            recipe,
            "front",
            guided_source=isinstance(recipe, GuidedCharacterReferenceRecipe),
        )
    if isinstance(recipe, PropReferenceRecipe):
        avoid = ", ".join(recipe.forbidden_co_subjects)
        state = f"\nVisible state: {recipe.visible_state}" if recipe.visible_state else ""
        return (
            "Create an isolated prop reference image.\n"
            f"Appearance: {recipe.appearance}{state}\n"
            f"Scale: {recipe.scale_cues}\n"
            f"Style: {recipe.style}\n"
            f"Avoid: other named subjects{': ' + avoid if avoid else ''}, text, watermark."
        )
    if isinstance(recipe, ScenePanoramaRecipe):
        layout = "; ".join(recipe.semantic_layout) or "no additional relation"
        elements = "; ".join(recipe.zones_and_landmarks)
        source_guidance = (
            (
                "Image 1 is the uploaded scene reference. Preserve its visible fixed "
                "architecture, layout, materials, lighting character, and visual style. "
                "Extend unseen directions coherently; inferred content must not contradict "
                "anything visible in Image 1."
            )
            if isinstance(recipe, GuidedScenePanoramaRecipe)
            else ""
        )
        return "\n".join(
            item
            for item in (
                "Create one seamless 2:1 equirectangular panorama.",
                source_guidance,
                f"Scene: {recipe.scene_visual_identity}",
                f"Explicit source station: {recipe.source_station.description}",
                f"Source-station zone key: {recipe.source_station.zone_key}",
                f"Zones, scene regions, and unique local anchors: {elements}",
                f"Semantic layout: {layout}",
                f"Lighting: {recipe.lighting}",
                f"Style: {recipe.style}",
                "Population: empty environment only; no characters or movable story props.",
                "Render from exactly the declared source station at eye level. Make every "
                "named anchor a single local structure with clear boundaries.",
                "Avoid: people, movable story props, duplicated anchors, broken seams, "
                "text, logos, watermark.",
            )
            if item
        )
    raise TypeError("provided assets are normalized, not generated")


def render_character_reference(
    recipe: CharacterReferenceRecipe,
    role: CharacterViewRole,
    *,
    guided_source: bool = False,
    corrections: tuple[str, ...] = (),
) -> str:
    avoid = ", ".join(recipe.forbidden_co_subjects)
    orientation = {
        "front": (
            "Image 1 is the authoritative uploaded character reference. Generate exactly "
            "one normalized full-body front view of that exact person or character. Preserve "
            "identity, face, hair, body proportions, clothing, colors, materials, and "
            "distinctive attachments; do not redesign, beautify, or substitute them."
            if guided_source
            else "Generate exactly one full-body front view. The character faces straight "
            "toward the camera in a neutral reference pose."
        ),
        "side": (
            "Image 1 is the authoritative front view. Generate exactly one full-body "
            "left-facing side view of that exact character."
        ),
        "back": (
            "Image 1 is the authoritative front view. Generate exactly one full-body "
            "back view of that exact character; no face should be visible."
        ),
    }[role]
    identity = (
        "Preserve the exact identity, silhouette, proportions, colors, materials, "
        "clothing, and attachments from Image 1. Do not redesign any part."
        if role != "front" or guided_source
        else ""
    )
    sections = [
        "Create an isolated character reference image.",
        orientation,
        identity,
        f"Appearance: {recipe.appearance}",
        f"Clothing: {recipe.clothing or 'as described in appearance'}",
        f"Pose: {recipe.neutral_pose}",
        f"Style: {recipe.style}",
        "Use one uncropped full-frame portrait on a plain uniform background.",
        "Avoid: other named subjects"
        f"{': ' + avoid if avoid else ''}, text, watermark, grid, collage, panel, "
        "turnaround sheet, or multiple views.",
    ]
    additions = _new_corrections(sections, corrections)
    if additions:
        sections.append(f"Correct: {'; '.join(additions)}")
    return "\n".join(section for section in sections if section)


def render_first_frame(
    view: FirstFrameView,
    *,
    max_prompt_characters: int,
    corrections: tuple[str, ...] = (),
) -> RenderedPrompt:
    sections = [
        *_scene_reference_sections(),
        *_first_frame_sections(view, reference_start_index=3),
    ]
    additions = _new_corrections(sections, corrections)
    if additions:
        sections.append(f"Correct: {'; '.join(additions)}")
    prompt = "\n".join(sections)
    _ensure_within_context(prompt, max_prompt_characters)
    return RenderedPrompt(
        text=prompt,
        attachments=(view.scene_view, view.scene_panorama, *view.reference_images),
    )


def render_reference_first_frame(
    view: FirstFrameView,
    *,
    previous_tail: ArtifactRef,
    guidance: tuple[FirstFrameGuidance, ...],
    max_prompt_characters: int,
    corrections: tuple[str, ...] = (),
) -> RenderedPrompt:
    preserve = tuple(
        _guidance_text(item) for item in guidance if item.action == GuidanceAction.PRESERVE
    )
    do_not_copy = tuple(
        _guidance_text(item) for item in guidance if item.action == GuidanceAction.DO_NOT_COPY
    )
    sections = [
        *_scene_reference_sections(),
        "Image 3 is the previous real tail and is continuity reference only.",
        *_first_frame_sections(view, reference_start_index=4),
        f"Preserve from Image 3: {'; '.join(preserve) or 'nothing'}",
        f"Do not copy from Image 3: {'; '.join(do_not_copy) or 'nothing'}",
        "Never replace the current scene background or viewpoint with Image 3.",
    ]
    additions = _new_corrections(sections, corrections)
    if additions:
        sections.append(f"Correct: {'; '.join(additions)}")
    prompt = "\n".join(sections)
    _ensure_within_context(prompt, max_prompt_characters)
    return RenderedPrompt(
        text=prompt,
        attachments=(
            view.scene_view,
            view.scene_panorama,
            previous_tail,
            *view.reference_images,
        ),
    )


def _scene_reference_sections() -> tuple[str, str]:
    return (
        "Create the opening frame. Image 1 is the authoritative current scene view and "
        "defines the exact opening viewpoint and visible background.",
        "Image 2 is the matching full scene panorama. Use it only to preserve the global "
        "room layout and static architecture beyond Image 1; do not copy its equirectangular "
        "projection or let it replace Image 1's camera.",
    )


def _first_frame_sections(
    view: FirstFrameView,
    *,
    reference_start_index: int,
) -> list[str]:
    subjects = "; ".join(
        _subject_text(subject.alias, subject.t0_facts, subject.semantic_description)
        for subject in view.visible_subjects
    )
    ordered_motion = "; ".join(
        f"{index}. {action}" for index, action in enumerate(view.ordered_beats, start=1)
    )
    shot_constraints = "; ".join(view.shot_constraints)
    avoid = ", ".join(_deduplicate(view.avoid))
    reference_lines = _subject_reference_lines(
        view,
        start_index=reference_start_index,
    )
    return [
        f"Scene: {view.scene_description}",
        f"Visible at t0: {subjects or 'no named subject'}",
        *(
            (
                "Subject identity references:",
                *reference_lines,
                "Use these images for identity only; do not copy their neutral pose "
                "or plain background.",
            )
            if reference_lines
            else ()
        ),
        "Upcoming ordered motion is staging context only; do not complete later beats "
        "or replace them with a different action in this still.",
        f"Upcoming ordered motion: {ordered_motion}",
        f"Shot-local complete-video constraints: {shot_constraints or 'none'}",
        "Stage t0 so the listed motion can proceed. Do not invent an alternate tool, "
        "container, recipient, or action target that conflicts with it.",
        f"Composition: {view.composition}",
        f"Camera: {view.camera}",
        f"Style: {view.visual_style}",
        f"Avoid: {avoid or 'text, watermark'}",
    ]


def _subject_reference_lines(
    view: FirstFrameView,
    *,
    start_index: int,
) -> tuple[str, ...]:
    lines: list[str] = []
    image_index = start_index
    for subject in view.subject_references:
        count = len(subject.images)
        label = (
            f"Image {image_index}"
            if count == 1
            else f"Images {image_index}-{image_index + count - 1}"
        )
        roles = ", ".join(image.role for image in subject.images)
        lines.append(f"- {label}: {subject.alias} ({roles}).")
        image_index += count
    return tuple(lines)


def _guidance_text(guidance: FirstFrameGuidance) -> str:
    return (
        f"{guidance.fact} "
        f"(canonical_keys={','.join(guidance.canonical_keys)}; "
        f"criterion_ids={','.join(guidance.criterion_ids)})"
    )


def render_novel_station(
    view: NovelStationView,
    *,
    max_prompt_characters: int,
    corrections: tuple[str, ...] = (),
) -> RenderedPrompt:
    sections = [
        (
            "Materialize one seamless 2:1 equirectangular 360-degree panorama from the new "
            "translated camera station in the attached canonical scene."
        ),
        f"Scene: {view.scene_description}",
        f"New station: {view.station_description}",
        f"The panorama must support this later perspective crop: {view.composition}",
        f"Fixed landmarks: {'; '.join(view.landmarks) or 'none specified'}",
        f"Required shot content: {'; '.join(view.required_content) or 'none specified'}",
        f"Semantic layout: {'; '.join(view.semantic_layout) or 'none specified'}",
        f"Style: {view.visual_style}",
        f"Avoid: {', '.join(_deduplicate(view.avoid)) or 'text, watermark'}",
        (
            "Preserve landmark identity and relations across the full sphere. The left and "
            "right edges must join continuously. Do not copy probe borders or invent text."
        ),
    ]
    additions = _new_corrections(sections, corrections)
    if additions:
        sections.append(f"Correct: {'; '.join(additions)}")
    prompt = "\n".join(sections)
    _ensure_within_context(prompt, max_prompt_characters)
    return RenderedPrompt(
        text=prompt,
        attachments=(view.panorama, *view.probes),
    )


def _subject_text(alias: str, facts: tuple[str, ...], description: str | None) -> str:
    components = [alias]
    if description:
        components.append(description)
    components.extend(_deduplicate(facts))
    return ", ".join(components)


def _deduplicate(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


def _new_corrections(
    sections: list[str],
    corrections: tuple[str, ...],
) -> tuple[str, ...]:
    canonical_prompt = "\n".join(sections)
    return tuple(
        correction for correction in _deduplicate(corrections) if correction not in canonical_prompt
    )


def _ensure_within_context(prompt: str, maximum: int) -> None:
    if len(prompt) > maximum:
        raise PlanComplexityError(
            f"canonical image prompt has {len(prompt)} characters; provider limit is {maximum}"
        )
