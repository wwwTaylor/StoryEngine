"""Local grounding prompt over one deterministic six-probe set."""

from __future__ import annotations

import json

from story_engine.errors import PlanComplexityError
from story_engine.planning.prompt_views import RenderedPrompt
from story_engine.spatial.grounding import GroundingView


def render_grounding(
    view: GroundingView,
    *,
    max_prompt_characters: int,
    correction: str | None = None,
) -> RenderedPrompt:
    aliases = tuple(target.alias for target in view.targets)
    allowed_aliases = json.dumps(aliases, ensure_ascii=False)
    targets = "\n".join(
        f"- target_alias={json.dumps(target.alias, ensure_ascii=False)}; "
        f"target_kind={json.dumps(target.kind.value)}"
        for target in view.targets
    )
    sections = [
        "Build one station-level Scene Anchor Map using only the six attached probes.",
        f"Scene alias: {view.scene_alias}",
        f"Station key: {view.station_key}",
        f"Allowed target_alias values (JSON): {allowed_aliases}",
        "Target rows (target_kind is context, never part of target_alias):",
        targets,
        "For every anchor return exactly one observation with observation_index=1 and a "
        "boolean is_unique. is_unique is true only for one static, local, clearly bounded "
        "instance. Repeated, distributed, ambiguous, or oversized structures are not unique.",
        "For every region return one row per distinct visible instance, numbered from 1, "
        "with is_unique=null. If a target is not visible, return exactly one row with "
        "observation_index=1, normalized_box=null, and its actual visibility.",
        "Use fixed probe roles front, right, back, left, up, or down. Apply this truth table "
        "to every item before returning:",
        "- visibility=visible or partially_visible: normalized_box MUST be an object tightly "
        "covering only the actually visible pixels.",
        "- visibility=not_visible or unknown: normalized_box MUST be null.",
        "- Never emit occluded. Do not infer the hidden extent of a partially visible target.",
        "Copy every JSON target_alias verbatim. Never append a colon, target_kind, synonym, "
        "description, or umbrella label to target_alias.",
        "Do not infer hidden geometry or discuss the wider story.",
    ]
    if correction is not None:
        sections.extend(
            (
                f"Current deterministic validation correction: {correction}",
                f"Discard every prior target_alias value. The complete legal set is "
                f"{allowed_aliases}; use only those strings exactly.",
                "Recheck the visibility/normalized_box truth table for every item. A box is "
                "legal only with visible or partially_visible; not_visible and unknown require "
                "null. Never emit occluded.",
                "Return a new complete response from the original probes with this correction "
                "applied; do not return a patch or explanation.",
            )
        )
    prompt = "\n".join(sections)
    if len(prompt) > max_prompt_characters:
        raise PlanComplexityError(
            f"canonical grounding prompt has {len(prompt)} characters; "
            f"provider limit is {max_prompt_characters}"
        )
    return RenderedPrompt(
        text=prompt,
        attachments=tuple(probe.artifact_ref for probe in view.probe_set.probes),
    )
