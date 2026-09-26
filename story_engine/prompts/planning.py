"""Story planning prompt."""

from __future__ import annotations

from story_engine.planning.prompt_views import StoryPlanningView


def render_story_planning(
    view: StoryPlanningView,
    *,
    correction: str | None = None,
) -> str:
    required = ", ".join(view.required_entities) or "none specified"
    forbidden = "; ".join(view.forbidden_content) or "none specified"
    assets = "; ".join(view.provided_asset_summaries) or "none"
    bindings = "; ".join(view.provided_asset_binding_summaries) or "none"
    durations = ", ".join(str(value) for value in view.supported_durations)
    sections = [
        "Create a concise StoryDraft for a deterministic video workflow.",
        f"Idea: {view.idea}",
        f"Exactly {view.shot_target} ordered shots.",
        f"Style: {view.visual_style}",
        f"Output language: {view.output_language}",
        f"Required entities: {required}",
        f"Forbidden content: {forbidden}",
        f"Allowed shot durations in seconds: {durations}",
        f"Provided asset summaries: {assets}",
        f"Confirmed provided-asset bindings: {bindings}",
        "First-frame image input limit (including one scene view and one matching scene "
        f"panorama): {view.image_input_limit}.",
        "Each frozen character visible in the opening uses three identity-image slots "
        "(front, side, back); each other frozen entity uses one.",
        f"Additional video reference-image limit after the opening frame: "
        f"{view.video_reference_limit}.",
        f"Per-shot hard complexity limits: at most {view.max_beats_per_shot} ordered "
        f"beats and at most {view.max_visible_entities_per_shot} visible entities.",
        "Use short aliases. Describe creative facts only; do not create IDs, hashes, "
        "paths, prompts, state snapshots, provider requests, or camera matrices.",
        "Draft requirements are only additional shot-local constraints. Every requirement "
        "must name exactly one shot_alias and must be independently pixel-observable in that "
        "shot's complete video. Never ask one shot to prove events from another shot. For a "
        "shot with a mid-shot scene change, the requirement must be independently "
        "pixel-observable in the shot's final scene segment.",
        "Represent narrative sequence through ordered shots and ordered beats. For every "
        "explicit cross-shot before/after dependency in the idea, add a typed shot_order "
        "plan invariant; do not restate the whole sequence as a media requirement.",
        "Cross-shot identity is represented by reusing one frozen entity alias and its "
        "canonical references. If an extra visual condition is needed in several shots, "
        "emit a separate locally worded requirement for each owning shot.",
        "Do not restate style, forbidden content, shot duration, resolution, FPS, audio, "
        "output language, or provider limits; those are enforced outside the StoryDraft.",
        "Alias closure is mandatory: every alias reference must resolve to a declaration "
        "in this same draft; never use an undeclared noun as an alias.",
        "Separate spatial declarations strictly. scene_regions are large, repeated, or "
        "distributed content such as rows of booths, windows, garden beds, long bars, or "
        "awnings. anchor_landmarks are static, single-instance, local structures with one "
        "clear visual boundary, such as a door frame, gate post, bed corner, or stall "
        "support. Never declare a person, movable prop, zone, open area, repeated object, "
        "or whole region as an anchor_landmark.",
        "Every scene region and anchor zone_alias must name one of that scene's zones. "
        "A scene region may list only local anchor aliases from the same zone as "
        "representative_anchor_aliases. Each semantic relation endpoint must name a zone, "
        "scene region, or anchor declared in that same scene.",
        "Entity placements and transitions may reference only declared scenes, zones, and "
        "entities. Shot scene/action-zone/visible/content aliases, requirement shot_alias, "
        "and plan-invariant shot aliases may reference only their matching declared alias "
        "domain.",
        "Every entity changed by a beat transition, and every entity used as that "
        "transition's placement target, must also appear in the same shot's "
        "visible_entity_aliases. Count them within the per-shot visible-entity limit.",
        "Placement target kinds are strict: on_surface targets a surface entity, "
        "in_container targets a container entity, held_by targets a character entity, "
        "and attached_to may target any other entity.",
        "Use every required entity string as an exact entity alias and make it visible "
        "in at least one shot.",
        "When an entity or scene must use a provided asset, copy its listed asset ID "
        "exactly into provided_asset_id; never infer a binding by similar names.",
        "Confirmed provided-asset bindings are authoritative. For each one, create exactly "
        "one entity or scene of the matching kind using its exact alias and asset ID. Display "
        "names are labels only and must never cause an additional entity or scene. Every bound "
        "character must be visible in at least one shot and every bound scene must be used by at "
        "least one shot, either as a shot's scene_alias or as a beat-level scene change target.",
        "Mark an attribute is_visual only when its values must look different in pixels; "
        "knowledge, intent, and other narrative facts are not visual reference states.",
        "Each shot must have one purpose, one spatial_intent, one action zone, one camera "
        "motion intent, and a short ordered beat list. story_target_aliases contain only "
        "visible entity aliases and express narrative importance; they are never camera "
        "aim points and never refer to zones, regions, or anchors.",
        "A shot normally uses exactly one scene for its whole duration, and its scene_alias "
        "is that scene. Only when the idea explicitly requires a scene change inside one shot "
        "(for example a character running through a portal from one scene into another and "
        "back) set scene_alias on the beat where the new scene begins, and describe that beat "
        "and all later beats as happening in that new scene. Returning to an earlier scene "
        "counts as another change. At most 2 scene changes per shot (at most 3 scene "
        "segments); whenever the idea allows, stage scene changes at shot boundaries instead. "
        "A scene change must be demanded by the idea; never use it just to vary the look. "
        "Attach the transition that moves the crossing character into the new scene to the "
        "last beat of the outgoing scene segment, so the new scene segment opens with the "
        "character already present in it. Every scene segment of a shot needs at least one "
        "story target that is present in that scene at the segment's opening.",
        "spatial_content contains only scene-region or anchor aliases. Use presentation "
        "simultaneous only when content must share the opening static frame; use sequential "
        "when the same shot may reveal it through motion. Use strength must only when "
        "dropping the content would violate the idea or a hard requirement; otherwise use "
        "preferred.",
        "viewpoint_intent describes a semantic scene location, never a station key. Set "
        "translation_expected=true when the view clearly requires moving to a different "
        "place such as a room end, entrance, or opposite side. Do not choose a final aim "
        "anchor, station key, yaw, pitch, or field of view.",
        "Set framing_scale to tight, medium, or wide as the structured field-of-view "
        "intent; keep framing as a short creative composition description.",
        "For adjacent shots on the same axis, keep camera_side (left or right) unchanged. "
        "Keep screen_direction unchanged when horizontal movement continues; otherwise "
        "set it to null. entry_edge and exit_edge, when used, are left or right only.",
        "Respect media input reachability. An entity with frozen appearance that is needed "
        "during a shot must be visible in that shot's planned opening state unless an "
        "additional video reference slot is available. Each distinct visual attribute "
        "state introduced after the opening consumes another video reference slot. For a "
        "start-only video provider, stage discoveries and visual state changes at a cut so "
        "the required entity/state is already visible in the next shot's opening frame.",
        "When the additional video reference limit is zero, every frozen entity listed in "
        "visible_entity_aliases must have a planned t0 placement that is neither offscreen "
        "nor in_container. Make a revealed object partly visible at t0, move its reveal "
        "across a cut, or use freeze_appearance=false only for a generic prop whose identity "
        "does not need cross-shot preservation.",
        "A sequential reveal must be plausible as one simple pan, arc, or reveal within the "
        "shot's duration. Do not use sequential presentation to disguise an explicit "
        "simultaneous hard requirement.",
    ]
    if correction is not None:
        sections.extend(
            (
                "The previous full draft was rejected by deterministic validation.",
                f"Current correction: {correction}",
                "Return a new complete draft from the original facts with this correction "
                "applied. Do not discuss the error or return a patch.",
            )
        )
    return "\n".join(sections)
