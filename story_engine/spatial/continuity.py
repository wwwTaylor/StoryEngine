"""Deterministic adjacent-shot spatial continuity contracts."""

from __future__ import annotations

from story_engine.domain.common import FrozenModel
from story_engine.domain.story import StoryPlan


class SpatialContinuityIssue(FrozenModel):
    code: str
    previous_shot: str
    current_shot: str
    detail: str


def validate_spatial_continuity(
    story_plan: StoryPlan,
) -> tuple[SpatialContinuityIssue, ...]:
    issues: list[SpatialContinuityIssue] = []
    for previous, current in zip(
        story_plan.ordered_shots, story_plan.ordered_shots[1:], strict=False
    ):
        if previous.scene_key != current.scene_key:
            continue
        previous_camera = previous.spatial_intent
        current_camera = current.spatial_intent
        if (
            previous_camera.axis_key
            and previous_camera.axis_key == current_camera.axis_key
            and previous_camera.camera_side
            and current_camera.camera_side
            and previous_camera.camera_side != current_camera.camera_side
        ):
            issues.append(
                SpatialContinuityIssue(
                    code="axis_crossing",
                    previous_shot=previous.shot_key,
                    current_shot=current.shot_key,
                    detail="camera side changes across an established axis",
                )
            )
        if (
            previous_camera.screen_direction
            and current_camera.screen_direction
            and previous_camera.screen_direction != current_camera.screen_direction
        ):
            issues.append(
                SpatialContinuityIssue(
                    code="screen_direction_change",
                    previous_shot=previous.shot_key,
                    current_shot=current.shot_key,
                    detail="screen direction changes without a planned reset",
                )
            )
        expected_entry = {"left": "right", "right": "left"}.get(previous_camera.exit_edge or "")
        if (
            expected_entry
            and current_camera.entry_edge
            and current_camera.entry_edge != expected_entry
        ):
            issues.append(
                SpatialContinuityIssue(
                    code="entry_exit_mismatch",
                    previous_shot=previous.shot_key,
                    current_shot=current.shot_key,
                    detail=f"expected entry {expected_entry}, got {current_camera.entry_edge}",
                )
            )
    return tuple(issues)
