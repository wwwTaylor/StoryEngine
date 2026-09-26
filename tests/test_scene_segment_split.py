from __future__ import annotations

import unittest
from types import SimpleNamespace

from story_engine.domain.request import ProjectRequest, ProvidedAssetBinding
from story_engine.domain.state import EntityKind
from story_engine.domain.story import FramingScale, RequirementKind
from story_engine.errors import StoryCompilationError
from story_engine.planning.story_compiler import StoryCompileConstraints, StoryCompiler
from story_engine.planning.story_planner import (
    DraftBeat,
    DraftEntity,
    DraftPlacement,
    DraftPlanInvariant,
    DraftRequirement,
    DraftScene,
    DraftShot,
    DraftShotSpatialIntent,
    DraftTransition,
    DraftViewpointIntent,
    DraftZone,
    StoryDraft,
)
from story_engine.providers.ports import ResponseContract
from story_engine.providers.schemas import schema_for_contract


def make_request(shot_target: int = 1) -> ProjectRequest:
    return ProjectRequest(
        task_id="segment-split-test",
        idea="A courier runs through a portal from a station into a canyon and back.",
        shot_target=shot_target,
        visual_style="cinematic realism",
        resolution={"width": 1280, "height": 720},
        generation_requirements={"asset_alignment_mode": "auto"},
    )


def make_scene(alias: str) -> DraftScene:
    return DraftScene(
        alias=alias,
        visual_identity=f"{alias} visual identity",
        zones=[DraftZone(alias="center", description=f"center of {alias}")],
        lighting="soft daylight",
        style="cinematic realism",
    )


def make_entity(alias: str, scene_alias: str, zone_alias: str = "center") -> DraftEntity:
    return DraftEntity(
        alias=alias,
        kind=EntityKind.CHARACTER,
        visual_identity=f"{alias} appearance",
        initial_placement=DraftPlacement(
            kind="in_scene_zone",
            scene_alias=scene_alias,
            zone_alias=zone_alias,
        ),
    )


def make_intent(story_targets: list[str]) -> DraftShotSpatialIntent:
    return DraftShotSpatialIntent(
        story_target_aliases=story_targets,
        action_zone_alias="center",
        spatial_content=[],
        viewpoint_intent=DraftViewpointIntent(description="center of the scene"),
        framing="side view of the action",
        framing_scale=FramingScale.MEDIUM,
        camera_motion_intent="static camera",
    )


def make_shot(
    alias: str,
    scene_alias: str,
    beats: list[DraftBeat],
    *,
    story_targets: list[str] | None = None,
    visible: list[str] | None = None,
) -> DraftShot:
    targets = story_targets if story_targets is not None else ["runner"]
    return DraftShot(
        alias=alias,
        scene_alias=scene_alias,
        purpose=f"{alias} purpose",
        duration=6,
        visible_entity_aliases=visible if visible is not None else targets,
        beats=beats,
        spatial_intent=make_intent(targets),
    )


def crossing_beat(
    action: str,
    scene_alias: str | None = None,
    *,
    move: str | None = None,
    to_scene: str | None = None,
) -> DraftBeat:
    transition = None
    if move is not None and to_scene is not None:
        transition = DraftTransition(
            kind="set_placement",
            entity_alias=move,
            placement=DraftPlacement(
                kind="in_scene_zone",
                scene_alias=to_scene,
                zone_alias="center",
            ),
        )
    return DraftBeat(action=action, transition=transition, scene_alias=scene_alias)


def compiler() -> StoryCompiler:
    return StoryCompiler(StoryCompileConstraints(supported_durations=(4, 6, 8)))


def make_two_scene_draft() -> StoryDraft:
    return StoryDraft(
        entities=[
            make_entity("runner", "s_a"),
        ],
        scenes=[make_scene("s_a"), make_scene("s_b")],
        shots=[
            make_shot(
                "sh01",
                "s_a",
                [
                    crossing_beat("run toward the portal"),
                    crossing_beat("step through the portal", move="runner", to_scene="s_b"),
                    crossing_beat("burst into the canyon", scene_alias="s_b"),
                    crossing_beat("keep running"),
                ],
            )
        ],
        requirements=[],
        plan_invariants=[],
    )


class SceneSegmentSplitTests(unittest.TestCase):
    def test_single_scene_change_splits_into_two_parts(self) -> None:
        plan = compiler().compile(make_request(), make_two_scene_draft())
        self.assertEqual(len(plan.ordered_shots), 2)
        first, second = plan.ordered_shots
        self.assertEqual((first.alias, second.alias), ("sh01_s1", "sh01_s2"))
        scene_by_alias = {scene.alias: scene.scene_key for scene in plan.scene_catalog}
        self.assertEqual(first.scene_key, scene_by_alias["s_a"])
        self.assertEqual(second.scene_key, scene_by_alias["s_b"])
        self.assertEqual(len(first.beats), 2)
        self.assertEqual(len(second.beats), 2)
        self.assertEqual(first.duration, 6)
        self.assertEqual(second.duration, 6)
        self.assertEqual(first.purpose, "sh01 purpose")
        self.assertNotEqual(first.shot_key, second.shot_key)

    def test_double_change_a_b_a_three_parts(self) -> None:
        draft = make_two_scene_draft().model_copy(
            update={
                "shots": [
                    make_shot(
                        "sh01",
                        "s_a",
                        [
                            crossing_beat("run toward the portal"),
                            crossing_beat("cross into swamp", move="runner", to_scene="s_b"),
                            crossing_beat("run through the swamp", scene_alias="s_b"),
                            crossing_beat("cross back", move="runner", to_scene="s_a"),
                            crossing_beat("emerge back in camp", scene_alias="s_a"),
                        ],
                    )
                ]
            }
        )
        plan = compiler().compile(make_request(), draft)
        self.assertEqual(
            tuple(shot.alias for shot in plan.ordered_shots),
            ("sh01_s1", "sh01_s2", "sh01_s3"),
        )
        scene_by_alias = {scene.alias: scene.scene_key for scene in plan.scene_catalog}
        self.assertEqual(
            tuple(shot.scene_key for shot in plan.ordered_shots),
            (scene_by_alias["s_a"], scene_by_alias["s_b"], scene_by_alias["s_a"]),
        )

    def test_no_override_leaves_shot_untouched(self) -> None:
        draft = make_two_scene_draft().model_copy(
            update={
                "shots": [
                    make_shot(
                        "sh01",
                        "s_a",
                        [
                            crossing_beat("run toward the portal"),
                            crossing_beat("step through the portal"),
                        ],
                    )
                ]
            }
        )
        plan = compiler().compile(make_request(), draft)
        self.assertEqual(len(plan.ordered_shots), 1)
        self.assertEqual(plan.ordered_shots[0].alias, "sh01")

    def test_scene_binding_satisfied_via_beat_override(self) -> None:
        binding = ProvidedAssetBinding(
            asset_id="asset_scene_01",
            kind="scene",
            canonical_alias="canyon",
            semantic_role="the canyon",
        )
        request = SimpleNamespace(provided_asset_bindings=(binding,))
        scene = SimpleNamespace(alias="canyon", provided_asset_id="asset_scene_01")
        draft = SimpleNamespace(
            entities=(),
            scenes=(scene,),
            shots=(
                SimpleNamespace(
                    scene_alias="station",
                    beats=(SimpleNamespace(scene_alias="canyon"),),
                ),
            ),
        )
        StoryCompiler._validate_asset_binding_contract(request, draft)

    def test_scene_binding_still_rejected_when_unreferenced(self) -> None:
        binding = ProvidedAssetBinding(
            asset_id="asset_scene_01",
            kind="scene",
            canonical_alias="canyon",
            semantic_role="the canyon",
        )
        request = SimpleNamespace(provided_asset_bindings=(binding,))
        scene = SimpleNamespace(alias="canyon", provided_asset_id="asset_scene_01")
        draft = SimpleNamespace(
            entities=(),
            scenes=(scene,),
            shots=(
                SimpleNamespace(
                    scene_alias="station",
                    beats=(SimpleNamespace(scene_alias=None),),
                ),
            ),
        )
        with self.assertRaises(StoryCompilationError) as caught:
            StoryCompiler._validate_asset_binding_contract(request, draft)
        self.assertIn("confirmed scene asset is never used", str(caught.exception))

    def test_override_undeclared_scene_rejected(self) -> None:
        draft = make_two_scene_draft().model_copy(
            update={
                "shots": [
                    make_shot(
                        "sh01",
                        "s_a",
                        [
                            crossing_beat("run toward the portal"),
                            crossing_beat("arrive somewhere else", scene_alias="s_missing"),
                        ],
                    )
                ]
            }
        )
        with self.assertRaises(StoryCompilationError) as caught:
            compiler().compile(make_request(), draft)
        self.assertIn("scene override names undeclared scene", str(caught.exception))

    def test_override_repeating_current_scene_rejected(self) -> None:
        draft = make_two_scene_draft().model_copy(
            update={
                "shots": [
                    make_shot(
                        "sh01",
                        "s_a",
                        [
                            crossing_beat("run toward the portal"),
                            crossing_beat("still here", scene_alias="s_a"),
                        ],
                    )
                ]
            }
        )
        with self.assertRaises(StoryCompilationError) as caught:
            compiler().compile(make_request(), draft)
        self.assertIn("repeats the current scene", str(caught.exception))

    def test_more_than_two_changes_rejected(self) -> None:
        draft = make_two_scene_draft().model_copy(
            update={
                "shots": [
                    make_shot(
                        "sh01",
                        "s_a",
                        [
                            crossing_beat("run", move="runner", to_scene="s_b"),
                            crossing_beat("in swamp", scene_alias="s_b"),
                            crossing_beat("back", move="runner", to_scene="s_a"),
                            crossing_beat("in camp", scene_alias="s_a"),
                            crossing_beat("again", scene_alias="s_b"),
                        ],
                    )
                ]
            }
        )
        with self.assertRaises(StoryCompilationError) as caught:
            compiler().compile(make_request(), draft)
        self.assertIn("more than 2 mid-shot scene changes", str(caught.exception))

    def test_first_beat_override_rejected(self) -> None:
        draft = make_two_scene_draft().model_copy(
            update={
                "shots": [
                    make_shot(
                        "sh01",
                        "s_a",
                        [
                            crossing_beat("already in the canyon", scene_alias="s_b"),
                            crossing_beat("keep running"),
                        ],
                    )
                ]
            }
        )
        with self.assertRaises(StoryCompilationError) as caught:
            compiler().compile(make_request(), draft)
        self.assertIn("first beat changes scene", str(caught.exception))

    def test_requirement_remapped_to_last_part(self) -> None:
        draft = make_two_scene_draft().model_copy(
            update={
                "requirements": [
                    DraftRequirement(
                        alias="req_portal",
                        kind=RequirementKind.REQUIREMENT,
                        description="the courier ends up inside the canyon",
                        priority=50,
                        shot_alias="sh01",
                    )
                ]
            }
        )
        plan = compiler().compile(make_request(), draft)
        owners = {
            requirement.description: requirement.owner_shot_key
            for requirement in plan.requirements
        }
        self.assertEqual(
            owners["the courier ends up inside the canyon"],
            plan.ordered_shots[1].shot_key,
        )

    def test_invariant_before_after_remap(self) -> None:
        draft = make_two_scene_draft().model_copy(
            update={
                "shots": [
                    *make_two_scene_draft().shots,
                    make_shot(
                        "sh02",
                        "s_a",
                        [
                            crossing_beat("collapse exhausted"),
                        ],
                    ),
                ],
                "plan_invariants": [
                    DraftPlanInvariant(
                        alias="inv_order",
                        kind="shot_order",
                        description="the crossing happens before the collapse",
                        before_shot_alias="sh01",
                        after_shot_alias="sh02",
                    )
                ],
            }
        )
        plan = compiler().compile(make_request(shot_target=2), draft)
        self.assertEqual(len(plan.ordered_shots), 3)
        invariant = plan.plan_invariants[0]
        self.assertEqual(invariant.before_shot_key, plan.ordered_shots[0].shot_key)
        self.assertEqual(invariant.after_shot_key, plan.ordered_shots[2].shot_key)

    def test_part_visible_derivation(self) -> None:
        plan = compiler().compile(make_request(), make_two_scene_draft())
        runner_key = plan.entity_catalog[0].entity_key
        first, second = plan.ordered_shots
        self.assertEqual(tuple(first.visible_entities), (runner_key,))
        self.assertEqual(tuple(second.visible_entities), (runner_key,))
        self.assertEqual(tuple(first.spatial_intent.story_targets), (runner_key,))
        self.assertEqual(tuple(second.spatial_intent.story_targets), (runner_key,))

    def test_segment_without_story_target_rejected(self) -> None:
        draft = make_two_scene_draft().model_copy(
            update={
                "shots": [
                    make_shot(
                        "sh01",
                        "s_a",
                        [
                            crossing_beat("run toward the portal"),
                            crossing_beat("meanwhile in the canyon", scene_alias="s_b"),
                        ],
                    )
                ]
            }
        )
        with self.assertRaises(StoryCompilationError) as caught:
            compiler().compile(make_request(), draft)
        self.assertIn("has no story target in that scene", str(caught.exception))

    def test_part_alias_collision_rejected(self) -> None:
        draft = make_two_scene_draft().model_copy(
            update={
                "shots": [
                    *make_two_scene_draft().shots,
                    make_shot(
                        "sh01_s2",
                        "s_a",
                        [
                            crossing_beat("collapse exhausted"),
                        ],
                    ),
                ],
            }
        )
        with self.assertRaises(StoryCompilationError) as caught:
            compiler().compile(make_request(shot_target=2), draft)
        self.assertIn("collides with a declared shot alias", str(caught.exception))

    def test_story_draft_schema_requires_scene_alias_on_beats(self) -> None:
        schema = schema_for_contract(ResponseContract.STORY_DRAFT)
        beat_schema = schema["properties"]["shots"]["items"]["properties"]["beats"]["items"]
        self.assertIn("scene_alias", beat_schema["required"])
        beat_scene_alias = beat_schema["properties"]["scene_alias"]
        self.assertIn("string", str(beat_scene_alias))
        self.assertIn("null", str(beat_scene_alias))


if __name__ == "__main__":
    unittest.main()
