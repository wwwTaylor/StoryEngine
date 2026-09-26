from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from pydantic import ValidationError

from story_engine.domain.reference import (
    GuidedCharacterReferenceRecipe,
    GuidedScenePanoramaRecipe,
    SourceStationSpec,
)
from story_engine.domain.request import (
    ProjectRequest,
    ProvidedAsset,
    ProvidedAssetBinding,
    project_request_from_mapping,
)
from story_engine.errors import ContractError, StoryCompilationError
from story_engine.planning.asset_alignment import (
    AssetAlignmentMatch,
    AssetAlignmentWireResponse,
    AssetIdeaConcept,
    validate_asset_alignment,
)
from story_engine.planning.story_compiler import StoryCompiler
from story_engine.prompts.image import render_character_reference, render_reference_image
from story_engine.providers.ports import ResponseContract
from story_engine.providers.schemas import schema_for_contract


class ProvidedAssetGuidanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.character_path = self.root / "character.png"
        self.scene_path = self.root / "scene.webp"
        Image.new("RGB", (128, 192), "white").save(self.character_path)
        Image.new("RGB", (192, 128), "gray").save(self.scene_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def request(self, *, mode: str = "auto") -> ProjectRequest:
        return ProjectRequest(
            task_id="asset-guidance-test",
            idea="A courier named Lin waits inside an old railway station.",
            shot_target=1,
            visual_style="cinematic realism",
            resolution={"width": 1280, "height": 720},
            generation_requirements={"asset_alignment_mode": mode},
            provided_assets=(
                ProvidedAsset(
                    asset_id="asset_character_01",
                    kind="character",
                    name="客户称呼阿青",
                    description="young courier with a red scarf",
                    path=self.character_path,
                ),
                ProvidedAsset(
                    asset_id="asset_scene_01",
                    kind="scene",
                    name="候车空间素材",
                    description="aged station hall with arched windows",
                    path=self.scene_path,
                ),
            ),
        )

    @staticmethod
    def wire() -> AssetAlignmentWireResponse:
        return AssetAlignmentWireResponse(
            concepts=[
                AssetIdeaConcept(
                    concept_id="character_lin",
                    kind="character",
                    label="Lin",
                    description="the courier",
                    idea_mentions=["Lin", "courier"],
                ),
                AssetIdeaConcept(
                    concept_id="scene_station",
                    kind="scene",
                    label="old railway station",
                    description="the station interior",
                    idea_mentions=["old railway station"],
                ),
            ],
            matches=[
                AssetAlignmentMatch(
                    asset_id="asset_character_01",
                    candidate_concept_ids=["character_lin"],
                    confidence=0.96,
                    rationale="description matches the courier",
                ),
                AssetAlignmentMatch(
                    asset_id="asset_scene_01",
                    candidate_concept_ids=["scene_station"],
                    confidence=0.94,
                    rationale="description matches the station",
                ),
            ],
        )

    def test_auto_alignment_uses_semantics_not_display_names(self) -> None:
        proposal = validate_asset_alignment(self.request(), self.wire(), mode="auto")
        self.assertEqual(proposal.status, "auto_approved")
        self.assertEqual(
            tuple(binding.canonical_alias for binding in proposal.suggested_bindings),
            ("provided_character_01", "provided_scene_01"),
        )
        self.assertEqual(proposal.suggested_bindings[0].semantic_role, "Lin")

    def test_manual_alignment_always_requires_confirmation(self) -> None:
        proposal = validate_asset_alignment(self.request(mode="manual"), self.wire(), mode="manual")
        self.assertEqual(proposal.status, "confirmation_required")
        self.assertFalse(proposal.issues)

    def test_alignment_reuses_matching_required_entity_alias(self) -> None:
        request = self.request().model_copy(
            update={
                "generation_requirements": self.request().generation_requirements.model_copy(
                    update={"required_entities": ("courier_alias",)}
                )
            }
        )
        wire = self.wire().model_copy(
            update={
                "concepts": [
                    self.wire().concepts[0].model_copy(
                        update={"required_entity_alias": "courier_alias"}
                    ),
                    self.wire().concepts[1],
                ]
            }
        )
        proposal = validate_asset_alignment(request, wire, mode="auto")
        self.assertEqual(proposal.suggested_bindings[0].canonical_alias, "courier_alias")

    def test_no_match_returns_editable_input_error(self) -> None:
        wire = self.wire().model_copy(
            update={
                "matches": [
                    self.wire().matches[0].model_copy(
                        update={"candidate_concept_ids": [], "confidence": 0.0}
                    ),
                    self.wire().matches[1],
                ]
            }
        )
        proposal = validate_asset_alignment(self.request(), wire, mode="auto")
        self.assertEqual(proposal.status, "input_invalid")
        self.assertEqual(proposal.issues[0].code, "ASSET_NO_MATCH")
        self.assertIn("idea", proposal.issues[0].message)

    def test_require_all_rejects_unbound_assets(self) -> None:
        values = self.request().model_dump()
        values["generation_requirements"]["provided_asset_policy"] = "require_all"
        with self.assertRaises(ValidationError):
            ProjectRequest.model_validate(values)

    def test_local_images_are_decoded_and_content_hashed(self) -> None:
        loaded = project_request_from_mapping(
            self.request().model_dump(mode="json"),
            source=self.root / "definition.json",
        )
        self.assertRegex(loaded.provided_assets[0].content_sha256 or "", r"^[0-9a-f]{64}$")
        corrupt = self.root / "corrupt.png"
        corrupt.write_bytes(b"not a real PNG")
        values = self.request().model_dump(mode="json")
        values["provided_assets"][0]["path"] = str(corrupt)
        with self.assertRaises(ContractError):
            project_request_from_mapping(values, source=self.root / "definition.json")

    def test_confirmed_binding_requires_exact_planner_alias_and_usage(self) -> None:
        binding = ProvidedAssetBinding(
            asset_id="asset_character_01",
            kind="character",
            canonical_alias="courier_alias",
            semantic_role="Lin",
        )
        request = SimpleNamespace(provided_asset_bindings=(binding,))
        entity = SimpleNamespace(
            alias="courier_alias",
            kind=SimpleNamespace(value="character"),
            provided_asset_id="asset_character_01",
        )
        valid = SimpleNamespace(
            entities=(entity,),
            scenes=(),
            shots=(
                SimpleNamespace(
                    visible_entity_aliases=("courier_alias",),
                    scene_alias="station",
                ),
            ),
        )
        StoryCompiler._validate_asset_binding_contract(request, valid)
        invalid = SimpleNamespace(
            entities=(entity,),
            scenes=(),
            shots=(SimpleNamespace(visible_entity_aliases=(), scene_alias="station"),),
        )
        with self.assertRaises(StoryCompilationError):
            StoryCompiler._validate_asset_binding_contract(request, invalid)

    def test_guided_prompts_make_uploaded_images_authoritative(self) -> None:
        character = GuidedCharacterReferenceRecipe(
            need_key="character-reference",
            subject_key="lin",
            appearance="young courier",
            style="cinematic realism",
            provided_asset_id="asset_character_01",
        )
        character_prompt = render_character_reference(
            character,
            "front",
            guided_source=True,
        )
        self.assertIn("authoritative uploaded character reference", character_prompt)
        self.assertIn("do not redesign", character_prompt)

        scene = GuidedScenePanoramaRecipe(
            need_key="scene-reference",
            subject_key="station",
            scene_visual_identity="old railway station",
            zones_and_landmarks=("arched ticket hall",),
            semantic_layout=(),
            lighting="soft daylight",
            style="cinematic realism",
            allowed_population=(),
            source_station=SourceStationSpec(
                station_key="source-station",
                zone_key="ticket-hall",
                description="center of ticket hall",
            ),
            provided_asset_id="asset_scene_01",
        )
        scene_prompt = render_reference_image(scene)
        self.assertIn("uploaded scene reference", scene_prompt)
        self.assertIn("Extend unseen directions coherently", scene_prompt)

    def test_alignment_response_schema_is_registered(self) -> None:
        schema = schema_for_contract(ResponseContract.ASSET_ALIGNMENT)
        self.assertEqual(schema["type"], "object")
        self.assertEqual(set(schema["required"]), {"concepts", "matches"})


if __name__ == "__main__":
    unittest.main()
