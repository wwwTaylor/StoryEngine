"""Incremental, human-readable output published by each workflow stage."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from story_engine.domain.evaluation import CandidateRecord, EvaluationReport
from story_engine.domain.manifest import DeliveryManifest
from story_engine.domain.reference import ReferenceLibrary, ReferenceRecipe
from story_engine.domain.render import RenderPlan
from story_engine.domain.request import ProjectRequest
from story_engine.domain.spatial import ResolvedSpatialPlan
from story_engine.domain.story import StoryPlan, StoryShot
from story_engine.domain.trace import ProviderPrompt
from story_engine.errors import WorkflowError
from story_engine.run_state import AttemptRecord, RunState, RunStore, StepRecord, StepStatus
from story_engine.spatial.grounding import SceneAnchorMap
from story_engine.spatial.probes import ProbeSet
from story_engine.storage import ArtifactRef, read_json
from story_engine.workflow_records import (
    FirstFrameOperationResult,
    MediaOperationResult,
    PanoramaPreflightOutcome,
    PanoramaPreflightValue,
    ReferenceOperationResult,
    SceneAnchorOperationResult,
    SpatialOperationResult,
)

STAGE_OUTPUT_VERSION = "stage-output-v7-grounding-attempt-evidence"


class StageOutputPublisher:
    """Publish one reviewable stage as soon as that stage has durable results."""

    def __init__(self, run_store: RunStore) -> None:
        self.run_store = run_store
        self.root = run_store.output_dir

    def publish_input(self, request: ProjectRequest, redacted_config: Any) -> None:
        directory = self.root / "00_input"
        _write_json(directory / "request.json", request)
        _write_json(directory / "config.redacted.json", redacted_config)
        requirements = request.generation_requirements
        lines = [
            "# Input requirements",
            "",
            f"- Task: `{request.task_id}`",
            f"- Idea: {request.idea}",
            f"- Shot target: {request.shot_target}",
            f"- Resolution: {request.resolution.width}x{request.resolution.height}",
            f"- Visual style: {request.visual_style}",
            "",
            "## Forbidden content",
            "",
            *[f"- {item}" for item in requirements.forbidden_content],
            "",
            "## Delivery",
            "",
            (
                "- Allowed shot durations: "
                + ", ".join(
                    str(item)
                    for item in requirements.delivery_requirements.allowed_video_duration_seconds
                )
                + " seconds"
            ),
            f"- FPS: {requirements.delivery_requirements.video_fps}",
            f"- Audio: {requirements.delivery_requirements.audio}",
            "",
        ]
        _write_text(directory / "requirements.md", "\n".join(lines))
        self._refresh_root()

    def publish_story_prompt(self, prompt: ProviderPrompt) -> None:
        self._write_prompt(self.root / "01_story" / "prompts", prompt)

    def publish_story(self, plan: StoryPlan, prompts: tuple[ProviderPrompt, ...]) -> None:
        directory = self.root / "01_story"
        _write_json(directory / "story_plan.json", plan)
        for prompt in prompts:
            self._write_prompt(directory / "prompts", prompt)
        self._mark_missing_legacy_prompts(directory / "prompts", prompts)
        shots_directory = directory / "shots"
        _reset_directory(shots_directory, within=self.root)
        rows: list[tuple[int, str, str, int, str]] = []
        storyboard = ["# Storyboard", ""]
        scene_aliases = {item.scene_key: item.alias for item in plan.scene_catalog}
        for index, shot in enumerate(plan.ordered_shots, start=1):
            shot_directory = shots_directory / _ordered_name(index, shot.alias)
            _write_json(shot_directory / "shot.json", shot)
            scene_alias = scene_aliases.get(shot.scene_key, shot.scene_key)
            summary = [
                f"# {index:03d} · {shot.alias}",
                "",
                f"- Scene: `{scene_alias}`",
                f"- Duration: {shot.duration} seconds",
                f"- Purpose: {shot.purpose}",
                "",
                "## Beats",
                "",
                *[f"{beat_index}. {beat.action}" for beat_index, beat in enumerate(shot.beats, 1)],
                "",
            ]
            _write_text(shot_directory / "summary.md", "\n".join(summary))
            rows.append((index, shot.alias, scene_alias, shot.duration, shot.purpose))
            storyboard.extend(summary)
        _write_text(directory / "storyboard.md", "\n".join(storyboard))
        _write_csv(
            directory / "timeline.csv",
            ("index", "shot_alias", "scene_alias", "duration_seconds", "purpose"),
            rows,
        )
        self._publish_structural_checks(plan)
        self.publish_asset_catalog(plan)
        self._refresh_root()

    def publish_asset_catalog(self, plan: StoryPlan) -> None:
        directory = self.root / "02_assets"
        _write_json(
            directory / "catalog.json",
            {
                "entities": [item.model_dump(mode="json") for item in plan.entity_catalog],
                "scenes": [item.model_dump(mode="json") for item in plan.scene_catalog],
                "reference_needs": [item.model_dump(mode="json") for item in plan.reference_needs],
            },
        )
        needs_by_subject: dict[str, list[str]] = {}
        for need in plan.reference_needs:
            needs_by_subject.setdefault(need.subject_key, []).append(need.need_key)
        for index, entity in enumerate(plan.entity_catalog, start=1):
            category = "characters" if entity.kind.value == "character" else "props"
            asset_directory = directory / category / _ordered_name(index, entity.alias)
            _write_json(asset_directory / "asset.json", entity)
            _write_text(
                asset_directory / "asset.md",
                "\n".join(
                    (
                        f"# {entity.alias}",
                        "",
                        f"- Kind: `{entity.kind.value}`",
                        f"- Visual identity: {entity.visual_identity}",
                        f"- Freeze appearance: {entity.freeze_appearance}",
                        "",
                    )
                ),
            )
            _write_json(
                asset_directory / "reference-status.json",
                {
                    "status": (
                        "pending" if entity.entity_key in needs_by_subject else "semantic_only"
                    ),
                    "reference_needs": needs_by_subject.get(entity.entity_key, []),
                },
            )
        for index, scene in enumerate(plan.scene_catalog, start=1):
            scene_directory = directory / "scenes" / _ordered_name(index, scene.alias)
            _write_json(scene_directory / "scene.json", scene)
            _write_text(
                scene_directory / "scene.md",
                "\n".join(
                    (
                        f"# {scene.alias}",
                        "",
                        f"- Visual identity: {scene.visual_identity}",
                        f"- Lighting: {scene.lighting}",
                        f"- Style: {scene.style}",
                        "",
                        "## Zones",
                        "",
                        *[f"- `{zone.alias}`: {zone.description}" for zone in scene.zones],
                        "",
                    )
                ),
            )
            _write_json(
                scene_directory / "reference-status.json",
                {
                    "status": (
                        "pending" if scene.scene_key in needs_by_subject else "semantic_only"
                    ),
                    "reference_needs": needs_by_subject.get(scene.scene_key, []),
                },
            )

    def publish_reference_prompt(
        self,
        plan: StoryPlan,
        recipe: ReferenceRecipe,
        prompt: ProviderPrompt,
    ) -> None:
        self._write_prompt(self._reference_directory(plan, recipe) / "prompts", prompt)

    def publish_reference(
        self,
        plan: StoryPlan,
        result: ReferenceOperationResult,
    ) -> None:
        directory = self._reference_directory(plan, result.recipe)
        selected_name = (
            "panorama-selected"
            if result.recipe.kind in {"scene_panorama", "guided_scene_panorama"}
            else "selected"
        )
        selected_report, statements = self._publish_media_operation(
            directory,
            result,
            selected_stem=selected_name,
        )
        selected = _selected_candidate(result)
        for media in selected.media:
            self.run_store.artifacts.materialize(
                media.artifact_ref,
                directory / f"{media.role}-selected"
                f"{_media_extension(media.artifact_ref.media_type)}",
            )
        _write_json(
            directory / "reference-status.json",
            {
                "status": "selected",
                "need_key": result.recipe.need_key,
                "candidate_key": result.decision.selected_candidate,
                "selection_outcome": result.decision.outcome.value,
            },
        )
        label = self._reference_label(plan, result.recipe)
        self._publish_evaluation_summary(
            "assets",
            label,
            selected_report,
            statements,
        )
        self._refresh_root()

    def publish_reference_library(self, library: ReferenceLibrary) -> None:
        _write_json(self.root / "02_assets" / "reference-library.json", library)
        self._refresh_root()

    def publish_spatial_prompt(
        self,
        plan: StoryPlan,
        shot_key: str,
        prompt: ProviderPrompt,
    ) -> None:
        directory = self._spatial_shot_directory(plan, shot_key)
        self._write_prompt(directory / "prompts", prompt)

    def publish_anchor_prompt(
        self,
        plan: StoryPlan,
        scene_key: str,
        prompt: ProviderPrompt,
    ) -> None:
        directory = self._spatial_scene_directory(plan, scene_key)
        self._write_prompt(directory / "anchor-prompts", prompt)

    def publish_grounding_attempt(
        self,
        plan: StoryPlan,
        scene_key: str,
        attempt: AttemptRecord,
    ) -> None:
        directory = self._spatial_scene_directory(plan, scene_key) / "anchor-prompts"
        for prompt in attempt.prompts:
            self._write_prompt(directory, prompt)
        for label, references in (
            ("response", attempt.response_refs),
            ("validation", attempt.validation_refs),
        ):
            for index, reference in enumerate(references, start=1):
                suffix = f"-call-{index:02d}" if len(references) > 1 else ""
                self.run_store.artifacts.materialize(
                    reference,
                    directory
                    / (f"grounding-attempt-{attempt.logical_attempt:02d}{suffix}-{label}.json"),
                )
        self._refresh_root()

    def publish_probes(
        self,
        plan: StoryPlan,
        scene_key: str,
        probe_set: ProbeSet,
    ) -> None:
        directory = self._spatial_scene_directory(plan, scene_key)
        _write_json(directory / "probe-set.json", probe_set)
        evidence_directory = directory / "probe-sets" / probe_set.source_panorama_hash[:16]
        _write_json(evidence_directory / "probe-set.json", probe_set)
        probes_directory = directory / "probes"
        _reset_directory(probes_directory, within=self.root)
        evidence_probes = evidence_directory / "probes"
        _reset_directory(evidence_probes, within=self.root)
        for probe in probe_set.probes:
            filename = f"{probe.role.value}{_media_extension(probe.artifact_ref.media_type)}"
            for target in (probes_directory / filename, evidence_probes / filename):
                self.run_store.artifacts.materialize(probe.artifact_ref, target)
        self._refresh_root()

    def publish_anchor_map(
        self,
        plan: StoryPlan,
        anchor_map: SceneAnchorMap,
    ) -> None:
        directory = self._spatial_scene_directory(plan, anchor_map.scene_key)
        _write_json(directory / "scene-anchor-map.json", anchor_map)
        _write_json(
            directory
            / "anchor-maps"
            / f"{anchor_map.reference_hash[:16]}_{_slug(anchor_map.station_key)}.json",
            anchor_map,
        )
        self._refresh_root()

    def publish_spatial_preflight(
        self,
        plan: StoryPlan,
        result: PanoramaPreflightValue,
    ) -> None:
        directory = self._spatial_scene_directory(plan, result.scene_key) / "preflight"
        _write_json(directory / f"{result.candidate_key}.json", result)
        self._refresh_root()

    def publish_spatial_resolution(
        self,
        plan: StoryPlan,
        resolution: ResolvedSpatialPlan,
    ) -> None:
        directory = self._spatial_shot_directory(plan, resolution.shot_key)
        _write_json(directory / "resolved-spatial-plan.json", resolution)
        _write_json(directory / "camera.json", resolution.camera_recipe)
        _write_json(
            directory / "repair-audit.json",
            [item.model_dump(mode="json") for item in resolution.audit],
        )
        self._write_shot_context(directory, plan, resolution.shot_key)
        self.run_store.artifacts.materialize(
            resolution.camera_recipe.scene_view,
            directory
            / ("scene-view" + _media_extension(resolution.camera_recipe.scene_view.media_type)),
        )
        self._refresh_root()

    def publish_novel_station(
        self,
        plan: StoryPlan,
        shot_key: str,
        result: MediaOperationResult,
    ) -> None:
        directory = self._spatial_shot_directory(plan, shot_key) / "novel-station"
        self._write_shot_context(directory, plan, shot_key)
        report, statements = self._publish_media_operation(
            directory,
            result,
            selected_stem="selected-station-panorama",
        )
        shot = _story_shot(plan, shot_key)
        self._publish_evaluation_summary(
            "spatial",
            f"{_shot_index(plan, shot_key):03d}_{shot.alias}_novel_station",
            report,
            statements,
        )
        self._refresh_root()

    def publish_render_plan(self, story_plan: StoryPlan, render_plan: RenderPlan) -> None:
        directory = self.root / "04_render_plan"
        _write_json(directory / "render_plan.json", render_plan)
        shots_directory = directory / "shots"
        _reset_directory(shots_directory, within=self.root)
        for index, (story_shot, render_shot) in enumerate(
            zip(
                story_plan.ordered_shots,
                render_plan.ordered_render_shots,
                strict=True,
            ),
            start=1,
        ):
            _write_json(
                shots_directory / f"{_ordered_name(index, story_shot.alias)}.json",
                render_shot,
            )
        continuity = ["# Continuity plan", ""]
        for index, render_shot in enumerate(render_plan.ordered_render_shots, start=1):
            story_shot = story_plan.ordered_shots[index - 1]
            continuity.extend(
                [
                    f"## {index:03d} · {story_shot.alias}",
                    "",
                    f"- Start visible entities: {', '.join(render_shot.start_visible_entities)}",
                    f"- End visible entities: {', '.join(render_shot.end_visible_entities)}",
                    "",
                ]
            )
        _write_text(directory / "continuity.md", "\n".join(continuity))
        self._refresh_root()

    def publish_first_frame_prompt(
        self,
        plan: StoryPlan,
        shot_key: str,
        prompt: ProviderPrompt,
    ) -> None:
        directory = self._shot_stage_directory("05_first_frames", plan, shot_key)
        self._write_prompt(directory / "prompts", prompt)

    def publish_first_frame(
        self,
        plan: StoryPlan,
        shot_key: str,
        result: MediaOperationResult,
    ) -> None:
        directory = self._shot_stage_directory("05_first_frames", plan, shot_key)
        self._write_shot_context(directory, plan, shot_key)
        report, statements = self._publish_media_operation(
            directory,
            result,
            selected_stem="selected",
        )
        _write_json(directory / "input-assets.json", _prompt_inputs(result.prompts))
        shot = _story_shot(plan, shot_key)
        self._publish_evaluation_summary(
            "first_frames",
            f"{_shot_index(plan, shot_key):03d}_{shot.alias}",
            report,
            statements,
        )
        self._refresh_root()

    def publish_video_prompt(
        self,
        plan: StoryPlan,
        shot_key: str,
        prompt: ProviderPrompt,
    ) -> None:
        directory = self._shot_stage_directory("06_shot_videos", plan, shot_key)
        self._write_prompt(directory / "prompts", prompt)

    def publish_video(
        self,
        plan: StoryPlan,
        shot_key: str,
        start_frame: ArtifactRef,
        result: MediaOperationResult,
    ) -> None:
        directory = self._shot_stage_directory("06_shot_videos", plan, shot_key)
        self._write_shot_context(directory, plan, shot_key)
        report, statements = self._publish_media_operation(
            directory,
            result,
            selected_stem="selected",
        )
        start_path = directory / ("start-frame" + _media_extension(start_frame.media_type))
        self.run_store.artifacts.materialize(start_frame, start_path)
        selected = _selected_candidate(result)
        video = _required_artifact(selected)
        selected_video_path = directory / ("selected" + _media_extension(video.media_type))
        _extract_first_frame(selected_video_path, directory / "video-frame-000.png")
        _write_json(directory / "input-assets.json", _prompt_inputs(result.prompts))
        shot = _story_shot(plan, shot_key)
        self._publish_evaluation_summary(
            "videos",
            f"{_shot_index(plan, shot_key):03d}_{shot.alias}",
            report,
            statements,
        )
        self._refresh_root()

    def publish_final(
        self,
        state: RunState,
        story_plan: StoryPlan,
        manifest: DeliveryManifest,
    ) -> None:
        if state.final_video is None:
            raise WorkflowError("final output requires the assembled video")
        directory = self.root / "08_final"
        final_video = directory / "final.mp4"
        self.run_store.artifacts.materialize(state.final_video, final_video)
        _write_json(directory / "manifest.json", manifest)
        rows = [
            (index, shot.alias, shot.duration, shot.purpose)
            for index, shot in enumerate(story_plan.ordered_shots, start=1)
        ]
        _write_csv(
            directory / "shot-timeline.csv",
            ("index", "shot_alias", "duration_seconds", "purpose"),
            rows,
        )
        checksum_lines = [
            f"{_sha256_file(final_video)}  final.mp4",
            f"{_sha256_file(directory / 'manifest.json')}  manifest.json",
        ]
        _write_text(directory / "checksums.sha256", "\n".join(checksum_lines) + "\n")
        self.publish_audit(state, manifest)
        self._refresh_root()

    def publish_audit(
        self,
        state: RunState,
        manifest: DeliveryManifest | None = None,
    ) -> None:
        directory = self.root / "09_audit"
        _write_json(directory / "state.json", state)
        if self.run_store.events_path.is_file():
            _atomic_copy(
                self.run_store.events_path,
                directory / "workflow-events.jsonl",
            )
        if manifest is not None:
            _write_json(
                directory / "provenance.json",
                [item.model_dump(mode="json") for item in manifest.provenance],
            )
            _write_json(
                directory / "provider-summary.json",
                [item.model_dump(mode="json") for item in manifest.providers],
            )
        self._refresh_root()

    def _publish_media_operation(
        self,
        directory: Path,
        result: MediaOperationResult,
        *,
        selected_stem: str,
    ) -> tuple[EvaluationReport, dict[str, str]]:
        directory.mkdir(parents=True, exist_ok=True)
        alternatives = directory / "alternatives"
        evaluations = directory / "evaluations"
        _reset_directory(alternatives, within=self.root)
        _reset_directory(evaluations, within=self.root)
        prompts_directory = directory / "prompts"
        _reset_directory(prompts_directory, within=self.root)
        for prompt in result.prompts:
            self._write_prompt(prompts_directory, prompt)
        self._mark_missing_legacy_prompts(prompts_directory, result.prompts)

        reports = {item.candidate_key: item for item in result.reports}
        selected = _selected_candidate(result)
        selected_artifact = _required_artifact(selected)
        selected_path = directory / (selected_stem + _media_extension(selected_artifact.media_type))
        self.run_store.artifacts.materialize(selected_artifact, selected_path)
        selected_report = reports.get(selected.candidate_key)
        if selected_report is None:
            raise WorkflowError("selected output has no evaluation report")
        statements = _criterion_statements(result.prompts)
        _write_json(directory / "evaluation.json", selected_report)
        _write_text(
            directory / "evaluation.md",
            _evaluation_markdown(selected_report, statements),
        )
        _write_json(directory / "selection.json", result.decision)
        _write_text(
            directory / "selection.md",
            _selection_markdown(result),
        )
        _write_json(directory / "operation.json", result)
        _write_json(
            directory / "technical-validation.json",
            {
                "status": selected.technical_status.value,
                "quality": selected.technical_quality,
                "findings": [item.model_dump(mode="json") for item in selected.technical_findings],
            },
        )

        alternative_records: list[dict[str, Any]] = []
        for candidate in result.candidates:
            report = reports.get(candidate.candidate_key)
            if report is not None:
                _write_json(
                    evaluations / f"{_slug(candidate.candidate_key)}.json",
                    report,
                )
                _write_text(
                    evaluations / f"{_slug(candidate.candidate_key)}.md",
                    _evaluation_markdown(report, statements),
                )
            if candidate.candidate_key == selected.candidate_key:
                continue
            record = candidate.model_dump(mode="json")
            if candidate.artifact_ref is not None:
                stem = f"attempt-{candidate.logical_attempt:02d}-{_slug(candidate.candidate_key)}"
                media_path = alternatives / (
                    stem + _media_extension(candidate.artifact_ref.media_type)
                )
                self.run_store.artifacts.materialize(candidate.artifact_ref, media_path)
                record["human_path"] = media_path.relative_to(directory).as_posix()
                if report is not None:
                    _write_json(alternatives / f"{stem}-evaluation.json", report)
                    _write_text(
                        alternatives / f"{stem}-evaluation.md",
                        _evaluation_markdown(report, statements),
                    )
            alternative_records.append(record)
        _write_json(alternatives / "candidates.json", alternative_records)
        return selected_report, statements

    def _publish_evaluation_summary(
        self,
        stage: str,
        label: str,
        report: EvaluationReport,
        statements: dict[str, str],
    ) -> None:
        directory = self.root / "07_evaluation_summary"
        result_path = directory / "results" / stage / f"{_slug(label)}.json"
        _write_json(
            result_path,
            {
                "stage": stage,
                "label": label,
                "report": report.model_dump(mode="json"),
                "criterion_statements": statements,
            },
        )
        self._rebuild_evaluation_summary(directory)

    def _publish_structural_checks(self, plan: StoryPlan) -> None:
        directory = self.root / "07_evaluation_summary"
        shot_aliases = {item.shot_key: item.alias for item in plan.ordered_shots}
        checks = [
            {
                "invariant_key": item.invariant_key,
                "kind": item.kind.value,
                "description": item.description,
                "before_shot": shot_aliases[item.before_shot_key],
                "after_shot": shot_aliases[item.after_shot_key],
                "status": "verified",
                "evidence": "Canonical StoryPlan order satisfies before_shot < after_shot.",
            }
            for item in plan.plan_invariants
        ]
        _write_json(
            self.root / "01_story" / "structural-checks.json",
            {
                "media_evaluation": False,
                "checks": checks,
            },
        )
        lines = [
            "# Structural plan checks",
            "",
            "These checks are verified deterministically by StoryCompiler.",
            "They are not media-evaluation PASS results.",
            "",
        ]
        if checks:
            lines.extend(
                f"- **VERIFIED** `{item['before_shot']}` → `{item['after_shot']}`: "
                f"{item['description']}"
                for item in checks
            )
        else:
            lines.append("- No explicit shot-order invariant was requested.")
        lines.append("")
        markdown = "\n".join(lines)
        _write_text(self.root / "01_story" / "structural-checks.md", markdown)
        _write_json(
            directory / "structural-checks.json",
            {
                "media_evaluation": False,
                "checks": checks,
            },
        )
        _write_text(directory / "structural-checks.md", markdown)
        self._rebuild_evaluation_summary(directory)

    def _rebuild_evaluation_summary(self, directory: Path) -> None:
        entries: list[dict[str, Any]] = []
        results_root = directory / "results"
        if results_root.is_dir():
            for path in sorted(results_root.rglob("*.json")):
                raw = read_json(path)
                if isinstance(raw, dict):
                    entries.append(raw)
        totals = {"pass": 0, "fail": 0, "unknown": 0}
        matrix_rows: list[tuple[str, str, str, str, str, str]] = []
        issues: list[str] = ["# Failed and unknown evaluations", ""]
        for entry in entries:
            report = entry.get("report", {})
            statements = entry.get("criterion_statements", {})
            for criterion in report.get("criterion_results", []):
                status = str(criterion["status"])
                criterion_id = str(criterion["criterion_id"])
                statement = str(statements.get(criterion_id, "Statement unavailable"))
                totals[status] = totals.get(status, 0) + 1
                matrix_rows.append(
                    (
                        str(entry["stage"]),
                        str(entry["label"]),
                        criterion_id,
                        status,
                        statement,
                        str(criterion["evidence"]),
                    )
                )
                if status != "pass":
                    issues.extend(
                        [
                            f"## {entry['stage']} · {entry['label']}",
                            "",
                            f"- Status: **{status.upper()}**",
                            f"- Criterion: `{criterion_id}`",
                            f"- Statement: {statement}",
                            f"- Evidence: {criterion['evidence']}",
                            "",
                        ]
                    )
        overview = [
            "# Evaluation overview",
            "",
            f"- PASS: {totals['pass']}",
            f"- FAIL: {totals['fail']}",
            f"- UNKNOWN: {totals['unknown']}",
            f"- Evaluated items: {len(entries)}",
            (
                "- Structurally verified plan invariants: "
                f"{len(read_json(directory / 'structural-checks.json').get('checks', []))}"
                if (directory / "structural-checks.json").is_file()
                else "- Structurally verified plan invariants: 0"
            ),
            "- Final assembled video semantic evaluation: not performed",
            "",
        ]
        _write_text(directory / "overview.md", "\n".join(overview))
        _write_text(directory / "failed-and-unknown.md", "\n".join(issues))
        _write_csv(
            directory / "requirement-matrix.csv",
            ("stage", "item", "criterion_id", "status", "statement", "evidence"),
            matrix_rows,
        )

    def _write_prompt(self, directory: Path, prompt: ProviderPrompt) -> None:
        candidate = (
            f"-candidate-{prompt.candidate_index:02d}" if prompt.candidate_index is not None else ""
        )
        key = f"-{_slug(prompt.candidate_key)}" if prompt.candidate_key is not None else ""
        stem = f"{prompt.purpose.value}-attempt-{prompt.logical_attempt:02d}{candidate}{key}"
        _write_text(directory / f"{stem}.md", prompt.prompt)
        _write_json(directory / f"{stem}-request.json", prompt)

    @staticmethod
    def _mark_missing_legacy_prompts(
        directory: Path,
        prompts: tuple[ProviderPrompt, ...],
    ) -> None:
        marker = directory / "NOT_RECORDED_IN_LEGACY_RUN.md"
        if prompts:
            marker.unlink(missing_ok=True)
            return
        _write_text(
            marker,
            "# Prompt unavailable\n\n"
            "This operation predates exact Provider-port prompt persistence. "
            "No prompt was reconstructed or fabricated.\n",
        )

    def _reference_directory(
        self,
        plan: StoryPlan,
        recipe: ReferenceRecipe,
    ) -> Path:
        subject_key = recipe.subject_key
        for index, scene in enumerate(plan.scene_catalog, start=1):
            if scene.scene_key == subject_key:
                return (
                    self.root
                    / "02_assets"
                    / "scenes"
                    / _ordered_name(
                        index,
                        scene.alias,
                    )
                )
        for index, entity in enumerate(plan.entity_catalog, start=1):
            if entity.entity_key != subject_key:
                continue
            category = "characters" if entity.kind.value == "character" else "props"
            base = self.root / "02_assets" / category / _ordered_name(index, entity.alias)
            matching = [item for item in plan.reference_needs if item.subject_key == subject_key]
            if len(matching) <= 1:
                return base
            need_index = next(
                (
                    offset
                    for offset, item in enumerate(matching, start=1)
                    if item.need_key == recipe.need_key
                ),
                1,
            )
            return base / "references" / f"{need_index:02d}_{_slug(recipe.need_key)}"
        raise WorkflowError(f"reference recipe has unknown subject {subject_key}")

    def _reference_label(self, plan: StoryPlan, recipe: ReferenceRecipe) -> str:
        for item in (*plan.entity_catalog, *plan.scene_catalog):
            key = getattr(item, "entity_key", None) or getattr(item, "scene_key", None)
            if key == recipe.subject_key:
                alias = str(getattr(item, "alias", recipe.subject_key))
                return f"{recipe.kind}_{alias}_{recipe.need_key}"
        return f"{recipe.kind}_{recipe.need_key}"

    def _spatial_scene_directory(self, plan: StoryPlan, scene_key: str) -> Path:
        for index, scene in enumerate(plan.scene_catalog, start=1):
            if scene.scene_key == scene_key:
                return self.root / "03_spatial" / _ordered_name(index, scene.alias)
        raise WorkflowError(f"unknown spatial scene {scene_key}")

    def _spatial_shot_directory(self, plan: StoryPlan, shot_key: str) -> Path:
        shot = _story_shot(plan, shot_key)
        return (
            self._spatial_scene_directory(plan, shot.scene_key)
            / "shots"
            / _ordered_name(_shot_index(plan, shot_key), shot.alias)
        )

    def _shot_stage_directory(
        self,
        stage: str,
        plan: StoryPlan,
        shot_key: str,
    ) -> Path:
        shot = _story_shot(plan, shot_key)
        return self.root / stage / _ordered_name(_shot_index(plan, shot_key), shot.alias)

    def _write_shot_context(
        self,
        directory: Path,
        plan: StoryPlan,
        shot_key: str,
    ) -> None:
        shot = _story_shot(plan, shot_key)
        scene = next(item for item in plan.scene_catalog if item.scene_key == shot.scene_key)
        lines = [
            f"# {_shot_index(plan, shot_key):03d} · {shot.alias}",
            "",
            f"- Scene: `{scene.alias}`",
            f"- Duration: {shot.duration} seconds",
            f"- Purpose: {shot.purpose}",
            "",
            "## Expected beats",
            "",
            *[f"{index}. {beat.action}" for index, beat in enumerate(shot.beats, start=1)],
            "",
        ]
        _write_text(directory / "shot.md", "\n".join(lines))

    def _refresh_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        stage_labels = (
            ("00_input", "Input"),
            ("01_story", "Story"),
            ("02_assets", "Assets"),
            ("03_spatial", "Spatial"),
            ("04_render_plan", "Render plan"),
            ("05_first_frames", "First frames"),
            ("06_shot_videos", "Shot videos"),
            ("07_evaluation_summary", "Evaluation summary"),
            ("08_final", "Final delivery"),
            ("09_audit", "Audit"),
        )
        available = [
            (directory, label)
            for directory, label in stage_labels
            if (self.root / directory).is_dir()
        ]
        readme = [
            "# StoryEngine human-review output",
            "",
            "Folders are written incrementally by their workflow stages.",
            "The content-addressed artifact store and workflow state remain outside",
            "`output/` and are authoritative for resume.",
            "",
            "Open `OPEN_ME.html` for selected-media galleries and playable videos.",
            "Each media folder keeps its exact prompts, all candidates, technical",
            "validation, selection rationale, and a nearby `evaluation.md`.",
            "`07_evaluation_summary/` provides the cross-stage view without replacing",
            "those per-media reports. A `NOT_RECORDED_IN_LEGACY_RUN.md` file means the",
            "historical run predates exact prompt tracing; no prompt was fabricated.",
            "Structural plan checks are listed separately and never counted as media PASS.",
            "The assembled final video receives technical validation, not semantic judging.",
            "",
            "## Available stages",
            "",
            *[f"- [{directory}]({directory}/): {label}" for directory, label in available],
            "",
        ]
        _write_text(self.root / "README.md", "\n".join(readme))
        links = "".join(
            f'<li><a href="{html.escape(directory, quote=True)}/">'
            f"{html.escape(directory)} · {html.escape(label)}</a></li>"
            for directory, label in available
        )
        asset_paths = sorted(
            {
                *self.root.glob("02_assets/**/selected.*"),
                *self.root.glob("02_assets/**/panorama-selected.*"),
            }
        )
        first_frame_paths = sorted(self.root.glob("05_first_frames/*/selected.*"))
        video_paths = sorted(self.root.glob("06_shot_videos/*/selected.mp4"))
        galleries = "".join(
            (
                self._image_gallery_html("Selected assets", asset_paths),
                self._image_gallery_html("Selected first frames", first_frame_paths),
                self._video_gallery_html("Selected shot videos", video_paths),
            )
        )
        final_video = (
            '<h2>Final video</h2><video controls preload="metadata" '
            'src="08_final/final.mp4"></video>'
            if (self.root / "08_final" / "final.mp4").is_file()
            else ""
        )
        page = (
            '<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>StoryEngine output</title><style>"
            "body{max-width:1100px;margin:auto;padding:24px;font:16px/1.5 system-ui}"
            "video{width:100%;background:#000}li{margin:.5em 0}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));"
            "gap:16px}.card{border:1px solid #ccc;border-radius:8px;padding:10px;"
            "overflow:hidden}.card img{display:block;width:100%;height:180px;"
            "object-fit:contain;background:#111}.card video{height:180px;object-fit:contain}"
            ".meta{font-size:.9em;overflow-wrap:anywhere}</style></head><body>"
            "<h1>StoryEngine human-review output</h1>"
            "<p>Each folder appears when its workflow stage completes.</p>"
            f"<ol>{links}</ol>{galleries}{final_video}</body></html>\n"
        )
        _write_text(self.root / "OPEN_ME.html", page)
        self._write_review_manifest()

    def _image_gallery_html(self, title: str, paths: list[Path]) -> str:
        if not paths:
            return ""
        cards = "".join(self._media_card(path, video=False) for path in paths)
        return f'<h2>{html.escape(title)}</h2><div class="grid">{cards}</div>'

    def _video_gallery_html(self, title: str, paths: list[Path]) -> str:
        if not paths:
            return ""
        cards = "".join(self._media_card(path, video=True) for path in paths)
        return f'<h2>{html.escape(title)}</h2><div class="grid">{cards}</div>'

    def _media_card(self, path: Path, *, video: bool) -> str:
        relative = path.relative_to(self.root).as_posix()
        escaped = html.escape(relative, quote=True)
        label = html.escape(path.parent.name)
        rejected_extensions = {".mp4"} if video else {".png", ".jpg", ".jpeg", ".webp"}
        rejected = sum(
            candidate.is_file() and candidate.suffix.lower() in rejected_extensions
            for candidate in (path.parent / "alternatives").glob("*")
        )
        evaluation = path.parent / "evaluation.md"
        evaluation_link = (
            f' · <a href="{html.escape(evaluation.relative_to(self.root).as_posix(), quote=True)}">'
            "evaluation</a>"
            if evaluation.is_file()
            else ""
        )
        media = (
            f'<video controls preload="metadata" src="{escaped}"></video>'
            if video
            else f'<a href="{escaped}"><img loading="lazy" src="{escaped}" alt="{label}"></a>'
        )
        return (
            f'<article class="card">{media}<div class="meta"><strong>{label}</strong><br>'
            f'<a href="{escaped}">media</a>{evaluation_link} · rejected: {rejected}'
            "</div></article>"
        )

    def _write_review_manifest(self) -> None:
        files: list[dict[str, Any]] = []
        excluded = {"review_manifest.json"}
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.name in excluded:
                continue
            files.append(
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        _write_json(
            self.root / "review_manifest.json",
            {
                "version": STAGE_OUTPUT_VERSION,
                "files": files,
            },
        )


def rebuild_stage_output(run_store: RunStore) -> Path:
    """Recover the staged projection from authoritative completed artifacts."""

    state = run_store.load()
    try:
        request = ProjectRequest.model_validate(read_json(run_store.run_dir / "request.json"))
        redacted_config = read_json(run_store.run_dir / "config.redacted.json")
    except ValueError as exc:
        raise WorkflowError(f"cannot rebuild staged output: {exc}") from exc
    publisher = StageOutputPublisher(run_store)
    for legacy_name in ("video.mp4", "manifest.json"):
        (publisher.root / legacy_name).unlink(missing_ok=True)
    publisher.publish_input(request, redacted_config)

    story_step = _completed_step(state, "story_plan")
    if story_step is None:
        publisher.publish_audit(state)
        return publisher.root
    story_plan = _load_completed_output(run_store, story_step, StoryPlan)
    story_prompts = tuple(
        prompt for attempt in story_step.attempt_records for prompt in attempt.prompts
    )
    publisher.publish_story(story_plan, story_prompts)

    for step in _completed_steps_with_prefix(state, "reference:"):
        reference_result = _load_completed_output(
            run_store,
            step,
            ReferenceOperationResult,
        )
        publisher.publish_reference(story_plan, reference_result)
    library_step = _completed_step(state, "reference_library_spatial") or _completed_step(
        state, "reference_library"
    )
    if library_step is not None:
        library = _load_completed_output(run_store, library_step, ReferenceLibrary)
        publisher.publish_reference_library(library)

    for step in _completed_steps_with_prefix(state, "probes:"):
        probes = _load_completed_output(run_store, step, ProbeSet)
        publisher.publish_probes(story_plan, probes.scene_key, probes)
    for step in sorted(
        (item for item in state.steps if item.step_name.startswith("anchor_map:")),
        key=lambda item: item.step_name,
    ):
        parts = step.step_name.split(":", maxsplit=3)
        if len(parts) >= 2:
            scene_key = parts[1]
            for attempt in step.attempt_records:
                publisher.publish_grounding_attempt(story_plan, scene_key, attempt)
        if step.status == StepStatus.COMPLETED:
            anchor_result = _load_completed_output(
                run_store,
                step,
                SceneAnchorOperationResult,
            )
            publisher.publish_anchor_map(story_plan, anchor_result.anchor_map)
    preflight_steps = (
        *_completed_steps_with_prefix(state, "spatial_preflight:"),
        *_completed_steps_with_prefix(state, "novel_station_preflight:"),
    )
    for step in preflight_steps:
        preflight_result = _load_completed_output(
            run_store,
            step,
            PanoramaPreflightOutcome,
        ).outcome
        publisher.publish_spatial_preflight(story_plan, preflight_result)
    for step in _completed_steps_with_prefix(state, "spatial:"):
        spatial_result = _load_completed_output(run_store, step, SpatialOperationResult)
        publisher.publish_spatial_resolution(story_plan, spatial_result.resolution)
    for step in _completed_steps_with_prefix(state, "novel_station:"):
        shot_key = step.step_name.removeprefix("novel_station:")
        novel_result = _load_completed_output(run_store, step, MediaOperationResult)
        publisher.publish_novel_station(story_plan, shot_key, novel_result)

    render_step = _completed_step(state, "render_plan")
    if render_step is not None:
        render_plan = _load_completed_output(run_store, render_step, RenderPlan)
        publisher.publish_render_plan(story_plan, render_plan)

    frames: dict[str, FirstFrameOperationResult] = {}
    for step in _completed_steps_with_prefix(state, "first_frame:"):
        shot_key = step.step_name.removeprefix("first_frame:")
        frame_result = _load_completed_output(
            run_store,
            step,
            FirstFrameOperationResult,
        )
        frames[shot_key] = frame_result
        publisher.publish_first_frame(story_plan, shot_key, frame_result)
    for step in _completed_steps_with_prefix(state, "video:"):
        shot_key = step.step_name.removeprefix("video:")
        frame = frames.get(shot_key)
        if frame is None:
            raise WorkflowError(f"video output has no completed first frame: {shot_key}")
        start_frame = _required_artifact(_selected_candidate(frame))
        video_result = _load_completed_output(run_store, step, MediaOperationResult)
        publisher.publish_video(story_plan, shot_key, start_frame, video_result)

    manifest: DeliveryManifest | None = None
    if state.manifest is not None:
        path = run_store.artifacts.verify(state.manifest)
        try:
            manifest = DeliveryManifest.model_validate_json(path.read_bytes())
        except ValueError as exc:
            raise WorkflowError(f"cannot rebuild invalid delivery manifest: {exc}") from exc
        publisher.publish_final(state, story_plan, manifest)
    publisher.publish_audit(state, manifest)
    return publisher.root


def _completed_step(state: RunState, name: str) -> StepRecord | None:
    return next(
        (
            step
            for step in state.steps
            if step.step_name == name and step.status == StepStatus.COMPLETED
        ),
        None,
    )


def _completed_steps_with_prefix(
    state: RunState,
    prefix: str,
) -> tuple[StepRecord, ...]:
    return tuple(
        step
        for step in state.steps
        if step.step_name.startswith(prefix) and step.status == StepStatus.COMPLETED
    )


def _load_completed_output[ModelT: BaseModel](
    run_store: RunStore,
    step: StepRecord,
    model: type[ModelT],
) -> ModelT:
    if step.selected_output is None:
        raise WorkflowError(f"completed step has no output: {step.step_name}")
    path = run_store.artifacts.verify(step.selected_output)
    try:
        return model.model_validate_json(path.read_bytes())
    except ValueError as exc:
        raise WorkflowError(f"invalid output for {step.step_name}: {exc}") from exc


def _selected_candidate(result: MediaOperationResult) -> CandidateRecord:
    for candidate in result.candidates:
        if candidate.candidate_key == result.decision.selected_candidate:
            return candidate
    raise WorkflowError("selected candidate is missing from operation result")


def _required_artifact(candidate: CandidateRecord) -> ArtifactRef:
    if candidate.artifact_ref is None:
        raise WorkflowError("selected candidate has no media artifact")
    return candidate.artifact_ref


def _story_shot(plan: StoryPlan, shot_key: str) -> StoryShot:
    for shot in plan.ordered_shots:
        if shot.shot_key == shot_key:
            return shot
    raise WorkflowError(f"unknown story shot {shot_key}")


def _shot_index(plan: StoryPlan, shot_key: str) -> int:
    for index, shot in enumerate(plan.ordered_shots, start=1):
        if shot.shot_key == shot_key:
            return index
    raise WorkflowError(f"unknown story shot {shot_key}")


def _prompt_inputs(prompts: tuple[ProviderPrompt, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for prompt in prompts:
        for attachment in prompt.attachments:
            key = (attachment.name, attachment.artifact_ref.sha256)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "name": attachment.name,
                    "artifact_ref": attachment.artifact_ref.model_dump(mode="json"),
                }
            )
    return rows


def _criterion_statements(prompts: tuple[ProviderPrompt, ...]) -> dict[str, str]:
    statements: dict[str, str] = {}
    pattern = re.compile(r"^- (.+?): \[[^\]]+\] (.+)$")
    for prompt in prompts:
        if prompt.purpose.value != "evaluation":
            continue
        for line in prompt.prompt.splitlines():
            match = pattern.fullmatch(line)
            if match is not None:
                statements[match.group(1)] = match.group(2)
    return statements


def _evaluation_markdown(
    report: EvaluationReport,
    statements: dict[str, str],
) -> str:
    lines = [
        "# Evaluation",
        "",
        f"- Candidate: `{report.candidate_key}`",
        f"- Evaluator call: `{report.evaluator_ref}`",
        "",
        "| Status | Criterion | Statement | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for criterion in report.criterion_results:
        lines.append(
            "| "
            + " | ".join(
                (
                    criterion.status.value.upper(),
                    f"`{_markdown_cell(criterion.criterion_id)}`",
                    _markdown_cell(
                        statements.get(
                            criterion.criterion_id,
                            "Statement unavailable",
                        )
                    ),
                    _markdown_cell(criterion.evidence),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _selection_markdown(result: MediaOperationResult) -> str:
    decision = result.decision
    lines = [
        "# Selection",
        "",
        f"- Outcome: **{decision.outcome.value}**",
        f"- Selected candidate: `{decision.selected_candidate}`",
        f"- Selected report: `{decision.selected_report}`",
        "",
        (
            "| Rank | Candidate | Requirement failures | Requirement unknowns | "
            "Priority loss | Continuity loss | Preference loss | Technical quality |"
        ),
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, rank in enumerate(decision.ranks, start=1):
        vector = rank.vector
        lines.append(
            f"| {index} | `{rank.candidate_key}` | "
            f"{vector.requirement_failures} | {vector.requirement_unknowns} | "
            f"{vector.priority_loss} | {vector.continuity_loss} | "
            f"{vector.preference_loss} | {vector.technical_quality:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _markdown_cell(value: str) -> str:
    return " ".join(value.split()).replace("|", r"\|")


def _write_json(path: Path, value: Any) -> None:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    _write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_csv(
    path: Path,
    header: tuple[str, ...],
    rows: list[tuple[Any, ...]],
) -> None:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    _write_text(path, stream.getvalue())


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise WorkflowError(f"cannot write human output {path.name}: {exc}") from exc


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise WorkflowError(f"cannot copy human output {destination.name}: {exc}") from exc


def _reset_directory(path: Path, *, within: Path) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(within.resolve()):
        raise WorkflowError("human output path escapes output root")
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise WorkflowError(f"unsafe human output directory {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _extract_first_frame(video: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(video),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                str(destination),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorkflowError(f"cannot extract frame zero from {video.name}") from exc
    if completed.returncode != 0 or not destination.is_file():
        detail = completed.stderr.strip()[-500:] or "ffmpeg produced no frame"
        raise WorkflowError(f"cannot extract frame zero from {video.name}: {detail}")


def _ordered_name(index: int, alias: str) -> str:
    return f"{index:03d}_{_slug(alias)}"


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("._-").lower()
    return result[:80] or "item"


def _media_extension(media_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "video/mp4": ".mp4",
    }.get(media_type.lower(), ".bin")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
