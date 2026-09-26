"""Fixed workflow state and step records."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from story_engine.domain.common import FrozenModel
from story_engine.domain.evaluation import ProviderUsage
from story_engine.domain.trace import ProviderPrompt
from story_engine.errors import ArtifactError, WorkflowError
from story_engine.ids import canonical_hash
from story_engine.storage import ArtifactRef, ArtifactStore, atomic_write_json, read_json


class ProjectStatus(StrEnum):
    CREATED = "created"
    PREFLIGHTED = "preflighted"
    STORY_PLANNED = "story_planned"
    REFERENCES_READY = "references_ready"
    RENDER_PLANNED = "render_planned"
    SHOTS_RENDERED = "shots_rendered"
    ASSEMBLED = "assembled"
    DELIVERED = "delivered"
    DELIVERED_DEGRADED = "delivered_degraded"
    PROCESS_FAILED = "process_failed"
    INTERRUPTED = "interrupted"


class ShotStatus(StrEnum):
    PENDING = "pending"
    FRAME_SELECTED = "frame_selected"
    VIDEO_SELECTED = "video_selected"
    COMPLETED = "completed"
    COMPLETED_DEGRADED = "completed_degraded"


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class AttemptRecord(FrozenModel):
    logical_attempt: int = Field(ge=1)
    attempt_ref: str | None = None
    provider_call_refs: tuple[str, ...] = ()
    candidate_refs: tuple[str, ...] = ()
    evaluation_refs: tuple[str, ...] = ()
    target_aliases: tuple[str, ...] = ()
    response_refs: tuple[ArtifactRef, ...] = ()
    validation_refs: tuple[ArtifactRef, ...] = ()
    validation_status: Literal["valid", "invalid"] | None = None
    validation_code: str | None = None
    provider_usage: ProviderUsage = ProviderUsage()
    prompts: tuple[ProviderPrompt, ...] = ()
    started_at: str
    completed_at: str | None = None


class StepRecord(FrozenModel):
    step_name: str
    step_key: str
    status: StepStatus
    input_refs: tuple[str, ...]
    attempt_records: tuple[AttemptRecord, ...] = ()
    selected_output: ArtifactRef | None = None
    selected_key: str | None = None
    pending_provider_job: str | None = None
    pending_logical_attempt: int | None = Field(default=None, ge=1)
    pending_candidate_index: int | None = Field(default=None, ge=1)
    pending_provider_usage: ProviderUsage | None = None
    pending_provider_call_refs: tuple[str, ...] = ()
    error: str | None = None

    @model_validator(mode="after")
    def validate_pending_job(self) -> StepRecord:
        coordinates = (self.pending_logical_attempt, self.pending_candidate_index)
        if self.pending_provider_job is None and any(item is not None for item in coordinates):
            raise ValueError("pending video coordinates require a provider job")
        if self.pending_provider_job is not None and any(item is None for item in coordinates):
            raise ValueError("pending provider job requires logical attempt and candidate index")
        if self.pending_provider_job is None and self.pending_provider_usage is not None:
            raise ValueError("pending provider usage requires a provider job")
        if self.pending_provider_job is None and self.pending_provider_call_refs:
            raise ValueError("pending provider call refs require a provider job")
        return self


class ShotRunState(FrozenModel):
    shot_key: str
    status: ShotStatus = ShotStatus.PENDING
    selected_frame_candidate: str | None = None
    selected_video_candidate: str | None = None
    degraded: bool = False
    boundary_conflicts: tuple[str, ...] = ()
    previous_end_frame_allowed: bool | None = None


class RunState(FrozenModel):
    run_id: str
    request_hash: str
    status: ProjectStatus
    active_story_plan: ArtifactRef | None = None
    active_reference_library: ArtifactRef | None = None
    active_render_plan: ArtifactRef | None = None
    steps: tuple[StepRecord, ...] = ()
    shots: tuple[ShotRunState, ...] = ()
    final_video: ArtifactRef | None = None
    manifest: ArtifactRef | None = None
    error: str | None = None
    updated_at: str

    @model_validator(mode="after")
    def validate_uniqueness(self) -> RunState:
        step_names = [item.step_name for item in self.steps]
        if len(step_names) != len(set(step_names)):
            raise ValueError("step names must be unique in fixed workflow")
        shot_keys = [item.shot_key for item in self.shots]
        if len(shot_keys) != len(set(shot_keys)):
            raise ValueError("shot run states must be unique")
        return self


ALLOWED_PROJECT_TRANSITIONS: dict[ProjectStatus, frozenset[ProjectStatus]] = {
    ProjectStatus.CREATED: frozenset(
        {ProjectStatus.PREFLIGHTED, ProjectStatus.PROCESS_FAILED, ProjectStatus.INTERRUPTED}
    ),
    ProjectStatus.PREFLIGHTED: frozenset(
        {ProjectStatus.STORY_PLANNED, ProjectStatus.PROCESS_FAILED, ProjectStatus.INTERRUPTED}
    ),
    ProjectStatus.STORY_PLANNED: frozenset(
        {ProjectStatus.REFERENCES_READY, ProjectStatus.PROCESS_FAILED, ProjectStatus.INTERRUPTED}
    ),
    ProjectStatus.REFERENCES_READY: frozenset(
        {ProjectStatus.RENDER_PLANNED, ProjectStatus.PROCESS_FAILED, ProjectStatus.INTERRUPTED}
    ),
    ProjectStatus.RENDER_PLANNED: frozenset(
        {ProjectStatus.SHOTS_RENDERED, ProjectStatus.PROCESS_FAILED, ProjectStatus.INTERRUPTED}
    ),
    ProjectStatus.SHOTS_RENDERED: frozenset(
        {ProjectStatus.ASSEMBLED, ProjectStatus.PROCESS_FAILED, ProjectStatus.INTERRUPTED}
    ),
    ProjectStatus.ASSEMBLED: frozenset(
        {
            ProjectStatus.DELIVERED,
            ProjectStatus.DELIVERED_DEGRADED,
            ProjectStatus.PROCESS_FAILED,
            ProjectStatus.INTERRUPTED,
        }
    ),
    ProjectStatus.INTERRUPTED: frozenset(
        {
            ProjectStatus.PREFLIGHTED,
            ProjectStatus.STORY_PLANNED,
            ProjectStatus.REFERENCES_READY,
            ProjectStatus.RENDER_PLANNED,
            ProjectStatus.SHOTS_RENDERED,
            ProjectStatus.ASSEMBLED,
            ProjectStatus.PROCESS_FAILED,
        }
    ),
    ProjectStatus.DELIVERED: frozenset(),
    ProjectStatus.DELIVERED_DEGRADED: frozenset(),
    ProjectStatus.PROCESS_FAILED: frozenset(),
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def step_key(
    *,
    kind: str,
    input_hashes: tuple[str, ...],
    implementation_version: str,
    renderer_version: str | None,
    provider_fingerprint: str | None,
    behavior_config: Any,
) -> str:
    return canonical_hash(
        {
            "kind": kind,
            "input_hashes": input_hashes,
            "implementation_version": implementation_version,
            "renderer_version": renderer_version,
            "provider_fingerprint": provider_fingerprint,
            "behavior_config": behavior_config,
        }
    )


class RunStore:
    """Run state is authoritative; events are append-only audit records."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Protect custom output roots as well as the conventional runs/ folder.
        ignore_path = self.run_dir / ".gitignore"
        if not ignore_path.exists():
            ignore_path.write_text("*\n", encoding="utf-8")
        self.artifacts = ArtifactStore(self.run_dir / "artifacts")
        self.output_dir = self.run_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "state.json"
        self.events_path = self.run_dir / "events.jsonl"

    def initialize(
        self, *, run_id: str, request_hash: str, request: Any, redacted_config: Any
    ) -> RunState:
        if self.state_path.exists():
            return self.load()
        atomic_write_json(self.run_dir / "request.json", request)
        atomic_write_json(self.run_dir / "config.redacted.json", redacted_config)
        state = RunState(
            run_id=run_id,
            request_hash=request_hash,
            status=ProjectStatus.CREATED,
            updated_at=utc_now(),
        )
        self.save(state)
        self.append_event("run_created", {"run_id": run_id})
        return state

    def load(self) -> RunState:
        try:
            return RunState.model_validate(read_json(self.state_path))
        except ValueError as exc:
            raise WorkflowError(f"invalid run state: {exc}") from exc

    def save(self, state: RunState) -> None:
        atomic_write_json(self.state_path, state)

    def transition(self, state: RunState, status: ProjectStatus, **updates: Any) -> RunState:
        if status not in ALLOWED_PROJECT_TRANSITIONS[state.status]:
            raise WorkflowError(
                f"invalid project transition {state.status.value} -> {status.value}"
            )
        changed = state.model_copy(update={"status": status, "updated_at": utc_now(), **updates})
        self.save(changed)
        self.append_event(
            "project_transition",
            {"from": state.status.value, "to": status.value},
        )
        return changed

    def upsert_step(self, state: RunState, step: StepRecord) -> RunState:
        records = {item.step_name: item for item in state.steps}
        records[step.step_name] = step
        changed = state.model_copy(
            update={
                "steps": tuple(sorted(records.values(), key=lambda item: item.step_name)),
                "updated_at": utc_now(),
            }
        )
        self.save(changed)
        return changed

    def upsert_shot(self, state: RunState, shot: ShotRunState) -> RunState:
        records = {item.shot_key: item for item in state.shots}
        records[shot.shot_key] = shot
        changed = state.model_copy(
            update={
                "shots": tuple(sorted(records.values(), key=lambda item: item.shot_key)),
                "updated_at": utc_now(),
            }
        )
        self.save(changed)
        return changed

    def reusable_step(self, state: RunState, name: str, key: str) -> StepRecord | None:
        for step in state.steps:
            if step.step_name != name or step.step_key != key:
                continue
            if step.status != StepStatus.COMPLETED:
                return None
            if step.selected_output is not None:
                try:
                    self.artifacts.verify(step.selected_output)
                except ArtifactError:
                    return None
            return step
        return None

    def append_event(self, event: str, data: Any) -> None:
        record = {
            "time": utc_now(),
            "event": event,
            "data": data,
        }
        line = __import__("json").dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        try:
            with self.events_path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
                file.flush()
                os.fsync(file.fileno())
        except OSError as exc:
            raise WorkflowError(f"cannot append event: {exc}") from exc
