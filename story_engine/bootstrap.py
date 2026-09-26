"""Composition root for CLI execution."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from story_engine.config import AppConfig
from story_engine.domain.request import ProjectRequest
from story_engine.errors import ConfigurationError, WorkflowError
from story_engine.providers.registry import build_provider_set, validate_provider_config
from story_engine.run_spec import load_run_spec
from story_engine.run_state import RunStore
from story_engine.storage import read_json
from story_engine.security import validation_summary
from story_engine.workflow import Workflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_command(args: argparse.Namespace) -> int:
    if args.command == "run":
        spec = load_run_spec(args.definition)
        config = spec.config
        request = spec.request
        output_root = _output_root(config)
        run_id = args.run_id or _new_run_id(request)
        _validate_run_id(run_id)
        run_store = RunStore(output_root / run_id)
        state = run_store.initialize(
            run_id=run_id,
            request_hash=request.request_hash,
            request=request,
            redacted_config=config.redacted_dict(),
        )
        if state.request_hash != request.request_hash:
            raise WorkflowError("run directory belongs to another request")
        if _load_persisted_config(run_store.run_dir) != config:
            raise WorkflowError(
                "run directory belongs to another runtime config; use a new run ID"
            )
    elif args.command == "resume":
        run_dir = _inside_project(args.run_dir, label="run directory")
        run_store = RunStore(run_dir)
        try:
            request = ProjectRequest.model_validate(read_json(run_dir / "request.json"))
        except ValueError as exc:
            raise WorkflowError(f"invalid persisted request: {validation_summary(exc)}") from None
        config = _load_persisted_config(run_dir)
        state = run_store.load()
        if state.request_hash != request.request_hash:
            raise WorkflowError("persisted request hash does not match run state")
    else:
        raise ConfigurationError(f"unsupported execution command: {args.command}")

    providers = build_provider_set(config, run_store.artifacts)
    final = asyncio.run(
        Workflow(
            request=request,
            config=config,
            run_store=run_store,
            providers=providers,
        ).execute()
    )
    print(
        json.dumps(
            {
                "run_id": final.run_id,
                "status": final.status.value,
                "run_dir": str(run_store.run_dir),
                "video": (
                    str(run_store.output_dir / "08_final" / "final.mp4")
                    if final.final_video is not None
                    else None
                ),
                "manifest": (
                    str(run_store.output_dir / "08_final" / "manifest.json")
                    if final.manifest is not None
                    else None
                ),
                "output_index": str(run_store.output_dir / "OPEN_ME.html"),
            },
            sort_keys=True,
        )
    )
    return 0


def _output_root(config: AppConfig) -> Path:
    if config.storage.artifact_root is not None:
        raise ConfigurationError(
            "storage.artifact_root is reserved; artifacts are fixed below each run"
        )
    raw = config.project.output_root
    candidate = raw if raw.is_absolute() else PROJECT_ROOT / raw
    return _inside_project(candidate, label="project.output_root")


def _load_persisted_config(run_dir: Path) -> AppConfig:
    try:
        config = AppConfig.model_validate(read_json(run_dir / "config.redacted.json"))
    except ValueError as exc:
        raise WorkflowError(
            f"invalid persisted runtime config: {validation_summary(exc)}"
        ) from None
    validate_provider_config(config)
    return config


def _inside_project(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ConfigurationError(f"{label} must remain inside {PROJECT_ROOT}")
    return resolved


def _new_run_id(request: ProjectRequest) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{request.task_id}-{timestamp}-{request.request_hash[:8]}"


def _validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise ConfigurationError("run_id must be a safe 1-128 character path segment")
