"""Single-file project request and runtime configuration boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from story_engine.config import AppConfig
from story_engine.domain.request import ProjectRequest, project_request_from_mapping
from story_engine.errors import ConfigurationError, ContractError
from story_engine.security import validation_summary


@dataclass(frozen=True, slots=True)
class RunSpec:
    request: ProjectRequest
    config: AppConfig


def load_run_spec(path: Path) -> RunSpec:
    """Load a strict request plus its per-file runtime configuration."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot read run definition {path}: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise ContractError(f"invalid YAML/JSON in run definition{location}") from None
    if not isinstance(raw, dict):
        raise ContractError("run definition root must be a mapping")

    request_raw = dict(raw)
    if "runtime" not in request_raw:
        raise ConfigurationError(f"run definition {path} requires a runtime section")
    runtime_raw = request_raw.pop("runtime")
    request = project_request_from_mapping(request_raw, source=path)
    try:
        config = AppConfig.model_validate(runtime_raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"invalid runtime configuration: {validation_summary(exc)}"
        ) from None
    # Validate options before any run files are created, without loading credentials
    # or constructing network clients.
    from story_engine.providers.registry import validate_provider_config

    validate_provider_config(config)
    return RunSpec(request=request, config=config)
