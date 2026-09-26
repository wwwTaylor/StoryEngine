"""Application configuration without serialized credentials."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

from pydantic import Field, StringConstraints, field_validator, model_validator

from story_engine.domain.common import FrozenModel
from story_engine.errors import ConfigurationError
from story_engine.ids import canonical_json
from story_engine.security import is_credential_key, redact_data

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
EnvironmentName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    ),
]


class ProviderSpec(FrozenModel):
    adapter: NonEmpty
    model: NonEmpty
    base_url: str | None = None
    api_key_env: EnvironmentName | None = None
    timeout_seconds: float = Field(default=120.0, gt=0, le=3600)
    options: tuple[tuple[str, str], ...] = ()

    @model_validator(mode="before")
    @classmethod
    def normalize_options(cls, value: Any) -> Any:
        if isinstance(value, dict) and "options" in value:
            raw_options = value["options"]
            if isinstance(raw_options, dict):
                decoded_options = raw_options
            elif isinstance(raw_options, list | tuple):
                decoded_options: dict[str, Any] = {}
                for pair in raw_options:
                    if not isinstance(pair, list | tuple) or len(pair) != 2:
                        raise ValueError(
                            "provider option pairs must contain a name and canonical JSON value"
                        )
                    key, encoded = pair
                    if not isinstance(key, str) or not isinstance(encoded, str):
                        raise ValueError("provider option pairs must contain strings")
                    if key in decoded_options:
                        raise ValueError("duplicate provider option")
                    try:
                        decoded_options[key] = json.loads(encoded)
                    except json.JSONDecodeError as exc:
                        raise ValueError("invalid canonical provider option") from None
            else:
                raise ValueError("provider options must be a mapping")
            forbidden = _credential_option_paths(decoded_options)
            if forbidden:
                raise ValueError(
                    "credentials are forbidden in provider options; use api_key_env"
                )
            copied = dict(value)
            copied["options"] = tuple(
                sorted(
                    (str(key), canonical_json(item))
                    for key, item in decoded_options.items()
                )
            )
            return copied
        return value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url cannot contain credentials, query, or fragment")
        return value.rstrip("/")

    def decoded_options(self) -> dict[str, Any]:
        return {key: json.loads(value) for key, value in self.options}


def _credential_option_paths(
    value: Any,
    path: tuple[str, ...] = (),
) -> set[str]:
    forbidden: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            label = str(key)
            current = (*path, label)
            if is_credential_key(label):
                forbidden.add(".".join(current))
            forbidden.update(_credential_option_paths(item, current))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            forbidden.update(_credential_option_paths(item, (*path, str(index))))
    return forbidden


class ProviderRoles(FrozenModel):
    planner: ProviderSpec
    image: ProviderSpec
    video: ProviderSpec
    judge: ProviderSpec


class SpatialFailurePolicy(StrEnum):
    STRICT = "strict"
    DEGRADE = "degrade"


class GenerationPolicy(FrozenModel):
    attempts: int = Field(default=3, ge=1, le=10)
    story_planning_attempts: int = Field(default=3, ge=1, le=10)
    grounding_attempts: int = Field(default=3, ge=1, le=10)
    candidates_per_attempt: int = Field(default=3, ge=1, le=16)
    provider_retries: int = Field(default=2, ge=0, le=8)
    max_concurrency: int = Field(default=4, ge=1, le=64)
    spatial_failure_policy: SpatialFailurePolicy = SpatialFailurePolicy.STRICT
    spatial_repair_attempts: int = Field(default=2, ge=0, le=10)
    novel_station_attempts: int = Field(default=2, ge=0, le=10)
    max_hfov_degrees: float = Field(default=110.0, gt=0, le=110)
    safe_fallback_hfov_min: float = Field(default=60.0, gt=0, le=90)
    safe_fallback_hfov_max: float = Field(default=90.0, gt=0, le=90)

    @model_validator(mode="after")
    def validate_spatial_limits(self) -> GenerationPolicy:
        if self.safe_fallback_hfov_min > self.safe_fallback_hfov_max:
            raise ValueError("safe fallback minimum HFOV cannot exceed maximum")
        if self.safe_fallback_hfov_max > self.max_hfov_degrees:
            raise ValueError("safe fallback HFOV cannot exceed the maximum HFOV")
        return self


class ProjectConfig(FrozenModel):
    output_root: Path = Path("runs")


class StorageConfig(FrozenModel):
    artifact_root: Path | None = None


class AppConfig(FrozenModel):
    project: ProjectConfig = ProjectConfig()
    providers: ProviderRoles
    generation: GenerationPolicy = GenerationPolicy()
    storage: StorageConfig = StorageConfig()

    def redacted_dict(self) -> dict[str, Any]:
        # Rejected at input; redaction remains a second persistence boundary for
        # callers that bypass validation with model_construct/model_copy.
        redacted = self.model_dump(mode="json")
        providers = redacted["providers"]
        for role in ("planner", "image", "video", "judge"):
            providers[role]["options"] = redact_data(
                getattr(self.providers, role).decoded_options()
            )
        return redacted


@dataclass(frozen=True, slots=True, repr=False)
class SecretValue:
    """A credential that cannot accidentally reveal itself through repr/str."""

    _value: str

    def __post_init__(self) -> None:
        if not self._value:
            raise ConfigurationError("credential is empty")

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "SecretValue(***)"

    def __str__(self) -> str:
        return "***"

    __hash__ = None  # type: ignore[assignment]


def credential_from_environment(spec: ProviderSpec) -> SecretValue | None:
    if spec.api_key_env is None:
        return None
    value = os.environ.get(spec.api_key_env)
    if not value:
        raise ConfigurationError(
            f"required credential environment variable is unset: {spec.api_key_env}"
        )
    return SecretValue(value)
