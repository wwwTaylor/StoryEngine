"""Typed errors used at StoryEngine ownership boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class StoryEngineError(Exception):
    """Base error for expected StoryEngine failures."""


class ConfigurationError(StoryEngineError):
    """Configuration is invalid or incomplete."""


class ContractError(StoryEngineError):
    """A domain or provider contract was violated."""


class PlanComplexityError(ContractError):
    """A request must be simplified upstream instead of being truncated."""


class StoryCompilationError(ContractError):
    """A StoryDraft cannot be compiled into a valid StoryPlan."""


class WorldStateError(ContractError):
    """A world state or transition is invalid."""


class ReferenceError(ContractError):
    """Reference generation or selection cannot satisfy its contract."""


class SpatialError(ContractError):
    """Spatial evidence is invalid or insufficient."""


@dataclass(slots=True)
class GroundingValidationExhausted(SpatialError):
    """All configured complete Grounding attempts failed deterministic validation."""

    scene_key: str
    station_key: str
    reference_hash: str
    attempt_count: int
    failed_target_aliases: tuple[str, ...]
    last_validation_code: str
    last_validation_error: str
    provider_call_refs: tuple[str, ...]
    logical_attempt_refs: tuple[str, ...] = ()
    response_refs: tuple[str, ...] = ()
    validation_refs: tuple[str, ...] = ()
    failure_code: str = "GROUNDING_VALIDATION_EXHAUSTED"

    def __str__(self) -> str:
        aliases = ", ".join(self.failed_target_aliases) or "unknown targets"
        return (
            f"{self.failure_code}: grounding failed validation after "
            f"{self.attempt_count} complete attempts for {aliases}; "
            f"{self.last_validation_code}: {self.last_validation_error}"
        )


@dataclass(slots=True)
class CandidatePreflightExhausted(SpatialError):
    """All eligible generated candidates failed their spatial preflight."""

    operation_key: str
    candidate_keys: tuple[str, ...]
    failure_codes: tuple[str, ...]
    detail: str

    def __str__(self) -> str:
        codes = ", ".join(self.failure_codes) or "spatial_preflight_failed"
        return f"candidate spatial preflight exhausted [{codes}]: {self.detail}"


class RenderCompilationError(ContractError):
    """A RenderPlan cannot be frozen."""


class ArtifactError(StoryEngineError):
    """An artifact cannot be persisted or verified."""


class MediaValidationError(StoryEngineError):
    """Media is not technically usable."""


class AssemblyError(StoryEngineError):
    """Selected shot media cannot be assembled."""


class WorkflowError(StoryEngineError):
    """The fixed workflow entered an invalid state."""


class ProviderErrorKind(StrEnum):
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RESPONSE_CONTRACT = "response_contract"
    JOB_FAILED = "job_failed"
    DOWNLOAD = "download"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class ProviderError(StoryEngineError):
    """A redacted, normalized provider failure."""

    kind: ProviderErrorKind
    message: str
    retryable: bool = False
    status_code: int | None = None
    details: tuple[tuple[str, Any], ...] = ()

    def __str__(self) -> str:
        code = f" status={self.status_code}" if self.status_code is not None else ""
        return f"{self.kind.value}{code}: {self.message}"
