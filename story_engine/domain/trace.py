"""Persisted descriptions of exact prompts sent to Provider ports."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from story_engine.domain.common import FrozenModel
from story_engine.domain.request import PixelSize
from story_engine.storage import ArtifactRef


class PromptPurpose(StrEnum):
    ASSET_ALIGNMENT = "asset_alignment"
    PLANNING = "planning"
    GROUNDING = "grounding"
    GENERATION = "generation"
    EVALUATION = "evaluation"


class PromptAttachment(FrozenModel):
    name: str
    artifact_ref: ArtifactRef


class ProviderPrompt(FrozenModel):
    purpose: PromptPurpose
    provider_role: str
    logical_attempt: int = Field(ge=1)
    candidate_index: int | None = Field(default=None, ge=1)
    candidate_key: str | None = None
    prompt: str
    attachments: tuple[PromptAttachment, ...] = ()
    response_contract: str | None = None
    aspect_ratio: str | None = None
    resolution: PixelSize | None = None
    duration_seconds: int | None = Field(default=None, gt=0)
    fps: int | None = Field(default=None, gt=0)
