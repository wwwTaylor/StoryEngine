"""Core provider contracts. Raw provider JSON is forbidden here."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Protocol

from pydantic import Field, model_validator

from story_engine.domain.common import FrozenModel
from story_engine.domain.request import PixelSize
from story_engine.ids import canonical_hash
from story_engine.storage import ArtifactRef


class StructuredOutputMode(StrEnum):
    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"
    TEXT = "text"


class ResponseContract(StrEnum):
    ASSET_ALIGNMENT = "asset_alignment_v1"
    STORY_DRAFT = "story_draft_v2"
    GROUNDING_ROWS = "grounding_rows_v3_conditional_visibility"
    EVALUATION_ROWS = "evaluation_rows_v1"
    FIRST_FRAME_PARTICIPATION = "first_frame_participation_v1"


class ProviderIdentity(FrozenModel):
    adapter: str
    model: str
    endpoint: str
    adapter_version: str


class TextCapabilities(FrozenModel):
    request_protocol: str = "typed_text"
    attachment_modes: tuple[str, ...] = ()
    structured_output_modes: tuple[StructuredOutputMode, ...]
    max_prompt_characters: int = Field(gt=0)
    max_attachments: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_modes(self) -> TextCapabilities:
        if not self.structured_output_modes:
            raise ValueError("text structured output modes cannot be empty")
        if len(self.structured_output_modes) != len(set(self.structured_output_modes)):
            raise ValueError("text structured output modes must be unique")
        if len(self.attachment_modes) != len(set(self.attachment_modes)):
            raise ValueError("text attachment modes must be unique")
        return self


class ImageCapabilities(FrozenModel):
    request_protocol: str = "typed_image"
    input_modes: tuple[str, ...] = ()
    output_modes: tuple[str, ...] = ("binary",)
    max_input_images: int = Field(default=0, ge=0)
    supported_aspect_ratios: tuple[str, ...]
    supported_resolutions: tuple[PixelSize, ...] = ()
    supports_native_negative_prompt: bool = False
    prompt_enhancement_behavior: str = "provider_defined"
    submit_is_idempotent: bool = False
    max_prompt_characters: int = Field(gt=0)

    def supports_resolution(self, resolution: PixelSize) -> bool:
        return not self.supported_resolutions or resolution in self.supported_resolutions

    @model_validator(mode="after")
    def validate_modes(self) -> ImageCapabilities:
        if not self.supported_aspect_ratios:
            raise ValueError("image aspect ratios cannot be empty")
        for label, values in (
            ("image input modes", self.input_modes),
            ("image output modes", self.output_modes),
            ("image aspect ratios", self.supported_aspect_ratios),
            ("image resolutions", self.supported_resolutions),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        if not self.output_modes:
            raise ValueError("image output modes cannot be empty")
        return self


class VideoCapabilities(FrozenModel):
    request_protocol: str = "typed_video"
    input_modes: tuple[str, ...] = ("artifact_ref",)
    output_modes: tuple[str, ...] = ("binary",)
    supports_start_image: bool
    max_reference_images: int = Field(default=0, ge=0)
    supported_durations: tuple[int, ...]
    supported_resolutions: tuple[PixelSize, ...]
    supported_fps: tuple[int, ...]
    supports_camera_motion: bool = True
    job_mode: Literal["synchronous", "asynchronous"] = "asynchronous"
    submit_is_idempotent: bool = False
    poll_interval_seconds: float = Field(default=10.0, gt=0, le=300)
    job_timeout_seconds: float = Field(default=1_800.0, gt=0, le=86_400)
    max_prompt_characters: int = Field(gt=0)

    def supports_resolution(self, resolution: PixelSize) -> bool:
        return resolution in self.supported_resolutions

    @model_validator(mode="after")
    def validate_modes(self) -> VideoCapabilities:
        for label, values in (
            ("video input modes", self.input_modes),
            ("video output modes", self.output_modes),
            ("video durations", self.supported_durations),
            ("video resolutions", self.supported_resolutions),
            ("video FPS values", self.supported_fps),
        ):
            if not values:
                raise ValueError(f"{label} cannot be empty")
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        if any(value <= 0 for value in self.supported_durations):
            raise ValueError("video durations must be positive")
        if any(value <= 0 for value in self.supported_fps):
            raise ValueError("video FPS values must be positive")
        return self


class JudgeCapabilities(FrozenModel):
    request_protocol: str = "typed_judge"
    attachment_modes: tuple[str, ...] = ()
    structured_output_modes: tuple[StructuredOutputMode, ...]
    max_media: int = Field(ge=1)
    supported_media_types: tuple[str, ...]
    max_prompt_characters: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_modes(self) -> JudgeCapabilities:
        for label, values in (
            ("judge structured output modes", self.structured_output_modes),
            ("judge attachment modes", self.attachment_modes),
            ("judge media types", self.supported_media_types),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        if not self.structured_output_modes or not self.supported_media_types:
            raise ValueError("judge output modes and media types cannot be empty")
        return self


class ProviderCapabilityProfile(FrozenModel):
    identity: ProviderIdentity
    behavior_fingerprint: str = Field(
        default_factory=lambda: canonical_hash({}),
        pattern=r"^[0-9a-f]{64}$",
    )
    text: TextCapabilities | None = None
    image: ImageCapabilities | None = None
    video: VideoCapabilities | None = None
    judge: JudgeCapabilities | None = None

    @property
    def fingerprint(self) -> str:
        return canonical_hash(self)


class RuntimeCapabilities(FrozenModel):
    planner: ProviderCapabilityProfile
    image: ProviderCapabilityProfile
    video: ProviderCapabilityProfile
    judge: ProviderCapabilityProfile

    @property
    def fingerprint(self) -> str:
        return canonical_hash(self)


class Attachment(FrozenModel):
    name: str
    artifact_ref: ArtifactRef


class TextRequest(FrozenModel):
    prompt: str
    attachments: tuple[Attachment, ...] = ()
    response_contract: ResponseContract | None = None


class ImageRequest(FrozenModel):
    prompt: str
    input_images: tuple[Attachment, ...] = ()
    aspect_ratio: str
    resolution: PixelSize


class VideoRequest(FrozenModel):
    prompt: str
    start_image: Attachment
    reference_images: tuple[Attachment, ...] = ()
    duration: int
    resolution: PixelSize
    fps: int


class JudgeRequest(FrozenModel):
    prompt: str
    media: tuple[Attachment, ...]
    response_contract: ResponseContract = ResponseContract.EVALUATION_ROWS


class ProviderCallMetrics(FrozenModel):
    call_ref: str
    provider_request_id: str | None = None
    elapsed_seconds: float = Field(ge=0)
    transport_retries: int = Field(default=0, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class TextResult(FrozenModel):
    text: str
    metrics: ProviderCallMetrics


class MediaPayload(FrozenModel):
    data: bytes
    media_type: str


class ImageResult(FrozenModel):
    payload: MediaPayload
    metrics: ProviderCallMetrics


class VideoResult(FrozenModel):
    payload: MediaPayload
    metrics: ProviderCallMetrics


class VideoJobStatus(StrEnum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoJob(FrozenModel):
    job_id: str
    status: VideoJobStatus
    progress: float | None = Field(default=None, ge=0, le=100)
    error: str | None = None
    metrics: ProviderCallMetrics


class JudgeResult(FrozenModel):
    text: str
    metrics: ProviderCallMetrics


class TextProvider(Protocol):
    @property
    def capability_profile(self) -> ProviderCapabilityProfile: ...

    async def preflight(self) -> None: ...

    async def generate_text(self, request: TextRequest) -> TextResult: ...

    async def aclose(self) -> None: ...


class ImageProvider(Protocol):
    @property
    def capability_profile(self) -> ProviderCapabilityProfile: ...

    async def preflight(self) -> None: ...

    async def generate_image(self, request: ImageRequest) -> ImageResult: ...

    async def aclose(self) -> None: ...


class VideoProvider(Protocol):
    @property
    def capability_profile(self) -> ProviderCapabilityProfile: ...

    async def preflight(self) -> None: ...

    async def submit_video(self, request: VideoRequest) -> VideoJob: ...

    async def poll_video(self, job_id: str) -> VideoJob: ...

    async def download_video(self, job_id: str) -> VideoResult: ...

    async def aclose(self) -> None: ...


class JudgeProvider(Protocol):
    @property
    def capability_profile(self) -> ProviderCapabilityProfile: ...

    async def preflight(self) -> None: ...

    async def judge(self, request: JudgeRequest) -> JudgeResult: ...

    async def aclose(self) -> None: ...
