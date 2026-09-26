"""OpenAI Responses, Images, and asynchronous Videos adapters."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Literal
from urllib.parse import quote

import httpx
from pydantic import Field, model_validator

from story_engine.config import ProviderSpec, SecretValue
from story_engine.domain.common import FrozenModel
from story_engine.domain.request import PixelSize
from story_engine.errors import ConfigurationError, ProviderError, ProviderErrorKind
from story_engine.ids import canonical_hash, sha256_bytes, stable_key
from story_engine.providers.ports import (
    Attachment,
    ImageCapabilities,
    ImageRequest,
    ImageResult,
    JudgeCapabilities,
    JudgeRequest,
    JudgeResult,
    MediaPayload,
    ProviderCallMetrics,
    ProviderCapabilityProfile,
    ProviderIdentity,
    ResponseContract,
    StructuredOutputMode,
    TextCapabilities,
    TextRequest,
    TextResult,
    VideoCapabilities,
    VideoJob,
    VideoJobStatus,
    VideoRequest,
    VideoResult,
)
from story_engine.providers.schemas import schema_for_contract
from story_engine.providers.transport import HttpTransport, TransportResponse
from story_engine.storage import ArtifactStore
from story_engine.security import validation_summary

OPENAI_ADAPTER_VERSION = "openai-adapter-v3"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIResponsesOptions(FrozenModel):
    preflight_mode: Literal["model_get", "model_list"] = "model_get"
    structured_output_mode: StructuredOutputMode = StructuredOutputMode.JSON_SCHEMA
    max_prompt_characters: int = Field(default=100_000, gt=0)
    max_attachments: int = Field(default=16, ge=0)
    judge_max_media: int = Field(default=16, ge=1)
    judge_media_types: tuple[str, ...] = ("image/png", "image/jpeg", "image/webp")

    @model_validator(mode="after")
    def validate_media_protocol(self) -> OpenAIResponsesOptions:
        if not self.judge_media_types or any(
            not item.startswith("image/") for item in self.judge_media_types
        ):
            raise ValueError("OpenAI Responses adapter currently declares image judge media only")
        return self


class OpenAIImageOptions(FrozenModel):
    preflight_mode: Literal["model_get", "model_list"] = "model_get"
    max_input_images: int = Field(default=16, ge=0)
    supported_aspect_ratios: tuple[str, ...]
    supported_resolutions: tuple[PixelSize, ...]
    max_prompt_characters: int = Field(default=32_000, gt=0)
    output_format: Literal["png", "jpeg", "webp"] = "png"
    quality: Literal["low", "medium", "high", "auto"] = "auto"


class OpenAIVideoOptions(FrozenModel):
    preflight_mode: Literal["model_get", "model_list"] = "model_get"
    supported_durations: tuple[int, ...]
    supported_resolutions: tuple[PixelSize, ...]
    supported_fps: tuple[int, ...]
    max_prompt_characters: int = Field(default=32_000, gt=0)
    poll_interval_seconds: float = Field(default=10.0, gt=0, le=300)
    job_timeout_seconds: float = Field(default=1_800.0, gt=0, le=86_400)


class OpenAIResponsesAdapter:
    """One protocol adapter can serve both TextProvider and JudgeProvider."""

    def __init__(
        self,
        spec: ProviderSpec,
        credential: SecretValue,
        store: ArtifactStore,
        *,
        max_retries: int,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._spec = spec
        self._store = store
        self._options = _validated_options(OpenAIResponsesOptions, spec)
        endpoint = spec.base_url or DEFAULT_BASE_URL
        self._transport = HttpTransport(
            base_url=endpoint,
            credential=credential,
            timeout_seconds=spec.timeout_seconds,
            max_retries=max_retries,
            client=client,
        )
        identity = ProviderIdentity(
            adapter=spec.adapter,
            model=spec.model,
            endpoint=endpoint,
            adapter_version=OPENAI_ADAPTER_VERSION,
        )
        modes = (self._options.structured_output_mode,)
        self._profile = ProviderCapabilityProfile(
            identity=identity,
            behavior_fingerprint=canonical_hash(self._options),
            text=TextCapabilities(
                request_protocol="openai_responses",
                attachment_modes=("data_url",),
                structured_output_modes=modes,
                max_prompt_characters=self._options.max_prompt_characters,
                max_attachments=self._options.max_attachments,
            ),
            judge=JudgeCapabilities(
                request_protocol="openai_responses",
                attachment_modes=("data_url",),
                structured_output_modes=modes,
                max_media=self._options.judge_max_media,
                supported_media_types=self._options.judge_media_types,
                max_prompt_characters=self._options.max_prompt_characters,
            ),
        )

    def __repr__(self) -> str:
        return (
            f"OpenAIResponsesAdapter(model={self._spec.model!r}, "
            f"endpoint={self._profile.identity.endpoint!r}, credential=***)"
        )

    @property
    def capability_profile(self) -> ProviderCapabilityProfile:
        return self._profile

    async def preflight(self) -> None:
        await _preflight_model(
            self._transport,
            self._spec.model,
            self._options.preflight_mode,
        )

    async def aclose(self) -> None:
        await self._transport.close()

    async def generate_text(self, request: TextRequest) -> TextResult:
        capability = self._profile.text
        if capability is None:
            raise AssertionError("text capability is missing")
        if len(request.prompt) > capability.max_prompt_characters:
            raise _invalid("text prompt exceeds declared provider capability")
        if len(request.attachments) > capability.max_attachments:
            raise _invalid("text request has too many attachments")
        body = self._responses_body(
            request.prompt,
            request.attachments,
            request.response_contract,
        )
        envelope = await self._transport.request("POST", "responses", json_body=body)
        payload = _json_object(envelope)
        return TextResult(
            text=_response_text(payload),
            metrics=self._metrics(envelope, payload),
        )

    async def judge(self, request: JudgeRequest) -> JudgeResult:
        capability = self._profile.judge
        if capability is None:
            raise AssertionError("judge capability is missing")
        if len(request.prompt) > capability.max_prompt_characters:
            raise _invalid("judge prompt exceeds declared provider capability")
        if len(request.media) > capability.max_media:
            raise _invalid("judge request has too many media attachments")
        for attachment in request.media:
            if attachment.artifact_ref.media_type not in capability.supported_media_types:
                raise _invalid(
                    f"judge media type is unsupported: {attachment.artifact_ref.media_type}"
                )
        body = self._responses_body(
            request.prompt,
            request.media,
            request.response_contract,
        )
        envelope = await self._transport.request("POST", "responses", json_body=body)
        payload = _json_object(envelope)
        return JudgeResult(
            text=_response_text(payload),
            metrics=self._metrics(envelope, payload),
        )

    def _responses_body(
        self,
        prompt: str,
        attachments: tuple[Attachment, ...],
        contract: ResponseContract | None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for attachment in attachments:
            media_type = attachment.artifact_ref.media_type
            if not media_type.startswith("image/"):
                raise _invalid(f"Responses attachment is not an image: {media_type}")
            content.append(
                {
                    "type": "input_image",
                    "image_url": self._data_url(attachment),
                }
            )
        body: dict[str, Any] = {
            "model": self._spec.model,
            "input": [{"role": "user", "content": content}],
            "store": False,
            "truncation": "disabled",
        }
        if contract is not None:
            mode = self._options.structured_output_mode
            if mode == StructuredOutputMode.JSON_SCHEMA:
                body["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": contract.value,
                        "strict": True,
                        "schema": schema_for_contract(contract),
                    }
                }
            elif mode == StructuredOutputMode.JSON_OBJECT:
                body["text"] = {"format": {"type": "json_object"}}
            elif mode == StructuredOutputMode.TEXT:
                body["text"] = {"format": {"type": "text"}}
        return body

    def _data_url(self, attachment: Attachment) -> str:
        data = self._store.verify(attachment.artifact_ref).read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{attachment.artifact_ref.media_type};base64,{encoded}"

    def _metrics(
        self,
        envelope: TransportResponse,
        payload: dict[str, Any],
    ) -> ProviderCallMetrics:
        request_id = _optional_string(payload.get("id"))
        namespace = request_id or canonical_hash(payload)
        return ProviderCallMetrics(
            call_ref=stable_key("provider_call", self._profile.fingerprint, namespace),
            provider_request_id=request_id,
            elapsed_seconds=envelope.elapsed_seconds,
            transport_retries=envelope.retries,
        )


class OpenAIImageAdapter:
    def __init__(
        self,
        spec: ProviderSpec,
        credential: SecretValue,
        store: ArtifactStore,
        *,
        max_retries: int,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._spec = spec
        self._store = store
        self._options = _validated_options(OpenAIImageOptions, spec)
        endpoint = spec.base_url or DEFAULT_BASE_URL
        self._transport = HttpTransport(
            base_url=endpoint,
            credential=credential,
            timeout_seconds=spec.timeout_seconds,
            max_retries=max_retries,
            client=client,
        )
        self._profile = ProviderCapabilityProfile(
            identity=ProviderIdentity(
                adapter=spec.adapter,
                model=spec.model,
                endpoint=endpoint,
                adapter_version=OPENAI_ADAPTER_VERSION,
            ),
            behavior_fingerprint=canonical_hash(self._options),
            image=ImageCapabilities(
                request_protocol="openai_images",
                input_modes=("multipart",),
                output_modes=("base64", "url"),
                max_input_images=self._options.max_input_images,
                supported_aspect_ratios=self._options.supported_aspect_ratios,
                supported_resolutions=self._options.supported_resolutions,
                prompt_enhancement_behavior="provider_default",
                submit_is_idempotent=False,
                max_prompt_characters=self._options.max_prompt_characters,
            ),
        )

    def __repr__(self) -> str:
        return (
            f"OpenAIImageAdapter(model={self._spec.model!r}, "
            f"endpoint={self._profile.identity.endpoint!r}, credential=***)"
        )

    @property
    def capability_profile(self) -> ProviderCapabilityProfile:
        return self._profile

    async def preflight(self) -> None:
        await _preflight_model(
            self._transport,
            self._spec.model,
            self._options.preflight_mode,
        )

    async def aclose(self) -> None:
        await self._transport.close()

    async def generate_image(self, request: ImageRequest) -> ImageResult:
        capability = self._profile.image
        if capability is None:
            raise AssertionError("image capability is missing")
        if len(request.prompt) > capability.max_prompt_characters:
            raise _invalid("image prompt exceeds declared provider capability")
        if len(request.input_images) > capability.max_input_images:
            raise _invalid("image request has too many input images")
        if request.aspect_ratio not in capability.supported_aspect_ratios:
            raise _invalid(f"unsupported image aspect ratio: {request.aspect_ratio}")
        if not capability.supports_resolution(request.resolution):
            raise _invalid("unsupported image resolution")

        size = f"{request.resolution.width}x{request.resolution.height}"
        if request.input_images:
            files = [
                (
                    "image",
                    (
                        attachment.name,
                        self._store.verify(attachment.artifact_ref).read_bytes(),
                        attachment.artifact_ref.media_type,
                    ),
                )
                for attachment in request.input_images
            ]
            envelope = await self._transport.request(
                "POST",
                "images/edits",
                data={
                    "model": self._spec.model,
                    "prompt": request.prompt,
                    "size": size,
                    "quality": self._options.quality,
                    "output_format": self._options.output_format,
                },
                files=files,
                allow_retries=False,
            )
        else:
            envelope = await self._transport.request(
                "POST",
                "images/generations",
                json_body={
                    "model": self._spec.model,
                    "prompt": request.prompt,
                    "n": 1,
                    "size": size,
                    "quality": self._options.quality,
                    "output_format": self._options.output_format,
                },
                allow_retries=False,
            )
        payload = _json_object(envelope)
        data, media_type, download = await self._image_payload(payload)
        return ImageResult(
            payload=MediaPayload(data=data, media_type=media_type),
            metrics=_media_metrics(
                self._profile,
                envelope,
                payload,
                download=download,
            ),
        )

    async def _image_payload(
        self,
        payload: dict[str, Any],
    ) -> tuple[bytes, str, TransportResponse | None]:
        try:
            first = payload["data"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise _contract_error("image response has no data item") from exc
        if not isinstance(first, dict):
            raise _contract_error("image response item is not an object")
        encoded = first.get("b64_json")
        media_type = _image_media_type(self._options.output_format)
        if isinstance(encoded, str):
            try:
                return base64.b64decode(encoded, validate=True), media_type, None
            except (ValueError, binascii.Error) as exc:
                raise _contract_error("image response contains invalid base64") from exc
        url = first.get("url")
        if isinstance(url, str) and url:
            downloaded = await self._transport.request(
                "GET",
                url,
                headers={"Accept": "image/*"},
            )
            content_type = downloaded.response.headers.get("content-type", media_type)
            return downloaded.response.content, content_type.split(";", 1)[0], downloaded
        raise _contract_error("image response contains neither base64 nor URL")


class OpenAIVideoAdapter:
    def __init__(
        self,
        spec: ProviderSpec,
        credential: SecretValue,
        store: ArtifactStore,
        *,
        max_retries: int,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._spec = spec
        self._store = store
        self._options = _validated_options(OpenAIVideoOptions, spec)
        endpoint = spec.base_url or DEFAULT_BASE_URL
        self._transport = HttpTransport(
            base_url=endpoint,
            credential=credential,
            timeout_seconds=spec.timeout_seconds,
            max_retries=max_retries,
            client=client,
        )
        self._profile = ProviderCapabilityProfile(
            identity=ProviderIdentity(
                adapter=spec.adapter,
                model=spec.model,
                endpoint=endpoint,
                adapter_version=OPENAI_ADAPTER_VERSION,
            ),
            behavior_fingerprint=canonical_hash(self._options),
            video=VideoCapabilities(
                request_protocol="openai_videos",
                input_modes=("multipart_start_image",),
                output_modes=("binary_download",),
                supports_start_image=True,
                max_reference_images=0,
                supported_durations=self._options.supported_durations,
                supported_resolutions=self._options.supported_resolutions,
                supported_fps=self._options.supported_fps,
                submit_is_idempotent=False,
                poll_interval_seconds=self._options.poll_interval_seconds,
                job_timeout_seconds=self._options.job_timeout_seconds,
                max_prompt_characters=self._options.max_prompt_characters,
            ),
        )

    def __repr__(self) -> str:
        return (
            f"OpenAIVideoAdapter(model={self._spec.model!r}, "
            f"endpoint={self._profile.identity.endpoint!r}, credential=***)"
        )

    @property
    def capability_profile(self) -> ProviderCapabilityProfile:
        return self._profile

    async def preflight(self) -> None:
        await _preflight_model(
            self._transport,
            self._spec.model,
            self._options.preflight_mode,
        )

    async def aclose(self) -> None:
        await self._transport.close()

    async def submit_video(self, request: VideoRequest) -> VideoJob:
        capability = self._profile.video
        if capability is None:
            raise AssertionError("video capability is missing")
        _validate_video_request(request, capability)
        start_path = self._store.verify(request.start_image.artifact_ref)
        envelope = await self._transport.request(
            "POST",
            "videos",
            data={
                "model": self._spec.model,
                "prompt": request.prompt,
                "seconds": str(request.duration),
                "size": f"{request.resolution.width}x{request.resolution.height}",
            },
            files=[
                (
                    "input_reference",
                    (
                        request.start_image.name,
                        start_path.read_bytes(),
                        request.start_image.artifact_ref.media_type,
                    ),
                )
            ],
            allow_retries=False,
        )
        payload = _json_object(envelope)
        return self._job(payload, envelope)

    async def poll_video(self, job_id: str) -> VideoJob:
        component = quote(job_id, safe="")
        envelope = await self._transport.request("GET", f"videos/{component}")
        return self._job(_json_object(envelope), envelope, expected_job_id=job_id)

    async def download_video(self, job_id: str) -> VideoResult:
        component = quote(job_id, safe="")
        envelope = await self._transport.request(
            "GET",
            f"videos/{component}/content",
            headers={"Accept": "video/mp4"},
        )
        media_type = envelope.response.headers.get("content-type", "video/mp4").split(";", 1)[0]
        payload_identity = {
            "id": job_id,
            "sha256": str(sha256_bytes(envelope.response.content)),
        }
        return VideoResult(
            payload=MediaPayload(data=envelope.response.content, media_type=media_type),
            metrics=ProviderCallMetrics(
                call_ref=stable_key(
                    "provider_call",
                    self._profile.fingerprint,
                    canonical_hash(payload_identity),
                ),
                provider_request_id=job_id,
                elapsed_seconds=envelope.elapsed_seconds,
                transport_retries=envelope.retries,
            ),
        )

    def _job(
        self,
        payload: dict[str, Any],
        envelope: TransportResponse,
        *,
        expected_job_id: str | None = None,
    ) -> VideoJob:
        job_id = _required_string(payload.get("id"), "video job id")
        if expected_job_id is not None and job_id != expected_job_id:
            raise _contract_error("video poll returned another job id")
        raw_status = _required_string(payload.get("status"), "video job status")
        status = _video_status(raw_status)
        progress = payload.get("progress")
        if progress is not None and not isinstance(progress, int | float):
            raise _contract_error("video job progress is not numeric")
        error = payload.get("error")
        if isinstance(error, dict):
            error = error.get("message")
        return VideoJob(
            job_id=job_id,
            status=status,
            progress=float(progress) if progress is not None else None,
            error=(self._transport.redact_text(str(error))[:1_000] if error else None),
            metrics=_media_metrics(self._profile, envelope, payload),
        )


def _validated_options[OptionsT: FrozenModel](
    model: type[OptionsT],
    spec: ProviderSpec,
) -> OptionsT:
    try:
        return model.model_validate(spec.decoded_options())
    except ValueError as exc:
        raise ConfigurationError(f"invalid provider options: {validation_summary(exc)}") from None


async def _preflight_model(
    transport: HttpTransport,
    model: str,
    mode: Literal["model_get", "model_list"],
) -> None:
    if mode == "model_get":
        await transport.request("GET", f"models/{quote(model, safe='')}")
        return
    envelope = await transport.request("GET", "models")
    payload = _json_object(envelope)
    rows = payload.get("data")
    if not isinstance(rows, list) or not any(
        isinstance(item, dict) and item.get("id") == model for item in rows
    ):
        raise ConfigurationError("configured model is absent from provider model discovery")


def _json_object(envelope: TransportResponse) -> dict[str, Any]:
    try:
        payload = envelope.response.json()
    except json.JSONDecodeError as exc:
        raise _contract_error("provider returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise _contract_error("provider response root is not an object")
    return payload


def _response_text(payload: dict[str, Any]) -> str:
    output = payload.get("output")
    if not isinstance(output, list):
        raise _contract_error("Responses output is missing")
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "refusal":
                raise _contract_error("model refused the response contract")
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
    if not texts:
        raise _contract_error("Responses output has no output_text")
    return "".join(texts)


def _media_metrics(
    profile: ProviderCapabilityProfile,
    envelope: TransportResponse,
    payload: dict[str, Any],
    *,
    download: TransportResponse | None = None,
) -> ProviderCallMetrics:
    request_id = _optional_string(payload.get("id")) or envelope.response.headers.get(
        "x-request-id"
    )
    namespace = request_id or canonical_hash(payload)
    return ProviderCallMetrics(
        call_ref=stable_key("provider_call", profile.fingerprint, namespace),
        provider_request_id=request_id,
        elapsed_seconds=envelope.elapsed_seconds
        + (download.elapsed_seconds if download is not None else 0.0),
        transport_retries=envelope.retries + (download.retries if download is not None else 0),
    )


def _validate_video_request(
    request: VideoRequest,
    capability: VideoCapabilities,
) -> None:
    if len(request.prompt) > capability.max_prompt_characters:
        raise _invalid("video prompt exceeds declared provider capability")
    if request.reference_images:
        raise _invalid("OpenAI Videos has no core additional-reference image channel")
    if request.duration not in capability.supported_durations:
        raise _invalid("unsupported video duration")
    if request.resolution not in capability.supported_resolutions:
        raise _invalid("unsupported video resolution")
    if request.fps not in capability.supported_fps:
        raise _invalid("unsupported video FPS")
    if not request.start_image.artifact_ref.media_type.startswith("image/"):
        raise _invalid("video start reference must be an image")


def _video_status(value: str) -> VideoJobStatus:
    if value == "queued":
        return VideoJobStatus.QUEUED
    if value in {"in_progress", "processing"}:
        return VideoJobStatus.IN_PROGRESS
    if value == "completed":
        return VideoJobStatus.COMPLETED
    if value in {"failed", "cancelled", "expired"}:
        return VideoJobStatus.FAILED
    raise _contract_error(f"unknown video job status: {value}")


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise _contract_error(f"provider response lacks {label}")
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _image_media_type(output_format: str) -> str:
    return "image/jpeg" if output_format == "jpeg" else f"image/{output_format}"


def _invalid(message: str) -> ProviderError:
    return ProviderError(
        kind=ProviderErrorKind.INVALID_REQUEST,
        message=message,
        retryable=False,
    )


def _contract_error(message: str) -> ProviderError:
    return ProviderError(
        kind=ProviderErrorKind.RESPONSE_CONTRACT,
        message=message,
        retryable=False,
    )
