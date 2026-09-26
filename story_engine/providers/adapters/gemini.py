"""Gemini generateContent adapter with inline image and short-video input."""

from __future__ import annotations

import base64
import copy
import json
from typing import Any, Literal
from urllib.parse import quote, urlsplit, urlunsplit

import httpx
from pydantic import Field, model_validator

from story_engine.config import ProviderSpec, SecretValue
from story_engine.domain.common import FrozenModel
from story_engine.errors import ConfigurationError, ProviderError, ProviderErrorKind
from story_engine.ids import canonical_hash, stable_key
from story_engine.providers.ports import (
    Attachment,
    JudgeCapabilities,
    JudgeRequest,
    JudgeResult,
    ProviderCallMetrics,
    ProviderCapabilityProfile,
    ProviderIdentity,
    ResponseContract,
    StructuredOutputMode,
    TextCapabilities,
    TextRequest,
    TextResult,
)
from story_engine.providers.schemas import schema_for_contract
from story_engine.providers.transport import HttpTransport, TransportResponse
from story_engine.storage import ArtifactStore
from story_engine.security import validation_summary

GEMINI_ADAPTER_VERSION = "gemini-adapter-v3"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiOptions(FrozenModel):
    credential_mode: Literal["google_api_key", "bearer"] = "google_api_key"
    preflight_mode: Literal["model_get", "model_list"] = "model_get"
    structured_output_mode: StructuredOutputMode = StructuredOutputMode.JSON_SCHEMA
    max_prompt_characters: int = Field(default=100_000, gt=0)
    max_attachments: int = Field(default=16, ge=6)
    max_inline_request_bytes: int = Field(
        default=19 * 1024 * 1024,
        gt=0,
        le=100 * 1024 * 1024,
    )
    judge_max_media: int = Field(default=16, ge=1)
    judge_media_types: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/webp",
        "video/mp4",
        "video/webm",
        "video/quicktime",
    )

    @model_validator(mode="after")
    def validate_modes(self) -> GeminiOptions:
        if self.structured_output_mode not in {
            StructuredOutputMode.JSON_SCHEMA,
            StructuredOutputMode.JSON_OBJECT,
            StructuredOutputMode.TEXT,
        }:
            raise ValueError("unsupported Gemini structured output mode")
        if not self.judge_media_types:
            raise ValueError("Gemini judge media types cannot be empty")
        return self


class GeminiGenerateContentAdapter:
    """Implements TextProvider and JudgeProvider over one stable REST protocol."""

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
        self._options = _options(GeminiOptions, spec)
        endpoint = spec.base_url or DEFAULT_BASE_URL
        self._endpoint = endpoint
        if self._options.credential_mode == "bearer":
            credential_header = "Authorization"
            credential_prefix = "Bearer "
        else:
            credential_header = "x-goog-api-key"
            credential_prefix = ""
        self._transport = HttpTransport(
            base_url=endpoint,
            credential=credential,
            timeout_seconds=spec.timeout_seconds,
            max_retries=max_retries,
            client=client,
            credential_header=credential_header,
            credential_prefix=credential_prefix,
        )
        modes = (self._options.structured_output_mode,)
        self._profile = ProviderCapabilityProfile(
            identity=ProviderIdentity(
                adapter=spec.adapter,
                model=spec.model,
                endpoint=endpoint,
                adapter_version=GEMINI_ADAPTER_VERSION,
            ),
            behavior_fingerprint=canonical_hash(self._options),
            text=TextCapabilities(
                request_protocol="gemini_generate_content",
                attachment_modes=("inline_base64",),
                structured_output_modes=modes,
                max_prompt_characters=self._options.max_prompt_characters,
                max_attachments=self._options.max_attachments,
            ),
            judge=JudgeCapabilities(
                request_protocol="gemini_generate_content",
                attachment_modes=("inline_base64",),
                structured_output_modes=modes,
                max_media=self._options.judge_max_media,
                supported_media_types=self._options.judge_media_types,
                max_prompt_characters=self._options.max_prompt_characters,
            ),
        )

    def __repr__(self) -> str:
        return (
            f"GeminiGenerateContentAdapter(model={self._spec.model!r}, "
            f"endpoint={self._profile.identity.endpoint!r}, credential=***)"
        )

    @property
    def capability_profile(self) -> ProviderCapabilityProfile:
        return self._profile

    async def preflight(self) -> None:
        if self._options.preflight_mode == "model_get":
            await self._transport.request("GET", f"models/{self._model_component()}")
            return
        envelope = await self._transport.request("GET", self._gateway_model_list_url())
        payload = _json_object(envelope)
        rows = payload.get("data")
        if not isinstance(rows, list) or not any(
            isinstance(item, dict) and item.get("id") == self._spec.model for item in rows
        ):
            raise ConfigurationError("configured model is absent from provider model discovery")

    async def aclose(self) -> None:
        await self._transport.close()

    async def generate_text(self, request: TextRequest) -> TextResult:
        capability = self._profile.text
        if capability is None:
            raise AssertionError("Gemini text capability is missing")
        if len(request.prompt) > capability.max_prompt_characters:
            raise _invalid("text prompt exceeds declared Gemini capability")
        if len(request.attachments) > capability.max_attachments:
            raise _invalid("text request has too many Gemini attachments")
        envelope, payload = await self._generate(
            request.prompt,
            request.attachments,
            request.response_contract,
        )
        return TextResult(
            text=_response_text(payload),
            metrics=self._metrics(envelope, payload),
        )

    async def judge(self, request: JudgeRequest) -> JudgeResult:
        capability = self._profile.judge
        if capability is None:
            raise AssertionError("Gemini judge capability is missing")
        if len(request.prompt) > capability.max_prompt_characters:
            raise _invalid("judge prompt exceeds declared Gemini capability")
        if len(request.media) > capability.max_media:
            raise _invalid("judge request has too many Gemini media inputs")
        for attachment in request.media:
            if attachment.artifact_ref.media_type not in capability.supported_media_types:
                raise _invalid(
                    f"Gemini judge media type is unsupported: {attachment.artifact_ref.media_type}"
                )
        envelope, payload = await self._generate(
            request.prompt,
            request.media,
            request.response_contract,
        )
        return JudgeResult(
            text=_response_text(payload),
            metrics=self._metrics(envelope, payload),
        )

    async def _generate(
        self,
        prompt: str,
        attachments: tuple[Attachment, ...],
        contract: ResponseContract | None,
    ) -> tuple[TransportResponse, dict[str, Any]]:
        parts: list[dict[str, Any]] = [{"text": prompt}]
        for attachment in attachments:
            data = self._store.verify(attachment.artifact_ref).read_bytes()
            parts.append(
                {
                    "inline_data": {
                        "mime_type": attachment.artifact_ref.media_type,
                        "data": base64.b64encode(data).decode("ascii"),
                    }
                }
            )
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": self._generation_config(contract),
        }
        encoded_size = len(
            json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        if encoded_size > self._options.max_inline_request_bytes:
            raise _invalid(
                "Gemini inline request exceeds the declared limit; a Files API adapter is required"
            )
        envelope = await self._transport.request(
            "POST",
            f"models/{self._model_component()}:generateContent",
            json_body=body,
        )
        return envelope, _json_object(envelope)

    def _generation_config(
        self,
        contract: ResponseContract | None,
    ) -> dict[str, Any]:
        if contract is None or self._options.structured_output_mode == StructuredOutputMode.TEXT:
            return {}
        config: dict[str, Any] = {"responseMimeType": "application/json"}
        if self._options.structured_output_mode == StructuredOutputMode.JSON_SCHEMA:
            config["responseJsonSchema"] = _gemini_schema(schema_for_contract(contract))
        return config

    def _model_component(self) -> str:
        value = self._spec.model.removeprefix("models/")
        return quote(value, safe="")

    def _gateway_model_list_url(self) -> str:
        parsed = urlsplit(self._endpoint)
        return urlunsplit((parsed.scheme, parsed.netloc, "/v1/models", "", ""))

    def _metrics(
        self,
        envelope: TransportResponse,
        payload: dict[str, Any],
    ) -> ProviderCallMetrics:
        request_id = payload.get("responseId")
        if not isinstance(request_id, str) or not request_id:
            request_id = envelope.response.headers.get("x-request-id")
        namespace = request_id or canonical_hash(payload)
        return ProviderCallMetrics(
            call_ref=stable_key(
                "provider_call",
                self._profile.fingerprint,
                namespace,
            ),
            provider_request_id=request_id,
            elapsed_seconds=envelope.elapsed_seconds,
            transport_retries=envelope.retries,
        )


def _gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline local refs and remove presentation-only Pydantic keywords."""

    source = copy.deepcopy(schema)
    definitions = source.get("$defs", {})

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                name = reference.removeprefix("#/$defs/")
                target = definitions.get(name)
                if not isinstance(target, dict):
                    raise ConfigurationError(f"unresolved Gemini schema ref: {reference}")
                merged = {**target, **{key: item for key, item in value.items() if key != "$ref"}}
                return normalize(merged)
            return {
                key: normalize(item)
                for key, item in value.items()
                if key not in {"$defs", "title", "default"}
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    result = normalize(source)
    if not isinstance(result, dict):
        raise ConfigurationError("Gemini response schema root must be an object")
    return result


def _response_text(payload: dict[str, Any]) -> str:
    try:
        candidates = payload["candidates"]
        first = candidates[0]
        parts = first["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise _contract("Gemini response has no candidate content") from exc
    texts = [
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    if not texts:
        raise _contract("Gemini response has no text part")
    return "".join(texts)


def _json_object(envelope: TransportResponse) -> dict[str, Any]:
    try:
        payload = envelope.response.json()
    except json.JSONDecodeError as exc:
        raise _contract("Gemini returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise _contract("Gemini response root is not an object")
    return payload


def _options[OptionsT: FrozenModel](
    model: type[OptionsT],
    spec: ProviderSpec,
) -> OptionsT:
    try:
        return model.model_validate(spec.decoded_options())
    except ValueError as exc:
        raise ConfigurationError(f"invalid provider options: {validation_summary(exc)}") from None


def _invalid(message: str) -> ProviderError:
    return ProviderError(
        kind=ProviderErrorKind.INVALID_REQUEST,
        message=message,
        retryable=False,
    )


def _contract(message: str) -> ProviderError:
    return ProviderError(
        kind=ProviderErrorKind.RESPONSE_CONTRACT,
        message=message,
        retryable=False,
    )
