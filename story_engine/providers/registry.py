"""Explicit adapter registry and provider composition."""

from __future__ import annotations

from dataclasses import dataclass

from story_engine.config import (
    AppConfig,
    ProviderSpec,
    SecretValue,
    credential_from_environment,
)
from story_engine.errors import ConfigurationError
from story_engine.providers.adapters.gemini import GeminiGenerateContentAdapter, GeminiOptions
from story_engine.providers.adapters.openai import (
    OpenAIImageAdapter,
    OpenAIImageOptions,
    OpenAIResponsesAdapter,
    OpenAIResponsesOptions,
    OpenAIVideoAdapter,
    OpenAIVideoOptions,
)
from story_engine.providers.ports import (
    ImageProvider,
    JudgeProvider,
    RuntimeCapabilities,
    TextProvider,
    VideoProvider,
)
from story_engine.storage import ArtifactStore
from story_engine.security import validation_summary


def validate_provider_config(config: AppConfig) -> None:
    """Validate every adapter's options before credentials or run files are used."""

    role_options = {
        "planner": {"openai_responses": OpenAIResponsesOptions, "gemini_native": GeminiOptions},
        "image": {"openai_images": OpenAIImageOptions},
        "video": {"openai_videos": OpenAIVideoOptions},
        "judge": {"openai_responses": OpenAIResponsesOptions, "gemini_native": GeminiOptions},
    }
    for role, adapters in role_options.items():
        spec = getattr(config.providers, role)
        model = adapters.get(spec.adapter)
        if model is None:
            raise ConfigurationError(f"unsupported {role} adapter")
        try:
            model.model_validate(spec.decoded_options())
        except ValueError as exc:
            raise ConfigurationError(
                f"invalid {role} provider options: {validation_summary(exc)}"
            ) from None


@dataclass(frozen=True, slots=True)
class ProviderSet:
    planner: TextProvider
    image: ImageProvider
    video: VideoProvider
    judge: JudgeProvider

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            planner=self.planner.capability_profile,
            image=self.image.capability_profile,
            video=self.video.capability_profile,
            judge=self.judge.capability_profile,
        )

    async def aclose(self) -> None:
        closed: set[int] = set()
        for provider in (self.planner, self.image, self.video, self.judge):
            identity = id(provider)
            if identity in closed:
                continue
            closed.add(identity)
            await provider.aclose()


def build_provider_set(
    config: AppConfig,
    store: ArtifactStore,
    *,
    credential_override: SecretValue | None = None,
) -> ProviderSet:
    validate_provider_config(config)
    retries = config.generation.provider_retries
    return ProviderSet(
        planner=_planner(
            config.providers.planner,
            store,
            retries,
            credential_override,
        ),
        image=_image(
            config.providers.image,
            store,
            retries,
            credential_override,
        ),
        video=_video(
            config.providers.video,
            store,
            retries,
            credential_override,
        ),
        judge=_judge(
            config.providers.judge,
            store,
            retries,
            credential_override,
        ),
    )


def _planner(
    spec: ProviderSpec,
    store: ArtifactStore,
    retries: int,
    override: SecretValue | None,
) -> TextProvider:
    if spec.adapter == "openai_responses":
        return OpenAIResponsesAdapter(
            spec,
            _credential(spec, override),
            store,
            max_retries=retries,
        )
    if spec.adapter == "gemini_native":
        return GeminiGenerateContentAdapter(
            spec,
            _credential(spec, override),
            store,
            max_retries=retries,
        )
    raise ConfigurationError(f"unknown planner adapter: {spec.adapter}")


def _image(
    spec: ProviderSpec,
    store: ArtifactStore,
    retries: int,
    override: SecretValue | None,
) -> ImageProvider:
    if spec.adapter == "openai_images":
        return OpenAIImageAdapter(
            spec,
            _credential(spec, override),
            store,
            max_retries=retries,
        )
    raise ConfigurationError(f"unknown image adapter: {spec.adapter}")


def _video(
    spec: ProviderSpec,
    store: ArtifactStore,
    retries: int,
    override: SecretValue | None,
) -> VideoProvider:
    if spec.adapter == "openai_videos":
        return OpenAIVideoAdapter(
            spec,
            _credential(spec, override),
            store,
            max_retries=retries,
        )
    raise ConfigurationError(f"unknown video adapter: {spec.adapter}")


def _judge(
    spec: ProviderSpec,
    store: ArtifactStore,
    retries: int,
    override: SecretValue | None,
) -> JudgeProvider:
    if spec.adapter == "openai_responses":
        return OpenAIResponsesAdapter(
            spec,
            _credential(spec, override),
            store,
            max_retries=retries,
        )
    if spec.adapter == "gemini_native":
        return GeminiGenerateContentAdapter(
            spec,
            _credential(spec, override),
            store,
            max_retries=retries,
        )
    raise ConfigurationError(f"unknown judge adapter: {spec.adapter}")


def _credential(spec: ProviderSpec, override: SecretValue | None) -> SecretValue:
    credential = override or credential_from_environment(spec)
    if credential is None:
        raise ConfigurationError(f"adapter {spec.adapter} requires a credential")
    return credential
