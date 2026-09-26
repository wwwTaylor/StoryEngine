"""Public project input contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from PIL import Image, UnidentifiedImageError
from pydantic import Field, StringConstraints, field_validator, model_validator

from story_engine.domain.common import FrozenModel
from story_engine.errors import ContractError
from story_engine.ids import canonical_hash
from story_engine.security import validation_summary

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
LongText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000)]
AssetAlias = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$",
    ),
]
AssetSha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
MAX_PROVIDED_IMAGE_BYTES = 50 * 1024 * 1024


class PixelSize(FrozenModel):
    width: int = Field(ge=64, le=16_384)
    height: int = Field(ge=64, le=16_384)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


class DeliveryRequirements(FrozenModel):
    allowed_video_duration_seconds: tuple[int, ...] = (4, 6, 8)
    video_resolution: PixelSize | None = None
    video_fps: int = Field(default=24, ge=1, le=240)
    audio: bool = False

    @model_validator(mode="after")
    def validate_durations(self) -> DeliveryRequirements:
        values = self.allowed_video_duration_seconds
        if not values:
            raise ValueError("allowed_video_duration_seconds cannot be empty")
        if any(value <= 0 or value > 120 for value in values):
            raise ValueError("video durations must be between 1 and 120 seconds")
        if tuple(sorted(set(values))) != values:
            raise ValueError("video durations must be unique and sorted")
        return self


class GenerationRequirements(FrozenModel):
    output_language: ShortText = "English"
    required_entities: tuple[ShortText, ...] = ()
    forbidden_content: tuple[ShortText, ...] = ()
    asset_alignment_mode: Literal["auto", "manual"] = "auto"
    provided_asset_policy: Literal["optional", "require_all"] = "optional"
    delivery_requirements: DeliveryRequirements = DeliveryRequirements()

    @model_validator(mode="after")
    def validate_unique_lists(self) -> GenerationRequirements:
        if len(self.required_entities) != len(set(self.required_entities)):
            raise ValueError("required_entities must be unique")
        if len(self.forbidden_content) != len(set(self.forbidden_content)):
            raise ValueError("forbidden_content must be unique")
        return self


class ProvidedAsset(FrozenModel):
    asset_id: ShortText
    kind: Literal["character", "prop", "scene", "panorama", "image", "video"]
    name: ShortText | None = None
    description: ShortText
    content_sha256: AssetSha256 | None = None
    path: Path | None = None
    uri: str | None = None

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("provided asset URI must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "provided asset URI cannot contain credentials, query, or fragment; "
                "download signed assets locally and use path"
            )
        return value

    @model_validator(mode="after")
    def exactly_one_source(self) -> ProvidedAsset:
        if (self.path is None) == (self.uri is None):
            raise ValueError("provided asset requires exactly one of path or uri")
        return self


class ProvidedAssetBinding(FrozenModel):
    asset_id: ShortText
    kind: Literal["character", "scene"]
    canonical_alias: AssetAlias
    semantic_role: ShortText
    idea_mentions: tuple[ShortText, ...] = ()

    @model_validator(mode="after")
    def validate_mentions(self) -> ProvidedAssetBinding:
        if len(self.idea_mentions) != len(set(self.idea_mentions)):
            raise ValueError("provided asset binding idea_mentions must be unique")
        return self


class ProjectRequest(FrozenModel):
    task_id: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=128,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
        ),
    ]
    idea: LongText
    shot_target: int = Field(ge=1, le=64)
    visual_style: LongText
    resolution: PixelSize
    generation_requirements: GenerationRequirements = GenerationRequirements()
    provided_assets: tuple[ProvidedAsset, ...] = ()
    provided_asset_bindings: tuple[ProvidedAssetBinding, ...] = ()
    format_version: int = Field(default=1, ge=1, le=1)

    @model_validator(mode="after")
    def validate_delivery_resolution(self) -> ProjectRequest:
        delivery = self.generation_requirements.delivery_requirements
        if delivery.video_resolution is not None and delivery.video_resolution != self.resolution:
            raise ValueError("resolution must match delivery_requirements.video_resolution")
        asset_ids = [asset.asset_id for asset in self.provided_assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("provided asset_id values must be unique")
        bindings_by_asset = {binding.asset_id: binding for binding in self.provided_asset_bindings}
        if len(bindings_by_asset) != len(self.provided_asset_bindings):
            raise ValueError("provided asset bindings must have unique asset_id values")
        aliases = [binding.canonical_alias for binding in self.provided_asset_bindings]
        if len(aliases) != len(set(aliases)):
            raise ValueError("provided asset bindings must have unique canonical_alias values")
        assets_by_id = {asset.asset_id: asset for asset in self.provided_assets}
        for binding in self.provided_asset_bindings:
            asset = assets_by_id.get(binding.asset_id)
            if asset is None:
                raise ValueError(f"provided asset binding references unknown asset: {binding.asset_id}")
            allowed_asset_kinds = (
                {"character", "image"}
                if binding.kind == "character"
                else {"scene", "panorama", "image"}
            )
            if asset.kind not in allowed_asset_kinds:
                raise ValueError(
                    f"provided asset binding kind does not match asset: {binding.asset_id}"
                )
        if self.generation_requirements.provided_asset_policy == "require_all":
            if set(bindings_by_asset) != set(assets_by_id):
                missing = sorted(set(assets_by_id) - set(bindings_by_asset))
                extra = sorted(set(bindings_by_asset) - set(assets_by_id))
                detail = ", ".join((*missing, *extra)) or "unknown"
                raise ValueError(f"require_all provided assets are not fully bound: {detail}")
        return self

    @property
    def request_hash(self) -> str:
        return canonical_hash(self)


def project_request_from_mapping(raw: Any, *, source: Path) -> ProjectRequest:
    """Validate one request mapping and resolve its local asset paths."""

    if not isinstance(raw, dict):
        raise ContractError("project request root must be a mapping")
    try:
        request = ProjectRequest.model_validate(raw)
    except ValueError as exc:
        raise ContractError(f"invalid project request: {validation_summary(exc)}") from None
    base = source.resolve().parent
    assets: list[ProvidedAsset] = []
    for asset in request.provided_assets:
        if asset.path is None:
            assets.append(asset)
            continue
        asset_source = asset.path if asset.path.is_absolute() else base / asset.path
        asset_source = asset_source.resolve()
        if not asset_source.is_file():
            raise ContractError(f"provided asset path is not a readable file: {asset.asset_id}")
        try:
            size_bytes = asset_source.stat().st_size
            with asset_source.open("rb") as file:
                digest = hashlib.sha256()
                while chunk := file.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as exc:
            raise ContractError(f"cannot inspect provided asset: {asset.asset_id}") from exc
        if asset.kind in {"character", "scene", "panorama", "image"}:
            if size_bytes > MAX_PROVIDED_IMAGE_BYTES:
                raise ContractError(
                    f"provided image exceeds 50 MiB: {asset.asset_id}"
                )
            try:
                with Image.open(asset_source) as image:
                    image_format = (image.format or "").lower()
                    image_mode = image.mode
                    width, height = image.size
                    image.verify()
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                raise ContractError(
                    f"provided image cannot be decoded: {asset.asset_id}"
                ) from exc
            if image_format not in {"png", "jpeg", "webp"}:
                raise ContractError(
                    f"provided image format must be PNG, JPEG, or WebP: {asset.asset_id}"
                )
            if image_mode not in {"RGB", "RGBA", "L", "LA"}:
                raise ContractError(
                    f"provided image color mode is unsupported: {asset.asset_id}"
                )
            if width > 16_384 or height > 16_384:
                raise ContractError(
                    f"provided image dimensions exceed 16384 pixels: {asset.asset_id}"
                )
        actual_sha256 = digest.hexdigest()
        if asset.content_sha256 is not None and asset.content_sha256 != actual_sha256:
            raise ContractError(f"provided asset SHA-256 mismatch: {asset.asset_id}")
        assets.append(
            asset.model_copy(
                update={"path": asset_source, "content_sha256": actual_sha256}
            )
        )
    return request.model_copy(update={"provided_assets": tuple(assets)})
