"""Canonical panorama evidence."""

from __future__ import annotations

from story_engine.domain.common import FrozenModel
from story_engine.errors import SpatialError
from story_engine.media.validate import ImageTechnicalValidator
from story_engine.storage import ArtifactRef, ArtifactStore


class PanoramaSource(FrozenModel):
    scene_key: str
    story_plan_hash: str
    reference_library_hash: str
    artifact_ref: ArtifactRef
    width: int
    height: int


def validate_panorama_source(
    store: ArtifactStore,
    artifact: ArtifactRef,
    *,
    scene_key: str,
    story_plan_hash: str,
    reference_library_hash: str,
) -> PanoramaSource:
    result = ImageTechnicalValidator().validate(store, artifact, require_panorama=True)
    if result.info is None or result.status.value != "valid":
        detail = "; ".join(
            f"{finding.code}={finding.detail}" for finding in result.findings if not finding.passed
        )
        raise SpatialError(f"invalid canonical panorama for {scene_key}: {detail}")
    return PanoramaSource(
        scene_key=scene_key,
        story_plan_hash=story_plan_hash,
        reference_library_hash=reference_library_hash,
        artifact_ref=artifact,
        width=result.info.width,
        height=result.info.height,
    )
