"""Compact delivery manifest and provenance contract."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field

from story_engine.domain.common import FrozenModel
from story_engine.domain.evaluation import SelectionOutcome, TechnicalFinding
from story_engine.domain.request import PixelSize
from story_engine.domain.spatial import SpatialResolutionStatus
from story_engine.ids import canonical_hash
from story_engine.spatial.camera import NovelStationRecipe, SourceStationRecipe
from story_engine.storage import ArtifactRef


class DeliveryStatus(StrEnum):
    DELIVERED = "delivered"
    DELIVERED_DEGRADED = "delivered_degraded"


class CriterionSummary(FrozenModel):
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    unknown: int = Field(ge=0)


class FinalMediaRecord(FrozenModel):
    artifact_ref: ArtifactRef
    duration_seconds: float = Field(gt=0)
    resolution: PixelSize
    fps: float = Field(gt=0)


type ManifestCameraRecipe = Annotated[
    SourceStationRecipe | NovelStationRecipe,
    Field(discriminator="kind"),
]


class ShotDeliveryRecord(FrozenModel):
    shot_key: str
    selected_frame_candidate: str
    selected_video_candidate: str
    selection_outcome: SelectionOutcome
    spatial_status: SpatialResolutionStatus
    spatial_resolution_hash: str
    spatial_failure_codes: tuple[str, ...] = ()
    dropped_spatial_content: tuple[str, ...] = ()
    fallback_camera_recipe: ManifestCameraRecipe | None = None
    evaluation_refs: tuple[str, ...]
    criterion_summary: CriterionSummary
    video_technical_findings: tuple[TechnicalFinding, ...] = ()
    boundary_conflicts: tuple[str, ...] = ()
    previous_end_frame_allowed: bool | None = None


class ProviderRecord(FrozenModel):
    role: str
    adapter: str
    model: str
    endpoint: str
    capability_fingerprint: str


class UsageRecord(FrozenModel):
    logical_attempts: int = Field(ge=0)
    provider_calls: int = Field(ge=0)
    transport_retries: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(ge=0)
    known_cost_usd: float | None = Field(default=None, ge=0)


class ProvenanceRecord(FrozenModel):
    record_key: str
    artifact_ref: ArtifactRef
    parents: tuple[str, ...] = ()


class DeliveryManifest(FrozenModel):
    manifest_version: str
    run_id: str
    delivery_status: DeliveryStatus
    story_plan_hash: str
    reference_library_hash: str
    render_plan_hash: str
    final_media: FinalMediaRecord
    shots: tuple[ShotDeliveryRecord, ...]
    requirement_summary: CriterionSummary
    providers: tuple[ProviderRecord, ...]
    usage: UsageRecord
    plan_revisions: tuple[str, ...] = ()
    provenance: tuple[ProvenanceRecord, ...] = ()
    created_at: str
    manifest_hash: str

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(exclude={"manifest_hash"}))

    def assert_hash(self) -> None:
        if self.content_hash() != self.manifest_hash:
            raise ValueError("DeliveryManifest hash mismatch")
