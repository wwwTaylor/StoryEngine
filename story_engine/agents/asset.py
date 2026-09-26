"""AssetAgent produces immutable media candidates and nothing else."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from story_engine.domain.evaluation import (
    CandidateRecord,
    ProviderUsage,
    TechnicalStatus,
)
from story_engine.errors import ProviderError, ProviderErrorKind
from story_engine.ids import stable_key
from story_engine.media.validate import ImageTechnicalValidator, VideoTechnicalValidator
from story_engine.providers.ports import (
    ImageProvider,
    ImageRequest,
    ProviderCallMetrics,
    VideoJobStatus,
    VideoProvider,
    VideoRequest,
)
from story_engine.storage import ArtifactStore
from story_engine.task_pool import TaskPool


class AssetAgent:
    def __init__(
        self,
        *,
        image_provider: ImageProvider,
        video_provider: VideoProvider,
        store: ArtifactStore,
        task_pool: TaskPool,
        image_validator: ImageTechnicalValidator | None = None,
        video_validator: VideoTechnicalValidator | None = None,
    ) -> None:
        self.image_provider = image_provider
        self.video_provider = video_provider
        self.store = store
        self.task_pool = task_pool
        self.image_validator = image_validator or ImageTechnicalValidator()
        self.video_validator = video_validator or VideoTechnicalValidator()

    async def produce_image(
        self,
        *,
        operation_key: str,
        request: ImageRequest,
        logical_attempt: int,
        candidate_index: int,
        require_panorama: bool = False,
    ) -> CandidateRecord:
        async with self.task_pool.slot():
            result = await self.image_provider.generate_image(request)
        artifact = self.store.put_bytes(result.payload.data, result.payload.media_type)
        validation = self.image_validator.validate(
            self.store,
            artifact,
            expected_size=(request.resolution.width, request.resolution.height),
            require_panorama=require_panorama,
        )
        candidate_key = stable_key(
            "candidate",
            operation_key,
            f"{logical_attempt}:{candidate_index}",
        )
        return CandidateRecord(
            candidate_key=candidate_key,
            operation_key=operation_key,
            artifact_ref=artifact,
            technical_status=validation.status,
            technical_findings=validation.findings,
            technical_quality=1.0 if validation.status == TechnicalStatus.VALID else 0.0,
            provider_call_ref=result.metrics.call_ref,
            provider_call_refs=(result.metrics.call_ref,),
            provider_usage=_usage(result.metrics),
            logical_attempt=logical_attempt,
        )

    async def produce_video(
        self,
        *,
        operation_key: str,
        request: VideoRequest,
        logical_attempt: int,
        candidate_index: int,
        on_job_progress: (Callable[[str, ProviderUsage, tuple[str, ...]], None] | None) = None,
        resume_job_id: str | None = None,
        prior_usage: ProviderUsage | None = None,
        prior_call_refs: tuple[str, ...] = (),
    ) -> CandidateRecord:
        call_refs = list(prior_call_refs)
        usage = prior_usage or ProviderUsage()
        if resume_job_id is None:
            async with self.task_pool.slot():
                job = await self.video_provider.submit_video(request)
        else:
            async with self.task_pool.slot():
                job = await self.video_provider.poll_video(resume_job_id)
        call_refs.append(job.metrics.call_ref)
        usage = ProviderUsage.combine((usage, _usage(job.metrics)))
        if on_job_progress is not None:
            on_job_progress(job.job_id, usage, tuple(call_refs))
        capability = self.video_provider.capability_profile.video
        if capability is None:
            raise ProviderError(
                kind=ProviderErrorKind.INVALID_REQUEST,
                message="VideoProvider exposes no video capability",
            )
        poll_started = time.monotonic()
        while job.status in {VideoJobStatus.QUEUED, VideoJobStatus.IN_PROGRESS}:
            if time.monotonic() - poll_started > capability.job_timeout_seconds:
                raise ProviderError(
                    kind=ProviderErrorKind.TIMEOUT,
                    message="video provider job exceeded the declared wait limit",
                    retryable=True,
                )
            await asyncio.sleep(capability.poll_interval_seconds)
            async with self.task_pool.slot():
                job = await self.video_provider.poll_video(job.job_id)
            call_refs.append(job.metrics.call_ref)
            usage = ProviderUsage.combine((usage, _usage(job.metrics)))
            if on_job_progress is not None:
                on_job_progress(job.job_id, usage, tuple(call_refs))
        if job.status == VideoJobStatus.FAILED:
            raise ProviderError(
                kind=ProviderErrorKind.JOB_FAILED,
                message=job.error or "video provider job failed",
                retryable=False,
            )
        async with self.task_pool.slot():
            result = await self.video_provider.download_video(job.job_id)
        call_refs.append(result.metrics.call_ref)
        usage = ProviderUsage.combine((usage, _usage(result.metrics)))
        artifact = self.store.put_bytes(result.payload.data, result.payload.media_type)
        validation = self.video_validator.validate(
            self.store,
            artifact,
            expected_duration=request.duration,
            expected_resolution=(request.resolution.width, request.resolution.height),
            expected_fps=request.fps,
            expected_start_image=request.start_image.artifact_ref,
        )
        candidate_key = stable_key(
            "candidate",
            operation_key,
            f"{logical_attempt}:{candidate_index}",
        )
        return CandidateRecord(
            candidate_key=candidate_key,
            operation_key=operation_key,
            artifact_ref=artifact,
            technical_status=validation.status,
            technical_findings=validation.findings,
            technical_quality=validation.quality_score,
            provider_call_ref=result.metrics.call_ref,
            provider_call_refs=tuple(call_refs),
            provider_usage=usage,
            logical_attempt=logical_attempt,
        )


def _usage(metrics: ProviderCallMetrics) -> ProviderUsage:
    return ProviderUsage(
        call_count=1,
        transport_retries=metrics.transport_retries,
        elapsed_seconds=metrics.elapsed_seconds,
        known_cost_usd=metrics.cost_usd,
    )
