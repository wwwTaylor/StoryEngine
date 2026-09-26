"""FFmpeg assembly with no synthetic fallback path."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from story_engine.domain.common import FrozenModel
from story_engine.domain.request import PixelSize
from story_engine.errors import AssemblyError
from story_engine.media.validate import VideoTechnicalInfo, VideoTechnicalValidator
from story_engine.storage import ArtifactRef, ArtifactStore


class AssemblyResult(FrozenModel):
    artifact_ref: ArtifactRef
    technical_info: VideoTechnicalInfo


class VideoAssembler:
    def __init__(
        self,
        *,
        ffmpeg_path: str = "ffmpeg",
        validator: VideoTechnicalValidator | None = None,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.validator = validator or VideoTechnicalValidator()

    def assemble(
        self,
        store: ArtifactStore,
        shots: tuple[ArtifactRef, ...],
        *,
        resolution: PixelSize,
        fps: int,
        expected_duration: float,
    ) -> AssemblyResult:
        if not shots:
            raise AssemblyError("assembly requires at least one selected shot")
        sources = tuple(store.verify(reference) for reference in shots)
        descriptor, temporary_name = tempfile.mkstemp(dir=store.root, suffix=".mp4")
        os.close(descriptor)
        temporary = Path(temporary_name)
        command = [self.ffmpeg_path, "-v", "error"]
        for source in sources:
            command.extend(("-i", str(source)))
        inputs = "".join(f"[{index}:v]" for index in range(len(sources)))
        command.extend(
            (
                "-filter_complex",
                f"{inputs}concat=n={len(sources)}:v=1:a=0[v]",
                "-map",
                "[v]",
                "-an",
                "-r",
                str(fps),
                "-s",
                f"{resolution.width}x{resolution.height}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-y",
                str(temporary),
            )
        )
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(120.0, expected_duration * 20.0),
            )
            if result.returncode != 0:
                raise AssemblyError(f"ffmpeg assembly failed: {result.stderr.strip()[:1_000]}")
            artifact = store.put_file(temporary, "video/mp4")
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AssemblyError(f"ffmpeg assembly failed: {exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)
        validation = self.validator.validate(
            store,
            artifact,
            expected_duration=expected_duration,
            expected_resolution=(resolution.width, resolution.height),
            expected_fps=fps,
        )
        if validation.info is None or validation.status.value != "valid":
            findings = "; ".join(
                f"{item.code}={item.detail}" for item in validation.findings if not item.passed
            )
            raise AssemblyError(f"assembled media failed validation: {findings}")
        return AssemblyResult(artifact_ref=artifact, technical_info=validation.info)
