"""Extract the actual last decoded video frame as an optional workflow input."""

from __future__ import annotations

import subprocess
from pathlib import Path

from story_engine.domain.evaluation import TechnicalStatus
from story_engine.domain.first_frame import TailFrameInput, TailFrameStatus
from story_engine.errors import ArtifactError
from story_engine.media.validate import ImageTechnicalValidator
from story_engine.storage import ArtifactRef, ArtifactStore


class VideoTailExtractor:
    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

    def extract(
        self,
        store: ArtifactStore,
        video: ArtifactRef,
        *,
        expected_size: tuple[int, int],
    ) -> TailFrameInput:
        try:
            video_path = store.verify(video)
            frame_index = self._last_decoded_frame_index(video_path)
            completed = subprocess.run(
                [
                    self.ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(video_path),
                    "-map",
                    "0:v:0",
                    "-vf",
                    f"select=eq(n\\,{frame_index})",
                    "-frames:v",
                    "1",
                    "-an",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-",
                ],
                check=False,
                capture_output=True,
                timeout=120,
            )
        except (ArtifactError, OSError, subprocess.TimeoutExpired) as exc:
            return _unavailable(video, f"tail extraction could not execute: {exc}")
        if completed.returncode != 0 or not completed.stdout:
            detail = completed.stderr.decode(errors="replace").strip()[-400:]
            return _unavailable(video, detail or "ffmpeg produced no tail frame")
        try:
            frame = store.put_bytes(completed.stdout, "image/png")
            validation = ImageTechnicalValidator().validate(
                store,
                frame,
                expected_size=expected_size,
            )
        except ArtifactError as exc:
            return _unavailable(video, f"tail frame could not be stored: {exc}")
        if validation.status != TechnicalStatus.VALID:
            detail = "; ".join(
                f"{item.code}={item.detail}" for item in validation.findings if not item.passed
            )
            return _unavailable(video, detail or "tail frame failed technical validation")
        return TailFrameInput(
            status=TailFrameStatus.AVAILABLE,
            source_video=video,
            frame=frame,
        )

    def _last_decoded_frame_index(self, video_path: Path) -> int:
        completed = subprocess.run(
            [
                self.ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(video_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0:
            raise OSError(completed.stderr.strip()[-400:] or "ffprobe rejected video")
        try:
            frame_count = int(completed.stdout.strip())
        except ValueError as exc:
            raise OSError("ffprobe did not return a decoded frame count") from exc
        if frame_count < 1:
            raise OSError("video contains no decoded frames")
        return frame_count - 1


def _unavailable(video: ArtifactRef, reason: str) -> TailFrameInput:
    return TailFrameInput(
        status=TailFrameStatus.UNAVAILABLE,
        source_video=video,
        reason=" ".join(reason.split())[:500],
    )
