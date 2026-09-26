"""Deterministic technical media validation."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps, UnidentifiedImageError
from pydantic import Field

from story_engine.domain.common import FrozenModel
from story_engine.domain.evaluation import TechnicalFinding, TechnicalStatus
from story_engine.errors import ArtifactError, MediaValidationError
from story_engine.storage import ArtifactRef, ArtifactStore


class ImageTechnicalInfo(FrozenModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: str
    mode: str


class ImageValidationResult(FrozenModel):
    status: TechnicalStatus
    findings: tuple[TechnicalFinding, ...]
    info: ImageTechnicalInfo | None = None


class ImageTechnicalValidator:
    def validate(
        self,
        store: ArtifactStore,
        artifact: ArtifactRef,
        *,
        expected_size: tuple[int, int] | None = None,
        require_panorama: bool = False,
    ) -> ImageValidationResult:
        findings: list[TechnicalFinding] = []
        try:
            path = store.verify(artifact)
            info = self._inspect(path)
            findings.append(TechnicalFinding(code="decodable", passed=True, detail="ok"))
        except (ArtifactError, MediaValidationError) as exc:
            # ArtifactError and Pillow failures are normalized into a technical
            # result. The exception string contains no provider credential.
            return ImageValidationResult(
                status=TechnicalStatus.INVALID,
                findings=(TechnicalFinding(code="decodable", passed=False, detail=str(exc)[:500]),),
            )
        format_supported = info.format in {"png", "jpeg", "webp"}
        findings.append(
            TechnicalFinding(
                code="format",
                passed=format_supported,
                detail=info.format,
            )
        )
        mode_supported = info.mode in {"RGB", "RGBA", "L", "LA"}
        findings.append(
            TechnicalFinding(
                code="color_mode",
                passed=mode_supported,
                detail=info.mode,
            )
        )
        decoded_media_type = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
        }.get(info.format)
        findings.append(
            TechnicalFinding(
                code="media_type",
                passed=decoded_media_type == artifact.media_type,
                detail=f"declared={artifact.media_type}, decoded={decoded_media_type}",
            )
        )
        if expected_size is not None:
            passed = (info.width, info.height) == expected_size
            findings.append(
                TechnicalFinding(
                    code="dimensions",
                    passed=passed,
                    detail=f"{info.width}x{info.height}",
                )
            )
        if require_panorama:
            passed = info.width == 2 * info.height
            findings.append(
                TechnicalFinding(
                    code="panorama_2_to_1",
                    passed=passed,
                    detail=f"{info.width}:{info.height}",
                )
            )
        status = (
            TechnicalStatus.VALID
            if all(finding.passed for finding in findings)
            else TechnicalStatus.INVALID
        )
        return ImageValidationResult(status=status, findings=tuple(findings), info=info)

    @staticmethod
    def _inspect(path: Path) -> ImageTechnicalInfo:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                if image.format is None:
                    raise MediaValidationError("image format is unavailable")
                return ImageTechnicalInfo(
                    width=image.width,
                    height=image.height,
                    format=image.format.lower(),
                    mode=image.mode,
                )
        except (OSError, UnidentifiedImageError) as exc:
            raise MediaValidationError(f"image cannot be decoded: {exc}") from exc


class VideoTechnicalInfo(FrozenModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    fps: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    codec: str
    container: str


class VideoValidationResult(FrozenModel):
    status: TechnicalStatus
    findings: tuple[TechnicalFinding, ...]
    info: VideoTechnicalInfo | None = None
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)


class VideoTechnicalValidator:
    _PERCEPTUAL_SCALES = (1, 2, 4)
    _PERCEPTUAL_WEIGHTS = (1, 2, 5)

    def __init__(
        self,
        ffprobe_path: str = "ffprobe",
        ffmpeg_path: str = "ffmpeg",
    ) -> None:
        self.ffprobe_path = ffprobe_path
        self.ffmpeg_path = ffmpeg_path

    def validate(
        self,
        store: ArtifactStore,
        artifact: ArtifactRef,
        *,
        expected_duration: float | None = None,
        expected_resolution: tuple[int, int] | None = None,
        expected_fps: float | None = None,
        expected_start_image: ArtifactRef | None = None,
        min_start_frame_similarity: float = 0.975,
    ) -> VideoValidationResult:
        if not 0.0 <= min_start_frame_similarity <= 1.0:
            raise ValueError("minimum start-frame similarity must be between zero and one")
        try:
            path = store.verify(artifact)
            info = self._probe(path)
        except (ArtifactError, MediaValidationError) as exc:
            return VideoValidationResult(
                status=TechnicalStatus.INVALID,
                findings=(
                    TechnicalFinding(
                        code="decodable",
                        passed=False,
                        detail=str(exc)[:500],
                    ),
                ),
            )
        findings: list[TechnicalFinding] = [
            TechnicalFinding(code="decodable", passed=True, detail="ok"),
            TechnicalFinding(
                code="frames_nonzero",
                passed=info.frame_count > 0,
                detail=str(info.frame_count),
            ),
            TechnicalFinding(
                code="media_type",
                passed=artifact.media_type.startswith("video/"),
                detail=artifact.media_type,
            ),
        ]
        if expected_duration is not None:
            fps_for_tolerance = expected_fps or info.fps
            duration_tolerance = max(0.35, 2.0 / fps_for_tolerance)
            findings.append(
                TechnicalFinding(
                    code="duration",
                    passed=abs(info.duration_seconds - expected_duration) <= duration_tolerance,
                    detail=f"{info.duration_seconds:.3f}s",
                )
            )
        if expected_resolution is not None:
            findings.append(
                TechnicalFinding(
                    code="resolution",
                    passed=(info.width, info.height) == expected_resolution,
                    detail=f"{info.width}x{info.height}",
                )
            )
        if expected_fps is not None:
            findings.append(
                TechnicalFinding(
                    code="fps",
                    passed=abs(info.fps - expected_fps) <= 0.1,
                    detail=f"{info.fps:.3f}",
                )
            )
        start_frame_similarity: float | None = None
        if expected_start_image is not None:
            try:
                reference_path = store.verify(expected_start_image)
                start_frame_similarity = self._start_frame_similarity(path, reference_path)
                findings.append(
                    TechnicalFinding(
                        code="start_frame_perceptual_similarity",
                        passed=start_frame_similarity >= min_start_frame_similarity,
                        detail=(
                            f"{start_frame_similarity:.6f} "
                            f"(minimum {min_start_frame_similarity:.6f})"
                        ),
                    )
                )
            except (ArtifactError, MediaValidationError) as exc:
                findings.append(
                    TechnicalFinding(
                        code="start_frame_perceptual_similarity",
                        passed=False,
                        detail=str(exc)[:500],
                    )
                )
        valid = all(item.passed for item in findings)
        return VideoValidationResult(
            status=TechnicalStatus.VALID if valid else TechnicalStatus.INVALID,
            findings=tuple(findings),
            info=info,
            quality_score=(
                (start_frame_similarity if start_frame_similarity is not None else 1.0)
                if valid
                else 0.0
            ),
        )

    def _start_frame_similarity(
        self,
        video_path: Path,
        reference_path: Path,
    ) -> float:
        command = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-an",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MediaValidationError(
                f"ffmpeg start-frame comparison failed to execute: {exc}"
            ) from exc
        if result.returncode != 0:
            raise MediaValidationError(
                "ffmpeg could not extract the video start frame: "
                f"{result.stderr.decode(errors='replace').strip()[:500]}"
            )
        try:
            with Image.open(reference_path) as opened:
                reference = ImageOps.exif_transpose(opened).convert("RGB")
            with Image.open(BytesIO(result.stdout)) as opened:
                first_frame = ImageOps.exif_transpose(opened).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise MediaValidationError(
                f"start-frame comparison image cannot be decoded: {exc}"
            ) from exc
        return self._perceptual_similarity(reference, first_frame)

    @classmethod
    def _perceptual_similarity(cls, reference: Image.Image, candidate: Image.Image) -> float:
        """Compare visual structure while tolerating codec texture and tiny motion steps."""

        reference = reference.convert("RGB")
        candidate = candidate.convert("RGB")
        if reference.size != candidate.size:
            return 0.0
        weighted_error = Fraction(0, 1)
        for scale, weight in zip(
            cls._PERCEPTUAL_SCALES,
            cls._PERCEPTUAL_WEIGHTS,
            strict=True,
        ):
            size = (
                max(1, reference.width // scale),
                max(1, reference.height // scale),
            )
            radius = scale * 3 / 5
            reference_level = reference.filter(ImageFilter.GaussianBlur(radius=radius))
            candidate_level = candidate.filter(ImageFilter.GaussianBlur(radius=radius))
            if scale != 1:
                reference_level = reference_level.resize(size, Image.Resampling.LANCZOS)
                candidate_level = candidate_level.resize(size, Image.Resampling.LANCZOS)
            histogram = ImageChops.difference(reference_level, candidate_level).histogram()
            absolute_error = sum((bucket % 256) * count for bucket, count in enumerate(histogram))
            maximum_error = size[0] * size[1] * 3 * 255
            weighted_error += weight * Fraction(absolute_error, maximum_error)
        mean_error = weighted_error / sum(cls._PERCEPTUAL_WEIGHTS)
        return max(0.0, 1.0 - float(mean_error))

    def _probe(self, path: Path) -> VideoTechnicalInfo:
        command = [
            self.ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration,format_name:stream=index,codec_type,codec_name,"
            "width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MediaValidationError(f"ffprobe failed to execute: {exc}") from exc
        if result.returncode != 0:
            raise MediaValidationError(f"ffprobe rejected video: {result.stderr.strip()[:500]}")
        try:
            payload = json.loads(result.stdout)
            streams = payload["streams"]
            video = next(stream for stream in streams if stream.get("codec_type") == "video")
            duration = float(video.get("duration") or payload["format"].get("duration"))
            fps = _parse_fraction(str(video["avg_frame_rate"]))
            raw_frames = video.get("nb_frames")
            frame_count = (
                int(raw_frames)
                if raw_frames not in {None, "N/A"}
                else max(1, round(duration * fps))
            )
            return VideoTechnicalInfo(
                width=int(video["width"]),
                height=int(video["height"]),
                duration_seconds=duration,
                fps=fps,
                frame_count=frame_count,
                codec=str(video.get("codec_name") or "unknown"),
                container=str(payload["format"].get("format_name") or "unknown"),
            )
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            raise MediaValidationError("ffprobe response lacks a valid video stream") from exc


def _parse_fraction(value: str) -> float:
    if "/" not in value:
        result = float(value)
    else:
        numerator, denominator = value.split("/", 1)
        result = float(numerator) / float(denominator)
    if result <= 0:
        raise ValueError("FPS must be positive")
    return result
