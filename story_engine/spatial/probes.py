"""Deterministic equirectangular-to-perspective projection."""

from __future__ import annotations

import math
from enum import StrEnum
from io import BytesIO

import numpy as np
from PIL import Image
from pydantic import Field, model_validator

from story_engine.domain.common import FrozenModel
from story_engine.errors import SpatialError
from story_engine.ids import canonical_hash
from story_engine.spatial.panorama import PanoramaSource
from story_engine.storage import ArtifactRef, ArtifactStore
from story_engine.version import PANORAMA_PROJECTOR_VERSION


class ProbeRole(StrEnum):
    FRONT = "front"
    RIGHT = "right"
    BACK = "back"
    LEFT = "left"
    UP = "up"
    DOWN = "down"


PROBE_ORDER = (
    ProbeRole.FRONT,
    ProbeRole.RIGHT,
    ProbeRole.BACK,
    ProbeRole.LEFT,
    ProbeRole.UP,
    ProbeRole.DOWN,
)


def _basis(
    forward: tuple[float, float, float],
    camera_up: tuple[float, float, float],
) -> tuple[float, ...]:
    forward_array = np.asarray(forward, dtype=np.float64)
    up_hint = np.asarray(camera_up, dtype=np.float64)
    right = np.cross(up_hint, forward_array)
    right /= np.linalg.norm(right)
    up = np.cross(forward_array, right)
    up /= np.linalg.norm(up)
    # Columns map camera right/up/forward axes into world XYZ.
    return tuple(float(item) for item in np.column_stack((right, up, forward_array)).ravel())


ORIENTATION_BY_ROLE: dict[ProbeRole, tuple[float, ...]] = {
    ProbeRole.FRONT: _basis((0, 0, 1), (0, 1, 0)),
    ProbeRole.RIGHT: _basis((1, 0, 0), (0, 1, 0)),
    ProbeRole.BACK: _basis((0, 0, -1), (0, 1, 0)),
    ProbeRole.LEFT: _basis((-1, 0, 0), (0, 1, 0)),
    ProbeRole.UP: _basis((0, 1, 0), (0, 0, -1)),
    ProbeRole.DOWN: _basis((0, -1, 0), (0, 0, 1)),
}


class ProbeView(FrozenModel):
    role: ProbeRole
    source_panorama_hash: str
    orientation_matrix: tuple[float, ...]
    hfov_degrees: float = Field(gt=0, lt=180)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    artifact_ref: ArtifactRef
    projector_version: str

    @model_validator(mode="after")
    def validate_matrix(self) -> ProbeView:
        if len(self.orientation_matrix) != 9:
            raise ValueError("orientation matrix must contain nine values")
        return self


class ProbeSet(FrozenModel):
    scene_key: str
    source_panorama_hash: str
    probes: tuple[ProbeView, ...]
    projector_version: str
    probe_set_hash: str

    @model_validator(mode="after")
    def validate_contract(self) -> ProbeSet:
        if tuple(probe.role for probe in self.probes) != PROBE_ORDER:
            raise ValueError("probe roles must use the fixed six-direction order")
        if any(probe.source_panorama_hash != self.source_panorama_hash for probe in self.probes):
            raise ValueError("all probes must share one source panorama")
        if any(probe.projector_version != self.projector_version for probe in self.probes):
            raise ValueError("all probes must share one projector version")
        return self

    def by_role(self, role: ProbeRole) -> ProbeView:
        for probe in self.probes:
            if probe.role == role:
                return probe
        raise KeyError(role)


class PanoramaProjector:
    """Nearest-neighbor projection with explicit pixel-center semantics."""

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    def generate_six_probes(
        self,
        panorama: PanoramaSource,
        *,
        size: int = 512,
        hfov_degrees: float = 90.0,
    ) -> ProbeSet:
        if size <= 0:
            raise SpatialError("probe size must be positive")
        source = self._load_panorama(panorama)
        probes: list[ProbeView] = []
        for role in PROBE_ORDER:
            orientation = ORIENTATION_BY_ROLE[role]
            projected = self._project(
                source,
                orientation=orientation,
                width=size,
                height=size,
                hfov_degrees=hfov_degrees,
            )
            artifact = self.store.put_bytes(self._encode_png(projected), "image/png")
            probes.append(
                ProbeView(
                    role=role,
                    source_panorama_hash=panorama.artifact_ref.sha256,
                    orientation_matrix=orientation,
                    hfov_degrees=hfov_degrees,
                    width=size,
                    height=size,
                    artifact_ref=artifact,
                    projector_version=PANORAMA_PROJECTOR_VERSION,
                )
            )
        probe_tuple = tuple(probes)
        payload = {
            "scene_key": panorama.scene_key,
            "source_panorama_hash": panorama.artifact_ref.sha256,
            "probes": probe_tuple,
            "projector_version": PANORAMA_PROJECTOR_VERSION,
        }
        return ProbeSet(
            scene_key=panorama.scene_key,
            source_panorama_hash=panorama.artifact_ref.sha256,
            probes=probe_tuple,
            projector_version=PANORAMA_PROJECTOR_VERSION,
            probe_set_hash=canonical_hash(payload),
        )

    def project_view(
        self,
        panorama: PanoramaSource,
        *,
        yaw_degrees: float,
        pitch_degrees: float,
        hfov_degrees: float,
        width: int,
        height: int,
    ) -> ArtifactRef:
        if not 1 <= hfov_degrees < 180:
            raise SpatialError("view HFOV must be between 1 and 180 degrees")
        orientation = orientation_from_yaw_pitch(yaw_degrees, pitch_degrees)
        projected = self._project(
            self._load_panorama(panorama),
            orientation=orientation,
            width=width,
            height=height,
            hfov_degrees=hfov_degrees,
        )
        return self.store.put_bytes(self._encode_png(projected), "image/png")

    def _load_panorama(self, panorama: PanoramaSource) -> np.ndarray:
        path = self.store.verify(panorama.artifact_ref)
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if array.shape[1] != 2 * array.shape[0]:
            raise SpatialError("panorama source is not exactly 2:1")
        return array

    @staticmethod
    def _project(
        source: np.ndarray,
        *,
        orientation: tuple[float, ...],
        width: int,
        height: int,
        hfov_degrees: float,
    ) -> np.ndarray:
        if width <= 0 or height <= 0:
            raise SpatialError("projection dimensions must be positive")
        matrix = np.asarray(orientation, dtype=np.float64).reshape(3, 3)
        tangent_x = math.tan(math.radians(hfov_degrees) / 2)
        vertical_fov = 2 * math.atan(tangent_x * height / width)
        tangent_y = math.tan(vertical_fov / 2)
        x = ((np.arange(width, dtype=np.float64) + 0.5) / width * 2 - 1) * tangent_x
        y = (1 - (np.arange(height, dtype=np.float64) + 0.5) / height * 2) * tangent_y
        xx, yy = np.meshgrid(x, y)
        local = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
        local /= np.linalg.norm(local, axis=-1, keepdims=True)
        world = local @ matrix.T

        longitude = np.arctan2(world[..., 0], world[..., 2])
        latitude = np.arcsin(np.clip(world[..., 1], -1.0, 1.0))
        source_height, source_width = source.shape[:2]
        source_x = (
            np.floor((longitude / (2 * math.pi) + 0.5) * source_width).astype(np.int64)
            % source_width
        )
        source_y = np.clip(
            np.floor((0.5 - latitude / math.pi) * source_height).astype(np.int64),
            0,
            source_height - 1,
        )
        result: np.ndarray = source[source_y, source_x]
        return result

    @staticmethod
    def _encode_png(array: np.ndarray) -> bytes:
        output = BytesIO()
        Image.fromarray(array, mode="RGB").save(
            output,
            format="PNG",
            optimize=False,
            compress_level=9,
        )
        return output.getvalue()


def orientation_from_yaw_pitch(yaw_degrees: float, pitch_degrees: float) -> tuple[float, ...]:
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    forward = (
        math.sin(yaw) * math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw) * math.cos(pitch),
    )
    if abs(pitch_degrees) >= 89.999:
        up = (0.0, 0.0, -1.0 if pitch_degrees > 0 else 1.0)
    else:
        up = (0.0, 1.0, 0.0)
    return _basis(forward, up)
