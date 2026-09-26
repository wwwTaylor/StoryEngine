"""Shared deterministic common-frustum mathematics for preflight and rendering."""

from __future__ import annotations

import math

import numpy as np
from pydantic import Field

from story_engine.domain.common import FrozenModel
from story_engine.errors import SpatialError
from story_engine.spatial.grounding import GroundingObservation
from story_engine.spatial.probes import orientation_from_yaw_pitch


class FovCornerDiagnostic(FrozenModel):
    target_key: str
    observation_key: str
    corner: str
    yaw_degrees: float
    pitch_degrees: float


class PairwiseAngleDiagnostic(FrozenModel):
    first_target_key: str
    second_target_key: str
    angle_degrees: float = Field(ge=0, le=180)


class CommonFovSolution(FrozenModel):
    yaw_degrees: float
    pitch_degrees: float
    minimum_hfov_degrees: float = Field(gt=0, lt=180)
    minimum_vfov_degrees: float = Field(gt=0, lt=180)
    hfov_degrees: float = Field(gt=0, lt=180)
    vfov_degrees: float = Field(gt=0, lt=180)
    feasible: bool
    limiting_corners: tuple[FovCornerDiagnostic, ...]
    pairwise_angles: tuple[PairwiseAngleDiagnostic, ...]


def observation_corner_rays(
    observation: GroundingObservation,
) -> tuple[tuple[str, np.ndarray], ...]:
    box = observation.normalized_box
    if box is None:
        raise SpatialError(f"observation {observation.observation_key} has no visible box")
    return tuple(
        (name, _ray(observation, x, y))
        for name, x, y in (
            ("top_left", box.x_min, box.y_min),
            ("top_right", box.x_max, box.y_min),
            ("bottom_right", box.x_max, box.y_max),
            ("bottom_left", box.x_min, box.y_max),
        )
    )


def solve_common_fov(
    observations: tuple[GroundingObservation, ...],
    *,
    output_width: int,
    output_height: int,
    framing_hfov_degrees: float,
    min_hfov_degrees: float,
    max_hfov_degrees: float,
    safe_margin_degrees: float,
) -> CommonFovSolution:
    if not observations:
        raise SpatialError("common FOV requires at least one visible observation")
    if output_width <= 0 or output_height <= 0:
        raise SpatialError("common FOV output dimensions must be positive")
    corners = tuple(
        (observation, name, ray)
        for observation in observations
        for name, ray in observation_corner_rays(observation)
    )
    rays = np.stack(tuple(item[2] for item in corners))
    aspect = output_width / output_height
    yaw, pitch, raw_hfov, raw_vfov = _optimize_axis(rays, aspect)
    minimum_hfov = max(
        raw_hfov + safe_margin_degrees,
        _horizontal_for_vertical(raw_vfov + safe_margin_degrees, aspect),
    )
    if minimum_hfov >= 179.999999:
        raise SpatialError("target corners plus the safety margin cannot share a perspective FOV")
    selected_hfov = max(min_hfov_degrees, framing_hfov_degrees, minimum_hfov)
    selected_vfov = _vertical_for_horizontal(selected_hfov, aspect)
    minimum_vfov = _vertical_for_horizontal(minimum_hfov, aspect)
    limiting = _limiting_corners(corners, yaw=yaw, pitch=pitch, aspect=aspect)
    pairs = _pairwise_angles(observations)
    feasible = minimum_hfov <= max_hfov_degrees and selected_hfov <= max_hfov_degrees
    if feasible:
        _verify_projection(
            rays,
            yaw=yaw,
            pitch=pitch,
            hfov=selected_hfov,
            vfov=selected_vfov,
        )
    return CommonFovSolution(
        yaw_degrees=yaw,
        pitch_degrees=pitch,
        minimum_hfov_degrees=minimum_hfov,
        minimum_vfov_degrees=minimum_vfov,
        hfov_degrees=selected_hfov,
        vfov_degrees=selected_vfov,
        feasible=feasible,
        limiting_corners=limiting,
        pairwise_angles=pairs,
    )


def _ray(observation: GroundingObservation, x: float, y: float) -> np.ndarray:
    matrix = np.asarray(observation.orientation_matrix, dtype=np.float64).reshape(3, 3)
    tangent_x = math.tan(math.radians(observation.probe_hfov_degrees) / 2)
    local = np.asarray(
        ((x - 0.5) * 2 * tangent_x, (0.5 - y) * 2 * tangent_x, 1.0),
        dtype=np.float64,
    )
    local /= np.linalg.norm(local)
    world = matrix @ local
    return world / np.linalg.norm(world)


def _optimize_axis(rays: np.ndarray, aspect: float) -> tuple[float, float, float, float]:
    starts: list[np.ndarray] = []
    summed = np.sum(rays, axis=0)
    if float(np.linalg.norm(summed)) > 1e-9:
        starts.append(summed / np.linalg.norm(summed))
    starts.extend(ray for ray in rays)
    for first in range(len(rays)):
        for second in range(first + 1, len(rays)):
            midpoint = rays[first] + rays[second]
            norm = float(np.linalg.norm(midpoint))
            if norm > 1e-9:
                starts.append(midpoint / norm)

    best = (math.inf, 0.0, 0.0, math.inf, math.inf)
    seen: set[tuple[int, int]] = set()
    for vector in starts:
        yaw = math.degrees(math.atan2(float(vector[0]), float(vector[2])))
        pitch = math.degrees(math.asin(float(np.clip(vector[1], -1.0, 1.0))))
        coordinate = (round(yaw * 1_000), round(pitch * 1_000))
        if coordinate in seen:
            continue
        seen.add(coordinate)
        candidate = _refine_axis(rays, aspect, yaw, pitch)
        if candidate < best:
            best = candidate
    if not math.isfinite(best[0]) or best[0] >= 179.999999:
        raise SpatialError("target corners cannot share a forward perspective hemisphere")
    _, yaw, pitch, hfov, vfov = best
    return yaw, pitch, hfov, vfov


def _refine_axis(
    rays: np.ndarray,
    aspect: float,
    yaw: float,
    pitch: float,
) -> tuple[float, float, float, float, float]:
    current = _axis_objective(rays, aspect, yaw, pitch)
    step = 30.0
    while step >= 1e-5:
        candidates = [current]
        for yaw_delta, pitch_delta in (
            (-step, 0.0),
            (step, 0.0),
            (0.0, -step),
            (0.0, step),
            (-step, -step),
            (-step, step),
            (step, -step),
            (step, step),
        ):
            candidates.append(
                _axis_objective(
                    rays,
                    aspect,
                    _wrap_yaw(current[1] + yaw_delta),
                    max(-89.9, min(89.9, current[2] + pitch_delta)),
                )
            )
        selected = min(candidates)
        if selected[0] + 1e-10 < current[0]:
            current = selected
        else:
            step /= 2
    return current


def _axis_objective(
    rays: np.ndarray,
    aspect: float,
    yaw: float,
    pitch: float,
) -> tuple[float, float, float, float, float]:
    orientation = np.asarray(orientation_from_yaw_pitch(yaw, pitch)).reshape(3, 3)
    local = rays @ orientation
    if np.any(local[:, 2] <= 1e-9):
        return (179.999999, yaw, pitch, 179.999999, 179.999999)
    horizontal_tangent = float(np.max(np.abs(local[:, 0] / local[:, 2])))
    vertical_tangent = float(np.max(np.abs(local[:, 1] / local[:, 2])))
    raw_hfov = math.degrees(2 * math.atan(horizontal_tangent))
    raw_vfov = math.degrees(2 * math.atan(vertical_tangent))
    objective = max(raw_hfov, _horizontal_for_vertical(raw_vfov, aspect))
    return (objective, yaw, pitch, raw_hfov, raw_vfov)


def _limiting_corners(
    corners: tuple[tuple[GroundingObservation, str, np.ndarray], ...],
    *,
    yaw: float,
    pitch: float,
    aspect: float,
) -> tuple[FovCornerDiagnostic, ...]:
    orientation = np.asarray(orientation_from_yaw_pitch(yaw, pitch)).reshape(3, 3)
    scored: list[tuple[float, FovCornerDiagnostic]] = []
    for observation, name, ray in corners:
        local = ray @ orientation
        horizontal = abs(math.atan2(float(local[0]), float(local[2])))
        vertical = abs(math.atan2(float(local[1]), float(local[2])))
        score = max(horizontal, math.atan(aspect * math.tan(vertical)))
        scored.append(
            (
                score,
                FovCornerDiagnostic(
                    target_key=observation.target_key,
                    observation_key=observation.observation_key,
                    corner=name,
                    yaw_degrees=math.degrees(math.atan2(float(ray[0]), float(ray[2]))),
                    pitch_degrees=math.degrees(math.asin(float(np.clip(ray[1], -1.0, 1.0)))),
                ),
            )
        )
    maximum = max(item[0] for item in scored)
    return tuple(
        diagnostic
        for score, diagnostic in sorted(
            scored,
            key=lambda item: (
                -item[0],
                item[1].target_key,
                item[1].observation_key,
                item[1].corner,
            ),
        )
        if maximum - score <= 1e-5
    )


def _pairwise_angles(
    observations: tuple[GroundingObservation, ...],
) -> tuple[PairwiseAngleDiagnostic, ...]:
    centers: list[tuple[str, np.ndarray]] = []
    for observation in observations:
        box = observation.normalized_box
        if box is None:
            continue
        centers.append(
            (
                observation.target_key,
                _ray(
                    observation,
                    (box.x_min + box.x_max) / 2,
                    (box.y_min + box.y_max) / 2,
                ),
            )
        )
    pairs = [
        PairwiseAngleDiagnostic(
            first_target_key=first_key,
            second_target_key=second_key,
            angle_degrees=math.degrees(
                math.acos(float(np.clip(np.dot(first_ray, second_ray), -1.0, 1.0)))
            ),
        )
        for first_index, (first_key, first_ray) in enumerate(centers)
        for second_key, second_ray in centers[first_index + 1 :]
    ]
    return tuple(
        sorted(
            pairs,
            key=lambda item: (-item.angle_degrees, item.first_target_key, item.second_target_key),
        )
    )


def _verify_projection(
    rays: np.ndarray,
    *,
    yaw: float,
    pitch: float,
    hfov: float,
    vfov: float,
) -> None:
    orientation = np.asarray(orientation_from_yaw_pitch(yaw, pitch)).reshape(3, 3)
    local = rays @ orientation
    horizontal_limit = math.tan(math.radians(hfov) / 2) + 1e-7
    vertical_limit = math.tan(math.radians(vfov) / 2) + 1e-7
    if np.any(local[:, 2] <= 0):
        raise SpatialError("common FOV verification placed a corner behind the camera")
    if np.any(np.abs(local[:, 0] / local[:, 2]) > horizontal_limit):
        raise SpatialError("common FOV verification lost a horizontal target corner")
    if np.any(np.abs(local[:, 1] / local[:, 2]) > vertical_limit):
        raise SpatialError("common FOV verification lost a vertical target corner")


def _horizontal_for_vertical(vfov_degrees: float, aspect: float) -> float:
    return math.degrees(2 * math.atan(aspect * math.tan(math.radians(vfov_degrees) / 2)))


def _vertical_for_horizontal(hfov_degrees: float, aspect: float) -> float:
    return math.degrees(2 * math.atan(math.tan(math.radians(hfov_degrees) / 2) / aspect))


def _wrap_yaw(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0
