"""Canonical serialization, stable identifiers, and hashes."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, NewType

EntityKey = NewType("EntityKey", str)
SceneKey = NewType("SceneKey", str)
ShotKey = NewType("ShotKey", str)
BeatKey = NewType("BeatKey", str)
PlanKey = NewType("PlanKey", str)
CandidateKey = NewType("CandidateKey", str)
ArtifactHash = NewType("ArtifactHash", str)


def _canonicalize(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return _canonicalize(value.model_dump(mode="python", by_alias=True, exclude_none=False))
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return value.isoformat(timespec="microseconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_canonicalize(item) for item in value]
        return sorted(normalized, key=lambda item: canonical_json(item))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical values cannot contain NaN or infinity")
        if value == 0:
            return 0.0
        return float(format(value, ".12g"))
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return deterministic UTF-8 JSON for a supported value."""

    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(data: bytes) -> ArtifactHash:
    return ArtifactHash(hashlib.sha256(data).hexdigest())


def canonical_hash(value: Any) -> str:
    return str(sha256_bytes(canonical_bytes(value)))


def stable_key(kind: str, namespace: str, alias: str) -> str:
    """Create a readable deterministic key without trusting model-provided IDs."""

    digest = hashlib.sha256(f"{kind}\0{namespace}\0{alias}".encode()).hexdigest()[:16]
    return f"{kind}_{digest}"
