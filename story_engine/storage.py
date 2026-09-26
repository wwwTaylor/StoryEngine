"""Atomic run-state and content-addressed artifact storage."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

from pydantic import Field

from story_engine.domain.common import FrozenModel
from story_engine.errors import ArtifactError
from story_engine.ids import canonical_bytes, sha256_bytes


class ArtifactRef(FrozenModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    media_type: str
    relative_path: str


class ArtifactStore:
    """Immutable SHA-256 storage rooted below one run directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for_hash(self, digest: str) -> Path:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ArtifactError("invalid artifact hash")
        return self.root / "sha256" / digest[:2] / digest

    def resolve(self, reference: ArtifactRef) -> Path:
        path = (self.root / reference.relative_path).resolve()
        if not path.is_relative_to(self.root):
            raise ArtifactError("artifact reference escapes store root")
        return path

    def put_bytes(self, data: bytes, media_type: str) -> ArtifactRef:
        digest = str(sha256_bytes(data))
        destination = self._path_for_hash(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            reference = self._reference(destination, digest, media_type)
            self.verify(reference)
            return reference
        self._atomic_write(destination, data)
        reference = self._reference(destination, digest, media_type)
        self.verify(reference)
        return reference

    def put_stream(self, source: BinaryIO, media_type: str) -> ArtifactRef:
        with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            digest = __import__("hashlib").sha256()
            size = 0
            while chunk := source.read(1024 * 1024):
                temporary.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())
        hex_digest = digest.hexdigest()
        destination = self._path_for_hash(hex_digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if destination.exists():
                temporary_path.unlink()
            else:
                os.replace(temporary_path, destination)
                _fsync_directory(destination.parent)
        finally:
            temporary_path.unlink(missing_ok=True)
        reference = ArtifactRef(
            sha256=hex_digest,
            size_bytes=size,
            media_type=media_type,
            relative_path=destination.relative_to(self.root).as_posix(),
        )
        self.verify(reference)
        return reference

    def put_file(self, source: Path, media_type: str) -> ArtifactRef:
        try:
            with source.open("rb") as file:
                return self.put_stream(file, media_type)
        except OSError as exc:
            raise ArtifactError(f"cannot store artifact {source}: {exc}") from exc

    def verify(self, reference: ArtifactRef) -> Path:
        path = self.resolve(reference)
        try:
            stat = path.stat()
            if stat.st_size != reference.size_bytes:
                raise ArtifactError(f"artifact size mismatch: {reference.sha256}")
            with path.open("rb") as file:
                actual = sha256_bytes(file.read())
        except OSError as exc:
            raise ArtifactError(f"cannot verify artifact {reference.sha256}: {exc}") from exc
        if str(actual) != reference.sha256:
            raise ArtifactError(f"artifact hash mismatch: {reference.sha256}")
        return path

    def materialize(self, reference: ArtifactRef, destination: Path) -> Path:
        source = self.verify(reference)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise ArtifactError(f"cannot materialize {destination}: {exc}") from exc
        return destination

    def _reference(self, path: Path, digest: str, media_type: str) -> ArtifactRef:
        return ArtifactRef(
            sha256=digest,
            size_bytes=path.stat().st_size,
            media_type=media_type,
            relative_path=path.relative_to(self.root).as_posix(),
        )

    @staticmethod
    def _atomic_write(destination: Path, data: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(dir=destination.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise ArtifactError(f"cannot write artifact: {exc}") from exc


def atomic_write_json(path: Path, value: Any) -> None:
    """Write canonical JSON through an fsync + atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ArtifactError(f"cannot atomically write {path}: {exc}") from exc


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read JSON {path}: {exc}") from exc


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ArtifactError(f"cannot fsync directory {path}: {exc}") from exc
