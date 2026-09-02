"""Content hashing and strict cache validation helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value with canonical ordering."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fsync_file(path: str | Path) -> None:
    """Synchronize an already-written file to stable storage."""
    with Path(path).open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path: str | Path) -> None:
    """Synchronize directory metadata where the platform supports it."""
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_npy_shard(
    path: str | Path,
    *,
    expected_shape: tuple[int, int, int],
    expected_dtype: np.dtype[Any] | str = np.float16,
    finite_chunk_rows: int = 256,
) -> dict[str, Any]:
    """Reopen an mmap shard and validate geometry, dtype, values, and hash."""
    shard_path = Path(path)
    values = np.load(shard_path, mmap_mode="r", allow_pickle=False)
    if values.shape != expected_shape:
        raise ValueError(
            f"Shard {shard_path} has shape {values.shape}, expected {expected_shape}"
        )
    expected = np.dtype(expected_dtype)
    if values.dtype != expected:
        raise ValueError(
            f"Shard {shard_path} has dtype {values.dtype}, expected {expected}"
        )
    for start in range(0, values.shape[0], finite_chunk_rows):
        if not np.isfinite(values[start : start + finite_chunk_rows]).all():
            raise ValueError(f"Shard {shard_path} contains NaN or Inf")
    del values
    return {
        "path": shard_path.name,
        "rows": expected_shape[0],
        "shape": list(expected_shape),
        "dtype": expected.name,
        "bytes": shard_path.stat().st_size,
        "sha256": sha256_file(shard_path),
    }
