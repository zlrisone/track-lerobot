"""Atomic mmap shard writing and resumable shard-state records."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from qwen_history_cache.validation import (
    fsync_directory,
    fsync_file,
    validate_npy_shard,
)


def atomic_write_json(path: str | Path, value: dict[str, Any]) -> None:
    """Write a JSON object using fsync followed by same-filesystem rename."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.{os.getpid()}.partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fsync_file(partial)
    os.replace(partial, output)
    fsync_directory(output.parent)


def ensure_build_spec(
    building_root: str | Path,
    spec: dict[str, Any],
    *,
    rank: int,
    wait_seconds: float = 300.0,
) -> Path:
    """Create the immutable build spec on rank zero or validate it elsewhere."""
    root = Path(building_root)
    profile_root = root.with_name(root.name.removeprefix(".").removesuffix(".building"))
    if profile_root.exists():
        raise FileExistsError(
            "Published cache already exists and will not be overwritten: "
            f"{profile_root}"
        )
    path = root / ".generation" / "build_spec.json"
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            atomic_write_json(path, spec)
    else:
        deadline = time.monotonic() + wait_seconds
        while not path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for rank 0 to create {path}")
            time.sleep(0.5)
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read immutable build spec: {path}") from exc
    if existing != spec:
        raise ValueError(
            "Existing building directory has a different run fingerprint/config; "
            f"refusing to mix outputs: {path}"
        )
    return path


def wait_for_build_spec(
    building_root: str | Path, *, wait_seconds: float = 300.0
) -> dict[str, Any]:
    """Wait for rank zero's immutable build spec and return it."""
    path = Path(building_root) / ".generation" / "build_spec.json"
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for rank 0 to create {path}"
                ) from None
            time.sleep(0.5)
        except json.JSONDecodeError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Build spec remained unreadable: {path}") from None
            time.sleep(0.1)


def shard_filename(shard_id: int) -> str:
    """Return the deterministic shard filename."""
    if shard_id < 0:
        raise ValueError("shard_id cannot be negative")
    return f"image_embeds_{shard_id:06d}.npy"


def shard_state_path(building_root: str | Path, shard_id: int) -> Path:
    """Return the private validation-record path for one shard."""
    return Path(building_root) / ".generation" / "shards" / f"{shard_id:06d}.json"


class AtomicShardWriter:
    """Write one cache shard sequentially and publish only after validation."""

    def __init__(
        self,
        building_root: str | Path,
        shard_id: int,
        *,
        rows: int,
        token_count: int,
        hidden_dim: int,
    ) -> None:
        if min(rows, token_count, hidden_dim) <= 0:
            raise ValueError("Shard dimensions must be positive")
        self.building_root = Path(building_root)
        self.shard_id = shard_id
        self.shape = (rows, token_count, hidden_dim)
        self.final_path = self.building_root / "shards" / shard_filename(shard_id)
        self.partial_path = self.final_path.with_name(self.final_path.name + ".partial")
        self.state_path = shard_state_path(self.building_root, shard_id)
        self._array: np.memmap | None = None
        self._next_row = 0

    def completed_metadata(self) -> dict[str, Any] | None:
        """Validate and return metadata for a completed shard, if present."""
        if not self.final_path.is_file():
            return None
        metadata = validate_npy_shard(
            self.final_path,
            expected_shape=self.shape,
            expected_dtype=np.float16,
        )
        metadata.update(
            {
                "path": self.final_path.name,
                "shard_id": self.shard_id,
                "shard_path": f"shards/{self.final_path.name}",
            }
        )
        if self.state_path.is_file():
            recorded = json.loads(self.state_path.read_text(encoding="utf-8"))
            if recorded != metadata:
                raise ValueError(
                    f"Shard state does not match validated shard: {self.final_path}"
                )
        else:
            atomic_write_json(self.state_path, metadata)
        return metadata

    def open(self) -> AtomicShardWriter:
        """Create a fresh partial mmap, discarding only an incomplete partial."""
        if self.final_path.exists():
            raise FileExistsError(f"Completed shard already exists: {self.final_path}")
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        if self.partial_path.exists():
            self.partial_path.unlink()
        self._array = np.lib.format.open_memmap(
            self.partial_path,
            mode="w+",
            dtype=np.float16,
            shape=self.shape,
        )
        self._next_row = 0
        return self

    def write(self, values: np.ndarray) -> None:
        """Append one contiguous batch to the partial shard."""
        if self._array is None:
            raise RuntimeError("Shard writer is not open")
        batch = np.asarray(values)
        expected_tail = self.shape[1:]
        if batch.ndim != 3 or batch.shape[1:] != expected_tail:
            raise ValueError(
                f"Shard batch has shape {batch.shape}, expected "
                f"[B, {expected_tail[0]}, "
                f"{expected_tail[1]}]"
            )
        if batch.dtype != np.float16:
            raise ValueError(f"Shard batch must be float16, got {batch.dtype}")
        if not np.isfinite(batch).all():
            raise ValueError("Shard batch contains NaN or Inf")
        end = self._next_row + batch.shape[0]
        if end > self.shape[0]:
            raise ValueError(
                f"Shard write would exceed {self.shape[0]} rows: {self._next_row}:{end}"
            )
        self._array[self._next_row : end] = batch
        self._next_row = end

    def close(self) -> None:
        """Flush and close an open partial mmap without publishing it."""
        if self._array is None:
            return
        self._array.flush()
        mmap_object = getattr(self._array, "_mmap", None)
        if mmap_object is not None:
            mmap_object.close()
        self._array = None

    def finalize(self) -> dict[str, Any]:
        """Validate the partial file, atomically publish it, and save its hash."""
        if self._array is None:
            raise RuntimeError("Shard writer is not open")
        if self._next_row != self.shape[0]:
            raise ValueError(
                f"Shard is incomplete: wrote {self._next_row}/{self.shape[0]} rows"
            )
        self.close()
        fsync_file(self.partial_path)
        metadata = validate_npy_shard(
            self.partial_path,
            expected_shape=self.shape,
            expected_dtype=np.float16,
        )
        os.replace(self.partial_path, self.final_path)
        fsync_directory(self.final_path.parent)
        metadata.update(
            {
                "path": self.final_path.name,
                "shard_id": self.shard_id,
                "shard_path": f"shards/{self.final_path.name}",
            }
        )
        atomic_write_json(self.state_path, metadata)
        return metadata
