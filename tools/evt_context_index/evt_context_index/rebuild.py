"""Build RLinf-compatible EVT-bench context indexes without TVI tokens.

The converted EVT-bench split already contains all information needed to rebuild
its context index. This module reads the checkpoint ``frame_records.parquet``
files, selects a chronological sliding window of recent front-camera frames and
writes the compact parquet/numpy representation consumed by RLinf.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROW_GROUP_SIZE = 131_072
CAMERA_NAME = "front"
BUDGET_MODEL_VERSION = "qwen_groot_no_tvi_v1"

ARRAY_DTYPES: dict[str, np.dtype] = {
    "history_step_timestamp": np.dtype(np.float32),
    "history_block_step_index": np.dtype(np.int32),
    "history_block_camera_id": np.dtype(np.int32),
    "history_block_ref_id": np.dtype(np.int64),
    "history_mask": np.dtype(bool),
    "long_memory_step_timestamp": np.dtype(np.float32),
    "long_memory_block_step_index": np.dtype(np.int32),
    "long_memory_block_camera_id": np.dtype(np.int32),
    "long_memory_block_ref_id": np.dtype(np.int64),
    "long_memory_mask": np.dtype(bool),
}

META_SCHEMA = pa.schema(
    [
        ("context.index_key", pa.string()),
        ("index", pa.int64()),
        # Kept for compatibility with the existing compact-index schema. It is
        # metadata only and has zero cost in the no-TVI budget model.
        ("current_tvi_time", pa.float64()),
        ("bats_k", pa.float64()),
        ("token_count_after", pa.int64()),
        ("context_policy_version", pa.string()),
        ("history_step_offset", pa.int64()),
        ("history_step_count", pa.int64()),
        ("history_block_offset", pa.int64()),
        ("history_block_count", pa.int64()),
        ("long_memory_step_offset", pa.int64()),
        ("long_memory_step_count", pa.int64()),
        ("long_memory_block_offset", pa.int64()),
        ("long_memory_block_count", pa.int64()),
    ]
)

REF_SCHEMA = pa.schema(
    [
        ("ref_id", pa.int64()),
        ("ref", pa.string()),
        ("episode_key", pa.string()),
        ("episode_id", pa.string()),
        ("frame_index", pa.int64()),
        ("camera_name", pa.string()),
    ]
)

DEBUG_SCHEMA = pa.schema(
    [
        ("index", pa.int64()),
        ("context.index_key", pa.string()),
        ("split", pa.string()),
        ("anchor_timestamp", pa.float64()),
        ("history_block_count", pa.int64()),
        ("current_visual_tokens", pa.int64()),
        ("history_visual_tokens", pa.int64()),
        ("total_context_tokens", pa.int64()),
        ("context_token_budget", pa.int64()),
        ("remaining_tokens", pa.int64()),
        ("tvi_tokens", pa.int64()),
    ]
)


@dataclass(frozen=True)
class BudgetSpec:
    """No-TVI context budget used by Qwen-GR00T.

    The budget is ``current_visual_tokens + N * history_visual_tokens``.
    Current/history TVI costs are deliberately absent.
    """

    token_budget: int = 1024
    current_visual_tokens: int = 64
    history_visual_tokens: int = 4

    def __post_init__(self) -> None:
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if self.current_visual_tokens < 0:
            raise ValueError("current_visual_tokens must be non-negative")
        if self.history_visual_tokens <= 0:
            raise ValueError("history_visual_tokens must be positive")
        if self.current_visual_tokens > self.token_budget:
            raise ValueError(
                "current_visual_tokens cannot exceed token_budget: "
                f"{self.current_visual_tokens} > {self.token_budget}"
            )

    @property
    def max_history_blocks(self) -> int:
        """Maximum number of cached history frames that fit in the budget."""
        return (
            self.token_budget - self.current_visual_tokens
        ) // self.history_visual_tokens

    def total_tokens(self, history_blocks: int) -> int:
        """Return total current-plus-history visual tokens for a row."""
        if history_blocks < 0:
            raise ValueError("history_blocks must be non-negative")
        return self.current_visual_tokens + history_blocks * self.history_visual_tokens


DEFAULT_BUDGET = BudgetSpec()


@dataclass(frozen=True)
class FrameRecord:
    """Minimal frame metadata needed to build a context row."""

    episode_index: int
    episode_id: str
    frame_index: int
    timestamp: float
    data_index: int


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata read from an already converted EVT-bench split."""

    dataset_name: str
    split: str
    policy_version: str
    expected_frames: int


@dataclass(frozen=True)
class DatasetInspection:
    """Read-only size estimate for one EVT-bench split."""

    dataset_root: str
    dataset_name: str
    split: str
    episodes: int
    frames: int
    history_blocks: int
    refs: int
    max_history_blocks: int
    estimated_array_bytes: int


@dataclass(frozen=True)
class BuildSummary:
    """Summary returned after a successful build and validation."""

    dataset_root: str
    dataset_name: str
    split: str
    token_budget: int
    current_visual_tokens: int
    history_visual_tokens: int
    max_history_blocks: int
    episodes: int
    frames: int
    history_blocks: int
    refs: int
    context_dir: str
    debug_path: str
    backup_dir: str | None


def select_recent_history(
    frames: Sequence[FrameRecord], anchor_pos: int, max_history_blocks: int
) -> list[FrameRecord]:
    """Select the immediately preceding frames in chronological order."""
    if anchor_pos < 0 or anchor_pos >= len(frames):
        raise IndexError(anchor_pos)
    if max_history_blocks < 0:
        raise ValueError("max_history_blocks must be non-negative")
    start = max(0, anchor_pos - max_history_blocks)
    return list(frames[start:anchor_pos])


def inspect_dataset(
    dataset_root: str | Path, budget: BudgetSpec = DEFAULT_BUDGET
) -> DatasetInspection:
    """Inspect a split and estimate the compact-array size without writing."""
    root = Path(dataset_root).resolve()
    metadata = _read_dataset_metadata(root)
    episodes = 0
    frames = 0
    history_blocks = 0
    refs = 0
    for episode in _iter_episodes(root):
        length = len(episode)
        episodes += 1
        frames += length
        refs += max(0, length - 1)
        history_blocks += _sliding_block_count(length, budget.max_history_blocks)
    _validate_frame_count(metadata, frames)
    bytes_per_block = sum(
        ARRAY_DTYPES[name].itemsize
        for name in (
            "history_step_timestamp",
            "history_block_step_index",
            "history_block_camera_id",
            "history_block_ref_id",
            "history_mask",
        )
    )
    return DatasetInspection(
        dataset_root=str(root),
        dataset_name=metadata.dataset_name,
        split=metadata.split,
        episodes=episodes,
        frames=frames,
        history_blocks=history_blocks,
        refs=refs,
        max_history_blocks=budget.max_history_blocks,
        estimated_array_bytes=history_blocks * bytes_per_block,
    )


def rebuild_dataset(
    dataset_root: str | Path,
    budget: BudgetSpec = DEFAULT_BUDGET,
    *,
    replace: bool = False,
    progress: Callable[[str], None] | None = None,
) -> BuildSummary:
    """Build, validate and publish one split's no-TVI context index.

    Existing indexes are never overwritten implicitly. With ``replace=True``
    they are moved to a timestamped backup after the staging build validates.
    """
    root = Path(dataset_root).resolve()
    metadata = _read_dataset_metadata(root)
    tag = f"budget_{budget.token_budget}"
    final_context_dir = root / "meta" / "context_index" / tag
    final_debug_path = (
        root / "cache" / "context_index_debug" / tag / (f"{metadata.split}.parquet")
    )
    if (final_context_dir.exists() or final_debug_path.exists()) and not replace:
        raise FileExistsError(
            f"context index already exists: {final_context_dir}; pass --replace "
            "to publish with a timestamped backup"
        )

    build_id = uuid.uuid4().hex
    staging_context_dir = final_context_dir.with_name(
        f".{tag}.no_tvi_building_{build_id}"
    )
    staging_debug_path = final_debug_path.with_name(
        f".{metadata.split}.no_tvi_building_{build_id}.parquet"
    )
    if staging_context_dir.exists() or staging_debug_path.exists():
        raise FileExistsError("unexpected context-index staging path collision")

    staging_context_dir.mkdir(parents=True)
    staging_debug_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        counts = _write_index(
            root,
            metadata=metadata,
            budget=budget,
            context_dir=staging_context_dir,
            debug_path=staging_debug_path,
            progress=progress,
        )
        if progress is not None:
            progress("validating staging index")
        _validate_index(
            staging_context_dir,
            staging_debug_path,
            expected_frames=counts["frames"],
            expected_history_blocks=counts["history_blocks"],
            expected_refs=counts["refs"],
            budget=budget,
        )
        backup_dir = _publish_index(
            root,
            metadata=metadata,
            budget=budget,
            staging_context_dir=staging_context_dir,
            staging_debug_path=staging_debug_path,
            final_context_dir=final_context_dir,
            final_debug_path=final_debug_path,
            replace=replace,
        )
        if progress is not None:
            progress(f"published {final_context_dir}")
    except Exception:
        if staging_context_dir.exists():
            shutil.rmtree(staging_context_dir)
        if staging_debug_path.exists():
            staging_debug_path.unlink()
        raise

    return BuildSummary(
        dataset_root=str(root),
        dataset_name=metadata.dataset_name,
        split=metadata.split,
        token_budget=budget.token_budget,
        current_visual_tokens=budget.current_visual_tokens,
        history_visual_tokens=budget.history_visual_tokens,
        max_history_blocks=budget.max_history_blocks,
        episodes=counts["episodes"],
        frames=counts["frames"],
        history_blocks=counts["history_blocks"],
        refs=counts["refs"],
        context_dir=str(final_context_dir),
        debug_path=str(final_debug_path),
        backup_dir=None if backup_dir is None else str(backup_dir),
    )


def validate_dataset(
    dataset_root: str | Path, budget: BudgetSpec = DEFAULT_BUDGET
) -> dict[str, int | str]:
    """Validate the published compact index and its no-TVI budget accounting."""
    root = Path(dataset_root).resolve()
    metadata = _read_dataset_metadata(root)
    tag = f"budget_{budget.token_budget}"
    context_dir = root / "meta" / "context_index" / tag
    debug_path = (
        root / "cache" / "context_index_debug" / tag / f"{metadata.split}.parquet"
    )
    refs = pq.ParquetFile(context_dir / "refs.parquet").metadata.num_rows
    meta_rows = pq.ParquetFile(context_dir / "context_meta.parquet").metadata.num_rows
    history_blocks = int(
        np.load(
            context_dir / "context_arrays" / "history_mask.npy",
            mmap_mode="r",
            allow_pickle=False,
        ).shape[0]
    )
    _validate_index(
        context_dir,
        debug_path,
        expected_frames=meta_rows,
        expected_history_blocks=history_blocks,
        expected_refs=refs,
        budget=budget,
    )
    _validate_frame_count(metadata, meta_rows)
    return {
        "dataset_root": str(root),
        "frames": meta_rows,
        "history_blocks": history_blocks,
        "refs": refs,
        "max_history_blocks": budget.max_history_blocks,
    }


def _read_dataset_metadata(root: Path) -> DatasetMetadata:
    info_path = root / "meta" / "info.json"
    schema_path = root / "meta" / "navvla_schema_ext.json"
    manifest_path = root / "meta" / "navvla_context_index_manifest.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing EVT metadata: {info_path}")
    if not schema_path.is_file():
        raise FileNotFoundError(f"missing EVT schema metadata: {schema_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    split = "train"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split = str(manifest.get("split", split))
    elif len(info.get("splits", {})) == 1:
        split = str(next(iter(info["splits"])))
    policy_version = str(
        schema.get(
            "context_policy_version",
            info.get("navvla", {}).get(
                "context_policy_version", "lerobot_sliding_short_long_v1"
            ),
        )
    )
    return DatasetMetadata(
        dataset_name=str(info["dataset_name"]),
        split=split,
        policy_version=policy_version,
        expected_frames=int(info["total_frames"]),
    )


def _checkpoint_paths(root: Path) -> list[Path]:
    paths = sorted((root / "meta" / "checkpoints").glob("*_frame_records.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no *_frame_records.parquet files under {root / 'meta' / 'checkpoints'}"
        )
    return paths


def _iter_episodes(root: Path) -> Iterator[list[FrameRecord]]:
    seen_episode_indices: set[int] = set()
    for path in _checkpoint_paths(root):
        table = pq.read_table(
            path,
            columns=[
                "episode_index",
                "episode_id",
                "frame_index",
                "timestamp",
                "index",
            ],
        )
        rows = table.to_pylist()
        rows.sort(key=lambda row: (int(row["episode_index"]), int(row["index"])))
        start = 0
        while start < len(rows):
            episode_index = int(rows[start]["episode_index"])
            end = start + 1
            while end < len(rows) and int(rows[end]["episode_index"]) == episode_index:
                end += 1
            if episode_index in seen_episode_indices:
                raise ValueError(
                    f"episode_index {episode_index} occurs in multiple checkpoint files"
                )
            seen_episode_indices.add(episode_index)
            group = rows[start:end]
            episode_ids = {str(row["episode_id"]) for row in group}
            if len(episode_ids) != 1:
                raise ValueError(
                    f"episode_index {episode_index} has multiple episode_id values"
                )
            frames = [
                FrameRecord(
                    episode_index=episode_index,
                    episode_id=str(row["episode_id"]),
                    frame_index=int(row["frame_index"]),
                    timestamp=float(row["timestamp"]),
                    data_index=int(row["index"]),
                )
                for row in group
            ]
            indexes = [frame.data_index for frame in frames]
            if len(indexes) != len(set(indexes)) or indexes != sorted(indexes):
                raise ValueError(
                    f"episode {frames[0].episode_id} has duplicate or unsorted data indexes"
                )
            yield frames
            start = end


def _sliding_block_count(frame_count: int, max_history_blocks: int) -> int:
    if frame_count <= 1 or max_history_blocks <= 0:
        return 0
    ramp = min(frame_count - 1, max_history_blocks)
    full_rows = max(0, (frame_count - 1) - ramp)
    return ramp * (ramp + 1) // 2 + full_rows * max_history_blocks


def _context_key(metadata: DatasetMetadata, episode_id: str, frame_index: int) -> str:
    return (
        f"{metadata.dataset_name}/{metadata.split}/{episode_id}/"
        f"f{frame_index:06d}/{metadata.policy_version}"
    )


def _history_ref(frame: FrameRecord) -> str:
    return f"{frame.episode_id}/{frame.frame_index:06d}/{CAMERA_NAME}"


class _ArraySpool:
    """Append compact arrays as raw binaries, then stream them into .npy files."""

    def __init__(self, context_dir: Path) -> None:
        self.spool_dir = context_dir / ".array_spool"
        self.spool_dir.mkdir()
        self.files = {
            name: (self.spool_dir / f"{name}.bin").open("wb") for name in ARRAY_DTYPES
        }
        self.counts = dict.fromkeys(ARRAY_DTYPES, 0)

    def append(self, name: str, values: Sequence[object]) -> None:
        array = np.asarray(values, dtype=ARRAY_DTYPES[name])
        if array.size:
            array.tofile(self.files[name])
        self.counts[name] += int(array.size)

    def close(self) -> None:
        for file in self.files.values():
            if not file.closed:
                file.close()

    def finish(self, arrays_dir: Path, *, chunk_elements: int = 1_000_000) -> None:
        self.close()
        temporary = arrays_dir.with_name(f"{arrays_dir.name}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        for name, dtype in ARRAY_DTYPES.items():
            count = int(self.counts[name])
            output = np.lib.format.open_memmap(
                temporary / f"{name}.npy", mode="w+", dtype=dtype, shape=(count,)
            )
            if count:
                source = np.memmap(
                    self.spool_dir / f"{name}.bin",
                    mode="r",
                    dtype=dtype,
                    shape=(count,),
                )
                for start in range(0, count, chunk_elements):
                    end = min(count, start + chunk_elements)
                    output[start:end] = source[start:end]
                del source
            output.flush()
            del output
        np.save(
            temporary / "camera_names.npy",
            np.asarray([CAMERA_NAME], dtype=str),
            allow_pickle=False,
        )
        temporary.rename(arrays_dir)
        shutil.rmtree(self.spool_dir)


def _table(rows: list[dict[str, object]], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=schema)


def _write_index(
    root: Path,
    *,
    metadata: DatasetMetadata,
    budget: BudgetSpec,
    context_dir: Path,
    debug_path: Path,
    progress: Callable[[str], None] | None,
) -> dict[str, int]:
    meta_path = context_dir / "context_meta.parquet"
    refs_path = context_dir / "refs.parquet"
    arrays_dir = context_dir / "context_arrays"
    spool = _ArraySpool(context_dir)
    meta_writer = pq.ParquetWriter(meta_path, META_SCHEMA)
    refs_writer = pq.ParquetWriter(refs_path, REF_SCHEMA)
    debug_writer = pq.ParquetWriter(debug_path, DEBUG_SCHEMA)
    history_step_offset = 0
    history_block_offset = 0
    ref_id_next = 0
    episodes = 0
    frame_count = 0
    history_block_count = 0
    try:
        for frames in _iter_episodes(root):
            episodes += 1
            frame_count += len(frames)
            episode_id = frames[0].episode_id

            # Every frame except the final frame can occur as a previous frame.
            # Assign refs once per episode instead of retaining a global map.
            ref_ids: dict[int, int] = {}
            ref_rows: list[dict[str, object]] = []
            for frame in frames[:-1]:
                ref_ids[frame.frame_index] = ref_id_next
                ref_rows.append(
                    {
                        "ref_id": ref_id_next,
                        "ref": _history_ref(frame),
                        "episode_key": episode_id,
                        "episode_id": episode_id,
                        "frame_index": frame.frame_index,
                        "camera_name": CAMERA_NAME,
                    }
                )
                ref_id_next += 1
            if ref_rows:
                refs_writer.write_table(
                    _table(ref_rows, REF_SCHEMA), row_group_size=ROW_GROUP_SIZE
                )

            meta_rows: list[dict[str, object]] = []
            debug_rows: list[dict[str, object]] = []
            episode_timestamps: list[float] = []
            episode_step_indexes: list[int] = []
            episode_camera_ids: list[int] = []
            episode_ref_ids: list[int] = []
            episode_masks: list[bool] = []
            for anchor_pos, frame in enumerate(frames):
                history = select_recent_history(
                    frames, anchor_pos, budget.max_history_blocks
                )
                count = len(history)
                total_tokens = budget.total_tokens(count)
                if total_tokens > budget.token_budget:
                    raise AssertionError(
                        f"internal budget error: {total_tokens} > {budget.token_budget}"
                    )
                key = _context_key(
                    metadata, episode_id=episode_id, frame_index=frame.frame_index
                )
                meta_rows.append(
                    {
                        "context.index_key": key,
                        "index": frame.data_index,
                        "current_tvi_time": frame.timestamp,
                        "bats_k": 0.0,
                        "token_count_after": total_tokens,
                        "context_policy_version": metadata.policy_version,
                        "history_step_offset": history_step_offset,
                        "history_step_count": count,
                        "history_block_offset": history_block_offset,
                        "history_block_count": count,
                        "long_memory_step_offset": 0,
                        "long_memory_step_count": 0,
                        "long_memory_block_offset": 0,
                        "long_memory_block_count": 0,
                    }
                )
                debug_rows.append(
                    {
                        "index": frame.data_index,
                        "context.index_key": key,
                        "split": metadata.split,
                        "anchor_timestamp": frame.timestamp,
                        "history_block_count": count,
                        "current_visual_tokens": budget.current_visual_tokens,
                        "history_visual_tokens": count * budget.history_visual_tokens,
                        "total_context_tokens": total_tokens,
                        "context_token_budget": budget.token_budget,
                        "remaining_tokens": budget.token_budget - total_tokens,
                        "tvi_tokens": 0,
                    }
                )
                episode_timestamps.extend(item.timestamp for item in history)
                episode_step_indexes.extend(range(count))
                episode_camera_ids.extend([0] * count)
                episode_ref_ids.extend(ref_ids[item.frame_index] for item in history)
                episode_masks.extend([True] * count)
                history_step_offset += count
                history_block_offset += count
                history_block_count += count

            meta_writer.write_table(
                _table(meta_rows, META_SCHEMA), row_group_size=ROW_GROUP_SIZE
            )
            debug_writer.write_table(
                _table(debug_rows, DEBUG_SCHEMA), row_group_size=ROW_GROUP_SIZE
            )
            spool.append("history_step_timestamp", episode_timestamps)
            spool.append("history_block_step_index", episode_step_indexes)
            spool.append("history_block_camera_id", episode_camera_ids)
            spool.append("history_block_ref_id", episode_ref_ids)
            spool.append("history_mask", episode_masks)
            if progress is not None and episodes % 250 == 0:
                progress(
                    f"indexed {episodes} episodes, {frame_count} frames, "
                    f"{history_block_count} history blocks"
                )
    finally:
        meta_writer.close()
        refs_writer.close()
        debug_writer.close()
        spool.close()

    _validate_frame_count(metadata, frame_count)
    if progress is not None:
        progress("converting array spools to mmap-compatible .npy files")
    spool.finish(arrays_dir)
    return {
        "episodes": episodes,
        "frames": frame_count,
        "history_blocks": history_block_count,
        "refs": ref_id_next,
    }


def _validate_frame_count(metadata: DatasetMetadata, actual: int) -> None:
    if actual != metadata.expected_frames:
        raise ValueError(
            f"checkpoint frame count {actual} does not match meta/info.json "
            f"total_frames={metadata.expected_frames}"
        )


def _validate_index(
    context_dir: Path,
    debug_path: Path,
    *,
    expected_frames: int,
    expected_history_blocks: int,
    expected_refs: int,
    budget: BudgetSpec,
) -> None:
    required = [
        context_dir / "context_meta.parquet",
        context_dir / "refs.parquet",
        debug_path,
        *(context_dir / "context_arrays" / f"{name}.npy" for name in ARRAY_DTYPES),
        context_dir / "context_arrays" / "camera_names.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"incomplete context index; missing: {missing}")

    meta_file = pq.ParquetFile(context_dir / "context_meta.parquet")
    refs_file = pq.ParquetFile(context_dir / "refs.parquet")
    if meta_file.metadata.num_rows != expected_frames:
        raise ValueError(
            f"context row count {meta_file.metadata.num_rows} != {expected_frames}"
        )
    if refs_file.metadata.num_rows != expected_refs:
        raise ValueError(
            f"ref row count {refs_file.metadata.num_rows} != {expected_refs}"
        )

    expected_offset = 0
    for batch in meta_file.iter_batches(
        columns=[
            "token_count_after",
            "history_step_offset",
            "history_step_count",
            "history_block_offset",
            "history_block_count",
            "long_memory_step_count",
            "long_memory_block_count",
        ],
        batch_size=ROW_GROUP_SIZE,
    ):
        payload = batch.to_pydict()
        for row in range(batch.num_rows):
            count = int(payload["history_block_count"][row])
            if int(payload["history_step_count"][row]) != count:
                raise ValueError("history step/block count mismatch")
            if int(payload["history_step_offset"][row]) != expected_offset:
                raise ValueError("non-contiguous history step offsets")
            if int(payload["history_block_offset"][row]) != expected_offset:
                raise ValueError("non-contiguous history block offsets")
            if count > budget.max_history_blocks:
                raise ValueError(
                    f"history count {count} exceeds {budget.max_history_blocks}"
                )
            expected_tokens = budget.total_tokens(count)
            if int(payload["token_count_after"][row]) != expected_tokens:
                raise ValueError("token_count_after does not match the no-TVI formula")
            if expected_tokens > budget.token_budget:
                raise ValueError("context row exceeds token budget")
            if int(payload["long_memory_step_count"][row]) != 0:
                raise ValueError(
                    "sliding index unexpectedly contains long-memory steps"
                )
            if int(payload["long_memory_block_count"][row]) != 0:
                raise ValueError(
                    "sliding index unexpectedly contains long-memory blocks"
                )
            expected_offset += count
    if expected_offset != expected_history_blocks:
        raise ValueError(
            f"history block total {expected_offset} != {expected_history_blocks}"
        )

    arrays_dir = context_dir / "context_arrays"
    for name, dtype in ARRAY_DTYPES.items():
        array = np.load(arrays_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if array.dtype != dtype:
            raise ValueError(f"{name} dtype {array.dtype} != {dtype}")
        expected = expected_history_blocks if name.startswith("history_") else 0
        if int(array.shape[0]) != expected:
            raise ValueError(f"{name} length {array.shape[0]} != {expected}")
    camera_names = np.load(
        arrays_dir / "camera_names.npy", mmap_mode="r", allow_pickle=False
    )
    if camera_names.tolist() != [CAMERA_NAME]:
        raise ValueError(f"expected only front camera, got {camera_names.tolist()}")

    debug_file = pq.ParquetFile(debug_path)
    if debug_file.metadata.num_rows != expected_frames:
        raise ValueError("debug/context row count mismatch")
    for batch in debug_file.iter_batches(
        columns=["total_context_tokens", "tvi_tokens"], batch_size=ROW_GROUP_SIZE
    ):
        payload = batch.to_pydict()
        if any(
            int(value) > budget.token_budget
            for value in payload["total_context_tokens"]
        ):
            raise ValueError("debug table contains an over-budget row")
        if any(int(value) != 0 for value in payload["tvi_tokens"]):
            raise ValueError("debug table contains non-zero TVI token cost")


def _publish_index(
    root: Path,
    *,
    metadata: DatasetMetadata,
    budget: BudgetSpec,
    staging_context_dir: Path,
    staging_debug_path: Path,
    final_context_dir: Path,
    final_debug_path: Path,
    replace: bool,
) -> Path | None:
    backup_root: Path | None = None
    old_context_backup: Path | None = None
    old_debug_backup: Path | None = None
    if final_context_dir.exists() or final_debug_path.exists():
        if not replace:
            raise FileExistsError(final_context_dir)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_root = (
            root
            / "meta"
            / "context_index_backups"
            / (f"budget_{budget.token_budget}_{stamp}")
        )
        if backup_root.exists():
            raise FileExistsError(f"backup path already exists: {backup_root}")
        backup_root.mkdir(parents=True)
        if final_context_dir.exists():
            old_context_backup = backup_root / "context_index"
            final_context_dir.rename(old_context_backup)
        if final_debug_path.exists():
            old_debug_backup = backup_root / "debug.parquet"
            final_debug_path.rename(old_debug_backup)

    try:
        final_context_dir.parent.mkdir(parents=True, exist_ok=True)
        final_debug_path.parent.mkdir(parents=True, exist_ok=True)
        staging_context_dir.rename(final_context_dir)
        staging_debug_path.rename(final_debug_path)
        _write_manifest(root, metadata=metadata, budget=budget)
    except Exception:
        if final_context_dir.exists() and staging_context_dir.parent.exists():
            final_context_dir.rename(staging_context_dir)
        if final_debug_path.exists() and staging_debug_path.parent.exists():
            final_debug_path.rename(staging_debug_path)
        if old_context_backup is not None and old_context_backup.exists():
            old_context_backup.rename(final_context_dir)
        if old_debug_backup is not None and old_debug_backup.exists():
            old_debug_backup.rename(final_debug_path)
        raise
    return backup_root


def _write_manifest(
    root: Path, *, metadata: DatasetMetadata, budget: BudgetSpec
) -> None:
    tag = f"budget_{budget.token_budget}"
    payload = {
        "version": 1,
        "split": metadata.split,
        "default_token_budget": budget.token_budget,
        "available_token_budgets": [budget.token_budget],
        "budget_model": {
            "version": BUDGET_MODEL_VERSION,
            "selection": "sliding_recent",
            "camera_names": [CAMERA_NAME],
            "formula": (
                "current_visual_tokens + history_blocks * "
                "history_visual_tokens_per_block"
            ),
            "current_visual_tokens": budget.current_visual_tokens,
            "history_visual_tokens_per_block": budget.history_visual_tokens,
            "tvi_tokens": 0,
            "max_history_blocks": budget.max_history_blocks,
        },
        "entries": {
            str(budget.token_budget): {
                "token_budget": budget.token_budget,
                "context_dir": f"meta/context_index/{tag}",
                "meta_path": f"meta/context_index/{tag}/context_meta.parquet",
                "refs_path": f"meta/context_index/{tag}/refs.parquet",
                "arrays_path": f"meta/context_index/{tag}/context_arrays",
                "debug_path": (
                    f"cache/context_index_debug/{tag}/{metadata.split}.parquet"
                ),
            }
        },
    }
    path = root / "meta" / "navvla_context_index_manifest.json"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def summary_json(value: object) -> str:
    """Serialize a dataclass summary for CLI output."""

    def encode(item: object) -> object:
        if is_dataclass(item) and not isinstance(item, type):
            return asdict(item)
        raise TypeError(
            f"Object of type {type(item).__name__} is not JSON serializable"
        )

    return json.dumps(value, ensure_ascii=False, indent=2, default=encode)
