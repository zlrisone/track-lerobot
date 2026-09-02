"""Build deterministic source-frame worklists from EVT-bench metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from qwen_history_cache.validation import (
    fsync_directory,
    fsync_file,
    sha256_file,
)
from qwen_history_cache.video_reader import OpenCVFrameReader, probe_video

WORKLIST_COLUMNS = (
    "ref_id",
    "ref",
    "episode_id",
    "frame_index",
    "camera_name",
    "data_index",
    "video_key",
    "video_frame_index",
    "chunk_index",
    "file_index",
    "video_path",
)


def _require_columns(table: pa.Table, columns: set[str], *, label: str) -> None:
    missing = sorted(columns - set(table.column_names))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    null_columns = [name for name in columns if table[name].null_count]
    if null_columns:
        raise ValueError(f"{label} contains nulls in columns: {null_columns}")


def _assert_unique(table: pa.Table, keys: list[str], *, label: str) -> None:
    marker = "__qwen_history_row_marker"
    marked = table.select(keys).append_column(
        marker, pa.array(np.ones(table.num_rows, dtype=np.int8))
    )
    grouped = marked.group_by(keys).aggregate([(marker, "count")])
    if grouped.num_rows != table.num_rows:
        raise ValueError(f"{label} must be unique by {keys}")


def _validate_refs(refs: pa.Table) -> None:
    required = {
        "ref_id",
        "ref",
        "episode_id",
        "frame_index",
        "camera_name",
    }
    _require_columns(refs, required, label="refs.parquet")
    if refs.num_rows == 0:
        raise ValueError("refs.parquet is empty")
    ids = np.asarray(refs["ref_id"].combine_chunks()).astype(np.int64, copy=False)
    if not np.array_equal(ids, np.arange(refs.num_rows, dtype=np.int64)):
        raise ValueError("ref_id must be ordered, unique, and contiguous from zero")
    if np.any(np.asarray(refs["frame_index"].combine_chunks()) < 0):
        raise ValueError("refs.parquet contains a negative frame_index")
    cameras = {str(item) for item in pc.unique(refs["camera_name"]).to_pylist()}
    if cameras != {"front"}:
        raise ValueError(f"Only front-camera refs are supported, found {cameras}")
    _assert_unique(refs, ["ref_id"], label="ref_id")
    _assert_unique(refs, ["ref"], label="ref")


def _read_frame_records(split_root: Path) -> pa.Table:
    paths = sorted(
        (split_root / "meta" / "checkpoints").glob("*_frame_records.parquet")
    )
    if not paths:
        raise FileNotFoundError(
            f"No *_frame_records.parquet files under {split_root / 'meta/checkpoints'}"
        )
    tables = [
        pq.read_table(path, columns=["episode_id", "frame_index", "index"])
        for path in paths
    ]
    records = pa.concat_tables(tables).rename_columns(
        ["episode_id", "frame_index", "data_index"]
    )
    _require_columns(
        records,
        {"episode_id", "frame_index", "data_index"},
        label="frame records",
    )
    _assert_unique(
        records,
        ["episode_id", "frame_index"],
        label="frame-record source key",
    )
    _assert_unique(records, ["data_index"], label="frame-record data index")
    return records


def _read_front_video_index(split_root: Path) -> pa.Table:
    path = split_root / "meta" / "navvla_video_index.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pq.read_table(
        path,
        columns=[
            "index",
            "video_key",
            "camera_name",
            "available",
            "video_frame_index",
            "chunk_index",
            "file_index",
        ],
        filters=[("camera_name", "=", "front"), ("available", "=", True)],
    ).drop(["camera_name", "available"])
    table = table.rename_columns(
        [
            "data_index",
            "video_key",
            "video_frame_index",
            "chunk_index",
            "file_index",
        ]
    )
    _require_columns(
        table,
        {
            "data_index",
            "video_key",
            "video_frame_index",
            "chunk_index",
            "file_index",
        },
        label="front video index",
    )
    _assert_unique(table, ["data_index"], label="front video data index")
    return table


def assemble_worklist(
    refs: pa.Table,
    frame_records: pa.Table,
    video_index: pa.Table,
) -> pa.Table:
    """Perform validated one-to-one joins and preserve ``ref_id`` ordering."""
    _validate_refs(refs)
    _require_columns(
        frame_records,
        {"episode_id", "frame_index", "data_index"},
        label="frame records",
    )
    _assert_unique(
        frame_records,
        ["episode_id", "frame_index"],
        label="frame-record source key",
    )
    _assert_unique(frame_records, ["data_index"], label="frame-record data index")
    _require_columns(
        video_index,
        {
            "data_index",
            "video_key",
            "video_frame_index",
            "chunk_index",
            "file_index",
        },
        label="front video index",
    )
    _assert_unique(video_index, ["data_index"], label="front video data index")

    joined = refs.select(
        ["ref_id", "ref", "episode_id", "frame_index", "camera_name"]
    ).join(
        frame_records.select(["episode_id", "frame_index", "data_index"]),
        keys=["episode_id", "frame_index"],
        join_type="inner",
    )
    if joined.num_rows != refs.num_rows:
        raise ValueError(
            f"refs -> frame-record join returned {joined.num_rows} rows for "
            f"{refs.num_rows} refs"
        )
    joined = joined.join(video_index, keys="data_index", join_type="inner")
    if joined.num_rows != refs.num_rows:
        raise ValueError(
            f"frame-record -> front-video join returned {joined.num_rows} rows "
            f"for {refs.num_rows} refs"
        )
    joined = joined.sort_by("ref_id")

    video_frames = np.asarray(joined["video_frame_index"].combine_chunks())
    if np.any(video_frames < 0):
        raise ValueError("Joined worklist contains a negative video_frame_index")
    chunks = np.asarray(joined["chunk_index"].combine_chunks())
    files = np.asarray(joined["file_index"].combine_chunks())
    if np.any(chunks < 0) or np.any(files < 0):
        raise ValueError("Joined worklist contains a negative chunk/file index")
    keys = joined["video_key"].to_pylist()
    paths = pa.array(
        [
            f"videos/{key}/chunk-{int(chunk):03d}/part-{int(file):03d}.mp4"
            for key, chunk, file in zip(keys, chunks, files, strict=True)
        ],
        type=pa.string(),
    )
    joined = joined.append_column("video_path", paths).select(WORKLIST_COLUMNS)
    _validate_refs(joined)
    return joined


def validate_worklist_videos(
    worklist: pa.Table,
    split_root: str | Path,
    *,
    mode: str = "bounds",
    decode_samples: int = 0,
) -> dict[str, Any]:
    """Validate all source files and optionally probe bounds and sample frames."""
    if mode not in {"none", "exists", "bounds"}:
        raise ValueError("video check mode must be one of: none, exists, bounds")
    if decode_samples < 0:
        raise ValueError("decode_samples cannot be negative")
    root = Path(split_root)
    grouped = (
        worklist.select(["video_path", "video_frame_index"])
        .group_by("video_path")
        .aggregate([("video_frame_index", "min"), ("video_frame_index", "max")])
    )
    groups = sorted(grouped.to_pylist(), key=lambda row: row["video_path"])
    probes: dict[str, dict[str, int | float]] = {}
    if mode != "none":
        missing = [
            row["video_path"]
            for row in groups
            if not (root / row["video_path"]).is_file()
        ]
        if missing:
            preview = missing[:5]
            raise FileNotFoundError(
                f"Missing {len(missing)} source videos; first entries: {preview}"
            )
    if mode == "bounds":
        for row in groups:
            relative = str(row["video_path"])
            info = probe_video(root / relative)
            maximum = int(row["video_frame_index_max"])
            if maximum >= int(info["frame_count"]):
                raise ValueError(
                    f"Worklist requests frame {maximum} from {relative}, but the "
                    f"video reports {info['frame_count']} frames"
                )
            probes[relative] = info

    decoded: list[dict[str, Any]] = []
    if decode_samples:
        indices = np.linspace(
            0, worklist.num_rows - 1, min(decode_samples, worklist.num_rows), dtype=int
        )
        rows = [worklist.slice(int(index), 1).to_pylist()[0] for index in indices]
        with OpenCVFrameReader(root) as reader:
            frames = reader.read_rows(rows)
        for row, frame in zip(rows, frames, strict=True):
            decoded.append(
                {
                    "ref_id": int(row["ref_id"]),
                    "video_path": str(row["video_path"]),
                    "video_frame_index": int(row["video_frame_index"]),
                    "height": int(frame.shape[0]),
                    "width": int(frame.shape[1]),
                    "channels": int(frame.shape[2]),
                }
            )
    return {
        "video_count": len(groups),
        "video_check": mode,
        "probed_video_count": len(probes),
        "decoded_samples": decoded,
    }


def write_worklist(
    table: pa.Table,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> str:
    """Atomically write a canonical worklist and return its file hash."""
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Worklist already exists (pass --force to replace it): {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    pq.write_table(table, partial, compression="zstd", write_statistics=True)
    fsync_file(partial)
    os.replace(partial, output)
    fsync_directory(output.parent)
    return sha256_file(output)


def write_worklist_summary(path: str | Path, summary: dict[str, Any]) -> Path:
    """Atomically persist the worklist digest and validation summary."""
    worklist_path = Path(path)
    output = worklist_path.with_suffix(".summary.json")
    partial = output.with_name(output.name + ".partial")
    partial.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fsync_file(partial)
    os.replace(partial, output)
    fsync_directory(output.parent)
    return output


def build_worklist(
    split_root: str | Path,
    output_path: str | Path,
    *,
    video_check: str = "bounds",
    decode_samples: int = 3,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build, validate, and atomically persist one EVT-bench worklist."""
    root = Path(split_root).resolve()
    refs_path = root / "meta" / "context_index" / "budget_1024" / "refs.parquet"
    refs = pq.read_table(
        refs_path,
        columns=[
            "ref_id",
            "ref",
            "episode_id",
            "frame_index",
            "camera_name",
        ],
    )
    frame_records = _read_frame_records(root)
    video_index = _read_front_video_index(root)
    worklist = assemble_worklist(refs, frame_records, video_index)
    video_summary = validate_worklist_videos(
        worklist,
        root,
        mode=video_check,
        decode_samples=decode_samples,
    )
    digest = write_worklist(worklist, output_path, overwrite=overwrite)
    summary = {
        "split": root.name,
        "split_root": str(root),
        "rows": worklist.num_rows,
        "worklist_path": str(Path(output_path).resolve()),
        "worklist_sha256": digest,
        **video_summary,
    }
    write_worklist_summary(output_path, summary)
    return summary


def load_worklist(path: str | Path) -> pa.Table:
    """Load and revalidate a persisted canonical worklist."""
    table = pq.read_table(path)
    missing = set(WORKLIST_COLUMNS) - set(table.column_names)
    if missing:
        raise ValueError(f"Worklist is missing columns: {sorted(missing)}")
    table = table.select(WORKLIST_COLUMNS)
    _validate_refs(table)
    return table
