"""Build fingerprints, index generation, profile validation, and publication."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from qwen_history_cache.metadata import load_worklist
from qwen_history_cache.shard_writer import (
    AtomicShardWriter,
    atomic_write_json,
    shard_filename,
)
from qwen_history_cache.validation import (
    fsync_directory,
    fsync_file,
    sha256_file,
    stable_json_sha256,
    validate_npy_shard,
)


def source_fingerprint() -> dict[str, Any]:
    """Hash the installed generator sources used for this run."""
    root = Path(__file__).resolve().parent
    files = {
        path.name: sha256_file(path)
        for path in sorted(root.glob("*.py"))
        if path.is_file()
    }
    return {"files": files, "aggregate_sha256": stable_json_sha256(files)}


def create_build_spec(
    *,
    split: str,
    split_root: str | Path,
    worklist_path: str | Path,
    worklist_sha256: str,
    row_count: int,
    checkpoint: dict[str, Any],
    processor: dict[str, Any],
    profile: str,
    encoder_revision: str,
    shard_size: int,
    token_count: int,
    hidden_dim: int,
    output_grid: tuple[int, int],
) -> dict[str, Any]:
    """Create the immutable configuration checked by every resumed rank."""
    identity = {
        "schema_version": 1,
        "split": split,
        "split_root": str(Path(split_root).resolve()),
        "worklist_path": str(Path(worklist_path).resolve()),
        "worklist_sha256": worklist_sha256,
        "row_count": row_count,
        "checkpoint": checkpoint,
        "processor": processor,
        "profile": profile,
        "encoder_family": "qwen3_vl",
        "encoder_revision": encoder_revision,
        "token_source": "model.visual.merger:main_image_embeds",
        "pooling": {
            "method": "adaptive_average_pool2d_fp32",
            "output_grid": list(output_grid),
            "token_order": ["top_left", "top_right", "bottom_left", "bottom_right"],
            "extra_normalization": "none",
        },
        "shard_size": shard_size,
        "token_count": token_count,
        "hidden_dim": hidden_dim,
        "dtype": "float16",
        "file_format": "mmap_npy",
    }
    return {**identity, "run_fingerprint": stable_json_sha256(identity)}


def _rows_in_shard(row_count: int, shard_size: int, shard_id: int) -> int:
    start = shard_id * shard_size
    return min(shard_size, row_count - start)


def collect_shard_metadata(
    building_root: str | Path,
    *,
    row_count: int,
    shard_size: int,
    token_count: int,
    hidden_dim: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Strictly validate completed shards and report any missing IDs."""
    shard_count = (row_count + shard_size - 1) // shard_size
    completed: list[dict[str, Any]] = []
    missing: list[int] = []
    for shard_id in range(shard_count):
        rows = _rows_in_shard(row_count, shard_size, shard_id)
        writer = AtomicShardWriter(
            building_root,
            shard_id,
            rows=rows,
            token_count=token_count,
            hidden_dim=hidden_dim,
        )
        metadata = writer.completed_metadata()
        if metadata is None:
            missing.append(shard_id)
        else:
            completed.append(metadata)
    return completed, missing


def write_index(
    worklist: pa.Table,
    output_path: str | Path,
    *,
    shard_size: int,
    token_count: int,
    hidden_dim: int,
) -> str:
    """Write the RLinf cache index in worklist/ref_id order."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ref_ids = np.asarray(worklist["ref_id"].combine_chunks(), dtype=np.int64)
    shard_ids = ref_ids // shard_size
    unique_paths = [
        f"shards/{shard_filename(int(shard_id))}"
        for shard_id in range(int(shard_ids.max()) + 1)
    ]
    shard_paths = pc.take(pa.array(unique_paths, type=pa.string()), pa.array(shard_ids))
    index = pa.table(
        {
            "ref": worklist["ref"],
            "shard_path": shard_paths,
            "row_index": pa.array(ref_ids % shard_size, type=pa.int64()),
            "token_count": pa.array(
                np.full(worklist.num_rows, token_count, dtype=np.int16)
            ),
            "hidden_dim": pa.array(
                np.full(worklist.num_rows, hidden_dim, dtype=np.int32)
            ),
            "ref_id": worklist["ref_id"],
            "video_path": worklist["video_path"],
            "video_frame_index": worklist["video_frame_index"],
        }
    )
    partial = output.with_name(output.name + ".partial")
    pq.write_table(index, partial, compression="zstd", write_statistics=True)
    fsync_file(partial)
    os.replace(partial, output)
    fsync_directory(output.parent)
    return sha256_file(output)


def create_manifest(
    build_spec: dict[str, Any],
    *,
    shards: list[dict[str, Any]],
    index_sha256: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Create the final manifest, including the mandatory RLinf fields."""
    return {
        "schema_version": 1,
        "encoder_family": "qwen3_vl",
        "encoder_revision": build_spec["encoder_revision"],
        "token_count": build_spec["token_count"],
        "hidden_dim": build_spec["hidden_dim"],
        "dtype": "float16",
        "file_format": "mmap_npy",
        "profile": build_spec["profile"],
        "split": build_spec["split"],
        "dataset_root": build_spec["split_root"],
        "ref_count": build_spec["row_count"],
        "worklist_sha256": build_spec["worklist_sha256"],
        "run_fingerprint": build_spec["run_fingerprint"],
        "checkpoint": build_spec["checkpoint"],
        "processor": build_spec["processor"],
        "token_source": build_spec["token_source"],
        "pooling": build_spec["pooling"],
        "shard_size": build_spec["shard_size"],
        "shard_count": len(shards),
        "shards": shards,
        "index_sha256": index_sha256,
        "runtime": runtime,
        "generator": source_fingerprint(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def validate_profile(
    profile_root: str | Path,
    *,
    worklist_path: str | Path | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Validate manifest, index mapping, shard geometry, values, and hashes."""
    root = Path(profile_root)
    manifest_path = root / "manifest.json"
    index_path = root / "index.parquet"
    if not manifest_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            f"Profile must contain manifest.json and index.parquet: {root}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "encoder_family": "qwen3_vl",
        "encoder_revision": "main",
        "token_count": 4,
        "hidden_dim": 2048,
        "dtype": "float16",
        "file_format": "mmap_npy",
    }
    errors = {
        key: {"actual": manifest.get(key), "expected": value}
        for key, value in required.items()
        if manifest.get(key) != value
    }
    if errors:
        raise ValueError(f"Manifest contract mismatch: {errors}")
    if verify_hashes and sha256_file(index_path) != manifest["index_sha256"]:
        raise ValueError("index.parquet SHA-256 does not match manifest")

    index = pq.read_table(
        index_path,
        columns=["ref", "shard_path", "row_index", "token_count", "hidden_dim"],
    )
    row_count = int(manifest["ref_count"])
    if index.num_rows != row_count:
        raise ValueError(
            f"Index has {index.num_rows} rows, manifest expects {row_count}"
        )
    expected_ids = np.arange(row_count, dtype=np.int64)
    expected_rows = expected_ids % int(manifest["shard_size"])
    actual_rows = np.asarray(index["row_index"].combine_chunks(), dtype=np.int64)
    if not np.array_equal(actual_rows, expected_rows):
        raise ValueError("Index row_index values do not match deterministic sharding")
    if not np.all(np.asarray(index["token_count"].combine_chunks()) == 4):
        raise ValueError("Index token_count is not uniformly 4")
    if not np.all(np.asarray(index["hidden_dim"].combine_chunks()) == 2048):
        raise ValueError("Index hidden_dim is not uniformly 2048")
    if worklist_path is not None:
        worklist_path = Path(worklist_path)
        if verify_hashes and sha256_file(worklist_path) != manifest["worklist_sha256"]:
            raise ValueError("Worklist SHA-256 does not match manifest")
        worklist = load_worklist(worklist_path)
        if not index["ref"].equals(worklist["ref"]):
            raise ValueError("Index refs differ from the canonical worklist")

    shard_records = sorted(manifest["shards"], key=lambda item: item["shard_id"])
    if len(shard_records) != int(manifest["shard_count"]):
        raise ValueError("Manifest shard_count does not match its shard records")
    total_rows = 0
    for expected_id, record in enumerate(shard_records):
        if int(record["shard_id"]) != expected_id:
            raise ValueError("Manifest shard IDs must be contiguous from zero")
        path = root / str(record["shard_path"])
        shape = tuple(int(value) for value in record["shape"])
        indexed_paths = index["shard_path"].slice(total_rows, shape[0])
        if not bool(pc.all(pc.equal(indexed_paths, str(record["shard_path"]))).as_py()):
            raise ValueError(
                f"Index shard_path mapping is incorrect for shard {expected_id}"
            )
        validated = validate_npy_shard(
            path,
            expected_shape=shape,
            expected_dtype=np.float16,
        )
        if verify_hashes and validated["sha256"] != record["sha256"]:
            raise ValueError(f"Shard SHA-256 mismatch: {path}")
        total_rows += shape[0]
    if total_rows != row_count:
        raise ValueError(
            f"Shards contain {total_rows} rows, manifest expects {row_count}"
        )
    return {
        "profile_root": str(root.resolve()),
        "rows": row_count,
        "shards": len(shard_records),
        "manifest_sha256": sha256_file(manifest_path),
        "verified_hashes": verify_hashes,
    }


def publish_profile(
    building_root: str | Path,
    final_root: str | Path,
    *,
    build_spec: dict[str, Any],
    worklist_path: str | Path,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Finalize index/manifest and atomically publish a complete profile."""
    building = Path(building_root)
    final = Path(final_root)
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite published profile: {final}")
    shards, missing = collect_shard_metadata(
        building,
        row_count=int(build_spec["row_count"]),
        shard_size=int(build_spec["shard_size"]),
        token_count=int(build_spec["token_count"]),
        hidden_dim=int(build_spec["hidden_dim"]),
    )
    if missing:
        preview = missing[:20]
        raise RuntimeError(
            f"Cannot publish: {len(missing)} shards are incomplete; "
            f"first IDs: {preview}"
        )
    worklist = load_worklist(worklist_path)
    if worklist.num_rows != int(build_spec["row_count"]):
        raise ValueError("Worklist row count changed after build initialization")
    index_sha256 = write_index(
        worklist,
        building / "index.parquet",
        shard_size=int(build_spec["shard_size"]),
        token_count=int(build_spec["token_count"]),
        hidden_dim=int(build_spec["hidden_dim"]),
    )
    manifest = create_manifest(
        build_spec,
        shards=shards,
        index_sha256=index_sha256,
        runtime=runtime,
    )
    atomic_write_json(building / "manifest.json", manifest)
    report = validate_profile(building, worklist_path=worklist_path, verify_hashes=True)
    generation_root = building / ".generation"
    if generation_root.exists():
        shutil.rmtree(generation_root)
        fsync_directory(building)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.rename(building, final)
    fsync_directory(final.parent)
    report["profile_root"] = str(final.resolve())
    return report
