"""End-to-end pilot and resumable cache generation pipelines."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from qwen_history_cache.config import CacheConfig
from qwen_history_cache.manifest import (
    collect_shard_metadata,
    create_build_spec,
    publish_profile,
)
from qwen_history_cache.metadata import load_worklist
from qwen_history_cache.qwen_encoder import (
    QwenHistoryEncoder,
    checkpoint_fingerprint,
    inspect_processor_contract,
    runtime_versions,
)
from qwen_history_cache.shard_writer import (
    AtomicShardWriter,
    ensure_build_spec,
    wait_for_build_spec,
)
from qwen_history_cache.validation import sha256_file, stable_json_sha256
from qwen_history_cache.video_reader import OpenCVFrameReader


def _processor_contract(config: CacheConfig) -> dict[str, Any]:
    return inspect_processor_contract(
        config.checkpoint_path,
        image_size=(config.image_width, config.image_height),
        expected_grid_thw=config.expected_grid_thw,
        expected_patch_size=config.expected_patch_size,
        expected_temporal_patch_size=config.expected_temporal_patch_size,
        expected_spatial_merge_size=config.expected_spatial_merge_size,
        expected_hidden_dim=config.hidden_dim,
    )


def _new_encoder(
    config: CacheConfig, *, device: str, model_dtype: str
) -> QwenHistoryEncoder:
    return QwenHistoryEncoder(
        config.checkpoint_path,
        device=device,
        model_dtype=model_dtype,
        image_size=(config.image_width, config.image_height),
        output_grid=config.output_grid,
        expected_grid_thw=config.expected_grid_thw,
        expected_patch_size=config.expected_patch_size,
        expected_temporal_patch_size=config.expected_temporal_patch_size,
        expected_spatial_merge_size=config.expected_spatial_merge_size,
        expected_hidden_dim=config.hidden_dim,
    )


def prepare_build_spec(
    config: CacheConfig,
    split: str,
    *,
    rank: int,
) -> tuple[dict[str, Any], Any]:
    """Hash all inputs, create the immutable spec, and return the worklist."""
    split = split.upper()
    worklist_path = config.worklist_path(split)
    if not worklist_path.is_file():
        raise FileNotFoundError(
            f"Generate the metadata worklist first: {worklist_path}"
        )
    worklist = load_worklist(worklist_path)
    spec_path = config.building_root(split) / ".generation" / "build_spec.json"
    if spec_path.is_file():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    elif rank != 0:
        spec = wait_for_build_spec(config.building_root(split))
    else:
        spec = create_build_spec(
            split=split,
            split_root=config.split_root(split),
            worklist_path=worklist_path,
            worklist_sha256=sha256_file(worklist_path),
            row_count=worklist.num_rows,
            checkpoint=checkpoint_fingerprint(
                config.checkpoint_path, revision=config.encoder_revision
            ),
            processor=_processor_contract(config),
            profile=config.profile,
            encoder_revision=config.encoder_revision,
            shard_size=config.shard_size,
            token_count=config.token_count,
            hidden_dim=config.hidden_dim,
            output_grid=config.output_grid,
        )
    _validate_reused_spec(config, split, spec, worklist_path, worklist.num_rows)
    ensure_build_spec(config.building_root(split), spec, rank=rank)
    return spec, worklist


def _validate_reused_spec(
    config: CacheConfig,
    split: str,
    spec: dict[str, Any],
    worklist_path: Path,
    row_count: int,
) -> None:
    """Reject stale inputs before reusing rank zero's checkpoint hashes."""
    expected = {
        "split": split,
        "split_root": str(config.split_root(split).resolve()),
        "worklist_path": str(worklist_path.resolve()),
        "worklist_sha256": sha256_file(worklist_path),
        "row_count": row_count,
        "profile": config.profile,
        "encoder_revision": config.encoder_revision,
        "shard_size": config.shard_size,
        "token_count": config.token_count,
        "hidden_dim": config.hidden_dim,
        "encoder_family": "qwen3_vl",
        "dtype": "float16",
        "file_format": "mmap_npy",
        "token_source": "model.visual.merger:main_image_embeds",
    }
    mismatches = {
        key: {"actual": spec.get(key), "expected": value}
        for key, value in expected.items()
        if spec.get(key) != value
    }
    checkpoint = spec.get("checkpoint", {})
    if checkpoint.get("checkpoint_path") != str(config.checkpoint_path.resolve()):
        mismatches["checkpoint_path"] = {
            "actual": checkpoint.get("checkpoint_path"),
            "expected": str(config.checkpoint_path.resolve()),
        }
    if spec.get("processor") != _processor_contract(config):
        mismatches["processor"] = "checkpoint processor contract changed"
    pooling = spec.get("pooling", {})
    if pooling.get("output_grid") != list(config.output_grid):
        mismatches["output_grid"] = {
            "actual": pooling.get("output_grid"),
            "expected": list(config.output_grid),
        }
    identity = dict(spec)
    recorded_run_fingerprint = identity.pop("run_fingerprint", None)
    if stable_json_sha256(identity) != recorded_run_fingerprint:
        mismatches["run_fingerprint"] = "build spec content hash is invalid"
    for name, recorded in checkpoint.get("files", {}).items():
        path = config.checkpoint_path / name
        if not path.is_file():
            mismatches[f"checkpoint_file:{name}"] = "missing"
            continue
        stat = path.stat()
        if stat.st_size != int(recorded["bytes"]) or stat.st_mtime_ns != int(
            recorded["mtime_ns"]
        ):
            mismatches[f"checkpoint_file:{name}"] = "size or mtime changed"
    if mismatches:
        raise ValueError(
            "Existing build spec no longer matches its inputs; refusing resume: "
            f"{mismatches}"
        )


def generate_split(
    config: CacheConfig,
    split: str,
    *,
    rank: int = 0,
    world_size: int = 1,
    device: str = "cuda:0",
    model_dtype: str = "auto",
    batch_size: int | None = None,
    max_shards: int | None = None,
    publish: bool = True,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Generate rank-owned shards and publish if every shard is complete."""
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("Require world_size > 0 and 0 <= rank < world_size")
    if max_shards is not None and max_shards <= 0:
        raise ValueError("max_shards must be positive when set")
    effective_batch_size = batch_size or config.batch_size
    if effective_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    split = split.upper()
    final_root = config.profile_root(split)
    if final_root.exists():
        raise FileExistsError(
            f"Published profile already exists; refusing to overwrite: {final_root}"
        )
    spec, worklist = prepare_build_spec(config, split, rank=rank)
    row_count = worklist.num_rows
    shard_count = (row_count + config.shard_size - 1) // config.shard_size
    owned_ids = [
        shard_id for shard_id in range(shard_count) if shard_id % world_size == rank
    ]
    incomplete: list[tuple[int, AtomicShardWriter]] = []
    reused: list[int] = []
    for shard_id in owned_ids:
        start = shard_id * config.shard_size
        rows = min(config.shard_size, row_count - start)
        writer = AtomicShardWriter(
            config.building_root(split),
            shard_id,
            rows=rows,
            token_count=config.token_count,
            hidden_dim=config.hidden_dim,
        )
        if writer.completed_metadata() is None:
            incomplete.append((shard_id, writer))
        else:
            reused.append(shard_id)
    if max_shards is not None:
        incomplete = incomplete[:max_shards]

    owned_rows = sum(
        min(config.shard_size, row_count - shard_id * config.shard_size)
        for shard_id in owned_ids
    )
    reused_rows = sum(
        min(config.shard_size, row_count - shard_id * config.shard_size)
        for shard_id in reused
    )
    generated: list[int] = []
    with tqdm(
        total=owned_rows,
        initial=reused_rows,
        desc=f"{split} rank {rank}",
        unit="ref",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as progress:
        if incomplete:
            progress.set_postfix_str("loading model", refresh=True)
            encoder = _new_encoder(config, device=device, model_dtype=model_dtype)
            with OpenCVFrameReader(config.split_root(split)) as reader:
                for shard_id, writer in incomplete:
                    shard_start = shard_id * config.shard_size
                    shard_rows = writer.shape[0]
                    progress.set_postfix_str(
                        f"shard {shard_id + 1}/{shard_count}", refresh=True
                    )
                    writer.open()
                    try:
                        for offset in range(0, shard_rows, effective_batch_size):
                            count = min(effective_batch_size, shard_rows - offset)
                            rows = worklist.slice(
                                shard_start + offset, count
                            ).to_pylist()
                            frames = reader.read_rows(rows)
                            writer.write(encoder.encode(frames))
                            progress.update(count)
                        writer.finalize()
                    except Exception:
                        writer.close()
                        raise
                    generated.append(shard_id)
            progress.set_postfix_str("complete", refresh=False)

    report: dict[str, Any] = {
        "split": split,
        "rank": rank,
        "world_size": world_size,
        "run_fingerprint": spec["run_fingerprint"],
        "owned_shards": owned_ids,
        "reused_shards": reused,
        "generated_shards": generated,
        "published": False,
    }
    if rank == 0 and publish and max_shards is None:
        _, missing = collect_shard_metadata(
            config.building_root(split),
            row_count=row_count,
            shard_size=config.shard_size,
            token_count=config.token_count,
            hidden_dim=config.hidden_dim,
        )
        report["missing_shards"] = missing
        if not missing:
            published = publish_profile(
                config.building_root(split),
                final_root,
                build_spec=spec,
                worklist_path=config.worklist_path(split),
                runtime=runtime_versions(),
            )
            report["published"] = True
            report["validation"] = published
    return report


def building_status(config: CacheConfig, split: str) -> dict[str, Any]:
    """Report strictly validated completed/missing shards without loading Qwen."""
    split = split.upper()
    spec_path = config.building_root(split) / ".generation" / "build_spec.json"
    if not spec_path.is_file():
        raise FileNotFoundError(f"No initialized build found: {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    completed, missing = collect_shard_metadata(
        config.building_root(split),
        row_count=int(spec["row_count"]),
        shard_size=int(spec["shard_size"]),
        token_count=int(spec["token_count"]),
        hidden_dim=int(spec["hidden_dim"]),
    )
    return {
        "split": split,
        "run_fingerprint": spec["run_fingerprint"],
        "completed_shards": [item["shard_id"] for item in completed],
        "missing_shards": missing,
        "ready_to_publish": not missing,
    }


def publish_completed_build(config: CacheConfig, split: str) -> dict[str, Any]:
    """Publish a complete distributed build from rank zero after workers exit."""
    split = split.upper()
    spec_path = config.building_root(split) / ".generation" / "build_spec.json"
    if not spec_path.is_file():
        raise FileNotFoundError(f"No initialized build found: {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    return publish_profile(
        config.building_root(split),
        config.profile_root(split),
        build_spec=spec,
        worklist_path=config.worklist_path(split),
        runtime=runtime_versions(),
    )


def _pilot_rows(worklist: Any, sample_count: int) -> list[dict[str, Any]]:
    grouped = (
        worklist.select(["video_path", "ref_id"])
        .group_by("video_path")
        .aggregate([("ref_id", "min")])
    )
    candidates = np.sort(
        np.asarray(grouped["ref_id_min"].combine_chunks(), dtype=np.int64)
    )
    indices = np.linspace(
        0, candidates.size - 1, min(sample_count, candidates.size), dtype=int
    )
    ref_ids = candidates[indices]
    return [worklist.slice(int(ref_id), 1).to_pylist()[0] for ref_id in ref_ids]


def run_pilot(
    config: CacheConfig,
    split: str,
    *,
    output_dir: str | Path,
    sample_count: int = 8,
    device: str = "cuda:0",
    model_dtype: str = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Encode diverse real frames twice and save a regression pilot artifact."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    split = split.upper()
    output = Path(output_dir)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Pilot output already exists (pass --force to replace): {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    worklist = load_worklist(config.worklist_path(split))
    rows = _pilot_rows(worklist, sample_count)
    with OpenCVFrameReader(config.split_root(split)) as reader:
        frames = reader.read_rows(rows)
    encoder = _new_encoder(config, device=device, model_dtype=model_dtype)
    first = encoder.encode(frames)
    second = encoder.encode(frames)
    max_abs_repeat_error = float(
        np.max(np.abs(first.astype(np.float32) - second.astype(np.float32)))
    )
    if max_abs_repeat_error != 0.0:
        raise ValueError(
            f"Repeated pilot encoding was not deterministic: {max_abs_repeat_error}"
        )
    np.save(output / "image_embeds.npy", first, allow_pickle=False)
    records = [
        {
            "ref_id": int(row["ref_id"]),
            "ref": str(row["ref"]),
            "video_path": str(row["video_path"]),
            "video_frame_index": int(row["video_frame_index"]),
            "token_mean": float(first[index].astype(np.float32).mean()),
            "token_std": float(first[index].astype(np.float32).std()),
        }
        for index, row in enumerate(rows)
    ]
    report = {
        "split": split,
        "samples": len(rows),
        "shape": list(first.shape),
        "dtype": str(first.dtype),
        "finite": bool(np.isfinite(first).all()),
        "max_abs_repeat_error": max_abs_repeat_error,
        "embeddings_sha256": sha256_file(output / "image_embeds.npy"),
        "processor": encoder.processor_metadata(),
        "runtime": runtime_versions(),
        "records": records,
    }
    report_path = output / "pilot_report.json"
    temporary = report_path.with_name(report_path.name + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, report_path)
    return report
