from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Callable, Iterator

from .schema import ContextEpisode, ContextFrame, DatasetSpec


ARRAY_DTYPES = {
    "history_step_timestamp": "float32",
    "history_block_step_index": "int32",
    "history_block_camera_id": "int32",
    "history_block_ref_id": "int64",
    "history_mask": "bool",
    "long_memory_step_timestamp": "float32",
    "long_memory_block_step_index": "int32",
    "long_memory_block_camera_id": "int32",
    "long_memory_block_ref_id": "int64",
    "long_memory_mask": "bool",
}

CONTEXT_META_COLUMNS = [
    "context.index_key",
    "index",
    "current_tvi_time",
    "bats_k",
    "token_count_after",
    "context_policy_version",
    "history_step_offset",
    "history_step_count",
    "history_block_offset",
    "history_block_count",
    "long_memory_step_offset",
    "long_memory_step_count",
    "long_memory_block_offset",
    "long_memory_block_count",
]

PARQUET_ROW_GROUP_SIZE = 131_072


def make_context_key(
    dataset_name: str,
    split: str,
    episode_id: str,
    frame_index: int,
    policy_version: str,
) -> str:
    return f"{dataset_name}/{split}/{episode_id}/f{frame_index:06d}/{policy_version}"


def normalize_context_cameras(names: tuple[str, ...], valid_names: tuple[str, ...]) -> tuple[str, ...]:
    if not names:
        raise ValueError("context cameras cannot be empty")
    unknown = [name for name in names if name not in valid_names]
    if unknown:
        raise ValueError(f"unknown context cameras: {unknown}; valid cameras: {list(valid_names)}")
    result: list[str] = []
    for name in names:
        if name not in result:
            result.append(name)
    return tuple(result)


def write_context_indexes(
    episode_factory: Callable[[], Iterator[ContextEpisode]],
    *,
    spec: DatasetSpec,
    dataset_root: Path,
    frame_hashes: dict[tuple[str, int, str], str],
    token_budgets: tuple[int, ...],
    camera_names: tuple[str, ...],
    valid_camera_names: tuple[str, ...],
    use_bats: bool,
    use_hash_dedup: bool,
    dhash_threshold: int,
) -> None:
    camera_names = normalize_context_cameras(camera_names, valid_camera_names)
    results: dict[int, dict[str, Path]] = {}
    for budget in dict.fromkeys(int(value) for value in token_budgets):
        if budget <= 0:
            raise ValueError(f"context token budget must be positive: {budget}")
        context_dir = dataset_root / "meta" / "context_index" / f"budget_{budget}"
        debug_path = (
            dataset_root
            / "cache"
            / "context_index_debug"
            / f"budget_{budget}"
            / f"{spec.split}.parquet"
        )
        context_dir.mkdir(parents=True, exist_ok=True)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = context_dir / "context_meta.parquet"
        refs_path = context_dir / "refs.parquet"
        arrays_path = context_dir / "context_arrays"
        _write_context_budget(
            episode_factory(),
            spec=spec,
            token_budget=budget,
            meta_path=meta_path,
            refs_path=refs_path,
            arrays_path=arrays_path,
            debug_path=debug_path,
            frame_hashes=frame_hashes,
            camera_names=camera_names,
            use_bats=use_bats,
            use_hash_dedup=use_hash_dedup,
            dhash_threshold=dhash_threshold,
        )
        results[budget] = {
            "context_dir": context_dir,
            "meta_path": meta_path,
            "refs_path": refs_path,
            "arrays_path": arrays_path,
            "debug_path": debug_path,
        }

    default_budget = next(iter(results))
    manifest = {
        "version": 1,
        "split": spec.split,
        "default_token_budget": default_budget,
        "available_token_budgets": sorted(results),
        "entries": {
            str(budget): {
                "token_budget": budget,
                **{
                    name: path.relative_to(dataset_root).as_posix()
                    for name, path in paths.items()
                },
            }
            for budget, paths in results.items()
        },
    }
    (dataset_root / "meta" / "navvla_context_index_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def build_context_rows(
    episode: ContextEpisode,
    *,
    spec: DatasetSpec,
    token_budget: int,
    frame_hashes: dict[tuple[str, int, str], str],
    camera_names: tuple[str, ...],
    use_bats: bool,
    use_hash_dedup: bool,
    dhash_threshold: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frames = list(episode.frames)
    num_cameras = len(camera_names)
    max_history = max(0, (token_budget - num_cameras * 65) // (num_cameras * 5))
    context_rows: list[dict[str, object]] = []
    debug_rows: list[dict[str, object]] = []

    for anchor_pos, frame in enumerate(frames):
        bats_k = bats_decay(
            anchor_pos, token_budget=token_budget, num_cameras=num_cameras
        ) if use_bats else 0.0
        selected, probabilities, distances = select_history(
            frames,
            anchor_pos,
            episode_id=episode.episode_id,
            dataset_name=spec.dataset_name,
            bats_k=bats_k,
            max_history=max_history,
            frame_hashes=frame_hashes,
            use_bats=use_bats,
            use_hash_dedup=use_hash_dedup,
            dhash_threshold=dhash_threshold,
        )
        steps, blocks, refs, masks = context_fields(
            selected,
            episode_id=episode.episode_id,
            camera_names=camera_names,
        )
        key = make_context_key(
            spec.dataset_name,
            episode.split,
            episode.episode_id,
            frame.frame_index,
            spec.context_policy_version,
        )
        context_rows.append(
            {
                "context.index_key": key,
                "index": frame.data_index,
                "current_tvi_time": frame.timestamp,
                "bats_k": bats_k,
                "history_steps": steps,
                "history_blocks": blocks,
                "history_token_refs": refs,
                "history_mask": masks,
                "long_memory_steps": [],
                "long_memory_blocks": [],
                "long_memory_token_refs": [],
                "long_memory_mask": [],
                "token_count_after": len(refs),
                "context_policy_version": spec.context_policy_version,
            }
        )
        debug_rows.append(
            {
                "index": frame.data_index,
                "context.index_key": key,
                "split": episode.split,
                "anchor_timestamp": frame.timestamp,
                "keep_probability": probabilities,
                "dhash_hamming_distance": distances,
                "token_count_before": anchor_pos * num_cameras,
                "bats_k": bats_k,
                "bats_expected_frames": float("nan"),
                "bats_target_frames": float(max_history),
                "bats_budget_tokens": token_budget,
                "bats_epsilon": 0.1,
                "bats_num_cameras": num_cameras,
                "bats_history_num_cameras": num_cameras,
                "bats_budget_feasible": token_budget >= num_cameras * 65,
            }
        )
    return context_rows, debug_rows


def select_history(
    frames: list[ContextFrame],
    anchor_pos: int,
    *,
    episode_id: str,
    dataset_name: str,
    bats_k: float,
    max_history: int,
    frame_hashes: dict[tuple[str, int, str], str],
    use_bats: bool,
    use_hash_dedup: bool,
    dhash_threshold: int,
) -> tuple[list[ContextFrame], list[float], list[int]]:
    if anchor_pos <= 0 or max_history <= 0:
        return [], [], []
    candidates = frames[:anchor_pos]
    if not use_bats:
        if not use_hash_dedup:
            selected = candidates[-max_history:]
            distances = [64 if index == 0 else 0 for index in range(len(selected))]
            return list(selected), [1.0] * len(selected), distances
        selected: list[ContextFrame] = []
        distances: list[int] = []
        last_hash: str | None = None
        for candidate in reversed(candidates):
            if len(selected) >= max_history:
                break
            distance, candidate_hash = _hash_distance(
                episode_id, candidate, last_hash, frame_hashes
            )
            if last_hash is not None and distance <= dhash_threshold:
                continue
            selected.append(candidate)
            distances.append(distance)
            last_hash = candidate_hash
        selected.reverse()
        distances.reverse()
        return selected, [1.0] * len(selected), distances

    anchor_frame = frames[anchor_pos]
    rng = random.Random(
        f"42:{dataset_name}:{episode_id}:{anchor_frame.frame_index}"
    )
    scored: list[tuple[float, int, float, ContextFrame]] = []
    denominator = max(1, anchor_frame.frame_index)
    for candidate in candidates:
        probability = (
            0.9
            * math.exp(
                bats_k
                * (candidate.frame_index - anchor_frame.frame_index)
                / denominator
            )
            + 0.1
        )
        scored.append((probability, candidate.frame_index, rng.random(), candidate))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    selected_with_scores: list[tuple[ContextFrame, float, int]] = []
    last_hash = None
    for probability, _frame_index, _jitter, candidate in scored:
        if len(selected_with_scores) >= max_history:
            break
        if use_hash_dedup:
            distance, candidate_hash = _hash_distance(
                episode_id, candidate, last_hash, frame_hashes
            )
            if last_hash is not None and distance <= dhash_threshold:
                continue
            last_hash = candidate_hash
        else:
            distance = 64 if not selected_with_scores else 0
        selected_with_scores.append((candidate, float(probability), distance))
    selected_with_scores.sort(key=lambda item: item[0].frame_index)
    return (
        [item[0] for item in selected_with_scores],
        [item[1] for item in selected_with_scores],
        [item[2] for item in selected_with_scores],
    )


def context_fields(
    frames: list[ContextFrame],
    *,
    episode_id: str,
    camera_names: tuple[str, ...],
) -> tuple[list[dict[str, float]], list[dict[str, object]], list[str], list[bool]]:
    steps: list[dict[str, float]] = []
    blocks: list[dict[str, object]] = []
    refs: list[str] = []
    masks: list[bool] = []
    for frame in frames:
        step_index = len(steps)
        steps.append({"timestamp": float(frame.timestamp)})
        for camera_name in camera_names:
            blocks.append({"step_index": step_index, "camera_name": camera_name})
            refs.append(f"{episode_id}/{frame.frame_index:06d}/{camera_name}")
            masks.append(True)
    return steps, blocks, refs, masks


def bats_decay(history_frames: int, *, token_budget: int, num_cameras: int) -> float:
    if history_frames <= 0:
        return 0.0
    if num_cameras <= 0:
        raise ValueError("num_cameras must be positive")
    target = max(
        0.0,
        (token_budget - num_cameras * 65) / (num_cameras * 5),
    )
    target = min(float(history_frames), target)
    floor = history_frames * 0.1
    if target <= floor:
        return 1.0e6
    if target >= history_frames:
        return 0.0

    def objective(value: float) -> float:
        expected = history_frames * (
            0.9 * (1.0 - math.exp(-value)) / value + 0.1
        )
        return expected - target

    lower = 1.0e-12
    upper = 1.0
    while objective(upper) > 0.0:
        upper *= 2.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if objective(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _hash_distance(
    episode_id: str,
    frame: ContextFrame,
    last_hash: str | None,
    frame_hashes: dict[tuple[str, int, str], str],
) -> tuple[int, str]:
    frame_hash = frame_hashes[(episode_id, frame.frame_index, "front")]
    distance = (
        64
        if last_hash is None
        else (int(frame_hash, 16) ^ int(last_hash, 16)).bit_count()
    )
    return distance, frame_hash


def _write_context_budget(
    episodes: Iterator[ContextEpisode],
    *,
    spec: DatasetSpec,
    token_budget: int,
    meta_path: Path,
    refs_path: Path,
    arrays_path: Path,
    debug_path: Path,
    frame_hashes: dict[tuple[str, int, str], str],
    camera_names: tuple[str, ...],
    use_bats: bool,
    use_hash_dedup: bool,
    dhash_threshold: int,
) -> None:
    np, pd, pa, pq = _context_dependencies()
    dtype_map = {name: np.dtype(dtype) for name, dtype in ARRAY_DTYPES.items()}
    ref_ids: dict[str, int] = {}
    ref_rows: list[dict[str, object]] = []
    camera_ids = {name: index for index, name in enumerate(camera_names)}
    offsets = {name: 0 for name in ARRAY_DTYPES}
    spool_dir = arrays_path.with_name(".context_arrays_spool")
    shutil.rmtree(spool_dir, ignore_errors=True)
    spool_dir.mkdir(parents=True, exist_ok=True)
    spool_files = {
        name: (spool_dir / f"{name}.bin").open("wb") for name in ARRAY_DTYPES
    }
    meta_writer = None
    debug_writer = None

    def get_ref_id(ref: str, camera_name: str) -> int:
        if ref not in ref_ids:
            parts = ref.rsplit("/", 2)
            value = len(ref_rows)
            ref_ids[ref] = value
            ref_rows.append(
                {
                    "ref_id": value,
                    "ref": ref,
                    "episode_key": parts[0],
                    "episode_id": parts[0],
                    "frame_index": int(parts[1]),
                    "camera_name": camera_name,
                }
            )
        return ref_ids[ref]

    try:
        for episode in episodes:
            context_rows, debug_rows = build_context_rows(
                episode,
                spec=spec,
                token_budget=token_budget,
                frame_hashes=frame_hashes,
                camera_names=camera_names,
                use_bats=use_bats,
                use_hash_dedup=use_hash_dedup,
                dhash_threshold=dhash_threshold,
            )
            meta_rows: list[dict[str, object]] = []
            batch_arrays: dict[str, list[object]] = {
                name: [] for name in ARRAY_DTYPES
            }
            for row in context_rows:
                meta_row: dict[str, object] = {
                    "context.index_key": row["context.index_key"],
                    "index": row["index"],
                    "current_tvi_time": row["current_tvi_time"],
                    "bats_k": row["bats_k"],
                    "token_count_after": row["token_count_after"],
                    "context_policy_version": row["context_policy_version"],
                }
                for prefix in ("history", "long_memory"):
                    steps = row[f"{prefix}_steps"]
                    blocks = row[f"{prefix}_blocks"]
                    refs = row[f"{prefix}_token_refs"]
                    masks = row[f"{prefix}_mask"]
                    step_key = f"{prefix}_step_timestamp"
                    block_key = f"{prefix}_block_step_index"
                    meta_row[f"{prefix}_step_offset"] = (
                        offsets[step_key] + len(batch_arrays[step_key])
                    )
                    meta_row[f"{prefix}_step_count"] = len(steps)
                    meta_row[f"{prefix}_block_offset"] = (
                        offsets[block_key] + len(batch_arrays[block_key])
                    )
                    meta_row[f"{prefix}_block_count"] = len(blocks)
                    batch_arrays[step_key].extend(
                        float(step["timestamp"]) for step in steps
                    )
                    for block, ref, mask in zip(blocks, refs, masks):
                        camera_name = str(block["camera_name"])
                        batch_arrays[f"{prefix}_block_step_index"].append(
                            int(block["step_index"])
                        )
                        batch_arrays[f"{prefix}_block_camera_id"].append(
                            camera_ids[camera_name]
                        )
                        batch_arrays[f"{prefix}_block_ref_id"].append(
                            get_ref_id(str(ref), camera_name)
                        )
                        batch_arrays[f"{prefix}_mask"].append(bool(mask))
                meta_rows.append(meta_row)

            if meta_rows:
                table = pa.Table.from_pandas(
                    pd.DataFrame(meta_rows, columns=CONTEXT_META_COLUMNS),
                    preserve_index=False,
                )
                if meta_writer is None:
                    meta_writer = pq.ParquetWriter(meta_path, table.schema)
                meta_writer.write_table(table, row_group_size=PARQUET_ROW_GROUP_SIZE)
            if debug_rows:
                table = pa.Table.from_pandas(
                    pd.DataFrame(debug_rows), preserve_index=False
                )
                if debug_writer is None:
                    debug_writer = pq.ParquetWriter(debug_path, table.schema)
                debug_writer.write_table(table, row_group_size=PARQUET_ROW_GROUP_SIZE)
            for name, values in batch_arrays.items():
                array = np.asarray(values, dtype=dtype_map[name])
                if array.size:
                    array.tofile(spool_files[name])
                offsets[name] += len(values)
    finally:
        if meta_writer is not None:
            meta_writer.close()
        if debug_writer is not None:
            debug_writer.close()
        for file in spool_files.values():
            file.close()

    if meta_writer is None:
        pd.DataFrame(columns=CONTEXT_META_COLUMNS).to_parquet(meta_path, index=False)
    if debug_writer is None:
        pd.DataFrame(columns=["index"]).to_parquet(debug_path, index=False)
    pd.DataFrame(
        ref_rows,
        columns=[
            "ref_id",
            "ref",
            "episode_key",
            "episode_id",
            "frame_index",
            "camera_name",
        ],
    ).to_parquet(refs_path, index=False, row_group_size=PARQUET_ROW_GROUP_SIZE)

    temporary_arrays = arrays_path.with_name(f"{arrays_path.name}.tmp")
    shutil.rmtree(temporary_arrays, ignore_errors=True)
    shutil.rmtree(arrays_path, ignore_errors=True)
    temporary_arrays.mkdir(parents=True)
    for name, dtype in dtype_map.items():
        values = np.fromfile(
            spool_dir / f"{name}.bin", dtype=dtype, count=offsets[name]
        )
        np.save(temporary_arrays / f"{name}.npy", values, allow_pickle=False)
    np.save(
        temporary_arrays / "camera_names.npy",
        np.asarray(camera_names, dtype=str),
        allow_pickle=False,
    )
    temporary_arrays.rename(arrays_path)
    shutil.rmtree(spool_dir, ignore_errors=True)


def _context_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot context conversion requires numpy, pandas and pyarrow. "
            "Install requirements-lerobot.txt."
        ) from exc
    return np, pd, pa, pq
