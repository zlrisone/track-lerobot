from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evt_context_index.rebuild import (
    BudgetSpec,
    FrameRecord,
    inspect_dataset,
    rebuild_dataset,
    select_recent_history,
    summary_json,
    validate_dataset,
)


def _make_dataset(root: Path) -> Path:
    split_root = root / "AT"
    checkpoints = split_root / "meta" / "checkpoints"
    checkpoints.mkdir(parents=True)
    info = {
        "dataset_name": "AT",
        "total_frames": 7,
        "splits": {"train": "0:2"},
        "navvla": {"context_policy_version": "lerobot_sliding_short_long_v1"},
    }
    (split_root / "meta" / "info.json").write_text(json.dumps(info))
    (split_root / "meta" / "navvla_schema_ext.json").write_text(
        json.dumps({"context_policy_version": "lerobot_sliding_short_long_v1"})
    )
    (split_root / "meta" / "navvla_context_index_manifest.json").write_text(
        json.dumps({"split": "train"})
    )
    rows = []
    data_index = 0
    for episode_index, (episode_id, length) in enumerate((("ep-a", 5), ("ep-b", 2))):
        for frame_index in range(length):
            rows.append(
                {
                    "episode_index": episode_index,
                    "episode_id": episode_id,
                    "frame_index": frame_index,
                    "timestamp": frame_index / 10,
                    "index": data_index,
                }
            )
            data_index += 1
    pq.write_table(
        pa.Table.from_pylist(rows),
        checkpoints / "chunk-000_part-000_frame_records.parquet",
    )
    return split_root


def test_no_tvi_budget_has_240_history_blocks() -> None:
    budget = BudgetSpec()
    assert budget.max_history_blocks == 240
    assert budget.total_tokens(240) == 1024


def test_sliding_history_is_recent_and_chronological() -> None:
    frames = [FrameRecord(0, "ep", i, i / 10, i) for i in range(6)]
    selected = select_recent_history(frames, anchor_pos=5, max_history_blocks=3)
    assert [frame.frame_index for frame in selected] == [2, 3, 4]


def test_build_and_validate_compact_index(tmp_path: Path) -> None:
    split_root = _make_dataset(tmp_path)
    budget = BudgetSpec(
        token_budget=72, current_visual_tokens=64, history_visual_tokens=4
    )

    inspection = inspect_dataset(split_root, budget)
    assert inspection.max_history_blocks == 2
    assert inspection.frames == 7
    assert inspection.history_blocks == 8
    assert inspection.refs == 5

    summary = rebuild_dataset(split_root, budget)
    assert summary.history_blocks == 8
    assert summary.refs == 5
    result = validate_dataset(split_root, budget)
    assert result["frames"] == 7

    context_dir = split_root / "meta" / "context_index" / "budget_72"
    meta = pq.read_table(context_dir / "context_meta.parquet").to_pylist()
    assert [row["history_block_count"] for row in meta] == [0, 1, 2, 2, 2, 0, 1]
    assert [row["token_count_after"] for row in meta] == [64, 68, 72, 72, 72, 64, 68]
    assert all(row["long_memory_block_count"] == 0 for row in meta)

    refs = pq.read_table(context_dir / "refs.parquet").to_pylist()
    assert [row["ref"] for row in refs] == [
        "ep-a/000000/front",
        "ep-a/000001/front",
        "ep-a/000002/front",
        "ep-a/000003/front",
        "ep-b/000000/front",
    ]
    arrays = context_dir / "context_arrays"
    assert np.load(arrays / "history_mask.npy").shape == (8,)
    assert np.load(arrays / "long_memory_mask.npy").shape == (0,)
    assert np.load(arrays / "camera_names.npy").tolist() == ["front"]

    manifest = json.loads(
        (split_root / "meta" / "navvla_context_index_manifest.json").read_text()
    )
    assert manifest["budget_model"]["tvi_tokens"] == 0
    assert manifest["budget_model"]["max_history_blocks"] == 2


def test_existing_index_requires_explicit_replace(tmp_path: Path) -> None:
    split_root = _make_dataset(tmp_path)
    budget = BudgetSpec(token_budget=72)
    rebuild_dataset(split_root, budget)
    with pytest.raises(FileExistsError, match="--replace"):
        rebuild_dataset(split_root, budget)

    replacement = rebuild_dataset(split_root, budget, replace=True)
    backup = Path(replacement.backup_dir)
    assert (backup / "context_index" / "context_meta.parquet").is_file()
    assert (backup / "debug.parquet").is_file()
    validate_dataset(split_root, budget)


def test_summary_json_supports_dataclass_collections(tmp_path: Path) -> None:
    split_root = _make_dataset(tmp_path)
    inspection = inspect_dataset(split_root)
    payload = json.loads(summary_json([inspection]))
    assert payload[0]["frames"] == 7
