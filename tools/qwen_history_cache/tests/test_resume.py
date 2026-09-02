"""Tests for immutable run specs and final profile publication."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from qwen_history_cache.manifest import (
    create_build_spec,
    publish_profile,
    validate_profile,
)
from qwen_history_cache.metadata import write_worklist
from qwen_history_cache.shard_writer import AtomicShardWriter, ensure_build_spec


def _worklist() -> pa.Table:
    return pa.table(
        {
            "ref_id": [0, 1],
            "ref": ["episode/000000/front", "episode/000001/front"],
            "episode_id": ["episode", "episode"],
            "frame_index": [0, 1],
            "camera_name": ["front", "front"],
            "data_index": [0, 1],
            "video_key": ["front_image", "front_image"],
            "video_frame_index": [0, 1],
            "chunk_index": [0, 0],
            "file_index": [0, 0],
            "video_path": [
                "videos/front_image/chunk-000/part-000.mp4",
                "videos/front_image/chunk-000/part-000.mp4",
            ],
        }
    )


def test_build_spec_is_immutable(tmp_path: Path) -> None:
    building = tmp_path / ".profile.building"
    ensure_build_spec(building, {"run_fingerprint": "a"}, rank=0)
    ensure_build_spec(building, {"run_fingerprint": "a"}, rank=0)
    with pytest.raises(ValueError, match="different run fingerprint"):
        ensure_build_spec(building, {"run_fingerprint": "b"}, rank=0)


def test_complete_build_publishes_rlinf_contract(tmp_path: Path) -> None:
    building = tmp_path / ".profile.building"
    final = tmp_path / "profile"
    worklist_path = tmp_path / "worklist.parquet"
    digest = write_worklist(_worklist(), worklist_path)
    spec = create_build_spec(
        split="DT",
        split_root=tmp_path / "DT",
        worklist_path=worklist_path,
        worklist_sha256=digest,
        row_count=2,
        checkpoint={"checkpoint_path": "/checkpoint", "aggregate_sha256": "x"},
        processor={"image_grid_thw": [1, 24, 24]},
        profile="profile",
        encoder_revision="main",
        shard_size=2,
        token_count=4,
        hidden_dim=2048,
        output_grid=(2, 2),
    )
    ensure_build_spec(building, spec, rank=0)
    writer = AtomicShardWriter(
        building, 0, rows=2, token_count=4, hidden_dim=2048
    ).open()
    values = np.arange(2 * 4 * 2048, dtype=np.float16).reshape(2, 4, 2048)
    writer.write(values)
    writer.finalize()
    report = publish_profile(
        building,
        final,
        build_spec=spec,
        worklist_path=worklist_path,
        runtime={"test": True},
    )
    assert report["rows"] == 2
    assert final.is_dir()
    assert not building.exists()
    mmap = np.load(final / "shards/image_embeds_000000.npy", mmap_mode="r")
    np.testing.assert_array_equal(mmap, values)
    assert validate_profile(final, worklist_path=worklist_path)["shards"] == 1


def test_publish_rejects_missing_shard(tmp_path: Path) -> None:
    worklist_path = tmp_path / "worklist.parquet"
    digest = write_worklist(_worklist(), worklist_path)
    spec = create_build_spec(
        split="DT",
        split_root=tmp_path / "DT",
        worklist_path=worklist_path,
        worklist_sha256=digest,
        row_count=2,
        checkpoint={},
        processor={},
        profile="profile",
        encoder_revision="main",
        shard_size=2,
        token_count=4,
        hidden_dim=2048,
        output_grid=(2, 2),
    )
    building = tmp_path / ".profile.building"
    ensure_build_spec(building, spec, rank=0)
    with pytest.raises(RuntimeError, match="incomplete"):
        publish_profile(
            building,
            tmp_path / "profile",
            build_spec=spec,
            worklist_path=worklist_path,
            runtime={},
        )
