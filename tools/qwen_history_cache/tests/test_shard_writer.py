"""Tests for mmap shard atomicity and validation."""

from pathlib import Path

import numpy as np
import pytest
from qwen_history_cache.shard_writer import AtomicShardWriter


def test_atomic_shard_publish_and_reopen(tmp_path: Path) -> None:
    writer = AtomicShardWriter(
        tmp_path / ".profile.building",
        0,
        rows=3,
        token_count=4,
        hidden_dim=8,
    ).open()
    values = np.arange(3 * 4 * 8, dtype=np.float16).reshape(3, 4, 8)
    writer.write(values[:2])
    writer.write(values[2:])
    metadata = writer.finalize()
    assert writer.final_path.is_file()
    assert not writer.partial_path.exists()
    assert metadata["sha256"]
    np.testing.assert_array_equal(
        np.load(writer.final_path, mmap_mode="r", allow_pickle=False), values
    )
    assert writer.completed_metadata() == metadata


def test_incomplete_shard_is_not_published(tmp_path: Path) -> None:
    writer = AtomicShardWriter(
        tmp_path / ".profile.building",
        0,
        rows=2,
        token_count=4,
        hidden_dim=8,
    ).open()
    writer.write(np.zeros((1, 4, 8), dtype=np.float16))
    with pytest.raises(ValueError, match="incomplete"):
        writer.finalize()
    writer.close()
    assert writer.partial_path.exists()
    assert not writer.final_path.exists()


def test_resume_discards_only_partial_shard(tmp_path: Path) -> None:
    root = tmp_path / ".profile.building"
    first = AtomicShardWriter(root, 0, rows=2, token_count=4, hidden_dim=8).open()
    first.write(np.ones((1, 4, 8), dtype=np.float16))
    first.close()
    resumed = AtomicShardWriter(root, 0, rows=2, token_count=4, hidden_dim=8).open()
    resumed.write(np.full((2, 4, 8), 2, dtype=np.float16))
    resumed.finalize()
    assert np.all(np.load(resumed.final_path) == 2)
