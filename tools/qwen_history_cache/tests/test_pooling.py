"""Tests for the fixed row-major Qwen visual pooling rule."""

import pytest
import torch
from qwen_history_cache.pooling import pool_main_image_embeddings


def test_pooling_is_fp32_and_row_major() -> None:
    grid = torch.arange(12 * 12, dtype=torch.float16).reshape(12, 12, 1)
    pooled = pool_main_image_embeddings(
        grid.reshape(144, 1),
        [1, 24, 24],
        spatial_merge_size=2,
        output_grid=(2, 2),
    )
    expected = torch.tensor(
        [
            grid[:6, :6].float().mean(),
            grid[:6, 6:].float().mean(),
            grid[6:, :6].float().mean(),
            grid[6:, 6:].float().mean(),
        ]
    ).reshape(4, 1)
    assert pooled.dtype == torch.float32
    torch.testing.assert_close(pooled, expected)


def test_pooling_rejects_nonfinite_and_wrong_geometry() -> None:
    values = torch.zeros(144, 8)
    values[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        pool_main_image_embeddings(values, [1, 24, 24], spatial_merge_size=2)
    with pytest.raises(ValueError, match="expected 144"):
        pool_main_image_embeddings(
            torch.zeros(143, 8), [1, 24, 24], spatial_merge_size=2
        )


def test_pooling_rejects_video_grid() -> None:
    with pytest.raises(ValueError, match="still images only"):
        pool_main_image_embeddings(
            torch.zeros(288, 8), [2, 24, 24], spatial_merge_size=2
        )
