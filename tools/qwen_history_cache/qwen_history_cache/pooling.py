"""Spatial pooling for Qwen3-VL main visual-merger embeddings."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def pool_main_image_embeddings(
    image_embeds: torch.Tensor,
    image_grid_thw: torch.Tensor | Sequence[int],
    *,
    spatial_merge_size: int,
    output_grid: tuple[int, int] = (2, 2),
) -> torch.Tensor:
    """Pool one image's merger output to row-major spatial memory tokens.

    Args:
        image_embeds: Main merger output with shape ``[tokens, hidden_dim]``.
        image_grid_thw: Pre-merger Qwen grid ``[temporal, height, width]``.
        spatial_merge_size: Spatial merge factor from the checkpoint config.
        output_grid: Requested pooled height and width.

    Returns:
        FP32 tensor with shape ``[output_height * output_width, hidden_dim]``.
    """
    if image_embeds.ndim != 2:
        raise ValueError(
            f"image_embeds must have shape [tokens, hidden_dim], got "
            f"{tuple(image_embeds.shape)}"
        )
    grid = torch.as_tensor(image_grid_thw, dtype=torch.long).reshape(-1)
    if grid.numel() != 3:
        raise ValueError("image_grid_thw must contain [temporal, height, width]")
    temporal, height, width = (int(item) for item in grid.tolist())
    if temporal != 1:
        raise ValueError(
            f"History cache accepts still images only; temporal grid is {temporal}"
        )
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive")
    if height % spatial_merge_size or width % spatial_merge_size:
        raise ValueError(
            f"Grid {(height, width)} is not divisible by merge size "
            f"{spatial_merge_size}"
        )
    merged_height = height // spatial_merge_size
    merged_width = width // spatial_merge_size
    expected_tokens = temporal * merged_height * merged_width
    if image_embeds.shape[0] != expected_tokens:
        raise ValueError(
            f"Merger output has {image_embeds.shape[0]} tokens, expected "
            f"{expected_tokens} from grid {tuple(grid.tolist())}"
        )
    if not torch.isfinite(image_embeds).all():
        raise ValueError("Qwen main image embeddings contain NaN or Inf")

    spatial = image_embeds.reshape(
        temporal, merged_height, merged_width, image_embeds.shape[-1]
    )[0]
    channels_first = spatial.permute(2, 0, 1).unsqueeze(0).float()
    pooled = F.adaptive_avg_pool2d(channels_first, output_grid)
    row_major = (
        pooled.squeeze(0)
        .permute(1, 2, 0)
        .reshape(output_grid[0] * output_grid[1], image_embeds.shape[-1])
    )
    if not torch.isfinite(row_major).all():
        raise ValueError("Pooled Qwen history tokens contain NaN or Inf")
    return row_major
