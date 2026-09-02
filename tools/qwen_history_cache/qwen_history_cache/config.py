"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROFILE = "qwen3_vl_2b_pooled_history_4_mmap"


@dataclass(frozen=True)
class CacheConfig:
    """Validated configuration for the fixed Qwen3-VL cache profile."""

    dataset_root: Path
    checkpoint_path: Path
    worklist_dir: Path
    splits: tuple[str, ...]
    profile: str
    encoder_revision: str
    shard_size: int
    batch_size: int
    image_width: int
    image_height: int
    token_count: int
    hidden_dim: int
    output_grid: tuple[int, int]
    expected_grid_thw: tuple[int, int, int]
    expected_patch_size: int
    expected_temporal_patch_size: int
    expected_spatial_merge_size: int

    def split_root(self, split: str) -> Path:
        """Return and validate the root for one configured split."""
        normalized = split.upper()
        if normalized not in self.splits:
            raise ValueError(
                f"Split {split!r} is not configured; choose from {self.splits}"
            )
        return self.dataset_root / normalized

    def worklist_path(self, split: str) -> Path:
        """Return the canonical worklist path for a split."""
        normalized = split.upper()
        self.split_root(normalized)
        return self.worklist_dir / f"{normalized}.parquet"

    def building_root(self, split: str) -> Path:
        """Return the hidden, resumable profile build directory."""
        return (
            self.split_root(split)
            / "cache"
            / "visual_tokens"
            / f".{self.profile}.building"
        )

    def profile_root(self, split: str) -> Path:
        """Return the final, published profile directory."""
        return self.split_root(split) / "cache" / "visual_tokens" / self.profile


def _pair(value: Any, *, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two integers")
    result = (int(value[0]), int(value[1]))
    if min(result) <= 0:
        raise ValueError(f"{name} values must be positive")
    return result


def _triple(value: Any, *, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three integers")
    result = (int(value[0]), int(value[1]), int(value[2]))
    if min(result) <= 0:
        raise ValueError(f"{name} values must be positive")
    return result


def load_config(path: str | Path) -> CacheConfig:
    """Load a YAML configuration without deriving hidden values in code."""
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")

    image_size = _pair(raw.get("image_size"), name="image_size")
    output_grid = _pair(raw.get("output_grid"), name="output_grid")
    expected_grid = _triple(
        raw.get("expected_image_grid_thw"), name="expected_image_grid_thw"
    )
    splits = tuple(str(item).upper() for item in raw.get("splits", ()))
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("splits must be a non-empty list without duplicates")

    config = CacheConfig(
        dataset_root=Path(raw["dataset_root"]).expanduser().resolve(),
        checkpoint_path=Path(raw["checkpoint_path"]).expanduser().resolve(),
        worklist_dir=Path(raw["worklist_dir"]).expanduser().resolve(),
        splits=splits,
        profile=str(raw["profile"]),
        encoder_revision=str(raw.get("encoder_revision", "main")),
        shard_size=int(raw.get("shard_size", 8192)),
        batch_size=int(raw.get("batch_size", 8)),
        image_width=image_size[0],
        image_height=image_size[1],
        token_count=int(raw.get("token_count", 4)),
        hidden_dim=int(raw.get("hidden_dim", 2048)),
        output_grid=output_grid,
        expected_grid_thw=expected_grid,
        expected_patch_size=int(raw.get("expected_patch_size", 16)),
        expected_temporal_patch_size=int(raw.get("expected_temporal_patch_size", 2)),
        expected_spatial_merge_size=int(raw.get("expected_spatial_merge_size", 2)),
    )
    if config.shard_size <= 0 or config.batch_size <= 0:
        raise ValueError("shard_size and batch_size must be positive")
    if config.output_grid[0] * config.output_grid[1] != config.token_count:
        raise ValueError("output_grid area must equal token_count")
    if not config.profile or "/" in config.profile:
        raise ValueError("profile must be one non-empty path component")
    if not config.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {config.dataset_root}")
    if not config.checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {config.checkpoint_path}"
        )
    fixed_contract = {
        "profile": (config.profile, PROFILE),
        "encoder_revision": (config.encoder_revision, "main"),
        "image_size": (
            (config.image_width, config.image_height),
            (384, 384),
        ),
        "output_grid": (config.output_grid, (2, 2)),
        "token_count": (config.token_count, 4),
        "hidden_dim": (config.hidden_dim, 2048),
        "expected_image_grid_thw": (
            config.expected_grid_thw,
            (1, 24, 24),
        ),
        "expected_patch_size": (config.expected_patch_size, 16),
        "expected_temporal_patch_size": (
            config.expected_temporal_patch_size,
            2,
        ),
        "expected_spatial_merge_size": (
            config.expected_spatial_merge_size,
            2,
        ),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in fixed_contract.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(
            f"Configuration does not match the fixed {PROFILE} contract: {mismatches}"
        )
    return config
