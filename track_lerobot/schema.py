from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class CameraSpec:
    name: str
    video_key: str
    azimuth_rad: float
    viewpoint_type: str

    def __post_init__(self) -> None:
        if not self.name or not self.video_key:
            raise ValueError("camera name and video_key must be non-empty")


@dataclass(frozen=True)
class SourceEpisode:
    scene_id: str
    source_episode_id: str
    metadata_path: Path
    frame_info_path: Path
    video_paths: Mapping[str, Path]

    @property
    def episode_id(self) -> str:
        return f"{self.scene_id}_{self.source_episode_id}"


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    fps: float
    height: int
    width: int


@dataclass(frozen=True)
class ValidatedEpisode:
    source: SourceEpisode
    frame_count: int


@dataclass(frozen=True)
class ContextFrame:
    frame_index: int
    timestamp: float
    data_index: int

    def __post_init__(self) -> None:
        if self.frame_index < 0 or self.data_index < 0:
            raise ValueError("frame_index and data_index must be non-negative")


@dataclass(frozen=True)
class ContextEpisode:
    episode_id: str
    split: str
    frames: tuple[ContextFrame, ...]

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if not self.frames:
            raise ValueError("context episode must contain frames")


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    split: str
    fps: float
    dt: float
    action_horizon: int
    episodes_per_file: int
    files_per_chunk: int
    context_policy_version: str
    cache_policy_version: str

    def __post_init__(self) -> None:
        if not self.dataset_name or not self.split:
            raise ValueError("dataset_name and split must be non-empty")
        if self.fps <= 0 or self.dt <= 0:
            raise ValueError("fps and dt must be positive")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.episodes_per_file <= 0 or self.files_per_chunk <= 0:
            raise ValueError("shard sizes must be positive")
