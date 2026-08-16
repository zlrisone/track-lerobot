from __future__ import annotations

import json
import math
import shutil
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from .context_index import make_context_key, normalize_context_cameras, write_context_indexes
from .schema import (
    CameraSpec,
    ContextEpisode,
    ContextFrame,
    DatasetSpec,
    SourceEpisode,
    ValidatedEpisode,
    VideoInfo,
)


DATA_PATH_PATTERN = "data/chunk-{chunk_index:03d}/part-{file_index:03d}.parquet"
CONTEXT_POLICY_VERSION = "lerobot_sliding_short_long_v1"
CACHE_POLICY_VERSION = "track-four-view-v1"
CONVERSION_CONFIG_NAME = "track_lerobot_conversion.json"
DEFAULT_CONTEXT_CAMERAS = ("front",)
PARQUET_ROW_GROUP_SIZE = 131_072

CAMERAS: dict[str, dict[str, Any]] = {
    "front": {"video_key": "front_image", "azimuth_rad": 0.0},
    "left": {"video_key": "left_image", "azimuth_rad": math.pi / 2.0},
    "right": {"video_key": "right_image", "azimuth_rad": -math.pi / 2.0},
    "rear": {"video_key": "rear_image", "azimuth_rad": math.pi},
}


class MultiCameraVideoWriter(AbstractContextManager["MultiCameraVideoWriter"]):
    def __init__(
        self,
        paths: Mapping[str, Path],
        *,
        fps: float,
        shapes: Mapping[str, tuple[int, int]],
    ) -> None:
        cv2, _np, _pd, _pa, _pq, _tqdm = _dependencies()
        self._writers: dict[str, Any] = {}
        for camera_name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            height, width = shapes[camera_name]
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(fps),
                (int(width), int(height)),
            )
            if not writer.isOpened():
                self.close()
                raise RuntimeError(f"unable to create video: {path}")
            self._writers[camera_name] = writer

    def write(self, frames: Mapping[str, Any]) -> None:
        for camera_name, writer in self._writers.items():
            writer.write(frames[camera_name])

    def close(self) -> None:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def discover_episodes(
    source_root: str | Path,
    *,
    max_episodes: int | None = None,
    skip_invalid: bool = False,
) -> tuple[list[SourceEpisode], list[dict[str, str]]]:
    root = Path(source_root)
    if not root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {root}")
    episodes: list[SourceEpisode] = []
    rejected: list[dict[str, str]] = []
    info_paths = sorted(
        root.glob("*/*_info.json"),
        key=lambda path: (
            path.parent.name,
            _natural_key(path.stem.removesuffix("_info")),
        ),
    )
    for info_path in info_paths:
        scene_id = info_path.parent.name
        source_episode_id = info_path.stem.removesuffix("_info")
        metadata_path = info_path.with_name(f"{source_episode_id}.json")
        video_paths = {
            camera_name: info_path.with_name(
                f"{source_episode_id}_{camera_name}.mp4"
            )
            for camera_name in CAMERAS
        }
        missing = [
            str(path)
            for path in (metadata_path, *video_paths.values())
            if not path.is_file()
        ]
        if missing:
            reason = "missing files: " + ", ".join(missing)
            rejected.append(
                {
                    "scene_id": scene_id,
                    "episode_id": source_episode_id,
                    "reason": reason,
                }
            )
            if not skip_invalid:
                raise FileNotFoundError(reason)
            continue
        episodes.append(
            SourceEpisode(
                scene_id=scene_id,
                source_episode_id=source_episode_id,
                metadata_path=metadata_path,
                frame_info_path=info_path,
                video_paths=video_paths,
            )
        )
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
    return episodes, rejected


def build_action_chunk(
    frames: list[dict[str, Any]],
    start: int,
    *,
    horizon: int = 8,
    dt: float = 0.1,
) -> list[list[float]]:
    """Integrate future normalized body velocities into [x, y, z, yaw]."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    values = [[0.0, 0.0, 0.0, 0.0] for _ in range(horizon)]
    x = 0.0
    y = 0.0
    yaw = 0.0
    valid_count = 0
    available = min(horizon, max(0, len(frames) - start))
    for offset in range(available):
        raw_velocity = frames[start + offset].get("base_velocity", [])
        if not isinstance(raw_velocity, (list, tuple)) or len(raw_velocity) < 3:
            break
        try:
            source_vx, source_vy, source_omega = (
                float(raw_velocity[0]),
                float(raw_velocity[1]),
                float(raw_velocity[2]),
            )
        except (TypeError, ValueError):
            break
        if not all(math.isfinite(value) for value in (source_vx, source_vy, source_omega)):
            break
        vx, vy, omega = _convert_action_axes(
            source_vx, source_vy, source_omega
        )
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x += (cos_yaw * vx - sin_yaw * vy) * dt
        y += (sin_yaw * vx + cos_yaw * vy) * dt
        yaw += omega * dt
        values[offset] = [x, y, 0.0, yaw]
        valid_count += 1
    if 0 < valid_count < horizon:
        last_pose = list(values[valid_count - 1])
        for index in range(valid_count, horizon):
            values[index] = list(last_pose)
    return values


def convert_rollout_dataset(
    *,
    source_root: str | Path,
    output_root: str | Path,
    dataset_name: str = "track",
    split: str = "train",
    fps: float | None = None,
    dt: float = 0.1,
    action_horizon: int = 8,
    episodes_per_file: int = 20,
    files_per_chunk: int = 50,
    max_episodes: int | None = None,
    overwrite: bool = False,
    resume: bool = False,
    skip_invalid: bool = False,
    show_progress: bool = True,
    context_token_budgets: tuple[int, ...] = (1024,),
    context_camera_names: tuple[str, ...] = DEFAULT_CONTEXT_CAMERAS,
    use_bats: bool = False,
    use_hash_dedup: bool = False,
    dhash_threshold: int = 10,
    build_context_index: bool = True,
) -> dict[str, Any]:
    _dependencies()
    source_root = Path(source_root)
    output_root = Path(output_root)
    _validate_dataset_name(dataset_name)
    if not split:
        raise ValueError("split must be non-empty")
    if action_horizon <= 0 or dt <= 0:
        raise ValueError("action_horizon and dt must be positive")
    if episodes_per_file <= 0 or files_per_chunk <= 0:
        raise ValueError("episodes_per_file and files_per_chunk must be positive")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if dhash_threshold < 0:
        raise ValueError("dhash_threshold must be non-negative")
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    context_camera_names = normalize_context_cameras(
        tuple(context_camera_names), tuple(CAMERAS)
    )
    budgets = tuple(dict.fromkeys(int(value) for value in context_token_budgets))
    if build_context_index and (not budgets or any(value <= 0 for value in budgets)):
        raise ValueError("context token budgets must contain positive integers")

    episodes, rejected = discover_episodes(
        source_root,
        max_episodes=max_episodes,
        skip_invalid=skip_invalid,
    )
    if not episodes:
        raise FileNotFoundError(
            f"no complete four-view rollout episodes found under {source_root}"
        )
    first_metadata = _read_json(episodes[0].metadata_path)
    if not isinstance(first_metadata, dict):
        raise ValueError(f"episode metadata must be an object: {episodes[0].metadata_path}")
    resolved_fps = float(
        fps if fps is not None else first_metadata.get("fps", 1.0 / dt)
    )
    if resolved_fps <= 0 or not math.isfinite(resolved_fps):
        raise ValueError(f"fps must be positive and finite: {resolved_fps}")

    dataset_root = output_root / dataset_name
    conversion_config = {
        "source_root": str(source_root.resolve()),
        "dataset_name": dataset_name,
        "split": split,
        "fps": resolved_fps,
        "dt": dt,
        "action_horizon": action_horizon,
        "episodes_per_file": episodes_per_file,
        "files_per_chunk": files_per_chunk,
        "max_episodes": max_episodes,
        "context_token_budgets": list(budgets),
        "context_camera_names": list(context_camera_names),
        "context_policy_version": CONTEXT_POLICY_VERSION,
        "cache_policy_version": CACHE_POLICY_VERSION,
        "use_bats": bool(use_bats),
        "use_hash_dedup": bool(use_hash_dedup),
        "dhash_threshold": int(dhash_threshold),
        "build_context_index": bool(build_context_index),
    }
    if dataset_root.exists():
        if overwrite:
            shutil.rmtree(dataset_root)
        elif resume:
            _validate_resume_config(dataset_root, conversion_config)
        else:
            raise FileExistsError(
                f"dataset already exists: {dataset_root}; use --resume or --overwrite"
            )
    _make_output_dirs(dataset_root)
    config_path = dataset_root / "meta" / CONVERSION_CONFIG_NAME
    if not config_path.exists():
        _write_json(config_path, conversion_config)

    validated, camera_shapes = _validate_episodes(
        episodes,
        rejected=rejected,
        fps=resolved_fps,
        skip_invalid=skip_invalid,
        show_progress=show_progress,
    )
    if not validated:
        raise ValueError("no episode passed metadata and media validation")

    episode_offsets = _episode_offsets(validated)
    shard_groups = _group_episodes(
        validated,
        episodes_per_file=episodes_per_file,
        files_per_chunk=files_per_chunk,
    )
    _cv2, _np, _pd, _pa, _pq, tqdm = _dependencies()
    skipped_shards = 0
    for (chunk_index, file_index), shard_episodes in tqdm(
        shard_groups,
        desc="convert shards",
        unit="shard",
        disable=not show_progress,
    ):
        expected_frames = sum(item.frame_count for _index, item in shard_episodes)
        status = _shard_status(
            dataset_root,
            chunk_index=chunk_index,
            file_index=file_index,
            expected_frames=expected_frames,
        )
        if resume and status == "complete":
            skipped_shards += 1
            continue
        if status != "missing":
            _clear_shard_outputs(
                dataset_root,
                chunk_index=chunk_index,
                file_index=file_index,
            )
        _convert_shard(
            dataset_root,
            shard_episodes=shard_episodes,
            episode_offsets=episode_offsets,
            chunk_index=chunk_index,
            file_index=file_index,
            fps=resolved_fps,
            dt=dt,
            action_horizon=action_horizon,
            split=split,
            dataset_name=dataset_name,
            camera_shapes=camera_shapes,
        )

    _merge_checkpoint_metadata(dataset_root)
    total_frames = sum(item.frame_count for item in validated)
    total_shards = len(shard_groups)
    spec = DatasetSpec(
        dataset_name=dataset_name,
        split=split,
        fps=resolved_fps,
        dt=dt,
        action_horizon=action_horizon,
        episodes_per_file=episodes_per_file,
        files_per_chunk=files_per_chunk,
        context_policy_version=CONTEXT_POLICY_VERSION,
        cache_policy_version=CACHE_POLICY_VERSION,
    )
    _write_camera_metadata(dataset_root)
    _write_modality_metadata(dataset_root, action_horizon=action_horizon)
    _write_info_metadata(
        dataset_root,
        dataset_name=dataset_name,
        split=split,
        fps=resolved_fps,
        dt=dt,
        action_horizon=action_horizon,
        episodes_per_file=episodes_per_file,
        files_per_chunk=files_per_chunk,
        total_episodes=len(validated),
        total_frames=total_frames,
        total_videos=total_shards * len(CAMERAS),
        camera_shapes=camera_shapes,
    )
    _write_schema_extension(dataset_root, spec=spec, has_context=build_context_index)
    _write_statistics(
        dataset_root,
        dataset_name=dataset_name,
        action_horizon=action_horizon,
        total_frames=total_frames,
        total_episodes=len(validated),
    )

    checkpoint_paths = _complete_checkpoints(dataset_root)
    if build_context_index:
        context_episode_factory = _make_context_episode_factory(
            checkpoint_paths, split=split
        )
        frame_hashes = (
            _load_frame_hashes(checkpoint_paths) if use_hash_dedup else {}
        )
        write_context_indexes(
            context_episode_factory,
            spec=spec,
            dataset_root=dataset_root,
            frame_hashes=frame_hashes,
            token_budgets=budgets,
            camera_names=context_camera_names,
            valid_camera_names=tuple(CAMERAS),
            use_bats=use_bats,
            use_hash_dedup=use_hash_dedup,
            dhash_threshold=dhash_threshold,
        )

    summary = {
        "dataset_root": str(dataset_root),
        "source_root": str(source_root),
        "dataset_name": dataset_name,
        "split": split,
        "fps": resolved_fps,
        "dt": dt,
        "action_horizon": action_horizon,
        "total_episodes": len(validated),
        "total_frames": total_frames,
        "total_videos": total_shards * len(CAMERAS),
        "total_shards": total_shards,
        "skipped_shards": skipped_shards,
        "rejected_episodes": len(rejected),
        "rejections": rejected,
        "context_index_built": bool(build_context_index),
        "context_token_budgets": list(budgets) if build_context_index else [],
        "context_camera_names": list(context_camera_names),
        "use_bats": bool(use_bats),
        "use_hash_dedup": bool(use_hash_dedup),
        "dhash_threshold": int(dhash_threshold),
    }
    _write_json(dataset_root / "conversion_report.json", summary)
    return summary


def _validate_episodes(
    episodes: list[SourceEpisode],
    *,
    rejected: list[dict[str, str]],
    fps: float,
    skip_invalid: bool,
    show_progress: bool,
) -> tuple[list[ValidatedEpisode], dict[str, tuple[int, int]]]:
    _cv2, _np, _pd, _pa, _pq, tqdm = _dependencies()
    validated: list[ValidatedEpisode] = []
    common_shapes: dict[str, tuple[int, int]] | None = None
    for episode in tqdm(
        episodes,
        desc="validate episodes",
        unit="episode",
        disable=not show_progress,
    ):
        try:
            frames = _read_json(episode.frame_info_path)
            if not isinstance(frames, list) or not frames:
                raise ValueError(
                    f"frame info must be a non-empty list: {episode.frame_info_path}"
                )
            metadata = _read_json(episode.metadata_path)
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"episode metadata must be an object: {episode.metadata_path}"
                )
            video_info = {
                name: _probe_video(path) for name, path in episode.video_paths.items()
            }
            _validate_episode_media(episode, frames, video_info, fps)
            episode_shapes = {
                name: (info.height, info.width) for name, info in video_info.items()
            }
            if common_shapes is None:
                common_shapes = episode_shapes
            elif episode_shapes != common_shapes:
                raise ValueError(
                    f"video dimensions differ from previous episodes: "
                    f"{episode_shapes} != {common_shapes}"
                )
            validated.append(
                ValidatedEpisode(source=episode, frame_count=len(frames))
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            rejected.append(
                {
                    "scene_id": episode.scene_id,
                    "episode_id": episode.source_episode_id,
                    "reason": str(exc),
                }
            )
            if not skip_invalid:
                raise
    if common_shapes is None:
        raise ValueError("unable to determine camera dimensions")
    return validated, common_shapes


def _convert_shard(
    dataset_root: Path,
    *,
    shard_episodes: list[tuple[int, ValidatedEpisode]],
    episode_offsets: dict[int, int],
    chunk_index: int,
    file_index: int,
    fps: float,
    dt: float,
    action_horizon: int,
    split: str,
    dataset_name: str,
    camera_shapes: dict[str, tuple[int, int]],
) -> None:
    cv2, _np, pd, _pa, _pq, _tqdm = _dependencies()
    output_videos = _shard_video_paths(
        dataset_root, chunk_index=chunk_index, file_index=file_index
    )
    data_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    video_index_rows: list[dict[str, Any]] = []
    frame_record_rows: list[dict[str, Any]] = []
    task_lines: list[str] = []
    frame_metadata_path = _checkpoint_artifact_path(
        dataset_root,
        chunk_index=chunk_index,
        file_index=file_index,
        suffix="frame_metadata.jsonl",
    )
    frame_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    shard_frame_index = 0

    try:
        with frame_metadata_path.open(
            "w", encoding="utf-8"
        ) as frame_metadata_file, MultiCameraVideoWriter(
            output_videos, fps=fps, shapes=camera_shapes
        ) as video_writer:
            for episode_index, descriptor in shard_episodes:
                episode = descriptor.source
                metadata = _read_json(episode.metadata_path)
                frames = _read_json(episode.frame_info_path)
                if len(frames) != descriptor.frame_count:
                    raise ValueError(
                        f"frame count changed after validation: {episode.frame_info_path}"
                    )
                instruction = str(
                    metadata.get("instruction")
                    or "Track and follow the target person."
                ).strip()
                task = {
                    "task_index": episode_index,
                    "task": instruction,
                    "task_type": str(metadata.get("task_type") or "tracking"),
                    "task_subtype": str(
                        metadata.get("task_subtype") or "human_following"
                    ),
                    "platform_text": (
                        "Platform: Habitat Spot mobile robot with front, left, "
                        "right and rear cameras. "
                        f"Action: {action_horizon} local waypoints [x, y, z, yaw], "
                        "with z fixed to 0."
                    ),
                    "dataset_source": dataset_name,
                    "scene_id": episode.scene_id,
                    "answer": None,
                }
                task_lines.append(json.dumps(task, ensure_ascii=False))
                episode_video_start = shard_frame_index
                captures = {
                    name: cv2.VideoCapture(str(path))
                    for name, path in episode.video_paths.items()
                }
                if not all(capture.isOpened() for capture in captures.values()):
                    for capture in captures.values():
                        capture.release()
                    raise ValueError(
                        f"unable to open all camera videos for {episode.episode_id}"
                    )
                try:
                    for frame_pos, frame in enumerate(frames):
                        images = _read_synchronized_frames(
                            captures, episode=episode, frame_pos=frame_pos
                        )
                        video_writer.write(images)
                        frame_index = int(frame.get("frame_index", frame_pos))
                        timestamp = float(
                            frame.get("timestamp", frame_index / fps)
                        )
                        global_index = episode_offsets[episode_index] + frame_pos
                        context_key = make_context_key(
                            dataset_name,
                            split,
                            episode.episode_id,
                            frame_index,
                            CONTEXT_POLICY_VERSION,
                        )
                        action = build_action_chunk(
                            frames,
                            frame_pos,
                            horizon=action_horizon,
                            dt=dt,
                        )
                        data_rows.append(
                            {
                                "episode_index": episode_index,
                                "frame_index": frame_index,
                                "timestamp": timestamp,
                                "task_index": episode_index,
                                "observation.state": _state_from_frame(frame),
                                "action": action,
                                "action.padding_mask": [False] * action_horizon,
                                "next.done": frame_pos == len(frames) - 1,
                                "sample.action_available": True,
                                "context.index_key": context_key,
                                "source_frame_index": frame_index,
                                "index": global_index,
                            }
                        )
                        for camera_name, camera in CAMERAS.items():
                            video_index_rows.append(
                                {
                                    "index": global_index,
                                    "video_key": camera["video_key"],
                                    "camera_name": camera_name,
                                    "available": True,
                                    "video_frame_index": shard_frame_index,
                                    "chunk_index": chunk_index,
                                    "file_index": file_index,
                                }
                            )
                        front_hash = _dhash(images["front"])
                        frame_record_rows.append(
                            {
                                "episode_index": episode_index,
                                "episode_id": episode.episode_id,
                                "frame_index": frame_index,
                                "timestamp": timestamp,
                                "index": global_index,
                                "front_dhash": front_hash,
                            }
                        )
                        frame_metadata_file.write(
                            json.dumps(
                                {
                                    "index": global_index,
                                    "source_frame_index": frame_index,
                                    "source_metadata": {
                                        **frame,
                                        "source_scene_id": episode.scene_id,
                                        "source_episode_id": episode.source_episode_id,
                                        "metadata_path": str(episode.metadata_path),
                                        "info_path": str(episode.frame_info_path),
                                        "video_paths": {
                                            name: str(path)
                                            for name, path in episode.video_paths.items()
                                        },
                                    },
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        shard_frame_index += 1
                finally:
                    for capture in captures.values():
                        capture.release()

                episode_row: dict[str, Any] = {
                    "episode_index": episode_index,
                    "episode_id": episode.episode_id,
                    "trajectory_id": episode.episode_id,
                    "task_index": episode_index,
                    "split": split,
                    "scene_id": episode.scene_id,
                    "tasks": [instruction],
                    "length": len(frames),
                    "data/chunk_index": chunk_index,
                    "data/file_index": file_index,
                }
                for camera in CAMERAS.values():
                    video_key = str(camera["video_key"])
                    episode_row[f"videos/{video_key}/chunk_index"] = chunk_index
                    episode_row[f"videos/{video_key}/file_index"] = file_index
                    episode_row[f"videos/{video_key}/from_timestamp"] = (
                        episode_video_start / fps
                    )
                    episode_row[f"videos/{video_key}/to_timestamp"] = (
                        shard_frame_index / fps
                    )
                episode_rows.append(episode_row)

        data_path = _shard_data_path(
            dataset_root, chunk_index=chunk_index, file_index=file_index
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(data_rows).to_parquet(data_path, index=False)
        episode_path = _shard_episode_path(
            dataset_root, chunk_index=chunk_index, file_index=file_index
        )
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(episode_rows).to_parquet(episode_path, index=False)
        pd.DataFrame(video_index_rows).to_parquet(
            _checkpoint_artifact_path(
                dataset_root,
                chunk_index=chunk_index,
                file_index=file_index,
                suffix="video_index.parquet",
            ),
            index=False,
        )
        pd.DataFrame(frame_record_rows).to_parquet(
            _checkpoint_artifact_path(
                dataset_root,
                chunk_index=chunk_index,
                file_index=file_index,
                suffix="frame_records.parquet",
            ),
            index=False,
        )
        _checkpoint_artifact_path(
            dataset_root,
            chunk_index=chunk_index,
            file_index=file_index,
            suffix="tasks.jsonl",
        ).write_text("\n".join(task_lines) + "\n", encoding="utf-8")
        _write_json(
            _checkpoint_json_path(
                dataset_root, chunk_index=chunk_index, file_index=file_index
            ),
            {
                "complete": True,
                "chunk_index": chunk_index,
                "file_index": file_index,
                "frame_count": len(data_rows),
                "episode_count": len(episode_rows),
                "video_count": len(CAMERAS),
            },
        )
    except Exception:
        _clear_shard_outputs(
            dataset_root, chunk_index=chunk_index, file_index=file_index
        )
        raise


def _merge_checkpoint_metadata(dataset_root: Path) -> None:
    _cv2, _np, pd, _pa, _pq, _tqdm = _dependencies()
    checkpoints = _complete_checkpoints(dataset_root)
    if not checkpoints:
        raise ValueError("no complete shard checkpoints to merge")
    _merge_parquet_files(
        [_checkpoint_sibling(path, "video_index.parquet") for path in checkpoints],
        dataset_root / "meta" / "navvla_video_index.parquet",
    )
    _concatenate_files(
        [_checkpoint_sibling(path, "frame_metadata.jsonl") for path in checkpoints],
        dataset_root / "meta" / "navvla_frame_metadata.jsonl",
    )
    _concatenate_files(
        [_checkpoint_sibling(path, "tasks.jsonl") for path in checkpoints],
        dataset_root / "meta" / "navvla_tasks.jsonl",
    )
    task_rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        for line in _checkpoint_sibling(
            checkpoint, "tasks.jsonl"
        ).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                task_rows.append(
                    {"task_index": int(row["task_index"]), "task": str(row["task"])}
                )
    pd.DataFrame(task_rows).drop_duplicates("task_index").set_index("task").to_parquet(
        dataset_root / "meta" / "tasks.parquet", index=True
    )


def _make_context_episode_factory(
    checkpoint_paths: list[Path], *, split: str
) -> Any:
    _cv2, _np, pd, _pa, _pq, _tqdm = _dependencies()

    def factory() -> Iterator[ContextEpisode]:
        for checkpoint in checkpoint_paths:
            records = pd.read_parquet(
                _checkpoint_sibling(checkpoint, "frame_records.parquet")
            )
            for _episode_index, episode_records in records.groupby(
                "episode_index", sort=True
            ):
                rows = episode_records.sort_values("index", kind="stable")
                episode_id = str(rows.iloc[0]["episode_id"])
                frames = tuple(
                    ContextFrame(
                        frame_index=int(row.frame_index),
                        timestamp=float(row.timestamp),
                        data_index=int(row.index),
                    )
                    for row in rows.itertuples(index=False)
                )
                yield ContextEpisode(
                    episode_id=episode_id,
                    split=split,
                    frames=frames,
                )

    return factory


def _load_frame_hashes(
    checkpoint_paths: list[Path],
) -> dict[tuple[str, int, str], str]:
    _cv2, _np, pd, _pa, _pq, _tqdm = _dependencies()
    hashes: dict[tuple[str, int, str], str] = {}
    for checkpoint in checkpoint_paths:
        records = pd.read_parquet(
            _checkpoint_sibling(checkpoint, "frame_records.parquet")
        )
        for row in records.itertuples(index=False):
            hashes[(str(row.episode_id), int(row.frame_index), "front")] = str(
                row.front_dhash
            )
    return hashes


def _write_statistics(
    dataset_root: Path,
    *,
    dataset_name: str,
    action_horizon: int,
    total_frames: int,
    total_episodes: int,
) -> None:
    _cv2, np, pd, _pa, _pq, _tqdm = _dependencies()
    temporary_dir = dataset_root / "meta" / ".statistics_tmp"
    shutil.rmtree(temporary_dir, ignore_errors=True)
    temporary_dir.mkdir(parents=True)
    actions = np.memmap(
        temporary_dir / "actions.dat",
        mode="w+",
        dtype=np.float32,
        shape=(total_frames * action_horizon, 4),
    )
    states = np.memmap(
        temporary_dir / "states.dat",
        mode="w+",
        dtype=np.float32,
        shape=(total_frames, 4),
    )
    action_offset = 0
    state_offset = 0
    try:
        for path in sorted((dataset_root / "data").glob("chunk-*/*.parquet")):
            data = pd.read_parquet(
                path, columns=["action", "observation.state"]
            )
            shard_actions = np.stack(
                [
                    _nested_float_array(
                        value, rows=action_horizon, columns=4
                    )
                    for value in data["action"]
                ]
            ).reshape(-1, 4)
            shard_states = np.stack(
                [
                    _nested_float_array(value, rows=1, columns=4).reshape(4)
                    for value in data["observation.state"]
                ]
            )
            actions[action_offset : action_offset + len(shard_actions)] = shard_actions
            states[state_offset : state_offset + len(shard_states)] = shard_states
            action_offset += len(shard_actions)
            state_offset += len(shard_states)
        if action_offset != total_frames * action_horizon:
            raise ValueError("action statistics length does not match total_frames")
        if state_offset != total_frames:
            raise ValueError("state statistics length does not match total_frames")
        _write_json(
            dataset_root / "dataset_statistics.json",
            {
                dataset_name: {
                    "action": _array_statistics(actions),
                    "state": _array_statistics(states),
                    "num_trajectories": total_episodes,
                    "num_transitions": total_frames,
                }
            },
        )
    finally:
        del actions
        del states
        shutil.rmtree(temporary_dir, ignore_errors=True)


def _nested_float_array(value: Any, *, rows: int, columns: int) -> Any:
    _cv2, np, _pd, _pa, _pq, _tqdm = _dependencies()
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            value = value.tolist()
        else:
            return value.astype(np.float32, copy=False).reshape(rows, columns)
    return np.asarray(
        [np.asarray(item, dtype=np.float32).reshape(-1) for item in value],
        dtype=np.float32,
    ).reshape(rows, columns)


def _array_statistics(values: Any) -> dict[str, Any]:
    _cv2, np, _pd, _pa, _pq, _tqdm = _dependencies()
    if values.size == 0:
        values = np.zeros((1, 4), dtype=np.float32)
    return {
        "mean": np.mean(values, axis=0, dtype=np.float64).astype(np.float32).tolist(),
        "std": np.maximum(
            np.std(values, axis=0, dtype=np.float64), 1e-6
        ).astype(np.float32).tolist(),
        "min": np.min(values, axis=0).astype(np.float32).tolist(),
        "max": np.max(values, axis=0).astype(np.float32).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(np.float32).tolist(),
        "normalization_modes": ["q01_q99"] * values.shape[1],
        "mask": [True] * values.shape[1],
        "binary_mask": [False] * values.shape[1],
    }


def _state_from_frame(frame: dict[str, Any]) -> list[float]:
    robot_pos = frame.get("robot_pos", [])
    if not isinstance(robot_pos, (list, tuple)) or len(robot_pos) < 3:
        raise ValueError("frame is missing robot_pos[3]")
    try:
        world_x = float(robot_pos[0])
        world_y = float(robot_pos[1])
        world_z = float(robot_pos[2])
        world_yaw = float(frame.get("robot_yaw"))
    except (TypeError, ValueError) as exc:
        raise ValueError("frame has invalid robot_pos or robot_yaw") from exc
    if not all(math.isfinite(value) for value in (world_x, world_y, world_z, world_yaw)):
        raise ValueError("frame has non-finite robot_pos or robot_yaw")
    return [world_x, -world_z, world_y, -world_yaw]


def _convert_action_axes(vx: float, vy: float, omega: float) -> tuple[float, float, float]:
    return float(vx), -float(vy), -float(omega)


def _dhash(image_bgr: Any) -> str:
    cv2, _np, _pd, _pa, _pq, _tqdm = _dependencies()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    differences = small[:, :-1] > small[:, 1:]
    value = 0
    for bit in differences.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _probe_video(path: Path) -> VideoInfo:
    cv2, _np, _pd, _pa, _pq, _tqdm = _dependencies()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"unable to open video: {path}")
        return VideoInfo(
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
    finally:
        capture.release()


def _validate_episode_media(
    episode: SourceEpisode,
    frames: list[dict[str, Any]],
    video_info: Mapping[str, VideoInfo],
    expected_fps: float,
) -> None:
    expected_frames = len(frames)
    counts = {name: info.frame_count for name, info in video_info.items()}
    if any(count != expected_frames for count in counts.values()):
        raise ValueError(
            f"{episode.episode_id} frame mismatch: json={expected_frames}, videos={counts}"
        )
    for camera_name, info in video_info.items():
        if info.width <= 0 or info.height <= 0:
            raise ValueError(
                f"{episode.episode_id}/{camera_name} has invalid dimensions"
            )
        if info.fps > 0 and not math.isclose(
            info.fps, expected_fps, rel_tol=0.01, abs_tol=0.01
        ):
            raise ValueError(
                f"{episode.episode_id}/{camera_name} fps={info.fps}, "
                f"expected {expected_fps}"
            )


def _read_synchronized_frames(
    captures: Mapping[str, Any],
    *,
    episode: SourceEpisode,
    frame_pos: int,
) -> dict[str, Any]:
    frames: dict[str, Any] = {}
    for camera_name, capture in captures.items():
        ok, image = capture.read()
        if not ok or image is None:
            raise ValueError(
                f"unable to read {episode.episode_id}/{camera_name} frame {frame_pos}"
            )
        frames[camera_name] = image
    return frames


def _episode_offsets(episodes: list[ValidatedEpisode]) -> dict[int, int]:
    offsets: dict[int, int] = {}
    cursor = 0
    for episode_index, episode in enumerate(episodes):
        offsets[episode_index] = cursor
        cursor += episode.frame_count
    return offsets


def _group_episodes(
    episodes: list[ValidatedEpisode],
    *,
    episodes_per_file: int,
    files_per_chunk: int,
) -> list[tuple[tuple[int, int], list[tuple[int, ValidatedEpisode]]]]:
    groups: dict[tuple[int, int], list[tuple[int, ValidatedEpisode]]] = {}
    for episode_index, episode in enumerate(episodes):
        global_file_index = episode_index // episodes_per_file
        shard_key = (
            global_file_index // files_per_chunk,
            global_file_index % files_per_chunk,
        )
        groups.setdefault(shard_key, []).append((episode_index, episode))
    return sorted(groups.items())


def _shard_data_path(dataset_root: Path, *, chunk_index: int, file_index: int) -> Path:
    return dataset_root / DATA_PATH_PATTERN.format(
        chunk_index=chunk_index, file_index=file_index
    )


def _shard_episode_path(dataset_root: Path, *, chunk_index: int, file_index: int) -> Path:
    return (
        dataset_root
        / "meta"
        / "episodes"
        / f"chunk-{chunk_index:03d}"
        / f"part-{file_index:03d}.parquet"
    )


def _shard_video_paths(
    dataset_root: Path, *, chunk_index: int, file_index: int
) -> dict[str, Path]:
    return {
        camera_name: dataset_root
        / "videos"
        / str(camera["video_key"])
        / f"chunk-{chunk_index:03d}"
        / f"part-{file_index:03d}.mp4"
        for camera_name, camera in CAMERAS.items()
    }


def _checkpoint_prefix(chunk_index: int, file_index: int) -> str:
    return f"chunk-{chunk_index:03d}_part-{file_index:03d}"


def _checkpoint_dir(dataset_root: Path) -> Path:
    return dataset_root / "meta" / "checkpoints"


def _checkpoint_json_path(
    dataset_root: Path, *, chunk_index: int, file_index: int
) -> Path:
    return _checkpoint_dir(dataset_root) / (
        _checkpoint_prefix(chunk_index, file_index) + ".json"
    )


def _checkpoint_artifact_path(
    dataset_root: Path,
    *,
    chunk_index: int,
    file_index: int,
    suffix: str,
) -> Path:
    return _checkpoint_dir(dataset_root) / (
        f"{_checkpoint_prefix(chunk_index, file_index)}_{suffix}"
    )


def _checkpoint_sibling(checkpoint_path: Path, suffix: str) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_{suffix}")


def _complete_checkpoints(dataset_root: Path) -> list[Path]:
    result: list[Path] = []
    for path in sorted(_checkpoint_dir(dataset_root).glob("chunk-*_part-*.json")):
        payload = _read_json(path)
        if isinstance(payload, dict) and payload.get("complete"):
            result.append(path)
    return result


def _shard_status(
    dataset_root: Path,
    *,
    chunk_index: int,
    file_index: int,
    expected_frames: int,
) -> str:
    _cv2, _np, _pd, _pa, pq, _tqdm = _dependencies()
    data_path = _shard_data_path(
        dataset_root, chunk_index=chunk_index, file_index=file_index
    )
    episode_path = _shard_episode_path(
        dataset_root, chunk_index=chunk_index, file_index=file_index
    )
    video_paths = _shard_video_paths(
        dataset_root, chunk_index=chunk_index, file_index=file_index
    )
    checkpoint_path = _checkpoint_json_path(
        dataset_root, chunk_index=chunk_index, file_index=file_index
    )
    main_artifacts = [data_path, episode_path, checkpoint_path, *video_paths.values()]
    if not any(path.exists() for path in main_artifacts):
        return "missing"
    if not all(path.exists() and path.stat().st_size > 0 for path in main_artifacts):
        return "incomplete"
    checkpoint_artifacts = [
        _checkpoint_artifact_path(
            dataset_root,
            chunk_index=chunk_index,
            file_index=file_index,
            suffix=suffix,
        )
        for suffix in (
            "video_index.parquet",
            "frame_records.parquet",
            "frame_metadata.jsonl",
            "tasks.jsonl",
        )
    ]
    if not all(
        path.exists() and path.stat().st_size > 0 for path in checkpoint_artifacts
    ):
        return "incomplete"
    try:
        checkpoint = _read_json(checkpoint_path)
        if not checkpoint.get("complete"):
            return "incomplete"
        if int(checkpoint.get("frame_count", -1)) != expected_frames:
            return "incomplete"
        if pq.ParquetFile(data_path).metadata.num_rows != expected_frames:
            return "incomplete"
        if any(_video_frame_count(path) != expected_frames for path in video_paths.values()):
            return "incomplete"
    except (OSError, ValueError, json.JSONDecodeError):
        return "incomplete"
    return "complete"


def _clear_shard_outputs(
    dataset_root: Path, *, chunk_index: int, file_index: int
) -> None:
    paths = [
        _shard_data_path(
            dataset_root, chunk_index=chunk_index, file_index=file_index
        ),
        _shard_episode_path(
            dataset_root, chunk_index=chunk_index, file_index=file_index
        ),
        *_shard_video_paths(
            dataset_root, chunk_index=chunk_index, file_index=file_index
        ).values(),
        _checkpoint_json_path(
            dataset_root, chunk_index=chunk_index, file_index=file_index
        ),
    ]
    prefix = _checkpoint_prefix(chunk_index, file_index)
    paths.extend(_checkpoint_dir(dataset_root).glob(f"{prefix}_*"))
    for path in paths:
        path.unlink(missing_ok=True)


def _video_frame_count(path: Path) -> int:
    cv2, _np, _pd, _pa, _pq, _tqdm = _dependencies()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return 0
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def _merge_parquet_files(paths: list[Path], output_path: Path) -> None:
    _cv2, _np, _pd, pa, pq, _tqdm = _dependencies()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    writer = None
    try:
        for path in paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=PARQUET_ROW_GROUP_SIZE):
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema)
                writer.write_table(table, row_group_size=PARQUET_ROW_GROUP_SIZE)
        if writer is None:
            raise ValueError("no parquet checkpoint files to merge")
        writer.close()
        writer = None
        temporary.replace(output_path)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def _concatenate_files(paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as target:
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    temporary.replace(output_path)


def _write_info_metadata(
    dataset_root: Path,
    *,
    dataset_name: str,
    split: str,
    fps: float,
    dt: float,
    action_horizon: int,
    episodes_per_file: int,
    files_per_chunk: int,
    total_episodes: int,
    total_frames: int,
    total_videos: int,
    camera_shapes: Mapping[str, tuple[int, int]],
) -> None:
    features: dict[str, Any] = {}
    video_paths: dict[str, str] = {}
    for camera_name, camera in CAMERAS.items():
        video_key = str(camera["video_key"])
        height, width = camera_shapes[camera_name]
        features[f"observation.images.{video_key}"] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": fps,
                "video.height": height,
                "video.width": width,
                "video.channels": 3,
            },
        }
        video_paths[video_key] = (
            f"videos/{video_key}/chunk-{{chunk_index:03d}}/"
            "part-{file_index:03d}.mp4"
        )
    features.update(
        {
            "observation.state": {
                "dtype": "float32",
                "shape": [4],
                "names": ["x", "y", "z", "yaw"],
            },
            "action": {
                "dtype": "float32",
                "shape": [action_horizon * 4],
                "names": ["action"],
            },
            "action.padding_mask": {
                "dtype": "bool",
                "shape": [action_horizon],
                "names": ["horizon"],
            },
            "context.index_key": {
                "dtype": "string",
                "shape": [1],
                "names": ["context_index_key"],
            },
            "timestamp": {
                "dtype": "float64",
                "shape": [1],
                "names": ["timestamp"],
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": ["task_index"],
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": ["episode_index"],
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": ["frame_index"],
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": ["index"],
            },
            "next.done": {
                "dtype": "bool",
                "shape": [1],
                "names": ["done"],
            },
            "sample.action_available": {
                "dtype": "bool",
                "shape": [1],
                "names": ["action_available"],
            },
        }
    )
    _write_json(
        dataset_root / "meta" / "info.json",
        {
            "codebase_version": "v3.0",
            "dataset_name": dataset_name,
            "robot_type": "habitat_spot",
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_episodes,
            "total_videos": total_videos,
            "chunks_size": files_per_chunk,
            "fps": fps,
            "splits": {split: f"0:{total_episodes}"},
            "data_path": DATA_PATH_PATTERN,
            "video_path": video_paths,
            "features": features,
            "navvla": {
                "schema_version": "0.1",
                "context_policy_version": CONTEXT_POLICY_VERSION,
                "cache_policy_version": CACHE_POLICY_VERSION,
                "action_horizon": action_horizon,
                "action_dim": 4,
                "state_dim": 4,
                "control_frequency_hz": 1.0 / dt,
                "episodes_per_file": episodes_per_file,
                "files_per_chunk": files_per_chunk,
                "action_mode": "anchor_relative_planar_pose",
                "action_axis_policy": (
                    "source_x_neg_y_neg_theta_then_rotate_by_integrated_theta"
                ),
                "action_anchor": "current_frame_initial_local_frame",
                "integration_dt": dt,
                "tail_action_policy": "repeat_last_accumulated_pose_to_horizon",
                "padding_mask_policy": "all_false_after_repeat_last_padding",
                "state_policy": "[world_x, -world_z, world_y_height, -world_yaw]",
                "source_state_fields": {
                    "robot_pos": "[world_x, world_y_height, world_z]",
                    "robot_yaw": "world_+Y_rad",
                },
            },
        },
    )


def _write_camera_metadata(dataset_root: Path) -> None:
    payload = {
        camera_name: {
            "name": camera_name,
            "video_key": camera["video_key"],
            "viewpoint_type": camera_name,
            "azimuth_rad": camera["azimuth_rad"],
            "intrinsics": None,
            "extrinsics_body": None,
            "calibration_status": "unknown",
        }
        for camera_name, camera in CAMERAS.items()
    }
    _write_json(dataset_root / "meta" / "navvla_cameras.json", payload)


def _write_modality_metadata(dataset_root: Path, *, action_horizon: int) -> None:
    _write_json(
        dataset_root / "meta" / "modality.json",
        {
            "video": {
                str(camera["video_key"]): {
                    "original_key": f"observation.images.{camera['video_key']}"
                }
                for camera in CAMERAS.values()
            },
            "state": {
                name: {
                    "start": index,
                    "end": index + 1,
                    "absolute": True,
                    "dtype": "float32",
                    "original_key": "observation.state",
                }
                for index, name in enumerate(("x", "y", "z", "yaw"))
            },
            "action": {
                name: {
                    "start": index,
                    "end": index + 1,
                    "absolute": False,
                    "dtype": "float32",
                    "original_key": "action",
                }
                for index, name in enumerate(("x", "y", "z", "yaw"))
            },
            "annotation": {
                "language.language_instruction": {"original_key": "task_index"}
            },
            "action_horizon": action_horizon,
        },
    )


def _write_schema_extension(
    dataset_root: Path, *, spec: DatasetSpec, has_context: bool
) -> None:
    payload: dict[str, Any] = {
        "schema_version": "0.1",
        "context_policy_version": spec.context_policy_version,
        "cache_policy_version": spec.cache_policy_version,
        "frame_metadata": "meta/navvla_frame_metadata.jsonl",
        "video_index": "meta/navvla_video_index.parquet",
    }
    if has_context:
        payload.update(
            {
                "history_fields": [
                    "context.index_key",
                    "current_tvi_time",
                    "history_steps",
                    "history_blocks",
                    "history_token_refs",
                    "history_mask",
                ],
                "context_index_manifest": "meta/navvla_context_index_manifest.json",
                "context_index": "meta/context_index/budget_<budget>",
                "context_meta": (
                    "meta/context_index/budget_<budget>/context_meta.parquet"
                ),
                "context_refs": "meta/context_index/budget_<budget>/refs.parquet",
                "context_arrays": (
                    "meta/context_index/budget_<budget>/context_arrays"
                ),
                "context_debug": (
                    f"cache/context_index_debug/budget_<budget>/{spec.split}.parquet"
                ),
            }
        )
    _write_json(dataset_root / "meta" / "navvla_schema_ext.json", payload)


def _validate_resume_config(dataset_root: Path, expected: dict[str, Any]) -> None:
    config_path = dataset_root / "meta" / CONVERSION_CONFIG_NAME
    if not config_path.exists():
        raise FileNotFoundError(f"resume config is missing: {config_path}")
    saved = _read_json(config_path)
    mismatches = {
        key: {"saved": saved.get(key), "expected": value}
        for key, value in expected.items()
        if saved.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "resume arguments differ from the existing conversion: "
            + json.dumps(mismatches, ensure_ascii=False)
        )


def _make_output_dirs(dataset_root: Path) -> None:
    relative_paths = [
        "meta",
        "meta/episodes",
        "meta/checkpoints",
        "data",
        "cache/dhash",
        *[f"videos/{camera['video_key']}" for camera in CAMERAS.values()],
    ]
    for relative_path in relative_paths:
        (dataset_root / relative_path).mkdir(parents=True, exist_ok=True)


def _validate_dataset_name(dataset_name: str) -> None:
    path = Path(dataset_name)
    if (
        not dataset_name
        or path.is_absolute()
        or len(path.parts) != 1
        or dataset_name in {".", ".."}
    ):
        raise ValueError("dataset_name must be one non-empty path component")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _natural_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def camera_specs() -> tuple[CameraSpec, ...]:
    return tuple(
        CameraSpec(
            name=name,
            video_key=str(camera["video_key"]),
            azimuth_rad=float(camera["azimuth_rad"]),
            viewpoint_type=name,
        )
        for name, camera in CAMERAS.items()
    )


def _dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "Track LeRobot conversion requires opencv-python-headless (or "
            "opencv-python), numpy, pandas, pyarrow and tqdm. Install "
            "requirements-lerobot.txt."
        ) from exc
    return cv2, np, pd, pa, pq, tqdm
