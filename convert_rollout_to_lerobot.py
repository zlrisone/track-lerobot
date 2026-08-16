#!/usr/bin/env python3
"""Convert Track four-view rollout episodes to a standalone LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from track_lerobot.converter import CAMERAS, DEFAULT_CONTEXT_CAMERAS, convert_rollout_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Track rollout data with front/left/right/rear videos to "
            "a sharded LeRobot v3 dataset."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Rollout root containing <scene_id>/*_info.json and four-view MP4 files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output parent directory; the dataset is written below <dataset-name>/.",
    )
    parser.add_argument("--dataset-name", default="track")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override FPS; otherwise read it from episode metadata, then fall back to 1/dt.",
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--episodes-per-file", type=int, default=20)
    parser.add_argument("--files-per-chunk", type=int, default=50)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--context-token-budgets",
        type=int,
        nargs="+",
        default=[1024],
        help="Token budgets for context indexes; the first value is the default budget.",
    )
    parser.add_argument(
        "--context-cameras",
        nargs="+",
        choices=list(CAMERAS),
        default=list(DEFAULT_CONTEXT_CAMERAS),
        help="Cameras referenced by context history; all four videos are always converted.",
    )
    parser.add_argument(
        "--use-bats",
        action="store_true",
        help="Use deterministic BATS history selection instead of recent-frame sliding history.",
    )
    parser.add_argument(
        "--use-hash-dedup",
        action="store_true",
        help="Remove near-duplicate history frames using front-camera dHash.",
    )
    parser.add_argument(
        "--dhash-threshold",
        type=int,
        default=10,
        help="Maximum dHash Hamming distance considered duplicate.",
    )
    parser.add_argument(
        "--no-context-index",
        action="store_true",
        help="Skip context-index generation while keeping context keys in data parquet files.",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete shard checkpoints and rebuild final metadata.",
    )
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing output dataset before conversion.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip incomplete or media-inconsistent episodes and report them at the end.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = convert_rollout_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        split=args.split,
        fps=args.fps,
        dt=args.dt,
        action_horizon=args.action_horizon,
        episodes_per_file=args.episodes_per_file,
        files_per_chunk=args.files_per_chunk,
        max_episodes=args.max_episodes,
        overwrite=args.overwrite,
        resume=args.resume,
        skip_invalid=args.skip_invalid,
        show_progress=not args.no_progress,
        context_token_budgets=tuple(args.context_token_budgets),
        context_camera_names=tuple(args.context_cameras),
        use_bats=args.use_bats,
        use_hash_dedup=args.use_hash_dedup,
        dhash_threshold=args.dhash_threshold,
        build_context_index=not args.no_context_index,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
