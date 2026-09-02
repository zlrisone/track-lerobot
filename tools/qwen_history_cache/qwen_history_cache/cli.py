"""Command-line interface for EVT-bench Qwen history-cache generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qwen_history_cache.config import CacheConfig, load_config
from qwen_history_cache.manifest import validate_profile
from qwen_history_cache.metadata import build_worklist
from qwen_history_cache.pipeline import (
    building_status,
    generate_split,
    publish_completed_build,
    run_pilot,
)


def _splits(args: argparse.Namespace, config: CacheConfig) -> list[str]:
    return [item.upper() for item in (args.splits or config.splits)]


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _metadata(args: argparse.Namespace, config: CacheConfig) -> None:
    reports = []
    for split in _splits(args, config):
        reports.append(
            build_worklist(
                config.split_root(split),
                config.worklist_path(split),
                video_check=args.video_check,
                decode_samples=args.decode_samples,
                overwrite=args.force,
            )
        )
    _print(reports)


def _pilot(args: argparse.Namespace, config: CacheConfig) -> None:
    split = args.split.upper()
    output = (
        Path(args.output_dir)
        if args.output_dir
        else config.worklist_dir.parent / "pilot" / split
    )
    _print(
        run_pilot(
            config,
            split,
            output_dir=output,
            sample_count=args.samples,
            device=args.device,
            model_dtype=args.model_dtype,
            overwrite=args.force,
        )
    )


def _generate(args: argparse.Namespace, config: CacheConfig) -> None:
    _print(
        generate_split(
            config,
            args.split,
            rank=args.rank,
            world_size=args.world_size,
            device=args.device,
            model_dtype=args.model_dtype,
            batch_size=args.batch_size,
            max_shards=args.max_shards,
            publish=args.publish,
            show_progress=args.progress,
        )
    )


def _status(args: argparse.Namespace, config: CacheConfig) -> None:
    _print(building_status(config, args.split))


def _publish(args: argparse.Namespace, config: CacheConfig) -> None:
    _print(publish_completed_build(config, args.split))


def _validate(args: argparse.Namespace, config: CacheConfig) -> None:
    reports = []
    for split in _splits(args, config):
        reports.append(
            validate_profile(
                config.profile_root(split),
                worklist_path=config.worklist_path(split),
                verify_hashes=not args.skip_hashes,
            )
        )
    _print(reports)


def build_parser() -> argparse.ArgumentParser:
    """Construct the command parser."""
    parser = argparse.ArgumentParser(
        description="Generate Qwen3-VL pooled history caches for EVT-bench"
    )
    parser.add_argument(
        "--config",
        default="configs/qwen3_vl_2b_evt_bench.yaml",
        help="Path to the generation YAML",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata = subparsers.add_parser(
        "metadata", help="Build and validate source-frame worklists without Qwen"
    )
    metadata.add_argument("--splits", nargs="+", choices=["AT", "DT", "STT"])
    metadata.add_argument(
        "--video-check", choices=["none", "exists", "bounds"], default="bounds"
    )
    metadata.add_argument("--decode-samples", type=int, default=3)
    metadata.add_argument("--force", action="store_true")
    metadata.set_defaults(handler=_metadata)

    pilot = subparsers.add_parser(
        "pilot", help="Encode a small diverse real-frame regression sample"
    )
    pilot.add_argument("--split", required=True, choices=["AT", "DT", "STT"])
    pilot.add_argument("--samples", type=int, default=8)
    pilot.add_argument("--device", default="cuda:0")
    pilot.add_argument(
        "--model-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    pilot.add_argument("--output-dir")
    pilot.add_argument("--force", action="store_true")
    pilot.set_defaults(handler=_pilot)

    generate = subparsers.add_parser(
        "generate", aliases=["resume"], help="Generate or resume deterministic shards"
    )
    generate.add_argument("--split", required=True, choices=["AT", "DT", "STT"])
    generate.add_argument("--rank", type=int, default=0)
    generate.add_argument("--world-size", type=int, default=1)
    generate.add_argument("--device", default="cuda:0")
    generate.add_argument(
        "--model-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    generate.add_argument("--batch-size", type=int)
    generate.add_argument(
        "--max-shards", type=int, help="Debug only: stop after this many new shards"
    )
    generate.add_argument(
        "--publish", action=argparse.BooleanOptionalAction, default=True
    )
    generate.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show per-rank generation progress (default: enabled)",
    )
    generate.set_defaults(handler=_generate)

    status = subparsers.add_parser("status", help="Validate and report build progress")
    status.add_argument("--split", required=True, choices=["AT", "DT", "STT"])
    status.set_defaults(handler=_status)

    publish = subparsers.add_parser(
        "publish", help="Validate and atomically publish a completed distributed build"
    )
    publish.add_argument("--split", required=True, choices=["AT", "DT", "STT"])
    publish.set_defaults(handler=_publish)

    validate = subparsers.add_parser(
        "validate", help="Validate published profiles against worklists"
    )
    validate.add_argument("--splits", nargs="+", choices=["AT", "DT", "STT"])
    validate.add_argument("--skip-hashes", action="store_true")
    validate.set_defaults(handler=_validate)
    return parser


def main() -> None:
    """Run the selected cache-generation command."""
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    args.handler(args, config)


if __name__ == "__main__":
    main()
