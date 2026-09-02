"""Command-line interface for rebuilding EVT-bench context indexes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evt_context_index.rebuild import (
    BudgetSpec,
    inspect_dataset,
    rebuild_dataset,
    summary_json,
    validate_dataset,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild EVT-bench sliding context indexes for Qwen-GR00T using "
            "current 64 + history 4*N tokens and no TVI token cost."
        )
    )
    parser.add_argument("command", choices=("inspect", "build", "validate"))
    parser.add_argument(
        "--evt-root",
        type=Path,
        default=Path("/data1/yizhang/data/evt-bench"),
        help="Directory containing the AT, DT and STT split roots.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["AT", "DT", "STT"], help="Splits to process."
    )
    parser.add_argument("--token-budget", type=int, default=1024)
    parser.add_argument("--current-visual-tokens", type=int, default=64)
    parser.add_argument("--history-visual-tokens", type=int, default=4)
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Publish over an existing budget directory after moving it to "
            "meta/context_index_backups/. Required for the current EVT data."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command != "build" and args.replace:
        raise SystemExit("--replace is only valid with the build command")
    budget = BudgetSpec(
        token_budget=args.token_budget,
        current_visual_tokens=args.current_visual_tokens,
        history_visual_tokens=args.history_visual_tokens,
    )
    results = []
    for split in args.splits:
        dataset_root = args.evt_root / split
        if args.command == "inspect":
            result = inspect_dataset(dataset_root, budget)
        elif args.command == "build":
            result = rebuild_dataset(
                dataset_root,
                budget,
                replace=args.replace,
                progress=lambda message, name=split: print(
                    f"[{name}] {message}", file=sys.stderr, flush=True
                ),
            )
        else:
            result = validate_dataset(dataset_root, budget)
        results.append(result)
    print(summary_json(results))


if __name__ == "__main__":
    main()
