#!/usr/bin/env python3
"""从 eval 落盘目录扫描各 scene 下的 episode 结果 json，聚合 SR / TR / CR。

不依赖 ``summary_split_*.json``，避免 split 跳过 episode 或重复写入导致合并不准。

扫描规则（``save_dir/<scene>/<episode_id>.json``）：
- 跳过 ``*_info.json``（逐步记录）
- 跳过 ``summary*.json``
- 仅保留含 ``success`` 字段的 episode 结果

指标（与 ``scripts/merge_track_eval_summaries.py`` 一致）：
- SR (``success_rate``)：``success`` 为真的 episode 占比
- TR (``tracking_rate`` / ``mean_following_rate``)：按 ``total_step`` 加权
  ``Σ(following_rate * total_step) / Σ total_step``，等价于 ``Σ following_step / Σ total_step``
- CR (``collision_rate``)：``collision`` 为真的 episode 占比

用法::

    python scripts/summarize_eval_from_episodes.py results/eval/stt/train_subset
    python scripts/summarize_eval_from_episodes.py results/eval/stt/train_subset -o summary_from_episodes.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return bool(v)


def _is_episode_result_json(path: Path) -> bool:
    name = path.name
    if name.endswith("_info.json"):
        return False
    if name.startswith("summary"):
        return False
    return path.suffix == ".json"


def _episode_row(result: dict, scene_key: str, episode_id: str) -> dict:
    total_step = int(result.get("total_step", 0))
    following_step = result.get("following_step")
    if following_step is not None:
        following_rate = float(following_step) / max(total_step, 1)
    else:
        following_rate = float(result.get("following_rate", 0.0))
    return {
        "episode_id": str(result.get("episode_id", episode_id)),
        "scene": str(result.get("scene_id", scene_key)),
        "success": _to_bool(result.get("success", False)),
        "status": str(result.get("status", "Unknown")),
        "following_rate": following_rate,
        "total_step": total_step,
        "collision": _to_bool(result.get("collision", False)),
        "finish": _to_bool(result.get("finish", False)),
    }


def collect_episodes(save_dir: Path) -> list[dict]:
    episodes: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for scene_dir in sorted(p for p in save_dir.iterdir() if p.is_dir()):
        scene_key = scene_dir.name
        for path in sorted(scene_dir.glob("*.json")):
            if not _is_episode_result_json(path):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    result = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(result, dict) or "success" not in result:
                continue
            episode_id = str(result.get("episode_id", path.stem))
            key = (scene_key, episode_id)
            if key in seen:
                continue
            seen.add(key)
            episodes.append(_episode_row(result, scene_key, episode_id))
    return episodes


def aggregate(episodes: list[dict]) -> dict:
    n = len(episodes)
    if n == 0:
        return {
            "source": "episode_json",
            "num_episodes": 0,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "mean_following_rate": 0.0,
            "tracking_rate": 0.0,
            "mean_total_step": 0.0,
            "collision_episodes": 0,
            "status_counts": {},
            "episodes": [],
        }

    successes = [bool(e["success"]) for e in episodes]
    status_counts: dict[str, int] = {}
    for e in episodes:
        s = e.get("status", "Unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    total_steps = sum(int(e["total_step"]) for e in episodes)
    if total_steps > 0:
        mean_following_rate = sum(
            float(e["following_rate"]) * int(e["total_step"]) for e in episodes
        ) / total_steps
    else:
        mean_following_rate = 0.0

    collisions = sum(1 for e in episodes if e.get("collision"))
    return {
        "source": "episode_json",
        "num_episodes": n,
        "success_rate": sum(successes) / n,
        "collision_rate": collisions / n,
        "mean_following_rate": mean_following_rate,
        "tracking_rate": mean_following_rate,
        "mean_total_step": total_steps / n,
        "collision_episodes": collisions,
        "status_counts": status_counts,
        "episodes": episodes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "save_dir",
        type=str,
        help="run_eval --save-path 目录（含 <scene>/<episode_id>.json）",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=str,
        default="summary_from_episodes.json",
        help="相对于 save_dir 的输出文件名；传空字符串则只打印不写文件",
    )
    args = ap.parse_args()

    root = Path(args.save_dir).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    episodes = collect_episodes(root)
    summary = aggregate(episodes)

    sr = summary["success_rate"]
    tr = summary["tracking_rate"]
    cr = summary["collision_rate"]
    n = summary["num_episodes"]
    print(
        f"[eval] {root}\n"
        f"  episodes={n}  SR={sr:.4f}  TR={tr:.4f}  CR={cr:.4f}\n"
        f"  status={summary['status_counts']}"
    )

    if args.output:
        out_path = root / args.output
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
