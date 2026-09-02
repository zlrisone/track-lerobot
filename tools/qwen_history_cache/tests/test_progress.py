"""Tests for generation progress reporting."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa

from qwen_history_cache import pipeline


class _ProgressRecorder:
    instances: list["_ProgressRecorder"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.options = kwargs
        self.updates: list[int] = []
        self.postfixes: list[str] = []
        self.__class__.instances.append(self)

    def __enter__(self) -> "_ProgressRecorder":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def update(self, count: int) -> None:
        self.updates.append(count)

    def set_postfix_str(self, value: str, *, refresh: bool) -> None:
        self.postfixes.append(value)


class _Reader:
    def __init__(self, root: Path) -> None:
        self.root = root

    def __enter__(self) -> "_Reader":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read_rows(self, rows: list[dict[str, int]]) -> list[object]:
        return [object() for _ in rows]


class _Encoder:
    def encode(self, frames: list[object]) -> np.ndarray:
        return np.zeros((len(frames), 1, 2), dtype=np.float16)


def test_generation_progress_tracks_rows_and_resume(
    tmp_path: Path, monkeypatch: Any
) -> None:
    worklist = pa.table({"ref_id": range(5)})
    config = SimpleNamespace(
        shard_size=2,
        batch_size=2,
        token_count=1,
        hidden_dim=2,
        building_root=lambda split: tmp_path / ".profile.building",
        profile_root=lambda split: tmp_path / "profile",
        split_root=lambda split: tmp_path,
    )
    monkeypatch.setattr(
        pipeline,
        "prepare_build_spec",
        lambda config, split, rank: ({"run_fingerprint": "test"}, worklist),
    )
    monkeypatch.setattr(pipeline, "_new_encoder", lambda *args, **kwargs: _Encoder())
    monkeypatch.setattr(pipeline, "OpenCVFrameReader", _Reader)
    monkeypatch.setattr(pipeline, "tqdm", _ProgressRecorder)
    _ProgressRecorder.instances.clear()

    report = pipeline.generate_split(config, "DT", publish=False)

    first = _ProgressRecorder.instances[-1]
    assert first.options["total"] == 5
    assert first.options["initial"] == 0
    assert first.options["unit"] == "ref"
    assert first.updates == [2, 2, 1]
    assert report["generated_shards"] == [0, 1, 2]

    resumed = pipeline.generate_split(config, "DT", publish=False)

    second = _ProgressRecorder.instances[-1]
    assert second.options["total"] == 5
    assert second.options["initial"] == 5
    assert second.updates == []
    assert resumed["reused_shards"] == [0, 1, 2]
