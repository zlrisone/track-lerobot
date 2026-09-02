"""Strict OpenCV frame decoding for EVT-bench front-camera videos."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def probe_video(path: str | Path) -> dict[str, int | float]:
    """Read geometry and frame count from a video container."""
    video_path = Path(path)
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        result: dict[str, int | float] = {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        }
    finally:
        capture.release()
    if result["width"] <= 0 or result["height"] <= 0:
        raise RuntimeError(f"Video reports invalid geometry: {video_path}: {result}")
    if result["frame_count"] <= 0:
        raise RuntimeError(f"Video reports no frames: {video_path}")
    return result


class OpenCVFrameReader:
    """Read exact indexed RGB frames while reusing a bounded set of handles."""

    def __init__(self, dataset_root: str | Path, *, max_open_videos: int = 4) -> None:
        self.dataset_root = Path(dataset_root)
        if max_open_videos <= 0:
            raise ValueError("max_open_videos must be positive")
        self.max_open_videos = max_open_videos
        self._captures: OrderedDict[Path, cv2.VideoCapture] = OrderedDict()

    def __enter__(self) -> OpenCVFrameReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Release all cached video handles."""
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()

    def _capture(self, relative_path: str) -> tuple[Path, cv2.VideoCapture]:
        path = self.dataset_root / relative_path
        capture = self._captures.pop(path, None)
        if capture is None:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Could not open video: {path}")
        self._captures[path] = capture
        while len(self._captures) > self.max_open_videos:
            _, old_capture = self._captures.popitem(last=False)
            old_capture.release()
        return path, capture

    def _read(self, relative_path: str, frame_index: int) -> np.ndarray:
        if frame_index < 0:
            raise ValueError(f"Frame index must be non-negative: {frame_index}")
        path, capture = self._capture(relative_path)
        current = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        if current != frame_index and not capture.set(
            cv2.CAP_PROP_POS_FRAMES, float(frame_index)
        ):
            raise RuntimeError(f"Could not seek {path} to frame {frame_index}")
        ok, bgr = capture.read()
        if not ok or bgr is None:
            raise RuntimeError(f"Could not decode {path} frame {frame_index}")
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise RuntimeError(
                f"Decoded frame is not HxWx3 at {path}:{frame_index}: {bgr.shape}"
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.dtype != np.uint8:
            raise RuntimeError(
                f"Decoded frame is not uint8 at {path}:{frame_index}: {rgb.dtype}"
            )
        return rgb

    def read_rows(self, rows: list[dict[str, Any]]) -> list[np.ndarray]:
        """Decode worklist rows, grouping seeks by video and preserving row order."""
        grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for output_index, row in enumerate(rows):
            grouped[str(row["video_path"])].append(
                (output_index, int(row["video_frame_index"]))
            )
        output: list[np.ndarray | None] = [None] * len(rows)
        for relative_path, requests in grouped.items():
            for output_index, frame_index in sorted(requests, key=lambda item: item[1]):
                output[output_index] = self._read(relative_path, frame_index)
        if any(frame is None for frame in output):
            raise RuntimeError("Internal decoder error left an output frame unset")
        return [frame for frame in output if frame is not None]
