"""Tests for the main-image-embedding extraction boundary."""

import numpy as np
import torch
from qwen_history_cache.qwen_encoder import QwenHistoryEncoder


class _FakeImageProcessor:
    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        batch = len(images)
        return {
            "pixel_values": torch.zeros(batch * 576, 8),
            "image_grid_thw": torch.tensor([[1, 24, 24]] * batch),
        }


class _FakeProcessor:
    image_processor = _FakeImageProcessor()


class _FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def get_image_features(self, *, pixel_values, image_grid_thw):
        self.calls += 1
        assert pixel_values.shape[0] == image_grid_thw.shape[0] * 576
        main = tuple(
            torch.arange(144 * 8, dtype=torch.float32).reshape(144, 8)
            for _ in range(image_grid_thw.shape[0])
        )
        return main, [torch.full((1,), float("nan"))]


def test_encoder_uses_only_get_image_features_main_output() -> None:
    encoder = QwenHistoryEncoder.__new__(QwenHistoryEncoder)
    encoder.device = torch.device("cpu")
    encoder.image_size = (384, 384)
    encoder.output_grid = (2, 2)
    encoder.expected_grid_thw = (1, 24, 24)
    encoder.expected_hidden_dim = 8
    encoder.spatial_merge_size = 2
    encoder.processor = _FakeProcessor()
    encoder.model = _FakeModel()

    frames = [np.zeros((384, 384, 3), dtype=np.uint8) for _ in range(2)]
    result = encoder.encode(frames)

    assert encoder.model.calls == 1
    assert result.shape == (2, 4, 8)
    assert result.dtype == np.float16
    assert np.isfinite(result).all()
