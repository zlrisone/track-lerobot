"""Qwen3-VL visual encoder used by offline cache generation and online RL."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from qwen_history_cache.pooling import pool_main_image_embeddings
from qwen_history_cache.validation import sha256_file, stable_json_sha256


def checkpoint_fingerprint(
    checkpoint_path: str | Path, *, revision: str = "main"
) -> dict[str, Any]:
    """Hash checkpoint configs and all local safetensors weight files."""
    root = Path(checkpoint_path).resolve()
    names = [
        "config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "model.safetensors.index.json",
    ]
    paths = [root / name for name in names if (root / name).is_file()]
    weight_paths = sorted(root.glob("*.safetensors"))
    if not weight_paths:
        raise FileNotFoundError(f"No safetensors weights found under {root}")
    paths.extend(weight_paths)
    files = {
        path.name: {
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256_file(path),
        }
        for path in paths
    }
    identity = {
        "checkpoint_path": str(root),
        "revision": revision,
        "files": files,
    }
    return {**identity, "aggregate_sha256": stable_json_sha256(identity)}


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        dtype = mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model dtype: {name}") from exc
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 Qwen inference on CPU is not supported by this tool")
    return dtype


def _processor_contract(
    processor: Any,
    vision_config: Any,
    *,
    image_size: tuple[int, int],
    expected_grid_thw: tuple[int, int, int],
    expected_patch_size: int,
    expected_temporal_patch_size: int,
    expected_spatial_merge_size: int,
    expected_hidden_dim: int,
) -> dict[str, Any]:
    image_processor = processor.image_processor
    actual = {
        "model_patch_size": int(vision_config.patch_size),
        "processor_patch_size": int(image_processor.patch_size),
        "model_temporal_patch_size": int(vision_config.temporal_patch_size),
        "processor_temporal_patch_size": int(image_processor.temporal_patch_size),
        "model_spatial_merge_size": int(vision_config.spatial_merge_size),
        "processor_merge_size": int(image_processor.merge_size),
        "model_out_hidden_size": int(vision_config.out_hidden_size),
    }
    expected = {
        "model_patch_size": expected_patch_size,
        "processor_patch_size": expected_patch_size,
        "model_temporal_patch_size": expected_temporal_patch_size,
        "processor_temporal_patch_size": expected_temporal_patch_size,
        "model_spatial_merge_size": expected_spatial_merge_size,
        "processor_merge_size": expected_spatial_merge_size,
        "model_out_hidden_size": expected_hidden_dim,
    }
    mismatches = {
        key: {"actual": actual[key], "expected": value}
        for key, value in expected.items()
        if actual[key] != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint/processor geometry mismatch: {mismatches}")
    return {
        "processor_class": type(processor).__name__,
        "image_processor_class": type(image_processor).__name__,
        "rgb_conversion": "OpenCV BGR decode -> cvtColor(BGR2RGB) -> PIL RGB",
        "resize": {
            "width": image_size[0],
            "height": image_size[1],
            "external_resample": "PIL.Image.Resampling.BICUBIC_if_needed",
        },
        "do_resize": bool(image_processor.do_resize),
        "processor_size": dict(image_processor.size),
        "do_rescale": bool(image_processor.do_rescale),
        "rescale_factor": float(image_processor.rescale_factor),
        "do_normalize": bool(image_processor.do_normalize),
        "image_mean": list(image_processor.image_mean),
        "image_std": list(image_processor.image_std),
        "do_convert_rgb": bool(image_processor.do_convert_rgb),
        "image_grid_thw": list(expected_grid_thw),
        "pre_merger_tokens": int(np.prod(expected_grid_thw)),
        "post_merger_grid": [
            expected_grid_thw[1] // expected_spatial_merge_size,
            expected_grid_thw[2] // expected_spatial_merge_size,
        ],
        "post_merger_tokens": int(
            np.prod(expected_grid_thw) // expected_spatial_merge_size**2
        ),
        **actual,
    }


def inspect_processor_contract(
    checkpoint_path: str | Path,
    *,
    image_size: tuple[int, int] = (384, 384),
    expected_grid_thw: tuple[int, int, int] = (1, 24, 24),
    expected_patch_size: int = 16,
    expected_temporal_patch_size: int = 2,
    expected_spatial_merge_size: int = 2,
    expected_hidden_dim: int = 2048,
) -> dict[str, Any]:
    """Validate processor/model configs without loading model weights."""
    from transformers import AutoConfig, AutoProcessor

    root = Path(checkpoint_path).resolve()
    processor = AutoProcessor.from_pretrained(root, local_files_only=True)
    config = AutoConfig.from_pretrained(root, local_files_only=True)
    return _processor_contract(
        processor,
        config.vision_config,
        image_size=image_size,
        expected_grid_thw=expected_grid_thw,
        expected_patch_size=expected_patch_size,
        expected_temporal_patch_size=expected_temporal_patch_size,
        expected_spatial_merge_size=expected_spatial_merge_size,
        expected_hidden_dim=expected_hidden_dim,
    )


class QwenHistoryEncoder:
    """Extract and spatially pool Qwen3-VL main image embeddings."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda:0",
        model_dtype: str = "auto",
        image_size: tuple[int, int] = (384, 384),
        output_grid: tuple[int, int] = (2, 2),
        expected_grid_thw: tuple[int, int, int] = (1, 24, 24),
        expected_patch_size: int = 16,
        expected_temporal_patch_size: int = 2,
        expected_spatial_merge_size: int = 2,
        expected_hidden_dim: int = 2048,
    ) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {device!r} was requested, but CUDA is unavailable"
            )
        self.model_dtype = _resolve_dtype(model_dtype, self.device)
        self.image_size = image_size
        self.output_grid = output_grid
        self.expected_grid_thw = expected_grid_thw
        self.expected_hidden_dim = expected_hidden_dim
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint_path, local_files_only=True
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.checkpoint_path,
            local_files_only=True,
            torch_dtype=self.model_dtype,
        ).to(self.device)
        self.model.eval()

        self._processor_contract = _processor_contract(
            self.processor,
            self.model.config.vision_config,
            image_size=image_size,
            expected_grid_thw=expected_grid_thw,
            expected_patch_size=expected_patch_size,
            expected_temporal_patch_size=expected_temporal_patch_size,
            expected_spatial_merge_size=expected_spatial_merge_size,
            expected_hidden_dim=expected_hidden_dim,
        )
        self.spatial_merge_size = expected_spatial_merge_size

    @staticmethod
    def _to_rgb_image(frame: np.ndarray, image_size: tuple[int, int]) -> Image.Image:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 RGB frame, got {frame.shape}")
        if frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB input, got {frame.dtype}")
        image = Image.fromarray(frame, mode="RGB")
        if image.size != image_size:
            image = image.resize(image_size, resample=Image.Resampling.BICUBIC)
        return image

    @torch.inference_mode()
    def encode(self, frames: list[np.ndarray]) -> np.ndarray:
        """Encode RGB frames to ``[batch, 4, 2048]`` float16 tokens."""
        if not frames:
            return np.empty(
                (
                    0,
                    self.output_grid[0] * self.output_grid[1],
                    self.expected_hidden_dim,
                ),
                dtype=np.float16,
            )
        images = [self._to_rgb_image(frame, self.image_size) for frame in frames]
        processed = self.processor.image_processor(images=images, return_tensors="pt")
        grid = processed["image_grid_thw"]
        expected_grid = torch.tensor(self.expected_grid_thw, dtype=grid.dtype)
        if grid.shape != (len(images), 3) or not torch.equal(
            grid.cpu(), expected_grid.unsqueeze(0).expand(len(images), -1)
        ):
            raise ValueError(
                f"Processor returned image_grid_thw={grid.tolist()}, expected "
                f"{[list(self.expected_grid_thw)] * len(images)}"
            )
        pixel_values = processed["pixel_values"].to(self.device)
        grid_device = grid.to(self.device)

        # Qwen3VL returns (main image embeddings split per image, deepstack
        # features). The main embeddings have already passed visual.merger.
        main_image_embeds, _deepstack_features = self.model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=grid_device,
        )
        if isinstance(main_image_embeds, torch.Tensor):
            split_size = (
                self.expected_grid_thw[0]
                * self.expected_grid_thw[1]
                * self.expected_grid_thw[2]
                // self.spatial_merge_size**2
            )
            main_image_embeds = torch.split(main_image_embeds, split_size)
        if len(main_image_embeds) != len(images):
            raise ValueError(
                f"Qwen returned {len(main_image_embeds)} image feature groups for "
                f"{len(images)} images"
            )
        pooled = [
            pool_main_image_embeddings(
                embedding,
                grid_device[index],
                spatial_merge_size=self.spatial_merge_size,
                output_grid=self.output_grid,
            )
            for index, embedding in enumerate(main_image_embeds)
        ]
        result = torch.stack(pooled).to(dtype=torch.float16, device="cpu").numpy()
        expected_shape = (
            len(images),
            self.output_grid[0] * self.output_grid[1],
            self.expected_hidden_dim,
        )
        if result.shape != expected_shape:
            raise ValueError(
                f"History encoder returned {result.shape}, expected {expected_shape}"
            )
        if not np.isfinite(result).all():
            raise ValueError("History encoder output contains NaN or Inf")
        return result

    def processor_metadata(self) -> dict[str, Any]:
        """Return the effective deterministic preprocessing contract."""
        return dict(self._processor_contract)


def runtime_versions() -> dict[str, Any]:
    """Collect software and accelerator versions for the manifest."""
    import cv2
    import PIL
    import pyarrow
    import transformers

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "pyarrow": pyarrow.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
