#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 sample 视频每一帧上叠加 planner 预测轨迹线。"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cache_gridpool import VisionCacheConfig, VisionFeatureCacher, grid_pool_tokens
from model import ModelConfig, OpenTrackVLA

DEFAULT_VIDEO = ROOT / "data/sample/IMG_7786.MOV"
DEFAULT_CKPT = Path("/nfsdata/zhl/zhangyi/OpenTrackVLA/outputs/stt/train_batch_size/best.pt")
DEFAULT_INSTRUCTION = "follow the person"
DEFAULT_FPS = 10
HISTORY = 31
DT = 0.1
FRAME_SIZE = 384


def resize_frame_rgb(rgb: np.ndarray, size: int = FRAME_SIZE) -> np.ndarray:
    """与 VisionFeatureCacher 一致，缩放到 size x size。"""
    from PIL import Image

    pil = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    pil = pil.resize((size, size), Image.BICUBIC)
    return np.asarray(pil)


def load_planner_model(ckpt_path: str, device: torch.device) -> OpenTrackVLA:
    obj = torch.load(ckpt_path, map_location=device)
    ck = obj if isinstance(obj, dict) else {}
    ck_cfg = ck.get("config", {})

    n_waypoints = int(ck_cfg.get("n_waypoints", 8))
    use_angle_tvi = bool(ck_cfg.get("use_angle_tvi", False))
    no_tanh_actions = bool(ck_cfg.get("no_tanh_actions", True))
    vision_feat_dim = int(ck_cfg.get("vision_feat_dim", 1536))
    alpha_xy = ck_cfg.get("alpha_xy", None)

    model = OpenTrackVLA(
        ModelConfig(
            n_waypoints=n_waypoints,
            beta_nav=float(ck_cfg.get("beta_nav", 10.0)),
            use_angle_tvi=use_angle_tvi,
            use_tanh_actions=(not no_tanh_actions),
            alpha_xy=alpha_xy,
        ),
        vision_feat_dim=vision_feat_dim,
    ).to(device).eval()

    msd = ck.get("model_state") or ck.get("model_state_dict")
    if msd:
        model.load_state_dict(msd, strict=False)
    return model


def render_frame_with_traj(rgb_frame_np: np.ndarray, traj_xyz: Optional[np.ndarray]) -> np.ndarray:
    """与 trained_agent._render_frame_with_traj 相同的轨迹绘制逻辑。"""
    try:
        if traj_xyz is None or not isinstance(traj_xyz, np.ndarray) or traj_xyz.size == 0:
            return rgb_frame_np
        from PIL import Image, ImageDraw

        img = Image.fromarray(rgb_frame_np.astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size
        base_x = w // 2
        base_y = int(h * 0.86)
        scale = 120.0
        pts = []
        npts = min(int(traj_xyz.shape[0]), 64)
        for i in range(npts):
            x = float(traj_xyz[i, 0])
            y = float(traj_xyz[i, 1]) if traj_xyz.shape[1] >= 2 else 0.0
            px = base_x - int(y * scale)
            py = base_y - int(x * scale)
            pts.append((px, py))
        for i in range(1, len(pts)):
            draw.line([pts[i - 1], pts[i]], fill=(0, 0, 0), width=8)
        for i in range(1, len(pts)):
            draw.line([pts[i - 1], pts[i]], fill=(0, 255, 180), width=8)
        if pts:
            r = 6
            sx, sy = pts[0]
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(0, 255, 0))
        return np.asarray(img)
    except Exception:
        return rgb_frame_np


class VideoPlanner:
    def __init__(
        self,
        ckpt_path: str,
        device: Optional[torch.device] = None,
        history: int = HISTORY,
        instruction: str = DEFAULT_INSTRUCTION,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = history
        self.instruction = instruction
        self.model = load_planner_model(ckpt_path, self.device)
        self._coarse_hist_tokens: deque = deque(maxlen=self.history)
        self._vision_cache: Optional[VisionFeatureCacher] = None

    def _ensure_vision_cache(self) -> Optional[VisionFeatureCacher]:
        if self._vision_cache is None:
            cfg = VisionCacheConfig(
                image_size=384,
                batch_size=1,
                device=str(self.device),
            )
            cache = VisionFeatureCacher(cfg)
            cache.eval()
            self._vision_cache = cache
        return self._vision_cache

    def _encode_frame_tokens(self, rgb_np: np.ndarray) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        enc = self._ensure_vision_cache()
        if enc is None:
            return None, None
        try:
            from PIL import Image

            pil = Image.fromarray(rgb_np.astype(np.uint8))
            tok_dino, Hp, Wp = enc._encode_dino([pil])
            tok_sigl = enc._encode_siglip([pil], out_hw=(Hp, Wp))
            vt_cat = torch.cat([tok_dino, tok_sigl], dim=-1)
            vfine = grid_pool_tokens(vt_cat, Hp, Wp, out_tokens=64)[0].float()
            vcoarse = grid_pool_tokens(vt_cat, Hp, Wp, out_tokens=4)[0].float()
            return vcoarse, vfine
        except Exception:
            return None, None

    @torch.inference_mode()
    def predict(self, rgb_frame_np: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
        vc, vf = self._encode_frame_tokens(rgb_frame_np)
        if vc is None or vf is None:
            return None, None

        self._coarse_hist_tokens.append(vc.cpu())
        hist = list(self._coarse_hist_tokens)
        if len(hist) < self.history:
            pad_needed = self.history - len(hist)
            first = hist[0] if hist else vc
            hist = [first] * pad_needed + hist
        else:
            hist = hist[-self.history :]

        coarse_list = []
        coarse_tidx = []
        for t, tok4 in enumerate(hist):
            tok4 = tok4.to(self.device)
            coarse_list.append(tok4)
            coarse_tidx.append(
                torch.full((tok4.size(0),), fill_value=t, dtype=torch.long, device=self.device)
            )
        coarse_tokens = torch.cat(coarse_list, dim=0).unsqueeze(0)
        coarse_tidx = torch.cat(coarse_tidx, dim=0).unsqueeze(0)

        fine_tokens = vf.to(self.device).unsqueeze(0)
        fine_tidx = torch.full(
            (1, fine_tokens.size(1)), fill_value=self.history, dtype=torch.long, device=self.device
        )
        instr = [self.instruction]

        tau = self.model(coarse_tokens, coarse_tidx, fine_tokens, fine_tidx, instr)
        traj = tau.detach().float().cpu().numpy()[0]

        wp0 = tau[0, 1]
        x, y = float(wp0[0].item()), float(wp0[1].item())
        theta = float(wp0[2].item()) if wp0.numel() >= 3 else 0.0
        vx = x / DT
        vy = y / DT
        wz = theta / DT
        action = [float(vx), float(vy), float(wz)]
        return traj, action


def read_video_frames(video_path: str, target_fps: float, max_frames: int = 0) -> Tuple[List[np.ndarray], float]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0:
        src_fps = 30.0
    sample_interval = max(1, int(round(src_fps / target_fps)))

    frames: List[np.ndarray] = []
    frame_idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_idx % sample_interval == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = resize_frame_rgb(rgb)
            frames.append(rgb)
            if max_frames > 0 and len(frames) >= max_frames:
                break
        frame_idx += 1
    cap.release()
    return frames, target_fps


def write_video(frames: List[np.ndarray], out_path: str, fps: float) -> None:
    import imageio

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="在视频帧上绘制 planner 预测轨迹")
    ap.add_argument("--video", type=str, default=str(DEFAULT_VIDEO), help="输入视频路径")
    ap.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT), help="轨迹规划模型 checkpoint")
    ap.add_argument("--output", type=str, default="", help="输出视频路径（默认与输入同目录，后缀 _traj.mp4）")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS, help="采样/输出帧率")
    ap.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION, help="语言指令")
    ap.add_argument("--history", type=int, default=HISTORY, help="coarse token 历史帧数")
    ap.add_argument("--max-frames", type=int, default=0, help="最多处理帧数，0 表示全部")
    ap.add_argument("--device", type=str, default="", help="cuda / cpu，默认自动选择")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    video_path = os.path.abspath(args.video)
    ckpt_path = os.path.abspath(args.ckpt)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"视频不存在: {video_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    if args.output:
        out_path = os.path.abspath(args.output)
    else:
        stem = Path(video_path).stem
        out_path = str(Path(video_path).with_name(f"{stem}_traj.mp4"))

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[draw_line] device={device}, ckpt={ckpt_path}", flush=True)
    print(f"[draw_line] video={video_path}, fps={args.fps}", flush=True)
    print("[draw_line] loading planner + vision encoder (may take a few minutes)...", flush=True)

    planner = VideoPlanner(
        ckpt_path=ckpt_path,
        device=device,
        history=args.history,
        instruction=args.instruction,
    )
    print("[draw_line] model ready", flush=True)

    frames, out_fps = read_video_frames(video_path, args.fps, max_frames=args.max_frames)
    print(
        f"[draw_line] sampled {len(frames)} frame(s) at {out_fps} fps, frame size {FRAME_SIZE}x{FRAME_SIZE}",
        flush=True,
    )

    try:
        from tqdm import trange
        frame_iter = trange(len(frames), desc="infer+draw")
    except ImportError:
        frame_iter = range(len(frames))

    out_frames: List[np.ndarray] = []
    for i in frame_iter:
        rgb = frames[i]
        traj, action = planner.predict(rgb)
        vis = render_frame_with_traj(rgb, traj)
        out_frames.append(vis)
        if action is not None:
            print(
                f"[frame {i:04d}] action vx={action[0]:.3f}, vy={action[1]:.3f}, wz={action[2]:.3f}",
                flush=True,
            )

    write_video(out_frames, out_path, out_fps)
    print(f"[draw_line] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
