# EVT-bench Qwen3-VL history cache generator

该工具实现 `QWEN_HISTORY_CACHE_GENERATION_PLAN.md` 中固定的第一版 profile：

```text
qwen3_vl_2b_pooled_history_4_mmap
```

它从 EVT-bench `front_image` MP4 读取 384×384 RGB 帧，调用
Qwen3-VL-2B-Instruct 的 `model.get_image_features()`，仅取已通过 final visual
merger 的主 `image_embeds`，再以 FP32 adaptive average pooling 从 12×12 压缩到
2×2，最终写为每个 ref `[4, 2048]` float16。代码不依赖 StarVLA、LeRobot、
MiniCPM、robot state、TVI、action placeholder 或 language hidden states。

## 环境

当前已验证的环境是：

```bash
PYTHON=/data1/yizhang/RLinf/.venv-habitat/bin/python
cd /data1/yizhang/cache
export PYTHONPATH=/data1/yizhang/cache
```

`requirements.txt` 只用于记录离线工具的直接依赖。若上述环境已有依赖，无需重复
安装。视频后端固定为 OpenCV，因此不要求 PyAV。

## 1. Metadata dry-run

此步骤不加载 Qwen。它会完成 refs → frame records → front video index 的一对一
join，检查所有视频存在、帧上界合法，并抽样解码真实帧；worklist 使用原子写入。

```bash
$PYTHON -m qwen_history_cache.cli \
  --config configs/qwen3_vl_2b_evt_bench.yaml \
  metadata --splits AT DT STT --video-check bounds --decode-samples 3
```

若 worklist 已存在，命令默认拒绝覆盖。只有在明确需要重建时才添加 `--force`。

## 2. GPU pilot

pilot 从多个视频各取样本，重复编码两次，验证 `[N, 4, 2048]`、float16、finite
和数值确定性。它只写独立 pilot 目录，不创建正式 cache profile。

```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON -m qwen_history_cache.cli \
  --config configs/qwen3_vl_2b_evt_bench.yaml \
  pilot --split DT --device cuda:0 --samples 8
```

## 3. 分片生成与 resume

单 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON -m qwen_history_cache.cli \
  --config configs/qwen3_vl_2b_evt_bench.yaml \
  generate --split DT --device cuda:0
```

生成时默认显示当前 rank 的 `ref` 进度、处理速度、预计剩余时间和当前 shard。
resume 时已完成 shard 会计入初始进度；如需写入纯净日志，可添加 `--no-progress`。

多 GPU 时每个进程获得唯一 `rank`，但都使用本进程可见的 `cuda:0`。rank 0 首次
计算 checkpoint SHA-256 并原子写入不可变 build spec，其余 rank 等待并复用该
fingerprint；每个 shard 的 owner 是 `shard_id % world_size`。

```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON -m qwen_history_cache.cli --config configs/qwen3_vl_2b_evt_bench.yaml generate --split DT --rank 0 --world-size 4 --device cuda:0 --no-publish
CUDA_VISIBLE_DEVICES=1 $PYTHON -m qwen_history_cache.cli --config configs/qwen3_vl_2b_evt_bench.yaml generate --split DT --rank 1 --world-size 4 --device cuda:0 --no-publish
CUDA_VISIBLE_DEVICES=2 $PYTHON -m qwen_history_cache.cli --config configs/qwen3_vl_2b_evt_bench.yaml generate --split DT --rank 2 --world-size 4 --device cuda:0 --no-publish
CUDA_VISIBLE_DEVICES=3 $PYTHON -m qwen_history_cache.cli --config configs/qwen3_vl_2b_evt_bench.yaml generate --split DT --rank 3 --world-size 4 --device cuda:0 --no-publish
```

这些命令应并行启动。所有进程结束后由一个进程检查并发布：

```bash
$PYTHON -m qwen_history_cache.cli \
  --config configs/qwen3_vl_2b_evt_bench.yaml status --split DT

$PYTHON -m qwen_history_cache.cli \
  --config configs/qwen3_vl_2b_evt_bench.yaml publish --split DT
```

中断后重新执行相同 `generate` 命令即可 resume。正式 `.npy` 会重新校验；
`.npy.partial` 从该 shard 的第 0 行重算。resume 可以改变 `world_size`，不会改变
shard 路径或行号。若 worklist、checkpoint stat、processor 或 profile 参数变化，
程序会拒绝混写。

## 4. 最终验证

```bash
$PYTHON -m qwen_history_cache.cli \
  --config configs/qwen3_vl_2b_evt_bench.yaml \
  validate --splits DT
```

验证默认重新计算 index 和所有 shard 的 SHA-256，并 mmap 分块检查 finite values。
`--skip-hashes` 只适合快速诊断，不能替代正式发布时的完整验证。

## 安全边界

- 正式 profile 已存在时，所有生成和发布命令都会拒绝覆盖。
- cache 先写入 `.qwen3_vl_2b_pooled_history_4_mmap.building`；只有全量 shard、
  index、manifest 和 hash 验证都通过，才原子 rename 为正式目录。
- `--max-shards` 仅用于小规模诊断，使用该选项时不会自动发布。
- 建议顺序是 metadata → 每个 split 的 pilot → DT 全量 → AT → STT。
