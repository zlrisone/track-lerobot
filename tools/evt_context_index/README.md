# EVT-bench Qwen-GR00T context index

这个工具重新生成 RLinf Qwen-GR00T 使用的 EVT-bench compact context index。历史选择仍然是确定性的 sliding recent window，但预算中不再包含 TVI：

```text
total = current_visual_tokens + history_blocks * history_visual_tokens
      = 64 + N * 4
```

当 `token_budget=1024` 时，`N_max=(1024-64)//4=240`。当前帧不属于 history；每个样本选择同一 episode 内紧邻当前帧的最多 240 个历史帧，按旧到新写入。只生成 `front` 相机 block，long-memory 数组保持为空。

工具直接读取已有 EVT 转换产物中的：

```text
<split>/meta/checkpoints/*_frame_records.parquet
```

因此不需要重新读取源视频、重新转换 action，也不依赖旧 MiniCPM token。ref 字符串仍为 `<episode_id>/<frame_index:06d>/front`，可以继续匹配已有 Qwen history cache。

## 环境

已验证 RLinf Habitat 环境包含所需依赖：

```bash
PYTHON=/data1/yizhang/RLinf/.venv-habitat/bin/python
cd /data1/yizhang/context_index
export PYTHONPATH=$PWD
```

也可以安装本目录记录的依赖：

```bash
python -m pip install -e '.[test]'
```

## 1. 只读检查

先统计 AT、DT、STT 的帧数、新 history block 数量和预计数组空间，不写文件：

```bash
$PYTHON -m evt_context_index.cli inspect \
  --evt-root /data1/yizhang/data/evt-bench \
  --splits AT DT STT
```

## 2. 生成并发布

现有数据已经有 `budget_1024`，所以必须显式传 `--replace`：

```bash
$PYTHON -m evt_context_index.cli build \
  --evt-root /data1/yizhang/data/evt-bench \
  --splits AT DT STT \
  --token-budget 1024 \
  --current-visual-tokens 64 \
  --history-visual-tokens 4 \
  --replace
```

每个 split 先写隐藏 staging 目录并完成全量格式/预算验证，验证通过后才发布。旧索引不会删除，而是移动到：

```text
<split>/meta/context_index_backups/budget_1024_<UTC timestamp>/
```

发布内容保持 RLinf reader 所需布局：

```text
meta/context_index/budget_1024/
├── context_meta.parquet
├── refs.parquet
└── context_arrays/*.npy
```

manifest 会新增 `budget_model=qwen_groot_no_tvi_v1`，明确记录 `tvi_tokens=0`。`current_tvi_time` 列仅为旧 compact schema 兼容而保留，不参与预算，也不会被 RLinf Qwen-GR00T dataset 读取。

## 3. 再次验证

```bash
$PYTHON -m evt_context_index.cli validate \
  --evt-root /data1/yizhang/data/evt-bench \
  --splits AT DT STT
```

验证包括：行数、offset 连续性、数组 dtype/长度、仅 front camera、每行 `64+4N<=1024`、TVI cost 为 0，以及 long-memory 为空。

## 测试

```bash
PYTHONPATH=$PWD $PYTHON -m pytest -q
```

