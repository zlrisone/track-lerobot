# Qwen3-VL EVT-bench History Cache 生成方案

> 状态：待确认，本文档不包含生成器代码，也不执行依赖安装或 cache 生成。  
> 代码计划位置：`/data1/yizhang/cache`  
> 数据集：`/data1/yizhang/data/evt-bench/{AT,DT,STT}`  
> Qwen checkpoint：`/data1/yizhang/Pretrained_model/Qwen3-VL-2B-Instruct`

## 1. 目标

为 EVT-bench 已有 history refs 生成与 Qwen3-VL-2B-Instruct 匹配的历史视觉 cache，供 RLinf 中的 `qwen_groot` 模型用于 SFT，并保证后续在线 RL 可以用同一编码规则实时生成 history token。

第一版的固定输出是：

```text
每个 history ref -> [4, 2048] float16
```

这些 token 是 Qwen 视觉编码器的主 `image_embeds`，即
`model.visual.merger` 输出，再做 `2 x 2` 空间池化得到。离线生成时不运行
Qwen language layers；训练和在线 RL 时必须将这些 cache token 插入 Qwen
`inputs_embeds` 序列，再与当前图像和指令一起经过 language layers。

## 2. 范围与边界

### 2.1 本工具负责

- 从 AT、DT、STT 的 `refs.parquet` 构建完整、可重现的 source-frame worklist。
- 通过 EVT-bench 自身 metadata 将 ref 定位到 `front_image` 视频及帧号。
- 使用本地 Qwen3-VL-2B-Instruct 的 processor、visual encoder 和 visual merger 生成 token。
- 进行多 GPU 分片、断点续跑、原子写入和完整性校验。
- 生成 RLinf reader 需要的 `manifest.json` 、`index.parquet` 和 mmap `.npy` shards。

### 2.2 第一版明确不做

- 不从 `/data1/yizhang/simplify/starVLA` 复用、import 或复制任何代码。
- 不运行 Qwen language layers，不缓存 prompt/instruction 相关 hidden state。
- 不使用 action placeholders、TVI、robot state、action、reward 或任务标签。
- 第一版不使用 Qwen deepstack 中间层输出；该决策需在实施前单独确认。
- 不读取 MiniCPM cache 作为输入，也不把 MiniCPM token 投影成 Qwen token。
- 不修改 RLinf 训练代码；本工具只生成 RLinf 消费的数据产物。
- 不在本轮安装依赖、下载模型、解码视频或写入 cache 产物。

## 3. 已确认的输入与规模

### 3.1 数据路径

```text
/data1/yizhang/data/evt-bench/
├── AT/
├── DT/
└── STT/
```

三个 split 都使用已有 context index 中的 history refs，不重新定义历史采样策略。

| Split | refs 数量 | `[N, 4, 2048]` FP16 约占用 |
| --- | ---: | ---: |
| AT | 870,903 | 13.29 GiB |
| DT | 808,182 | 12.33 GiB |
| STT | 1,171,091 | 17.87 GiB |
| 合计 | 2,850,176 | 43.49 GiB |

上述体积只是 embedding shard 的理论值，还需预留 index、worklist、临时 shard 和文件系统开销。实施前建议保留至少 100 GiB 可用空间，以支持 `.partial` 与最终文件的短暂共存。

### 3.2 Qwen 维度

本地 checkpoint 已确认：

```text
model type: qwen3_vl
text hidden size: 2048
vision hidden size: 1024
visual merger output size: 2048
spatial merge size: 2
```

因此 cache 取 `model.get_image_features()` 返回的主 `image_embeds`，也就是
`model.visual.merger` 之后的 2048 维 token。它本来就是 Qwen 用来替换
image placeholder、然后送入 language model 的视觉 embedding，不需要额外
`1024 -> 2048` projection。生成器不得对 `get_image_features()` 的结果再调用一次
merger。

## 4. History token 的精确定义

### 4.1 推荐的编码链路

```text
384 x 384 RGB front frame
  -> Qwen3-VL image processor
  -> image_grid_thw = [1, 24, 24]
  -> Qwen3-VL visual encoder
  -> model.visual.merger
  -> [12, 12, 2048]
  -> adaptive average pooling to [2, 2, 2048]
  -> flatten spatially to [4, 2048]
  -> cast/store as float16
```

4 个 token 的固定顺序为 row-major：

```text
0: top-left
1: top-right
2: bottom-left
3: bottom-right
```

不对单个 token 做额外 LayerNorm 或 L2 normalization。第一版 RLinf 也不在插入前
强制执行 history LayerNorm，以保持与 Qwen 主 `image_embeds` 输入路径一致。
若 pilot 显示 pooling 后分布需要校准，应将显式、可训练且可配置的 adapter 作为新的模型变体，
不改变离线 cache 数值。

### 4.2 为什么 cache 不取 language hidden state

- language hidden state 依赖 instruction、chat template、文本 tokenization 和图文排列，cache 不再是纯历史视觉表征。
- RLinf 中计划训练 Qwen language layers/LoRA。如果 cache 来自 language layers，训练后缓存立即与当前模型参数不一致。
- 后续在线 RL 只需跑相同的 visual encoder + merger + pooling，无需为每个历史帧运行整个 LLM。

“cache 不取 language hidden state”不等于“history 不进入 language layers”。离线 cache 仅保存
visual merger token；每次训练/rollout 时，这些 token 都会作为 Qwen 输入序列的一部分，
使用当前 language-layer 参数重新计算 contextualized hidden states。

这也意味着：使用该 cache 训练时，生成 cache 的 Qwen visual encoder 和
`model.visual.merger` 应保持冻结。如果将来解冻这两部分，必须同步重生成离线
cache，并更新在线 history encoder。language layers 可以训练，因为 history cache
会在每次 forward 中重新经过当前 language layers。

### 4.3 Processor 确定性

生成器必须显式记录并校验：

- RGB 转换规则。
- resize 后的实际宽高。
- processor 的 rescale/normalize 配置。
- 从 checkpoint model config 和 processor config 分别读取 patch/merge 参数，并断言两边一致。
- 当前 Qwen3-VL-2B profile 的预期值为 `patch_size=16`、`temporal_patch_size=2`、`spatial_merge_size/merge_size=2`；不得将这些值作为适用所有 Qwen checkpoint 的硬编码常量。
- `image_grid_thw`、merger 前后 token 数和 pooling 前形状。

第一个 batch 和每个 shard 都必须做 shape/finite-value 检查，不允许在 processor 输出尺寸变化时静默使用不同 pooling 网格。

### 4.4 History cache 在 Qwen 输入序列中的位置

本 cache profile 的 consumer 契约是 **pre-LLM fusion**，不是 action-head late fusion。

```text
history cache [B, Nh, 4, 2048]
  -> 按时间从旧到新展平为 [B, Nh*4, 2048]
  -> 作为 history prefix 插入 Qwen inputs_embeds

当前图像
  -> Qwen visual encoder + main merger
  -> 替换当前图像的 image placeholder embeddings

[history prefix, Qwen 当前图像+指令序列]
  -> combined attention_mask + explicit mRoPE position_ids
  -> Qwen language layers
  -> contextualized last_hidden_state
  -> GR00T flow-matching action head cross-attention
```

必须遵守以下约束：

- history token 位于当前图像和指令之前，使当前 token 在 causal attention 中可以关注所有历史。
- history block 按时间从旧到新排列；每块内部固定为 top-left、top-right、bottom-left、bottom-right。
- padding history 位置必须在 combined attention mask 中为 false，不参与任何 attention。
- 为 history token 构造明确的 Qwen3-VL 3-axis mRoPE 位置：时间轴表达 history-block 顺序，高/宽轴表达 `2 x 2` quadrant。当前 Qwen 序列的原生 mRoPE 位置保留，并按每个样本的有效 history 长度做无冲突偏移。
- history 是直接 embedding prefix，不增加 action placeholder，不增加 action token，不需要修改 Qwen vocabulary。
- action head 只消费 Qwen language model 的 contextualized `last_hidden_state` 和有效 mask；不再将 raw history cache 与 final hidden states 二次拼接。

当前 RLinf `HistoryConditioner -> cat([history_hidden, current_hidden])` 属于 late-fusion 实现，
不符合本契约，后续代码阶段必须改为 input-embedding injector。

### 4.5 Qwen3-VL DeepStack 边界

Qwen3-VL 的当前图像除主 `image_embeds` 外，还会在前几个 language layers 加入
deepstack visual features。如果 history cache 只保存本文定义的 4 个主 token，则：

- history positions 作为自定义 memory embeddings 进入 language model。
- history positions 不加入 `visual_pos_masks`，因此不要求 deepstack 残差。
- 当前图像仍使用 Qwen3-VL 原生 main image embeds + deepstack 路径。

这是第一版推荐：history 是压缩 memory token，不伪装成完整原生图像 token。
如果要求 history 与当前图像严格遵循相同 Qwen3-VL deepstack 路径，就必须额外缓存
3 组 `[4, 2048]` deepstack features，总 embedding 体积将从约 43.49 GiB 增加到约
173.96 GiB，并扩展 RLinf cache schema。这是实施前需用户确认的独立取舍。

## 5. Source-frame worklist

### 5.1 映射链路

不依赖旧 MiniCPM cache 的 index。每个 split 独立通过 EVT-bench metadata 构建 worklist：

1. 读取 `meta/context_index/budget_1024/refs.parquet`，保留 `ref_id`、`ref`、`episode_id`、`frame_index`、`camera_name`。
2. 读取 `meta/checkpoints/*_frame_records.parquet`，按 `(episode_id, frame_index)` 找到 `data_index`。
3. 读取 `meta/navvla_video_index.parquet`，只保留 `camera_name=front` 且 `available=true` 的记录，获得 `video_key`、`chunk_index`、`file_index`、`video_frame_index`。
4. 生成完整视频路径：

   ```text
   videos/{video_key}/chunk-{chunk_index:03d}/part-{file_index:03d}.mp4
   ```

5. 保持 refs 中的 `ref_id` 顺序，将 source 信息固化为 worklist。

### 5.2 全量生成前的强制检查

- 每个 `ref` 必须恰好映射到一个 source frame。
- `ref` 不可重复，`ref_id` 必须连续且唯一。
- join 前后行数必须完全一致。
- 不得出现缺失视频、负帧号或超过视频长度的帧号。
- 仅允许 front camera；其他 camera 值直接报错，不做静默替换。
- 为 worklist 生成 SHA-256，写入 manifest，保证 resume 时输入未变。

worklist 是中间产物，建议按 split 保存，以避免每次 resume 都重复扫描全部 metadata。

## 6. 图像解码与推理

### 6.1 解码策略

- 按 `video_path` 对 worklist 分组，同一视频的帧按 `video_frame_index` 升序读取，减少重复 seek/open。
- 解码后立即校验帧尺寸和 RGB 通道数；不使用视频中的其他相机 key。
- CPU 解码与 GPU 推理使用有界队列解耦，避免内存无上限增长。
- batch size 由运行参数控制，首次 pilot 用保守值测量 GPU 峰值显存后再调整。

解码失败不得写入零 token、NaN placeholder 或复制相邻帧。应记录明确的 split/ref/video/frame 定位信息并使当前 shard 失败。

### 6.2 精度

- 模型计算优先遵循 checkpoint 的 `bfloat16`，关键 pooling 可先在 FP32 中累加。
- 最终落盘统一转换为 `float16`。
- 每个 batch 检查 `isfinite`，不允许 NaN/Inf 进入已完成 shard。

## 7. 多 GPU、分片与断点续跑

### 7.1 确定性分片

每个 split 独立编号，默认：

```text
shard_size = 8192
shard_id   = ref_id // shard_size
row_index  = ref_id % shard_size
owner_rank = shard_id % world_size
```

shard 文件名不包含 rank：

```text
shards/image_embeds_000000.npy
shards/image_embeds_000001.npy
...
```

因此重启时可以改变 GPU 数量；新 world size 仅重新分配未完成 shard，不改变已生成文件的路径或 row index。

### 7.2 shard 写入协议

1. 一个 shard 只由一个 rank 写入。
2. 预分配精确 shape 的 NumPy `.npy` mmap，最后一个 shard 使用实际行数。
3. 先写入 `image_embeds_XXXXXX.npy.partial`。
4. 写完后 flush/fsync，关闭并以 read-only mmap 重新打开。
5. 校验 dtype、shape、文件大小、finite values 和内容 hash。
6. 通过同文件系统原子 rename 发布完成 shard。

仅存在且通过完整校验的正式 `.npy` 才视为已完成。`.partial` 文件可在 resume 时覆盖重算，不得直接改名为正式 shard。

### 7.3 构建与发布目录

每个 split 先写入隐藏的 building 目录：

```text
cache/visual_tokens/.qwen3_vl_2b_pooled_history_4_mmap.building/
```

全部 shard、index 和 manifest 都通过校验后，才原子重命名为：

```text
cache/visual_tokens/qwen3_vl_2b_pooled_history_4_mmap/
```

若正式目录已存在，默认立即拒绝运行；不自动删除、清空或覆盖任何已有 cache。

## 8. 输出数据契约

### 8.1 每个 split 的目录

```text
/data1/yizhang/data/evt-bench/<SPLIT>/cache/visual_tokens/
└── qwen3_vl_2b_pooled_history_4_mmap/
    ├── manifest.json
    ├── index.parquet
    └── shards/
        ├── image_embeds_000000.npy
        └── ...
```

代码、运行配置和可选的 worklist 存在 `/data1/yizhang/cache`；最终 cache 回写各 split 的数据目录，便于 RLinf 按 dataset root 定位。

### 8.2 `manifest.json`

RLinf 强制字段：

```json
{
  "encoder_family": "qwen3_vl",
  "encoder_revision": "main",
  "token_count": 4,
  "hidden_dim": 2048,
  "dtype": "float16",
  "file_format": "mmap_npy"
}
```

`encoder_revision: main` 是为了与当前 RLinf YAML 预案保持一致，但 `main` 本身不能唯一标识权重。manifest 还必须记录：

- checkpoint 绝对路径或用户指定的逻辑 ID。
- `config.json`、processor/preprocessor 配置和权重索引的 SHA-256。
- 可重现的 aggregate encoder fingerprint。
- `token_source=model.visual.merger:main_image_embeds`。
- pooling 方法、输入/输出网格和 token 顺序。
- processor 参数与实际 `image_grid_thw`。
- split、dataset root、ref 数量、worklist SHA-256。
- shard size、shard 数和每个 shard 的 hash/行数。
- Python、PyTorch、Transformers、NumPy、CUDA 及视频解码器版本。
- 生成时间和生成工具的 Git commit；如目录不在 Git 中，则记录主脚本 SHA-256。

### 8.3 `index.parquet`

每个 ref 一行，至少包含：

```text
ref: string
shard_path: string       # 相对 profile 目录
row_index: int
token_count: int         # 固定为 4
hidden_dim: int          # 固定为 2048
```

可附加 `ref_id`、source video/frame 和 shard hash 用于追溯，但 RLinf 不应依赖这些可选列。`shard_path` 必须使用 POSIX 相对路径，不把当前机器的绝对路径写入 reader 契约。

### 8.4 shard 形状

```text
full shard: [8192, 4, 2048], float16
last shard: [remaining_rows, 4, 2048], float16
one row:    [4, 2048], float16
```

`.npy` 必须能被 `numpy.load(path, mmap_mode="r")` 直接打开，不使用 object array、pickle 或额外压缩层。

## 9. Resume 和失败处理

启动或 resume 时，生成器先重建并校验运行 fingerprint：

```text
checkpoint fingerprint
+ processor fingerprint
+ token source/pooling spec
+ dataset split/worklist hash
+ dtype/token_count/hidden_dim/shard_size
```

仅在 fingerprint 完全一致时才允许 resume。不一致时拒绝混合旧新 shard，由用户手动选择新 profile/building 目录。

错误分类：

- metadata/join 错误：在加载模型前终止。
- 视频缺失或解码错误：当前 shard 失败，保留可定位错误日志。
- CUDA OOM：当前 `.partial` 不发布；调小 batch 后重算整个 shard。
- worker/rank 崩溃：其他已原子发布的 shard 保持可复用。
- index/manifest 失败：不发布 profile 目录。

不建议进行“跳过错误 ref 继续”，因为这会破坏 refs 全覆盖契约，并把数据问题推迟到训练期才暴露。

## 10. 验证计划

### 10.1 Stage A：metadata dry-run

不加载 Qwen，对 AT/DT/STT 全量执行 join 与文件存在性检查：

- refs 数量等于预期。
- join 一对一，无缺失和重复。
- 样本视频帧号可解码。
- 输出 worklist 和 hash，但不写 embedding。

### 10.2 Stage B：小样本 pilot

每个 split 抽取覆盖多个视频和不同帧位置的小样本：

- 验证 processor 与 visual merger 形状。
- 验证输出为 `[4, 2048]` FP16 且全部 finite。
- 对同一帧重复生成，比较输出一致性。
- 保存若干帧的 token 均值、标准差和 hash 作为 regression fixture。
- 测量显存、解码速度、GPU 利用率和预估总耗时。

pilot 通过且由用户确认 token 数值语义后，才进入全量生成。

### 10.3 Stage C：单 split 全量

建议先完整生成体量最小的 DT，检查：

- shard 数、行数总和与 refs 完全一致。
- index 中每行都能 mmap 到正确 shape。
- 随机抽样重算 token 与 cache 比较。
- 中断后使用不同 GPU 数 resume，已完成 shard 不改变。

DT 通过后再运行 AT 和 STT。

### 10.4 Stage D：RLinf consumer 验证

- RLinf reader 能加载 manifest/index 和 mmap shard。
- 任意 ref 返回 `[4, 2048]` FP16。
- batch 后形状为 `[B, Nh, 4, 2048]`，history mask 与 refs 数量一致。
- history 在 Qwen language model 之前插入 `inputs_embeds`，不在 Qwen final hidden states 之后拼接。
- 有效 history token 可被后续当前图像/文本 token 关注，padding history token 始终被 mask。
- history 和当前 Qwen token 共同经过所有 language layers，action head 仅接收一份 contextualized sequence。
- mRoPE 位置在不同 history 长度及 batch padding 下仍正确，无 position collision。
- checkpoint fingerprint/profile 不匹配时直接报错，不回退到 MiniCPM cache。
- 用同一帧调用未来的在线 history encoder，与离线 cache 按设定容差比较。

## 11. 未来代码组织（本轮不创建）

```text
/data1/yizhang/cache/
├── QWEN_HISTORY_CACHE_GENERATION_PLAN.md
├── README.md
├── requirements.txt
├── configs/
│   └── qwen3_vl_2b_evt_bench.yaml
├── qwen_history_cache/
│   ├── __init__.py
│   ├── cli.py
│   ├── metadata.py
│   ├── video_reader.py
│   ├── qwen_encoder.py
│   ├── pooling.py
│   ├── shard_writer.py
│   ├── manifest.py
│   └── validation.py
└── tests/
    ├── test_metadata.py
    ├── test_pooling.py
    ├── test_shard_writer.py
    └── test_resume.py
```

职责分开的目的是让“在线 RL history encoder”后续可直接复用 `qwen_encoder.py` 和 `pooling.py` 中的精确规则，而不复用离线 worklist/shard 逻辑。

RLinf 中的 input-embedding injector 不属于这个离线工具目录，但必须与本文 4.4 节的
token 顺序、mask 和 mRoPE 契约保持一致。

## 12. 计划依赖及安装理由

本轮不安装任何依赖。代码实施时会先检查 RLinf/Qwen 环境现有包，只补充缺失项，并用独立 requirements 记录而不改动 RLinf 的核心依赖。

| 依赖 | 用途 | 为什么需要 |
| --- | --- | --- |
| PyTorch | Qwen 权重加载与 GPU 推理 | Qwen visual encoder 的运行时 |
| Transformers（需支持 Qwen3-VL） | model/processor 实现 | 避免自行重写 Qwen 模型；版本必须经 pilot 锁定 |
| NumPy | mmap `.npy` shard | RLinf cache reader 的目标格式 |
| PyArrow | 读写 Parquet metadata/index | EVT-bench metadata 和 RLinf `index.parquet` 都是 Parquet |
| 视频解码库（PyAV 或环境已有的同等实现） | MP4 精确按帧解码 | 源图像位于 EVT-bench MP4；最终只选一种并锁版本 |
| Pillow | RGB 图像容器/processor 接口 | 与 Hugging Face image processor 的常用输入契约一致 |
| pytest | 单元和 resume 测试 | 防止分片、pooling 和断点续跑逻辑破坏数据 |

不需要 StarVLA、LeRobot、机器人环境、TVI 或 RL 运行时依赖。cache 生成是纯离线的“MP4 帧 -> Qwen 视觉 token”流程。

## 13. 实施顺序

1. 用户确认本文档中的 token 语义、pooling 和输出位置。
2. 只实现 metadata dry-run、worklist 与全量 join 校验。
3. 实现 Qwen visual encoder 抽取和 `12 x 12 -> 2 x 2` pooling。
4. 实现 shard writer、fingerprint、resume 和 atomic publish。
5. 补充单元测试，再运行小样本 pilot。
6. 用户检查 pilot 产物后，先生成 DT，然后 AT 和 STT。
7. 用 RLinf cache reader 做 consumer 端联调。

## 14. 实施前需确认的决策

请确认以下 5 点；确认后再开始写代码：

1. **Token 来源和插入点**：离线取 Qwen3-VL `model.visual.merger` 的主
   `image_embeds`，不缓存 language hidden state；训练/rollout 时将其作为 history
   prefix 插入 Qwen `inputs_embeds`，再经过 language layers。禁止在 final hidden states
   后做 late fusion。
2. **池化方式**：将 `12 x 12` visual tokens 用 adaptive average pooling 压缩到 `2 x 2`，得到 4 个 row-major token。
3. **位置边界**：生成工具位于 `/data1/yizhang/cache`，最终 cache 分别回写 AT/DT/STT 各自的 `cache/visual_tokens/qwen3_vl_2b_pooled_history_4_mmap/`。
4. **Revision 标识**：RLinf 契约字段保持 `encoder_revision: main`，同时用精确 checkpoint/processor SHA-256 和 aggregate fingerprint 防止权重混用。
5. **DeepStack**：第一版推荐只缓存主 `image_embeds`，history 作为自定义
   memory prefix 经过 language layers，但不接收 deepstack 残差；当前图像仍走原生
   deepstack。若要求 history 也使用 deepstack，需改用约 173.96 GiB 的扩展 cache
   profile，不再是当前 43.49 GiB 方案。

未得到上述确认前，不实现生成器，不生成 pilot 或全量 cache。
