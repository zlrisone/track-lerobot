# Track Rollout Data Generation

本仓库基于 Habitat-Lab/Habitat-Sim 生成多行人场景中的目标人物跟随 rollout。机器人使用连续 Oracle Teacher 产生底盘速度，经 RVO2 动态避障和主行人近距离安全护栏修正后执行，并保存逐步状态、动作、指标和四视角 RGB 视频。

仓库聚焦 rollout 数据采集与数据转换，不包含模型训练、checkpoint 转换或训练后模型评测代码。

## 功能概览

- 支持 `AT`、`DT`、`STT` 三套 tracking dataset/config。
- 使用 Spot 机器人和一个主行人目标，场景中可包含多个其他行人。
- Oracle Teacher 结合行人速度估计、跟随目标平滑、NavMesh 路径和速度前馈生成连续动作。
- 默认使用 RVO2 对主行人及其他行人进行动态避障。
- 在动作执行前应用预测式主行人近距离安全护栏。
- 保存前、左、右、后四路 `384 × 384` RGB 视频。
- 支持多 GPU split 调度、断点续跑和失败 split 自动重试。
- 可将四视角 rollout 转换为独立、分片的 LeRobot v3 数据集。

## 仓库结构

```text
.
├── baseline_agent.py       # rollout 主循环、RVO2、安全护栏、数据与视频保存
├── run.py                  # 单个 dataset split 的启动入口
├── train_data.sh           # AT/DT/STT 统一多 GPU 采集入口
├── convert_rollout_to_lerobot.py # 四视角 rollout → LeRobot v3 CLI
├── track_lerobot/          # 独立转换、schema 和 context index 实现
├── requirements-lerobot.txt # 数据转换所需依赖
├── make_tracking_data.py   # 可选的逐帧 JSON/JSONL 转换工具
├── evt_bench/              # 自定义 action、sensor、metric 和 simulator 注册
├── habitat-lab/            # 仓库内 Habitat-Lab 实现与 tracking 配置
├── data/                   # dataset、场景、humanoid 和机器人资源目录
├── humanoid_infos.json     # humanoid 名称与 semantic id 映射
└── scripts/                # 辅助脚本
```

Tracking benchmark 配置位于：

```text
habitat-lab/habitat/config/benchmark/nav/track/
```

rollout 使用以下三份配置：

```text
track_train_at.yaml
track_train_dt.yaml
track_train_stt.yaml
```

## 环境准备

### Python 与系统依赖

建议使用 Habitat-Lab 支持的 Python 3.9–3.11，并安装与 CUDA/驱动匹配的 Habitat-Sim。随后安装仓库内 Habitat-Lab 依赖：

```bash
python -m pip install -r habitat-lab/requirements.txt
```

rollout 默认开启 RVO2，因此还需要提供可被 `import rvo2` 加载的 Python-RVO2/pyrvo2 安装。视频保存和数据转换需要 `ffmpeg`。

运行时使用仓库内 Habitat-Lab：

```bash
export PYTHONPATH="habitat-lab:${PYTHONPATH}"
```

LeRobot 转换器不依赖 Habitat-Lab，也不依赖其他仓库。可单独安装转换依赖：

```bash
python -m pip install -r requirements-lerobot.txt
```

### 数据资源

大体积 dataset、场景和 humanoid 资源需要单独准备。三种任务的默认 dataset 路径为：

```text
data/datasets/track/AT/train/train.json.gz
data/datasets/track/DT/train/train.json.gz
data/datasets/track/STT/train/train.json.gz
```

此外需要：

- dataset episode 的 `scene_id` 所引用的场景资源；
- benchmark YAML 中 `scene_dataset` 指向的 scene dataset config；
- `data/humanoids/humanoid_data/<name>/` 下的 humanoid URDF 与动作数据；
- `data/robots/hab_spot_arm/` 下的 Spot URDF 与 mesh。

三份 rollout YAML 的 `scene_dataset` 均设置为：

```text
data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json
```

实际使用的数据集还可能通过 episode `scene_id` 引用 HM3D 等场景路径；发布或部署数据时应确保所有引用均能从 Habitat 数据根目录解析。

### GPU 与 EGL

Habitat-Sim GPU 渲染通常要求 CUDA 与 EGL 绑定到同一张物理 GPU：

```bash
export CUDA_VISIBLE_DEVICES=0
export EGL_DEVICE_ID=0
export EGL_VISIBLE_DEVICES=0
```

`train_data.sh` 会为每个并行槽位设置：

```text
CUDA_VISIBLE_DEVICES=<slot id>
EGL_DEVICE_ID=0
EGL_VISIBLE_DEVICES=<slot id>
```

因此默认使用编号 `0..NUM_PARALLEL-1` 的 GPU。

`train_data.sh` 会在启动前检查 `nvidia-smi`。推荐显式使用指定虚拟环境和 GPU 列表：

```bash
PYTHON_BIN=/data1/yizhang/RLinf/.venv-habitat/bin/python \
GPU_IDS="0 1 2 3" \
NUM_PARALLEL=4 \
TASK=stt bash train_data.sh
```

也支持逗号格式或 Slurm/CUDA 注入的 token：

```bash
GPU_IDS="0,2" NUM_PARALLEL=2 TASK=stt bash train_data.sh
CUDA_VISIBLE_DEVICES="GPU-xxxxxxxx,GPU-yyyyyyyy" NUM_PARALLEL=2 TASK=stt bash train_data.sh
```

每个子进程只看到一张卡，因此脚本内部统一设置 `EGL_DEVICE_ID=0`；`EGL_VISIBLE_DEVICES` 和 `CUDA_VISIBLE_DEVICES` 使用同一个 GPU token。若启动前提示 `nvidia-smi cannot see any GPU`，说明当前节点或容器没有 GPU 设备透传，需先申请 GPU 节点或使用 NVIDIA Container Toolkit/Apptainer `--nv`。

如果 `nvidia-smi` 正常、PyTorch 能看到 GPU，但出现：

```text
unable to find CUDA device 0 among N EGL devices
```

检查 NVIDIA EGL 用户态库：

```bash
ls -l /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ldconfig -p | grep libEGL_nvidia
```

当前节点的内核驱动版本是 `550.54.15`。需要由管理员安装同版本的 NVIDIA GL/EGL 用户态库（通常由 `libnvidia-gl-550` 提供），或在容器启动时使用 NVIDIA Container Toolkit/Apptainer `--nv` 注入这些库。不要直接安装与内核驱动不匹配的其他版本。`train_data.sh` 会通过 `EGL_VENDOR_FILE` 和 `__EGL_VENDOR_LIBRARY_FILENAMES` 指定 NVIDIA EGL vendor 文件，并在启动前检查库是否存在。

注意：如果通过 Codex、远程执行器或其他 sandbox 检查 GPU，执行器可能使用独立的 `/dev` 命名空间，看到的设备列表不代表你登录 shell 的宿主机视图。应在实际运行 `train_data.sh` 的 shell 中执行 `nvidia-smi -L` 和 `ls -l /dev/nvidia*`。

脚本默认 `GPU_PRECHECK=1`，会在创建子进程前执行 `nvidia-smi -L`。仅在已经确认 Habitat-Sim 能通过其他方式访问 GPU 时，才可关闭检查：

```bash
GPU_PRECHECK=0 TASK=stt bash train_data.sh
```

## 快速开始

### 多 GPU 采集

统一入口通过 `TASK` 选择任务：

```bash
TASK=stt bash train_data.sh
TASK=dt bash train_data.sh
TASK=at bash train_data.sh
```

脚本默认参数为：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `TASK` | `at` | `at`、`dt` 或 `stt` |
| `CHUNKS` | `900` | 将 dataset 划分成的 split 数量 |
| `NUM_PARALLEL` | `8` | 并行 GPU/进程数量 |
| `SEED` | `101` | Habitat simulator seed |
| `SAVE_PATH` | `sim_data/v4-new/<task>/train/seed_<seed>` | 输出目录 |
| `FORCE_FULL` | `0` | 设为 `1` 时忽略失败列表并全量执行 |
| `DATA_QUIET` | `0` | 设为 `1` 时减少逐 step 输出 |

单 GPU 示例：

```bash
TASK=stt NUM_PARALLEL=1 CHUNKS=900 bash train_data.sh
```

自定义输出与 seed：

```bash
TASK=dt \
SEED=202 \
SAVE_PATH=sim_data/dt/train/seed_202 \
NUM_PARALLEL=4 \
bash train_data.sh
```

Hydra 配置覆盖可通过 `EXTRA_OPTS` 传入：

```bash
TASK=at \
EXTRA_OPTS='habitat.environment.max_episode_steps=100' \
bash train_data.sh
```

脚本需要 Bash 4.3 或更高版本，以使用 `wait -n` 管理 GPU 槽位。

### 单 split 采集

```bash
CUDA_VISIBLE_DEVICES=0 \
EGL_DEVICE_ID=0 \
EGL_VISIBLE_DEVICES=0 \
PYTHONPATH="habitat-lab" \
python run.py \
  --split-num 900 \
  --split-id 0 \
  --exp-config habitat-lab/habitat/config/benchmark/nav/track/track_train_stt.yaml \
  --run-type eval \
  --save-path sim_data/stt/train/seed_101 \
  habitat.simulator.seed=101
```

`run.py` 只实现 `--run-type eval` 分支。这里的 `eval` 是 Habitat rollout 执行模式，不表示运行训练后模型评测。

可选参数 `--target-id` 可指定目标人物 semantic id；`--target-name` 会通过 `humanoid_infos.json` 解析对应 id。

### 断点续跑与失败重试

对每个已保存 episode，程序检查：

- `<episode_id>.json`；
- `<episode_id>_info.json`；
- 四个视角的 MP4 文件。

文件完整时，重复运行相同输出目录会跳过该 episode。

`train_data.sh` 将进程级失败记录到：

```text
<save_path>/failed_splits.txt
```

再次执行相同 `TASK` 和 `SAVE_PATH` 时，脚本会读取最后一行失败 split 列表并自动进入 retry 模式。也可以手动指定：

```bash
TASK=stt FAILED_SPLIT_IDS="3 18 42" bash train_data.sh
```

强制重新运行全部 split：

```bash
TASK=stt FORCE_FULL=1 bash train_data.sh
```

## 任务配置

三种任务共享 10 Hz 控制频率和相同的机器人速度上限，主要区别如下：

| 配置 | Dataset | 最大 step | 主目标检测像素阈值 | 其他行人检测像素阈值 | NavMesh footprint | Teacher 软安全距离/增益 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `STT` | `data/datasets/track/STT/train/train.json.gz` | 500 | 40 | 40 | `±0.18 m` | `1.2 m / 6.0` |
| `DT` | `data/datasets/track/DT/train/train.json.gz` | 500 | 1000 | 10000 | `±0.25 m` | `0.85 m / 4.0` |
| `AT` | `data/datasets/track/AT/train/train.json.gz` | 300 | 1000 | 10000 | `±0.25 m` | `0.85 m / 4.0` |

任务语义、目标人物、其他行人、自然语言指令和 waypoint 由对应 dataset episode 的 `info` 字段提供。

## Rollout 控制链路

```text
主行人状态
    ↓
连续 Oracle Teacher
    ↓
RVO2 动态避障
    ↓
预测式主行人近距离安全护栏
    ↓
BaseVelNonCylinderAction
    ↓
Habitat 仿真与逐步数据记录
```

### Oracle Teacher

Oracle Teacher 实现在 `evt_bench/additional_action.py` 的 `OracleNavCoordinateActionForRobot` 中。

每个控制 step 执行以下计算：

1. 根据主行人相邻位置估计世界 XZ 平面速度，并使用 EMA 平滑。
2. 在主行人运动方向后方 `1.5 m` 构造跟随目标，并对目标位置使用 EMA 平滑。
3. 将目标投影到 NavMesh，使用与底盘执行器一致的 footprint 检查可达性。
4. 缓存短期路径，从机器人在路径上的投影点向前选择 lookahead point。
5. 使用路径位置反馈与主行人速度前馈计算期望世界速度。
6. 距离主行人过近时加入径向退让速度。
7. 根据机器人朝向与主行人方向的夹角生成连续 yaw 速度。
8. 将世界速度投影到纵向/横向非对称速度边界，并施加物理加速度限制。

主要参数来自三份 `track_train_*.yaml`：

| 参数 | 值 |
| --- | ---: |
| 期望跟随距离 `dist_thresh` | `1.5 m` |
| 位置反馈增益 `follow_position_gain` | `1.4` |
| 主行人速度前馈 `human_velocity_feedforward` | `0.9` |
| 主行人速度 EMA `human_velocity_ema_alpha` | `0.35` |
| 跟随目标 EMA `follow_target_ema_alpha` | `0.4` |
| 最小有效行人运动速度 | `0.08 m/s` |
| 最大跟踪行人速度 | `2.5 m/s` |
| 路径 lookahead | `0.8 m` |
| 路径重规划间隔 | `3 steps` |
| 目标移动重规划阈值 | `0.25 m` |
| yaw 增益 `follow_yaw_gain` | `2.2` |
| yaw deadband | `0.025 rad` |
| 最大纵向加速度 | `3.0 m/s²` |
| 最大横向加速度 | `2.5 m/s²` |
| 最大 yaw 加速度 | `6.0 rad/s²` |

Teacher 对其他行人的势场排斥默认关闭：

```text
TEACHER_PED_REPULSE=0
```

如需启用，可设置 `TEACHER_PED_REPULSE=1`；`train_data.sh` 提供的半径和增益默认分别为 `2.5 m` 与 `0.8`。

### RVO2 动态避障

RVO2 在 `baseline_agent.py` 中运行。它将 Teacher 的本体速度转换到世界 XZ 平面，计算安全速度，再转换回底盘纵向/横向动作。RVO2 修正平移速度，yaw 保留 Teacher 输出。

`train_data.sh` 的默认参数：

| 环境变量 | 默认值 | 说明 |
| --- | ---: | --- |
| `DYN_OBSTACLE_AVOID` | `1` | RVO2 总开关 |
| `AVOID_OTHER_PEDESTRIANS` | `1` | 同时避让其他行人；设为 `0` 时只处理主行人 |
| `RVO_NEIGHBOR_DIST` | `2.2 m` | 邻居搜索距离 |
| `RVO_TIME_HORIZON` | `1.5 s` | ORCA time horizon |
| `RVO_AGENT_RADIUS` | `0.35 m` | 基础 agent 半径 |
| `LEADER_RVO_RADIUS_SCALE` | `1.0` | 主行人半径缩放 |
| `OTHER_RVO_RADIUS_SCALE` | `0.8` | 其他行人半径缩放 |
| `RVO_MAX_SPEED` | `2.0 m/s` | RVO 最大平移速度 |
| `RVO_VELOCITY_EMA_ALPHA` | `0.35` | RVO 输入速度 EMA |

如未安装 RVO2，可在排查环境时临时关闭：

```bash
TASK=stt DYN_OBSTACLE_AVOID=0 bash train_data.sh
```

### 主行人近距离安全护栏

RVO2 之后还会执行一次主行人专用安全投影。护栏使用主行人径向速度预测短期距离，限制机器人朝向主行人的闭合速度；进入紧急区时强制径向退让，并限制切向速度和 yaw 速度。

| 环境变量 | 默认值 |
| --- | ---: |
| `MAIN_HUMAN_PROXIMITY_GUARD` | `1` |
| `MAIN_HUMAN_GUARD_START_DISTANCE` | `1.8 m` |
| `MAIN_HUMAN_GUARD_STOP_DISTANCE` | `1.0 m` |
| `MAIN_HUMAN_GUARD_APPROACH_GAIN` | `1.5` |
| `MAIN_HUMAN_GUARD_CLOSING_MARGIN` | `0.25 m/s` |
| `MAIN_HUMAN_GUARD_PREDICTION_HORIZON` | `0.25 s` |
| `MAIN_HUMAN_GUARD_EMERGENCY_DISTANCE` | `1.15 m` |
| `MAIN_HUMAN_GUARD_MIN_RETREAT_SPEED` | `0.8 m/s` |
| `MAIN_HUMAN_GUARD_MAX_TANGENT_SPEED` | `0.15 m/s` |
| `MAIN_HUMAN_GUARD_MAX_YAW_SPEED` | `0.5 rad/s` |

实际参数会写入 episode 结果 JSON 的 `main_human_guard_config` 字段。

## 动作定义

### `base_velocity`

机器人动作是三维归一化本体速度：

```text
base_velocity = [v_forward, v_lateral, yaw_rate]
```

| 维度 | 正方向 | 负方向 |
| --- | --- | --- |
| `v_forward` | 向前 | 向后 |
| `v_lateral` | 向左平移 | 向右平移 |
| `yaw_rate` | 向左转/俯视逆时针 | 向右转/俯视顺时针 |

执行器会先将每一维裁剪到 `[-1, 1]`。三种 rollout 配置使用相同物理上限：

| 维度 | 物理上限 |
| --- | ---: |
| 纵向 | `2.0 m/s` |
| 横向 | `1.0 m/s` |
| yaw | `2.5 rad/s` |

物理速度换算为：

```text
v_forward_physical = clip(v_forward, -1, 1) × 2.0
v_lateral_physical = clip(v_lateral, -1, 1) × 1.0
yaw_rate_physical  = clip(yaw_rate, -1, 1) × 2.5
```

`BaseVelNonCylinderAction` 设置的局部速度为：

```text
linear_velocity  = [v_forward_physical, 0, -v_lateral_physical]
angular_velocity = [0, yaw_rate_physical, 0]
```

YAML 中 Oracle Teacher 与 `agent_1_base_velocity` 的三个速度上限必须一致；rollout 在每个 episode 开始时进行检查，不一致会抛出 `ValueError`。

### 逐步动作字段

`<episode_id>_info.json` 中每一行对应一个视频 frame，主要动作字段为：

| 字段 | 含义 |
| --- | --- |
| `base_velocity_raw` | Oracle Teacher 输出的归一化动作，尚未经过 RVO2 和最终安全护栏 |
| `base_velocity` | 实际传给底盘执行器的归一化动作 |
| `base_velocity_physical` | 实际动作反归一化后的 `[m/s, m/s, rad/s]` |
| `target_pose` | `[v_forward × dt, v_lateral × dt, yaw_rate × dt]`，使用归一化动作 |
| `action` | `[target_pose_x, target_pose_y, 0, target_pose_yaw]` |
| `dyn_obs_correction` | RVO2 世界平面速度修正幅度 |
| `main_human_guard_correction` | 主行人安全护栏修正幅度 |

`target_pose` 和 `action` 使用归一化速度乘以控制周期，并不是严格的米和弧度物理位移。

## 四视角 RGB 相机

四个相机均挂载在 Spot base，`attached_link_id=-1`，图像参数均为：

```text
height = 384
width  = 384
hfov   = 90°
```

相机位置均为：

```text
cam_offset_pos = [0.24, 0.24, 0.0]
```

| 视角 | Observation key | `cam_orientation` | 相对机器人 front 的 yaw | 输出文件 |
| --- | --- | ---: | ---: | --- |
| front | `agent_1_articulated_agent_jaw_rgb` | `[0, -1.571, 0]` | `0` | `<id>_front.mp4` |
| left | `agent_1_articulated_agent_left_rgb` | `[0, 0, 0]` | `+1.571` | `<id>_left.mp4` |
| rear | `agent_1_articulated_agent_back_rgb` | `[0, 1.571, 0]` | `+3.142` | `<id>_rear.mp4` |
| right | `agent_1_articulated_agent_right_rgb` | `[0, 3.142, 0]` | `-1.571` | `<id>_right.mp4` |

视频帧率与 simulator `ctrl_freq` 一致，三份 rollout 配置均为 `10 FPS`。采集器只保存带视角后缀的四个视频，不额外生成 `<episode_id>.mp4`。

相机注册位于：

```text
habitat-lab/habitat/articulated_agents/robots/spot_robot.py
```

图像参数位于：

```text
habitat-lab/habitat/config/habitat/simulator/sensor_setups/spot_agent_simplified.yaml
```

## Episode 保存条件

非 debug 模式下，episode 只有同时满足以下条件才会保存 JSON、逐步信息和四路视频：

- 至少记录一个 step；
- `status == "Normal"`；
- episode 最终满足跟随条件或最后一个 step 处于有效跟随状态；
- 最终主目标 detector 可见；
- `following_rate >= 0.5`；
- 未被主行人 stuck filter 丢弃。

有效跟随 metric 要求机器人与主行人距离不超过 `3.0 m`，并满足几何视野或 detector 可见。`human_collision` 在机器人与主行人距离小于 `0.4 m` 后保持为真。

rollout 还会提前终止以下异常 episode：

- 未处于有效跟随状态且距离大于 `4.0 m` 连续超过 20 step；
- 检测到主行人碰撞；
- 主行人在 XZ 平面的单步 L1 位移小于 `1e-4 m`，连续达到默认 60 step。

stuck filter 可通过以下变量调整：

```text
HUMAN_STUCK_FILTER=1
HUMAN_STUCK_XZ_EPS=1e-4
HUMAN_STUCK_MIN_STEPS=60
```

设置 `DATA_DEBUG=1` 后，只要 episode 产生了 step，就会保存完整轨迹与视频；成功判定和失败日志仍会照常计算。

## 输出格式

### 成功 episode

```text
<save_path>/<scene_key>/
├── <episode_id>.json
├── <episode_id>_info.json
├── <episode_id>_front.mp4
├── <episode_id>_left.mp4
├── <episode_id>_right.mp4
└── <episode_id>_rear.mp4
```

`<episode_id>.json` 包含：

- episode 状态、成功标记与自然语言指令；
- `following_rate`、`following_step`、`total_step` 和碰撞信息；
- scene、episode、控制频率和避障开关；
- 底盘速度上限；
- 主行人安全护栏参数。

`<episode_id>_info.json` 是逐 step 数组，视频 frame 与数组下标一一对应。每一步包括：

- frame index、timestamp 和 simulator step；
- 机器人与主行人执行前/后的世界位置和 yaw；
- 原始动作、实际动作、物理速度和单步 target pose；
- 跟随、距离和碰撞指标；
- RVO2 与主行人安全护栏诊断字段。

### 运行日志

```text
<save_path>/runtime_logs/split_<split_id>.log
<save_path>/runtime_logs/split_<split_id>_episode_failures.jsonl
<save_path>/failed_splits.txt
```

episode 失败日志包含稳定 reason code、距离统计、护栏统计以及失败前若干 step。常见 reason code 包括：

```text
lost_main_human
main_human_stuck
human_collision
empty_trajectory
episode_incomplete
following_not_achieved
target_not_visible_at_end
low_following_rate
```

可通过 `ROLLOUT_FAILURE_TAIL_STEPS` 调整失败日志中保留的末尾 step 数，默认值为 `20`。

## 转换为 LeRobot v3

`convert_rollout_to_lerobot.py` 将本仓库直接生成的四视角 rollout 转换为 LeRobot v3 数据集。转换实现全部位于本项目的 `track_lerobot/`，运行时不引用其他项目代码。

### 输入结构

`--source-root` 应直接指向包含场景子目录的某个 rollout 根目录，例如 `seed_101`：

```text
<source_root>/
└── <scene_id>/
    ├── <episode_id>.json
    ├── <episode_id>_info.json
    ├── <episode_id>_front.mp4
    ├── <episode_id>_left.mp4
    ├── <episode_id>_right.mp4
    └── <episode_id>_rear.mp4
```

转换器要求一个 episode 的六个文件全部存在，并检查 JSON、四路视频帧数、FPS、分辨率以及跨 episode 的相机尺寸一致性。四路带视角后缀的视频会被直接读取，不需要建立软链接或复制视频。

### 基本用法

```bash
python convert_rollout_to_lerobot.py \
  --source-root sim_data/v4-new/stt/train/seed_101 \
  --output-root data/lerobot \
  --dataset-name track-stt-seed101 \
  --skip-invalid
```

实际数据集写入：

```text
data/lerobot/track-stt-seed101/
```

小规模检查可限制 episode 数量，并让每个 shard 只包含一个 episode：

```bash
python convert_rollout_to_lerobot.py \
  --source-root /path/to/seed_101 \
  --output-root /tmp/track-lerobot \
  --dataset-name track-smoke \
  --max-episodes 2 \
  --episodes-per-file 1 \
  --context-token-budgets 512 \
  --skip-invalid \
  --overwrite
```

### Action 与 state 坐标

Action 使用 `_info.json` 中实际执行的归一化速度：

```text
base_velocity = [vx_source, vy_source, omega_source]
```

先转换为目标坐标约定：

```text
vx_target    =  vx_source
vy_target    = -vy_source
omega_target = -omega_source
```

目标坐标中 `+x` 向前、`+y` 向右、`+yaw` 表示右转。转换器从当前帧开始，使用累计 yaw 将每一步本体速度旋转到当前帧的初始局部坐标系：

```text
x   += (cos(yaw) × vx - sin(yaw) × vy) × dt
y   += (sin(yaw) × vx + cos(yaw) × vy) × dt
yaw += omega × dt
```

每帧的输出 action 为：

```text
shape = [action_horizon, 4]
value = [x, y, 0, yaw]
```

默认 `action_horizon=8`、`dt=0.1`。episode 尾部不足 horizon 时重复最后一个有效累计位姿，`action.padding_mask` 全部为 `False`。这里积分的是归一化速度，因此 action 是统一坐标下的局部轨迹表示，不是按底盘物理速度上限换算后的米制轨迹。

绝对 observation state 来自执行动作前的机器人状态：

```text
robot_pos = [world_x, world_y_height, world_z]
observation.state = [world_x, -world_z, world_y_height, -world_yaw]
```

### Shard 与输出结构

默认每个 parquet/video part 包含 20 个 episode，每个 chunk 包含 50 个 part。可通过 `--episodes-per-file` 和 `--files-per-chunk` 调整。

```text
<output_root>/<dataset_name>/
├── data/chunk-000/part-000.parquet
├── videos/
│   ├── front_image/chunk-000/part-000.mp4
│   ├── left_image/chunk-000/part-000.mp4
│   ├── right_image/chunk-000/part-000.mp4
│   └── rear_image/chunk-000/part-000.mp4
├── meta/
│   ├── info.json
│   ├── modality.json
│   ├── tasks.parquet
│   ├── episodes/
│   ├── checkpoints/
│   ├── navvla_tasks.jsonl
│   ├── navvla_cameras.json
│   ├── navvla_frame_metadata.jsonl
│   ├── navvla_video_index.parquet
│   ├── navvla_schema_ext.json
│   ├── navvla_context_index_manifest.json
│   └── context_index/
├── cache/context_index_debug/
├── dataset_statistics.json
└── conversion_report.json
```

Data parquet 每帧一行，包含：

```text
episode_index
frame_index
timestamp
task_index
observation.state
action
action.padding_mask
next.done
sample.action_available
context.index_key
source_frame_index
index
```

`index` 是数据集内连续的全局帧索引；episode ID 为 `<scene_id>_<source_episode_id>`。`conversion_report.json` 记录 episode、frame、shard、视频数量以及所有被跳过 episode 的原因。

### Context index

转换器默认生成 token budget 为 1024、仅引用 front 相机的 sliding-recent context index；四路视频本身始终完整写出。可选择多个 budget 或相机：

```bash
--context-token-budgets 512 1024 2048 \
--context-cameras front left right rear
```

可选策略：

- `--use-bats`：启用确定性的 BATS 历史选择；
- `--use-hash-dedup`：用 front dHash 去除近重复历史帧；
- `--dhash-threshold`：设置 dHash Hamming 距离阈值，默认 10；
- `--no-context-index`：不生成 context 文件，但 data parquet 仍保留稳定的 `context.index_key`。

每个 budget 写入 `meta/context_index/budget_<budget>/`，其中包含 `context_meta.parquet`、`refs.parquet` 和扁平化的 NumPy context arrays。默认 budget 和可用 budget 列表记录在 `meta/navvla_context_index_manifest.json`。

### 断点续转与无效数据

首次转换中断后，使用完全相同的参数并增加 `--resume`：

```bash
python convert_rollout_to_lerobot.py \
  --source-root /path/to/seed_101 \
  --output-root data/lerobot \
  --dataset-name track-stt-seed101 \
  --skip-invalid \
  --resume
```

Resume 会校验转换配置、parquet 行数、四路视频帧数和 shard checkpoint；完整 shard 会跳过，不完整 shard 会清理后重做。统计信息、合并元数据和 context index 会重新生成。

若输出已存在且需要全部重做，使用 `--overwrite`。`--overwrite` 会删除 `<output_root>/<dataset_name>`，不能与 `--resume` 同时使用。

源目录存在残缺 episode 时使用 `--skip-invalid`。未使用该选项时，发现第一个不完整或媒体不一致的 episode 就会停止转换。

完整参数可通过以下命令查看：

```bash
python convert_rollout_to_lerobot.py --help
```

## 常见问题

### `ModuleNotFoundError: hydra`

安装仓库依赖，并确认运行脚本的 Python 与安装依赖的 Python 一致：

```bash
python -m pip install -r habitat-lab/requirements.txt
```

### `DYN_OBSTACLE_AVOID=1 but pyrvo2 is not installed`

安装提供 `rvo2` Python 模块的 RVO2 binding，或使用以下命令暂时关闭动态避障：

```bash
TASK=stt DYN_OBSTACLE_AVOID=0 bash train_data.sh
```

### LeRobot 转换依赖缺失

若转换器提示缺少 NumPy、Pandas、PyArrow 或 OpenCV，安装独立转换依赖：

```bash
python -m pip install -r requirements-lerobot.txt
```

### EGL 与 CUDA 设备不匹配

确认 `CUDA_VISIBLE_DEVICES`、`EGL_DEVICE_ID` 和 `EGL_VISIBLE_DEVICES` 指向同一张物理卡。多 GPU 脚本默认使用从 0 开始的连续 GPU 编号。

### Episode 没有落盘

检查：

```text
<save_path>/runtime_logs/split_<id>.log
<save_path>/runtime_logs/split_<id>_episode_failures.jsonl
```

非 debug 模式只保存满足完整成功条件的 episode。调试时可使用 `DATA_DEBUG=1` 保存所有产生了 step 的 episode。

### 重跑时没有跳过已有 episode

默认断点检查要求结果 JSON、逐步信息 JSON 和四个视角视频全部存在。任何一路视频缺失都会重新执行该 episode。

### Teacher 与执行器速度不一致

确保同一 YAML 中以下两个 action 的纵向、横向和 yaw 速度上限完全一致：

```text
agent_1_oracle_follow_action
agent_1_base_velocity
```
