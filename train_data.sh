#!/usr/bin/env bash
#
# 统一数据采集脚本：通过 TASK 切换 at / dt / stt。
# 若 ${SAVE_PATH}/failed_splits.txt 中已有 failed list，则只重跑失败 split；
# 否则跑全量 0..CHUNKS-1。
#
# 用法：
#   TASK=at bash train_data.sh
#   TASK=dt bash train_data.sh
#   TASK=stt bash train_data.sh
#   TASK=at FAILED_SPLIT_IDS="10 20 30" bash train_data.sh   # 手动指定重跑
#   FORCE_FULL=1 TASK=dt bash train_data.sh                  # 忽略 failed list，强制全量
#
# Background jobs ignore Ctrl+C SIGINT unless we forward cleanup explicitly.
# 并行：机器有 8 卡时 NUM_PARALLEL=8 可接近 2× 吞吐（每卡一个 Habitat 进程）。
# CHUNKS：7257 train episodes；越大单 job 越短、并行 wave 越多，但进程启动开销也略增。
#   推荐 64–120（约 60–110 ep/job）；全量跑完约 7257/8 ≈ 900 ep/GPU。
# HUMAN_STUCK_MIN_STEPS=20 TASK=dt bash train_data.sh
# AVOID_OTHER_PEDESTRIANS=0 TASK=dt bash train_data.sh  # 仅对主行人避障
# DYN_OBSTACLE_AVOID=0 TASK=dt bash train_data.sh      # 关闭 RVO 动态避障
# PYTHON_BIN=/data1/yizhang/RLinf/.venv-habitat/bin/python bash train_data.sh
set -uo pipefail

TASK="${TASK:-dt}"
case "${TASK}" in
  at|dt|stt) ;;
  *)
    echo "Unknown TASK='${TASK}'. Use: at | dt | stt" >&2
    exit 1
    ;;
esac

CHUNKS="${CHUNKS:-900}"
NUM_PARALLEL="${NUM_PARALLEL:-8}"
SEED="${SEED:-101}"
SAVE_PATH="${SAVE_PATH:-sim_data/v4-new/${TASK}/train/seed_${SEED}}"
FAIL_LOG="${SAVE_PATH}/failed_splits.txt"
RUNTIME_LOG_DIR="${SAVE_PATH}/runtime_logs"
EXP_CONFIG="habitat-lab/habitat/config/benchmark/nav/track/track_train_${TASK}.yaml"
FORCE_FULL="${FORCE_FULL:-0}"

# 关闭逐步 print，可显著减少 I/O 等待（见 baseline_agent.py DATA_QUIET）
DATA_QUIET="${DATA_QUIET:-0}"
DATA_DEBUG="${DATA_DEBUG:-0}"
ROLLOUT_FAILURE_TAIL_STEPS="${ROLLOUT_FAILURE_TAIL_STEPS:-20}"
# RVO 动态避障：1=对所有行人避障（默认）；0=仅对主行人 agent_0 避障
AVOID_OTHER_PEDESTRIANS="${AVOID_OTHER_PEDESTRIANS:-1}"
# 动态避障总开关：1=开启 RVO（默认）；0=关闭
DYN_OBSTACLE_AVOID="${DYN_OBSTACLE_AVOID:-1}"
# RVO 细调：减弱对其他行人避障，减少绕远跟丢主行人（见 baseline_agent.py）
OTHER_RVO_RADIUS_SCALE="${OTHER_RVO_RADIUS_SCALE:-0.8}"
RVO_NEIGHBOR_DIST="${RVO_NEIGHBOR_DIST:-2.2}"
RVO_TIME_HORIZON="${RVO_TIME_HORIZON:-1.5}"
RVO_AGENT_RADIUS="${RVO_AGENT_RADIUS:-0.35}"
LEADER_RVO_RADIUS_SCALE="${LEADER_RVO_RADIUS_SCALE:-1.0}"
RVO_MAX_SPEED="${RVO_MAX_SPEED:-2.0}"
RVO_VELOCITY_EMA_ALPHA="${RVO_VELOCITY_EMA_ALPHA:-0.35}"
# 主行人最终动作硬护栏（在 Teacher 平滑和 RVO 之后执行）：
# 使用主行人相对速度预测近距闭合；紧急区内主动径向后退，并压低横移/偏航。
MAIN_HUMAN_PROXIMITY_GUARD="${MAIN_HUMAN_PROXIMITY_GUARD:-1}"
MAIN_HUMAN_GUARD_START_DISTANCE="${MAIN_HUMAN_GUARD_START_DISTANCE:-1.8}"
MAIN_HUMAN_GUARD_STOP_DISTANCE="${MAIN_HUMAN_GUARD_STOP_DISTANCE:-1.0}"
MAIN_HUMAN_GUARD_APPROACH_GAIN="${MAIN_HUMAN_GUARD_APPROACH_GAIN:-1.5}"
MAIN_HUMAN_GUARD_CLOSING_MARGIN="${MAIN_HUMAN_GUARD_CLOSING_MARGIN:-0.25}"
MAIN_HUMAN_GUARD_PREDICTION_HORIZON="${MAIN_HUMAN_GUARD_PREDICTION_HORIZON:-0.25}"
MAIN_HUMAN_GUARD_EMERGENCY_DISTANCE="${MAIN_HUMAN_GUARD_EMERGENCY_DISTANCE:-1.15}"
MAIN_HUMAN_GUARD_MIN_RETREAT_SPEED="${MAIN_HUMAN_GUARD_MIN_RETREAT_SPEED:-0.8}"
MAIN_HUMAN_GUARD_MAX_TANGENT_SPEED="${MAIN_HUMAN_GUARD_MAX_TANGENT_SPEED:-0.15}"
MAIN_HUMAN_GUARD_MAX_YAW_SPEED="${MAIN_HUMAN_GUARD_MAX_YAW_SPEED:-0.5}"
# 默认只使用 RVO 动态避障；可手动开启 teacher 势场用于消融实验。
TEACHER_PED_REPULSE="${TEACHER_PED_REPULSE:-0}"
TEACHER_PED_REPULSE_RADIUS="${TEACHER_PED_REPULSE_RADIUS:-2.5}"
TEACHER_PED_REPULSE_GAIN="${TEACHER_PED_REPULSE_GAIN:-0.8}"
# 可选：略降单局上限、略提仿真速度（需与训练侧 ctrl_freq/速度一致时再改）
# EXTRA_OPTS='habitat.environment.max_episode_steps=400'
# teacher 与 agent_1_base_velocity 在 YAML 中统一为 2.0/1.0/2.5；不要单边覆盖。
EXTRA_OPTS="${EXTRA_OPTS:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# 可显式指定物理 GPU 或 Slurm 分配的 GPU token，例如 GPU_IDS="2 5"。
# 未指定时优先使用外部 CUDA_VISIBLE_DEVICES；否则从 nvidia-smi 自动发现。
GPU_IDS="${GPU_IDS:-}"
GPU_PRECHECK="${GPU_PRECHECK:-1}"
# masked：每个进程只暴露一张卡；full：保留全部 GPU 可见性，并通过
# habitat.simulator.habitat_sim_v0.gpu_device_id 选择物理卡（用于 EGL 映射异常节点）。
GPU_BIND_MODE="${GPU_BIND_MODE:-masked}"
# 可复用无 sudo 的 NVIDIA 用户态库目录。优先使用系统库；若系统缺失，
# 自动使用 /data1/wxwu/github 中已准备好的、与 550.54.15 驱动匹配的归档。
NVIDIA_LIB_DIR="${NVIDIA_LIB_DIR:-}"
if [[ -z "${NVIDIA_LIB_DIR}" && -d /data1/wxwu/github/nvidia-550.54.15/nvidia_driver-linux-x86_64-550.54.15-archive/lib ]]; then
  NVIDIA_LIB_DIR="/data1/wxwu/github/nvidia-550.54.15/nvidia_driver-linux-x86_64-550.54.15-archive/lib"
fi
if [[ -z "${NVIDIA_LIB_DIR}" && -d /usr/lib/x86_64-linux-gnu ]]; then
  NVIDIA_LIB_DIR="/usr/lib/x86_64-linux-gnu"
fi
EGL_VENDOR_FILE="${EGL_VENDOR_FILE:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
if [[ ! -f "${EGL_VENDOR_FILE}" && -n "${NVIDIA_LIB_DIR}" && -f "$(dirname "${NVIDIA_LIB_DIR}")/etc/10_nvidia.json" ]]; then
  EGL_VENDOR_FILE="$(dirname "${NVIDIA_LIB_DIR}")/etc/10_nvidia.json"
fi
# 无 sudo 时使用项目外部已准备好的 NVIDIA 用户态库。系统 GLVND 通用库
# 必须放在归档目录之前，否则 OpenCV/Mesa 可能与 NVIDIA libGLdispatch 混用。
if [[ -n "${NVIDIA_LIB_DIR}" ]]; then
  export NVIDIA_LIB_DIR
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${NVIDIA_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export __EGL_VENDOR_LIBRARY_FILENAMES="${EGL_VENDOR_FILE}"
export EGL_VENDOR_LIBRARY_DIRS="$(dirname "${EGL_VENDOR_FILE}")"
# 多卡并行时 Habitat 需把 EGL 绑到与 CUDA_VISIBLE_DEVICES 同一张物理卡；
# GPU_IDS 使用物理编号时，EGL_DEVICE_ID 也应使用该物理编号。
# 若仍失败，在 shell 里先 export（无 root 时常用）：
#   export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
#   export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${NVIDIA_LIB_DIR}
# 不要把 NVIDIA 归档目录放在系统目录之前，否则会覆盖系统 libGLdispatch。
EGL_DEVICE_ID="${EGL_DEVICE_ID:-}"

discover_gpu_ids() {
  local visible count index normalized
  if [[ -n "${GPU_IDS}" ]]; then
    normalized="${GPU_IDS//,/ }"
    read -r -a DISCOVERED_GPU_IDS <<< "${normalized}"
    return 0
  fi
  visible="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -n "${visible}" ]]; then
    normalized="${visible//,/ }"
    read -r -a DISCOVERED_GPU_IDS <<< "${normalized}"
    return 0
  fi
  if [[ "${GPU_PRECHECK}" == "0" ]]; then
    echo "GPU_PRECHECK=0 requires GPU_IDS or CUDA_VISIBLE_DEVICES to be set." >&2
    return 1
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "No nvidia-smi found. Run train_data.sh on a GPU node with NVIDIA drivers." >&2
    return 1
  fi
  count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)"
  if [[ "${count}" -le 0 ]]; then
    echo "nvidia-smi cannot see any GPU. The node/container has no NVIDIA device or driver." >&2
    echo "Check /dev/nvidia*, request a GPU allocation, or start the container with NVIDIA GPU passthrough." >&2
    return 1
  fi
  DISCOVERED_GPU_IDS=()
  for ((index = 0; index < count; index++)); do
    DISCOVERED_GPU_IDS+=("${index}")
  done
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if ! discover_gpu_ids || [[ ${#DISCOVERED_GPU_IDS[@]} -eq 0 ]]; then
  exit 1
fi
if [[ "${GPU_PRECHECK}" == "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "NVIDIA GPU precheck failed: nvidia-smi cannot communicate with the driver." >&2
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} GPU_IDS=${GPU_IDS:-<unset>}" >&2
    echo "Visible NVIDIA device nodes:" >&2
    ls -l /dev/nvidia* 2>&1 | head -30 >&2 || true
    echo "This node/container has no usable NVIDIA device. Request a GPU node or enable GPU passthrough before running train_data.sh." >&2
    exit 1
  fi
  if [[ ! -f "${EGL_VENDOR_FILE}" ]] && [[ ! -e "${NVIDIA_LIB_DIR}/libEGL_nvidia.so.0" ]] && [[ ! -e /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 ]] && [[ ! -e /lib/x86_64-linux-gnu/libEGL_nvidia.so.0 ]]; then
    echo "NVIDIA CUDA is available, but NVIDIA EGL userspace libraries are missing." >&2
    echo "Expected ${EGL_VENDOR_FILE} and libEGL_nvidia.so.0; Habitat-Sim windowless rendering will fail." >&2
    echo "Install the matching libnvidia-gl package for the host driver, or run the container with NVIDIA graphics libraries mounted." >&2
    exit 1
  fi
fi
GPU_COUNT=${#DISCOVERED_GPU_IDS[@]}
if [[ "${NUM_PARALLEL}" -gt "${GPU_COUNT}" ]]; then
  echo "NUM_PARALLEL=${NUM_PARALLEL} exceeds visible GPU count=${GPU_COUNT}; using ${GPU_COUNT}." >&2
  NUM_PARALLEL="${GPU_COUNT}"
fi
echo "Using GPU tokens: ${DISCOVERED_GPU_IDS[*]}"
echo "Using Python: ${PYTHON_BIN}"
echo "Using NVIDIA library dir: ${NVIDIA_LIB_DIR:-<system default>}"
echo "Using EGL vendor file: ${EGL_VENDOR_FILE}"
if [[ "${GPU_BIND_MODE}" == "full" ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES_ALL:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES_ALL="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/[[:space:]]/, ""); ids = ids (ids ? "," : "") $0} END {print ids}')"
  fi
  CUDA_VISIBLE_DEVICES_ALL="${CUDA_VISIBLE_DEVICES_ALL:-0,1,2,3,4,5,6,7}"
  echo "GPU bind mode: full (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_ALL})"
fi

# ---------------------------------------------------------------------------
# 决定跑全量还是重跑 failed list
# ---------------------------------------------------------------------------
SPLIT_QUEUE=()
MODE="full"

load_failed_splits() {
  local line
  # 允许外部手动传入：FAILED_SPLIT_IDS="1 2 3"
  if [[ -n "${FAILED_SPLIT_IDS:-}" ]]; then
    # shellcheck disable=SC2206
    SPLIT_QUEUE=(${FAILED_SPLIT_IDS})
    return 0
  fi
  if [[ ! -f "${FAIL_LOG}" ]]; then
    return 1
  fi
  line="$(grep '^# failed split-id list:' "${FAIL_LOG}" | tail -1 || true)"
  if [[ -z "${line}" ]]; then
    return 1
  fi
  # shellcheck disable=SC2206
  SPLIT_QUEUE=(${line#*: })
  [[ ${#SPLIT_QUEUE[@]} -gt 0 ]]
}

if [[ "${FORCE_FULL}" != "1" ]] && load_failed_splits; then
  MODE="retry"
  echo "Detected failed list (${#SPLIT_QUEUE[@]}): ${SPLIT_QUEUE[*]}"
  echo "Mode=retry TASK=${TASK} SAVE_PATH=${SAVE_PATH}"
else
  MODE="full"
  SPLIT_QUEUE=()
  local_i=0
  for ((local_i = 0; local_i < CHUNKS; local_i++)); do
    SPLIT_QUEUE+=("${local_i}")
  done
  echo "Mode=full TASK=${TASK} chunks=${CHUNKS} SAVE_PATH=${SAVE_PATH}"
fi

TOTAL=${#SPLIT_QUEUE[@]}
IDX=0

CHILD_PIDS=()
cleanup_jobs() {
  local pid
  for pid in "${CHILD_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${CHILD_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}

shutdown() {
  echo "Interrupted — stopping child python processes..."
  cleanup_jobs
  exit 130
}

trap shutdown INT TERM

mkdir -p "${SAVE_PATH}" "${RUNTIME_LOG_DIR}"
echo "# train_data.sh started $(date '+%Y-%m-%dT%H:%M:%S%z') task=${TASK} mode=${MODE} seed=${SEED} chunks=${CHUNKS} total=${TOTAL}" >> "${FAIL_LOG}"

# GPU 槽位池：某 split 结束后立刻在同卡启动下一个，避免「等本批最慢 job」导致空转。
# 需要 bash >= 4.3（支持 wait -n）。
declare -a SLOT_PID=()
declare -a SLOT_SPLIT=()
FAILED_SPLITS=()

launch_split_on_gpu() {
  local gpu="$1"
  local split_id="$2"
  local gpu_token="${DISCOVERED_GPU_IDS[$gpu]}"
  local egl_device_id="${EGL_DEVICE_ID:-${gpu_token}}"
  local cuda_visible_devices="${gpu_token}"
  local egl_visible_devices="${gpu_token}"
  local sim_gpu_opt=""
  if [[ "${GPU_BIND_MODE}" == "full" ]]; then
    cuda_visible_devices="${CUDA_VISIBLE_DEVICES_ALL}"
    egl_visible_devices="${CUDA_VISIBLE_DEVICES_ALL}"
    sim_gpu_opt="habitat.simulator.habitat_sim_v0.gpu_device_id=${gpu_token}"
  fi
  local runtime_log="${RUNTIME_LOG_DIR}/split_${split_id}.log"
  if [[ "${MODE}" == "retry" ]]; then
    echo "Retry split-id=${split_id} on GPU=${gpu} (queue $((IDX + 1))/${TOTAL})"
  else
    echo "Launching split-id=${split_id} on GPU=${gpu} ($((split_id + 1))/${CHUNKS})"
  fi
  echo "Runtime log: ${runtime_log}"
  printf '\n# launch %s task=%s mode=%s split=%s gpu=%s\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${TASK}" "${MODE}" "${split_id}" "${gpu}" \
    >> "${runtime_log}"
  # gpu_token 可以是物理编号、UUID 或 Slurm 注入的 CUDA_VISIBLE_DEVICES token。
  # 物理编号场景下 EGL_DEVICE_ID 必须与 gpu_token 一致；否则会出现
  # "unable to find CUDA device 0 among N EGL devices"。
  CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" \
  EGL_DEVICE_ID="${egl_device_id}" \
  EGL_VISIBLE_DEVICES="${egl_visible_devices}" \
  __EGL_VENDOR_LIBRARY_FILENAMES="${EGL_VENDOR_FILE}" \
  EGL_VENDOR_LIBRARY_DIRS="$(dirname "${EGL_VENDOR_FILE}")" \
  # 系统 GLVND 通用库必须优先；归档目录仅提供 NVIDIA vendor 库，避免
  # archive/lib/libGLdispatch.so 覆盖 Mesa/系统 libGLdispatch，触发
  # "_glapi_tls_Current" undefined symbol。
  LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${NVIDIA_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  PYTHONUNBUFFERED=1 \
  SAVE_VIDEO=1 DATA_QUIET=${DATA_QUIET} \
  ROLLOUT_FAILURE_TAIL_STEPS=${ROLLOUT_FAILURE_TAIL_STEPS} \
  DYN_OBSTACLE_AVOID=${DYN_OBSTACLE_AVOID} \
  AVOID_OTHER_PEDESTRIANS=${AVOID_OTHER_PEDESTRIANS} OTHER_RVO_RADIUS_SCALE=${OTHER_RVO_RADIUS_SCALE} \
  RVO_NEIGHBOR_DIST=${RVO_NEIGHBOR_DIST} RVO_TIME_HORIZON=${RVO_TIME_HORIZON} \
  RVO_AGENT_RADIUS=${RVO_AGENT_RADIUS} RVO_MAX_SPEED=${RVO_MAX_SPEED} \
  LEADER_RVO_RADIUS_SCALE=${LEADER_RVO_RADIUS_SCALE} \
  RVO_VELOCITY_EMA_ALPHA=${RVO_VELOCITY_EMA_ALPHA} \
  MAIN_HUMAN_PROXIMITY_GUARD=${MAIN_HUMAN_PROXIMITY_GUARD} \
  MAIN_HUMAN_GUARD_START_DISTANCE=${MAIN_HUMAN_GUARD_START_DISTANCE} \
  MAIN_HUMAN_GUARD_STOP_DISTANCE=${MAIN_HUMAN_GUARD_STOP_DISTANCE} \
  MAIN_HUMAN_GUARD_APPROACH_GAIN=${MAIN_HUMAN_GUARD_APPROACH_GAIN} \
  MAIN_HUMAN_GUARD_CLOSING_MARGIN=${MAIN_HUMAN_GUARD_CLOSING_MARGIN} \
  MAIN_HUMAN_GUARD_PREDICTION_HORIZON=${MAIN_HUMAN_GUARD_PREDICTION_HORIZON} \
  MAIN_HUMAN_GUARD_EMERGENCY_DISTANCE=${MAIN_HUMAN_GUARD_EMERGENCY_DISTANCE} \
  MAIN_HUMAN_GUARD_MIN_RETREAT_SPEED=${MAIN_HUMAN_GUARD_MIN_RETREAT_SPEED} \
  MAIN_HUMAN_GUARD_MAX_TANGENT_SPEED=${MAIN_HUMAN_GUARD_MAX_TANGENT_SPEED} \
  MAIN_HUMAN_GUARD_MAX_YAW_SPEED=${MAIN_HUMAN_GUARD_MAX_YAW_SPEED} \
  TEACHER_PED_REPULSE=${TEACHER_PED_REPULSE} \
  TEACHER_PED_REPULSE_RADIUS=${TEACHER_PED_REPULSE_RADIUS} \
  TEACHER_PED_REPULSE_GAIN=${TEACHER_PED_REPULSE_GAIN} \
  PYTHONPATH="habitat-lab${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" run.py \
    --split-num "$CHUNKS" \
    --split-id "$split_id" \
    --exp-config "${EXP_CONFIG}" \
    --run-type 'eval' \
    --save-path "$SAVE_PATH" \
    habitat.simulator.seed="${SEED}" \
    ${sim_gpu_opt} \
    ${EXTRA_OPTS} >> "${runtime_log}" 2>&1 &
  SLOT_SPLIT[$gpu]=$split_id
  SLOT_PID[$gpu]=$!
  CHILD_PIDS+=($!)
}

reap_split_on_gpu() {
  local gpu="$1"
  local pid="${SLOT_PID[$gpu]}"
  local split_id="${SLOT_SPLIT[$gpu]}"
  local rc=0
  wait "${pid}" || rc=$?
  if [[ ${rc} -eq 0 ]]; then
    echo "[done] split-id=${split_id} GPU=${gpu}"
  else
    echo "[FAIL] split-id=${split_id} GPU=${gpu} exit=${rc}" >&2
    printf '%s\n' "[FAIL] split-id=${split_id} GPU=${gpu} exit=${rc} $(date '+%Y-%m-%dT%H:%M:%S%z')" >> "${FAIL_LOG}"
    FAILED_SPLITS+=("${split_id}")
  fi
  unset "SLOT_PID[$gpu]" "SLOT_SPLIT[$gpu]"
}

count_running_slots() {
  local gpu n=0
  for ((gpu = 0; gpu < NUM_PARALLEL; gpu++)); do
    if [[ -n "${SLOT_PID[$gpu]:-}" ]] && kill -0 "${SLOT_PID[$gpu]}" 2>/dev/null; then
      ((n++)) || true
    fi
  done
  echo "$n"
}

count_orphaned_slots() {
  local gpu n=0
  for ((gpu = 0; gpu < NUM_PARALLEL; gpu++)); do
    if [[ -n "${SLOT_PID[$gpu]:-}" ]] && ! kill -0 "${SLOT_PID[$gpu]}" 2>/dev/null; then
      ((n++)) || true
    fi
  done
  echo "$n"
}

reap_and_maybe_launch() {
  local gpu="$1"
  reap_split_on_gpu "${gpu}"
  if [[ ${IDX} -lt ${TOTAL} ]]; then
    launch_split_on_gpu "${gpu}" "${SPLIT_QUEUE[IDX]}"
    ((IDX++)) || true
  fi
}

maintain_slots() {
  local gpu
  for ((gpu = 0; gpu < NUM_PARALLEL; gpu++)); do
    if [[ -n "${SLOT_PID[$gpu]:-}" ]]; then
      if ! kill -0 "${SLOT_PID[$gpu]}" 2>/dev/null; then
        reap_and_maybe_launch "${gpu}"
      fi
    elif [[ ${IDX} -lt ${TOTAL} ]]; then
      launch_split_on_gpu "${gpu}" "${SPLIT_QUEUE[IDX]}"
      ((IDX++)) || true
    fi
  done
}

has_pending_work() {
  [[ ${IDX} -lt ${TOTAL} ]] && return 0
  [[ $(count_running_slots) -gt 0 ]] && return 0
  [[ $(count_orphaned_slots) -gt 0 ]] && return 0
  return 1
}

wait_for_any_child() {
  # wait -n -p 在子进程非零退出时也会返回非零，不能用 if ! wait 否则会二次 wait 丢事件
  local finished_pid=""
  wait -n -p finished_pid 2>/dev/null || true
  if [[ -z "${finished_pid}" ]]; then
    wait -n || true
  fi
}

for ((gpu = 0; gpu < NUM_PARALLEL && IDX < TOTAL; gpu++)); do
  launch_split_on_gpu "${gpu}" "${SPLIT_QUEUE[IDX]}"
  ((IDX++)) || true
done

while has_pending_work; do
  maintain_slots
  [[ $(count_running_slots) -gt 0 ]] || continue
  wait_for_any_child
done
maintain_slots

CHILD_PIDS=()
if [[ ${#FAILED_SPLITS[@]} -gt 0 ]]; then
  {
    echo "# ${MODE} finished $(date '+%Y-%m-%dT%H:%M:%S%z') count=${#FAILED_SPLITS[@]}"
    echo "# failed split-id list: ${FAILED_SPLITS[*]}"
  } >> "${FAIL_LOG}"
  echo "Failed split-id(s): ${FAILED_SPLITS[*]}" >&2
  echo "Logged to ${FAIL_LOG}" >&2
  echo "Re-run: TASK=${TASK} bash train_data.sh  (auto-retry from failed list)" >&2
  exit 1
fi

if [[ "${MODE}" == "retry" ]]; then
  echo "All ${TOTAL} retry split(s) completed successfully."
else
  echo "All ${TOTAL} split(s) completed successfully."
fi
