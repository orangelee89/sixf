#!/usr/bin/env bash
# ========= train_supervisor.sh (TMUX-friendly 版本) =========
set -euo pipefail

GPU_UTIL_TH=40
GPU_IDLE_SEC_THRESHOLD=10
CHECK_INTERVAL=5
WARMUP_AFTER_LAUNCH=30

WS="/home/lee/EE_ws/src/sixfeet_src/ckpt"
TRAIN_SH="$WS/train_all.sh"
PY_SCRIPT_NAME="train.py"
PY_SCRIPT_PATTERN="$PY_SCRIPT_NAME"

FRAMEWORK_ARG=""
USER_DECLARED_TOTAL_TRAINING_UNITS="0"
LOG_TAG="[Supervisor]"

########## 选择 RL 框架 ##########
select_framework() {
  local choice
  while true; do
    echo "------------------------------------------"
    echo "请选择要使用的 RL 框架："
    echo "1: RL-Games"
    echo "2: RSL-RL"
    echo "------------------------------------------"
    read -r -p "请输入选项 (1 或 2): " choice
    case "$choice" in
      1) FRAMEWORK_ARG="rl_games"; echo "$LOG_TAG 选择 RL-Games"; break ;;
      2) FRAMEWORK_ARG="rsl_rl";   echo "$LOG_TAG 选择 RSL-RL";   break ;;
      *) echo "$LOG_TAG 无效输入 '$choice'" ;;
    esac
  done
}

prompt_for_training_units() {
  echo "------------------------------------------"
  read -r -p "请输入目标总训练单元数（如 10000，留空或输入 0 则使用默认）: " units
  if [[ "$units" =~ ^[1-9][0-9]*$ ]]; then
    USER_DECLARED_TOTAL_TRAINING_UNITS="$units"
  else
    USER_DECLARED_TOTAL_TRAINING_UNITS="0"
  fi
  echo "$LOG_TAG 设置训练单元数: $USER_DECLARED_TOTAL_TRAINING_UNITS"
}

cleanup_and_exit() {
  echo "$LOG_TAG 清理并退出"
  pkill -9 -f "$PY_SCRIPT_PATTERN" 2>/dev/null || true
  pkill -9 -f "$TRAIN_SH" 2>/dev/null || true
  rm -f "$WS/train_finished.flag"
  exit 0
}
trap cleanup_and_exit SIGINT SIGTERM

select_framework
prompt_for_training_units

iterations_launched=0
idle_seconds_counter=0

while true; do
  if [[ -f "$WS/train_finished.flag" ]]; then
    echo "$LOG_TAG 训练完成，退出"
    cleanup_and_exit
  fi

  if ! pgrep -f "$PY_SCRIPT_PATTERN" > /dev/null; then
    echo "$LOG_TAG 未检测到 Python 训练进程，启动新训练"
    bash "$TRAIN_SH" "$FRAMEWORK_ARG" "$USER_DECLARED_TOTAL_TRAINING_UNITS" &
    LAUNCH_TIME=$(date +%s)
    sleep 5
    continue
  fi

  now=$(date +%s)
  time_since_launch=$((now - LAUNCH_TIME))

  if (( time_since_launch > WARMUP_AFTER_LAUNCH )); then
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0)
    if (( gpu_util <= GPU_UTIL_TH )); then
      idle_seconds_counter=$((idle_seconds_counter + CHECK_INTERVAL))
      echo "$LOG_TAG GPU util ${gpu_util}% 过低，空闲时间 ${idle_seconds_counter}/${GPU_IDLE_SEC_THRESHOLD}s"
    else
      idle_seconds_counter=0
      echo "$LOG_TAG GPU util ${gpu_util}% 正常"
    fi
  else
    echo "$LOG_TAG 预热期 (${time_since_launch}s)..."
  fi

  if (( idle_seconds_counter >= GPU_IDLE_SEC_THRESHOLD )); then
    echo "$LOG_TAG GPU 持续空闲，重启训练进程"
    pkill -9 -f "$PY_SCRIPT_PATTERN" 2>/dev/null || true
    pkill -9 -f "$TRAIN_SH" 2>/dev/null || true
    sleep 2
    bash "$TRAIN_SH" "$FRAMEWORK_ARG" "$USER_DECLARED_TOTAL_TRAINING_UNITS" &
    LAUNCH_TIME=$(date +%s)
    idle_seconds_counter=0
  fi

  sleep "$CHECK_INTERVAL"
done
