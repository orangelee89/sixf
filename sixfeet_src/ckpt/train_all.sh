#!/usr/bin/env bash
# ========= train_all.sh (V4 – 加入 CPU 亲和性与 nice) =========
# 用法示例：
#   ./train_all_with_affinity.sh rsl_rl 500000
#   CPU_CORE_RANGE="8-31" CPU_NICE_VALUE=5 ./train_all_with_affinity.sh rl_games

set -euo pipefail

########################################
# 0) 基本环境准备
########################################
if [[ $# -lt 1 ]]; then
  echo "用法: $0 <rsl_rl | rl_games> [total_training_units]"
  exit 1
fi
SELECTED_FRAMEWORK="$1"
USER_PROVIDED_TRAINING_UNITS_ARG="${2:-0}"

# ==== 新增：CPU 亲和性 / nice 配置 ====
CPU_CORE_RANGE="${CPU_CORE_RANGE:-8-31}"      # 默认给训练进程用的核心范围
CPU_NICE_VALUE="${CPU_NICE_VALUE:-10}"        # 默认 nice 优先级（越大越低）
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

WS1="/home/lee/EE_ws/src/sixfeet_src/ckpt"
WS="/home/lee/EE_ws/src/sixfeet_src/sixfeet"
cd "$WS1"

DEFAULT_INTERNAL_TOTAL_TRAINING_UNITS=10000
TOTAL_TRAINING_UNITS=$DEFAULT_INTERNAL_TOTAL_TRAINING_UNITS

if [[ "$USER_PROVIDED_TRAINING_UNITS_ARG" =~ ^[1-9][0-9]*$ ]]; then
  TOTAL_TRAINING_UNITS="$USER_PROVIDED_TRAINING_UNITS_ARG"
  echo "[$(date)] INFO: 使用目标总训练单元数: $TOTAL_TRAINING_UNITS"
elif [[ "$USER_PROVIDED_TRAINING_UNITS_ARG" == "0" ]]; then
  echo "[$(date)] INFO: 使用脚本默认目标训练单元数: $TOTAL_TRAINING_UNITS"
else
  echo "[$(date)] WARNING: 参数 '$USER_PROVIDED_TRAINING_UNITS_ARG' 无效，使用默认值: $TOTAL_TRAINING_UNITS"
fi

TASK_NAME="Template-Sixfeet-Direct-v0"
PYTHON_CMD_EXEC=()
iterations_for_this_run=$TOTAL_TRAINING_UNITS

########################################
# 1) 选择框架并构造 Python 命令
########################################

if [[ "$SELECTED_FRAMEWORK" == "rsl_rl" ]]; then
  LOG_DIR="$WS1/logs/rsl_rl/sixfeet_ppo"
  PY_SCRIPT="$WS/scripts/rsl_rl/train.py"
  EXP_NAME="sixfeet_ppo"

  mkdir -p "$LOG_DIR"
  ckpt_path=$(ls -1t "$LOG_DIR"/*/model_*.pt 2>/dev/null | head -n1 || true)

  if [[ -n "$ckpt_path" ]]; then
    run_name=$(basename "$(dirname "$ckpt_path")")
    ckpt_file=$(basename "$ckpt_path")
    echo "[$(date)] 检测到 checkpoint: $ckpt_path"

    completed_iterations=0
    if [[ "$ckpt_file" =~ model_([0-9]+)\.pt ]]; then
      completed_iterations="${BASH_REMATCH[1]}"
    fi

    if [[ "$TOTAL_TRAINING_UNITS" -gt "$completed_iterations" ]]; then
      iterations_for_this_run=$((TOTAL_TRAINING_UNITS - completed_iterations))
    else
      iterations_for_this_run=0
    fi

    PYTHON_CMD_EXEC=(python -u "$PY_SCRIPT" \
        --task "$TASK_NAME" \
        --resume \
        --experiment_name "$EXP_NAME" \
        --load_run "$run_name" \
        --checkpoint "$ckpt_file" \
        --max_iterations "$iterations_for_this_run" \
        --headless)
  else
    echo "[$(date)] 未找到 checkpoint，开始新训练"
    PYTHON_CMD_EXEC=(python -u "$PY_SCRIPT" \
        --task "$TASK_NAME" \
        --experiment_name "$EXP_NAME" \
        --max_iterations "$iterations_for_this_run" \
        --headless)
  fi

elif [[ "$SELECTED_FRAMEWORK" == "rl_games" ]]; then
  LOG_ROOT="$WS1/logs/rl_games"
  EXP_NAME="sixfeet_ppo"
  PY_SCRIPT="$WS/scripts/rl_games/train.py"

  best_ckpt=$(ls -1t "$LOG_ROOT/$EXP_NAME"/*/nn/sixfeet_ppo.pth 2>/dev/null | head -n1 || true)
  if [[ -z "$best_ckpt" ]]; then
    best_ckpt=$(ls -1t "$LOG_ROOT/$EXP_NAME"/*/nn/last_*.pth 2>/dev/null | head -n1 || true)
  fi

  if [[ -n "$best_ckpt" ]]; then
    PYTHON_CMD_EXEC=(python -u "$PY_SCRIPT" \
        --task "$TASK_NAME" \
        --headless \
        --checkpoint "$best_ckpt")
  else
    PYTHON_CMD_EXEC=(python -u "$PY_SCRIPT" \
        --task "$TASK_NAME" \
        --headless)
  fi

else
  echo "未知框架标识: '$SELECTED_FRAMEWORK'"
  exit 1
fi

########################################
# 2) 执行：加 taskset + nice 包裹
########################################

TASKSET_NICE_PREFIX=(taskset -c "$CPU_CORE_RANGE" nice -n "$CPU_NICE_VALUE")
echo "[$(date)] 启动训练，绑定核心范围: $CPU_CORE_RANGE，nice 优先级: $CPU_NICE_VALUE"
echo "[$(date)] 执行命令: ${TASKSET_NICE_PREFIX[*]} ${PYTHON_CMD_EXEC[*]}"
"${TASKSET_NICE_PREFIX[@]}" "${PYTHON_CMD_EXEC[@]}"
PYTHON_EXIT_CODE=$?

########################################
# 3) 退出处理
########################################
if [[ $PYTHON_EXIT_CODE -eq 0 ]]; then
  echo "[$(date)] Python 训练脚本成功完成，写入 train_finished.flag"
  touch "$WS1/train_finished.flag"
else
  echo "[$(date)] Python 训练脚本失败，退出码: $PYTHON_EXIT_CODE"
fi

exit $PYTHON_EXIT_CODE
