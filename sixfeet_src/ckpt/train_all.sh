#!/usr/bin/env bash
# ========= train_all.sh (V3 – 支持 RL-Games / RSL-RL) =========
set -euo pipefail

########################################
# 0) 基本环境准备
########################################
if [[ $# -lt 1 ]]; then
  echo "用法: $0 <rsl_rl | rl_games>"
  exit 1
fi
SELECTED_FRAMEWORK="$1"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
WS1="/home/lee/EE_ws/src/sixfeet_src/ckpt"
WS="/home/lee/EE_ws/src/sixfeet_src/sixfeet"
cd "$WS1"

TOTAL_TRAINING_UNITS=10000         # 视作总训练步数 / 迭代数
TASK_NAME="Template-Sixfeet-Direct-v0" # Isaac Lab task

########################################
# 1) 选择框架并构造命令
########################################
PYTHON_CMD_EXEC=()

if [[ "$SELECTED_FRAMEWORK" == "rsl_rl" ]]; then
  # ---------- RSL-RL ----------
  LOG_DIR="$WS1/logs/rsl_rl/sixfeet_ppo"
  PY_SCRIPT="$WS/scripts/rsl_rl/train.py"
  EXP_NAME="sixfeet_ppo"

  mkdir -p "$LOG_DIR"
  ckpt_path=$(ls -1t "$LOG_DIR"/*/model_*.pt 2>/dev/null | head -n1 || true)
  if [[ -n "$ckpt_path" ]]; then
    run_name=$(basename "$(dirname "$ckpt_path")")
    ckpt_file=$(basename "$ckpt_path")
    echo "[$(date)] 检测到 RSL-RL checkpoint: $ckpt_path — 继续训练"

# 从 ckpt_file (例如 model_499.pt) 中提取数字部分 499
    completed_iterations_str=""
    if [[ "$ckpt_file" =~ model_([0-9]+)\.pt ]]; then
      completed_iterations_str="${BASH_REMATCH[1]}"
    fi

    if [[ -n "$completed_iterations_str" ]] && [[ "$completed_iterations_str" =~ ^[0-9]+$ ]]; then
      completed_iterations=$((completed_iterations_str))
      echo "[$(date)] 从checkpoint文件名解析得到已完成迭代数: $completed_iterations"
      if [[ $TOTAL_TRAINING_UNITS -gt $completed_iterations ]]; then
        iterations_for_this_run=$((TOTAL_TRAINING_UNITS - completed_iterations))
        echo "[$(date)] 目标总迭代数: $TOTAL_TRAINING_UNITS. 本次继续训练将运行剩余的 $iterations_for_this_run 迭代."
      else
        echo "[$(date)] 目标总迭代数 $TOTAL_TRAINING_UNITS 已达到或超过。本次运行迭代数设为默认值."
      fi
    else
      echo "[$(date)] 警告: 无法从checkpoint文件名 '$ckpt_file' 中解析出有效的已完成迭代数。"
      echo "[$(date)] 将默认运行 DECLARED_TOTAL_TRAINING_UNITS ($DECLARED_TOTAL_TRAINING_UNITS) 指定的迭代次数（如果train.py支持）。"
      # iterations_for_this_run 保持为 DECLARED_TOTAL_TRAINING_UNITS
      # 或者你可以选择在这里设置一个较小的默认值，或直接退出
    fi
    
    echo "[$(date)] — 继续训练，运行 $iterations_for_this_run 迭代。"
    PYTHON_CMD_EXEC=(python -u "$PY_SCRIPT"            \
        --task "$TASK_NAME"                            \
        --resume                                      \
        --experiment_name "$EXP_NAME"                 \
        --max_iterations "$iterations_for_this_run" \
        --load_run "$run_name"                        \
        --checkpoint "$ckpt_file"                     \
        --headless)
  else
    echo "[$(date)] 未找到 RSL-RL checkpoint — 从头开始训练"
    PYTHON_CMD_EXEC=(python -u "$PY_SCRIPT"            \
        --task "$TASK_NAME"                            \
        --experiment_name "$EXP_NAME"                 \
        --max_iterations "$TOTAL_TRAINING_UNITS" \
        --headless)
  fi

elif [[ "$SELECTED_FRAMEWORK" == "rl_games" ]]; then
  # ---------- RL-Games ----------
  LOG_ROOT="$WS1/logs/rl_games"
  EXP_NAME="sixfeet_ppo"
  PY_SCRIPT="$WS/scripts/rl_games/train.py"

  # 1) 先找 best —— 文件名固定为 sixfeet_ppo.pth
best_ckpt=$(ls -1t "$LOG_ROOT/$EXP_NAME"/*/nn/sixfeet_ppo.pth 2>/dev/null | head -n1 || true)

if [[ -n "$best_ckpt" ]]; then
  ckpt="$best_ckpt"
  echo "[$(date)] 选用 BEST checkpoint: $ckpt"
else
  # 2) 若无 best，再找最新 last_*.pth
  ckpt=$(ls -1t "$LOG_ROOT/$EXP_NAME"/*/nn/last_*.pth 2>/dev/null | head -n1 || true)
  [[ -n "$ckpt" ]] \
    && echo "[$(date)] 未找到 sixfeet_ppo.pth，改用最新 last_: $ckpt"
fi

  if [[ -n "$ckpt" ]]; then
    echo "[$(date)] 检测到 RL-Games checkpoint: $ckpt — 继续训练"
    PYTHON_CMD_EXEC=(python -u "$PY_SCRIPT"        \
        --task "$TASK_NAME"                        \
        --headless                                 \
        --checkpoint "$ckpt")
  else
    echo "[$(date)] 未找到 RL-Games checkpoint — 从头开始训练"
    PYTHON_CMD_EXEC=(python -u "$PY_SCRIPT"        \
        --task "$TASK_NAME"                        \
        --headless)
  fi

else
  echo "未知框架标识: '$SELECTED_FRAMEWORK'"
  exit 1
fi

########################################
# 2) 执行
########################################
echo "[$(date)] 即将执行: ${PYTHON_CMD_EXEC[*]}"
"${PYTHON_CMD_EXEC[@]}"
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
