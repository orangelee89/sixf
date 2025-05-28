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

WS="/home/lee/EE_ws/src/sixfeet_src/sixfeet"
cd "$WS"

TOTAL_TRAINING_UNITS=5000000          # 视作总训练步数 / 迭代数
TASK_NAME="Template-Sixfeet-Direct-v0" # Isaac Lab task

########################################
# 1) 选择框架并构造命令
########################################
PYTHON_CMD_EXEC=()

if [[ "$SELECTED_FRAMEWORK" == "rsl_rl" ]]; then
  # ---------- RSL-RL ----------
  LOG_DIR="$WS/logs/rsl_rl/sixfeet_ppo"
  PY_SCRIPT="$WS/scripts/rsl_rl/train.py"
  EXP_NAME="sixfeet_ppo"

  mkdir -p "$LOG_DIR"
  ckpt_path=$(ls -1t "$LOG_DIR"/*/model_*.pt 2>/dev/null | head -n1 || true)

  if [[ -n "$ckpt_path" ]]; then
    run_name=$(basename "$(dirname "$ckpt_path")")
    ckpt_file=$(basename "$ckpt_path")
    echo "[$(date)] 检测到 RSL-RL checkpoint: $ckpt_path — 继续训练"
    PYTHON_CMD_EXEC=(python -u "$PY_SCRIPT"            \
        --task "$TASK_NAME"                            \
        --resume                                      \
        --experiment_name "$EXP_NAME"                 \
        --load_run "$run_name"                        \
        --checkpoint "$ckpt_file"                     \
        --headless)
  else
    echo "[$(date)] 未找到 RSL-RL checkpoint — 从头开始训练"
    PYTHON_CMD_EXEC=(python -u "$PY_SCRIPT"            \
        --task "$TASK_NAME"                            \
        --experiment_name "$EXP_NAME"                 \
        runner_cfg.max_iterations="$TOTAL_TRAINING_UNITS" \
        --headless)
  fi

elif [[ "$SELECTED_FRAMEWORK" == "rl_games" ]]; then
  # ---------- RL-Games ----------
  LOG_ROOT="$WS/logs/rl_games"
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
  touch "$WS/train_finished.flag"
else
  echo "[$(date)] Python 训练脚本失败，退出码: $PYTHON_EXIT_CODE"
fi

exit $PYTHON_EXIT_CODE
