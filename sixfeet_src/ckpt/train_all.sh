#!/usr/bin/env bash
# ========= train_all.sh (V3.2 – 接收并处理 supervisor 传递的训练单元数) =========
set -euo pipefail

########################################
# 0) 基本环境准备
########################################
if [[ $# -lt 1 ]]; then # 至少需要框架参数
  echo "用法: $0 <rsl_rl | rl_games> [total_training_units]"
  echo "  [total_training_units] 是可选的。如果提供并为正整数，则用作目标总迭代数。"
  echo "  如果为0、空或无效，则使用脚本内定义的默认值。"
  exit 1
fi
SELECTED_FRAMEWORK="$1"
# 从第二个参数获取用户提供的训练单元数，如果未提供，默认为 "0"
USER_PROVIDED_TRAINING_UNITS_ARG="${2:-0}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab # 确保这是你正确的conda环境名
WS1="/home/lee/EE_ws/src/sixfeet_src/ckpt"
WS="/home/lee/EE_ws/src/sixfeet_src/sixfeet"
cd "$WS1"

# 脚本内定义的默认总训练单元数
DEFAULT_INTERNAL_TOTAL_TRAINING_UNITS=10000
TOTAL_TRAINING_UNITS=$DEFAULT_INTERNAL_TOTAL_TRAINING_UNITS # 初始化为内部默认值

# 检查并使用从 supervisor 传递过来的 TOTAL_TRAINING_UNITS 参数
if [[ "$USER_PROVIDED_TRAINING_UNITS_ARG" =~ ^[1-9][0-9]*$ ]]; then # 如果是有效的正整数
    TOTAL_TRAINING_UNITS="$USER_PROVIDED_TRAINING_UNITS_ARG"
    echo "[$(date)] INFO: 使用 supervisor 传递的目标总训练单元数: $TOTAL_TRAINING_UNITS"
elif [[ "$USER_PROVIDED_TRAINING_UNITS_ARG" == "0" ]]; then # "0" 表示用户选择使用脚本内默认值
    echo "[$(date)] INFO: Supervisor 请求使用 train_all.sh 中的默认目标总训练单元数: $TOTAL_TRAINING_UNITS (为 $DEFAULT_INTERNAL_TOTAL_TRAINING_UNITS)"
else # 其他无效输入
    echo "[$(date)] WARNING: 从 supervisor 收到无效的目标总训练单元数参数 ('$USER_PROVIDED_TRAINING_UNITS_ARG')。将使用默认值: $TOTAL_TRAINING_UNITS (为 $DEFAULT_INTERNAL_TOTAL_TRAINING_UNITS)"
fi

TASK_NAME="Template-Sixfeet-Direct-v0" # Isaac Lab task

########################################
# 1) 选择框架并构造命令
########################################
PYTHON_CMD_EXEC=()
# 本次脚本执行的迭代次数
# 对于新训练，它等于上面确定的 TOTAL_TRAINING_UNITS
# 对于继续训练，它将是 TOTAL_TRAINING_UNITS 减去已完成的迭代数
iterations_for_this_run=$TOTAL_TRAINING_UNITS


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
    echo "[$(date)] 检测到 RSL-RL checkpoint: $ckpt_path"

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
        iterations_for_this_run=0 # 已达到或超过目标，本次不运行或运行0次迭代
        echo "[$(date)] 目标总迭代数 $TOTAL_TRAINING_UNITS 已达到或超过checkpoint中的 $completed_iterations 迭代。本次运行迭代数设为0."
      fi
    else
      echo "[$(date)] 警告: 无法从checkpoint文件名 '$ckpt_file' 中解析出有效的已完成迭代数。"
      echo "[$(date)] 将默认基于已确定的 iterations_for_this_run ($iterations_for_this_run) 执行（这通常意味着按总目标减去0，即整个目标长度）。"
      # iterations_for_this_run 此时等于 TOTAL_TRAINING_UNITS
    fi
    
    echo "[$(date)] — 继续训练 (RSL-RL)，计划运行 $iterations_for_this_run 迭代。"
    PYTHON_CMD_EXEC=(python -u "$PY_SCRIPT"            \
        --task "$TASK_NAME"                            \
        --resume                                      \
        --experiment_name "$EXP_NAME"                 \
        --load_run "$run_name"                        \
        --checkpoint "$ckpt_file"                     \
        --max_iterations "$iterations_for_this_run"   \
        --headless)
  else
    echo "[$(date)] 未找到 RSL-RL checkpoint — 从头开始训练，计划运行 $iterations_for_this_run 迭代。"
    PYTHON_CMD_EXEC=(python -u "$PY_SCRIPT"            \
        --task "$TASK_NAME"                            \
        --experiment_name "$EXP_NAME"                 \
        --max_iterations "$iterations_for_this_run" \
        --headless)
  fi

elif [[ "$SELECTED_FRAMEWORK" == "rl_games" ]]; then
  # ---------- RL-Games ----------
  # RL-Games 的迭代控制通常由其配置文件中的 train_steps 或 max_epochs 决定。
  # --checkpoint 参数使其从检查点加载，然后它会继续跑到配置中定义的总训练量。
  # 如果要传递类似 "本次运行多少步" 的参数，需要你的 rl_games/train.py 支持这个。
  # 目前，我们假设它会自行处理。
  LOG_ROOT="$WS1/logs/rl_games"
  EXP_NAME="sixfeet_ppo" 
  PY_SCRIPT="$WS/scripts/rl_games/train.py"

  best_ckpt=$(ls -1t "$LOG_ROOT/$EXP_NAME"/*/nn/sixfeet_ppo.pth 2>/dev/null | head -n1 || true)

  if [[ -n "$best_ckpt" ]]; then
    ckpt="$best_ckpt"
    echo "[$(date)] 选用 RL-Games BEST checkpoint: $ckpt"
  else
    ckpt=$(ls -1t "$LOG_ROOT/$EXP_NAME"/*/nn/last_*.pth 2>/dev/null | head -n1 || true)
    [[ -n "$ckpt" ]] \
      && echo "[$(date)] 未找到 RL-Games sixfeet_ppo.pth，改用最新 last_: $ckpt"
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