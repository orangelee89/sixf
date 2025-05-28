#!/usr/bin/env bash
# ========= train_cmd.sh (Fixed version) =========
set -euo pipefail

# ① 激活 conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# ② 配置路径
WS="/home/lee/EE_ws/src/sixfeet_src/sixfeet"
cd "$WS"

LOG_DIR="$WS/logs/rsl_rl/sixfeet_ppo"
PY_SCRIPT="$WS/scripts/rsl_rl/train.py"
TASK="Template-Sixfeet-Direct-v0"
# TOTAL_ITERS=5000000

# ③ 查找最新 checkpoint
ckpt_path=$(ls -1t "$LOG_DIR"/*/model_*.pt 2>/dev/null | head -n1 || true)
# echo "[INFO] 检测到的最新 checkpoint: $ckpt_path"
if [[ -n "$ckpt_path" ]]; then
    echo "[INFO] 检测到 checkpoint: $ckpt_path"

    # 提取轮数，例如 model_4500.pt → 4500
    ckpt_filename=$(basename "$ckpt_path")
    ckpt_iter=$(echo "$ckpt_filename" | grep -oP '\d+(?=\.pt$)')
    echo "[INFO] 已训练迭代数: $ckpt_iter"
    # echo "debug] ckpt_filename: $ckpt_filename"
    run_name=$(basename "$(dirname "$ckpt_path")")    # 提取 run 文件夹名
    echo "[INFO] run_name: $run_name"

    echo "[INFO] 即将继续训练，总目标 $TOTAL_ITERS 轮，从迭代 $ckpt_iter 开始"
    echo "[DEBUG] LOG_DIR = $LOG_DIR"
    python -u "$PY_SCRIPT" \
        --task "$TASK" \
        --resume \
        --experiment_name sixfeet_ppo \
        --load_run "$run_name" \
        # --max_iterations 5000000 \
        --headless \
        --checkpoint "$ckpt_filename" 
else
    echo "[INFO] 未检测到任何 checkpoint，重新训练 $TOTAL_ITERS 轮"

    python -u "$PY_SCRIPT" \
        --task "$TASK" \
        --experiment_name sixfeet_ppo \
        agent_cfg.max_iterations="$TOTAL_ITERS" \
        --headless
fi

echo "[$(date)] 训练全部完成，写入结束标志"
exit 0
