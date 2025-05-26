#!/usr/bin/env bash
# ========= train_cmd.sh =========
set -euo pipefail

# ① 激活 conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# ② 路径配置
WS="/home/lee/EE_ws/src/sixfeet_src/sixfeet"
cd "$WS"
LOG_DIR="$WS/logs/rl_games/sixfeet_ppo"
PY_SCRIPT="$WS/scripts/rl_games/train.py"
TASK="Template-Sixfeet-Direct-v0"
MAX_ITERS=5000000

# ③ 找最新 .pth
ckpt=$(ls -1t "$LOG_DIR"/*/nn/*.pth 2>/dev/null | head -n1 || true)

echo "[$(date)] ========= 训练启动 ========="
if [[ -n "$ckpt" ]]; then
    echo "使用 checkpoint: $ckpt"
    python -u "$PY_SCRIPT" --task "$TASK" --headless \
           --checkpoint "$ckpt" --max_iterations "$MAX_ITERS"
else
    echo "未检测到 checkpoint，重新训练"
    python -u "$PY_SCRIPT" --task "$TASK" --headless \
           --max_iterations "$MAX_ITERS"
fi

# ④ 正常完成：写标志并退出 0
echo "[$(date)] 训练全部完成，写入结束标志"
touch "$WS/train_finished.flag"
exit 0
