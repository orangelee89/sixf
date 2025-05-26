#!/usr/bin/env bash
# ========= train_supervisor.sh (V3) =========
set -euo pipefail

########## 可调参数 ##########
GPU_UTIL_TH=40                # GPU 利用率阈值 (%)
GPU_IDLE_SEC_THRESHOLD=10     # 连续空闲秒数 → 重启     ← 已改成 10
CHECK_INTERVAL=5              # 主循环检查间隔
WARMUP_AFTER_LAUNCH=30        # 新启动后忽略 GPU 空闲的秒数
MAX_CONSECUTIVE_FAST_FAILURES=3
FAST_FAILURE_WINDOW_SEC=300
##############################################

WS="/home/lee/EE_ws/src/sixfeet_src/sixfeet"
TRAIN_SH="$WS/train_cmd.sh"
PY_SCRIPT_NAME="train.py"
TERMINAL_TITLE="IsaacLab_Sixfeet_RL_Train"

PID_FILE="$WS/.term_pid"
PY_SCRIPT_PATTERN="$PY_SCRIPT_NAME"

########## 清理函数 ##########
cleanup_and_exit() {
  echo "[$(date)] Supervisor 退出，清理标志文件..."
  rm -f "$WS/train_finished.flag" "$PID_FILE"
  exit 0
}
trap cleanup_and_exit SIGINT SIGTERM

########## 打开新终端 ##########
open_new_terminal() {
  echo "[$(date)] 打开新终端 ($TERMINAL_TITLE) 并启动训练..."
  gnome-terminal --title="$TERMINAL_TITLE" --geometry=100x24 --working-directory="$WS" \
    -- bash -c "$TRAIN_SH; exec bash" &

  echo $! >"$PID_FILE"          # 记录 gnome-terminal-server PID
  sleep 3
  LAUNCH_TIME=$(date +%s)
  idle_seconds_counter=0
}

########## 杀掉旧会话 ##########
kill_previous_session() {
  echo "[$(date)] 终止旧训练进程与终端..."

  # 1) 杀掉 python / train_cmd.sh
  pkill -9 -f "$PY_SCRIPT_PATTERN" 2>/dev/null || true
  pkill -9 -f "$TRAIN_SH"          2>/dev/null || true

  # 2) 先尝试优雅关闭终端窗口
  if command -v wmctrl &>/dev/null; then
    for win in $(wmctrl -l | grep "$TERMINAL_TITLE" | awk '{print $1}'); do
      wmctrl -i -c "$win" || true
    done
  elif command -v xdotool &>/dev/null; then
    for win in $(xdotool search --onlyvisible --name "$TERMINAL_TITLE"); do
      xdotool windowclose "$win" || true
    done
  fi
  sleep 0.8

  # 3) 如果还活着 → 读 PID 文件强制 kill
  if [[ -f "$PID_FILE" ]]; then
    TERM_PID=$(cat "$PID_FILE" || true)
    if [[ -n "${TERM_PID:-}" ]] && ps -p "$TERM_PID" &>/dev/null; then
      echo "[$(date)] 窗口未关闭，发送 SIGTERM → SIGKILL ..."
      kill -TERM "$TERM_PID" 2>/dev/null || true
      sleep 0.8
      kill -KILL "$TERM_PID" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi

  # 4) 兜底：再扫一遍匹配标题的 gnome-terminal
  pkill -9 -f "$TERMINAL_TITLE" 2>/dev/null || true

  echo "[$(date)] 旧会话清理完成。"
}

########## 主逻辑 ##########
kill_previous_session
sleep 2
open_new_terminal

while true; do
  # A) 正常结束标志
  if [[ -f "$WS/train_finished.flag" ]]; then
    echo "[$(date)] train_finished.flag 检测到，Supervisor 退出。"
    kill_previous_session
    rm -f "$WS/train_finished.flag"
    exit 0
  fi

  # B) 预热期内只关心“是否崩了”
  now=$(date +%s)
  if (( now - LAUNCH_TIME < WARMUP_AFTER_LAUNCH )); then
    if ! pgrep -f "$PY_SCRIPT_PATTERN" >/dev/null; then
      echo "[$(date)] 预热期内训练进程退出，立即重启..."
      kill_previous_session
      sleep 3
      open_new_terminal
    fi
    sleep "$CHECK_INTERVAL"
    continue
  fi

  # C) GPU 利用率 / 进程存活
  gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0)
  python_alive=$(pgrep -f "$PY_SCRIPT_PATTERN" >/dev/null && echo 1 || echo 0)

  if (( python_alive == 0 )); then
    idle_seconds_counter=$GPU_IDLE_SEC_THRESHOLD
  elif (( gpu_util <= GPU_UTIL_TH )); then
    idle_seconds_counter=$(( idle_seconds_counter + CHECK_INTERVAL ))
    echo "[$(date)] GPU util ${gpu_util}% ≤ ${GPU_UTIL_TH}%  (idle ${idle_seconds_counter}/${GPU_IDLE_SEC_THRESHOLD}s)"
  else
    idle_seconds_counter=0
  fi

  # D) 到阈值 → 重启
  if (( idle_seconds_counter >= GPU_IDLE_SEC_THRESHOLD )); then
    echo "[$(date)] GPU 持续空闲或进程异常，重启训练..."
    kill_previous_session
    sleep 3
    open_new_terminal
  fi

  sleep "$CHECK_INTERVAL"
done
