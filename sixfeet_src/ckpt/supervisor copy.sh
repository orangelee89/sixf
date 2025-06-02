#!/usr/bin/env bash
# ========= train_supervisor.sh (V3.1 - 支持用户输入训练单元数) =========
set -euo pipefail

########## 可调参数 ##########
GPU_UTIL_TH=40
GPU_IDLE_SEC_THRESHOLD=10
CHECK_INTERVAL=5
WARMUP_AFTER_LAUNCH=30
MAX_CONSECUTIVE_FAST_FAILURES=3
FAST_FAILURE_WINDOW_SEC=300
##############################################

WS="/home/lee/EE_ws/src/sixfeet_src/ckpt"
TRAIN_SH="$WS/train_all.sh"
PY_SCRIPT_NAME="train.py"
TERMINAL_TITLE="IsaacLab_Sixfeet_RL_Train"
PID_FILE="$WS/.term_pid"
PY_SCRIPT_PATTERN="$PY_SCRIPT_NAME"

FRAMEWORK_ARG=""
USER_DECLARED_TOTAL_TRAINING_UNITS="0" # 初始化为 "0"，表示使用 train_all.sh 的默认值

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
      1) FRAMEWORK_ARG="rl_games"; echo "[$(date)] 已选择 RL-Games。"; break ;;
      2) FRAMEWORK_ARG="rsl_rl"; echo "[$(date)] 已选择 RSL-RL。"; break ;;
      *) echo "[$(date)] 无效选项 '$choice'。请重新输入。" ;;
    esac
  done
}

########## 提示用户输入训练单元数 ##########
prompt_for_training_units() {
  local units_input
  echo "------------------------------------------"
  read -r -p "请输入目标总训练单元数 (例如 10000, 直接回车或输入0则使用 train_all.sh 中的默认值): " units_input
  if [[ "$units_input" =~ ^[1-9][0-9]*$ ]]; then # 检查是否为正整数
    USER_DECLARED_TOTAL_TRAINING_UNITS="$units_input"
    echo "[$(date)] 用户设定目标总训练单元数: $USER_DECLARED_TOTAL_TRAINING_UNITS"
  elif [[ -z "$units_input" ]] || [[ "$units_input" == "0" ]]; then
    USER_DECLARED_TOTAL_TRAINING_UNITS="0" # 明确传递 "0" 表示使用默认值
    echo "[$(date)] 用户选择使用 train_all.sh 中的默认目标总训练单元数。"
  else
    USER_DECLARED_TOTAL_TRAINING_UNITS="0" # 无效输入也视为使用默认值
    echo "[$(date)] 无效输入 '$units_input'。将使用 train_all.sh 中的默认目标总训练单元数。"
  fi
  echo "------------------------------------------"
}

########## 清理函数 ##########
cleanup_and_exit() {
  echo "[$(date)] Supervisor 退出，清理标志文件和终端..."
  kill_previous_session || true
  rm -f "$WS/train_finished.flag"
  echo "[$(date)] 清理完成。"
  exit 0
}
trap cleanup_and_exit SIGINT SIGTERM

########## 打开新终端 ##########
open_new_terminal() {
  echo "[$(date)] 打开新终端 ($TERMINAL_TITLE) 并使用框架 '$FRAMEWORK_ARG' 和训练单元数 '$USER_DECLARED_TOTAL_TRAINING_UNITS' 启动训练..."
  # 将 FRAMEWORK_ARG 和 USER_DECLARED_TOTAL_TRAINING_UNITS 作为参数传递给 TRAIN_SH
  gnome-terminal --title="$TERMINAL_TITLE" --geometry=100x24 --working-directory="$WS" \
    -- bash -c "$TRAIN_SH \"$FRAMEWORK_ARG\" \"$USER_DECLARED_TOTAL_TRAINING_UNITS\"; echo -e \"\n[$(date)] '$TRAIN_SH' 执行完毕。此终端将保持打开。\n按 Ctrl+D 或输入 exit 关闭此终端。\"; exec bash" &

  echo $! >"$PID_FILE"
  sleep 3
  LAUNCH_TIME=$(date +%s)
  idle_seconds_counter=0
}

########## 杀掉旧会话 ##########
kill_previous_session() {
  echo "[$(date)] 终止旧训练进程与终端..."
  pkill -9 -f "$PY_SCRIPT_PATTERN" 2>/dev/null || echo "[$(date)] 未找到或无法终止 $PY_SCRIPT_PATTERN 进程。"
  pkill -9 -f "$TRAIN_SH"          2>/dev/null || echo "[$(date)] 未找到或无法终止 $TRAIN_SH 进程。"
  local terminal_closed_gracefully=false
  if command -v wmctrl &>/dev/null; then
    for win_id in $(wmctrl -l | grep "$TERMINAL_TITLE" | awk '{print $1}'); do
      echo "[$(date)] 尝试通过 wmctrl 关闭窗口 ID: $win_id ..."; wmctrl -i -c "$win_id" && terminal_closed_gracefully=true; sleep 0.2;
    done
  elif command -v xdotool &>/dev/null; then
    for win_id in $(xdotool search --onlyvisible --name "$TERMINAL_TITLE"); do
      echo "[$(date)] 尝试通过 xdotool 关闭窗口 ID: $win_id ..."; xdotool windowclose "$win_id" && terminal_closed_gracefully=true; sleep 0.2;
    done
  fi
  if $terminal_closed_gracefully; then echo "[$(date)] 已尝试优雅关闭匹配标题的终端窗口。"; sleep 0.8; fi
  if [[ -f "$PID_FILE" ]]; then
    TERM_PID_FROM_FILE=$(cat "$PID_FILE" || true)
    if [[ -n "${TERM_PID_FROM_FILE:-}" ]] && ps -p "$TERM_PID_FROM_FILE" -o comm= | grep -q "gnome-terminal-" ; then
      echo "[$(date)] 终端 PID: $TERM_PID_FROM_FILE 存活，发送 SIGTERM → SIGKILL ..."; kill -TERM "$TERM_PID_FROM_FILE" 2>/dev/null || true; sleep 0.8; kill -KILL "$TERM_PID_FROM_FILE" 2>/dev/null || true; echo "[$(date)] 已发送 SIGKILL 至 PID: $TERM_PID_FROM_FILE。";
    else echo "[$(date)] PID 文件中的 PID ($TERM_PID_FROM_FILE) 无效或进程已不存在。"; fi
    rm -f "$PID_FILE";
  else echo "[$(date)] PID 文件 '$PID_FILE' 未找到。"; fi
  pkill -9 -f "$TERMINAL_TITLE" 2>/dev/null || echo "[$(date)] 尝试按标题 '$TERMINAL_TITLE' 清理终端进程，可能无匹配或已关闭。"
  echo "[$(date)] 旧会话清理完成。"
}

########## 主逻辑 ##########
# 1. 用户选择框架
select_framework
# 2. 用户输入训练单元数
prompt_for_training_units # <<< 新增调用

# 3. 清理旧会话
kill_previous_session
sleep 2

# 4. 打开新终端开始训练 (现在会传递训练单元数)
open_new_terminal

# 5. 主监控循环 (逻辑保持不变)
while true; do
  if [[ -f "$WS/train_finished.flag" ]]; then
    echo "[$(date)] train_finished.flag 检测到，训练完成。Supervisor 退出。"
    cleanup_and_exit
  fi
  current_time_seconds=$(date +%s)
  time_since_launch=$((current_time_seconds - LAUNCH_TIME))
  python_process_alive=$(pgrep -f "$PY_SCRIPT_PATTERN" >/dev/null && echo 1 || echo 0)
  train_cmd_alive=$(pgrep -f "$TRAIN_SH" >/dev/null && echo 1 || echo 0)

  if (( python_process_alive == 0 && train_cmd_alive == 0 )); then
    if (( time_since_launch > WARMUP_AFTER_LAUNCH )); then
      echo "[$(date)] 训练进程未找到 (已运行 ${time_since_launch}s)，判定为崩溃，准备重启..."
      idle_seconds_counter=$GPU_IDLE_SEC_THRESHOLD
    elif (( time_since_launch <= WARMUP_AFTER_LAUNCH )); then
       echo "[$(date)] 预热期内训练进程退出，立即重启..."
       kill_previous_session; sleep 3;
       # select_framework # 如果需要在每次预热期崩溃后都重新选择，则取消注释
       # prompt_for_training_units # 如果需要每次预热期崩溃后都重新输入，则取消注释
       open_new_terminal;
       sleep "$CHECK_INTERVAL"; continue;
    fi
  else
    if (( time_since_launch > WARMUP_AFTER_LAUNCH )); then
      gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0)
      if (( gpu_util <= GPU_UTIL_TH )); then
        if (( python_process_alive == 1 )); then
            idle_seconds_counter=$((idle_seconds_counter + CHECK_INTERVAL))
            echo "[$(date)] GPU util ${gpu_util}% ≤ ${GPU_UTIL_TH}% (Python 进程存活, idle ${idle_seconds_counter}/${GPU_IDLE_SEC_THRESHOLD}s)"
        else
            echo "[$(date)] Python 进程未找到，但 train_all.sh 存活。GPU util ${gpu_util}%。计入空闲。"
            idle_seconds_counter=$((idle_seconds_counter + CHECK_INTERVAL))
        fi
      else
        idle_seconds_counter=0
        echo "[$(date)] GPU util ${gpu_util}% > ${GPU_UTIL_TH}%. 训练正常。"
      fi
    else
      idle_seconds_counter=0
      echo "[$(date)] 预热期 (${time_since_launch}s / ${WARMUP_AFTER_LAUNCH}s), 脚本进程存活。"
    fi
  fi

  if (( idle_seconds_counter >= GPU_IDLE_SEC_THRESHOLD )); then
    if (( time_since_launch > WARMUP_AFTER_LAUNCH )); then
        echo "[$(date)] GPU 持续空闲或进程异常 (已运行 ${time_since_launch}s)，重启训练..."
        kill_previous_session; sleep 3;
        # select_framework # 同上
        # prompt_for_training_units # 同上
        open_new_terminal
    fi
  fi
  sleep "$CHECK_INTERVAL"
done