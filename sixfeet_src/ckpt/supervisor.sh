#!/usr/bin/env bash
# ========= train_supervisor.sh (V3 - Modified for Framework Choice) =========
set -euo pipefail

########## 可调参数 ##########
GPU_UTIL_TH=40                # GPU 利用率阈值 (%)
GPU_IDLE_SEC_THRESHOLD=10     # 连续空闲秒数 → 重启
CHECK_INTERVAL=5              # 主循环检查间隔
WARMUP_AFTER_LAUNCH=30        # 新启动后忽略 GPU 空闲的秒数
# MAX_CONSECUTIVE_FAST_FAILURES 和 FAST_FAILURE_WINDOW_SEC 保留原样
MAX_CONSECUTIVE_FAST_FAILURES=3
FAST_FAILURE_WINDOW_SEC=300
##############################################

WS="/home/lee/EE_ws/src/sixfeet_src/ckpt"
TRAIN_SH="$WS/train_all.sh"
# 注意: PY_SCRIPT_NAME 和 PY_SCRIPT_PATTERN 可能需要根据实际Python脚本名调整
# 如果两个框架的Python训练脚本都叫 train.py (在不同目录下)，这个可能仍然有效。
# 否则，你可能需要一个更通用的模式，或者让 train_all.sh 将实际的脚本名/PID 传递出来。
PY_SCRIPT_NAME="train.py"
TERMINAL_TITLE="IsaacLab_Sixfeet_RL_Train" # 终端窗口标题

PID_FILE="$WS/.term_pid"              # 存储终端进程PID的文件
PY_SCRIPT_PATTERN="$PY_SCRIPT_NAME"   # 用于 pgrep 匹配 Python 训练进程

FRAMEWORK_ARG="" # 用于存储用户选择的框架参数 (rl_games 或 rsl_rl)

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
      1)
        FRAMEWORK_ARG="rl_games"
        echo "[$(date)] 已选择 RL-Games (传递给 train_all.sh 的参数: $FRAMEWORK_ARG)。"
        break
        ;;
      2)
        FRAMEWORK_ARG="rsl_rl"
        echo "[$(date)] 已选择 RSL-RL (传递给 train_all.sh 的参数: $FRAMEWORK_ARG)。"
        break
        ;;
      *)
        echo "[$(date)] 无效选项 '$choice'。请重新输入。"
        ;;
    esac
  done
}

########## 清理函数 ##########
cleanup_and_exit() {
  echo "[$(date)] Supervisor 退出，清理标志文件和终端..."
  # 调用 kill_previous_session 来确保终端也被关闭
  kill_previous_session || true # 忽略可能的错误，确保 rm 执行
  rm -f "$WS/train_finished.flag" # PID_FILE 会在 kill_previous_session 中删除
  echo "[$(date)] 清理完成。"
  exit 0
}
trap cleanup_and_exit SIGINT SIGTERM

########## 打开新终端 ##########
open_new_terminal() {
  echo "[$(date)] 打开新终端 ($TERMINAL_TITLE) 并使用框架 '$FRAMEWORK_ARG' 启动训练..."
  # 将 FRAMEWORK_ARG 作为参数传递给 TRAIN_SH
  gnome-terminal --title="$TERMINAL_TITLE" --geometry=100x24 --working-directory="$WS" \
    -- bash -c "$TRAIN_SH \"$FRAMEWORK_ARG\"; echo -e \"\n[$(date)] '$TRAIN_SH' 执行完毕。此终端将保持打开。\n按 Ctrl+D 或输入 exit 关闭此终端。\"; exec bash" &

  echo $! >"$PID_FILE"          # 记录 gnome-terminal 进程的 PID
  sleep 3                      # 等待终端和命令启动
  LAUNCH_TIME=$(date +%s)      # 更新启动时间
  idle_seconds_counter=0       # 重置空闲计数器
}

########## 杀掉旧会话 ##########
kill_previous_session() {
  echo "[$(date)] 终止旧训练进程与终端..."

  # 1) 杀掉 python 训练脚本和 train_all.sh 脚本
  # 使用 pkill -f 来匹配完整命令行，避免误杀
  pkill -9 -f "$PY_SCRIPT_PATTERN" 2>/dev/null || echo "[$(date)] 未找到或无法终止 $PY_SCRIPT_PATTERN 进程。"
  pkill -9 -f "$TRAIN_SH"          2>/dev/null || echo "[$(date)] 未找到或无法终止 $TRAIN_SH 进程。"

  # 2) 先尝试优雅关闭终端窗口 (如果窗口管理器工具可用)
  local terminal_closed_gracefully=false
  if command -v wmctrl &>/dev/null; then
    for win_id in $(wmctrl -l | grep "$TERMINAL_TITLE" | awk '{print $1}'); do
      echo "[$(date)] 尝试通过 wmctrl 关闭窗口 ID: $win_id ..."
      wmctrl -i -c "$win_id" && terminal_closed_gracefully=true
      sleep 0.2 # 给窗口一点反应时间
    done
  elif command -v xdotool &>/dev/null; then
    for win_id in $(xdotool search --onlyvisible --name "$TERMINAL_TITLE"); do
      echo "[$(date)] 尝试通过 xdotool 关闭窗口 ID: $win_id ..."
      xdotool windowclose "$win_id" && terminal_closed_gracefully=true
      sleep 0.2 # 给窗口一点反应时间
    done
  fi

  if $terminal_closed_gracefully; then
    echo "[$(date)] 已尝试优雅关闭匹配标题的终端窗口。"
    sleep 0.8 # 等待窗口关闭过程
  fi

  # 3) 如果 PID 文件存在且进程存活，则强制 kill gnome-terminal 进程
  if [[ -f "$PID_FILE" ]]; then
    TERM_PID_FROM_FILE=$(cat "$PID_FILE" || true) # 读取PID
    if [[ -n "${TERM_PID_FROM_FILE:-}" ]] && ps -p "$TERM_PID_FROM_FILE" -o comm= | grep -q "gnome-terminal-" ; then # 检查是否是gnome-terminal进程且存活
      echo "[$(date)] 终端窗口 '$TERMINAL_TITLE' (PID: $TERM_PID_FROM_FILE 来自文件) 似乎仍存活，发送 SIGTERM → SIGKILL ..."
      kill -TERM "$TERM_PID_FROM_FILE" 2>/dev/null || true
      sleep 0.8 # 等待 SIGTERM
      kill -KILL "$TERM_PID_FROM_FILE" 2>/dev/null || true
      echo "[$(date)] 已发送 SIGKILL 至 PID: $TERM_PID_FROM_FILE。"
    else
      echo "[$(date)] PID 文件中的 PID ($TERM_PID_FROM_FILE) 无效或进程已不存在。"
    fi
    rm -f "$PID_FILE" # 清理PID文件
  else
    echo "[$(date)] PID 文件 '$PID_FILE' 未找到。"
  fi

  # 4) 兜底：再次按标题模式查杀可能残留的 gnome-terminal 进程
  # 这一步要非常小心，确保不会误杀其他窗口。
  # pkill -f "gnome-terminal.*--title=$TERMINAL_TITLE" 是一种更精确的方式，但仍需谨慎。
  # 为了更安全，可以考虑注释掉，或者只在特定情况下启用。
  # 鉴于上面已经用了更精确的 PID 文件方法，这里的 pkill -9 -f "$TERMINAL_TITLE" 风险较高，已在原脚本中，此处保留。
  # 如果需要更安全，应替换为更精确的 pgrep + kill 组合，或完全依赖PID文件。
  pkill -9 -f "$TERMINAL_TITLE" 2>/dev/null || echo "[$(date)] 尝试按标题 '$TERMINAL_TITLE' 清理终端进程，可能无匹配或已关闭。"


  echo "[$(date)] 旧会话清理完成。"
}

########## 主逻辑 ##########
# 1. 在脚本开始时让用户选择框架
select_framework

# 2. 清理任何可能存在的旧会话
kill_previous_session
sleep 2 # 清理后稍作等待，确保端口等资源释放

# 3. 打开新终端并开始第一次训练
open_new_terminal

# 4. 主监控循环 (基本逻辑保持不变，但open_new_terminal会使用选定的框架)
while true; do
  # A) 检查正常结束标志
  if [[ -f "$WS/train_finished.flag" ]]; then
    echo "[$(date)] train_finished.flag 检测到，训练完成。Supervisor 退出。"
    cleanup_and_exit # 使用包含 kill_previous_session 的清理函数
  fi

  current_time_seconds=$(date +%s)
  time_since_launch=$((current_time_seconds - LAUNCH_TIME))

  # B) 检查 Python 训练进程是否存活
  # 同时检查 PY_SCRIPT_PATTERN (如 train.py) 和 TRAIN_SH (train_all.sh)
  # 因为 train_all.sh 可能会在 python 脚本崩溃后很快退出
  python_process_alive=$(pgrep -f "$PY_SCRIPT_PATTERN" >/dev/null && echo 1 || echo 0)
  train_cmd_alive=$(pgrep -f "$TRAIN_SH" >/dev/null && echo 1 || echo 0)

  if (( python_process_alive == 0 && train_cmd_alive == 0 )); then # 两者都不在了
    if (( time_since_launch > WARMUP_AFTER_LAUNCH )); then
      echo "[$(date)] 训练进程 ($PY_SCRIPT_NAME 和 $TRAIN_SH) 未找到 (已运行 ${time_since_launch}s)，判定为崩溃，准备重启..."
      idle_seconds_counter=$GPU_IDLE_SEC_THRESHOLD # 直接触发重启条件
    elif (( time_since_launch <= WARMUP_AFTER_LAUNCH )); then
       echo "[$(date)] 预热期内训练进程 ($PY_SCRIPT_NAME 和 $TRAIN_SH) 退出，立即重启..."
       kill_previous_session
       sleep 3
       # 预热期崩溃，默认使用上次的选择，如果需要重新选择，取消下一行注释
       # select_framework
       open_new_terminal
       # WARMUP_AFTER_LAUNCH 和 LAUNCH_TIME 会在 open_new_terminal 中重置
       sleep "$CHECK_INTERVAL" # 等待下一次检查
       continue
    fi
  else # 至少有一个脚本在运行，检查 GPU 利用率 (仅在预热期后)
    if (( time_since_launch > WARMUP_AFTER_LAUNCH )); then
      gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0)
      if (( gpu_util <= GPU_UTIL_TH )); then
        if (( python_process_alive == 1 )); then # Python 脚本还在，但GPU空闲
            idle_seconds_counter=$((idle_seconds_counter + CHECK_INTERVAL))
            echo "[$(date)] GPU util ${gpu_util}% ≤ ${GPU_UTIL_TH}% (Python 进程存活, idle ${idle_seconds_counter}/${GPU_IDLE_SEC_THRESHOLD}s)"
        else # Python 脚本不在了，但 train_all.sh 可能还在 (不太可能，但作为一种情况)
            echo "[$(date)] Python 进程 ($PY_SCRIPT_PATTERN) 未找到，但 train_all.sh 存活。GPU util ${gpu_util}%。计入空闲。"
            idle_seconds_counter=$((idle_seconds_counter + CHECK_INTERVAL))
        fi
      else
        # GPU 忙碌 (且至少一个相关脚本存活)，重置空_idle_seconds_counter=0
        echo "[$(date)] GPU util ${gpu_util}% > ${GPU_UTIL_TH}%. 训练正常。"
      fi
    else
      # 在预热期内，只要进程活着就不增加 idle_seconds_counter
      idle_seconds_counter=0
      echo "[$(date)] 预热期 (${time_since_launch}s / ${WARMUP_AFTER_LAUNCH}s), 脚本进程存活。"
    fi
  fi

  # D) 达到空闲/崩溃重启阈值
  if (( idle_seconds_counter >= GPU_IDLE_SEC_THRESHOLD )); then
    if (( time_since_launch > WARMUP_AFTER_LAUNCH )); then
        echo "[$(date)] GPU 持续空闲或进程异常 (已运行 ${time_since_launch}s)，重启训练..."
        kill_previous_session
        sleep 3
        # select_framework # 同上，决定是否在每次重启时都重新选择框架
        open_new_terminal
    fi
  fi

  sleep "$CHECK_INTERVAL"
done