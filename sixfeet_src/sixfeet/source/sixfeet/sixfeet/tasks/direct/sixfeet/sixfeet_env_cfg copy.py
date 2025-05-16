# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# 说明：本文件仅整理了缩进 / 空格 / 注释排版，功能与原代码一致。

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class SixfeetEnvCfg(DirectRLEnvCfg):
    # ─────────────────────────── 基础设置 ───────────────────────────
    decimation: int = 2                    # 控制频率 60 Hz (=120/decimation)
    episode_length_s: float = 6.0

    # 观测 / 动作空间尺寸
    action_space: int = 18                 # 18 × 关节目标位置
    observation_space: int = 4 + 3 + 18*2  # 4(quat)+3(ang vel)+18(pos)+18(vel)
    state_space: int = 0                   # 不使用 “state” channel

    # ─────────────────────────── 仿真配置 ───────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
    )

    # ─────────────────────────── 执行器统一 PD 参数 ────────────────
    all_joints_pd = ImplicitActuatorCfg(
    joint_names_expr=".*",          # 全部关节
    stiffness=600.0,                # 这里随你调
    damping=3.0                     # 这里随你调
    )

    # ─────────────────────────── 机器人 USD ────────────────────────
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/lee/EE_ws/src/sixfeet_src/sixfeet/source/"
                     "sixfeet/sixfeet/assets/hexapod_2/hexapod_2.usd",
        ),
        actuators={
            "all": all_joints_pd,
        },
    )

    # ─────────────────────────── 批量环境 ──────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2,                   # 2 个环境
        env_spacing=4.0,
        replicate_physics=True,
    )

    # ─────────────────────────── 其他超参 ──────────────────────────
    action_scale: float = 0.5              # [-1,1] → ±0.5 rad

    rew_scale_upright:    float = +5.0
    rew_scale_angvel:     float = -0.1
    rew_scale_torque:     float = -2e-4
    # rew_scale_collision:  float = -10.0

    # 初始 root 姿态在 ±π 随机
    root_orientation_range: float = math.pi
