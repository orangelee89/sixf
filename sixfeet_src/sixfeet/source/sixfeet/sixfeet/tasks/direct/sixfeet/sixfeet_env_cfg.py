# sixfeet_env_cfg.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import math

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg

@configclass
class SixfeetEnvCfg(DirectRLEnvCfg):
    # -------- Basic Environment Configuration --------
    decimation: int = 2
    episode_length_s: float = 25.0
    action_space: int = 18
    observation_space: int = 4 + 3 + 18 + 2 # root_quat(4) + root_ang_vel(3) + joint_pos(18) + fixed_forward_cmd(2)
    state_space: int = 0

    # -------- Simulation Configuration --------
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=decimation,
        device="cuda:0",
        physx=PhysxCfg(enable_ccd=True, solver_type=1),
    )

    # -------- Actuator Configuration --------
    all_pd: ImplicitActuatorCfg = ImplicitActuatorCfg(
        joint_names_expr=".*",
        stiffness=150.0,
        damping=15.0
    )

    # -------- Robot Configuration --------
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        actuators={"all": all_pd},
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/lee/EE_ws/src/sixfeet_src/sixfeet/source/sixfeet/sixfeet/assets/hexapod_2/hexapod_2.usd", # <<<--- 你的六足机器人USD路径
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
    )
    # --- 指定脚趾关节的名称 ---
    # 使用正则表达式匹配所有以 '3' 结尾的 joint_X3 类型的关节
    toe_joint_names_expr: str | list[int] | None = "joint_.[3]" # <<<--- 使用正则表达式

    # -------- Scene Configuration --------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # -------- Reward Scaling Factors --------
    action_scale: float = 0.5

    rew_scale_forward_progress: float = +6.0  # 奖励机器人沿自身Y轴前进的速度
    rew_scale_body_z_align: float = +2.0      # 保持身体直立

    # 惩罚项权重 (设为负值，在奖励计算中直接相加)
    rew_scale_root_ang_vel: float = -0.15      # 惩罚机器人整体的角速度（晃动）
    rew_scale_torque: float = -0.0002          # 惩罚关节扭矩
    rew_scale_joint_vel: float = -0.005        # 惩罚过大关节角速度
    soft_joint_vel_limit: float = 10.0         # 关节角速度软限制 (rad/s)

    rew_scale_toe_orientation_penalty: float = -2.5 # 惩罚脚趾关节角度大于0的权重

    # -------- Reset Randomization --------
    root_orientation_yaw_range: float = math.pi
    reset_height_offset: float = 0.10

    # -------- Termination Conditions --------
    termination_body_z_thresh: float = 0.5
    termination_height_thresh: float = 0.15