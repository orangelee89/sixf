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
from isaaclab.terrains import TerrainImporterCfg

@configclass
class SixfeetEnvCfg(DirectRLEnvCfg):
    # -------- 基本环境配置 --------
    decimation: int = 2
    episode_length_s: float = 20.0 # 可以先缩短 episode 长度，专注于快速站立
    action_space: int = 18
    observation_space: int = 4 + 3 + 18 + 2 # root_quat(4) + root_ang_vel_local(3) + joint_pos(18) + fixed_forward_cmd(2)
    state_space: int = 0

    # -------- 仿真配置 --------
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=decimation,
        device="cuda:0",
        physx=PhysxCfg(
            enable_ccd=True,
            solver_type=1,
            # num_position_iterations=8,
            # num_velocity_iterations=1
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.0,
            friction_combine_mode="average",
            restitution_combine_mode="average"
        )
    )

    # -------- 地形配置 --------
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
    )

    # -------- 机器人配置 --------
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/lee/EE_ws/src/sixfeet_src/sixfeet/source/sixfeet/sixfeet/assets/hexapod_2/hexapod_2.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, # 保持启用，看是否能学会避免
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.15), # 初始Z高度设低一些，给学习站立的空间
            rot=(1.0, 0.0, 0.0, 0.0), # (w,x,y,z)
            joint_pos={ # !! 确保这些初始关节角度能让机器人大致趴着或蹲着 !!
                "joint_11": 0.0, "joint_21": 0.0, "joint_31": 0.0, "joint_41": 0.0, "joint_51": 0.0, "joint_61": 0.0,
                "joint_12": math.radians(45), "joint_22": math.radians(45), "joint_32": math.radians(45), # 腿部更弯曲
                "joint_42": math.radians(45), "joint_52": math.radians(45), "joint_62": math.radians(45),
                "joint_13": math.radians(-90), "joint_23": math.radians(-90), "joint_33": math.radians(-90), # 腿部更弯曲
                "joint_43": math.radians(-90), "joint_53": math.radians(-90), "joint_63": math.radians(-90),
            }
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=".*",
                stiffness=200.0, # 尝试一个中等刚度
                damping=20.0,    # 匹配的阻尼
                effort_limit=25.0, 
                velocity_limit=15.0 
            )
        }
    )

    # --- 指定脚趾关节的名称 (暂时禁用其惩罚) ---
    toe_joint_names_expr: str | list[int] | None = "joint_.[3]" 

    # -------- 场景配置 --------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # -------- 奖励缩放因子 (专注于站立) --------
    action_scale: float = 0.5

    # --- 主要的正向激励 ---
    rew_scale_target_height: float = +25.0      # <<--- 强烈的站立高度奖励 (核心)
    target_height_m: float = 0.30              # <<--- 目标站立高度 (米)
    
    rew_scale_forward_progress: float = +0.1   # <<--- 暂时大幅降低前进奖励
    rew_scale_upright: float = +1.5              # 保持身体Z轴与世界Z轴对齐仍然重要
    rew_scale_alive: float = +0.01               # 非常小的存活奖励

    # --- 行为平滑与效率相关的惩罚 (暂时减小绝对值或设为0) ---
    rew_scale_action_cost: float = -0.0001       # 大幅减小
    rew_scale_action_rate: float = -0.001        # 大幅减小

    rew_scale_torque: float = -0.00001          # 大幅减小
    rew_scale_joint_vel: float = -0.0001         # 大幅减小
    soft_joint_vel_limit: float = 15.0           # 可以适当放宽

    # --- 姿态与运动稳定性相关的惩罚 (保持一些，避免完全乱动) ---
    rew_scale_root_ang_vel: float = -0.05        # 减小

    # --- 行为约束相关的惩罚 (权重为负) ---
    rew_scale_dof_at_limit: float = -0.05
    rew_scale_toe_orientation_penalty: float = -1.0 # <<--- 脚趾惩罚权重 (之前可能是0)
    # --- 新增：低高度惩罚 ---
    rew_scale_low_height_penalty: float = -15.0 # <<--- 低于特定高度的惩罚权重 (可以设得比较大)
    min_height_penalty_threshold: float = 0.28 # <<--- 低于此高度 (0.28m) 则开始惩罚 (略低于目标站立高度0.30m)


    # --- 终止状态相关的惩罚 (保持，让机器人学会避免失败) ---
    rew_scale_termination: float = -10.0         # 因摔倒等失败的惩罚

    # -------- 重置随机化 --------
    root_orientation_yaw_range: float = math.pi # 初始朝向仍然随机，以增加鲁棒性
    reset_height_offset: float = 0.02           # 较小的重置高度偏移，因为初始姿态已经较低

    # -------- 终止条件 --------
    termination_body_z_thresh: float = 0.35 # 身体Z轴投影低于此值则终止 (衡量是否严重翻倒)
    termination_height_thresh: float = 0.10 # 机器人根部高度低于此值则终止 (绝对高度，比目标低很多)