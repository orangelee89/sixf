# sixfeet_env_cfg.py
from __future__ import annotations
import math
from typing import List, Optional, Dict # 确保导入 Dict

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg

@configclass
class SixfeetEnvCfg(DirectRLEnvCfg):
    # -------- 基本环境配置 --------
    decimation: int = 2
    episode_length_s: float = 20.0
    action_space: int = 18
    observation_space: int = 45 # 根据你的定义
    state_space: int = 0

    # -------- 仿真配置 --------
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=decimation,
        device="cuda:0",
        physx=PhysxCfg(enable_ccd=True, solver_type=1),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8, dynamic_friction=0.6, restitution=0.0,
            friction_combine_mode="multiply", restitution_combine_mode="multiply"
        )
    )

    # -------- 地形配置 --------
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=0.8, restitution=0.0,
            friction_combine_mode="multiply", restitution_combine_mode="multiply"
        ),
    )

    # -------- 机器人配置 --------
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/lee/EE_ws/src/sixfeet_src/sixfeet/source/sixfeet/sixfeet/assets/hexapod/hexapod.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=10.0, enable_gyroscopic_forces=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=12,
                solver_velocity_iteration_count=1, sleep_threshold=0.005, stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0001), #   
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_vel={ f"joint_{l}{j}": 0.0 for l in range(1, 7) for j in range(1, 4) }
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=".*", stiffness=20.0, damping=12.0,
                effort_limit_sim=5.0, velocity_limit_sim=5.0
            )
        }
    )

    # -------- 传感器配置 --------
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=1, update_period=0.0,
        track_air_time=True
    )

    # --- 关节和连杆名称表达式 ---
    toe_joint_names_expr: str | list[int] | None = "joint_.[3]"
    undesired_contact_link_names_expr: str = "(thigh_link_[1-6]1|shin_link_[1-6]2)"
    base_link_name: str = "base_link"
    foot_link_name_pattern_for_custom_align: str = r"foot_link_[1-6]3"


    # -------- 场景配置 --------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True,
    )

    # -------- 离散指令配置 --------
    command_profile: Dict[str, float] = {
        "reference_linear_speed": 0.0,
        "reference_angular_rate": 0.0,
        "command_mode_duration_s": 20.0,  # <<< 已修正为浮点数
        "stand_still_prob": 1.0,
        "num_command_modes": 7.0          # <<< 已修正为浮点数
    }
    # ---初始姿态随机化模式 ---
    initial_pose_randomization_mode: int = 1 # 0 翻转， 1 随机（Roll，Pitch，yaw）
    randomize_initial_joint_poses: bool = False # 随机 joint angle

    # --- 目标站立姿态的关节角度 (弧度) ---
    # 键名必须与 self.robot.dof_names 中的实际关节名称完全对应，或在env.__init__中正确映射
    target_standing_joint_angles_dict: Dict[str, float] = {
         "joint_11": 0.0, "joint_21": 0.0, "joint_31": 0.0,
         "joint_41": 0.0, "joint_51": 0.0, "joint_61": 0.0,
         "joint_12": math.radians(45), "joint_22": math.radians(45), "joint_32": math.radians(45),
         "joint_42": math.radians(45), "joint_52": math.radians(45), "joint_62": math.radians(45),
         "joint_13": math.radians(-90), "joint_23": math.radians(-90), "joint_33": math.radians(-90),
         "joint_43": math.radians(-90), "joint_53": math.radians(-90), "joint_63": math.radians(-90),
    }

    # -------- 奖励缩放因子 --------
    action_scale: float = 0.5
    rew_scale_move_in_commanded_direction: float = 0.0
    rew_scale_achieve_reference_angular_rate: float = 0.0
    rew_scale_alive: float = +0.1               #  生存奖励

    rew_scale_target_height: float = +20.0      #  达到高度奖励
    target_height_m: float = 0.30              # 目标高度
    target_height_reward_sharpness: float = 100.0 # 高度奖励比例

    rew_scale_orientation_deviation: float = -50.0 # baselink 朝向偏差惩罚
    rew_scale_successful_flip: float = 40.0    # 成功翻转奖励
    rew_scale_self_collision: float = -100.0      # 自碰撞惩罚

    rew_scale_custom_foot_axis_alignment: float = 25.0 # 自定义脚轴对齐奖励 （foot link 轴对准世界Z）
    rew_scale_all_feet_stable_stand: float = 25.0    # 所有脚稳定站立奖励
    rew_scale_airborne_feet_penalty: float = -50.0   # 空中脚惩罚(原为-2.0)

    rew_scale_standing_pose_imitation: float = 25.0  # 站立姿态模仿奖励
    standing_pose_imitation_sharpness: float = 2.0 # 站立姿态模仿奖励锐化系数

    rew_scale_action_cost: float = -0.0001                 # 动作成本
    rew_scale_action_rate: float = -0.01                  # 动作速率成本
    rew_scale_joint_torques: float = -1.0e-6                    # 关节扭矩成本
    rew_scale_joint_accel: float = -1.0e-7                    # 关节加速度成本
    rew_scale_lin_vel_z_penalty: float = -3.0              # 线速度Z轴惩罚
    rew_scale_ang_vel_xy_penalty: float = -0.1              # 角速度XY轴惩罚
    rew_scale_flat_orientation: float = -25.0                 # 平坦朝向惩罚
    rew_scale_unwanted_movement_penalty: float = -10.0      # 不必要的移动惩罚
    rew_scale_dof_at_limit: float = -0.5                     # 关节在极限位置的惩罚
    rew_scale_toe_orientation_penalty: float = -10.0           # 脚趾朝向惩罚
    rew_scale_low_height_penalty: float = -20.0                   # 低高度惩罚
    min_height_penalty_threshold: float = 0.30                  # 低于此高度时施加惩罚

    rew_scale_undesired_contact: float = -15.0               # 不希望的接触惩罚                      
    rew_scale_termination: float = -20.0                        # 终止条件惩罚

    joint_limit_penalty_threshold_percent: float = 0.05            # 关节极限惩罚阈值百分比

    # -------- 重置随机化 --------
    root_orientation_yaw_range: float = math.pi                # 根朝向偏航范围
    reset_height_offset: float = 0.0                           # 重置高度偏移

    # -------- 终止条件 --------
    termination_body_z_thresh: float = 0.95                 # 身体Z轴位置阈值
    termination_height_thresh: float = 0.00      # 身体高度阈值
    termination_base_contact: bool = False        #  身体接触地面终止
    
    orientation_termination_angle_limit_deg: float = 90.0 # 身体朝向终止角度限制（度）(原为95，改为90)
    flatness_threshold_for_height_termination: float = 0.2 # 平坦度阈值（高度终止）