# sixfeet_env_cfg.py (Simplified Commands, No root_lin_vel_b in obs)
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
from isaaclab.sensors import ContactSensorCfg

@configclass
class SixfeetEnvCfg(DirectRLEnvCfg):
    # -------- 基本环境配置 --------
    decimation: int = 2
    episode_length_s: float = 20.0
    action_space: int = 18

    # ---- 新的观测空间定义 (移除 root_lin_vel_b) ----
    # projected_gravity_b (3) + root_ang_vel_b (3) + discrete_commands (3)
    # + joint_pos_rel (18) + joint_vel (18)
    # Total = 3 + 3 + 3 + 18 + 18 = 45 dimensions
    observation_space: int = 45

    state_space: int = 0

    # -------- 仿真配置 --------
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=decimation,
        device="cuda:0",
        physx=PhysxCfg(
            enable_ccd=True,
            solver_type=1,
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
    )

    # -------- 机器人配置 --------
    # robot: ArticulationCfg = ArticulationCfg(
    #     prim_path="/World/envs/env_.*/Robot",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path="/home/lee/EE_ws/src/sixfeet_src/sixfeet/source/sixfeet/sixfeet/assets/hexapod/hexapod.usd", # !! 确保路径正确 !!
    #         activate_contact_sensors=True,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=None, max_depenetration_velocity=10.0, enable_gyroscopic_forces=True
    #         ),
    #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #             enabled_self_collisions=True, solver_position_iteration_count=12,
    #             solver_velocity_iteration_count=1, sleep_threshold=0.005, stabilization_threshold=0.001,
    #         ),
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, 0.3),
    #         # rot=(1.0, 0.0, 0.0, 0.0),
    #         joint_pos={
    #             "joint_11": 0.0, "joint_21": 0.0, "joint_31": 0.0, "joint_41": 0.0, "joint_51": 0.0, "joint_61": 0.0,
    #             "joint_12": math.radians(45), "joint_22": math.radians(45), "joint_32": math.radians(45),
    #             "joint_42": math.radians(45), "joint_52": math.radians(45), "joint_62": math.radians(45),
    #             "joint_13": math.radians(-90), "joint_23": math.radians(-90), "joint_33": math.radians(-90),
    #             "joint_43": math.radians(-90), "joint_53": math.radians(-90), "joint_63": math.radians(-90),
    #         }
    #     ),
    #     actuators={
    #         "all_joints": ImplicitActuatorCfg(
    #             joint_names_expr=".*", stiffness=60.0,damping=4.0,
    #             effort_limit_sim=6.0, velocity_limit_sim=8.0
    #         )
    #     }
    # )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/lee/EE_ws/src/sixfeet_src/sixfeet/source/sixfeet/sixfeet/assets/hexapod/hexapod.usd",
            # usd_path="/home/lee/EE_ws/src/robot_urdf/urdf/hexapod_2/hexapod_2.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,  # <<<--- 修复：可能导致不稳定
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.15),
            rot=(1.0, 0.0, 0.0, 0.0),  # <<<--- 修复：明确设置四元数
            joint_pos={
                "joint_11": 0.0, "joint_21": 0.0, "joint_31": 0.0, "joint_41": 0.0, "joint_51": 0.0, "joint_61": 0.0,
                "joint_12": math.radians(45), "joint_22": math.radians(45), 
                "joint_32": math.radians(45), "joint_42": math.radians(45), 
                "joint_52": math.radians(45), "joint_62": math.radians(45),
                "joint_13": math.radians(-90), "joint_23": math.radians(-90), 
                "joint_33": math.radians(-90), "joint_43": math.radians(-90), 
                "joint_53": math.radians(-90), "joint_63": math.radians(-90),
            },
            joint_vel={
                # <<<--- 新增：显式设置关节初始速度为0
                "joint_11": 0.0, "joint_21": 0.0, "joint_31": 0.0,
                "joint_41": 0.0, "joint_51": 0.0, "joint_61": 0.0,
                "joint_12": 0.0, "joint_22": 0.0, "joint_32": 0.0,
                "joint_42": 0.0, "joint_52": 0.0, "joint_62": 0.0,
                "joint_13": 0.0, "joint_23": 0.0, "joint_33": 0.0,
                "joint_43": 0.0, "joint_53": 0.0, "joint_63": 0.0,
            }
        ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=".*", 
            stiffness=30.0,  # <<<--- 修复：增加刚度，提高稳定性
            damping=5.0,     # <<<--- 修复：增加阻尼，减少振荡
            effort_limit_sim=8.0,    # <<<--- 修复：增加扭矩限制
            velocity_limit_sim=10.0  # <<<--- 修复：增加速度限制
        )
    }
)
    # -------- 传感器配置 --------
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3,update_period=0.005,
        track_air_time=True
    )

    # --- 关节和连杆名称表达式 ---
    toe_joint_names_expr: str | list[int] | None = "joint_.[3]"
    undesired_contact_link_names_expr: str = "(thigh_link_[1-6]1|shin_link_[1-6]2)"
    base_link_name: str = "base_link" # 用于检测基座触地

    # -------- 场景配置 --------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # -------- 离散指令配置 (Discrete Command Profile) --------
    command_profile: dict = {
        # 当机器人被指令移动时，奖励函数会参考这个速度，但不再强制精确匹配
        # 可以理解为“期望的平均移动速度”或“最大期望速度”
        "reference_linear_speed": 0.3,  # m/s
        "reference_angular_rate": 0.4,  # rad/s
        # 指令模式: (前后, 左右, 转向)。值为 -1, 0, 或 1。
        # [1,0,0] 前进; [0,-1,0] 左移; [0,0,1] 右转
        "command_mode_duration_s": 3.0, # 训练时一个指令模式持续的秒数
        "stand_still_prob": 0.15, # 训练时指令为“站立不动”的概率
        "num_command_modes": 7 # (站立, 前, 后, 左, 右, 左转, 右转)
    }

    # -------- 奖励缩放因子 --------
    action_scale: float = 0.5

    # --- 主要的正向激励 ---
    rew_scale_move_in_commanded_direction: float = +2.5 # 奖励在指令方向上的移动
    rew_scale_achieve_reference_angular_rate: float = +0.5 # 奖励达到参考角速度 (如果指令转向)
    # 如果不追求特定线速度，可以移除或大幅减小此项，或者将其变为一个“不要太快”的惩罚
    # rew_scale_achieve_reference_linear_speed: float = +0.5

    rew_scale_alive: float = +0.05
    rew_scale_target_height: float = +5.0
    target_height_m: float = 0.2

    # --- 行为平滑与效率相关的惩罚 ---
    rew_scale_action_cost: float = -0.0002
    rew_scale_action_rate: float = -0.02
    rew_scale_joint_torques: float = -2.0e-5
    rew_scale_joint_accel: float = -2.0e-7

    # --- 姿态与运动稳定性相关的惩罚 ---
    rew_scale_lin_vel_z_penalty: float = -2.0   # 惩罚Z轴线速度 (不希望上下跳动)
    rew_scale_ang_vel_xy_penalty: float = -0.05 # 惩罚XY轴角速度 (不希望翻滚/侧倾)
    rew_scale_flat_orientation: float = -5.0    # 惩罚身体不水平 (基于projected_gravity_b)
    rew_scale_unwanted_movement_penalty: float = -2.0 # 当指令为静止时，惩罚移动

    # --- 行为约束相关的惩罚 ---
    rew_scale_dof_at_limit: float = -0.2
    rew_scale_toe_orientation_penalty: float = -1.5
    rew_scale_low_height_penalty: float = -25.0
    min_height_penalty_threshold: float = 0.15

    # --- 不期望的接触惩罚 ---
    rew_scale_undesired_contact: float = -2.0   # 惩罚大腿/小腿触地

    # --- 终止状态相关的惩罚 ---
    rew_scale_termination: float = -15.0

    # -------- 重置随机化 --------
    root_orientation_yaw_range: float = math.pi
    reset_height_offset: float = 0.2

    # -------- 终止条件 --------
    termination_body_z_thresh: float = 0.98 # 基于projected_gravity_z
    termination_height_thresh: float = 0.05
    termination_base_contact: bool = True