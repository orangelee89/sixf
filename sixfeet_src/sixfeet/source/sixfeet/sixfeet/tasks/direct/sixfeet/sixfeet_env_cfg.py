# sixfeet_env_cfg.py (Focus on Standing Up)
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
    episode_length_s: float = 20 # 可以适当缩短，如果只学站立
    action_space: int = 18

    # ---- 观测空间定义 ----
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
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply"
        )
    )

    # -------- 地形配置 --------
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
    )

    # -------- 机器人配置 --------
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/lee/EE_ws/src/sixfeet_src/sixfeet/source/sixfeet/sixfeet/assets/hexapod/hexapod.usd", # 请确保路径正确!
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=False, # 通常设为False，除非特殊需求
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=12, # 可以适当增加迭代次数提高稳定性
                solver_velocity_iteration_count=1,  # 可以适当增加迭代次数提高稳定性
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0001), # 初始高度较低，便于学习站起
            rot=(1.0, 0.0, 0.0, 0.0), # 标准初始姿态 (w,x,y,z)
            # joint_pos 已被注释掉，将在 env.py 的 _reset_idx 中实现完全随机关节初始姿态
            joint_vel={ # 初始关节速度为0
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
                stiffness=20.0, # 驱动器P项
                damping=12.0,   # 驱动器D项
                effort_limit_sim=5.0, # 关节力矩限制
                velocity_limit_sim=5.0 # 关节速度限制
            )
        }
    )

    # -------- 传感器配置 --------
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3,update_period=0.005,
        track_air_time=True
    )

    # --- 关节和连杆名称表达式 ---
    toe_joint_names_expr: str | list[int] | None = "joint_.[3]" # 用于足端方向惩罚
    undesired_contact_link_names_expr: str = "(thigh_link_[1-6]1|shin_link_[1-6]2)" # 非足端接触惩罚
    base_link_name: str = "base_link" # 用于检测基座触地

    # -------- 场景配置 --------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # -------- 离散指令配置 (Discrete Command Profile) --------
    command_profile: dict = {
        "reference_linear_speed": 0.0,
        "reference_angular_rate": 0.0,
        "command_mode_duration_s": episode_length_s,
        "stand_still_prob": 1.0,       # 专注于站立
        "num_command_modes": 7
    }
     # ---初始姿态随机化模式 ---
    # 模式 1: 完全随机的翻滚(Roll), 俯仰(Pitch), 偏航(Yaw)角度, 包括180度翻转
    # 模式 0: 特殊初始姿态 - 机器人Z轴与世界Z轴相反 (完全倒置)，Yaw角随机
    initial_pose_randomization_mode: int = 0
    randomize_initial_joint_poses: bool = False


    # -------- 奖励缩放因子 --------
    action_scale: float = 0.5
    rew_scale_move_in_commanded_direction: float = 0.0
    rew_scale_achieve_reference_angular_rate: float = 0.0
    rew_scale_alive: float = +0.1
    rew_scale_target_height: float = +20.0 # 将在 env.py 中条件化
    target_height_m: float = 0.15        # 你的 "Focus on Standing Up" cfg 中 target_height_m 为 0.20，这里用了你新cfg中的0.23
    target_height_reward_sharpness: float = 100.0 #target_height_reward_sharpness: float = 100.0

    rew_scale_orientation_deviation: float = -50.0
     # --- 新增：自碰撞惩罚 ---
    rew_scale_self_collision: float = -30.0  # 一个较大的负值，当发生自碰撞时施加
    rew_scale_successful_flip: float = 40.0 # 当从Z轴朝下翻转到Z轴朝上时给予
    rew_scale_action_cost: float = -0.0001
    rew_scale_action_rate: float = -0.01
    rew_scale_joint_torques: float = -1.0e-6
    rew_scale_joint_accel: float = -1.0e-7
    rew_scale_lin_vel_z_penalty: float = -3.0
    rew_scale_ang_vel_xy_penalty: float = -0.1
    rew_scale_flat_orientation: float = -25.0
    rew_scale_unwanted_movement_penalty: float = -5.0
    rew_scale_dof_at_limit: float = -0.5
    rew_scale_toe_orientation_penalty: float = -4.0 # 在 env.py 中条件化生效
    rew_scale_low_height_penalty: float = -30.0
    min_height_penalty_threshold: float = 0.12 # 你的 "Focus on Standing Up" cfg 中是 0.15

    rew_scale_undesired_contact: float = -5.0   # 在 env.py 中条件化生效
    rew_scale_termination: float = -20.0

    # --- 新增：可配置的关节极限惩罚阈值 (百分比) ---
    joint_limit_penalty_threshold_percent: float = 0.05  # 例如 0.05 代表边缘5%

    # -------- 重置随机化 --------
    root_orientation_yaw_range: float = math.pi
    reset_height_offset: float = 0.0

    # -------- 终止条件 --------
    termination_body_z_thresh: float = 0.95
    termination_height_thresh: float = 0.00
    termination_base_contact: bool = False # 已设为 True
    
    # --- 用于条件化终止的姿态角度限制 (度) ---
    orientation_termination_angle_limit_deg: float = 95.0
     # 值越小，表示要求机器人越平坦。例如 0.1*0.1 = 0.01 (约5.7度倾斜)，0.3*0.3 = 0.09 (约17度倾斜)
    flatness_threshold_for_height_termination: float = 0.02 # 可调整