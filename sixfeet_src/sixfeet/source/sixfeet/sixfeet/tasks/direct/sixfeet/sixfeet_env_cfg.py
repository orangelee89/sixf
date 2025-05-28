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
    episode_length_s: float = 20.0 # 可以适当缩短，如果只学站立
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
            pos=(0.0, 0.0, 0.15), # 初始高度较低，便于学习站起
            rot=(1.0, 0.0, 0.0, 0.0), # 标准初始姿态 (w,x,y,z)
            # joint_pos={ # 初始关节角度，可以是一个趴下的姿态
            #     "joint_11": 0.0, "joint_21": 0.0, "joint_31": 0.0, "joint_41": 0.0, "joint_51": 0.0, "joint_61": 0.0,
            #     "joint_12": math.radians(45), "joint_22": math.radians(45),
            #     "joint_32": math.radians(45), "joint_42": math.radians(45),
            #     "joint_52": math.radians(45), "joint_62": math.radians(45),
            #     "joint_13": math.radians(-90), "joint_23": math.radians(-90),
            #     "joint_33": math.radians(-90), "joint_43": math.radians(-90),
            #     "joint_53": math.radians(-90), "joint_63": math.radians(-90),
            # },
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
    # 为了只学习站立，我们将让机器人始终接收“站立”指令
    command_profile: dict = {
        "reference_linear_speed": 0.0,  # m/s (站立时参考速度为0)
        "reference_angular_rate": 0.0,  # rad/s (站立时参考角速度为0)
        "command_mode_duration_s": episode_length_s, # 指令持续整个episode
        "stand_still_prob": 1.0,       # !!! 始终发出站立指令 !!!
        "num_command_modes": 7 # (虽然概率为1，但结构保留)
    }

    # -------- 奖励缩放因子 (专注站立) --------
    action_scale: float = 0.5

    # --- 主要的正向激励 (站立) ---
    rew_scale_move_in_commanded_direction: float = 0.0  # !!! 禁用移动奖励 !!!
    rew_scale_achieve_reference_angular_rate: float = 0.0 # !!! 禁用转向奖励 !!!

    rew_scale_alive: float = +0.1 # 略微增加存活奖励，鼓励持续站立
    rew_scale_target_height: float = +8.0 # !!! 增加目标高度的奖励权重 !!!
    target_height_m: float = 0.20 # 目标站立高度 (米)

    # --- 行为平滑与效率相关的惩罚 (可设为0或保留较小值) ---
    rew_scale_action_cost: float = -0.0001 # 可以保留一个非常小的动作成本
    rew_scale_action_rate: float = -0.01   # 可以保留一个小的动作变化率惩罚，使动作更平滑
    rew_scale_joint_torques: float = -1.0e-6 # 极小的力矩惩罚
    rew_scale_joint_accel: float = -1.0e-7   # 极小的加速度惩罚

    # --- 姿态与运动稳定性相关的惩罚 (对站立很重要) ---
    rew_scale_lin_vel_z_penalty: float = -3.0   # !!! 略微增加Z轴速度惩罚，避免上下晃动 !!!
    rew_scale_ang_vel_xy_penalty: float = -0.1 # !!! 略微增加XY轴角速度惩罚，避免摇晃和摔倒 !!!
    rew_scale_flat_orientation: float = -25.0    # !!! 增加身体水平的惩罚权重 !!!
    rew_scale_unwanted_movement_penalty: float = -5.0 # !!! 增加在站立指令下移动的惩罚权重 !!!

    # --- 行为约束相关的惩罚 (对站立很重要) ---
    rew_scale_dof_at_limit: float = -0.5 # 略微增加关节到达极限的惩罚
    rew_scale_toe_orientation_penalty: float = -4.0 # 增加足端不良姿态的惩罚
    rew_scale_low_height_penalty: float = -30.0 # !!! 大幅增加低高度惩罚 !!!
    min_height_penalty_threshold: float = 0.15 # 低于此高度则开始惩罚 (米)

    # --- 不期望的接触惩罚 (对站立很重要) ---
    rew_scale_undesired_contact: float = -5.0   # !!! 增加大腿/小腿触地的惩罚权重 !!!

    # --- 终止状态相关的惩罚 ---
    rew_scale_termination: float = -20.0 # 失败终止的惩罚 (如摔倒)

    # -------- 重置随机化 --------
    root_orientation_yaw_range: float = math.pi # 重置时初始Yaw角随机范围
    reset_height_offset: float = 0.0 # 从较低的初始高度开始，不需要额外偏移

    # -------- 终止条件 --------
    termination_body_z_thresh: float = 0.95 # 身体倾斜到一定程度终止 (值越小越容易因为倾斜终止)
    termination_height_thresh: float = 0.05 # 高度过低终止
    termination_base_contact: bool = False   # 身体躯干接触地面时终止