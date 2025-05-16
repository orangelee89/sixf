from __future__ import annotations

import math
import torch
# 初始的CUDA设备检查打印
# print(torch.cuda.is_available())
# print("CUDA Device Count:", torch.cuda.device_count())
# print("Current CUDA Device:", torch.cuda.current_device())
# print("Device Name:", torch.cuda.get_device_name(0))

from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
# from isaaclab.utils.math import sample_uniform # 未在您的代码中使用

from .sixfeet_env_cfg import SixfeetEnvCfg # 假设配置文件在同一目录下
from isaaclab.utils.math import quat_rotate # 确保这是您Isaac Lab版本中正确的导入路径和函数

@torch.jit.script
def reward_upright(root_quat_xyzw: torch.Tensor) -> torch.Tensor:
    """
    计算基于机器人Z轴与世界Z轴对齐程度的奖励。
    输入:  root_quat_xyzw (B,4)  —— 机体四元数，顺序为 (x,y,z,w)。
    输出:  (B,)                  —— 奖励值, exp(-(1-cos_theta)/0.1)，theta是机体+Z与世界+Z的夹角。
    """
    B: int = root_quat_xyzw.shape[0]

    # 世界坐标系的 +Z 轴向量
    world_z_axis = torch.zeros((B, 3), device=root_quat_xyzw.device, dtype=root_quat_xyzw.dtype)
    world_z_axis[:, 2] = 1.0

    # 机体局部坐标系的 +Z 轴向量
    body_z_in_body_frame = torch.zeros((B, 3), device=root_quat_xyzw.device, dtype=root_quat_xyzw.dtype)
    body_z_in_body_frame[:, 2] = 1.0
    
    # 将机体的局部+Z轴旋转到世界坐标系下
    body_z_in_world_frame = quat_rotate(root_quat_xyzw, body_z_in_body_frame)

    # 计算机体Z轴（在世界系下）与世界Z轴的点积，即夹角的余弦值
    cos_theta = body_z_in_world_frame[..., 2].clamp(-1.0, 1.0)

    return torch.exp(-(1.0 - cos_theta) / 0.1)

@torch.jit.script
def penalty_ang_vel(root_ang_vel: torch.Tensor) -> torch.Tensor:
    """计算基于角速度大小的惩罚项。"""
    return torch.linalg.norm(root_ang_vel, dim=-1)

@torch.jit.script
def penalty_torque(joint_tau: torch.Tensor) -> torch.Tensor:
    """计算基于关节力矩平方和的惩罚项。"""
    return (joint_tau ** 2).sum(-1)


class SixfeetEnv(DirectRLEnv):
    cfg: SixfeetEnvCfg

    def __init__(self, cfg: SixfeetEnvCfg, render_mode: str | None = None, **kwargs):
        # 调用父类初始化，这会间接调用 _setup_scene
        super().__init__(cfg, render_mode, **kwargs)

        # 初始化可能在 _setup_scene 或 _post_reset 中才被赋值的变量
        self.joint_lower_limits: torch.Tensor | None = None
        self.joint_upper_limits: torch.Tensor | None = None
        self.all_joint_ids: torch.Tensor | None = None # 如果需要所有关节的索引
        self.target_joint_positions: torch.Tensor | None = None # 目标站立关节角度
        self.joint_pos_target: torch.Tensor | None = None # PD控制器的目标关节角度

        # 调试信息：检查碰撞报告是否在配置中启用
        contact_reporting_enabled = False
        if hasattr(self.cfg, 'robot_cfg') and hasattr(self.cfg.robot_cfg, 'collision_props') and self.cfg.robot_cfg.collision_props.report_contacts:
            contact_reporting_enabled = True
        elif hasattr(self.cfg, 'robot') and hasattr(self.cfg.robot, 'collision_props') and self.cfg.robot.collision_props.report_contacts: # Isaac Lab 0.6+ style
            contact_reporting_enabled = True
        
        if contact_reporting_enabled:
            print("[INFO] SixfeetEnv: Contact reporting is ENABLED in robot configuration.")
        else:
            print("[WARNING] SixfeetEnv: Contact reporting might NOT be enabled in robot configuration. Collision penalty may not work as expected.")


    def _setup_scene(self):
        # 机器人实例化
        # 根据配置中是 robot_cfg 还是 robot 来获取 ArticulationCfg
        if hasattr(self.cfg, 'robot_cfg'): # 兼容您之前的cfg命名
            self.robot = Articulation(self.cfg.robot_cfg)
        elif hasattr(self.cfg, 'robot'): # Isaac Lab 0.6+ 风格
            self.robot = Articulation(self.cfg.robot)
        else:
            raise AttributeError("SixfeetEnvCfg is missing 'robot_cfg' or 'robot' attribute of type ArticulationCfg.")
        
        # 将机器人添加到场景的关节对象字典中，这对于Isaac Lab环境很重要
        self.scene.articulations["robot"] = self.robot

        # 生成地面
        spawn_ground_plane("/World/ground", GroundPlaneCfg())

        # 添加光源
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 注意: self.robot.data 中的数据（如joint_pos_limits）在物理引擎至少运行一步或重置后才完全可靠。
        # _post_reset 是一个更安全的地方来初始化这些依赖于运行时数据的变量。
        # 但通常在 Articulation 对象创建后，其静态属性（如关节数量、名称）和配置的限制应该是可读的。


    def _pre_physics_step(self, actions: torch.Tensor):
        # 克隆动作张量并确保它在正确的设备上
        self.actions = actions.clone().to(self.device)
        # print(f"Actions shape: {self.actions.shape}") # 用于调试

    def _apply_action(self):
        # 确保关节限制已初始化 (通常在 _post_reset 中完成)
        if self.joint_lower_limits is None or self.joint_upper_limits is None:
            # 这是一个后备，理想情况下不应该在这里初始化
            if self.robot.is_initialized and self.robot.num_articulated_joints > 0:
                print("[Warning] SixfeetEnv._apply_action: Lazily initializing joint limits.")
                limits = self.robot.data.joint_pos_limits.to(self.device)
                if limits.ndim == 3 and limits.shape[0] == 1: # 处理可能的额外env维度
                    limits = limits.squeeze(0)
                self.joint_lower_limits = limits[:, 0]
                self.joint_upper_limits = limits[:, 1]
            else:
                raise RuntimeError("Joint limits not initialized in SixfeetEnv, and robot not ready.")

        # 将动作值从 [-1, 1] 缩放到每个关节的 [lower_limit, upper_limit] 范围
        action_midpoint = (self.joint_upper_limits + self.joint_lower_limits) / 2.0
        action_half_range = (self.joint_upper_limits - self.joint_lower_limits) / 2.0
        
        # 根据您的配置，action_scale 可能会进一步调整这个范围或作为偏移的乘数
        # 您之前的代码是: self.target_joint_positions + actions * self.cfg.action_scale
        # 这表示 actions 是相对于 target_joint_positions 的偏移。
        # 如果 self.actions 直接代表目标位置（在[-1,1]标准化后），则使用下面的映射：
        scaled_actions = self.actions * action_half_range + action_midpoint
        
        # 如果 self.actions 是相对于目标站立姿态的偏移:
        # if self.target_joint_positions is None: # 确保目标站立姿态已初始化
        #     self._init_target_joint_positions()
        # scaled_actions = self.target_joint_positions + self.actions * self.cfg.action_scale
        # scaled_actions = torch.clamp(scaled_actions, self.joint_lower_limits, self.joint_upper_limits) # 裁剪

        self.robot.set_joint_position_target(scaled_actions)

    def _get_observations(self) -> dict:
        # 直接从 self.robot.data 获取最新的状态数据
        root_quat_obs = self.robot.data.root_quat_w.to(self.device)         # (x,y,z,w)
        root_ang_vel_obs = self.robot.data.root_ang_vel_w.to(self.device)
        joint_pos_obs = self.robot.data.joint_pos.to(self.device)
        joint_vel_obs = self.robot.data.joint_vel.to(self.device)
        
        obs = torch.cat(
            (
                root_quat_obs,       # 4
                root_ang_vel_obs,    # 3
                joint_pos_obs,       # num_actions (例如18)
                joint_vel_obs,       # num_actions (例如18)
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # 获取当前状态用于奖励计算
        current_root_quat = self.robot.data.root_quat_w.to(self.device) # (x,y,z,w)
        current_root_ang_vel = self.robot.data.root_ang_vel_w.to(self.device)
        current_joint_tau = self.robot.data.applied_torque.to(self.device) # 注意: applied_torque 可能不总是理想的惩罚项

        r_up = reward_upright(current_root_quat)
        p_av = penalty_ang_vel(current_root_ang_vel)
        p_tau = penalty_torque(current_joint_tau) # 或者 penalty_torque(self.actions) 如果想惩罚动作幅度

        # --- 碰撞惩罚计算 ---
        computed_collision_penalty_value = torch.zeros_like(r_up) # 默认惩罚为0

        if hasattr(self.robot.data, "net_contact_forces") and self.robot.data.net_contact_forces is not None:
            contact_forces = self.robot.data.net_contact_forces.to(self.device) # 形状: (num_envs, num_links, 3)
            
            # 从配置中获取身体连杆索引和力分量阈值
            # 您需要在 SixfeetEnvCfg.py 中定义这些！
            body_link_indices = getattr(self.cfg, "body_link_indices_for_collision", [0]) # 示例：默认只检查连杆0
            contact_component_threshold = getattr(self.cfg, "contact_component_threshold", 1.0) # 示例：默认阈值1N

            valid_body_link_indices = [idx for idx in body_link_indices if idx < contact_forces.shape[1]]
            if valid_body_link_indices:
                # 获取指定身体连杆的接触力
                selected_body_forces = contact_forces[:, valid_body_link_indices, :] # 形状: (num_envs, num_valid_body_links, 3)
                
                # 检查是否有任何力分量的绝对值超过阈值
                component_exceeds_threshold = torch.abs(selected_body_forces) > contact_component_threshold # 形状: (num_envs, num_valid_body_links, 3)
                
                # 如果该连杆的x,y,z任何一个力分量超阈值，则为True
                link_has_strong_force_component = torch.any(component_exceeds_threshold, dim=2) # 形状: (num_envs, num_valid_body_links)
                
                # 如果任何一个指定的身体连杆有显著的力分量，则认为发生碰撞
                is_body_colliding = torch.any(link_has_strong_force_component, dim=1).float() # 形状: (num_envs,)
                
                computed_collision_penalty_value = is_body_colliding # 惩罚值为1.0（如果碰撞），0.0（如果不碰撞）
            else:
                # 仅在调试时或偶尔打印警告，避免刷屏
                if self.episode_count % 100 == 0 and self.num_envs > 0 : 
                    print(f"[Warning] SixfeetEnv._get_rewards: No valid body_link_indices ({body_link_indices}) for collision check. Max link index: {contact_forces.shape[1]-1 if contact_forces.shape[1]>0 else -1}")
        else:
            if self.episode_count % 100 == 0 and self.num_envs > 0:
                 print("[Warning] SixfeetEnv._get_rewards: 'net_contact_forces' not found or is None in self.robot.data. Collision penalty will be zero.")
        
        # self.cfg.rew_scale_collision 应该是一个负数 (例如 -10.0)
        total_reward = (
            self.cfg.rew_scale_upright * r_up
            + self.cfg.rew_scale_angvel * (-p_av)  # p_av 是正的范数，所以乘以负权重
            + self.cfg.rew_scale_torque * (-p_tau)  # p_tau 是正的平方和，所以乘以负权重
            + self.cfg.rew_scale_collision * computed_collision_penalty_value # computed_collision_penalty_value 是正的 "坏的程度"
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # 当前：只有超时会重置环境。
        # 如果需要因为跌倒或严重碰撞而提前终止，可以在这里添加逻辑。
        # 例如:
        # current_root_quat = self.robot.data.root_quat_w.to(self.device)
        # r_up_val = reward_upright(current_root_quat)
        # fallen_threshold = getattr(self.cfg, "fallen_threshold", 0.2) # 在cfg中定义
        # is_fallen = r_up_val < fallen_threshold
        # terminated = torch.logical_or(time_out, is_fallen) # 如果想因为跌倒而终止

        terminated = torch.zeros_like(time_out) # 总是False，所以只有time_out会触发RL框架的重置
        return terminated, time_out # 在RL中，通常 terminated=True 也会导致环境重置

    def _reset_idx(self, env_ids: Sequence[int] | None): 
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, Sequence) and not isinstance(env_ids, torch.Tensor):
            env_ids_tensor = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = env_ids # 假设已经是tensor

        super()._reset_idx(env_ids_tensor) # 处理父类的重置逻辑，如 episode_length_buf

        num_resets = len(env_ids_tensor)

        # 随机化根节点姿态
        # cfg.root_orientation_range 应为包含3个浮点数的列表/元组，对应R,P,Y的范围
        # 如果是单个浮点数，则应用到所有轴
        if isinstance(self.cfg.root_orientation_range, float):
            orientation_ranges = torch.full((3,), self.cfg.root_orientation_range, device=self.device, dtype=torch.float32)
        elif isinstance(self.cfg.root_orientation_range, (list, tuple)) and len(self.cfg.root_orientation_range) == 3:
            orientation_ranges = torch.tensor(self.cfg.root_orientation_range, device=self.device, dtype=torch.float32)
        else:
            raise ValueError("cfg.root_orientation_range must be a float or a list/tuple of 3 floats.")

        rpy = (torch.rand((num_resets, 3), device=self.device, dtype=torch.float32) - 0.5) * 2.0 * orientation_ranges.unsqueeze(0)

        # 欧拉角到四元数 (RPY -> w,x,y,z)
        cr, sr = torch.cos(rpy[:, 0] / 2.0), torch.sin(rpy[:, 0] / 2.0) # Roll
        cp, sp = torch.cos(rpy[:, 1] / 2.0), torch.sin(rpy[:, 1] / 2.0) # Pitch
        cy, sy = torch.cos(rpy[:, 2] / 2.0), torch.sin(rpy[:, 2] / 2.0) # Yaw

        qw = cy * cp * cr + sy * sp * sr
        qx = cy * cp * sr - sy * sp * cr
        qy = sy * cp * sr + cy * sp * cr
        qz = sy * cp * cr - cy * sp * sr
        
        # 转换为 Isaac Lab 通常使用的 (x,y,z,w) 顺序
        quat_xyzw = torch.stack([qx, qy, qz, qw], dim=-1)

        # 获取对应环境的默认根状态并修改
        default_root_state_selected = self.robot.data.default_root_state[env_ids_tensor].clone() # 使用 .clone() 避免修改原始默认状态
        
        # 设置新的随机姿态
        default_root_state_selected[:, 3:7] = quat_xyzw
        
        # 调整Z轴高度以防穿模
        reset_height_offset = getattr(self.cfg, "reset_height_offset", 0.1) # 从cfg获取，如果不存在则默认为0.1
        default_root_state_selected[:, 2] += reset_height_offset
        
        # 将新的根状态写入仿真
        self.robot.write_root_state_to_sim(default_root_state_selected, env_ids=env_ids_tensor)

        # 重置关节状态 (例如，到目标站立姿态并加上一些噪声)
        # 确保 self.target_joint_positions 已经初始化 (通常在 _post_reset 中完成)
        if self.target_joint_positions is None:
            self._init_target_joint_positions() # 辅助函数来初始化

        target_joints_for_reset = self.target_joint_positions[env_ids_tensor]
        reset_joint_pos_noise_val = getattr(self.cfg, "reset_joint_pos_noise", 0.1) # 从cfg获取，默认为0.1
        joint_pos_noise = (torch.rand_like(target_joints_for_reset) - 0.5) * 2.0 * reset_joint_pos_noise_val
        reset_joint_pos = target_joints_for_reset + joint_pos_noise
        
        # 裁剪到关节限制 (确保 self.joint_lower_limits 和 self.joint_upper_limits 已初始化)
        if self.joint_lower_limits is not None and self.joint_upper_limits is not None:
             reset_joint_pos = torch.clamp(reset_joint_pos, self.joint_lower_limits.unsqueeze(0).repeat(num_resets,1), self.joint_upper_limits.unsqueeze(0).repeat(num_resets,1))
        else: 
            # 这个警告不应该经常出现，如果_post_reset正确工作
            print("[Warning] SixfeetEnv._reset_idx: Joint limits not available for clamping reset positions. Attempting to initialize them now.")
            self._init_joint_limits() # 尝试初始化
            if self.joint_lower_limits is not None and self.joint_upper_limits is not None:
                 reset_joint_pos = torch.clamp(reset_joint_pos, self.joint_lower_limits.unsqueeze(0).repeat(num_resets,1), self.joint_upper_limits.unsqueeze(0).repeat(num_resets,1))


        self.robot.write_joint_state_to_sim(
            positions=reset_joint_pos, 
            velocities=self.robot.data.default_joint_vel[env_ids_tensor], # 通常重置为0速度
            # torques=None, # 通常在重置时不设置力矩
            joint_ids=None, # None 表示所有关节
            env_ids=env_ids_tensor,
        )
        
        # 初始化下一拍的PD控制器目标关节位置
        if self.joint_pos_target is not None: # 确保 joint_pos_target 已初始化
            self.joint_pos_target[env_ids_tensor] = reset_joint_pos[:]


    def _init_target_joint_positions(self):
        """辅助函数，用于初始化目标站立关节角度。"""
        if hasattr(self.cfg, 'target_standing_joint_angles'):
            self.target_joint_positions = torch.tensor(
                self.cfg.target_standing_joint_angles, device=self.device, dtype=torch.float32
            ).repeat(self.num_envs, 1) # 扩展到所有环境
        else:
            # 如果配置中没有定义，则回退到机器人的默认关节位置
            if self.robot.is_initialized and self.robot.num_articulated_joints > 0:
                self.target_joint_positions = self.robot.data.default_joint_pos.clone().to(self.device)
                print("[Warning] SixfeetEnv._init_target_joint_positions: 'target_standing_joint_angles' not found in cfg. Using default_joint_pos as target.")
            else:
                # 这种情况理论上不应发生，如果发生在 _post_reset 之后
                raise RuntimeError("Cannot initialize target_joint_positions: Robot not ready or has no joints.")

    def _init_joint_limits(self):
        """辅助函数，用于初始化关节限制。"""
        if self.robot.is_initialized and self.robot.num_articulated_joints > 0:
            limits = self.robot.data.joint_pos_limits.to(self.device)
            if limits.ndim == 3 and limits.shape[0] == 1: # 处理可能的额外env维度
                limits = limits.squeeze(0)
            self.joint_lower_limits = limits[:, 0]
            self.joint_upper_limits = limits[:, 1]
            if self.all_joint_ids is None: # 如果需要，也在这里初始化
                 self.all_joint_ids = torch.arange(self.robot.num_articulated_joints, device=self.device)
            print("[INFO] SixfeetEnv._init_joint_limits: Joint limits initialized.")
        else:
            print("[Error] SixfeetEnv._init_joint_limits: Cannot initialize joint limits, robot not ready or no joints.")


    def _post_reset(self, env_ids: torch.Tensor | None) -> None:
        """在环境重置后调用，适合初始化依赖于运行时数据的变量。"""
        super()._post_reset(env_ids) # 调用父类方法
        
        # 初始化目标站立关节角度 (如果尚未初始化)
        if self.target_joint_positions is None:
            self._init_target_joint_positions()

        # 初始化关节限制 (如果尚未初始化)
        if self.joint_lower_limits is None:
            self._init_joint_limits()

        # 初始化PD控制器的目标关节角度 (如果尚未初始化)
        if self.joint_pos_target is None:
            if self.target_joint_positions is not None:
                self.joint_pos_target = self.target_joint_positions.clone()
            elif self.robot.is_initialized and self.robot.num_articulated_joints > 0:
                # 作为最后的后备，使用机器人的默认关节位置
                self.joint_pos_target = self.robot.data.default_joint_pos.clone().to(self.device)
                print("[Warning] SixfeetEnv._post_reset: joint_pos_target initialized to default_joint_pos as target_joint_positions was None.")
            else:
                # 如果机器人也未准备好，这会是一个问题
                print("[Error] SixfeetEnv._post_reset: Cannot initialize joint_pos_target, robot not ready or target_joint_positions is None.")
