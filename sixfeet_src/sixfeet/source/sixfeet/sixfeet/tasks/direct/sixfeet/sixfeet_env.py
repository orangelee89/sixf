# sixfeet_env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import (
    quat_rotate, quat_from_angle_axis, quat_conjugate,
    convert_quat, euler_xyz_from_quat
)
import torch.nn.functional as F

from .sixfeet_env_cfg import SixfeetEnvCfg

# ───────────────── 辅助函数 ──────────────────
@torch.jit.script
def normalize_angle_for_obs(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))

@torch.jit.script
def penalty_root_angular_velocity_l2(root_ang_vel_w: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(root_ang_vel_w, dim=-1)

@torch.jit.script
def penalty_joint_torques_sq(applied_torque: torch.Tensor) -> torch.Tensor:
    return (applied_torque ** 2).sum(-1)

@torch.jit.script
def penalty_joint_velocities_with_limit(
    joint_vel: torch.Tensor, soft_vel_limit: float
) -> torch.Tensor:
    vel_sq_penalty = torch.sum(joint_vel**2, dim=-1)
    exceeded_vel_sq_penalty = torch.sum(torch.relu(torch.abs(joint_vel) - soft_vel_limit)**2, dim=-1)
    return vel_sq_penalty + exceeded_vel_sq_penalty

@torch.jit.script
def compute_intermediate_robot_states(
    root_pos_w: torch.Tensor, root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor, root_ang_vel_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inv_root_quat_w = quat_conjugate(root_quat_w)
    root_lin_vel_b = quat_rotate(inv_root_quat_w, root_lin_vel_w)
    root_ang_vel_b = quat_rotate(inv_root_quat_w, root_ang_vel_w)

    root_quat_wxyz = torch.cat((root_quat_w[..., 3:4], root_quat_w[..., 0:3]), dim=-1)
    roll_b, pitch_b, yaw_b = euler_xyz_from_quat(root_quat_wxyz)

    body_z_axis_local = torch.zeros_like(root_pos_w)
    body_z_axis_local[..., 2] = 1.0
    body_z_axis_world = quat_rotate(root_quat_w, body_z_axis_local)
    body_z_proj_w = body_z_axis_world[..., 2]

    return root_lin_vel_b, root_ang_vel_b, roll_b, pitch_b, yaw_b, body_z_proj_w

@torch.jit.script
def compute_sixfeet_rewards(
    # --- 机器人状态 ---
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    body_z_proj_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    applied_torque: torch.Tensor,
    q_lower_limits: torch.Tensor,
    q_upper_limits: torch.Tensor,
    actions: torch.Tensor,

    # --- 奖励权重和参数 (从 cfg 传入) ---
    rew_scale_target_height: float,
    target_height_m: float,
    rew_scale_forward_progress: float,
    rew_scale_upright: float,
    rew_scale_alive: float,
    rew_scale_root_ang_vel: float,
    rew_scale_action_cost: float,
    rew_scale_torque: float,
    rew_scale_joint_vel: float,
    soft_joint_vel_limit: float,
    rew_scale_dof_at_limit: float,
    rew_scale_toe_orientation_penalty: float,
    rew_scale_low_height_penalty: float,       # <<--- 新增低高度惩罚权重
    min_height_penalty_threshold: float,    # <<--- 新增低高度惩罚阈值
    
    # --- 其他 ---
    toe_joint_indices: torch.Tensor | None
) -> torch.Tensor:
    
    # 1. 目标高度奖励
    current_height_z = root_pos_w[:, 2]
    height_ratio = current_height_z / target_height_m
    reward_target_height_raw = torch.clamp(height_ratio, max=1.1)
    reward_target_height_term = reward_target_height_raw * rew_scale_target_height

    # 2. 前进奖励
    local_forward_y_axis = torch.tensor([0.0, 1.0, 0.0], device=root_quat_w.device).expand_as(root_lin_vel_w)
    robot_forward_dir_w = quat_rotate(root_quat_w, local_forward_y_axis)
    robot_forward_dir_xy_w = F.normalize(robot_forward_dir_w[:, :2], p=2.0, dim=-1)
    current_lin_vel_xy_w = root_lin_vel_w[:, :2]
    forward_velocity_component = torch.sum(current_lin_vel_xy_w * robot_forward_dir_xy_w, dim=1)
    reward_forward_progress_term = forward_velocity_component * rew_scale_forward_progress

    # 3. 直立奖励
    reward_upright_term = body_z_proj_w.clamp(min=0.0) * rew_scale_upright

    # 4. 存活奖励
    reward_alive_term = torch.ones_like(forward_velocity_component) * rew_scale_alive

    # 5. 动作成本惩罚
    penalty_action_cost_term = torch.sum(actions**2, dim=-1) * rew_scale_action_cost

    # 6. 根部角速度/晃动惩罚
    penalty_root_ang_vel_term = penalty_root_angular_velocity_l2(root_ang_vel_w) * rew_scale_root_ang_vel

    # 7. 关节扭矩惩罚
    penalty_torque_term = penalty_joint_torques_sq(applied_torque) * rew_scale_torque

    # 8. 实际关节角速度惩罚
    penalty_joint_vel_term = penalty_joint_velocities_with_limit(joint_vel, soft_joint_vel_limit) * rew_scale_joint_vel
    
    # 9. 关节限位惩罚
    dof_range = q_upper_limits - q_lower_limits
    dof_range = torch.where(dof_range < 1e-6, torch.ones_like(dof_range), dof_range)
    dof_pos_scaled_01 = (joint_pos - q_lower_limits.unsqueeze(0)) / dof_range.unsqueeze(0)
    near_lower_limit = torch.relu(0.05 - dof_pos_scaled_01)**2
    near_upper_limit = torch.relu(dof_pos_scaled_01 - 0.95)**2
    penalty_dof_at_limit_term = torch.sum(near_lower_limit + near_upper_limit, dim=-1) * rew_scale_dof_at_limit

    # 10. 脚趾朝向惩罚
    penalty_toe_orientation_term = torch.zeros_like(forward_velocity_component)
    if rew_scale_toe_orientation_penalty != 0.0 and toe_joint_indices is not None: # 检查权重是否启用
        toe_joint_positions = joint_pos[:, toe_joint_indices]
        penalty_toe_orientation_term = torch.sum(torch.relu(toe_joint_positions)**2, dim=-1) * rew_scale_toe_orientation_penalty
    
    # --- 11. 新增：低高度惩罚 ---
    is_too_low = (current_height_z < min_height_penalty_threshold).float()
    penalty_low_height_term = is_too_low * rew_scale_low_height_penalty
    
    total_reward_terms = (
        reward_target_height_term
        + reward_forward_progress_term
        + reward_upright_term
        + reward_alive_term
        + penalty_action_cost_term
        + penalty_root_ang_vel_term
        + penalty_torque_term
        + penalty_joint_vel_term
        + penalty_dof_at_limit_term
        + penalty_toe_orientation_term
        + penalty_low_height_term # <<--- 加入新的惩罚项
    )
    return total_reward_terms

# -- SixfeetEnv Class --
class SixfeetEnv(DirectRLEnv):
    cfg: SixfeetEnvCfg
    terrain: TerrainImporter

    def __init__(self, cfg: SixfeetEnvCfg, render_mode: str | None = None, **kwargs):
        # ... (与上一版本相同的 __init__ 实现) ...
        super().__init__(cfg, render_mode, **kwargs)
        joint_limits = self.robot.data.joint_pos_limits[0].to(self.device)
        self._q_lower_limits = joint_limits[:, 0]
        self._q_upper_limits = joint_limits[:, 1]
        self._fixed_forward_command_local = torch.tensor([0.0, 1.0], device=self.device)
        self.constant_forward_command_obs = self._fixed_forward_command_local.unsqueeze(0).expand(self.num_envs, -1)
        self._resolve_toe_joint_indices()
        if self.cfg.rew_scale_action_rate < 0:
            self.previous_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.root_lin_vel_b: torch.Tensor
        self.root_ang_vel_b: torch.Tensor
        self.roll_b: torch.Tensor
        self.pitch_b: torch.Tensor
        self.yaw_b: torch.Tensor
        self.body_z_proj_w: torch.Tensor

    def _resolve_toe_joint_indices(self):
        # ... (与上一版本相同的 _resolve_toe_joint_indices 实现) ...
        self._toe_joint_indices: torch.Tensor | None = None
        expr_or_list = getattr(self.cfg, 'toe_joint_names_expr', None)
        if isinstance(expr_or_list, str):
            joint_indices_list, joint_names_list = self.robot.find_joints(expr_or_list)
            if not joint_indices_list: print(f"[WARNING] SixfeetEnv: No toe joints found using regex: '{expr_or_list}'.")
            else:
                self._toe_joint_indices = torch.tensor(joint_indices_list, device=self.device, dtype=torch.long)
                print(f"[INFO] SixfeetEnv: Found toe joint indices via regex: {self._toe_joint_indices.tolist()}, names: {joint_names_list}")
        elif isinstance(expr_or_list, list) and all(isinstance(i, int) for i in expr_or_list):
            if not expr_or_list: print(f"[WARNING] SixfeetEnv: 'toe_joint_names_expr' in cfg is an empty list.")
            else:
                temp_indices = torch.tensor(expr_or_list, device=self.device, dtype=torch.long)
                if torch.any(temp_indices < 0) or torch.any(temp_indices >= self.robot.num_articulated_joints):
                    print(f"[ERROR] SixfeetEnv: Invalid toe joint indices in list: {expr_or_list}.")
                else: self._toe_joint_indices = temp_indices; print(f"[INFO] SixfeetEnv: Using pre-defined toe joint indices: {self._toe_joint_indices.tolist()}")
        elif expr_or_list is not None: print(f"[WARNING] SixfeetEnv: 'toe_joint_names_expr' ('{expr_or_list}') in cfg is invalid type: {type(expr_or_list)}.")
        if self._toe_joint_indices is None:
            suffix = "Toe orientation penalty will not be applied."
            if expr_or_list is None: print(f"[INFO] SixfeetEnv: 'toe_joint_names_expr' not specified. {suffix}")
            else: print(f"[INFO] SixfeetEnv: Could not resolve toe joints from '{expr_or_list}'. {suffix}")

    def _setup_scene(self):
        # ... (与上一版本相同的 _setup_scene 实现，确保正确使用 TerrainImporter) ...
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        if hasattr(self.cfg, "terrain") and self.cfg.terrain is not None:
            if hasattr(self.scene, "cfg"): # Check if scene.cfg exists (it should for InteractiveScene)
                self.cfg.terrain.num_envs = self.scene.cfg.num_envs
                self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            
            terrain_class_path = getattr(self.cfg.terrain, "class_type", None)
            if isinstance(terrain_class_path, str):
                # Dynamically import the class
                # This part can be tricky with relative imports if class_type is not a fully qualified path.
                # Assuming class_type is like "module.submodule.ClassName"
                try:
                    module_path, class_name = terrain_class_path.rsplit('.', 1)
                    module = __import__(module_path, fromlist=[class_name])
                    terrain_class = getattr(module, class_name)
                except Exception as e:
                    print(f"[ERROR] Failed to import terrain class {terrain_class_path}: {e}")
                    from isaaclab.terrains import TerrainImporter
                    terrain_class = TerrainImporter # Fallback
            elif terrain_class_path is None: # Default if not specified
                from isaaclab.terrains import TerrainImporter
                terrain_class = TerrainImporter
            else: # If class_type is already a class object (e.g. directly assigned in Hydra)
                terrain_class = terrain_class_path

            self._terrain = terrain_class(self.cfg.terrain) # type: ignore
        else:
            print("[WARNING] SixfeetEnv: No terrain configuration. Spawning default plane.")
            from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane # Local import
            spawn_ground_plane("/World/ground", GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.scene.clone_environments(copy_from_source=False)


    def _pre_physics_step(self, actions: torch.Tensor):
        # ... (与上一版本相同的 _pre_physics_step 实现) ...
        self.actions = actions.to(self.device).clamp(-1.0, 1.0)

    def _apply_action(self):
        # ... (与上一版本相同的 _apply_action 实现) ...
        mid_point = (self._q_upper_limits + self._q_lower_limits) * 0.5
        half_range = (self._q_upper_limits - self._q_lower_limits) * 0.5
        joint_pos_target = mid_point + half_range * (self.actions * self.cfg.action_scale)
        self.robot.set_joint_position_target(joint_pos_target)

    def _compute_and_store_intermediate_states(self):
        # ... (与上一版本相同的 _compute_and_store_intermediate_states 实现) ...
        root_pos_w = self.robot.data.root_pos_w.to(self.device)
        root_quat_w = self.robot.data.root_quat_w.to(self.device)
        root_lin_vel_w = self.robot.data.root_lin_vel_w.to(self.device)
        root_ang_vel_w = self.robot.data.root_ang_vel_w.to(self.device)
        (
            self.root_lin_vel_b, self.root_ang_vel_b,
            self.roll_b, self.pitch_b, self.yaw_b,
            self.body_z_proj_w
        ) = compute_intermediate_robot_states(
            root_pos_w, root_quat_w, root_lin_vel_w, root_ang_vel_w
        )

    def _get_observations(self) -> dict:
        # ... (与上一版本相同的 _get_observations 实现) ...
        self._compute_and_store_intermediate_states()
        root_quat_w = self.robot.data.root_quat_w.to(self.device)
        joint_pos = self.robot.data.joint_pos.to(self.device)
        obs_list = [
            root_quat_w, self.root_ang_vel_b, joint_pos, self.constant_forward_command_obs,
        ]
        observations = torch.cat(obs_list, dim=-1)
        return {"policy": observations}

    def _get_rewards(self) -> torch.Tensor:
        # 获取状态
        root_pos_w = self.robot.data.root_pos_w.to(self.device) # 获取 root_pos_w
        root_quat_w = self.robot.data.root_quat_w.to(self.device)
        root_lin_vel_w = self.robot.data.root_lin_vel_w.to(self.device)
        world_root_ang_vel = self.robot.data.root_ang_vel_w.to(self.device)
        applied_torque = self.robot.data.applied_torque.to(self.device)
        joint_pos = self.robot.data.joint_pos.to(self.device)
        joint_vel = self.robot.data.joint_vel.to(self.device)

        # 确保 body_z_proj_w 已计算
        if not hasattr(self, 'body_z_proj_w') or self.body_z_proj_w is None:
            self._compute_and_store_intermediate_states()

        # 调用 JIT 脚本计算基础奖励和惩罚
        base_reward_terms = compute_sixfeet_rewards(
            root_pos_w,         # <<--- 传递 root_pos_w
            root_quat_w,
            root_lin_vel_w,
            world_root_ang_vel,
            self.body_z_proj_w,
            joint_pos,
            joint_vel,
            applied_torque,
            self._q_lower_limits,
            self._q_upper_limits,
            self.actions,
            # 奖励权重和参数从 cfg 获取
            self.cfg.rew_scale_target_height,
            self.cfg.target_height_m,
            self.cfg.rew_scale_forward_progress,
            self.cfg.rew_scale_upright,
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_root_ang_vel,
            self.cfg.rew_scale_action_cost,
            self.cfg.rew_scale_torque,
            self.cfg.rew_scale_joint_vel,
            self.cfg.soft_joint_vel_limit,
            self.cfg.rew_scale_dof_at_limit,
            self.cfg.rew_scale_toe_orientation_penalty,
            self.cfg.rew_scale_low_height_penalty,    # <<--- 新增
            self.cfg.min_height_penalty_threshold, # <<--- 新增
            self._toe_joint_indices
        )

        # 动作变化率惩罚
        penalty_action_rate_value = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.rew_scale_action_rate < 0:
            if hasattr(self, 'previous_actions'):
                 penalty_action_rate_value = torch.sum((self.actions - self.previous_actions)**2, dim=-1)
            self.previous_actions = self.actions.clone()

        current_total_reward = base_reward_terms + (self.cfg.rew_scale_action_rate * penalty_action_rate_value)
        
        # 应用终止惩罚
        current_terminated, current_time_out = self._get_dones_for_reward_calculation()
        just_failed_termination = current_terminated & (~current_time_out)
        
        final_reward = torch.where(just_failed_termination,
                                   torch.full_like(current_total_reward, self.cfg.rew_scale_termination),
                                   current_total_reward)

        # 日志记录 (简化版，确保与cfg和compute_sixfeet_rewards中的项对应)
        current_height_for_log = root_pos_w[:, 2]
        is_too_low_for_log = (current_height_for_log < self.cfg.min_height_penalty_threshold).float()

        self.extras["log"] = {
            "reward/total": final_reward.mean(),
            "reward_term/target_height": (self.cfg.rew_scale_target_height * torch.clamp(current_height_for_log / self.cfg.target_height_m, max=1.1)).mean(),
            "penalty_term/low_height": (self.cfg.rew_scale_low_height_penalty * is_too_low_for_log).mean(),
            "penalty_term/termination_if_fail": (self.cfg.rew_scale_termination * just_failed_termination.float()).mean(),
            "metrics/current_height_z": current_height_for_log.mean(),
            "metrics/body_z_proj_w_actual": self.body_z_proj_w.mean(),
            # 你需要将所有在 compute_sixfeet_rewards 中计算的、且在 cfg 中有对应权重的项，在这里乘以权重并记录
            # 例如，如果 compute_sixfeet_rewards 中计算了 reward_forward_progress (未乘以权重):
            # "reward_term/forward_progress": (self.cfg.rew_scale_forward_progress * reward_forward_progress_from_compute_func).mean(),
            # "penalty_term/toe_orientation": (self.cfg.rew_scale_toe_orientation_penalty * penalty_toe_orientation_from_compute_func).mean(),
            # 等等...
        }
        if self._toe_joint_indices is not None and self.cfg.rew_scale_toe_orientation_penalty != 0.0 :
             self.extras["log"]["metrics/avg_toe_pos_raw"] = joint_pos[:, self._toe_joint_indices][joint_pos[:, self._toe_joint_indices] > 0].mean() if torch.any(joint_pos[:, self._toe_joint_indices] > 0) else torch.tensor(0.0, device=self.device) # 只记录上翻脚趾的平均角度
        return final_reward

    def _get_dones_for_reward_calculation(self) -> tuple[torch.Tensor, torch.Tensor]:
        # ... (与上一版本相同的实现) ...
        if not hasattr(self, 'body_z_proj_w') or self.body_z_proj_w is None:
            self._compute_and_store_intermediate_states()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fallen_over = self.body_z_proj_w < self.cfg.termination_body_z_thresh
        root_pos_w = self.robot.data.root_pos_w.to(self.device)
        height_too_low = root_pos_w[:, 2] < self.cfg.termination_height_thresh
        terminated = fallen_over | height_too_low
        return terminated, time_out

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # ... (与上一版本相同的实现) ...
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fallen_over = self.body_z_proj_w < self.cfg.termination_body_z_thresh
        root_pos_w = self.robot.data.root_pos_w.to(self.device)
        height_too_low = root_pos_w[:, 2] < self.cfg.termination_height_thresh
        terminated = fallen_over | height_too_low
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # ... (与上一版本相同的实现) ...
        if env_ids is None: env_ids = self.robot._ALL_INDICES # type: ignore
        super()._reset_idx(env_ids)
        eids = torch.as_tensor(env_ids, device=self.device)
        num_resets = len(eids)
        root_state_reset = self.robot.data.default_root_state[eids].clone()
        root_state_reset[:, 2] = self.robot.data.default_root_state[eids, 2] + self.cfg.reset_height_offset
        random_yaw = (torch.rand(num_resets, device=self.device) - 0.5) * 2.0 * self.cfg.root_orientation_yaw_range
        world_z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        orientation_quat_xyzw = quat_from_angle_axis(random_yaw, world_z_axis)
        root_state_reset[:, 3:7] = orientation_quat_xyzw
        root_state_reset[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state_reset, eids)
        num_joints = self._q_lower_limits.shape[0]
        random_proportions = torch.rand(num_resets, num_joints, device=self.device)
        joint_ranges = self._q_upper_limits.unsqueeze(0) - self._q_lower_limits.unsqueeze(0)
        random_joint_pos = self._q_lower_limits.unsqueeze(0) + random_proportions * joint_ranges
        reset_joint_vel = torch.zeros_like(random_joint_pos)
        self.robot.write_joint_state_to_sim(random_joint_pos, reset_joint_vel, env_ids=eids)
        if self.cfg.rew_scale_action_rate < 0:
            self.previous_actions[eids] = 0.0

# 需要定义 penalty_dof_at_limit_calc (如果日志中使用了它)
@torch.jit.script
def penalty_dof_at_limit_calc(joint_pos: torch.Tensor, q_lower_limits: torch.Tensor, q_upper_limits: torch.Tensor) -> torch.Tensor:
    dof_range = q_upper_limits - q_lower_limits
    dof_range = torch.where(dof_range < 1e-6, torch.ones_like(dof_range), dof_range) # 避免除以0
    dof_pos_scaled_01 = (joint_pos - q_lower_limits.unsqueeze(0)) / dof_range.unsqueeze(0)
    near_lower_limit = torch.relu(0.05 - dof_pos_scaled_01)**2
    near_upper_limit = torch.relu(dof_pos_scaled_01 - 0.95)**2
    return torch.sum(near_lower_limit + near_upper_limit, dim=-1)