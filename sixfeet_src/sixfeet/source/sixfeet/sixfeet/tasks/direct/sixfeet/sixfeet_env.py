# sixfeet_env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
import math
from collections.abc import Sequence

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate, quat_from_angle_axis, quat_mul # 确保导入了所有用到的 math 函数
import torch.nn.functional as F

# Your local project imports
from .sixfeet_env_cfg import SixfeetEnvCfg

# ───────────────── Reward Helper Functions ──────────────────
@torch.jit.script
def penalty_root_angular_velocity_l2(root_ang_vel_w: torch.Tensor) -> torch.Tensor:
    """Computes the L2 norm of the root angular velocity."""
    return torch.linalg.norm(root_ang_vel_w, dim=-1)

@torch.jit.script
def penalty_joint_torques_sq(applied_torque: torch.Tensor) -> torch.Tensor:
    """Computes the sum of squared joint torques."""
    return (applied_torque ** 2).sum(-1)

@torch.jit.script
def penalty_joint_velocities_with_limit(
    joint_vel: torch.Tensor, soft_vel_limit: float
) -> torch.Tensor:
    """Computes a penalty for joint velocities, including an increased penalty for exceeding a soft limit."""
    vel_sq_penalty = torch.sum(joint_vel**2, dim=-1)
    exceeded_vel_sq_penalty = torch.sum(torch.relu(torch.abs(joint_vel) - soft_vel_limit)**2, dim=-1)
    return vel_sq_penalty + exceeded_vel_sq_penalty

# ────────────────────── SixfeetEnv Class Definition ─────────────────────────
class SixfeetEnv(DirectRLEnv):
    cfg: SixfeetEnvCfg

    def __init__(self, cfg: SixfeetEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment."""
        super().__init__(cfg, render_mode, **kwargs)

        joint_limits = self.robot.data.joint_pos_limits[0].to(self.device)
        self._q_lower_limits = joint_limits[:, 0]
        self._q_upper_limits = joint_limits[:, 1]

        self._fixed_forward_command_local = torch.tensor([0.0, 1.0], device=self.device)
        self.constant_forward_command_obs = self._fixed_forward_command_local.unsqueeze(0).expand(self.num_envs, -1)

        self._resolve_toe_joint_indices()

    def _resolve_toe_joint_indices(self):
        """Helper function to resolve toe joint indices from configuration."""
        self._toe_joint_indices: torch.Tensor | None = None
        expr_or_list = getattr(self.cfg, 'toe_joint_names_expr', None)

        if isinstance(expr_or_list, str):
            # 使用正则表达式查找关节
            joint_indices_list, joint_names_list = self.robot.find_joints(expr_or_list)
            if not joint_indices_list: # 检查返回的列表是否为空
                print(f"[WARNING] SixfeetEnv: No toe joints found using regex: '{expr_or_list}'.")
            else:
                self._toe_joint_indices = torch.tensor(joint_indices_list, device=self.device, dtype=torch.long)
                print(f"[INFO] SixfeetEnv: Found toe joint indices via regex: {self._toe_joint_indices.tolist()}")
                print(f"[INFO] SixfeetEnv: Corresponding toe joint names: {joint_names_list}")
        elif isinstance(expr_or_list, list) and all(isinstance(i, int) for i in expr_or_list):
            # 使用预定义的整数索引列表
            if not expr_or_list: # 检查列表是否为空
                print(f"[WARNING] SixfeetEnv: 'toe_joint_names_expr' is an empty list.")
            else:
                temp_indices = torch.tensor(expr_or_list, device=self.device, dtype=torch.long)
                if torch.any(temp_indices < 0) or torch.any(temp_indices >= self.robot.num_articulated_joints):
                    print(f"[ERROR] SixfeetEnv: Invalid toe joint indices in list: {expr_or_list}. Indices out of bounds for {self.robot.num_articulated_joints} joints.")
                else:
                    self._toe_joint_indices = temp_indices
                    print(f"[INFO] SixfeetEnv: Using pre-defined toe joint indices: {self._toe_joint_indices.tolist()}")
        elif expr_or_list is not None: # 如果提供了但类型不对
            print(f"[WARNING] SixfeetEnv: 'toe_joint_names_expr' is of invalid type: {type(expr_or_list)}. Expected str or list[int].")

        # 如果最终没有成功解析出脚趾索引
        if self._toe_joint_indices is None:
            message_suffix = "Toe orientation penalty will not be applied."
            if expr_or_list is None: # 情况：cfg中根本没有定义
                print(f"[INFO] SixfeetEnv: 'toe_joint_names_expr' not specified. {message_suffix}")
            else: # 情况：cfg中定义了，但解析失败 (空列表，无效索引，类型错误等)
                print(f"[INFO] SixfeetEnv: Could not resolve toe joints from 'toe_joint_names_expr': '{expr_or_list}'. {message_suffix}")


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        spawn_ground_plane("/World/ground", GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)))
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.scene.clone_environments(copy_from_source=False)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.to(self.device).clamp(-1.0, 1.0)

    def _apply_action(self):
        mid_point = (self._q_upper_limits + self._q_lower_limits) * 0.5
        half_range = (self._q_upper_limits - self._q_lower_limits) * 0.5
        joint_pos_target = mid_point + half_range * (self.actions * self.cfg.action_scale)
        self.robot.set_joint_position_target(joint_pos_target)

    def _get_observations(self) -> dict:
        # print(f"[DEBUG] SixfeetEnv: Initial device of root_quat_w: {self.robot.data.root_quat_w.device}")
        # print(f"[DEBUG] SixfeetEnv: Initial device of root_ang_vel_w: {self.robot.data.root_ang_vel_w.device}") # 关键的速度张量
        # print(f"[DEBUG] SixfeetEnv: Initial device of joint_pos: {self.robot.data.joint_pos.device}")

        obs_list = [
            self.robot.data.root_quat_w.to(self.device),
            self.robot.data.root_ang_vel_w.to(self.device),
            self.robot.data.joint_pos.to(self.device),
            self.constant_forward_command_obs,
        ]
        observations = torch.cat(obs_list, dim=-1)
        # print(f"[DEBUG] SixfeetEnv: Final observation tensor device: {observations.device}")
        return {"policy": observations}

    def _get_rewards(self) -> torch.Tensor:
        root_quat_w = self.robot.data.root_quat_w.to(self.device)
        root_lin_vel_w = self.robot.data.root_lin_vel_w.to(self.device)
        root_ang_vel_w = self.robot.data.root_ang_vel_w.to(self.device)
        applied_torque = self.robot.data.applied_torque.to(self.device)
        joint_pos = self.robot.data.joint_pos.to(self.device)
        joint_vel = self.robot.data.joint_vel.to(self.device)

        # Forward Progress Reward
        local_forward_y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, -1)
        robot_forward_dir_w = quat_rotate(root_quat_w, local_forward_y_axis)
        robot_forward_dir_xy_w = F.normalize(robot_forward_dir_w[:, :2], p=2, dim=-1)
        current_lin_vel_xy_w = root_lin_vel_w[:, :2]
        forward_velocity_component = torch.sum(current_lin_vel_xy_w * robot_forward_dir_xy_w, dim=1)
        reward_forward_progress = forward_velocity_component

        # Body Z-axis Alignment Reward
        local_z_axis = torch.zeros_like(root_lin_vel_w)
        local_z_axis[:, 2] = 1.0
        body_z_in_world = quat_rotate(root_quat_w, local_z_axis)
        reward_body_z_align = body_z_in_world[:, 2].clamp(min=0.0)

        # Toe Orientation Penalty
        penalty_toe_orientation = torch.zeros(self.num_envs, device=self.device)
        if self._toe_joint_indices is not None: # 只有在成功获取索引后才计算
            toe_joint_positions = joint_pos[:, self._toe_joint_indices]
            penalty_toe_orientation = torch.sum(torch.relu(toe_joint_positions)**2, dim=-1)

        # Other Penalties
        penalty_r_ang_vel = penalty_root_angular_velocity_l2(root_ang_vel_w)
        penalty_trq = penalty_joint_torques_sq(applied_torque)
        penalty_j_vel = penalty_joint_velocities_with_limit(joint_vel, self.cfg.soft_joint_vel_limit)

        # Combine Rewards and Penalties (assuming penalty weights in cfg are negative)
        reward = (
            self.cfg.rew_scale_forward_progress * reward_forward_progress
            + self.cfg.rew_scale_body_z_align * reward_body_z_align
            + self.cfg.rew_scale_root_ang_vel * penalty_r_ang_vel
            + self.cfg.rew_scale_torque       * penalty_trq
            + self.cfg.rew_scale_joint_vel    * penalty_j_vel
            + self.cfg.rew_scale_toe_orientation_penalty * penalty_toe_orientation
        )

        self.extras["log"] = {
            "reward/total": reward.mean(),
            "reward_term/forward_progress": (self.cfg.rew_scale_forward_progress * reward_forward_progress).mean(),
            "reward_term/body_z_align": (self.cfg.rew_scale_body_z_align * reward_body_z_align).mean(),
            "penalty_term/root_ang_vel": (self.cfg.rew_scale_root_ang_vel * penalty_r_ang_vel).mean(),
            "penalty_term/torque": (self.cfg.rew_scale_torque * penalty_trq).mean(),
            "penalty_term/joint_vel": (self.cfg.rew_scale_joint_vel * penalty_j_vel).mean(),
            "penalty_term/toe_orientation": (self.cfg.rew_scale_toe_orientation_penalty * penalty_toe_orientation).mean(),
            "metrics/forward_velocity_actual": forward_velocity_component.mean(),
            "metrics/avg_abs_joint_vel": torch.abs(joint_vel).mean(),
            "metrics/body_z_alignment_raw": body_z_in_world[:, 2].mean(),
        }
        if self._toe_joint_indices is not None:
             self.extras["log"]["metrics/avg_toe_pos_raw"] = joint_pos[:, self._toe_joint_indices].mean()
             self.extras["log"]["metrics/max_toe_pos_raw"] = joint_pos[:, self._toe_joint_indices].max()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        root_quat_w = self.robot.data.root_quat_w.to(self.device)
        root_pos_w = self.robot.data.root_pos_w.to(self.device)
        local_z_axis = torch.zeros_like(root_pos_w)
        local_z_axis[:, 2] = 1.0
        body_z_in_world = quat_rotate(root_quat_w, local_z_axis)
        fallen_over = body_z_in_world[:, 2] < self.cfg.termination_body_z_thresh
        height_too_low = root_pos_w[:, 2] < self.cfg.termination_height_thresh
        terminated = fallen_over | height_too_low
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES # type: ignore
        super()._reset_idx(env_ids)

        eids = torch.as_tensor(env_ids, device=self.device)
        num_resets = len(eids)

        # ---- 1. Reset Root State ----
        root_state_reset = self.robot.data.default_root_state[eids].clone()
        root_state_reset[:, 2] += self.cfg.reset_height_offset
        random_yaw = (torch.rand(num_resets, device=self.device) - 0.5) * 2.0 * self.cfg.root_orientation_yaw_range
        world_z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        orientation_quat = quat_from_angle_axis(random_yaw, world_z_axis)
        root_state_reset[:, 3:7] = orientation_quat
        root_state_reset[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state_reset, eids)

        # ---- 2. Reset Joint State ----
        num_joints = self._q_lower_limits.shape[0]
        random_proportions = torch.rand(num_resets, num_joints, device=self.device)
        joint_ranges = self._q_upper_limits.unsqueeze(0) - self._q_lower_limits.unsqueeze(0)
        random_joint_pos = self._q_lower_limits.unsqueeze(0) + random_proportions * joint_ranges
        reset_joint_vel = torch.zeros_like(random_joint_pos)
        self.robot.write_joint_state_to_sim(
            random_joint_pos,
            reset_joint_vel,
            env_ids=eids
        )