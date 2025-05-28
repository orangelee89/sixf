# sixfeet_env.py
from __future__ import annotations
import torch
import math
from collections.abc import Sequence
from typing import List, Dict, Optional

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import (
    quat_rotate,
    quat_from_angle_axis,
    # quat_mul, # 如果下面的RPY转换不理想，可能需要这个
    # quat_from_euler_xyz, # Isaac Lab 可能没有直接提供这个，但有类似工具
    convert_quat,
    euler_xyz_from_quat
)
# from isaaclab.terrains import TerrainImporter

from .sixfeet_env_cfg import SixfeetEnvCfg

# ───────────────── 辅助函数 ──────────────────
@torch.jit.script
def normalize_angle_for_obs(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))

@torch.jit.script
def compute_sixfeet_rewards_directional(
    # ... (大部分现有参数不变) ...
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    projected_gravity_b: torch.Tensor, # 世界Z轴在机器人本体坐标系下的投影 (取反后归一化)
    joint_pos_rel: torch.Tensor,
    joint_vel: torch.Tensor,
    applied_torque: torch.Tensor,
    joint_acc: torch.Tensor,
    q_lower_limits: torch.Tensor,
    q_upper_limits: torch.Tensor,
    current_joint_pos_abs: torch.Tensor,
    actions_from_policy: torch.Tensor,
    previous_actions_from_policy: torch.Tensor,
    root_pos_w: torch.Tensor,
    undesired_contacts_active: torch.Tensor,
    commands_discrete: torch.Tensor,
    cfg_cmd_profile: Dict[str, float],
    cfg_rew_scale_move_in_commanded_direction: float,
    cfg_rew_scale_achieve_reference_angular_rate: float,
    cfg_rew_scale_alive: float,
    cfg_rew_scale_target_height: float,
    cfg_target_height_m: float,
    cfg_rew_scale_action_cost: float,
    cfg_rew_scale_action_rate: float,
    cfg_rew_scale_joint_torques: float,
    cfg_rew_scale_joint_accel: float,
    cfg_rew_scale_lin_vel_z_penalty: float,
    cfg_rew_scale_ang_vel_xy_penalty: float,
    cfg_rew_scale_flat_orientation: float,
    cfg_rew_scale_unwanted_movement_penalty: float,
    cfg_rew_scale_dof_at_limit: float,
    cfg_rew_scale_toe_orientation_penalty: float,
    cfg_toe_joint_indices: Optional[torch.Tensor],
    cfg_rew_scale_low_height_penalty: float,
    cfg_min_height_penalty_threshold: float,
    cfg_rew_scale_undesired_contact: float,
    dt: float,
    # --- 新增参数 ---
    cfg_rew_scale_orientation_deviation: float # 新的Z轴偏差惩罚的scale
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    ref_ang_rate = cfg_cmd_profile.get("reference_angular_rate", 0.0)

    # 1. Movement Rewards
    linear_vel_x_local = root_lin_vel_b[:, 0]
    linear_vel_y_local = root_lin_vel_b[:, 1]
    reward_fwd_bkwd_move = commands_discrete[:, 0] * linear_vel_x_local
    reward_left_right_move = commands_discrete[:, 1] * linear_vel_y_local
    is_linear_cmd_active = (torch.abs(commands_discrete[:, 0]) > 0.5) | (torch.abs(commands_discrete[:, 1]) > 0.5)
    reward_linear_direction = (reward_fwd_bkwd_move + reward_left_right_move) * is_linear_cmd_active.float()
    reward_move_in_commanded_direction = reward_linear_direction * cfg_rew_scale_move_in_commanded_direction
    angular_vel_z_local = root_ang_vel_b[:, 2]
    reward_angular_direction_raw = -commands_discrete[:, 2] * angular_vel_z_local
    is_turn_cmd_active = torch.abs(commands_discrete[:, 2]) > 0.5
    turn_rate_error = torch.abs(torch.abs(angular_vel_z_local) - ref_ang_rate)
    reward_achieve_ref_ang_rate = torch.exp(-5.0 * turn_rate_error) * is_turn_cmd_active.float() \
                                   * cfg_rew_scale_achieve_reference_angular_rate
    reward_turn = (reward_angular_direction_raw * is_turn_cmd_active.float() * cfg_rew_scale_move_in_commanded_direction) + \
                  reward_achieve_ref_ang_rate

    # 2. Alive and Height Rewards
    reward_alive = torch.ones_like(commands_discrete[:,0]) * cfg_rew_scale_alive
    current_height_z = root_pos_w[:, 2]
    height_check = torch.clamp(current_height_z / cfg_target_height_m, max=1.1)
    reward_target_height = height_check * cfg_rew_scale_target_height

    # 3. Action Penalties
    penalty_action_cost = torch.sum(actions_from_policy**2, dim=-1) * cfg_rew_scale_action_cost
    penalty_action_rate = torch.sum((actions_from_policy - previous_actions_from_policy)**2, dim=-1) * cfg_rew_scale_action_rate

    # 4. Efficiency Penalties
    penalty_joint_torques = torch.sum(applied_torque**2, dim=-1) * cfg_rew_scale_joint_torques
    penalty_joint_accel = torch.sum(joint_acc**2, dim=-1) * cfg_rew_scale_joint_accel
    
    # 5. Stability Penalties
    penalty_lin_vel_z = torch.square(root_lin_vel_b[:, 2]) * cfg_rew_scale_lin_vel_z_penalty
    penalty_ang_vel_xy = torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1) * cfg_rew_scale_ang_vel_xy_penalty
    penalty_flat_orientation = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1) * cfg_rew_scale_flat_orientation # 惩罚XY方向的重力投影，即身体不水平
    is_stand_cmd_active = torch.all(commands_discrete == 0, dim=1)
    unwanted_lin_vel_sq = torch.sum(torch.square(root_lin_vel_b[:, :2]), dim=1)
    unwanted_ang_vel_sq = torch.square(root_ang_vel_b[:, 2])
    penalty_unwanted_movement = (unwanted_lin_vel_sq + unwanted_ang_vel_sq) * \
                                is_stand_cmd_active.float() * cfg_rew_scale_unwanted_movement_penalty

    # 6. Constraint Penalties
    dof_range = q_upper_limits - q_lower_limits
    dof_range = torch.where(dof_range < 1e-6, torch.ones_like(dof_range), dof_range)
    q_lower_expanded = q_lower_limits.unsqueeze(0) if q_lower_limits.ndim == 1 else q_lower_limits
    dof_range_expanded = dof_range.unsqueeze(0) if dof_range.ndim == 1 else dof_range
    dof_pos_scaled_01 = (current_joint_pos_abs - q_lower_expanded) / dof_range_expanded
    near_lower_limit = torch.relu(0.05 - dof_pos_scaled_01)**2
    near_upper_limit = torch.relu(dof_pos_scaled_01 - 0.95)**2
    penalty_dof_at_limit = torch.sum(near_lower_limit + near_upper_limit, dim=-1) * cfg_rew_scale_dof_at_limit
    
    # --- 条件化惩罚逻辑 ---
    # is_severely_tilted: 机器人 Z 轴是否大致朝下 (与世界Z轴夹角 > 90度)
    is_severely_tilted = projected_gravity_b[:, 2] > 0.0

    # 条件化足端方向惩罚
    _base_penalty_toe_orientation = torch.zeros_like(commands_discrete[:,0], device=commands_discrete.device)
    if cfg_rew_scale_toe_orientation_penalty != 0.0 and cfg_toe_joint_indices is not None:
        if cfg_toe_joint_indices.numel() > 0:
            toe_joint_positions = current_joint_pos_abs[:, cfg_toe_joint_indices]
            _base_penalty_toe_orientation = torch.sum(torch.relu(toe_joint_positions)**2, dim=-1) * cfg_rew_scale_toe_orientation_penalty
    penalty_toe_orientation = torch.where(
        is_severely_tilted, 
        torch.zeros_like(_base_penalty_toe_orientation), 
        _base_penalty_toe_orientation
    )
    
    # 低高度惩罚 (这个不受 is_severely_tilted 影响，除非你也想改)
    is_too_low = (current_height_z < cfg_min_height_penalty_threshold).float()
    penalty_low_height = is_too_low * cfg_rew_scale_low_height_penalty

    # 条件化不期望的接触惩罚
    _base_penalty_undesired_contact = undesired_contacts_active.float() * cfg_rew_scale_undesired_contact
    penalty_undesired_contact = torch.where(
        is_severely_tilted,
        torch.zeros_like(_base_penalty_undesired_contact),
        _base_penalty_undesired_contact
    )

    # --- 新增：Z轴方向偏差惩罚 ---
    # projected_gravity_b[:, 2] 的范围是 [-1, 1]
    # -1: 机器人Z轴与世界Z轴同向 (直立)
    # +1: 机器人Z轴与世界Z轴反向 (倒置)
    # acos的输入范围是 [-1, 1]。为防止计算误差导致超出范围，进行clamp。
    cos_angle_robot_z_with_world_z = -projected_gravity_b[:, 2] # 这是机器人Z轴与世界Z轴点积
    angle_deviation_from_world_z = torch.acos(torch.clamp(cos_angle_robot_z_with_world_z, -1.0 + 1e-7, 1.0 - 1e-7)) # 弧度制, 范围 [0, pi]
    # 当直立时，angle_deviation_from_world_z 接近 0
    # 当倒置时，angle_deviation_from_world_z 接近 pi
    # scale 应该是负的，所以偏差越大，惩罚越大（负的越多）
    penalty_orientation_deviation = cfg_rew_scale_orientation_deviation * angle_deviation_from_world_z
    
    # --- 总奖励计算 ---
    # penalty_orientation_deviation 不乘以 dt，直接作用
    # penalty_flat_orientation 保持在你基准文件中的处理方式（乘以dt）
    total_reward = (
        reward_move_in_commanded_direction + reward_turn + reward_alive + reward_target_height + penalty_orientation_deviation
        + (penalty_action_cost + penalty_action_rate + penalty_joint_torques + penalty_joint_accel
        + penalty_lin_vel_z + penalty_ang_vel_xy + penalty_flat_orientation + penalty_unwanted_movement
        + penalty_dof_at_limit + penalty_toe_orientation + penalty_low_height
        + penalty_undesired_contact) * dt
    )
    
    reward_terms: Dict[str, torch.Tensor] = {
        "move_in_commanded_direction": reward_move_in_commanded_direction,
        "turn_reward_combined": reward_turn,
        "alive": reward_alive,
        "target_height": reward_target_height,
        "orientation_deviation_penalty": penalty_orientation_deviation, # 新增
        "action_cost_penalty": penalty_action_cost * dt,
        "action_rate_penalty": penalty_action_rate * dt,
        "joint_torques_penalty": penalty_joint_torques * dt,
        "joint_accel_penalty": penalty_joint_accel * dt,
        "lin_vel_z_penalty": penalty_lin_vel_z * dt,
        "ang_vel_xy_penalty": penalty_ang_vel_xy * dt,
        "flat_orientation_penalty": penalty_flat_orientation * dt, # 按你的基准文件处理
        "unwanted_movement_penalty": penalty_unwanted_movement * dt,
        "dof_at_limit_penalty": penalty_dof_at_limit * dt,
        "toe_orientation_penalty": penalty_toe_orientation * dt, 
        "low_height_penalty": penalty_low_height * dt,
        "undesired_contact_penalty": penalty_undesired_contact * dt,
    }
    return total_reward, reward_terms


class SixfeetEnv(DirectRLEnv):
    cfg: SixfeetEnvCfg
    _contact_sensor: ContactSensor

    def __init__(self, cfg: SixfeetEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._default_joint_pos = self.robot.data.default_joint_pos.clone()
        if self._default_joint_pos.ndim > 1 and self._default_joint_pos.shape[0] == self.num_envs:
            self._default_joint_pos = self._default_joint_pos[0]
        
        joint_limits = self.robot.data.joint_pos_limits[0].to(self.device)
        self._q_lower_limits = joint_limits[:, 0]
        self._q_upper_limits = joint_limits[:, 1]
        
        self._policy_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_policy_actions = torch.zeros_like(self._policy_actions)
        self._processed_actions = torch.zeros_like(self._policy_actions)

        self._commands = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self._time_since_last_command_change = torch.zeros(self.num_envs, device=self.device)
        
        self._resolve_toe_joint_indices() # num_dof 修正已在方法内

        self._undesired_contact_body_ids: Optional[List[int]] = None
        if self.cfg.undesired_contact_link_names_expr and self.cfg.rew_scale_undesired_contact != 0.0:
            indices, names = self._contact_sensor.find_bodies(self.cfg.undesired_contact_link_names_expr)
            if indices:
                self._undesired_contact_body_ids = indices
                print(f"[INFO] SixfeetEnv: Undesired contact body IDs: {self._undesired_contact_body_ids} for names {names}")
            else:
                print(f"[WARNING] SixfeetEnv: No bodies for undesired contact expr: {self.cfg.undesired_contact_link_names_expr}")

        self._base_body_id: Optional[List[int]] = None
        if self.cfg.termination_base_contact and self.cfg.base_link_name:
            indices, names = self._contact_sensor.find_bodies(self.cfg.base_link_name)
            if indices:
                self._base_body_id = indices
                print(f"[INFO] SixfeetEnv: Base body ID for termination: {self._base_body_id} for names {names}")
            else:
                print(f"[WARNING] SixfeetEnv: No body for base contact termination: {self.cfg.base_link_name}")
        
        self._episode_reward_terms_sum: Dict[str, torch.Tensor] = {}

    def _resolve_toe_joint_indices(self):
        self._toe_joint_indices: Optional[torch.Tensor] = None
        expr_or_list = getattr(self.cfg, 'toe_joint_names_expr', None)
        if self.cfg.rew_scale_toe_orientation_penalty == 0.0 or not expr_or_list:
            return

        num_dof_val = self._q_lower_limits.numel() # 使用 _q_lower_limits 获取 DoF 数量
        joint_names_list_for_logging = []

        if isinstance(expr_or_list, str):
            joint_indices_list, joint_names_list_for_logging = self.robot.find_joints(expr_or_list)
            if joint_indices_list:
                self._toe_joint_indices = torch.tensor(joint_indices_list, device=self.device, dtype=torch.long)
        elif isinstance(expr_or_list, list) and all(isinstance(i, int) for i in expr_or_list):
            if expr_or_list:
                temp_indices = torch.tensor(expr_or_list, device=self.device, dtype=torch.long)
                if torch.any(temp_indices < 0) or torch.any(temp_indices >= num_dof_val): # 使用 num_dof_val
                    print(f"[ERROR] SixfeetEnv: Invalid toe joint indices in list: {expr_or_list}. Max allowable index: {num_dof_val - 1}")
                    self._toe_joint_indices = None
                else:
                    self._toe_joint_indices = temp_indices
        elif expr_or_list is not None:
            print(f"[WARNING] SixfeetEnv: 'toe_joint_names_expr' ('{expr_or_list}') in cfg has invalid type: {type(expr_or_list)}. Expected str or list[int].")

        if self._toe_joint_indices is not None:
            if self._toe_joint_indices.numel() == 0:
                self._toe_joint_indices = None
            elif torch.any(self._toe_joint_indices < 0) or torch.any(self._toe_joint_indices >= num_dof_val): # 使用 num_dof_val
                print(f"[ERROR] SixfeetEnv: Invalid toe joint indices after processing: {self._toe_joint_indices.tolist()}. Max allowable index: {num_dof_val - 1}")
                self._toe_joint_indices = None
            else:
                log_msg = f"[INFO] SixfeetEnv: Validated toe joint indices for penalty: {self._toe_joint_indices.tolist()}"
                if joint_names_list_for_logging:
                    log_msg += f", names: {joint_names_list_for_logging}"
                print(log_msg)
        
        if self._toe_joint_indices is None and expr_or_list is not None:
             print(f"[WARNING] SixfeetEnv: No valid toe joint indices resolved from '{expr_or_list}'. Toe orientation penalty might not apply effectively.")
        elif self._toe_joint_indices is None and expr_or_list is None and self.cfg.rew_scale_toe_orientation_penalty != 0.0:
             print(f"[INFO] SixfeetEnv: 'toe_joint_names_expr' not specified, but toe penalty > 0. Toe orientation penalty will not be applied.")

    def _setup_scene(self): # 与你提供的版本一致
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if hasattr(self.cfg, "terrain") and self.cfg.terrain is not None:
            if hasattr(self.scene, "cfg"):
                self.cfg.terrain.num_envs = self.scene.cfg.num_envs
                self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            terrain_class_path = getattr(self.cfg.terrain, "class_type", None)
            if isinstance(terrain_class_path, str):
                try:
                    module_path, class_name = terrain_class_path.rsplit('.', 1)
                    module = __import__(module_path, fromlist=[class_name])
                    terrain_class = getattr(module, class_name)
                except Exception as e:
                    print(f"[ERROR] Failed to import terrain class {terrain_class_path}: {e}")
                    from isaaclab.terrains import TerrainImporter
                    terrain_class = TerrainImporter
            elif terrain_class_path is None:
                from isaaclab.terrains import TerrainImporter
                terrain_class = TerrainImporter
            else:
                terrain_class = terrain_class_path
            self._terrain = terrain_class(self.cfg.terrain)
        else:
            print("[WARNING] SixfeetEnv: No terrain configuration. Spawning default plane.")
            from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
            spawn_ground_plane("/World/ground", GroundPlaneCfg())
            class DummyTerrain:
                def __init__(self, num_envs, device): self.env_origins = torch.zeros((num_envs, 3), device=device)
            self._terrain = DummyTerrain(self.cfg.scene.num_envs, self.device)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.scene.clone_environments(copy_from_source=False)

    def _update_commands(self, env_ids: torch.Tensor): # 与你提供的版本一致
        self._time_since_last_command_change[env_ids] += self.physics_dt
        cmd_duration_str = str(self.cfg.command_profile.get("command_mode_duration_s", "20.0"))
        if cmd_duration_str == "episode_length_s":
            cmd_duration = self.cfg.episode_length_s
        else:
            cmd_duration = float(cmd_duration_str)

        stand_still_prob = self.cfg.command_profile.get("stand_still_prob", 0.0)
        num_cmd_modes = self.cfg.command_profile.get("num_command_modes", 1)
        change_command_mask = self._time_since_last_command_change[env_ids] >= cmd_duration
        envs_to_change = env_ids[change_command_mask]
        if envs_to_change.numel() > 0:
            self._time_since_last_command_change[envs_to_change] = 0.0
            num_to_change = envs_to_change.shape[0]
            new_commands_for_changed_envs = torch.zeros(num_to_change, 3, device=self.device, dtype=torch.float)
            if stand_still_prob == 1.0:
                command_modes = torch.zeros(num_to_change, device=self.device, dtype=torch.long)
            elif num_cmd_modes > 0:
                command_modes = torch.randint(0, num_cmd_modes, (num_to_change,), device=self.device)
                if stand_still_prob > 0.0 and stand_still_prob < 1.0:
                    stand_mask = torch.rand(num_to_change, device=self.device) < stand_still_prob
                    command_modes[stand_mask] = 0
            else:
                command_modes = torch.zeros(num_to_change, device=self.device, dtype=torch.long)
            new_commands_for_changed_envs[command_modes == 1, 0] = 1.0
            new_commands_for_changed_envs[command_modes == 2, 0] = -1.0
            new_commands_for_changed_envs[command_modes == 3, 1] = -1.0
            new_commands_for_changed_envs[command_modes == 4, 1] = 1.0
            new_commands_for_changed_envs[command_modes == 5, 2] = -1.0
            new_commands_for_changed_envs[command_modes == 6, 2] = 1.0
            self._commands[envs_to_change] = new_commands_for_changed_envs

    def _pre_physics_step(self, actions: torch.Tensor): # 与你提供的版本一致
        self._policy_actions = actions.clone().to(self.device)
        if torch.any(torch.isnan(actions)) or torch.any(torch.isinf(actions)):
            print(f"[WARNING] Invalid actions detected: {actions}")
            actions = torch.zeros_like(actions)
        cur_pos = self.robot.data.joint_pos
        self._processed_actions = cur_pos + self.cfg.action_scale * self._policy_actions
        self._processed_actions = torch.clamp(
            self._processed_actions,
            self._q_lower_limits.unsqueeze(0),
            self._q_upper_limits.unsqueeze(0)
        )
        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._update_commands(all_env_ids)

    def _apply_action(self): # 与你提供的版本一致
        self.robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict: # 与你提供的版本一致
        self._previous_policy_actions = self._policy_actions.clone()
        default_pos_expanded = self._default_joint_pos.unsqueeze(0) if self._default_joint_pos.ndim == 1 else self._default_joint_pos
        joint_pos_rel = self.robot.data.joint_pos - default_pos_expanded
        obs_list = [
            self.robot.data.projected_gravity_b,
            self.robot.data.root_ang_vel_b,
            self._commands,
            normalize_angle_for_obs(joint_pos_rel),
            self.robot.data.joint_vel,
        ]
        observations_tensor = torch.cat(obs_list, dim=-1)
        if hasattr(self.cfg, "observation_space") and observations_tensor.shape[1] != self.cfg.observation_space:
            print(f"[ERROR] SixfeetEnv: Obs dim mismatch! Expected {self.cfg.observation_space}, got {observations_tensor.shape[1]}")
        return {"policy": observations_tensor}

    def _get_rewards(self) -> torch.Tensor:
        root_lin_vel_b = self.robot.data.root_lin_vel_b
        root_ang_vel_b = self.robot.data.root_ang_vel_b
        projected_gravity_b = self.robot.data.projected_gravity_b
        default_pos_expanded = self._default_joint_pos.unsqueeze(0) if self._default_joint_pos.ndim == 1 else self._default_joint_pos
        joint_pos_rel = self.robot.data.joint_pos - default_pos_expanded
        current_joint_pos_abs = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        applied_torque = self.robot.data.applied_torque
        joint_acc = getattr(self.robot.data, "joint_acc", torch.zeros_like(joint_vel, device=self.device))
        root_pos_w = self.robot.data.root_pos_w

        undesired_contacts_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.cfg.rew_scale_undesired_contact != 0.0 and self._undesired_contact_body_ids and len(self._undesired_contact_body_ids) > 0:
             if hasattr(self._contact_sensor.data, 'net_forces_w_history') and self._contact_sensor.data.net_forces_w_history is not None:
                all_forces_history = self._contact_sensor.data.net_forces_w_history
                if all_forces_history.ndim == 4 and all_forces_history.shape[1] > 0 and \
                   self._undesired_contact_body_ids and max(self._undesired_contact_body_ids) < all_forces_history.shape[2]: # 安全检查
                    current_net_forces_w = all_forces_history[:, -1, :, :]
                    forces_on_undesired_bodies = current_net_forces_w[:, self._undesired_contact_body_ids, :]
                    force_magnitudes = torch.norm(forces_on_undesired_bodies, dim=-1)
                    undesired_contacts_active = torch.any(force_magnitudes > 1.0, dim=1)
        
        total_reward, reward_terms_dict = compute_sixfeet_rewards_directional(
            root_lin_vel_b, root_ang_vel_b, projected_gravity_b,
            joint_pos_rel, joint_vel, applied_torque, joint_acc,
            self._q_lower_limits, self._q_upper_limits, current_joint_pos_abs,
            self._policy_actions, self._previous_policy_actions,
            root_pos_w, undesired_contacts_active, self._commands,
            self.cfg.command_profile,
            self.cfg.rew_scale_move_in_commanded_direction, self.cfg.rew_scale_achieve_reference_angular_rate,
            self.cfg.rew_scale_alive, self.cfg.rew_scale_target_height, self.cfg.target_height_m,
            self.cfg.rew_scale_action_cost, self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_joint_torques, self.cfg.rew_scale_joint_accel,
            self.cfg.rew_scale_lin_vel_z_penalty, self.cfg.rew_scale_ang_vel_xy_penalty,
            self.cfg.rew_scale_flat_orientation, self.cfg.rew_scale_unwanted_movement_penalty,
            self.cfg.rew_scale_dof_at_limit, self.cfg.rew_scale_toe_orientation_penalty, self._toe_joint_indices,
            self.cfg.rew_scale_low_height_penalty, self.cfg.min_height_penalty_threshold,
            self.cfg.rew_scale_undesired_contact, 
            self.sim.cfg.dt, # 从 self.sim.cfg.dt 获取
            # --- 新增参数传递 ---
            cfg_rew_scale_orientation_deviation=self.cfg.rew_scale_orientation_deviation
        )

        if "log" not in self.extras or self.extras["log"] is None : self.extras["log"] = {}
        for key, value in reward_terms_dict.items():
            term_mean = value.mean()
            self.extras["log"][f"reward_term/{key}_step_avg"] = term_mean.item() if torch.is_tensor(term_mean) else term_mean
            if key not in self._episode_reward_terms_sum:
                self._episode_reward_terms_sum[key] = torch.zeros(self.num_envs, device=self.device)
            self._episode_reward_terms_sum[key] += value.squeeze(-1) if value.ndim > 1 and value.shape[-1] == 1 else value

        current_terminated, current_time_out = self._get_dones()
        just_failed_termination = current_terminated & (~current_time_out)
        final_reward = torch.where(
            just_failed_termination, torch.full_like(total_reward, self.cfg.rew_scale_termination), total_reward
        )
        self.extras["log"]["reward/final_reward_avg"] = final_reward.mean().item() if torch.is_tensor(final_reward) else final_reward.mean()
        return final_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        root_pos_w = self.robot.data.root_pos_w
        projected_gravity_b = self.robot.data.projected_gravity_b

        is_severely_tilted = projected_gravity_b[:, 2] > 0.0

        height_too_low_orig = root_pos_w[:, 2] < self.cfg.termination_height_thresh
        height_too_low = torch.where(
            is_severely_tilted, 
            torch.zeros_like(height_too_low_orig, dtype=torch.bool), # 确保是布尔类型
            height_too_low_orig
        )
        
        fallen_over_orig = projected_gravity_b[:, 2] > self.cfg.termination_body_z_thresh
        fallen_over = torch.where(
            is_severely_tilted,
            torch.zeros_like(fallen_over_orig, dtype=torch.bool), # 确保是布尔类型
            fallen_over_orig
        )

        base_contact_termination = torch.zeros_like(time_out, dtype=torch.bool)
        if self.cfg.termination_base_contact and self._base_body_id and len(self._base_body_id) > 0:
             if hasattr(self._contact_sensor.data, 'net_forces_w_history') and self._contact_sensor.data.net_forces_w_history is not None:
                all_forces_history = self._contact_sensor.data.net_forces_w_history
                if all_forces_history.ndim == 4 and all_forces_history.shape[1] > 0 and \
                   self._base_body_id and max(self._base_body_id) < all_forces_history.shape[2]: # 安全检查
                    current_net_forces_w = all_forces_history[:, -1, :, :]
                    forces_on_base = current_net_forces_w[:, self._base_body_id, :]
                    force_magnitudes_base = torch.norm(forces_on_base, dim=-1)
                    base_contact_termination = torch.any(force_magnitudes_base > 1.0, dim=1)
            
        terminated = height_too_low | fallen_over | base_contact_termination
        return terminated, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        eids = torch.arange(self.num_envs, device=self.device, dtype=torch.long) if env_ids is None \
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if eids.numel() == 0:
            return

        root_state_reset = self.robot.data.default_root_state[eids].clone()
        if hasattr(self._terrain, 'env_origins') and self._terrain.env_origins is not None:
             root_state_reset[:, :3] += self._terrain.env_origins[eids]
        
        initial_height_base = self.cfg.robot.init_state.pos[2] if self.cfg.robot.init_state.pos is not None and len(self.cfg.robot.init_state.pos) == 3 else 0.3
        root_state_reset[:, 2] = initial_height_base + self.cfg.reset_height_offset

        # --- Full RPY Randomization ---
        num_resets = len(eids)
        random_rolls = (torch.rand(num_resets, device=self.device) - 0.5) * 2.0 * math.pi
        random_pitches = (torch.rand(num_resets, device=self.device) - 0.5) * 2.0 * math.pi
        random_yaws = (torch.rand(num_resets, device=self.device) - 0.5) * 2.0 * math.pi

        quats_xyzw = torch.zeros(num_resets, 4, device=self.device)
        for i in range(num_resets):
            roll, pitch, yaw = random_rolls[i], random_pitches[i], random_yaws[i]
            cy = torch.cos(yaw * 0.5)
            sy = torch.sin(yaw * 0.5)
            cp = torch.cos(pitch * 0.5)
            sp = torch.sin(pitch * 0.5)
            cr = torch.cos(roll * 0.5)
            sr = torch.sin(roll * 0.5)
            # Standard ZYX Euler to Quaternion conversion (qx, qy, qz, qw)
            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy
            quats_xyzw[i, 0] = qx
            quats_xyzw[i, 1] = qy
            quats_xyzw[i, 2] = qz
            quats_xyzw[i, 3] = qw 
        root_state_reset[:, 3:7] = convert_quat(quats_xyzw, to="wxyz") # Isaac Sim uses wxyz for root state
        # --- End RPY Randomization ---
        
        root_state_reset[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state_reset, eids)
        
        # --- Joint state reset (in joint limits fully random) ---
        num_dof = self._q_lower_limits.numel() # CORRECTED
        
        random_proportions = torch.rand(len(eids), num_dof, device=self.device)
        q_lower_expanded = self._q_lower_limits.unsqueeze(0)
        q_upper_expanded = self._q_upper_limits.unsqueeze(0)
        q_range = q_upper_expanded - q_lower_expanded
        joint_pos_reset = q_lower_expanded + random_proportions * q_range
        
        zero_joint_vel = torch.zeros_like(joint_pos_reset)
        self.robot.write_joint_state_to_sim(joint_pos_reset, zero_joint_vel, env_ids=eids)
        self.robot.set_joint_position_target(joint_pos_reset, env_ids=eids)

        cmd_profile = self.cfg.command_profile
        cmd_duration_str = str(cmd_profile.get("command_mode_duration_s", "20.0"))
        if cmd_duration_str == "episode_length_s":
            cmd_duration = self.cfg.episode_length_s
        else:
            cmd_duration = float(cmd_duration_str)
            
        self._time_since_last_command_change[eids] = cmd_duration
        self._update_commands(eids)

        if hasattr(self, '_previous_policy_actions'): self._previous_policy_actions[eids] = 0.0
        if hasattr(self, '_policy_actions'): self._policy_actions[eids] = 0.0
        for key in list(self._episode_reward_terms_sum.keys()):
            if self._episode_reward_terms_sum[key].shape[0] == self.num_envs :
                self._episode_reward_terms_sum[key][eids] = 0.0