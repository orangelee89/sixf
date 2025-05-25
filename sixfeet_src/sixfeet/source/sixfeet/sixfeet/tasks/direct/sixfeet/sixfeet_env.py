# sixfeet_env.py (Corrected JIT Type Hints)
from __future__ import annotations
import torch
import math
from collections.abc import Sequence
from typing import List, Dict, Optional # <<--- Added Optional

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import (
    quat_rotate,
    quat_from_angle_axis, # <<--- 确保这一行存在
    quat_conjugate,
    convert_quat,
    euler_xyz_from_quat
)
# from isaaclab.terrains import TerrainImporter # Used via cfg

from .sixfeet_env_cfg import SixfeetEnvCfg

# ───────────────── 辅助函数 ──────────────────
@torch.jit.script
def normalize_angle_for_obs(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))

@torch.jit.script
def compute_sixfeet_rewards_directional(
    # --- Robot states (some might be used for reward even if not in obs) ---
    root_lin_vel_b: torch.Tensor, # 本体线速度，用于计算奖励，但不在策略观测中
    root_ang_vel_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    joint_pos_rel: torch.Tensor,
    joint_vel: torch.Tensor,
    applied_torque: torch.Tensor,
    joint_acc: torch.Tensor,
    q_lower_limits: torch.Tensor,
    q_upper_limits: torch.Tensor,
    current_joint_pos_abs: torch.Tensor,
    actions_from_policy: torch.Tensor, # Raw actions from policy
    previous_actions_from_policy: torch.Tensor,
    root_pos_w: torch.Tensor,

    # --- Sensor data ---
    undesired_contacts_active: torch.Tensor, # Boolean

    # --- Commands (discrete directional) ---
    # commands_discrete: (num_envs, 3) -> (cmd_fwd_bkwd, cmd_left_right, cmd_turn_lr) values from {-1, 0, 1}
    commands_discrete: torch.Tensor,

    # --- Reward weights and parameters (from cfg) ---
    cfg_cmd_profile: Dict[str, float], # Contains target speeds and probabilities
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
    cfg_toe_joint_indices: Optional[torch.Tensor], # <<--- MODIFIED: Used Optional
    cfg_rew_scale_low_height_penalty: float,
    cfg_min_height_penalty_threshold: float,
    cfg_rew_scale_undesired_contact: float,
    dt: float
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]: # <<--- MODIFIED: Return Dict of Tensors

    # --- Command Interpretation ---
    ref_ang_rate = cfg_cmd_profile["reference_angular_rate"]

    # 1. Movement in Commanded Direction Reward
    linear_vel_x_local = root_lin_vel_b[:, 0]
    linear_vel_y_local = root_lin_vel_b[:, 1]
    
    reward_fwd_bkwd_move = commands_discrete[:, 0] * linear_vel_x_local
    reward_left_right_move = commands_discrete[:, 1] * linear_vel_y_local
    
    is_linear_cmd_active = (torch.abs(commands_discrete[:, 0]) > 0.5) | (torch.abs(commands_discrete[:, 1]) > 0.5)
    reward_linear_direction = (reward_fwd_bkwd_move + reward_left_right_move) * is_linear_cmd_active.float()
    reward_move_in_commanded_direction = reward_linear_direction * cfg_rew_scale_move_in_commanded_direction

    angular_vel_z_local = root_ang_vel_b[:, 2]
    reward_angular_direction_raw = -commands_discrete[:, 2] * angular_vel_z_local # Assume cmd +1 is R turn, omega_z + is L turn
    
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
    penalty_flat_orientation = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1) * cfg_rew_scale_flat_orientation

    is_stand_cmd_active = torch.all(commands_discrete == 0, dim=1)
    unwanted_lin_vel_sq = torch.sum(torch.square(root_lin_vel_b[:, :2]), dim=1)
    unwanted_ang_vel_sq = torch.square(root_ang_vel_b[:, 2])
    penalty_unwanted_movement = (unwanted_lin_vel_sq + unwanted_ang_vel_sq) * \
                                is_stand_cmd_active.float() * cfg_rew_scale_unwanted_movement_penalty

    # 6. Constraint Penalties
    dof_range = q_upper_limits - q_lower_limits
    dof_range = torch.where(dof_range < 1e-6, torch.ones_like(dof_range), dof_range)
    # Ensure current_joint_pos_abs is (num_envs, num_dof) and q_lower_limits is (num_dof)
    q_lower_expanded = q_lower_limits.unsqueeze(0) if q_lower_limits.ndim == 1 else q_lower_limits
    dof_range_expanded = dof_range.unsqueeze(0) if dof_range.ndim == 1 else dof_range
    dof_pos_scaled_01 = (current_joint_pos_abs - q_lower_expanded) / dof_range_expanded
    
    near_lower_limit = torch.relu(0.05 - dof_pos_scaled_01)**2
    near_upper_limit = torch.relu(dof_pos_scaled_01 - 0.95)**2
    penalty_dof_at_limit = torch.sum(near_lower_limit + near_upper_limit, dim=-1) * cfg_rew_scale_dof_at_limit

    penalty_toe_orientation = torch.zeros_like(commands_discrete[:,0], device=commands_discrete.device)
    if cfg_rew_scale_toe_orientation_penalty != 0.0 and cfg_toe_joint_indices is not None:
        # Ensure cfg_toe_joint_indices is not empty before indexing
        if cfg_toe_joint_indices.numel() > 0:
            toe_joint_positions = current_joint_pos_abs[:, cfg_toe_joint_indices]
            penalty_toe_orientation = torch.sum(torch.relu(toe_joint_positions)**2, dim=-1) * cfg_rew_scale_toe_orientation_penalty
    
    is_too_low = (current_height_z < cfg_min_height_penalty_threshold).float()
    penalty_low_height = is_too_low * cfg_rew_scale_low_height_penalty

    penalty_undesired_contact = undesired_contacts_active.float() * cfg_rew_scale_undesired_contact
    
    total_reward = (
        reward_move_in_commanded_direction + reward_turn + reward_alive + reward_target_height
        + (penalty_action_cost + penalty_action_rate + penalty_joint_torques + penalty_joint_accel
        + penalty_lin_vel_z + penalty_ang_vel_xy + penalty_flat_orientation + penalty_unwanted_movement
        + penalty_dof_at_limit + penalty_toe_orientation + penalty_low_height
        + penalty_undesired_contact) * dt
    )
    
    # Ensure all dictionary values are Tensors for Dict[str, torch.Tensor]
    reward_terms: Dict[str, torch.Tensor] = { # Explicit type for the dict literal
        "move_in_commanded_direction": reward_move_in_commanded_direction,
        "turn_reward_combined": reward_turn,
        "alive": reward_alive,
        "target_height": reward_target_height,
        "action_cost_penalty": penalty_action_cost * dt,
        "action_rate_penalty": penalty_action_rate * dt,
        "joint_torques_penalty": penalty_joint_torques * dt,
        "joint_accel_penalty": penalty_joint_accel * dt,
        "lin_vel_z_penalty": penalty_lin_vel_z * dt,
        "ang_vel_xy_penalty": penalty_ang_vel_xy * dt,
        "flat_orientation_penalty": penalty_flat_orientation * dt,
        "unwanted_movement_penalty": penalty_unwanted_movement * dt,
        "dof_at_limit_penalty": penalty_dof_at_limit * dt,
        "toe_orientation_penalty": penalty_toe_orientation, # Already scaled if active
        "low_height_penalty": penalty_low_height * dt, # This was penalty_low_height before * dt, ensure consistency. Let's assume it should be scaled by dt if it's a penalty rate.
        "undesired_contact_penalty": penalty_undesired_contact * dt,
    }
    # Correction for penalty_toe_orientation, it's already scaled by its weight.
    # If other penalties are rates, they are scaled by dt. If they are "costs per step", then no dt.
    # Given Anymal-C scales most, let's assume it's fine. Re-evaluating:
    # penalty_toe_orientation term already has its scale cfg_rew_scale_toe_orientation_penalty
    # If it's a cost per step due to bad orientation, it's fine without dt.
    # If it's a continuous penalty over time, then * dt.
    # Let's be consistent: if the `cfg_rew_scale_` implies a per-step cost, then dt might not be needed for all.
    # However, to match Anymal-C's style which seems to scale most penalties by dt:
    reward_terms["toe_orientation_penalty"] = penalty_toe_orientation * dt # If applying dt scaling consistently

    return total_reward, reward_terms


class SixfeetEnv(DirectRLEnv):
    cfg: SixfeetEnvCfg
    _contact_sensor: ContactSensor

    def __init__(self, cfg: SixfeetEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._default_joint_pos = self.robot.data.default_joint_pos.clone()
        # Ensure _default_joint_pos is (num_dof) or (1, num_dof) for broadcasting
        if self._default_joint_pos.ndim > 1 and self._default_joint_pos.shape[0] == self.num_envs:
            self._default_joint_pos = self._default_joint_pos[0] # Take from first env, assume same for all defaults
        
        joint_limits = self.robot.data.joint_pos_limits[0].to(self.device)
        self._q_lower_limits = joint_limits[:, 0]
        self._q_upper_limits = joint_limits[:, 1]
        
        self._policy_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_policy_actions = torch.zeros_like(self._policy_actions)
        self._processed_actions = torch.zeros_like(self._policy_actions)

        self._commands = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self._time_since_last_command_change = torch.zeros(self.num_envs, device=self.device)
        
        self._resolve_toe_joint_indices()

        self._undesired_contact_body_ids: Optional[List[int]] = None # Use Optional
        if self.cfg.undesired_contact_link_names_expr:
            indices, names = self._contact_sensor.find_bodies(self.cfg.undesired_contact_link_names_expr)
            if indices: # Ensure indices list is not empty
                self._undesired_contact_body_ids = indices
                print(f"[INFO] SixfeetEnv: Undesired contact body IDs: {self._undesired_contact_body_ids} for names {names}")
            else:
                print(f"[WARNING] SixfeetEnv: No bodies for undesired contact expr: {self.cfg.undesired_contact_link_names_expr}")

        self._base_body_id: Optional[List[int]] = None # Use Optional
        if self.cfg.termination_base_contact and self.cfg.base_link_name:
            indices, names = self._contact_sensor.find_bodies(self.cfg.base_link_name)
            if indices: # Ensure indices list is not empty
                self._base_body_id = indices
                print(f"[INFO] SixfeetEnv: Base body ID for termination: {self._base_body_id} for names {names}")
            else:
                print(f"[WARNING] SixfeetEnv: No body for base contact termination: {self.cfg.base_link_name}")
        
        self._episode_reward_terms_sum: Dict[str, torch.Tensor] = {} # Ensure it's a Dict of Tensors

    def _resolve_toe_joint_indices(self):
        from typing import Optional # 确保 Optional 已导入

        self._toe_joint_indices: Optional[torch.Tensor] = None 
        expr_or_list = getattr(self.cfg, 'toe_joint_names_expr', None)
        joint_names_list_for_logging = [] # 用于日志记录

        if isinstance(expr_or_list, str):
            joint_indices_list, joint_names_list_for_logging = self.robot.find_joints(expr_or_list)
            if joint_indices_list: 
                self._toe_joint_indices = torch.tensor(joint_indices_list, device=self.device, dtype=torch.long)
                # print(f"[INFO] SixfeetEnv: Found toe joint indices via regex: {self._toe_joint_indices.tolist()}, names: {joint_names_list_for_logging}")
            # else:
            #     print(f"[WARNING] SixfeetEnv: No toe joints found using regex: '{expr_or_list}'.")
        elif isinstance(expr_or_list, list) and all(isinstance(i, int) for i in expr_or_list):
            if expr_or_list: 
                temp_indices = torch.tensor(expr_or_list, device=self.device, dtype=torch.long)
                # !! 使用 self.robot.num_dof !!
                if torch.any(temp_indices < 0) or torch.any(temp_indices >= self.robot.num_dof): # <<--- 修改在这里
                    print(f"[ERROR] SixfeetEnv: Invalid toe joint indices in list: {expr_or_list}. Max allowable index: {self.robot.num_dof - 1}")
                    self._toe_joint_indices = None # 标记为无效
                else:
                    self._toe_joint_indices = temp_indices
                    # 获取名称用于日志 (如果需要)
                    # joint_names_list_for_logging = [self.robot.data.joint_names[i] for i in temp_indices.tolist()]
        elif expr_or_list is not None:
            print(f"[WARNING] SixfeetEnv: 'toe_joint_names_expr' ('{expr_or_list}') in cfg has invalid type: {type(expr_or_list)}. Expected str or list[int].")

        # 统一的验证和日志打印
        if self._toe_joint_indices is not None:
            if self._toe_joint_indices.numel() == 0: 
                print(f"[WARNING] SixfeetEnv: Resolved toe joint indices tensor is empty from '{expr_or_list}'.")
                self._toe_joint_indices = None
            # 再次验证索引范围，确保之前的逻辑（如果设置了None）在这里也能被正确处理
            elif self.robot is not None and hasattr(self.robot, 'num_dof') and \
                 (torch.any(self._toe_joint_indices < 0) or torch.any(self._toe_joint_indices >= self.robot.num_dof)):
                print(f"[ERROR] SixfeetEnv: Invalid toe joint indices after processing: {self._toe_joint_indices.tolist()}. Max allowable index: {self.robot.num_dof - 1}")
                self._toe_joint_indices = None
            else:
                # 只有在所有检查通过后才打印最终有效的索引
                # （注意：joint_names_list_for_logging 可能只在 regex 分支被赋值）
                print(f"[INFO] SixfeetEnv: Validated toe joint indices for penalty: {self._toe_joint_indices.tolist()}")
        
        if self._toe_joint_indices is None and expr_or_list is not None:
             print(f"[WARNING] SixfeetEnv: No valid toe joint indices resolved from '{expr_or_list}'. Toe orientation penalty might not apply effectively.")
        elif self._toe_joint_indices is None and expr_or_list is None:
             print(f"[INFO] SixfeetEnv: 'toe_joint_names_expr' not specified. Toe orientation penalty will not be applied.")

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        
        if hasattr(self.cfg, "terrain") and self.cfg.terrain is not None:
            if hasattr(self.scene, "cfg"): # Should exist for InteractiveScene
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
            from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
            spawn_ground_plane("/World/ground", GroundPlaneCfg())
            class DummyTerrain:
                def __init__(self, num_envs, device): self.env_origins = torch.zeros((num_envs, 3), device=device)
            self._terrain = DummyTerrain(self.cfg.scene.num_envs, self.device) # type: ignore

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.scene.clone_environments(copy_from_source=False)

    def _update_commands(self, env_ids: torch.Tensor):
        # 使用 self.sim.dt 来获取仿真步长
        self._time_since_last_command_change[env_ids] += self.physics_dt 
        
        change_command_mask = self._time_since_last_command_change[env_ids] >= self.cfg.command_profile["command_mode_duration_s"]
        envs_to_change = env_ids[change_command_mask]

        if envs_to_change.numel() > 0:
            self._time_since_last_command_change[envs_to_change] = 0.0
            num_to_change = envs_to_change.shape[0]
            
            # Sample new command modes
            # 0: stand, 1: fwd, 2: bkwd, 3: left, 4: right, 5: turn_L, 6: turn_R
            command_modes = torch.randint(0, self.cfg.command_profile["num_command_modes"], (num_to_change,), device=self.device)
            
            new_commands_for_changed_envs = torch.zeros(num_to_change, 3, device=self.device, dtype=torch.float)

            # Stand still with higher probability
            stand_mask = torch.rand(num_to_change, device=self.device) < self.cfg.command_profile["stand_still_prob"]
            command_modes[stand_mask] = 0 # Force stand still for these

            new_commands_for_changed_envs[command_modes == 1, 0] = 1.0  # Forward (X+)
            new_commands_for_changed_envs[command_modes == 2, 0] = -1.0 # Backward (X-)
            new_commands_for_changed_envs[command_modes == 3, 1] = -1.0 # Left Strafe (Y-)
            new_commands_for_changed_envs[command_modes == 4, 1] = 1.0  # Right Strafe (Y+)
            new_commands_for_changed_envs[command_modes == 5, 2] = -1.0 # Turn Left (Yaw-)
            new_commands_for_changed_envs[command_modes == 6, 2] = 1.0  # Turn Right (Yaw+)
            
            self._commands[envs_to_change] = new_commands_for_changed_envs

    def _pre_physics_step(self, actions: torch.Tensor):
        self._policy_actions = actions.clone().to(self.device)
        # _default_joint_pos is (num_dof), need to expand for broadcasting with (num_envs, num_dof) actions
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

    def _apply_action(self):
        self.robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_policy_actions = self._policy_actions.clone()
        
        # default_pos_expanded = self._default_joint_pos.unsqueeze(0)
        # joint_pos_rel = self.robot.data.joint_pos - default_pos_expanded
        joint_pos_rel = torch.zeros_like(self.robot.data.joint_pos)
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
            print(f"Details: proj_grav({self.robot.data.projected_gravity_b.shape}), ang_vel({self.robot.data.root_ang_vel_b.shape}), cmds({self._commands.shape}), jpos_rel({joint_pos_rel.shape}), jvel({self.robot.data.joint_vel.shape})")


        return {"policy": observations_tensor}

    def _get_rewards(self) -> torch.Tensor:
        # Get necessary states (including root_lin_vel_b for reward calculation only)
        root_lin_vel_b = self.robot.data.root_lin_vel_b # Used for reward, not obs
        root_ang_vel_b = self.robot.data.root_ang_vel_b
        projected_gravity_b = self.robot.data.projected_gravity_b
        
        # Ensure default_joint_pos is correctly shaped for broadcasting
        if self._default_joint_pos.ndim == 1:
             default_pos_expanded = self._default_joint_pos.unsqueeze(0)
        else: # Should be (1, num_dof) or already (num_envs, num_dof)
             default_pos_expanded = self._default_joint_pos

        joint_pos_rel = self.robot.data.joint_pos - default_pos_expanded
        current_joint_pos_abs = self.robot.data.joint_pos
        
        joint_vel = self.robot.data.joint_vel
        applied_torque = self.robot.data.applied_torque
        joint_acc = getattr(self.robot.data, "joint_acc", torch.zeros_like(joint_vel))
        root_pos_w = self.robot.data.root_pos_w

        # --- undesired_contacts_active 计算修改开始 ---
        undesired_contacts_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self._undesired_contact_body_ids and len(self._undesired_contact_body_ids) > 0 :
            if hasattr(self._contact_sensor.data, 'net_forces_w_history'):
                all_forces_history = self._contact_sensor.data.net_forces_w_history
                # 确保历史数据有效且包含足够的body信息
                if all_forces_history is not None and all_forces_history.ndim == 4 and all_forces_history.shape[1] > 0 and \
                   all_forces_history.shape[2] > max(self._undesired_contact_body_ids):
                    current_net_forces_w = all_forces_history[:, -1, :, :] # 获取当前时间步的力
                    forces_on_undesired_bodies = current_net_forces_w[:, self._undesired_contact_body_ids, :]
                    force_magnitudes = torch.norm(forces_on_undesired_bodies, dim=-1)
                    undesired_contacts_active = torch.any(force_magnitudes > 1.0, dim=1)
            # else:
            #     # 如果需要，可以在这里添加警告或日志
            #     # print_once("[WARNING] ContactSensor data does not have 'net_forces_w_history' for undesired contacts in _get_rewards.")
            #     pass
        # --- undesired_contacts_active 计算修改结束 ---
        
        total_reward, reward_terms_dict = compute_sixfeet_rewards_directional(
            root_lin_vel_b, root_ang_vel_b, projected_gravity_b,
            joint_pos_rel, joint_vel, applied_torque, joint_acc,
            self._q_lower_limits, self._q_upper_limits, current_joint_pos_abs,
            self._policy_actions, self._previous_policy_actions,
            root_pos_w,
            undesired_contacts_active,
            self._commands,
            self.cfg.command_profile,
            self.cfg.rew_scale_move_in_commanded_direction,
            self.cfg.rew_scale_achieve_reference_angular_rate,
            self.cfg.rew_scale_alive, self.cfg.rew_scale_target_height, self.cfg.target_height_m,
            self.cfg.rew_scale_action_cost, self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_joint_torques, self.cfg.rew_scale_joint_accel,
            self.cfg.rew_scale_lin_vel_z_penalty, self.cfg.rew_scale_ang_vel_xy_penalty,
            self.cfg.rew_scale_flat_orientation, self.cfg.rew_scale_unwanted_movement_penalty,
            self.cfg.rew_scale_dof_at_limit,
            self.cfg.rew_scale_toe_orientation_penalty, self._toe_joint_indices,
            self.cfg.rew_scale_low_height_penalty, self.cfg.min_height_penalty_threshold,
            self.cfg.rew_scale_undesired_contact,
            self.sim.cfg.dt # <<--- 使用 self.sim.cfg.dt 获取物理步长
        )

        if "log" not in self.extras or self.extras["log"] is None : self.extras["log"] = {}
        for key, value in reward_terms_dict.items():
            term_mean = value.mean()
            # 确保记录的是Python标量
            self.extras["log"][f"reward_term/{key}_step_avg"] = term_mean.item() if torch.is_tensor(term_mean) else term_mean
            if key not in self._episode_reward_terms_sum:
                self._episode_reward_terms_sum[key] = torch.zeros(self.num_envs, device=self.device)
            self._episode_reward_terms_sum[key] += value.squeeze()


        current_terminated, current_time_out = self._get_dones()
        just_failed_termination = current_terminated & (~current_time_out)
        
        final_reward = torch.where(
            just_failed_termination,
            torch.full_like(total_reward, self.cfg.rew_scale_termination),
            total_reward
        )
        # 确保记录的是Python标量
        self.extras["log"]["reward/final_reward_avg"] = final_reward.mean().item() if torch.is_tensor(final_reward) else final_reward.mean()
        return final_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        root_pos_w = self.robot.data.root_pos_w
        height_too_low = root_pos_w[:, 2] < self.cfg.termination_height_thresh
        
        projected_gravity_z = self.robot.data.projected_gravity_b[:, 2]
        fallen_over = projected_gravity_z > self.cfg.termination_body_z_thresh

        base_contact_termination = torch.zeros_like(time_out, dtype=torch.bool)
        if self.cfg.termination_base_contact and self._base_body_id and len(self._base_body_id) > 0:
            # --- base_contact_termination 计算修改开始 ---
            if hasattr(self._contact_sensor.data, 'net_forces_w_history'):
                all_forces_history = self._contact_sensor.data.net_forces_w_history
                # 确保历史数据有效且包含足够的body信息
                if all_forces_history is not None and all_forces_history.ndim == 4 and all_forces_history.shape[1] > 0 and \
                   all_forces_history.shape[2] > max(self._base_body_id): # 确保 body 索引有效
                    current_net_forces_w = all_forces_history[:, -1, :, :] # 获取当前时间步的力
                    forces_on_base = current_net_forces_w[:, self._base_body_id, :]
                    force_magnitudes_base = torch.norm(forces_on_base, dim=-1)
                    base_contact_termination = torch.any(force_magnitudes_base > 1.0, dim=1)
            # else:
            #     # print_once("[WARNING] ContactSensor data does not have 'net_forces_w_history' for base contact in _get_dones.")
            #     pass
            # --- base_contact_termination 计算修改结束 ---
            
        terminated = height_too_low | fallen_over | base_contact_termination
        # terminated = height_too_low | fallen_over 
        # if torch.any(terminated):
        #     step_idx = int(self.episode_length_buf.max()) 
        #     print(
        #         f"[DEBUG _get_dones]  step={step_idx} | "
        #         f"height_too_low={height_too_low.nonzero(as_tuple=True)[0].numel()}  "
        #         f"fallen_over={fallen_over.nonzero(as_tuple=True)[0].numel()}  "
        #         f"base_contact={base_contact_termination.nonzero(as_tuple=True)[0].numel()}"
        #     )
        return terminated, time_out
    
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        # 1) 通用父类逻辑 ----------------------------------------------------
        super()._reset_idx(env_ids)
        # print("task reseted env_ids:", env_ids)
        eids = torch.arange(self.num_envs, device=self.device) if env_ids is None \
            else torch.as_tensor(env_ids, device=self.device)
        if eids.numel() == 0:
            return

        # 2) root 位置与姿态 --------------------------------------------------
        root_state = self.robot.data.default_root_state[eids].clone()

        # 加地形起点
        if hasattr(self._terrain, "env_origins"):
            root_state[:, :3] += self._terrain.env_origins[eids]

        # 高度 + 偏移
        h0 = self.cfg.robot.init_state.pos[2] if self.cfg.robot.init_state.pos is not None else 0.3
        root_state[:, 2] = h0 + self.cfg.reset_height_offset

        # 随机朝向
        yaw = (torch.rand(len(eids), device=self.device) - 0.5) * 2.0 * self.cfg.root_orientation_yaw_range
        root_state[:, 3:7] = quat_from_angle_axis(yaw, torch.tensor([0., 0., 1.], device=self.device))
        root_state[:, 7:] = 0.0                       # 线 / 角速度 = 0

        self.robot.write_root_state_to_sim(root_state, eids)

        # 3) 随机关节角 -------------------------------------------------------
        n_dof   = self._q_lower_limits.numel()
        rand_p  = torch.rand(len(eids), n_dof, device=self.device)
        q_range = self._q_upper_limits - self._q_lower_limits
        rand_q  = self._q_lower_limits + rand_p * q_range         # 均匀分布
        # ---- 如想“围绕 init_state 抖动”可改成： ----
        # center = self.robot.cfg.init_state.joint_pos_tensor      # (dof,)
        # sigma  = torch.deg2rad(torch.tensor(5.0, device=self.device))
        # noise  = torch.randn(len(eids), n_dof, device=self.device) * sigma
        # rand_q = torch.clamp(center + noise, self._q_lower_limits, self._q_upper_limits)

        zero_qv = torch.zeros_like(rand_q)
        self.robot.write_joint_state_to_sim(rand_q, zero_qv, env_ids=eids)

        # **同步 PD 目标 = 当前姿态**  ← 关键一行
        self.robot.set_joint_position_target(rand_q, env_ids=eids)

        # 4) 其余缓存 ----------------------------------------------------------
        self._time_since_last_command_change[eids] = self.cfg.command_profile["command_mode_duration_s"]
        self._update_commands(eids)

        if self.cfg.rew_scale_action_rate < 0 and hasattr(self, "previous_actions"):
            self.previous_actions[eids] = 0.0

        # 清 episodic 累加器（如有）
        for k in self._episode_reward_terms_sum.keys():
            self._episode_reward_terms_sum[k][eids] = 0.0

    # def _reset_idx(self, env_ids: Sequence[int] | None):
    #     # Call parent first to handle episode_length_buf, reset_buf, etc.
    #     super()._reset_idx(env_ids) 

    #     if env_ids is None: # If None, superclass might have reset all, or we operate on all
    #         eids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    #     else:
    #         eids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        
    #     if eids.numel() == 0: # No environments to reset specifically by this function call
    #         return

    #     # Reset robot state
    #     root_state_reset = self.robot.data.default_root_state[eids].clone()
    #     if hasattr(self._terrain, 'env_origins') and self._terrain.env_origins is not None:
    #          root_state_reset[:, :3] += self._terrain.env_origins[eids]
        
    #     # Use initial height from cfg.robot.init_state.pos[2] + reset_height_offset
    #     initial_height = self.cfg.robot.init_state.pos[2] if self.cfg.robot.init_state.pos is not None else 0.3 # Fallback height
    #     root_state_reset[:, 2] = initial_height + self.cfg.reset_height_offset

    #     random_yaw = (torch.rand(eids.shape[0], device=self.device) - 0.5) * 2.0 * self.cfg.root_orientation_yaw_range
    #     world_z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=random_yaw.dtype)
    #     orientation_quat_xyzw = quat_from_angle_axis(random_yaw, world_z_axis)
    #     root_state_reset[:, 3:7] = orientation_quat_xyzw
    #     root_state_reset[:, 7:] = 0.0 #lin_vel, ang_vel
        
    #     self.robot.write_root_state_to_sim(root_state_reset, eids)
        
    #     default_pos_expanded = self._default_joint_pos.unsqueeze(0).expand(eids.shape[0], -1)
    #     zero_vel = torch.zeros_like(default_pos_expanded)
    #     self.robot.write_joint_state_to_sim(
    #     default_pos_expanded,  # 使用默认位置，不是当前位置
    #     zero_vel,              # 明确设置零速度
    #     env_ids=eids
    #     )

    #     self._time_since_last_command_change[eids] = self.cfg.command_profile["command_mode_duration_s"] # Force immediate change
    #     self._update_commands(eids)

    #     if hasattr(self, '_previous_policy_actions'): self._previous_policy_actions[eids] = 0.0
    #     if hasattr(self, '_policy_actions'): self._policy_actions[eids] = 0.0

    #     # Logging episodic sums for environments that just reset
    #     # This should ideally use the reset_buf from before it's cleared by super()._reset_idx
    #     # For now, log the average of what was summed for the reset envs and then clear.
    #     if "log" not in self.extras or self.extras["log"] is None : self.extras["log"] = {}
    #     for key in list(self._episode_reward_terms_sum.keys()): # Iterate over a copy of keys
    #         if eids.numel() > 0 and self._episode_reward_terms_sum[key].shape[0] == self.num_envs :
    #             # This logs sum for envs that are resetting now.
    #             # A better approach is to log sums for envs that *were* terminated in the previous step.
    #             # This requires more careful state management of `reset_buf`.
    #             # For simplicity, this will show the sum accumulated *until reset* for these envs.
    #             # self.extras["log"][f"EpisodeSum/{key}_on_reset"] = self._episode_reward_terms_sum[key][eids].mean().item()
    #             self._episode_reward_terms_sum[key][eids] = 0.0 # Clear sums for reset envs