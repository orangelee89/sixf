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
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
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

    # --- Sensor data ---
    undesired_contacts_active: torch.Tensor, # Boolean

    # --- Commands (discrete directional) ---
    commands_discrete: torch.Tensor,

    # --- Reward weights and parameters (from cfg) ---
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
    dt: float
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # --- Command Interpretation (used for some rewards, even if scales are 0) ---
    # Accessing dict items, ensure they exist or use .get() with defaults if they might be missing
    ref_ang_rate = cfg_cmd_profile.get("reference_angular_rate", 0.0)
    # ref_lin_speed = cfg_cmd_profile.get("reference_linear_speed", 0.0) # 如果之后用到

    # 1. Movement in Commanded Direction Reward (will be 0 if scale in cfg is 0)
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
    height_check = torch.clamp(current_height_z / cfg_target_height_m, max=1.1) # Clip to avoid extreme rewards if much higher
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
    
    # Flat orientation penalty (already includes its scale)
    penalty_flat_orientation = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1) * cfg_rew_scale_flat_orientation

    # Unwanted movement penalty (when command is stand still)
    is_stand_cmd_active = torch.all(commands_discrete == 0, dim=1) # True if cmd is [0,0,0]
    unwanted_lin_vel_sq = torch.sum(torch.square(root_lin_vel_b[:, :2]), dim=1) # Unwanted planar linear velocity
    unwanted_ang_vel_sq = torch.square(root_ang_vel_b[:, 2]) # Unwanted yaw angular velocity
    penalty_unwanted_movement = (unwanted_lin_vel_sq + unwanted_ang_vel_sq) * \
                                is_stand_cmd_active.float() * cfg_rew_scale_unwanted_movement_penalty

    # 6. Constraint Penalties
    dof_range = q_upper_limits - q_lower_limits
    dof_range = torch.where(dof_range < 1e-6, torch.ones_like(dof_range), dof_range) # Avoid division by zero
    q_lower_expanded = q_lower_limits.unsqueeze(0) if q_lower_limits.ndim == 1 else q_lower_limits
    dof_range_expanded = dof_range.unsqueeze(0) if dof_range.ndim == 1 else dof_range
    dof_pos_scaled_01 = (current_joint_pos_abs - q_lower_expanded) / dof_range_expanded
    near_lower_limit = torch.relu(0.05 - dof_pos_scaled_01)**2 # Penalize being within 5% of lower limit
    near_upper_limit = torch.relu(dof_pos_scaled_01 - 0.95)**2 # Penalize being within 5% of upper limit
    penalty_dof_at_limit = torch.sum(near_lower_limit + near_upper_limit, dim=-1) * cfg_rew_scale_dof_at_limit

    penalty_toe_orientation = torch.zeros_like(commands_discrete[:,0], device=commands_discrete.device)
    if cfg_rew_scale_toe_orientation_penalty != 0.0 and cfg_toe_joint_indices is not None:
        if cfg_toe_joint_indices.numel() > 0: # Ensure indices are not empty
            toe_joint_positions = current_joint_pos_abs[:, cfg_toe_joint_indices]
            # Example: Penalize positive toe joint angles (if positive means toe pointing down into ground)
            # This depends on your specific toe joint definition.
            # The original code penalized torch.relu(toe_joint_positions)**2.
            # Adjust as needed e.g. torch.abs(toe_joint_positions) or (toe_joint_positions - desired_angle)**2
            penalty_toe_orientation = torch.sum(torch.relu(toe_joint_positions)**2, dim=-1) * cfg_rew_scale_toe_orientation_penalty
    
    is_too_low = (current_height_z < cfg_min_height_penalty_threshold).float()
    penalty_low_height = is_too_low * cfg_rew_scale_low_height_penalty # Already includes scale

    penalty_undesired_contact = undesired_contacts_active.float() * cfg_rew_scale_undesired_contact # Already includes scale
    
    # Total reward calculation:
    # penalty_flat_orientation is added directly (not scaled by dt here again).
    # Other terms like reward_alive, reward_target_height are also direct additions.
    # A group of penalties is scaled by dt.
    total_reward = (
        reward_move_in_commanded_direction + reward_turn + reward_alive + reward_target_height + penalty_flat_orientation
        + (penalty_action_cost + penalty_action_rate + penalty_joint_torques + penalty_joint_accel
        + penalty_lin_vel_z + penalty_ang_vel_xy + penalty_unwanted_movement # penalty_flat_orientation was removed from this dt-scaled group
        + penalty_dof_at_limit + penalty_toe_orientation + penalty_low_height
        + penalty_undesired_contact) * dt
    )
    
    reward_terms: Dict[str, torch.Tensor] = {
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
        "flat_orientation_penalty": penalty_flat_orientation, # Log the direct scaled term
        "unwanted_movement_penalty": penalty_unwanted_movement * dt,
        "dof_at_limit_penalty": penalty_dof_at_limit * dt,
        "toe_orientation_penalty": penalty_toe_orientation * dt, 
        "low_height_penalty": penalty_low_height * dt, # Note: penalty_low_height var already has its scale
        "undesired_contact_penalty": penalty_undesired_contact * dt, # Note: penalty_undesired_contact var already has its scale
    }
    # If penalty_low_height and penalty_undesired_contact should NOT be scaled by dt again,
    # they should be moved out of the dt block in total_reward and logged directly in reward_terms.
    # For now, keeping structure similar to original user file for these, but this is a point of attention.
    # If their scales in CFG are meant as direct per-step penalties, then the *dt here is a further reduction.

    return total_reward, reward_terms


class SixfeetEnv(DirectRLEnv):
    cfg: SixfeetEnvCfg
    _contact_sensor: ContactSensor # Defines the type of _contact_sensor

    def __init__(self, cfg: SixfeetEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Default joint positions from robot articulation data
        self._default_joint_pos = self.robot.data.default_joint_pos.clone()
        # Ensure _default_joint_pos is (num_dof) or (1, num_dof) for broadcasting
        if self._default_joint_pos.ndim > 1 and self._default_joint_pos.shape[0] == self.num_envs:
            self._default_joint_pos = self._default_joint_pos[0] # Take from first env, assume same for all defaults
        
        # Joint limits
        joint_limits = self.robot.data.joint_pos_limits[0].to(self.device) # Assuming limits are same across all envs
        self._q_lower_limits = joint_limits[:, 0]
        self._q_upper_limits = joint_limits[:, 1]
        
        # Action buffers
        self._policy_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_policy_actions = torch.zeros_like(self._policy_actions)
        self._processed_actions = torch.zeros_like(self._policy_actions)

        # Command buffers
        self._commands = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float) # (cmd_fwd_bkwd, cmd_left_right, cmd_turn_lr)
        self._time_since_last_command_change = torch.zeros(self.num_envs, device=self.device)
        
        # Resolve joint indices for specific penalties if needed
        self._resolve_toe_joint_indices()

        # Resolve body indices for contact penalties/terminations
        self._undesired_contact_body_ids: Optional[List[int]] = None
        if self.cfg.undesired_contact_link_names_expr and self.cfg.rew_scale_undesired_contact != 0.0:
            indices, names = self._contact_sensor.find_bodies(self.cfg.undesired_contact_link_names_expr)
            if indices:
                self._undesired_contact_body_ids = indices
                print(f"[INFO] SixfeetEnv: Undesired contact body IDs: {self._undesired_contact_body_ids} for names {names}")
            else:
                print(f"[WARNING] SixfeetEnv: No bodies found for undesired contact expr: {self.cfg.undesired_contact_link_names_expr}")

        self._base_body_id: Optional[List[int]] = None
        if self.cfg.termination_base_contact and self.cfg.base_link_name:
            indices, names = self._contact_sensor.find_bodies(self.cfg.base_link_name)
            if indices:
                self._base_body_id = indices
                print(f"[INFO] SixfeetEnv: Base body ID for termination: {self._base_body_id} for names {names}")
            else:
                print(f"[WARNING] SixfeetEnv: No body found for base contact termination: {self.cfg.base_link_name}")
        
        # For logging episodic reward term sums
        self._episode_reward_terms_sum: Dict[str, torch.Tensor] = {}

    def _resolve_toe_joint_indices(self):
        # from typing import Optional # Already imported at the top
        self._toe_joint_indices: Optional[torch.Tensor] = None
        
        # Only resolve if the penalty scale is non-zero and expression is provided
        expr_or_list = getattr(self.cfg, 'toe_joint_names_expr', None)
        if self.cfg.rew_scale_toe_orientation_penalty == 0.0 or not expr_or_list:
            if expr_or_list and self.cfg.rew_scale_toe_orientation_penalty != 0.0: # Should not happen if logic is or
                 print(f"[INFO] SixfeetEnv: Toe orientation penalty relevant but expr_or_list is '{expr_or_list}'. No indices resolved.")
            # else: # Be less verbose if penalty is zero or no expression
            # print(f"[INFO] SixfeetEnv: Toe orientation penalty disabled or expr not provided. Skipping toe joint index resolution.")
            return

        joint_names_list_for_logging = [] # For logging names if found by regex

        if isinstance(expr_or_list, str):
            joint_indices_list, joint_names_list_for_logging = self.robot.find_joints(expr_or_list)
            if joint_indices_list:
                self._toe_joint_indices = torch.tensor(joint_indices_list, device=self.device, dtype=torch.long)
        elif isinstance(expr_or_list, list) and all(isinstance(i, int) for i in expr_or_list):
            if expr_or_list: # Ensure list is not empty
                temp_indices = torch.tensor(expr_or_list, device=self.device, dtype=torch.long)
                if torch.any(temp_indices < 0) or torch.any(temp_indices >= self.robot.num_dof):
                    print(f"[ERROR] SixfeetEnv: Invalid toe joint indices in list: {expr_or_list}. Max allowable index: {self.robot.num_dof - 1}")
                    self._toe_joint_indices = None # Mark as invalid
                else:
                    self._toe_joint_indices = temp_indices
        elif expr_or_list is not None: # It's not str, not list[int], but not None
            print(f"[WARNING] SixfeetEnv: 'toe_joint_names_expr' ('{expr_or_list}') in cfg has invalid type: {type(expr_or_list)}. Expected str or list[int].")

        # Unified validation and logging
        if self._toe_joint_indices is not None:
            if self._toe_joint_indices.numel() == 0:
                # print(f"[WARNING] SixfeetEnv: Resolved toe joint indices tensor is empty from '{expr_or_list}'.")
                self._toe_joint_indices = None
            elif hasattr(self.robot, 'num_dof') and \
                 (torch.any(self._toe_joint_indices < 0) or torch.any(self._toe_joint_indices >= self.robot.num_dof)):
                print(f"[ERROR] SixfeetEnv: Invalid toe joint indices after processing: {self._toe_joint_indices.tolist()}. Max allowable index: {self.robot.num_dof - 1}")
                self._toe_joint_indices = None
            else:
                # Log only if relevant and successfully resolved
                log_msg = f"[INFO] SixfeetEnv: Validated toe joint indices for penalty: {self._toe_joint_indices.tolist()}"
                if joint_names_list_for_logging: # Add names if available
                    log_msg += f", names: {joint_names_list_for_logging}"
                print(log_msg)
        
        if self._toe_joint_indices is None and expr_or_list is not None: # If an expression was given but failed
             print(f"[WARNING] SixfeetEnv: No valid toe joint indices resolved from '{expr_or_list}'. Toe orientation penalty might not apply effectively.")
        # No message if expr_or_list was None from the start and penalty was 0.

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        
        if hasattr(self.cfg, "terrain") and self.cfg.terrain is not None:
            if hasattr(self.scene, "cfg"): # Should exist for InteractiveScene
                # These attributes might not exist on TerrainImporterCfg directly, handle carefully
                # Or ensure your custom TerrainImporterCfg handles them if needed by the terrain class
                self.cfg.terrain.num_envs = self.scene.cfg.num_envs
                self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
                pass # Terrain class usually gets num_envs, device from its init call
            
            terrain_class_path = getattr(self.cfg.terrain, "class_type", None)
            if isinstance(terrain_class_path, str):
                try:
                    module_path, class_name = terrain_class_path.rsplit('.', 1)
                    module = __import__(module_path, fromlist=[class_name])
                    terrain_class = getattr(module, class_name)
                except Exception as e:
                    print(f"[ERROR] Failed to import terrain class {terrain_class_path}: {e}")
                    from isaaclab.terrains import TerrainImporter # Fallback
                    terrain_class = TerrainImporter
            elif terrain_class_path is None: # Default if not specified
                from isaaclab.terrains import TerrainImporter
                terrain_class = TerrainImporter
            else: # If class_type is already a class object
                terrain_class = terrain_class_path
            # Pass necessary args like num_envs, device to terrain constructor if it expects them.
            # The TerrainImporter base class in Isaac Lab typically handles this.
            self._terrain = terrain_class(self.cfg.terrain) # type: ignore
        else:
            print("[WARNING] SixfeetEnv: No terrain configuration. Spawning default plane.")
            from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
            spawn_ground_plane("/World/ground", GroundPlaneCfg())
            # Dummy terrain for env_origins if no terrain object provides it
            class DummyTerrain: # type: ignore
                def __init__(self, num_envs, device): self.env_origins = torch.zeros((num_envs, 3), device=device)
            self._terrain = DummyTerrain(self.cfg.scene.num_envs, self.device) # type: ignore

        # Add a light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg) # pyright: ignore [reportGeneralTypeIssues]

        # Clone environments
        self.scene.clone_environments(copy_from_source=False) # Crucial for multi-env setup

    def _update_commands(self, env_ids: torch.Tensor):
        self._time_since_last_command_change[env_ids] += self.physics_dt # self.sim.dt or self.physics_dt
        
        # Check if command_profile exists and has the key
        cmd_duration = self.cfg.command_profile.get("command_mode_duration_s", float('inf'))
        stand_still_prob = self.cfg.command_profile.get("stand_still_prob", 0.0)
        num_cmd_modes = self.cfg.command_profile.get("num_command_modes", 1) # Default to 1 mode (e.g. stand)

        change_command_mask = self._time_since_last_command_change[env_ids] >= cmd_duration
        envs_to_change = env_ids[change_command_mask]

        if envs_to_change.numel() > 0:
            self._time_since_last_command_change[envs_to_change] = 0.0
            num_to_change = envs_to_change.shape[0]
            
            new_commands_for_changed_envs = torch.zeros(num_to_change, 3, device=self.device, dtype=torch.float)

            if stand_still_prob == 1.0: # Always stand still
                command_modes = torch.zeros(num_to_change, device=self.device, dtype=torch.long) # Mode 0 for stand
            elif num_cmd_modes > 0 : # Sample new command modes if not always standing
                 command_modes = torch.randint(0, num_cmd_modes, (num_to_change,), device=self.device)
                 # Force stand still for a portion based on probability, if not already 1.0
                 if stand_still_prob > 0.0 and stand_still_prob < 1.0:
                    stand_mask = torch.rand(num_to_change, device=self.device) < stand_still_prob
                    command_modes[stand_mask] = 0 # Mode 0 for stand
            else: # Should not happen if configured correctly
                command_modes = torch.zeros(num_to_change, device=self.device, dtype=torch.long)


            # Map command_modes to command vectors [vx, vy, vyaw] where values are -1, 0, 1
            # Mode 0: stand still (already zeros)
            new_commands_for_changed_envs[command_modes == 1, 0] = 1.0  # Forward (X+)
            new_commands_for_changed_envs[command_modes == 2, 0] = -1.0 # Backward (X-)
            new_commands_for_changed_envs[command_modes == 3, 1] = -1.0 # Left Strafe (Y-) (Note: Isaac Sim Y is often right)
            new_commands_for_changed_envs[command_modes == 4, 1] = 1.0  # Right Strafe (Y+)
            new_commands_for_changed_envs[command_modes == 5, 2] = -1.0 # Turn Left (Yaw-)
            new_commands_for_changed_envs[command_modes == 6, 2] = 1.0  # Turn Right (Yaw+)
            # Ensure mode 0 (stand) results in [0,0,0], which it does by default init.
            
            self._commands[envs_to_change] = new_commands_for_changed_envs

    def _pre_physics_step(self, actions: torch.Tensor):
        self._policy_actions = actions.clone().to(self.device)
        
        if torch.any(torch.isnan(actions)) or torch.any(torch.isinf(actions)):
            print(f"[WARNING] Invalid actions (NaN/Inf) detected in pre_physics_step. Clamping to zeros.")
            # Consider more sophisticated handling or logging env_ids with bad actions
            actions = torch.zeros_like(actions) # Replace NaN/inf with zeros
        
        cur_pos = self.robot.data.joint_pos
        # Target position is current position + scaled action delta
        # This is a form of residual control if actions are deltas to current position
        self._processed_actions = cur_pos + self.cfg.action_scale * self._policy_actions
        
        # Clamp actions to joint limits
        self._processed_actions = torch.clamp(
            self._processed_actions, 
            self._q_lower_limits.unsqueeze(0), # Ensure broadcasting (num_dof) -> (1, num_dof)
            self._q_upper_limits.unsqueeze(0)  # Ensure broadcasting
        )
        
        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._update_commands(all_env_ids)

    def _apply_action(self):
        self.robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_policy_actions = self._policy_actions.clone() # Store for action rate penalty
        
        # Relative joint positions (to default/initial pose)
        default_pos_expanded = self._default_joint_pos.unsqueeze(0) if self._default_joint_pos.ndim == 1 else self._default_joint_pos
        joint_pos_rel = self.robot.data.joint_pos - default_pos_expanded
        
        obs_list = [
            self.robot.data.projected_gravity_b,    # (num_envs, 3)
            self.robot.data.root_ang_vel_b,         # (num_envs, 3)
            self._commands,                         # (num_envs, 3)
            normalize_angle_for_obs(joint_pos_rel), # (num_envs, num_dof) - angles normalized
            self.robot.data.joint_vel,              # (num_envs, num_dof)
        ]
        observations_tensor = torch.cat(obs_list, dim=-1)
        
        # Observation dimension check
        if hasattr(self.cfg, "observation_space") and observations_tensor.shape[1] != self.cfg.observation_space:
            print(f"[ERROR] SixfeetEnv: Obs dim mismatch! Expected {self.cfg.observation_space}, got {observations_tensor.shape[1]}")
            print(f"  Details: proj_grav({self.robot.data.projected_gravity_b.shape}), ang_vel({self.robot.data.root_ang_vel_b.shape}), cmds({self._commands.shape}), jpos_rel({joint_pos_rel.shape}), jvel({self.robot.data.joint_vel.shape})")

        return {"policy": observations_tensor}

    def _get_rewards(self) -> torch.Tensor:
        # Gather all necessary states from robot data
        root_lin_vel_b = self.robot.data.root_lin_vel_b
        root_ang_vel_b = self.robot.data.root_ang_vel_b
        projected_gravity_b = self.robot.data.projected_gravity_b
        
        default_pos_expanded = self._default_joint_pos.unsqueeze(0) if self._default_joint_pos.ndim == 1 else self._default_joint_pos
        joint_pos_rel = self.robot.data.joint_pos - default_pos_expanded
        current_joint_pos_abs = self.robot.data.joint_pos # Absolute joint positions
        
        joint_vel = self.robot.data.joint_vel
        applied_torque = self.robot.data.applied_torque
        # joint_acc might not be available on all ArticulationData, provide a default
        joint_acc = getattr(self.robot.data, "joint_acc", torch.zeros_like(joint_vel, device=self.device))
        root_pos_w = self.robot.data.root_pos_w # Root position in world frame

        # Calculate undesired contacts
        undesired_contacts_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.cfg.rew_scale_undesired_contact != 0.0 and self._undesired_contact_body_ids and len(self._undesired_contact_body_ids) > 0:
            if hasattr(self._contact_sensor.data, 'net_forces_w_history') and self._contact_sensor.data.net_forces_w_history is not None:
                all_forces_history = self._contact_sensor.data.net_forces_w_history # (num_envs, history_len, num_bodies, 3)
                # Ensure history and body indices are valid
                if all_forces_history.ndim == 4 and all_forces_history.shape[1] > 0 and \
                   all_forces_history.shape[2] > max(self._undesired_contact_body_ids):
                    current_net_forces_w = all_forces_history[:, -1, :, :] # Get current time step forces
                    forces_on_undesired_bodies = current_net_forces_w[:, self._undesired_contact_body_ids, :]
                    force_magnitudes = torch.norm(forces_on_undesired_bodies, dim=-1) # Norm across xyz force components
                    undesired_contacts_active = torch.any(force_magnitudes > 1.0, dim=1) # Check if any undesired body has contact force > 1N
            # else:
            #   (Optional: print a warning once if sensor data is missing but penalty is active)
            #   pass # Silently pass if sensor data is not ready or penalty is zero.
        
        # Compute rewards using the JIT-compiled function
        total_reward, reward_terms_dict = compute_sixfeet_rewards_directional(
            root_lin_vel_b, root_ang_vel_b, projected_gravity_b,
            joint_pos_rel, joint_vel, applied_torque, joint_acc,
            self._q_lower_limits, self._q_upper_limits, current_joint_pos_abs,
            self._policy_actions, self._previous_policy_actions,
            root_pos_w,
            undesired_contacts_active,
            self._commands,
            self.cfg.command_profile, # pyright: ignore [reportGeneralTypeIssues]
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
            self.physics_dt # Use self.physics_dt (dt per physics step)
        )

        # Log reward terms
        if "log" not in self.extras or self.extras["log"] is None: self.extras["log"] = {} # Initialize log dict if needed
        for key, value in reward_terms_dict.items():
            term_mean = value.mean()
            self.extras["log"][f"reward_term/{key}_step_avg"] = term_mean.item() if torch.is_tensor(term_mean) else term_mean
            # Accumulate episodic sums
            if key not in self._episode_reward_terms_sum:
                self._episode_reward_terms_sum[key] = torch.zeros(self.num_envs, device=self.device)
            # Ensure value is compatible for addition (e.g., squeeze if it's (N,1) instead of (N))
            self._episode_reward_terms_sum[key] += value.squeeze(-1) if value.ndim > 1 and value.shape[-1] == 1 else value


        # Apply termination penalty
        current_terminated, current_time_out = self._get_dones()
        just_failed_termination = current_terminated & (~current_time_out) # Terminated not due to timeout
        
        final_reward = torch.where(
            just_failed_termination,
            torch.full_like(total_reward, self.cfg.rew_scale_termination), # Apply large negative reward for failure
            total_reward
        )
        
        self.extras["log"]["reward/final_reward_avg"] = final_reward.mean().item() if torch.is_tensor(final_reward) else final_reward.mean()
        return final_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Time out condition
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Robot state based termination conditions
        root_pos_w = self.robot.data.root_pos_w
        height_too_low = root_pos_w[:, 2] < self.cfg.termination_height_thresh
        
        projected_gravity_z = self.robot.data.projected_gravity_b[:, 2]
        # If z component of gravity vector in base frame is large positive, it means base's z-axis points downwards (fallen over)
        fallen_over = projected_gravity_z > self.cfg.termination_body_z_thresh 

        # Base contact termination
        base_contact_termination = torch.zeros_like(time_out, dtype=torch.bool)
        if self.cfg.termination_base_contact and self._base_body_id and len(self._base_body_id) > 0:
            if hasattr(self._contact_sensor.data, 'net_forces_w_history') and self._contact_sensor.data.net_forces_w_history is not None:
                all_forces_history = self._contact_sensor.data.net_forces_w_history
                if all_forces_history.ndim == 4 and all_forces_history.shape[1] > 0 and \
                   all_forces_history.shape[2] > max(self._base_body_id): # Check index validity
                    current_net_forces_w = all_forces_history[:, -1, :, :]
                    forces_on_base = current_net_forces_w[:, self._base_body_id, :]
                    force_magnitudes_base = torch.norm(forces_on_base, dim=-1)
                    base_contact_termination = torch.any(force_magnitudes_base > 1.0, dim=1) # Threshold for base contact
            # else:
            #   (Optional: print warning if sensor data missing but termination active)
            #   pass
            
        terminated = height_too_low | fallen_over | base_contact_termination
        return terminated, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        # Standard parent class reset (handles episode_length_buf, reset_buf etc.)
        super()._reset_idx(env_ids)
        
        # Determine which environment indices to reset
        eids = torch.arange(self.num_envs, device=self.device, dtype=torch.long) if env_ids is None \
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if eids.numel() == 0: # No environments to reset for this call
            return

        # --- Robot root state reset ---
        root_state_reset = self.robot.data.default_root_state[eids].clone() # Start from default
        
        # Add terrain origins if available
        if hasattr(self._terrain, 'env_origins') and self._terrain.env_origins is not None:
             root_state_reset[:, :3] += self._terrain.env_origins[eids]
        
        # Set initial height based on cfg (init_state.pos[2] + reset_height_offset)
        initial_height_base = self.cfg.robot.init_state.pos[2] if self.cfg.robot.init_state.pos is not None and len(self.cfg.robot.init_state.pos) == 3 else 0.3 # Fallback
        root_state_reset[:, 2] = initial_height_base + self.cfg.reset_height_offset

        # Randomize initial yaw orientation
        random_yaw = (torch.rand(eids.shape[0], device=self.device) - 0.5) * 2.0 * self.cfg.root_orientation_yaw_range
        world_z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=random_yaw.dtype)
        # quat_from_angle_axis produces (x,y,z,w) by default with Isaac Lab utils
        orientation_quat_xyzw = quat_from_angle_axis(random_yaw, world_z_axis)
        root_state_reset[:, 3:7] = convert_quat(orientation_quat_xyzw, to="wxyz") # Ensure wxyz for Isaac Sim root state
        
        # Set initial linear and angular velocities to zero
        root_state_reset[:, 7:] = 0.0
        
        self.robot.write_root_state_to_sim(root_state_reset, eids)
        
        # --- Joint state reset ---
        # Use default joint positions defined by the articulation's init_state in cfg
        # self._default_joint_pos should reflect the joint_pos map from cfg.robot.init_state
        default_pos_for_reset = self._default_joint_pos.unsqueeze(0).expand(eids.shape[0], -1)
        joint_pos_reset = default_pos_for_reset

        # (Optional: Add small noise to default joint positions for randomization if desired)
        # noise = (torch.rand_like(joint_pos_reset) - 0.5) * torch.deg2rad(torch.tensor(5.0, device=self.device)) # Example: +/- 2.5 deg noise
        # joint_pos_reset = torch.clamp(joint_pos_reset + noise, self._q_lower_limits.unsqueeze(0), self._q_upper_limits.unsqueeze(0))

        zero_joint_vel = torch.zeros_like(joint_pos_reset)
        self.robot.write_joint_state_to_sim(joint_pos_reset, zero_joint_vel, env_ids=eids)
        
        # Crucially, set PD controller targets to these reset joint positions
        self.robot.set_joint_position_target(joint_pos_reset, env_ids=eids)

        # --- Reset other environment-specific buffers ---
        # Reset command-related timers and ensure commands are updated for new episode
        cmd_duration = self.cfg.command_profile.get("command_mode_duration_s", float('inf'))
        self._time_since_last_command_change[eids] = cmd_duration # Force immediate command update
        self._update_commands(eids) # Generate initial commands (e.g., "stand still")

        # Reset action history buffers
        if hasattr(self, '_previous_policy_actions'): self._previous_policy_actions[eids] = 0.0
        if hasattr(self, '_policy_actions'): self._policy_actions[eids] = 0.0

        # Clear episodic reward term sums for environments that just reset
        for key in list(self._episode_reward_terms_sum.keys()): # Iterate over a copy of keys
            if self._episode_reward_terms_sum[key].shape[0] == self.num_envs : # Ensure it's per-env
                # Log summed rewards for episodes that just ended (on these env_ids) BEFORE clearing
                # This typically happens in the RL runner based on reset_buf from *previous* step.
                # For direct logging here, it would be the sum accumulated until reset.
                # if "log" not in self.extras or self.extras["log"] is None : self.extras["log"] = {}
                # self.extras["log"][f"EpisodeSum/{key}_on_reset"] = self._episode_reward_terms_sum[key][eids].mean().item()

                self._episode_reward_terms_sum[key][eids] = 0.0 # Clear sums for reset envs