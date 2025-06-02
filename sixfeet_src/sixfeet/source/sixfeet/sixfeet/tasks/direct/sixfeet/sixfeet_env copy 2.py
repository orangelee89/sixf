# sixfeet_env.py
from __future__ import annotations
import torch
import math
from collections.abc import Sequence
from typing import List, Dict, Optional
import re # 导入正则表达式模块

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import (
    quat_rotate,
    quat_from_angle_axis,
    convert_quat,
    euler_xyz_from_quat,
    matrix_from_quat # 如果需要，可以导入
)

from .sixfeet_env_cfg import SixfeetEnvCfg

# ───────────────── 辅助函数 ──────────────────
@torch.jit.script
def normalize_angle_for_obs(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def compute_sixfeet_rewards_directional(
    # --- 基础参数 (与你提供的 "Corrected JIT Type Hints" 版本一致) ---
    root_lin_vel_b: torch.Tensor, root_ang_vel_b: torch.Tensor, projected_gravity_b: torch.Tensor,
    joint_pos_rel: torch.Tensor, joint_vel: torch.Tensor, applied_torque: torch.Tensor,
    joint_acc: torch.Tensor, q_lower_limits: torch.Tensor, q_upper_limits: torch.Tensor,
    current_joint_pos_abs: torch.Tensor, actions_from_policy: torch.Tensor,
    previous_actions_from_policy: torch.Tensor, root_pos_w: torch.Tensor,
    undesired_contacts_active: torch.Tensor, commands_discrete: torch.Tensor,
    cfg_cmd_profile: Dict[str, float], cfg_rew_scale_move_in_commanded_direction: float,
    cfg_rew_scale_achieve_reference_angular_rate: float, cfg_rew_scale_alive: float,
    cfg_rew_scale_target_height: float, cfg_target_height_m: float,
    cfg_rew_scale_action_cost: float, cfg_rew_scale_action_rate: float,
    cfg_rew_scale_joint_torques: float, cfg_rew_scale_joint_accel: float,
    cfg_rew_scale_lin_vel_z_penalty: float, cfg_rew_scale_ang_vel_xy_penalty: float,
    cfg_rew_scale_flat_orientation: float, cfg_rew_scale_unwanted_movement_penalty: float,
    cfg_rew_scale_dof_at_limit: float, cfg_rew_scale_toe_orientation_penalty: float,
    cfg_toe_joint_indices: Optional[torch.Tensor], cfg_rew_scale_low_height_penalty: float,
    cfg_min_height_penalty_threshold: float, cfg_rew_scale_undesired_contact: float,
    dt: float,
    # --- 之前添加的参数 ---
    cfg_rew_scale_orientation_deviation: float,
    cfg_orientation_termination_angle_limit_rad: float,
    cfg_joint_limit_penalty_threshold_percent: float,
    num_self_collisions_per_env: torch.Tensor, # 自碰撞计数
    cfg_rew_scale_self_collision: float,       # 自碰撞惩罚scale
    was_severely_tilted_last_step: torch.Tensor,
    cfg_rew_scale_successful_flip: float,
    cfg_target_height_reward_sharpness: float,
    foot_quats_w: Optional[torch.Tensor],      # 脚部世界坐标系四元数 (用于轴对齐)
    foot_contact_mask: Optional[torch.Tensor], # 脚部是否接触 (用于轴对齐)
    # cfg_rew_scale_foot_z_alignment: float,    # 被下面的 custom_foot_axis_alignment 替代
    foot_link_target_align_local_axes: Optional[torch.Tensor],
    foot_link_target_align_world_axes: Optional[torch.Tensor],
    cfg_rew_scale_custom_foot_axis_alignment: float,
    # --- 新增参数用于所有脚触地奖励 ---
    num_feet_in_contact: torch.Tensor, # (num_envs,) 当前实际接触地面的脚的数量
    cfg_rew_scale_all_feet_stable_stand: float, # 所有脚稳定站立的奖励scale
    cfg_rew_scale_airborne_feet_penalty: float # （可选）悬空脚惩罚的scale

) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    device = projected_gravity_b.device

    # --- 1. 基础奖励和惩罚项计算 (基于你提供的文件结构) ---
    ref_ang_rate = cfg_cmd_profile.get("reference_angular_rate", 0.0) # 确保 cfg_cmd_profile 的值都是 float
    # ... (所有其他基础奖励计算，如 movement, alive 等，与你提供的文件一致) ...
    linear_vel_x_local = root_lin_vel_b[:, 0]; linear_vel_y_local = root_lin_vel_b[:, 1]
    reward_fwd_bkwd_move = commands_discrete[:, 0] * linear_vel_x_local; reward_left_right_move = commands_discrete[:, 1] * linear_vel_y_local
    is_linear_cmd_active = (torch.abs(commands_discrete[:, 0]) > 0.5) | (torch.abs(commands_discrete[:, 1]) > 0.5)
    reward_linear_direction = (reward_fwd_bkwd_move + reward_left_right_move) * is_linear_cmd_active.float()
    reward_move_in_commanded_direction = reward_linear_direction * cfg_rew_scale_move_in_commanded_direction
    angular_vel_z_local = root_ang_vel_b[:, 2]; reward_angular_direction_raw = -commands_discrete[:, 2] * angular_vel_z_local
    is_turn_cmd_active = torch.abs(commands_discrete[:, 2]) > 0.5
    turn_rate_error = torch.abs(torch.abs(angular_vel_z_local) - ref_ang_rate)
    reward_achieve_ref_ang_rate = torch.exp(-5.0 * turn_rate_error) * is_turn_cmd_active.float() * cfg_rew_scale_achieve_reference_angular_rate
    reward_turn = (reward_angular_direction_raw * is_turn_cmd_active.float() * cfg_rew_scale_move_in_commanded_direction) + reward_achieve_ref_ang_rate
    reward_alive = torch.ones_like(commands_discrete[:,0], device=device) * cfg_rew_scale_alive
    current_height_z = root_pos_w[:, 2]

    # --- 条件判断：机器人是否在允许的“正面”姿态范围内 ---
    cos_angle_robot_z_with_world_z_cond = -projected_gravity_b[:, 2]
    is_within_orientation_limit = cos_angle_robot_z_with_world_z_cond >= torch.cos(torch.tensor(cfg_orientation_termination_angle_limit_rad, device=device, dtype=torch.float32))

    # --- 2. 条件化目标高度奖励 (cfg_target_height_m = 0.30) ---
    height_error_sq = torch.square(current_height_z - cfg_target_height_m)
    _base_reward_target_height = torch.exp(-cfg_target_height_reward_sharpness * height_error_sq) * cfg_rew_scale_target_height
    reward_target_height = torch.where(is_within_orientation_limit, _base_reward_target_height, torch.zeros_like(_base_reward_target_height))

    # --- 3. 条件化速度惩罚 ---
    height_threshold_for_vel_penalties = cfg_target_height_m - 0.02 # 直接计算
    _base_penalty_lin_vel_z = torch.square(root_lin_vel_b[:, 2]) * cfg_rew_scale_lin_vel_z_penalty
    _base_penalty_ang_vel_xy = torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1) * cfg_rew_scale_ang_vel_xy_penalty
    is_above_height_for_vel_penalties = current_height_z >= height_threshold_for_vel_penalties
    vel_penalties_active_condition = is_within_orientation_limit & is_above_height_for_vel_penalties
    penalty_lin_vel_z = torch.where(vel_penalties_active_condition, _base_penalty_lin_vel_z, torch.zeros_like(_base_penalty_lin_vel_z))
    penalty_ang_vel_xy = torch.where(vel_penalties_active_condition, _base_penalty_ang_vel_xy, torch.zeros_like(_base_penalty_ang_vel_xy))

    # --- 4. 其他惩罚项 (保持你文件中的逻辑，确保所有变量都被定义) ---
    penalty_action_cost = torch.sum(actions_from_policy**2, dim=-1) * cfg_rew_scale_action_cost
    penalty_action_rate = torch.sum((actions_from_policy - previous_actions_from_policy)**2, dim=-1) * cfg_rew_scale_action_rate
    penalty_joint_torques = torch.sum(applied_torque**2, dim=-1) * cfg_rew_scale_joint_torques
    penalty_joint_accel = torch.sum(joint_acc**2, dim=-1) * cfg_rew_scale_joint_accel
    penalty_flat_orientation = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1) * cfg_rew_scale_flat_orientation
    is_stand_cmd_active = torch.all(commands_discrete == 0, dim=1)
    unwanted_lin_vel_sq = torch.sum(torch.square(root_lin_vel_b[:, :2]), dim=1)
    unwanted_ang_vel_sq = torch.square(root_ang_vel_b[:, 2])
    penalty_unwanted_movement = (unwanted_lin_vel_sq + unwanted_ang_vel_sq) * is_stand_cmd_active.float() * cfg_rew_scale_unwanted_movement_penalty
    dof_range = q_upper_limits - q_lower_limits
    dof_range = torch.where(dof_range < 1e-6, torch.ones_like(dof_range), dof_range)
    q_lower_expanded = q_lower_limits.unsqueeze(0) if q_lower_limits.ndim == 1 else q_lower_limits
    dof_range_expanded = dof_range.unsqueeze(0) if dof_range.ndim == 1 else dof_range
    dof_pos_scaled_01 = (current_joint_pos_abs - q_lower_expanded) / dof_range_expanded
    threshold_percent = cfg_joint_limit_penalty_threshold_percent
    near_lower_limit = torch.relu(threshold_percent - dof_pos_scaled_01)**2
    near_upper_limit = torch.relu(dof_pos_scaled_01 - (1.0 - threshold_percent))**2
    penalty_dof_at_limit = torch.sum(near_lower_limit + near_upper_limit, dim=-1) * cfg_rew_scale_dof_at_limit
    is_severely_tilted_for_other_penalties = projected_gravity_b[:, 2] > 0.0
    _base_penalty_toe_orientation = torch.zeros_like(commands_discrete[:,0], device=device)
    if cfg_rew_scale_toe_orientation_penalty != 0.0 and cfg_toe_joint_indices is not None:
        if cfg_toe_joint_indices.numel() > 0:
            toe_joint_positions = current_joint_pos_abs[:, cfg_toe_joint_indices]
            _base_penalty_toe_orientation = torch.sum(torch.relu(toe_joint_positions + math.radians(10.0))**2, dim=-1) * cfg_rew_scale_toe_orientation_penalty
    penalty_toe_orientation = torch.where(is_severely_tilted_for_other_penalties, torch.zeros_like(_base_penalty_toe_orientation), _base_penalty_toe_orientation)
    is_too_low = (current_height_z < cfg_min_height_penalty_threshold).float()
    penalty_low_height = is_too_low * cfg_rew_scale_low_height_penalty
    _base_penalty_undesired_contact = undesired_contacts_active.float() * cfg_rew_scale_undesired_contact
    penalty_undesired_contact = torch.where(is_severely_tilted_for_other_penalties, torch.zeros_like(_base_penalty_undesired_contact), _base_penalty_undesired_contact)
    cos_angle_robot_z_with_world_z_dev = -projected_gravity_b[:, 2]
    angle_deviation_from_world_z = torch.acos(torch.clamp(cos_angle_robot_z_with_world_z_dev, -1.0 + 1e-7, 1.0 - 1e-7))
    penalty_orientation_deviation = cfg_rew_scale_orientation_deviation * angle_deviation_from_world_z
    penalty_self_collision = cfg_rew_scale_self_collision * (num_self_collisions_per_env > 0).float()
    current_is_severely_tilted_for_flip = projected_gravity_b[:, 2] > 0.0
    transitioned_to_upright_hemisphere = was_severely_tilted_last_step & (~current_is_severely_tilted_for_flip)
    reward_successful_flip = cfg_rew_scale_successful_flip * transitioned_to_upright_hemisphere.float()

     # --- 新增：所有脚稳定站立奖励 ---
    reward_all_feet_stable_stand = torch.zeros_like(root_pos_w[:, 0], device=device)
    if cfg_rew_scale_all_feet_stable_stand != 0.0:
        num_total_feet = 6 # 假设六足机器人
        all_feet_contacting = (num_feet_in_contact == num_total_feet)
        # is_stand_cmd_active 和 is_within_orientation_limit 已在上面计算
        stable_stand_condition = is_stand_cmd_active & is_within_orientation_limit & all_feet_contacting
        
        reward_all_feet_stable_stand = torch.where(
            stable_stand_condition,
            torch.full_like(reward_all_feet_stable_stand, cfg_rew_scale_all_feet_stable_stand),
            torch.zeros_like(reward_all_feet_stable_stand)
        )

    # --- （可选）惩罚悬空的脚 ---
    penalty_airborne_feet = torch.zeros_like(root_pos_w[:, 0], device=device)
    if cfg_rew_scale_airborne_feet_penalty != 0.0:
        num_total_feet = 6 
        airborne_feet_count = num_total_feet - num_feet_in_contact
        airborne_feet_count_float = torch.relu(airborne_feet_count.float())
        # 仅在站立指令且姿态良好时惩罚悬空脚可能更合理
        penalty_airborne_feet = torch.where(
            is_stand_cmd_active & is_within_orientation_limit,
            cfg_rew_scale_airborne_feet_penalty * airborne_feet_count_float,
            torch.zeros_like(airborne_feet_count_float)
        )



    # --- 新增：自定义足部特定轴向对齐奖励 ---
    reward_custom_foot_axis_alignment = torch.zeros_like(root_pos_w[:, 0], device=device)
    if foot_quats_w is not None and foot_contact_mask is not None and \
       foot_link_target_align_local_axes is not None and foot_link_target_align_world_axes is not None and \
       cfg_rew_scale_custom_foot_axis_alignment != 0.0:
        
        num_feet_to_check = foot_quats_w.shape[1]
        all_feet_alignment_quality = torch.zeros_like(foot_contact_mask, dtype=torch.float32) # (num_envs, num_feet)

        for i in range(num_feet_to_check):
            # foot_quats_w[:, i, :] shape: (num_envs, 4)
            # foot_link_target_align_axes[i, :] shape: (3,)
            # foot_link_target_align_world_axes[i, :] shape: (3,)
            
            # 扩展局部轴和目标世界轴以匹配批次大小
            local_axis_to_align = foot_link_target_align_local_axes[i, :].expand(foot_quats_w.shape[0], 3)
            target_world_axis = foot_link_target_align_world_axes[i, :].expand(foot_quats_w.shape[0], 3)
            
            # 将脚的局部待对齐轴旋转到世界坐标系
            # quat_rotate(quat_wxyz, vector_xyz)
            foot_local_axis_in_world = quat_rotate(foot_quats_w[:, i, :], local_axis_to_align) # (num_envs, 3)
            
            # 计算旋转后的局部轴与目标世界轴的点积 (cosine similarity)
            # (N,3) * (N,3) -> sum over last dim -> (N,)
            cos_sim = torch.sum(foot_local_axis_in_world * target_world_axis, dim=1) # (num_envs,)
            
            # 将 cos_sim (-1 to 1) 映射到奖励质量 (0 to 1), 1 表示完美对齐
            alignment_quality = (cos_sim + 1.0) / 2.0
            all_feet_alignment_quality[:, i] = alignment_quality
            
        active_foot_alignment_reward = all_feet_alignment_quality * foot_contact_mask.float()
        num_contacting_feet = torch.sum(foot_contact_mask.float(), dim=1)
        sum_alignment_reward_per_env = torch.sum(active_foot_alignment_reward, dim=1)
        
        reward_custom_foot_axis_alignment_values = sum_alignment_reward_per_env / torch.clamp(num_contacting_feet, min=1.0)
        reward_custom_foot_axis_alignment = torch.where(
            num_contacting_feet > 0,
            reward_custom_foot_axis_alignment_values * cfg_rew_scale_custom_foot_axis_alignment,
            torch.zeros_like(sum_alignment_reward_per_env)
        )
        
    # --- 总奖励计算 ---
    total_reward = (
        reward_move_in_commanded_direction + reward_turn + reward_alive + reward_target_height 
        + penalty_orientation_deviation + penalty_self_collision + reward_successful_flip
        + reward_custom_foot_axis_alignment 
        + reward_all_feet_stable_stand # 新增
        + penalty_airborne_feet      # 新增
        + (penalty_action_cost + penalty_action_rate + penalty_joint_torques + penalty_joint_accel
        + penalty_lin_vel_z + penalty_ang_vel_xy + penalty_flat_orientation + penalty_unwanted_movement # penalty_flat_orientation 在此 (dt组)
        + penalty_dof_at_limit + penalty_toe_orientation + penalty_low_height
        + penalty_undesired_contact) * dt
    )
    
    reward_terms: Dict[str, torch.Tensor] = {
        "move_in_commanded_direction": reward_move_in_commanded_direction,
        "turn_reward_combined": reward_turn,
        "alive": reward_alive,
        "target_height": reward_target_height,
        "orientation_deviation_penalty": penalty_orientation_deviation,
        "self_collision_penalty": penalty_self_collision,
        "successful_flip_reward": reward_successful_flip,
        "custom_foot_axis_alignment": reward_custom_foot_axis_alignment,
        "all_feet_stable_stand_reward": reward_all_feet_stable_stand, # 新增
        "airborne_feet_penalty": penalty_airborne_feet,             # 新增
        "action_cost_penalty": penalty_action_cost * dt,
        "action_rate_penalty": penalty_action_rate * dt,
        "joint_torques_penalty": penalty_joint_torques * dt,
        "joint_accel_penalty": penalty_joint_accel * dt,
        "lin_vel_z_penalty": penalty_lin_vel_z, 
        "ang_vel_xy_penalty": penalty_ang_vel_xy,
        "flat_orientation_penalty": penalty_flat_orientation * dt,
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
    _orientation_termination_angle_limit_rad: float
    _was_severely_tilted_last_step: torch.Tensor
    
    # 用于自定义足部姿态对齐
    _foot_link_articulation_indices: List[int] 
    _foot_link_sensor_indices: List[int]     
    _foot_link_names_found_for_align: List[str]
    _foot_link_target_align_axes_tensor: Optional[torch.Tensor] = None # (num_feet, 3)
    _foot_link_target_align_world_axes_tensor: Optional[torch.Tensor] = None # (num_feet, 3)
    _all_foot_tip_sensor_indices: List[int]

    def __init__(self, cfg: SixfeetEnvCfg, render_mode: str | None = None, **kwargs):
        # 1. 初始化不依赖父类初始化的属性
        if hasattr(cfg, "orientation_termination_angle_limit_deg"):
            self._orientation_termination_angle_limit_rad = math.radians(cfg.orientation_termination_angle_limit_deg)
        else:
            self._orientation_termination_angle_limit_rad = math.radians(90.0)
        
        self._foot_link_articulation_indices = []
        self._foot_link_sensor_indices = []
        self._foot_link_names_found_for_align = []
        target_align_axes_list = []
        target_world_axes_list = []

        # 2. 调用父类的 __init__
        super().__init__(cfg, render_mode, **kwargs) # self.device 在此之后可用

        # --- 在 super().__init__ 之后，self.device 可用 ---
        world_z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=torch.float32)
        local_x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
        local_neg_x_axis = torch.tensor([-1.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
        local_y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=torch.float32)
        local_neg_y_axis = torch.tensor([0.0, -1.0, 0.0], device=self.device, dtype=torch.float32)
        local_z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=torch.float32) # 默认对齐轴

        self._was_severely_tilted_last_step = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        if hasattr(self.cfg, "foot_link_name_pattern_for_custom_align") and self.cfg.foot_link_name_pattern_for_custom_align:
            foot_pattern_str = self.cfg.foot_link_name_pattern_for_custom_align
            try: foot_pattern_re = re.compile(foot_pattern_str)
            except re.error: print(f"[ERROR] Invalid regex: {foot_pattern_str}"); foot_pattern_re = None

            if foot_pattern_re and hasattr(self.robot, "body_names") and self.robot.body_names is not None:
                temp_foot_infos: List[Tuple[int, str]] = []
                for i, body_name in enumerate(self.robot.body_names):
                    if foot_pattern_re.fullmatch(body_name):
                        temp_foot_infos.append((i, body_name))
                temp_foot_infos.sort(key=lambda x: x[1]) 

                for artic_idx, body_name in temp_foot_infos:
                    self._foot_link_articulation_indices.append(artic_idx)
                    self._foot_link_names_found_for_align.append(body_name)
                    
                    # 根据你的描述确定局部轴
                    # foot_link_13 (-Y), foot_link_23 (X), foot_link_33 (Y)
                    # foot_link_43 (-Y), foot_link_53 (-X), foot_link_63 (Y)
                    if body_name.endswith("33") or body_name.endswith("63"): # 脚3, 6
                        target_align_axes_list.append(local_y_axis)
                    elif body_name.endswith("53"): # 脚5
                        target_align_axes_list.append(local_neg_x_axis)
                    elif body_name.endswith("23"): # 脚2
                        target_align_axes_list.append(local_x_axis)
                    elif body_name.endswith("43") or body_name.endswith("13"): # 脚1, 4
                        target_align_axes_list.append(local_neg_y_axis)
                    else:
                        print(f"[WARNING] Foot link {body_name} matched pattern but has no specific axis rule. Defaulting to local Z.")
                        target_align_axes_list.append(local_z_axis)
                    target_world_axes_list.append(world_z_axis) # 所有都与世界Z轴对齐
                
                if self._foot_link_names_found_for_align:
                    sensor_indices_found, _ = self._contact_sensor.find_bodies(self._foot_link_names_found_for_align)
                    if sensor_indices_found and len(sensor_indices_found) == len(self._foot_link_names_found_for_align):
                        self._foot_link_sensor_indices = sensor_indices_found
                        if target_align_axes_list: # 确保列表非空
                            self._foot_link_target_align_local_axes_tensor = torch.stack(target_align_axes_list)
                            self._foot_link_target_align_world_axes_tensor = torch.stack(target_world_axes_list)
                            print(f"[INFO] Articulation indices for custom align: {self._foot_link_articulation_indices}")
                            print(f"[INFO] Sensor body indices for custom align/contact: {self._foot_link_sensor_indices} for names {self._foot_link_names_found_for_align}")
                        else: # 如果 target_align_axes_list 为空 (例如所有脚都没有特定规则且默认了)
                             print(f"[WARNING] Target axes list is empty for custom foot alignment.")
                             self._foot_link_articulation_indices = []; self._foot_link_sensor_indices = []; self._foot_link_names_found_for_align = []

                    else: # 清理
                        self._foot_link_articulation_indices = []; self._foot_link_sensor_indices = []; self._foot_link_names_found_for_align = []
                        print(f"[WARNING] Could not map all foot link names to sensor indices consistently.")
                else: print(f"[WARNING] No robot body names matched foot pattern for custom alignment: {foot_pattern_str}")
            # ... (其他elif和else分支)
        else: print(f"[INFO] 'foot_link_name_pattern_for_custom_align' not in Cfg. Custom foot align/all_feet_contact rewards disabled.")


        if not self._foot_link_articulation_indices or \
           len(self._foot_link_articulation_indices) != len(target_align_axes_list) or \
           not self._foot_link_sensor_indices or \
           len(self._foot_link_sensor_indices) != len(self._foot_link_articulation_indices) or \
           self._foot_link_target_align_local_axes_tensor is None: # 额外检查tensor是否创建
            if hasattr(self.cfg, "rew_scale_custom_foot_axis_alignment") and getattr(self.cfg, "rew_scale_custom_foot_axis_alignment", 0.0) != 0.0:
                print(f"[WARNING] Custom foot axis alignment reward may be disabled due to setup issues in __init__.")
            self._foot_link_articulation_indices = []; self._foot_link_sensor_indices = []; self._foot_link_names_found_for_align = []
            self._foot_link_target_align_local_axes_tensor = None; self._foot_link_target_align_world_axes_tensor = None
        
        # ... (其他 __init__ 内容与上一版本相同) ...
        self._default_joint_pos = self.robot.data.default_joint_pos.clone()
        if self._default_joint_pos.ndim > 1 and self._default_joint_pos.shape[0] == self.num_envs: self._default_joint_pos = self._default_joint_pos[0]
        joint_limits = self.robot.data.joint_pos_limits[0].to(self.device)
        self._q_lower_limits = joint_limits[:, 0]; self._q_upper_limits = joint_limits[:, 1]
        self._policy_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_policy_actions = torch.zeros_like(self._policy_actions); self._processed_actions = torch.zeros_like(self._policy_actions)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self._time_since_last_command_change = torch.zeros(self.num_envs, device=self.device)
        self._resolve_toe_joint_indices()
        self._undesired_contact_body_ids: Optional[List[int]] = None
        if hasattr(self.cfg, "undesired_contact_link_names_expr") and self.cfg.undesired_contact_link_names_expr and \
           hasattr(self.cfg, "rew_scale_undesired_contact") and self.cfg.rew_scale_undesired_contact != 0.0:
            indices, names = self._contact_sensor.find_bodies(self.cfg.undesired_contact_link_names_expr)
            if indices: self._undesired_contact_body_ids = indices; print(f"[INFO] Undesired contact IDs: {indices} for {names}")
            else: print(f"[WARNING] No bodies for undesired contact: {self.cfg.undesired_contact_link_names_expr}")
        self._base_body_id: Optional[List[int]] = None
        if hasattr(self.cfg, "termination_base_contact") and self.cfg.termination_base_contact and \
           hasattr(self.cfg, "base_link_name") and self.cfg.base_link_name:
            indices, names = self._contact_sensor.find_bodies(self.cfg.base_link_name)
            if indices: self._base_body_id = indices; print(f"[INFO] Base body ID: {indices} for {names}")
            else: print(f"[WARNING] No body for base contact: {self.cfg.base_link_name}")
        self._episode_reward_terms_sum: Dict[str, torch.Tensor] = {}


    # ... (_resolve_toe_joint_indices, _setup_scene, _update_commands, _pre_physics_step, _apply_action, _get_observations 与上一版本相同) ...
    def _resolve_toe_joint_indices(self): # 与上一版本一致
        self._toe_joint_indices: Optional[torch.Tensor] = None
        expr_or_list = getattr(self.cfg, 'toe_joint_names_expr', None)
        if not hasattr(self.cfg, "rew_scale_toe_orientation_penalty") or \
           self.cfg.rew_scale_toe_orientation_penalty == 0.0 or not expr_or_list: return
        num_dof_val = self._q_lower_limits.numel() 
        joint_names_list_for_logging = []
        if isinstance(expr_or_list, str):
            joint_indices_list, joint_names_list_for_logging = self.robot.find_joints(expr_or_list)
            if joint_indices_list: self._toe_joint_indices = torch.tensor(joint_indices_list, device=self.device, dtype=torch.long)
        elif isinstance(expr_or_list, list) and all(isinstance(i, int) for i in expr_or_list):
            if expr_or_list:
                temp_indices = torch.tensor(expr_or_list, device=self.device, dtype=torch.long)
                if torch.any(temp_indices < 0) or torch.any(temp_indices >= num_dof_val):
                    print(f"[ERROR] Invalid toe joint indices in list: {expr_or_list}. Max: {num_dof_val - 1}"); self._toe_joint_indices = None
                else: self._toe_joint_indices = temp_indices
        elif expr_or_list is not None: print(f"[WARNING] 'toe_joint_names_expr' ('{expr_or_list}') invalid type.")
        if self._toe_joint_indices is not None:
            if self._toe_joint_indices.numel() == 0: self._toe_joint_indices = None
            elif torch.any(self._toe_joint_indices < 0) or torch.any(self._toe_joint_indices >= num_dof_val):
                print(f"[ERROR] Invalid toe joint indices after processing: {self._toe_joint_indices.tolist()}. Max: {num_dof_val - 1}"); self._toe_joint_indices = None
            else: 
                log_msg = f"[INFO] Validated toe joint indices: {self._toe_joint_indices.tolist()}"
                if joint_names_list_for_logging: log_msg += f", names: {joint_names_list_for_logging}"
                print(log_msg)
        if self._toe_joint_indices is None and expr_or_list is not None: print(f"[WARNING] No valid toe joint indices from '{expr_or_list}'.")
        elif self._toe_joint_indices is None and expr_or_list is None and self.cfg.rew_scale_toe_orientation_penalty != 0.0: print(f"[INFO] Toe penalty active but no expr.")

    def _setup_scene(self): # 与上一版本一致
        self.robot = Articulation(self.cfg.robot); self.scene.articulations["robot"] = self.robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor); self.scene.sensors["contact_sensor"] = self._contact_sensor
        if hasattr(self.cfg, "terrain") and self.cfg.terrain is not None:
            if hasattr(self.scene, "cfg"): self.cfg.terrain.num_envs = self.scene.cfg.num_envs; self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            terrain_class_path = getattr(self.cfg.terrain, "class_type", None)
            if isinstance(terrain_class_path, str):
                try: module_path, class_name = terrain_class_path.rsplit('.', 1); module = __import__(module_path, fromlist=[class_name]); terrain_class = getattr(module, class_name)
                except Exception as e: print(f"[ERROR] Failed to import terrain class {terrain_class_path}: {e}"); from isaaclab.terrains import TerrainImporter; terrain_class = TerrainImporter
            elif terrain_class_path is None: from isaaclab.terrains import TerrainImporter; terrain_class = TerrainImporter
            else: terrain_class = terrain_class_path
            self._terrain = terrain_class(self.cfg.terrain)
        else:
            print("[WARNING] SixfeetEnv: No terrain cfg. Spawning default plane."); from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane; spawn_ground_plane("/World/ground", GroundPlaneCfg())
            class DummyTerrain: 
                def __init__(self, num_envs, device): self.env_origins = torch.zeros((num_envs, 3), device=device)
            self._terrain = DummyTerrain(self.cfg.scene.num_envs, self.device)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75,0.75,0.75)); light_cfg.func("/World/Light", light_cfg)
        self.scene.clone_environments(copy_from_source=False)

    def _update_commands(self, env_ids: torch.Tensor): # 与上一版本一致
        self._time_since_last_command_change[env_ids] += self.physics_dt
        cmd_profile = self.cfg.command_profile
        cmd_duration_s_config = cmd_profile.get("command_mode_duration_s", self.cfg.episode_length_s if hasattr(self.cfg, "episode_length_s") else 20.0)
        if isinstance(cmd_duration_s_config, str) and cmd_duration_s_config == "episode_length_s": cmd_duration = self.cfg.episode_length_s
        else:
            try: cmd_duration = float(cmd_duration_s_config)
            except (ValueError, TypeError) : cmd_duration = self.cfg.episode_length_s if hasattr(self.cfg, "episode_length_s") else 20.0
        stand_still_prob = cmd_profile.get("stand_still_prob", 0.0); num_cmd_modes = cmd_profile.get("num_command_modes", 1)
        change_command_mask = self._time_since_last_command_change[env_ids] >= cmd_duration
        envs_to_change = env_ids[change_command_mask]
        if envs_to_change.numel() > 0:
            self._time_since_last_command_change[envs_to_change] = 0.0; num_to_change = envs_to_change.shape[0]
            new_commands_for_changed_envs = torch.zeros(num_to_change, 3, device=self.device, dtype=torch.float)
            if stand_still_prob == 1.0: command_modes = torch.zeros(num_to_change, device=self.device, dtype=torch.long)
            elif num_cmd_modes > 0:
                command_modes = torch.randint(0, num_cmd_modes, (num_to_change,), device=self.device)
                if stand_still_prob > 0.0 and stand_still_prob < 1.0: stand_mask = torch.rand(num_to_change, device=self.device) < stand_still_prob; command_modes[stand_mask] = 0
            else: command_modes = torch.zeros(num_to_change, device=self.device, dtype=torch.long)
            new_commands_for_changed_envs[command_modes == 1, 0] = 1.0; new_commands_for_changed_envs[command_modes == 2, 0] = -1.0
            new_commands_for_changed_envs[command_modes == 3, 1] = -1.0; new_commands_for_changed_envs[command_modes == 4, 1] = 1.0
            new_commands_for_changed_envs[command_modes == 5, 2] = -1.0; new_commands_for_changed_envs[command_modes == 6, 2] = 1.0
            self._commands[envs_to_change] = new_commands_for_changed_envs

    def _pre_physics_step(self, actions: torch.Tensor): # 与上一版本一致
        self._policy_actions = actions.clone().to(self.device)
        if torch.any(torch.isnan(actions)) or torch.any(torch.isinf(actions)): print(f"[WARNING] Invalid actions: {actions}"); actions = torch.zeros_like(actions)
        cur_pos = self.robot.data.joint_pos; self._processed_actions = cur_pos + self.cfg.action_scale * self._policy_actions
        self._processed_actions = torch.clamp(self._processed_actions, self._q_lower_limits.unsqueeze(0), self._q_upper_limits.unsqueeze(0))
        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long); self._update_commands(all_env_ids)

    def _apply_action(self): self.robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict: # 与上一版本一致
        self._previous_policy_actions = self._policy_actions.clone()
        default_pos_expanded = self._default_joint_pos.unsqueeze(0) if self._default_joint_pos.ndim == 1 else self._default_joint_pos
        joint_pos_rel = self.robot.data.joint_pos - default_pos_expanded
        obs_list = [self.robot.data.projected_gravity_b, self.robot.data.root_ang_vel_b, self._commands, normalize_angle_for_obs(joint_pos_rel), self.robot.data.joint_vel]
        observations_tensor = torch.cat(obs_list, dim=-1)
        if hasattr(self.cfg, "observation_space") and observations_tensor.shape[1] != self.cfg.observation_space: print(f"[ERROR] Obs dim mismatch! Exp {self.cfg.observation_space}, got {observations_tensor.shape[1]}")
        return {"policy": observations_tensor}


    def _get_rewards(self) -> torch.Tensor:
        # ... (获取基础状态、不期望接触、自碰撞计数、当前是否严重倾斜的逻辑保持不变) ...
        # ... (获取 foot_quats_w_for_reward 和 foot_contact_mask_for_custom_align (用于轴对齐) 的逻辑保持不变) ...
        # ... (获取 num_feet_in_contact (用于所有脚稳定站立/悬空脚惩罚) 的逻辑保持不变) ...
        # (这些变量应该在你当前的 _get_rewards 方法中已经正确计算了)
        root_lin_vel_b = self.robot.data.root_lin_vel_b; root_ang_vel_b = self.robot.data.root_ang_vel_b; projected_gravity_b = self.robot.data.projected_gravity_b
        default_pos_expanded = self._default_joint_pos.unsqueeze(0) if self._default_joint_pos.ndim == 1 else self._default_joint_pos
        joint_pos_rel = self.robot.data.joint_pos - default_pos_expanded
        current_joint_pos_abs = self.robot.data.joint_pos; joint_vel = self.robot.data.joint_vel
        applied_torque = self.robot.data.applied_torque; joint_acc = getattr(self.robot.data, "joint_acc", torch.zeros_like(joint_vel, device=self.device))
        root_pos_w = self.robot.data.root_pos_w
        undesired_contacts_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool) # 确保此变量已定义
        if hasattr(self.cfg, "rew_scale_undesired_contact") and self.cfg.rew_scale_undesired_contact != 0.0 and \
           self._undesired_contact_body_ids and len(self._undesired_contact_body_ids) > 0:
             if hasattr(self._contact_sensor.data, 'net_forces_w_history') and self._contact_sensor.data.net_forces_w_history is not None:
                all_forces_contact_sensor = self._contact_sensor.data.net_forces_w_history
                if all_forces_contact_sensor.ndim == 4 and all_forces_contact_sensor.shape[1] > 0 and self._undesired_contact_body_ids and max(self._undesired_contact_body_ids) < all_forces_contact_sensor.shape[2]:
                    forces_undesired = all_forces_contact_sensor[:, -1, self._undesired_contact_body_ids, :]; undesired_contacts_active = torch.any(torch.norm(forces_undesired, dim=-1) > 1.0, dim=1)
        num_self_collisions_per_env = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 确保此变量已定义
        contact_data = self._contact_sensor.data
        if hasattr(contact_data, 'body_indices_in_contact_buffer') and contact_data.body_indices_in_contact_buffer is not None and \
           hasattr(self.robot.data, 'body_indices') and self.robot.data.body_indices is not None:
            contact_pairs_global_indices = contact_data.body_indices_in_contact_buffer; robot_global_body_indices = self.robot.data.body_indices
            if robot_global_body_indices.numel() > 0 and contact_pairs_global_indices.numel() > 0:
                num_envs_contact, max_contacts_per_env, _ = contact_pairs_global_indices.shape; num_envs_robot, num_robot_bodies = robot_global_body_indices.shape
                if num_envs_contact == self.num_envs and num_envs_robot == self.num_envs:
                    robot_indices_expanded = robot_global_body_indices.unsqueeze(1); body0_global_idx_pairs = contact_pairs_global_indices[..., 0].unsqueeze(-1); body1_global_idx_pairs = contact_pairs_global_indices[..., 1].unsqueeze(-1)
                    is_body0_robot_contact = torch.any(body0_global_idx_pairs == robot_indices_expanded, dim=2); is_body1_robot_contact = torch.any(body1_global_idx_pairs == robot_indices_expanded, dim=2)
                    is_self_collision_pair = is_body0_robot_contact & is_body1_robot_contact
                    valid_contact_indices = (contact_pairs_global_indices[..., 0] != -1) & (contact_pairs_global_indices[..., 1] != -1) & (contact_pairs_global_indices[..., 0] != contact_pairs_global_indices[..., 1])
                    if hasattr(contact_data, 'num_contacts_in_buffer') and contact_data.num_contacts_in_buffer is not None:
                        valid_contact_mask_sensor = torch.arange(max_contacts_per_env, device=self.device).unsqueeze(0) < contact_data.num_contacts_in_buffer.unsqueeze(1)
                        is_self_collision_pair_valid = is_self_collision_pair & valid_contact_mask_sensor & valid_contact_indices
                    else: is_self_collision_pair_valid = is_self_collision_pair & valid_contact_indices
                    num_self_collisions_per_env = torch.sum(is_self_collision_pair_valid, dim=1)
        current_is_severely_tilted_for_flip_reward = (projected_gravity_b[:, 2] > 0.0)
        foot_quats_w_for_reward: Optional[torch.Tensor] = None
        foot_contact_mask_for_custom_align: Optional[torch.Tensor] = None
        num_feet_in_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        if self._foot_link_articulation_indices and self._foot_link_sensor_indices and \
           len(self._foot_link_articulation_indices) > 0 and \
           len(self._foot_link_articulation_indices) == len(self._foot_link_sensor_indices):
            if hasattr(self.robot.data, "body_quat_w") and self.robot.data.body_quat_w is not None:
                articulation_indices_tensor = torch.tensor(self._foot_link_articulation_indices, device=self.device, dtype=torch.long)
                if articulation_indices_tensor.numel() > 0 and articulation_indices_tensor.max() < self.robot.data.body_quat_w.shape[1]:
                     foot_quats_w_for_reward = self.robot.data.body_quat_w[:, articulation_indices_tensor]
            if hasattr(self._contact_sensor.data, 'net_forces_w_history') and \
               self._contact_sensor.data.net_forces_w_history is not None:
                all_forces_contact = self._contact_sensor.data.net_forces_w_history
                sensor_indices_tensor = torch.tensor(self._foot_link_sensor_indices, device=self.device, dtype=torch.long)
                if sensor_indices_tensor.numel() > 0 and all_forces_contact.ndim == 4 and all_forces_contact.shape[1] > 0 and \
                   sensor_indices_tensor.max() < all_forces_contact.shape[2]:
                    forces_on_these_feet = all_forces_contact[:, -1, sensor_indices_tensor, :]
                    foot_contact_magnitudes = torch.norm(forces_on_these_feet, dim=-1)
                    foot_contact_mask_for_custom_align = foot_contact_magnitudes > 1.0
                    num_feet_in_contact = torch.sum(foot_contact_mask_for_custom_align.long(), dim=1)


        total_reward, reward_terms_dict = compute_sixfeet_rewards_directional(
            root_lin_vel_b, root_ang_vel_b, projected_gravity_b, joint_pos_rel, joint_vel, applied_torque, joint_acc,
            self._q_lower_limits, self._q_upper_limits, current_joint_pos_abs, self._policy_actions, self._previous_policy_actions,
            root_pos_w, undesired_contacts_active, self._commands, self.cfg.command_profile,
            self.cfg.rew_scale_move_in_commanded_direction, self.cfg.rew_scale_achieve_reference_angular_rate,
            self.cfg.rew_scale_alive, self.cfg.rew_scale_target_height, self.cfg.target_height_m,
            self.cfg.rew_scale_action_cost, self.cfg.rew_scale_action_rate, self.cfg.rew_scale_joint_torques, self.cfg.rew_scale_joint_accel,
            self.cfg.rew_scale_lin_vel_z_penalty, self.cfg.rew_scale_ang_vel_xy_penalty, self.cfg.rew_scale_flat_orientation,
            self.cfg.rew_scale_unwanted_movement_penalty, self.cfg.rew_scale_dof_at_limit, self.cfg.rew_scale_toe_orientation_penalty,
            self._toe_joint_indices, self.cfg.rew_scale_low_height_penalty, self.cfg.min_height_penalty_threshold,
            self.cfg.rew_scale_undesired_contact,
            self.sim.cfg.dt,
            getattr(self.cfg, "rew_scale_orientation_deviation", 0.0),
            self._orientation_termination_angle_limit_rad,
            getattr(self.cfg, "joint_limit_penalty_threshold_percent", 0.05),
            num_self_collisions_per_env,
            getattr(self.cfg, "rew_scale_self_collision", 0.0),
            self._was_severely_tilted_last_step,
            getattr(self.cfg, "rew_scale_successful_flip", 0.0),
            getattr(self.cfg, "target_height_reward_sharpness", 50.0),
            # vv --- 确保以下参数的顺序与函数定义中的一致 --- vv
            foot_quats_w_for_reward,                         # Tensor? foot_quats_w
            foot_contact_mask_for_custom_align,              # Tensor? foot_contact_mask (用于特定轴对齐)
            self._foot_link_target_align_axes_tensor,        # Tensor? foot_link_target_align_axes
            self._foot_link_target_align_world_axes_tensor,  # Tensor? foot_link_target_align_world_axes
            getattr(self.cfg, "rew_scale_custom_foot_axis_alignment", 0.0), # float cfg_rew_scale_custom_foot_axis_alignment
            num_feet_in_contact,                             # Tensor num_feet_in_contact
            getattr(self.cfg, "rew_scale_all_feet_stable_stand", 0.0), # float cfg_rew_scale_all_feet_stable_stand
            getattr(self.cfg, "rew_scale_airborne_feet_penalty", 0.0)  # float cfg_rew_scale_airborne_feet_penalty
            # ^^ --- 参数顺序调整结束 --- ^^
        )

        self._was_severely_tilted_last_step = current_is_severely_tilted_for_flip_reward.clone()

        # ... (日志记录和返回逻辑) ...
        if "log" not in self.extras or self.extras["log"] is None : self.extras["log"] = {}
        for key, value in reward_terms_dict.items():
            term_mean = value.mean(); self.extras["log"][f"reward_term/{key}_step_avg"] = term_mean.item() if torch.is_tensor(term_mean) else term_mean
            if key not in self._episode_reward_terms_sum: self._episode_reward_terms_sum[key] = torch.zeros(self.num_envs, device=self.device)
            self._episode_reward_terms_sum[key] += value.squeeze(-1) if value.ndim > 1 and value.shape[-1] == 1 else value
        current_terminated, current_time_out = self._get_dones()
        just_failed_termination = current_terminated & (~current_time_out)
        final_reward = torch.where(just_failed_termination, torch.full_like(total_reward, getattr(self.cfg, "rew_scale_termination", -200.0)), total_reward)
        self.extras["log"]["reward/final_reward_avg"] = final_reward.mean().item() if torch.is_tensor(final_reward) else final_reward.mean()
        return final_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]: # 与上一版本相同
        # ... (代码与上一版本完全一致，包含了条件化终止逻辑) ...
        time_out = self.episode_length_buf >= self.max_episode_length - 1; root_pos_w = self.robot.data.root_pos_w; projected_gravity_b = self.robot.data.projected_gravity_b
        orientation_limit_rad = getattr(self, "_orientation_termination_angle_limit_rad", math.radians(90.0)); cos_angle_robot_z_with_world_z = -projected_gravity_b[:, 2]
        is_within_termination_orientation_limit = cos_angle_robot_z_with_world_z >= math.cos(orientation_limit_rad)
        height_too_low_orig = root_pos_w[:, 2] < self.cfg.termination_height_thresh
        sum_sq_proj_grav_xy = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
        flatness_thresh = getattr(self.cfg, "flatness_threshold_for_height_termination", 0.2)
        is_relatively_flat = sum_sq_proj_grav_xy < flatness_thresh
        height_too_low = height_too_low_orig & is_within_termination_orientation_limit & is_relatively_flat
        fallen_over_orig = projected_gravity_b[:, 2] > self.cfg.termination_body_z_thresh
        fallen_over = fallen_over_orig & is_within_termination_orientation_limit
        base_contact_termination = torch.zeros_like(time_out, dtype=torch.bool)
        if hasattr(self.cfg, "termination_base_contact") and self.cfg.termination_base_contact:
            base_contact_raw_detection = torch.zeros_like(time_out, dtype=torch.bool)
            if self._base_body_id and len(self._base_body_id) > 0:
                if hasattr(self._contact_sensor.data, 'net_forces_w_history') and self._contact_sensor.data.net_forces_w_history is not None:
                    all_forces = self._contact_sensor.data.net_forces_w_history
                    if all_forces.ndim == 4 and all_forces.shape[1] > 0 and self._base_body_id and max(self._base_body_id) < all_forces.shape[2]:
                        forces = all_forces[:, -1, self._base_body_id, :]; base_contact_raw_detection = torch.any(torch.norm(forces, dim=-1) > 1.0, dim=1)
            current_time_in_episode = (self.episode_length_buf + 1).float() * self.physics_dt
            grace_period_seconds = 1.0; ignore_initial_fall_contact = current_time_in_episode <= grace_period_seconds
            valid_base_contact_trigger = base_contact_raw_detection & (~ignore_initial_fall_contact)
            base_contact_termination = valid_base_contact_trigger & is_within_termination_orientation_limit
        terminated = height_too_low | fallen_over | base_contact_termination | time_out
        # if torch.is_tensor(terminated) and terminated.any():
        #     terminated_env_indices = terminated.nonzero(as_tuple=False).squeeze(-1)
        #     if terminated_env_indices.ndim == 0: terminated_env_indices = terminated_env_indices.unsqueeze(0)
        #     for env_idx_tensor in terminated_env_indices:
        #         env_idx = env_idx_tensor.item(); reasons = []
        #         if height_too_low[env_idx]: reasons.append("height_too_low(cond_flat)")
        #         if fallen_over[env_idx]: reasons.append("fallen_over(cond_orient)")
        #         if base_contact_termination[env_idx]: reasons.append("base_contact(cond_orient_time)")
        #         if time_out[env_idx]: reasons.append("time_out")
        #         if reasons: episode_step = self.episode_length_buf[env_idx].item() + 1; print(f"[Termination Info] Env {env_idx}: Reset at ep step {episode_step} due to: {', '.join(reasons)}")
        return terminated, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None): # 与上一版本相同
        super()._reset_idx(env_ids)
        eids = torch.arange(self.num_envs, device=self.device, dtype=torch.long) if env_ids is None \
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if eids.numel() == 0: return
        root_state_reset = self.robot.data.default_root_state[eids].clone()
        if hasattr(self._terrain, 'env_origins') and self._terrain.env_origins is not None: root_state_reset[:, :3] += self._terrain.env_origins[eids]
        initial_height_base = self.cfg.robot.init_state.pos[2] if self.cfg.robot.init_state.pos is not None and len(self.cfg.robot.init_state.pos) == 3 else 0.3
        reset_height_offset = getattr(self.cfg, "reset_height_offset", 0.0)
        root_state_reset[:, 2] = initial_height_base + reset_height_offset
        num_resets = len(eids); quats_xyzw = torch.zeros(num_resets, 4, device=self.device)
        randomization_mode = getattr(self.cfg, "initial_pose_randomization_mode", 1)
        if randomization_mode == 1:
            random_rolls=(torch.rand(num_resets,device=self.device)-0.5)*2.0*math.pi; random_pitches=(torch.rand(num_resets,device=self.device)-0.5)*2.0*math.pi; random_yaws=(torch.rand(num_resets,device=self.device)-0.5)*2.0*math.pi
            for i in range(num_resets): r,p,y=random_rolls[i],random_pitches[i],random_yaws[i]; cy,sy=torch.cos(y*0.5),torch.sin(y*0.5); cp,sp=torch.cos(p*0.5),torch.sin(p*0.5); cr,sr=torch.cos(r*0.5),torch.sin(r*0.5); quats_xyzw[i,3]=cr*cp*cy+sr*sp*sy; quats_xyzw[i,0]=sr*cp*cy-cr*sp*sy; quats_xyzw[i,1]=cr*sp*cy+sr*cp*sy; quats_xyzw[i,2]=cr*cp*sy-sr*sp*cy
        elif randomization_mode == 0:
            rolls=torch.full((num_resets,),math.pi,device=self.device); pitches=torch.zeros(num_resets,device=self.device); random_yaws=(torch.rand(num_resets,device=self.device)-0.5)*2.0*math.pi
            for i in range(num_resets): r,p,y=rolls[i],pitches[i],random_yaws[i]; cy,sy=torch.cos(y*0.5),torch.sin(y*0.5); cp,sp=torch.cos(p*0.5),torch.sin(p*0.5); cr,sr=torch.cos(r*0.5),torch.sin(r*0.5); quats_xyzw[i,3]=cr*cp*cy+sr*sp*sy; quats_xyzw[i,0]=sr*cp*cy-cr*sp*sy; quats_xyzw[i,1]=cr*sp*cy+sr*cp*sy; quats_xyzw[i,2]=cr*cp*sy-sr*sp*cy
        elif randomization_mode == 2: print_once("[INFO] Resetting with USD default root orientation.")
        else: print_once(f"[WARNING] Unknown initial_pose_randomization_mode: {randomization_mode}. Using default.")
        if randomization_mode==1 or randomization_mode==0: root_state_reset[:,3:7]=convert_quat(quats_xyzw,to="wxyz")
        root_state_reset[:, 7:] = 0.0; self.robot.write_root_state_to_sim(root_state_reset, eids)
        num_dof = self._q_lower_limits.numel()
        if getattr(self.cfg, "randomize_initial_joint_poses", True):
            random_proportions = torch.rand(len(eids), num_dof, device=self.device); q_lower_expanded = self._q_lower_limits.unsqueeze(0); q_range = self._q_upper_limits.unsqueeze(0) - q_lower_expanded; joint_pos_reset = q_lower_expanded + random_proportions * q_range
        else: joint_pos_reset = self._default_joint_pos.unsqueeze(0).expand(len(eids), -1)
        zero_joint_vel = torch.zeros_like(joint_pos_reset)
        self.robot.write_joint_state_to_sim(joint_pos_reset, zero_joint_vel, env_ids=eids); self.robot.set_joint_position_target(joint_pos_reset, env_ids=eids)
        if hasattr(self, "_was_severely_tilted_last_step"): self._was_severely_tilted_last_step[eids] = True 
        cmd_profile = self.cfg.command_profile
        cmd_duration_s_config = cmd_profile.get("command_mode_duration_s", self.cfg.episode_length_s if hasattr(self.cfg, "episode_length_s") else 20.0)
        if isinstance(cmd_duration_s_config, str) and cmd_duration_s_config == "episode_length_s": cmd_duration = self.cfg.episode_length_s
        else:
            try: cmd_duration = float(cmd_duration_s_config)
            except (ValueError, TypeError): cmd_duration = self.cfg.episode_length_s if hasattr(self.cfg, "episode_length_s") else 20.0
        self._time_since_last_command_change[eids] = cmd_duration; self._update_commands(eids)
        if hasattr(self, '_previous_policy_actions'): self._previous_policy_actions[eids] = 0.0
        if hasattr(self, '_policy_actions'): self._policy_actions[eids] = 0.0
        for key in list(self._episode_reward_terms_sum.keys()):
            if self._episode_reward_terms_sum[key].shape[0] == self.num_envs : self._episode_reward_terms_sum[key][eids] = 0.0