# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# 六足机器人环境：保持水平 + 扶正

from __future__ import annotations

import torch
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate

from .sixfeet_env_cfg import SixfeetEnvCfg


# ───────────────────────────────── TorchScript 奖励函数 ──────────────────────────
@torch.jit.script
def reward_upright(root_quat: torch.Tensor) -> torch.Tensor:
    """
    指标: 机体 +Z 与世界 +Z 夹角 θ
    公式: exp(-(1-cosθ)/0.1)  → θ=0°:1.0, θ=90°:0.36, θ=180°:0.14
    """
    B = root_quat.shape[0]
    world_up = root_quat.new_zeros((B, 3))
    world_up[:, 2] = 1.0                         # (0,0,1)
    body_up = quat_rotate(root_quat, world_up)   # (B,3)
    z_cos = body_up[..., 2].clamp(-1.0, 1.0)     # 取对齐度
    return torch.exp(-(1.0 - z_cos) / 0.1)       # (B,)


@torch.jit.script
def penalty_ang_vel(root_ang_vel: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(root_ang_vel, dim=-1)


@torch.jit.script
def penalty_torque(joint_tau: torch.Tensor) -> torch.Tensor:
    return (joint_tau ** 2).sum(-1)
# ────────────────────────────────────────────────────────────────────────────────


class SixfeetEnv(DirectRLEnv):
    """六足机器人在任意朝向下扶正并保持水平"""
    cfg: SixfeetEnvCfg

    # --------------------------------------------------------------------- #
    # 初始化
    # --------------------------------------------------------------------- #
    def __init__(self, cfg: SixfeetEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 方便访问的缓存
        self.root_quat = self.robot.data.root_quat_w(device=self.device)
        self.root_ang_vel = self.robot.data.root_ang_vel_w(device=self.device)
        self.joint_pos = self.robot.data.joint_pos(device=self.device)
        self.joint_vel = self.robot.data.joint_vel(device=self.device)
        self.joint_tau = self.robot.data.applied_torque(device=self.device)

        # 读取关节限位
        limits = self.robot.data.joint_pos_limits(device=self.device)[0]  # (18,2)
        self.joint_lower_limits = limits[:, 0]
        self.joint_upper_limits = limits[:, 1]

        self.all_joint_ids = torch.arange(self.joint_pos.shape[1], device=self.device)

    # --------------------------------------------------------------------- #
    # 场景搭建
    # --------------------------------------------------------------------- #
    def _setup_scene(self):
        # 机器人
        self.robot = Articulation(self.cfg.robot_cfg)

        # 地面
        spawn_ground_plane("/World/ground", GroundPlaneCfg())

        # 克隆 env
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        # 环境光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # --------------------------------------------------------------------- #
    # RL 生命周期钩子
    # --------------------------------------------------------------------- #
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()(device=self.device)

    def _apply_action(self) -> None:
        # 将 [-1,1] 缩放到各关节限位
        scaled = self.actions * (self.joint_upper_limits - self.joint_lower_limits) / 2
        scaled += (self.joint_upper_limits + self.joint_lower_limits) / 2
        self.robot.set_joint_position_target(scaled)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (self.root_quat,                 # 4
             self.root_ang_vel,              # 3
             self.joint_pos,                 # 18
             self.joint_vel),                # 18
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        r_up = reward_upright(self.root_quat)
        p_av = penalty_ang_vel(self.root_ang_vel)
        p_tau = penalty_torque(self.joint_tau)
        # p_col = self.robot.get_collision_penalty()

        return (
            self.cfg.rew_scale_upright * r_up
            + self.cfg.rew_scale_angvel * (-p_av)
            + self.cfg.rew_scale_torque * (-p_tau)
            # + self.cfg.rew_scale_collision * p_col
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    # --------------------------------------------------------------------- #
    # Reset 随机化
    # --------------------------------------------------------------------- #
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # 随机 root 姿态
        rng = self.cfg.root_orientation_range
        rpy = (torch.rand((len(env_ids), 3), device=self.device) - 0.5) * 2 * rng

        cr, sr = torch.cos(rpy[:, 0] / 2), torch.sin(rpy[:, 0] / 2)
        cp, sp = torch.cos(rpy[:, 1] / 2), torch.sin(rpy[:, 1] / 2)
        cy, sy = torch.cos(rpy[:, 2] / 2), torch.sin(rpy[:, 2] / 2)

        quat = torch.stack(
            (
                cy * cp * cr + sy * sp * sr,
                cy * cp * sr - sy * sp * cr,
                sy * cp * sr + cy * sp * cr,
                sy * cp * cr - cy * sp * sr,
            ),
            dim=-1,
        )

        default_root = self.robot.data.default_root_state[env_ids]
        default_root[:, 3:7] = quat                   # orientation
        default_root[:, 2] += 0.1                    # 抬高 0.1 m 防穿地

        self.robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)

        # 关节归零
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids](device=self.device),
            self.robot.data.default_joint_vel[env_ids](device=self.device),
            None,
            env_ids,
        )