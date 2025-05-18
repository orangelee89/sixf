# sixfeet_env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import torch, math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets  import Articulation
from isaaclab.envs    import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate

from .sixfeet_env_cfg import SixfeetEnvCfg

# ───────────────── 奖励/惩罚 ──────────────────
@torch.jit.script
def reward_upright(q_xyzw: torch.Tensor) -> torch.Tensor:
    z = torch.zeros((q_xyzw.shape[0], 3), device=q_xyzw.device)
    z[:, 2] = 1.0
    body_z = quat_rotate(q_xyzw, z)
    cos_t  = body_z[..., 2].clamp(-1.0, 1.0)
    return torch.exp(-(1.0 - cos_t) / 0.1)

@torch.jit.script
def penalty_ang_vel(w: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(w, dim=-1)

@torch.jit.script
def penalty_torque(tau: torch.Tensor) -> torch.Tensor:
    return (tau ** 2).sum(-1)

# ─────────────────── Env ──────────────────────
class SixfeetEnv(DirectRLEnv):
    cfg: SixfeetEnvCfg

    def __init__(self, cfg: SixfeetEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)   # <-- 让父类把 robot / scene 建好

        # 一次性读取关节限位（CPU tensor）
        lim = self.robot.data.joint_pos_limits[0].to(self.device)   # 形状 (J, 2)
        self._q_lo = lim[:, 0]                                      # (18,)
        self._q_hi = lim[:, 1]                          # (18,)

    # ---------- scene ----------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=2_000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)  

        self.scene.clone_environments(copy_from_source=False)

    # ---------- RL loop ----------
    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.to(self.device)   # (N,18)
        abs_max = torch.abs(actions).amax(dim=1, keepdim=True)   # (B,1)
        scale   = torch.clamp(abs_max, min=1.0)                  # 不足 1 -> 1
        self.actions = actions / scale  

    def _apply_action(self):
        mid  = (self._q_hi + self._q_lo) * 0.5                         # (18,)
        half = (self._q_hi - self._q_lo) * 0.5                         # (18,)

        q_tgt = mid + half * (self.actions * self.cfg.action_scale)  # broadcasting OK
        self.robot.set_joint_position_target(q_tgt)
        self.robot.set_joint_position_target(q_tgt) 



    def _get_observations(self):
        obs = torch.cat(
            (
                self.robot.data.root_quat_w.to(self.device),
                self.robot.data.root_ang_vel_w.to(self.device),
                self.robot.data.joint_pos.to(self.device),
                self.robot.data.joint_vel.to(self.device),
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self):
        quat_w   = self.robot.data.root_quat_w.to(self.device)
        ang_vel  = self.robot.data.root_ang_vel_w.to(self.device)
        tau      = self.robot.data.applied_torque.to(self.device)
        root_pos = self.robot.data.root_pos_w.to(self.device)

        # -------- upright / align --------
        r_up = reward_upright(quat_w)

        N = quat_w.shape[0]
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(N, 3)
        body_z = quat_rotate(quat_w, z_axis)           # (N,3)
        r_align = body_z[..., 2].clamp(0.0, 1.0)

        # 1° 阈值
        cos_thr = math.cos(math.radians(1.0))
        success = (root_pos[:, 2] > 0.35) & (body_z[..., 2] >= cos_thr)
        done_bonus = torch.where(
            success,
            torch.full_like(root_pos[:, 2], self.cfg.bonus_upright_done),
            torch.zeros_like(root_pos[:, 2])
        )

        # 其余惩罚……
        p_av  = penalty_ang_vel(ang_vel)
        p_tau = penalty_torque(tau)
        ground_punish    = torch.zeros(N, device=self.device)
        collision_punish = torch.zeros(N, device=self.device)

        reward = (
            self.cfg.rew_scale_upright  * r_up
            + self.cfg.rew_scale_align_z * r_align
            + self.cfg.rew_scale_angvel  * (-p_av)
            + self.cfg.rew_scale_torque  * (-p_tau)
            + ground_punish
            + collision_punish
            + done_bonus
        )

        self.extras["log"] = {
            "align_z": (self.cfg.rew_scale_align_z * r_align).mean(),
            "success_rate": success.float().mean(),
            "reward": reward.mean(),    
        }
        return reward

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    # ---------- reset ----------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        eids = torch.as_tensor(env_ids, device=self.device)

        # 随机根姿态
        rng  = self.cfg.root_orientation_range
        rpy  = (torch.rand((len(eids), 3), device=self.device) - 0.5) * 2 * rng
        cr, sr = torch.cos(rpy[:, 0]/2), torch.sin(rpy[:, 0]/2)
        cp, sp = torch.cos(rpy[:, 1]/2), torch.sin(rpy[:, 1]/2)
        cy, sy = torch.cos(rpy[:, 2]/2), torch.sin(rpy[:, 2]/2)
        q_xyzw = torch.stack([cy*cp*sr - sy*sp*cr,
                              sy*cp*sr + cy*sp*cr,
                              sy*cp*cr - cy*sp*sr,
                              cy*cp*cr + sy*sp*sr], dim=-1)

        root = self.robot.data.default_root_state[eids].clone()
        root[:, 3:7] = q_xyzw
        root[:, 2]  += self.cfg.reset_height_offset        # ← 用 cfg 参数抬高
        self.robot.write_root_state_to_sim(root, eids)

        # 关节置回 USD 默认
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[eids],
            self.robot.data.default_joint_vel[eids],
            None, env_ids=eids,
        )