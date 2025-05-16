# SPDX-License-Identifier: BSD-3-Clause
#
# Hexapod self-righting task — Manager-Based RL workflow.
# Drop this file in: isaaclab_tasks/manager_based/hexapod_balance/

import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate
import torch
from isaaclab.utils.math import quat_rotate 

# ---------- Helper reward / penalty functions ---------- #

WORLD_UP = torch.tensor([0.0, 0.0, 1.0])

def reward_upright(asset, env_ids, params=None):
    body_up = quat_rotate(asset.data.root_quat[env_ids], WORLD_UP.to(asset.device))
    sin_theta = torch.linalg.norm(torch.cross(body_up, WORLD_UP.to(asset.device)), dim=-1)
    return torch.exp(-sin_theta / 0.05)

def penalty_root_ang_vel(asset, env_ids, params=None):
    ang_vel = asset.data.root_ang_vel[env_ids]
    return torch.linalg.norm(ang_vel, dim=-1)

# torque 惩罚无需改动


def penalty_joint_torque(asset, env_ids, params=None):
    """全部关节扭矩平方和"""
    torque = asset.data.applied_torque[env_ids]
    return (torque ** 2).sum(-1)

# ---------- Asset ---------- #

HEXAPOD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/lee/EE_ws/src/robot_urdf/urdf/hexapod_2/hexapod.usd",
        # 如需额外物理属性覆盖，取消下行注释：
        # articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #     enabled_self_collisions=False
        # ),
    ),
)

# ---------- Scene ---------- #

@configclass
class HexapodSceneCfg(InteractiveSceneCfg):
    """地面 + 六足机器人"""
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(20.0, 20.0)),
    )
    robot: ArticulationCfg = HEXAPOD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"   # 每个并行 env 独立路径
    )

# ---------- Random-orientation reset ---------- #

def reset_root_with_random_orientation(asset, env_ids, params=None):
    roll  = (torch.rand(len(env_ids), device=asset.device) - 0.5) * 2 * math.pi
    pitch = (torch.rand(len(env_ids), device=asset.device) - 0.5) * 2 * math.pi
    yaw   = (torch.rand(len(env_ids), device=asset.device) - 0.5) * 2 * math.pi
    quat = torch.stack([
        torch.cos(roll/2)*torch.cos(pitch/2)*torch.cos(yaw/2) +
        torch.sin(roll/2)*torch.sin(pitch/2)*torch.sin(yaw/2),

        torch.sin(roll/2)*torch.cos(pitch/2)*torch.cos(yaw/2) -
        torch.cos(roll/2)*torch.sin(pitch/2)*torch.sin(yaw/2),

        torch.cos(roll/2)*torch.sin(pitch/2)*torch.cos(yaw/2) +
        torch.sin(roll/2)*torch.cos(pitch/2)*torch.sin(yaw/2),

        torch.cos(roll/2)*torch.cos(pitch/2)*torch.sin(yaw/2) -
        torch.sin(roll/2)*torch.sin(pitch/2)*torch.cos(yaw/2),
    ], dim=-1)
    asset._default_root_state[env_ids, 3:7] = quat

# ---------- Events ---------- #

@configclass
class EventCfg:
    random_orientation = EventTerm(
        func=reset_root_with_random_orientation,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

# ---------- Rewards ---------- #

@configclass
class RewardsCfg:
    upright = RewTerm(
        func=reward_upright, weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    ang_vel = RewTerm(
        func=penalty_root_ang_vel, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    torque  = RewTerm(
        func=penalty_joint_torque, weight=-2e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

# ---------- Terminations ---------- #

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=lambda *a, **kw: False, time_out=True)  # 仅超时结束

# ---------- Environment master config ---------- #

@configclass
class HexapodBalanceEnvCfg(ManagerBasedRLEnvCfg):
    """六足自扶正（Manager-Based 工作流）"""
    scene: HexapodSceneCfg = HexapodSceneCfg(num_envs=4096, env_spacing=4.0)
    actions = ManagerBasedRLEnvCfg.actions   # 默认：全 DOF 位置控制
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2          # 60 Hz 控制
        self.episode_length_s = 6.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
