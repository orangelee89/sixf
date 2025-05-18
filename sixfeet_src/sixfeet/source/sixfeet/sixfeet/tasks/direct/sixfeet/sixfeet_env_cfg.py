# sixfeet_env_cfg.py
# ────────────────────────────────────────────────────────────
from __future__ import annotations
import math
import isaaclab.sim as sim_utils
from isaaclab.utils      import configclass
from isaaclab.envs       import DirectRLEnvCfg
from isaaclab.sim        import SimulationCfg, PhysxCfg
from isaaclab.assets     import ArticulationCfg
from isaaclab.actuators  import ImplicitActuatorCfg
from isaaclab.sensors    import ContactSensorCfg
from isaaclab.scene      import InteractiveSceneCfg

@configclass
class SixfeetEnvCfg(DirectRLEnvCfg):
    # ────────── 基础 ──────────
    decimation           = 2                # 控制 60 Hz (=120/2)
    episode_length_s     = 1_000
    action_space         = 18
    observation_space    = 4 + 3 + 18 * 2
    state_space          = 0

    # ────────── 物理 ──────────
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=decimation,
        device="cuda:0",
        physx=PhysxCfg(enable_ccd=True, solver_type=1),
    )

    # ────────── 执行器 ─────────
    all_pd = ImplicitActuatorCfg(joint_names_expr=".*", stiffness=600.0, damping=3.0)

    # ────────── 机器人 ─────────
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        actuators={"all": all_pd},
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/lee/EE_ws/src/sixfeet_src/sixfeet/source/"
                     "sixfeet/sixfeet/assets/hexapod_2/hexapod_2.usd",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
            ),
        ),
    )



    # ────────── 场景 ──────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=4.0,
        replicate_physics=True,
        
    )

    # ────────── 奖励缩放 ────────
    action_scale         = 0.5
    rew_scale_upright    = +5.0
    rew_scale_angvel     = -0.1
    rew_scale_torque     = -2e-4
    rew_scale_collision  = -10.0
    rew_scale_align_z  = +3.0   # Z 轴对齐奖励权重
    bonus_upright_done = +50.0 

    # ────────── Reset 随机 ──────
    root_orientation_range = math.pi
    reset_height_offset    = 0.10
