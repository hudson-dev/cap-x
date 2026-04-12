"""MuJoCo closed-loop evaluation for BC-trained RobotAgent policies.

Single-process eval with async-equivalent timing semantics: actions are
timestamped with sim_time and applied via ActionBuffer during physics substeps.
sim_time advances only from MuJoCo stepping (decoupled from wall-clock time).

Renders from fixed exo camera (fisheye) + wrist cameras (pinhole). No eye
control. Observation format matches bc_robot.py training: exo_image,
wrist_left_image, wrist_right_image, proprio.

Usage:
    python -m eye.mujoco.eval_bc --ckpt runs/bc_run/checkpoint_final.pt --n-episodes 10

    # With side-view video panel
    python -m eye.mujoco.eval_bc --ckpt runs/bc_run/checkpoint_final.pt --side-view

    # Custom exo resolution (gets resized to crop_size for DINO anyway)
    python -m eye.mujoco.eval_bc --ckpt runs/bc_run/checkpoint_final.pt --exo-width 1920 --exo-height 1200
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from eye.mujoco.action_buffer import ActionBuffer
from eye.mujoco.eval import (
    XML_PATH,
    JOINT_NAMES,
    FINGER_JOINT_MIN,
    FINGER_JOINT_RANGE,
    _gripper_agent_to_mujoco,
    make_eval_dir,
)
from eye.mujoco.mjwarp_utils import MujocoRenderer
from eye.mujoco.tasks import TASK_REGISTRY
from eye.ik_utils import process_action_chunk


class BCEvalEnv:
    """MuJoCo eval environment for BC-trained RobotAgent policies.

    Renders from fixed exo + wrist cameras (no eye control) and produces
    observations matching the bc_robot.py training format.

    Args:
        xml_path: Path to base scene XML.
        task: Task instance defining the manipulation task.
        agent_config: RobotAgentConfig from checkpoint.
        exo_width: Exo camera render width (default 768).
        exo_height: Exo camera render height (default 480).
        agent_fps: Agent inference frequency in Hz.
    """

    def __init__(
        self,
        xml_path: str | Path,
        task,
        agent_config,
        exo_width: int = 768,
        exo_height: int = 480,
        agent_fps: float = 30.0,
        ensemble_k: float = 0.01,
    ):
        # --- MjSpec pipeline: base XML -> task objects -> compile ---
        spec = mujoco.MjSpec.from_file(str(xml_path))
        task.configure_scene(spec)
        self.mjm = spec.compile()
        self.mjd = mujoco.MjData(self.mjm)

        self.physics_dt = self.mjm.opt.timestep
        self.agent_fps = agent_fps
        self.agent_dt = 1.0 / agent_fps
        self.substeps = int(round(self.agent_dt / self.physics_dt))

        # --- Task setup ---
        self.task = task
        task.setup(self.mjm, self.mjd)

        # --- Camera renderers (matching postprocess_sim_teleop.py setup) ---
        self.load_exo = getattr(agent_config, "load_exo", True)
        self.load_wrist_left = getattr(agent_config, "load_wrist_left", True)
        self.load_wrist_right = getattr(agent_config, "load_wrist_right", True)

        self.exo_renderer = None
        self.wrist_left_renderer = None
        self.wrist_right_renderer = None

        if self.load_exo:
            self.exo_renderer = MujocoRenderer(
                self.mjm, "exo_camera", exo_width, exo_height, fisheye=True,
            )
        if self.load_wrist_left:
            self.wrist_left_renderer = MujocoRenderer(
                self.mjm, "left_wrist_camera", 640, 480, fisheye=False,
            )
        if self.load_wrist_right:
            self.wrist_right_renderer = MujocoRenderer(
                self.mjm, "right_wrist_camera", 640, 480, fisheye=False,
            )

        # --- Agent config ---
        self.crop_size = agent_config.crop_size
        self.config = agent_config

        # --- FK infrastructure for SE3 proprio ---
        self._has_joint = any(at in ('joint_abs', 'joint_rel') for at in agent_config.action_types)
        self._has_se3 = any(at.startswith('se3_') for at in agent_config.action_types)
        if self._has_se3:
            import yourdfpy
            from pyroki import Robot
            urdf_path = Path('./urdf/yam_bimanual.urdf')
            urdf = yourdfpy.URDF.load(urdf_path, load_meshes=False)
            self._pk_robot = Robot.from_urdf(urdf)
            self._left_ee_link_index = self._pk_robot.links.names.index("left_tool0")
            self._right_ee_link_index = self._pk_robot.links.names.index("right_tool0")
        else:
            self._pk_robot = None

        # --- Joint/actuator IDs ---
        self.joint_ids = [
            mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]

        # --- Action buffer ---
        self.action_buffer = ActionBuffer(
            chunk_size=agent_config.action_chunk_size,
            action_dim=14,
            agent_fps=agent_fps,
            k=ensemble_k,
        )

        # --- Home position from config ---
        self.home_joints: np.ndarray | None = None
        if hasattr(agent_config, "mean_start_position") and agent_config.mean_start_position:
            home = agent_config.mean_start_position
            if isinstance(home, torch.Tensor):
                self.home_joints = home.float().cpu().numpy().flatten()
            else:
                self.home_joints = np.array(home, dtype=np.float32).flatten()
            print(f"Demo home position ({len(self.home_joints)} DOF): {self.home_joints}")

        # --- Timing ---
        self.sim_time = 0.0
        self.agent_step_count = 0

        # --- Cached frames for video ---
        self.last_exo_rgb: np.ndarray | None = None
        self.last_wrist_left_rgb: np.ndarray | None = None
        self.last_wrist_right_rgb: np.ndarray | None = None

        # --- Optional external-view renderers ---
        self.side_view_renderer: mujoco.Renderer | None = None
        self.side_view_cam: mujoco.MjvCamera | None = None
        self.side_view_scene_option: mujoco.MjvOption | None = None

        self.top_view_renderer: mujoco.Renderer | None = None
        self.top_view_cam: mujoco.MjvCamera | None = None
        self.top_view_scene_option: mujoco.MjvOption | None = None

    def enable_side_view(self, width: int = 640, height: int = 480):
        self.side_view_renderer = mujoco.Renderer(self.mjm, height, width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.35, 0.0, 0.0]
        cam.distance = 1.2
        cam.azimuth = 90.0
        cam.elevation = -5.0
        self.side_view_cam = cam
        self.side_view_scene_option = mujoco.MjvOption()

    def render_side_view(self) -> np.ndarray | None:
        if self.side_view_renderer is None:
            return None
        self.side_view_renderer.update_scene(
            self.mjd, self.side_view_cam, scene_option=self.side_view_scene_option,
        )
        self.side_view_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        return self.side_view_renderer.render()

    def enable_top_view(self, width: int = 640, height: int = 480):
        self.top_view_renderer = mujoco.Renderer(self.mjm, height, width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.35, 0.0, 0.0]
        cam.distance = 1.2
        cam.azimuth = 90.0
        cam.elevation = -90.0
        self.top_view_cam = cam
        self.top_view_scene_option = mujoco.MjvOption()

    def render_top_view(self) -> np.ndarray | None:
        if self.top_view_renderer is None:
            return None
        self.top_view_renderer.update_scene(
            self.mjd, self.top_view_cam, scene_option=self.top_view_scene_option,
        )
        self.top_view_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        return self.top_view_renderer.render()

    # ──────────────────────────────────────────────
    # Reset / Step / Obs
    # ──────────────────────────────────────────────

    def reset(self, config: dict) -> dict[str, torch.Tensor]:
        """Reset environment for a new episode.

        Args:
            config: Task config dict from task.generate_eval_configs().

        Returns:
            Observation dict with shapes (B=1, T=1, ...).
        """
        # 1. Reset to home keyframe
        mujoco.mj_resetDataKeyframe(self.mjm, self.mjd, 0)

        # 2. Override arm joints with demo home position
        if self.home_joints is not None:
            for i, jid in enumerate(self.joint_ids):
                val = self.home_joints[i]
                if i == 6 or i == 13:  # gripper indices
                    val = _gripper_agent_to_mujoco(val)
                self.mjd.qpos[self.mjm.jnt_qposadr[jid]] = val
            # Set ctrl so PD controllers target home position
            self.mjd.ctrl[2:8] = self.home_joints[:6]
            self.mjd.ctrl[8] = _gripper_agent_to_mujoco(self.home_joints[6])
            self.mjd.ctrl[9:15] = self.home_joints[7:13]
            self.mjd.ctrl[15] = _gripper_agent_to_mujoco(self.home_joints[13])

        # 3. Apply task config (object poses)
        self.task.apply_eval_config(self.mjm, self.mjd, config)

        # 4. Forward kinematics
        mujoco.mj_forward(self.mjm, self.mjd)

        # 5. Warmup physics so PD controllers settle (grippers especially)
        for _ in range(500):
            mujoco.mj_step(self.mjm, self.mjd)

        # 6. Reset action buffer + timing
        self.action_buffer.reset()
        self.sim_time = 0.0
        self.agent_step_count = 0

        return self.get_obs()

    def get_obs(self) -> dict[str, torch.Tensor]:
        """Build BC observation dict.

        Returns dict with shapes (B=1, T=1, ...) matching RobotAgent input.
        Only keys the agent actually reads: exo_image, wrist cameras, proprio.
        """
        device = "cuda"
        obs = {}

        # 1. Exo camera — pass raw resolution to model (model handles resize/pad
        # in _prepare_common_tokens, matching training preprocessing)
        if self.exo_renderer is not None:
            exo_gpu = self.exo_renderer.render_torch(self.mjm, self.mjd, device=device)
            self.last_exo_rgb = self.exo_renderer.last_cpu_rgb
            obs["exo_image"] = exo_gpu.half().unsqueeze(0).unsqueeze(0)  # (3,H,W) -> (1,1,3,H,W) float16 to match training

        # 2. Wrist cameras
        if self.wrist_left_renderer is not None:
            wl_gpu = self.wrist_left_renderer.render_torch(self.mjm, self.mjd, device=device)
            self.last_wrist_left_rgb = self.wrist_left_renderer.last_cpu_rgb
            obs["wrist_left_image"] = wl_gpu.half().unsqueeze(0).unsqueeze(0)  # float16 to match training

        if self.wrist_right_renderer is not None:
            wr_gpu = self.wrist_right_renderer.render_torch(self.mjm, self.mjd, device=device)
            self.last_wrist_right_rgb = self.wrist_right_renderer.last_cpu_rgb
            obs["wrist_right_image"] = wr_gpu.half().unsqueeze(0).unsqueeze(0)  # float16 to match training

        # 3. Proprio — conditional on action types (joints, SE3, or both)
        raw_proprio = self._read_proprio()  # (1, 14) unnormalized
        parts = []
        if self._has_joint:
            normalized_joints = self._normalize_proprio(raw_proprio.to(device))
            parts.append(normalized_joints)
        if self._has_se3:
            from eye.inference.fk_se3 import proprio_to_10vec
            se3_proprio = proprio_to_10vec(
                raw_proprio.to(device), self._pk_robot,
                self._left_ee_link_index, self._right_ee_link_index, device
            )  # (1, 20)
            parts.append(se3_proprio)
        obs["proprio"] = torch.cat(parts, dim=-1).unsqueeze(0)  # (1, 1, D)

        return obs

    def step(
        self, joint_chunk: np.ndarray | None = None,
    ) -> tuple[dict[str, torch.Tensor], bool, dict[str, bool]]:
        """Apply joint actions and step physics (no eye control).

        Actions are timestamped with sim_time and interpolated via ActionBuffer
        during physics substeps. sim_time advances only from MuJoCo stepping.

        Args:
            joint_chunk: (chunk_size, 14) joint position targets, or None to
                step physics using existing buffer contents (for receding
                horizon mode between inference steps).

        Returns:
            Tuple of (observation dict, success bool, stage results dict).
        """
        # 1. Timestamp chunk with current sim_time (if provided)
        if joint_chunk is not None:
            self.action_buffer.add_chunk(joint_chunk, self.sim_time)


        # 2. Step physics (substeps per agent step)
        for _ in range(self.substeps):
            joint_targets = self.action_buffer.get_action(self.sim_time)
            if joint_targets is not None:
                # Left arm (actuators 2-7) + gripper (8)
                self.mjd.ctrl[2:8] = joint_targets[:6]
                self.mjd.ctrl[8] = _gripper_agent_to_mujoco(joint_targets[6])
                # Right arm (actuators 9-14) + gripper (15)
                self.mjd.ctrl[9:15] = joint_targets[7:13]
                self.mjd.ctrl[15] = _gripper_agent_to_mujoco(joint_targets[13])

            mujoco.mj_step(self.mjm, self.mjd)
            self.sim_time += self.physics_dt

        self.agent_step_count += 1

        obs = self.get_obs()
        stages = self.task.check_stages(self.mjm, self.mjd)
        success = stages[self.task.stages[-1]]
        return obs, success, stages

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    def _resize_to_crop(self, image: torch.Tensor) -> torch.Tensor:
        """Resize GPU image to (1, 1, 3, crop_size, crop_size).

        Scales by crop_size / min(h, w) then center-crops.
        Matches robot_eye_demo_bc_yam.py._resize_to_crop().

        Args:
            image: (3, H, W) float32 on GPU.
        """
        h, w = image.shape[-2:]
        cs = self.crop_size
        scale = cs / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = F.interpolate(
            image.unsqueeze(0), size=(new_h, new_w),
            mode="bilinear", antialias=True,
        )

        overflow_h = new_h - cs
        overflow_w = new_w - cs
        start_h = overflow_h // 2
        start_w = overflow_w // 2
        cropped = resized[:, :, start_h:start_h + cs, start_w:start_w + cs]
        return cropped.unsqueeze(0)  # (1, 1, 3, cs, cs)

    def _read_proprio(self) -> torch.Tensor:
        """Read 14-DOF joint positions from mjd.qpos.

        Gripper values mapped from MuJoCo finger joint space to agent [0, 1].

        Returns:
            (1, 14) tensor (unnormalized).
        """
        joints = np.array([
            self.mjd.qpos[self.mjm.jnt_qposadr[jid]]
            for jid in self.joint_ids
        ])
        joints[6] = np.clip((joints[6] - FINGER_JOINT_MIN) / FINGER_JOINT_RANGE, 0.0, 1.0)
        joints[13] = np.clip((joints[13] - FINGER_JOINT_MIN) / FINGER_JOINT_RANGE, 0.0, 1.0)
        return torch.from_numpy(joints).float().unsqueeze(0)  # (1, 14)

    def _normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        """Normalize proprio using config bounds.

        Matches robot_eye_demo_bc_yam.py._normalize_proprio().
        """
        from eye.sim.demo_data import DemonstrationData

        if self.config.normalization_type == "min_max":
            return DemonstrationData.normalize_joint_data_flexible(
                proprio,
                method="min_max",
                joint_min=self.config.proprio_min,
                joint_max=self.config.proprio_max,
                epsilon=self.config.normalization_epsilon,
            )
        else:
            return DemonstrationData.normalize_joint_data_flexible(
                proprio,
                method="mean_std",
                joint_mean=self.config.proprio_mean,
                joint_std=self.config.proprio_std,
            )


# ──────────────────────────────────────────────
# Viser live viewer
# ──────────────────────────────────────────────

class BCViserViewer:
    """Live browser viewer for BC eval debugging.

    Shows URDF robot in 3D scene with action chunk ghosts (red=first, green=last),
    camera images in GUI sidebar, and play/pause controls.
    """

    def __init__(self, port: int = 8080, urdf_path: str = "./urdf/yam_bimanual.urdf"):
        import viser
        from viser.extras import ViserUrdf

        self._server = viser.ViserServer(port=port)
        self._server.scene.enable_default_lights()

        # --- GUI controls ---
        self._paused = self._server.gui.add_checkbox("Pause", initial_value=False)
        self._step_btn = self._server.gui.add_button("Step")
        self._step_event = threading.Event()
        self._step_btn.on_click(lambda _: self._step_event.set())

        # Status text
        self._status = self._server.gui.add_text("Status", initial_value="", disabled=True)
        self._stage_text = self._server.gui.add_text("Stages", initial_value="", disabled=True)

        # Image handles — created lazily
        self._exo_handle = None
        self._wrist_left_handle = None
        self._wrist_right_handle = None

        # --- URDF scene: ground truth (white) + chunk ghosts (red/green) ---
        self._urdf_path = Path(urdf_path)
        self.gt_urdf = ViserUrdf(
            self._server,
            urdf_or_path=self._urdf_path,
            root_node_name="/gt",
        )
        self.chunk_start_urdf = ViserUrdf(
            self._server,
            urdf_or_path=self._urdf_path,
            root_node_name="/chunk_start",
            mesh_color_override=(1.0, 0.0, 0.0),  # red = chunk[0]
        )
        self.chunk_end_urdf = ViserUrdf(
            self._server,
            urdf_or_path=self._urdf_path,
            root_node_name="/chunk_end",
            mesh_color_override=(0.0, 0.8, 0.0),  # green = chunk[-1]
        )

        # Grid at robot base height
        z_min = self.gt_urdf._urdf.scene.bounds[0, 2]
        self._server.scene.add_grid("/grid", width=2.0, height=2.0, position=(0.3, 0, z_min))

        print(f"Viser server started at http://localhost:{port}")

    @staticmethod
    def _joints_to_urdf_cfg(joints: np.ndarray) -> np.ndarray:
        """Convert 14-DOF agent joints to URDF config (scale grippers)."""
        left_gripper = np.clip(joints[6], 0.3, 1.3) * 0.065
        right_gripper = np.clip(joints[13], 0.3, 1.3) * 0.065
        return np.concatenate([joints[:6], [left_gripper], joints[7:13], [right_gripper]])

    def update_robot(self, current_joints: np.ndarray):
        """Update ground truth URDF pose from current joint state."""
        self.gt_urdf.update_cfg(self._joints_to_urdf_cfg(current_joints))

    def update_chunk_ghosts(self, joint_chunk: np.ndarray):
        """Show red (chunk[0]) and green (chunk[-1]) ghost robots."""
        self.chunk_start_urdf.update_cfg(self._joints_to_urdf_cfg(joint_chunk[0]))
        self.chunk_end_urdf.update_cfg(self._joints_to_urdf_cfg(joint_chunk[-1]))

    def hide_chunk_ghosts(self):
        """Hide ghosts during chunk execution."""
        self.chunk_start_urdf.update_cfg(self._joints_to_urdf_cfg(np.zeros(14)))
        self.chunk_end_urdf.update_cfg(self._joints_to_urdf_cfg(np.zeros(14)))

    def update(
        self,
        exo_rgb: np.ndarray | None,
        wrist_left_rgb: np.ndarray | None,
        wrist_right_rgb: np.ndarray | None,
        status_str: str,
        stages_dict: dict[str, bool],
    ):
        if not self._server.get_clients():
            return
        # All images in GUI sidebar
        if exo_rgb is not None:
            if self._exo_handle is None:
                self._exo_handle = self._server.gui.add_image(
                    image=exo_rgb, label="Exo", format="jpeg",
                )
            else:
                self._exo_handle.image = exo_rgb
        if wrist_left_rgb is not None:
            if self._wrist_left_handle is None:
                self._wrist_left_handle = self._server.gui.add_image(
                    image=wrist_left_rgb, label="Wrist Left", format="jpeg",
                )
            else:
                self._wrist_left_handle.image = wrist_left_rgb
        if wrist_right_rgb is not None:
            if self._wrist_right_handle is None:
                self._wrist_right_handle = self._server.gui.add_image(
                    image=wrist_right_rgb, label="Wrist Right", format="jpeg",
                )
            else:
                self._wrist_right_handle.image = wrist_right_rgb
        # Status text
        self._status.value = status_str
        stage_parts = [f"{k}: {'Y' if v else 'N'}" for k, v in stages_dict.items()]
        self._stage_text.value = " | ".join(stage_parts)

    def is_paused(self) -> bool:
        return self._paused.value

    def consume_step(self) -> bool:
        """Returns True once per Step click, clears the latch."""
        if self._step_event.is_set():
            self._step_event.clear()
            return True
        return False

    def has_clients(self) -> bool:
        return len(self._server.get_clients()) > 0

    def set_paused(self, paused: bool):
        """Programmatically set the pause checkbox."""
        self._paused.value = paused

    def wait_for_unpause(self):
        """Block until user unchecks Pause."""
        while self.is_paused():
            time.sleep(0.05)


# ──────────────────────────────────────────────
# CLI eval script
# ──────────────────────────────────────────────

def _build_normalization_params(config) -> dict:
    """Build normalization_params dict from RobotAgentConfig for process_action_chunk."""
    return {
        "absolute_actions_min": config.absolute_actions_min,
        "absolute_actions_max": config.absolute_actions_max,
        "absolute_actions_mean": config.absolute_actions_mean,
        "absolute_actions_std": config.absolute_actions_std,
        "relative_actions_min": config.relative_actions_min,
        "relative_actions_max": config.relative_actions_max,
        "relative_actions_mean": config.relative_actions_mean,
        "relative_actions_std": config.relative_actions_std,
        "normalization_epsilon": config.normalization_epsilon,
    }


@dataclasses.dataclass
class RecedingHorizon:
    """Infer every H steps, execute H actions from chunk, no blending. H=chunk_size is open loop."""
    horizon: int
    """Number of steps to execute per chunk (1 to chunk_size). chunk_size = open loop."""


@dataclasses.dataclass
class TemporalEnsembling:
    """Infer every step, blend overlapping chunks with exp(-k*i) weighting (ACT/LeRobot convention)."""
    ensemble_k: float
    """w_i = exp(-k * i), i=0 oldest. k>0 = older weighted more (ACT default), k=0 = uniform, k<0 = newer weighted more."""


@dataclasses.dataclass
class Args:
    ckpt: str
    """path to BC agent checkpoint (or glob pattern for multi-checkpoint sweep, e.g. 'runs/my_run/checkpoint_*.pt')"""
    task: str
    """task name (tape_handover_random, tape_handover_grid, double_marker_in_cup)"""
    n_episodes: int
    """number of evaluation episodes"""
    mode: RecedingHorizon | TemporalEnsembling
    """Action execution mode."""
    max_steps: int = 600
    """max steps per episode (~30s at 30Hz)"""
    seed: int = 0
    """random seed for eval config generation"""
    eval_dir: str | None = None
    """override eval output directory"""
    rank: int = 0
    """worker rank for distributed eval"""
    world_size: int = 1
    """total number of workers"""
    no_video: bool = False
    """skip saving episode videos"""
    video_fps: int = 30
    """video playback FPS"""
    side_view: bool = False
    """add side camera panel to video"""
    top_view: bool = False
    """add top-down camera panel to video"""
    exo_width: int = 768
    """exo camera render width"""
    exo_height: int = 480
    """exo camera render height"""
    dump_obs: bool = False
    """save per-episode .npz with obs tensors and actions"""
    viser: bool = False
    """enable live Viser visualization in browser"""
    port: int = 8080
    """Viser server port"""
    action_type: str | None = None
    """Override action type for multi-head models (e.g. joint_rel, se3_world_rel). Default: first from config."""
    flow_steps: int | None = None
    """Override flow_num_inference_steps from checkpoint config (e.g. 3, 5, 10, 20)."""

    # --- Flow matching inference tuning ---
    num_steps: int = 0
    """Flow matching: number of ODE integration steps. 0 = use config default (typically 10). Try 20-50 for better quality."""


def main():
    import tyro
    import glob as glob_module
    args = tyro.cli(Args)

    # --- Resolve checkpoint glob for multi-checkpoint sweeps ---
    if '*' in args.ckpt or '?' in args.ckpt:
        ckpt_paths = sorted(glob_module.glob(args.ckpt))
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoints matched pattern: {args.ckpt}")
        print(f"Found {len(ckpt_paths)} checkpoints to sweep:")
        for p in ckpt_paths:
            print(f"  {p}")
    else:
        ckpt_paths = [args.ckpt]

    for ckpt_idx, ckpt_path in enumerate(ckpt_paths):
        if len(ckpt_paths) > 1:
            print(f"\n{'='*60}")
            print(f"Checkpoint {ckpt_idx + 1}/{len(ckpt_paths)}: {ckpt_path}")
            print(f"{'='*60}")
        _run_eval(args, ckpt_path)


def _run_eval(args: Args, ckpt_path: str):

    from eye.agent_configs import AgentConfigManager, RobotAgentConfig, FlowRobotAgentConfig
    from eye.agents.robot_agent import RobotAgent
    from eye.agents.flow_robot_agent import FlowRobotAgent

    # --- Output directory ---
    if args.eval_dir:
        eval_dir = Path(args.eval_dir)
        # For multi-checkpoint sweeps with explicit eval_dir, add checkpoint stem as subdirectory
        if '*' in args.ckpt or '?' in args.ckpt:
            eval_dir = eval_dir / Path(ckpt_path).stem
        eval_dir.mkdir(parents=True, exist_ok=True)
    else:
        eval_dir = make_eval_dir(ckpt_path, args.task, label="bc")
    print(f"Eval output -> {eval_dir}")

    # --- Load agent ---
    config = AgentConfigManager.load_config_from_checkpoint(args.ckpt)
    assert isinstance(config, (RobotAgentConfig, FlowRobotAgentConfig)), (
        f"Expected RobotAgentConfig or FlowRobotAgentConfig, got {type(config).__name__}. "
        f"This script is for BC-trained models only."
    )

    if args.flow_steps is not None and isinstance(config, FlowRobotAgentConfig):
        config.flow_num_inference_steps = args.flow_steps

    ckpt = torch.load(args.ckpt, map_location="cuda")
    is_flow = any("velocity_head" in k for k in ckpt["model_state_dict"].keys())
    if is_flow:
        agent = FlowRobotAgent(config=config)
    else:
        agent = RobotAgent(config=config)
    load_result = agent.load_state_dict(ckpt["model_state_dict"], strict=False)
    agent.eval().cuda()
    if load_result.missing_keys:
        print(f"  WARNING missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"  WARNING unexpected keys: {load_result.unexpected_keys}")

    # --- Action config (from checkpoint, or CLI override for multi-head) ---
    action_mode = args.action_type or config.action_types[0]
    agent_type = "FlowRobotAgent" if is_flow else "RobotAgent"
    print(f"Agent: {agent_type}, Action type: {action_mode} (available: {config.action_types})")
    # --- Inference mode ---
    is_receding = isinstance(args.mode, RecedingHorizon)
    if is_receding:
        assert 1 <= args.mode.horizon <= config.action_chunk_size, (
            f"horizon={args.mode.horizon} must be between 1 and chunk_size={config.action_chunk_size}"
        )
        H = args.mode.horizon
        ensemble_k = 0.01  # unused, but ActionBuffer needs something
        print(f"Inference: mode=receding_horizon, horizon={H} (chunk_size={config.action_chunk_size})")
    else:
        H = 1  # temporal ensembling queries every step
        ensemble_k = args.mode.ensemble_k
        print(f"Inference: mode=temporal_ensembling, ensemble_k={ensemble_k}")

    # --- Flow matching inference params ---
    flow_sample_kwargs = {}
    if is_flow:
        if args.num_steps > 0:
            flow_sample_kwargs["num_steps"] = args.num_steps
        effective_steps = args.num_steps if args.num_steps > 0 else config.flow_num_inference_steps
        print(f"Flow matching: num_steps={effective_steps}")

    normalization_params = _build_normalization_params(config)

    # IK solver for SE3 action types
    pk_robot, left_link_name, right_link_name = None, "", ""
    if action_mode.startswith("se3"):
        from eye.ik_utils import setup_bimanual_ik as _setup_ik_solver
        print("Setting up PyRoKi IK solver for SE3 actions...")
        pk_robot, left_link_name, right_link_name = _setup_ik_solver()

    # --- Create task + environment ---
    task = TASK_REGISTRY[args.task]()
    env = BCEvalEnv(
        xml_path=XML_PATH,
        task=task,
        agent_config=config,
        exo_width=args.exo_width,
        exo_height=args.exo_height,
        ensemble_k=ensemble_k,
    )
    if args.side_view:
        env.enable_side_view()
    if args.top_view:
        env.enable_top_view()

    # --- Warmup inference (triggers torch.compile + CUDA kernel caching) ---
    print("Warming up model...")
    warmup_configs = task.generate_eval_configs(1, base_seed=99999)
    warmup_obs = env.reset(config=warmup_configs[0])
    with torch.no_grad():
        if is_flow:
            agent.sample(warmup_obs, **flow_sample_kwargs)
        else:
            agent(warmup_obs, inference=True)
    print("Warmup done.")

    # --- Viser viewer (rank 0 only) ---
    viewer = BCViserViewer(port=args.port) if args.viser and args.rank == 0 else None

    # --- Generate eval configs ---
    all_configs = task.generate_eval_configs(args.n_episodes, args.seed)
    my_work = [(i, all_configs[i]) for i in range(args.rank, args.n_episodes, args.world_size)]

    successes = []
    episode_results = []
    save_video = not args.no_video

    # --- Progress file for eval_launcher compatibility ---
    progress_path = eval_dir / f"progress_rank{args.rank}.jsonl"

    rank_desc = f"rank {args.rank}" if args.world_size > 1 else "Episodes"
    _first_inference_logged = False
    ep_bar = tqdm(my_work, desc=rank_desc, unit="ep")
    for global_idx, ep_config in ep_bar:
        obs = env.reset(config=ep_config)
        frames: list[np.ndarray] = []
        stages_achieved = {s: False for s in task.stages}

        # Update viewer with reset state — wait for a client, then show everything
        if viewer is not None:
            print("Waiting for viser client to connect...")
            while not viewer.has_clients():
                time.sleep(0.1)
            viewer.update_robot(env._read_proprio()[0].numpy())
            viewer.update(
                env.last_exo_rgb,
                env.last_wrist_left_rgb,
                env.last_wrist_right_rgb,
                f"Ep {global_idx} | Running first inference...",
                stages_achieved,
            )

        # Per-step metric accumulators
        metric_joint_chunks = []
        metric_joint_positions = []
        stage_first_steps = {}

        # Debug log: every step's actual joint position + action buffer target
        debug_step_joints = []     # actual qpos each step
        debug_step_targets = []    # action buffer output each step
        debug_chunk_steps = []     # (step_idx, chunk) pairs for each inference call

        def _capture_frame():
            if not save_video:
                return
            primary_rgb = env.last_exo_rgb
            if primary_rgb is None:
                return
            panels = [primary_rgb.copy()]

            # Wrist camera panels (scaled to match primary height)
            from PIL import Image
            for wrist_rgb in (env.last_wrist_left_rgb, env.last_wrist_right_rgb):
                if wrist_rgb is not None:
                    scale = primary_rgb.shape[0] / wrist_rgb.shape[0]
                    wrist_img = Image.fromarray(wrist_rgb).resize(
                        (int(wrist_rgb.shape[1] * scale), primary_rgb.shape[0]),
                    )
                    panels.append(np.array(wrist_img))

            # Side / top views
            for render_fn in (env.render_side_view, env.render_top_view):
                ext_rgb = render_fn()
                if ext_rgb is not None:
                    scale = primary_rgb.shape[0] / ext_rgb.shape[0]
                    ext_img = Image.fromarray(ext_rgb).resize(
                        (int(ext_rgb.shape[1] * scale), primary_rgb.shape[0]),
                    )
                    panels.append(np.array(ext_img))

            combined = np.concatenate(panels, axis=1) if len(panels) > 1 else panels[0]
            frames.append(combined)

        # Capture initial frame
        _capture_frame()

        # Obs dump
        obs_log = {k: [] for k in [
            "proprio", "agent_raw", "joint_chunk",
        ]} if args.dump_obs else None

        success = False
        joint_chunk = None
        step_bar = tqdm(range(args.max_steps), desc=f"  Ep {global_idx}", unit="step", leave=False)
        for step_idx in step_bar:
            # --- Pause gate (blocks until unpaused or step clicked) ---
            if viewer is not None:
                while viewer.is_paused() and not viewer.consume_step():
                    time.sleep(0.05)

            # Record wall-clock AFTER pause exits (so pause time isn't counted)
            step_start = time.perf_counter()

            # --- Agent inference: every H steps (receding horizon) or every step (temporal ensembling) ---
            run_inference = (step_idx % H == 0) or joint_chunk is None
            if run_inference:
                torch.compiler.cudagraph_mark_step_begin()
                with torch.no_grad():
                    if is_flow:
                        sampled_dict = agent.sample(obs, **flow_sample_kwargs)  # Dict[str, (B, chunk, dim)]
                        # Wrap each type to (1, B, chunk, dim) for process_action_chunk
                        actions_dict = {k: v.unsqueeze(0) for k, v in sampled_dict.items()}
                    else:
                        actions_dict, _debug_info = agent(obs, inference=True)

                # --- Denormalize actions via canonical pipeline ---
                raw_proprio = env._read_proprio()  # (1, 14) unnormalized

                # Compute current EE pose for SE3 action types that need it
                if action_mode == "se3_ee_rel":
                    from eye.inference.fk_se3 import proprio_to_10vec
                    ee_to_world = proprio_to_10vec(
                        raw_proprio.cuda(), env._pk_robot,
                        env._left_ee_link_index, env._right_ee_link_index, "cuda",
                    )  # (1, 20)
                else:
                    ee_to_world = torch.zeros(1, 20, device="cuda")

                # --- Raw network output (before denormalization) ---
                _raw_actions = actions_dict[action_mode]

                joint_chunk = process_action_chunk(
                    action_mode=action_mode,
                    joint_actions_output=actions_dict,
                    cur_joints=raw_proprio.cuda(),
                    ee_to_world=ee_to_world,
                    eye_to_base_se3_tensor=torch.zeros(1, 7, device="cuda"),  # unused for BC eval (no eye)
                    normalization_params=normalization_params,
                    normalization_type=config.normalization_type,
                    pk_robot=pk_robot,
                    left_link_name=left_link_name,
                    right_link_name=right_link_name,
                )  # (chunk_size, 14) numpy

                # --- Viser: show chunk ghosts ---
                if viewer is not None:
                    viewer.update_chunk_ghosts(joint_chunk)

                # Receding horizon: clear old chunks before adding new one
                if is_receding:
                    _pre_reset_n = len(env.action_buffer.action_chunks)
                    env.action_buffer.reset()

                # --- Step environment with new chunk ---
                obs, success, stage_results = env.step(joint_chunk)

            else:
                # --- Step environment using existing buffer (no new chunk) ---
                obs, success, stage_results = env.step(None)


            for k, v in stage_results.items():
                stages_achieved[k] |= v

            # --- Debug: log actual joint position every step ---
            debug_step_joints.append(env._read_proprio()[0].numpy().copy())
            buf_action = env.action_buffer.get_action(env.sim_time)
            debug_step_targets.append(buf_action.copy() if buf_action is not None else np.zeros(14))
            if run_inference:
                debug_chunk_steps.append((step_idx, joint_chunk.copy()))

            # --- Per-step metrics ---
            if run_inference:
                metric_joint_chunks.append(joint_chunk.copy())
                metric_joint_positions.append(raw_proprio[0].numpy().copy())
            for k, v in stage_results.items():
                if v and k not in stage_first_steps:
                    stage_first_steps[k] = step_idx

            # --- Obs dump ---
            if obs_log is not None and run_inference:
                obs_log["proprio"].append(env._read_proprio()[0].numpy())
                obs_log["agent_raw"].append(
                    actions_dict[action_mode][-1, 0, 0].float().cpu().numpy()
                )
                obs_log["joint_chunk"].append(joint_chunk[0])

            _capture_frame()

            # --- Viser update + real-time throttle ---
            if viewer is not None:
                viewer.update_robot(env._read_proprio()[0].numpy())
                viewer.update(
                    env.last_exo_rgb,
                    env.last_wrist_left_rgb,
                    env.last_wrist_right_rgb,
                    f"Ep {global_idx} | Step {step_idx}/{args.max_steps} | t={env.sim_time:.2f}s",
                    stages_achieved,
                )
                # Throttle to real-time only when clients are watching
                if viewer.has_clients():
                    elapsed = time.perf_counter() - step_start
                    remaining = env.agent_dt - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

            if success:
                break

        step_bar.close()

        # --- Show episode result in viewer ---
        if viewer is not None:
            result_str = "SUCCESS" if success else f"FAIL ({step_idx + 1} steps)"
            viewer.update(
                env.last_exo_rgb,
                env.last_wrist_left_rgb,
                env.last_wrist_right_rgb,
                f"Ep {global_idx} done: {result_str}",
                stages_achieved,
            )

        # --- Episode-level metrics ---
        # Chunk overlap agreement
        chunk_diffs = []
        for i in range(1, len(metric_joint_chunks)):
            prev = metric_joint_chunks[i - 1][1:]
            curr = metric_joint_chunks[i][:-1]
            overlap_len = min(len(prev), len(curr))
            if overlap_len > 0:
                chunk_diffs.append(np.mean(np.linalg.norm(
                    prev[:overlap_len] - curr[:overlap_len], axis=-1)))
        chunk_overlap_mean = float(np.mean(chunk_diffs)) if chunk_diffs else 0.0

        # Joint acceleration (smoothness)
        if len(metric_joint_positions) >= 3:
            positions = np.array(metric_joint_positions)
            vel = np.diff(positions, axis=0)
            accel = np.diff(vel, axis=0)
            joint_accel_mean = float(np.mean(np.linalg.norm(accel, axis=-1)))
        else:
            joint_accel_mean = 0.0

        # Steps to each stage
        steps_to_stage = {}
        for s in task.stages:
            steps_to_stage[s] = stage_first_steps.get(s, step_idx + 1)

        # --- Save video ---
        if save_video and len(frames) > 0:
            video_path = eval_dir / f"episode_{global_idx:03d}.mp4"
            imageio.mimwrite(
                str(video_path), frames, fps=args.video_fps, codec="h264",
                output_params=["-crf", "23", "-pix_fmt", "yuv420p"],
            )

        # --- Save debug diagnostic npz (always, for first 3 episodes) ---
        if global_idx < 3:
            diag_path = eval_dir / f"episode_{global_idx:03d}_debug.npz"
            chunk_starts = np.array([s for s, _ in debug_chunk_steps])
            chunk_arrays = [c for _, c in debug_chunk_steps]
            # Pad chunks to same length for stacking
            max_chunk_len = max(c.shape[0] for c in chunk_arrays) if chunk_arrays else 0
            if chunk_arrays:
                padded = np.stack([
                    np.pad(c, ((0, max_chunk_len - c.shape[0]), (0, 0)), constant_values=np.nan)
                    for c in chunk_arrays
                ])
            else:
                padded = np.zeros((0, 0, 14))
            np.savez_compressed(str(diag_path),
                step_joints=np.array(debug_step_joints),      # (num_steps, 14)
                step_targets=np.array(debug_step_targets),     # (num_steps, 14)
                chunk_starts=chunk_starts,                      # (num_inferences,)
                chunks=padded,                                  # (num_inferences, chunk_size, 14)
            )
            print(f"  Debug diagnostic saved: {diag_path}")

        # --- Save obs dump ---
        if obs_log is not None:
            npz_path = eval_dir / f"episode_{global_idx:03d}_obs.npz"
            np.savez_compressed(str(npz_path), **{k: np.array(v) for k, v in obs_log.items()})

        # --- Record results ---
        ep_result = {
            "episode": global_idx,
            "success": bool(success),
            "stages_achieved": {k: bool(v) for k, v in stages_achieved.items()},
            "steps": step_idx + 1,
            "chunk_overlap_mean": chunk_overlap_mean,
            "joint_accel_mean": joint_accel_mean,
        }
        for s in task.stages:
            ep_result[f"steps_to_{s}"] = steps_to_stage[s]
        episode_results.append(ep_result)
        successes.append(success)

        ep_bar.set_postfix(success=f"{sum(successes)}/{len(successes)}")

        # Write progress (eval_launcher compatible)
        with open(progress_path, "a") as f:
            f.write(json.dumps(ep_result) + "\n")

    # --- Final summary ---
    success_rate = sum(successes) / len(successes) if successes else 0.0
    per_stage_rates = {}
    for s in task.stages:
        per_stage_rates[s] = sum(
            r["stages_achieved"].get(s, False) for r in episode_results
        ) / len(episode_results) if episode_results else 0.0

    summary = {
        "checkpoint": ckpt_path,
        "task": args.task,
        "action_type": action_mode,
        "n_episodes": len(successes),
        "success_rate": success_rate,
        "stage_rates": per_stage_rates,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "exo_resolution": f"{args.exo_width}x{args.exo_height}",
        "mode": "receding_horizon" if is_receding else "temporal_ensembling",
        "horizon": H,
        "ensemble_k": ensemble_k,
        "num_steps": args.num_steps if args.num_steps > 0 else (config.flow_num_inference_steps if is_flow else 0),
        "episodes": episode_results,
    }

    results_path = eval_dir / f"results_rank{args.rank}.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"BC Eval Results: {ckpt_path}")
    print(f"  Task: {args.task}, Action: {action_mode}")
    if is_flow:
        print(f"  Flow: steps={summary['num_steps']}")
    print(f"  Success rate: {success_rate:.1%} ({sum(successes)}/{len(successes)})")
    for s, rate in per_stage_rates.items():
        print(f"  {s}: {rate:.1%}")
    print(f"  Results: {results_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
