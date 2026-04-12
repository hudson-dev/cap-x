"""MuJoCo evaluation environment for EyeRobotAgent policies.

CPU MuJoCo physics (with noslip_iterations for grasping) + built-in EGL
renderer. Perspective camera with foveated multicrop observations matching the
training format.

Usage:
    python -m eye.mujoco.eval --ckpt runs/my_run/checkpoint_final.pt --n-episodes 10
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from eye.foveal_encoders import create_foveated_batch, crop_sizes_from_levels
from eye.mujoco.action_buffer import ActionBuffer
from eye.mujoco.mjwarp_utils import MujocoRenderer, Equirect360Renderer
from eye.sim.spherical_video import _get_local_cam_rays, _multicrop_rays, _sample_from_equiv
from eye.stereo import (
    DEPTH_MIN, DEPTH_MAX, gaze_dir_to_so3,
    eye_state_to_so3, compute_fixation_point, fixation_to_eye_so3s,
)
from eye.table_frame import (
    compute_table_frame_se3, ee_to_table_frame,
    ee_to_gaze_frame, compute_gaze_frame_se3, frame_relative_to_world,
)
from eye.mujoco.tasks import (
    Task, TapeHandoverTask, DoubleMarkerInCupTask, PickUpTigerTask,
    TASK_REGISTRY,
)
from eye.transforms import SE3, SO3



# ─── DEBUG: Gripper debug overlay for video frames ───────────────────────────
# Set to True to overlay gripper telemetry on eval videos.
# Safe to set back to False or delete this entire block when done debugging.
DEBUG_GRIPPER_OVERLAY = True


def _debug_gripper_overlay(
    frame: np.ndarray,
    step_idx: int,
    stages_achieved: dict[str, bool] | None = None,
    **_kwargs,
) -> np.ndarray:
    """Draw stage progress overlay onto the top-left corner of a video frame."""
    if not stages_achieved:
        return frame
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18)
    except (OSError, IOError):
        font = ImageFont.load_default()

    markers = " > ".join(
        (f"[{s}]" if stages_achieved[s] else s) for s in stages_achieved
    )
    line = f"t={step_idx:>3d}  {markers}"

    x0, y0 = 10, 10
    line_h = 22
    w = draw.textlength(line, font=font) + 10
    draw.rectangle([x0, y0, x0 + w, y0 + line_h + 10], fill=(0, 0, 0, 180))
    draw.text((x0 + 5, y0 + 5), line, fill=(0, 255, 0), font=font)

    return np.array(img)
# ─── END DEBUG ───────────────────────────────────────────────────────────────

URDF_PATH = Path(__file__).parents[2] / "urdf" / "yam_bimanual.urdf"

XML_PATH = Path(__file__).parent / "robot_xmls" / "eyeball_bimanual_scene.xml"

# MuJoCo finger joint range (both left_left_finger and right_left_finger)
# Agent gripper values are in [0, 1] (0=closed, 1=open) but MuJoCo finger
# joints are in meters. This linear mapping converts between the two.
FINGER_JOINT_MIN = -0.00205
FINGER_JOINT_MAX = 0.04517
FINGER_JOINT_RANGE = FINGER_JOINT_MAX - FINGER_JOINT_MIN  # ~0.0396 m


def _gripper_agent_to_mujoco(gripper_val: float | np.ndarray) -> float | np.ndarray:
    """Map agent gripper command [0, 1] → MuJoCo finger joint position."""
    return np.clip(gripper_val, 0.0, 1.0) * FINGER_JOINT_RANGE + FINGER_JOINT_MIN

# Joint names for 14-DOF proprioception (6 arm joints + 1 finger per arm)
JOINT_NAMES = [
    "left_joint1", "left_joint2", "left_joint3",
    "left_joint4", "left_joint5", "left_joint6",
    "left_left_finger",
    "right_joint1", "right_joint2", "right_joint3",
    "right_joint4", "right_joint5", "right_joint6",
    "right_left_finger",
]


class EvalEnv:
    """Single-world MuJoCo eval environment with CPU physics + EGL rendering.

    The MjSpec pipeline:
      1. Load base XML (robot + table + eye camera only)
      2. Task adds objects via configure_scene(spec)
      3. Compile to get full MjModel with task objects
      4. Task caches IDs via setup(mjm, mjd)

    Args:
        xml_path: Path to base scene XML (no task objects).
        task: Task instance defining the manipulation task.
        agent_config: Agent config (for multicrop sizes, action chunk size, etc.).
        render_width: GPU render width in pixels.
        render_height: GPU render height in pixels.
        agent_fps: Agent inference frequency in Hz.
        eye_ema_alpha: EMA smoothing factor for eye velocity (1.0 = no smoothing).
        eye_vel_scale: Scaling factor from raw agent radians to deg/s.
    """

    def __init__(
        self,
        xml_path: str | Path,
        task: Task,
        agent_config,
        render_width: int = 1920,
        render_height: int = 1200,
        agent_fps: float = 30.0,
        eye_ema_alpha: float = 0.8,
        eye_vel_scale: float = 15.0,
        normalization_params: dict | None = None,
        fisheye: bool = True,
        vignette_alpha: float | None = None,
        video_mode: str = "center",
    ):
        self.vignette_alpha = vignette_alpha
        # --- MjSpec pipeline: base XML → task adds objects → compile ---
        spec = mujoco.MjSpec.from_file(str(xml_path))
        task.configure_scene(spec)
        # Task objects add freejoints that invalidate the base XML keyframe qpos sizes.
        # Delete all keyframes so compile() doesn't throw (MuJoCo <3.5 enforces this).
        for k in list(spec.keys):
            if hasattr(k, 'delete'):
                k.delete()
        self.mjm = spec.compile()
        self.mjd = mujoco.MjData(self.mjm)

        self.physics_dt = self.mjm.opt.timestep
        self.agent_dt = 1.0 / agent_fps
        self.substeps = int(round(self.agent_dt / self.physics_dt))

        # --- Renderer (after compile, model now includes task objects) ---
        # When fisheye=True, renders a wider pinhole internally and applies
        # the equidistant fisheye distortion remap to match the real camera.
        self.renderer = MujocoRenderer(
            self.mjm, "eye_camera", render_width, render_height, fisheye=fisheye,
        )

        # --- Task: cache IDs post-compilation ---
        self.task = task
        task.setup(self.mjm, self.mjd)

        # Oracle CLIP toggle is handled via OracleTargetSelector loaded from checkpoint.
        # The eval loop swaps target_clip_vec the same way as the learned selector.
        # No special setup needed here.

        # --- Agent config (for multicrop) ---
        self.stereo = getattr(agent_config, 'stereo', False)
        self.video_mode = video_mode
        self.elp_mono = video_mode in ("left", "right")
        self.crop_sizes = crop_sizes_from_levels(
            agent_config.n_levels, agent_config.fovea_size, agent_config.sphere_size
        )
        self.window_size = agent_config.window_size

        # --- ELP cameras (stereo or mono left/right) ---
        if self.stereo:
            self._init_stereo()
        elif self.elp_mono:
            self._init_elp_mono(video_mode)

        # --- Cache joint/actuator/site IDs ---
        self.joint_ids = [
            mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]

        self.eye_pan_act_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, "eye_pan")
        self.eye_tilt_act_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, "eye_tilt")

        self.left_grasp_site_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_site")
        self.right_grasp_site_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_site")

        # Eye camera ID for reading the actual camera frame from MuJoCo
        self.eye_cam_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_CAMERA, "eye_camera")

        # --- PyRoKi FK (matches training data's EE computation exactly) ---
        # MuJoCo grasp_sites are NOT at the same position/orientation as the
        # URDF's tool0 frames that training uses. We use PyRoKi FK instead.
        import yourdfpy
        from pyroki import Robot as PKRobot
        urdf = yourdfpy.URDF.load(URDF_PATH, load_meshes=False)
        self.pk_robot = PKRobot.from_urdf(urdf)
        self.left_ee_link_idx = self.pk_robot.links.names.index("left_tool0")
        self.right_ee_link_idx = self.pk_robot.links.names.index("right_tool0")

        # Actuator indices: eye=[0,1], left_arm=[2..7], left_grip=[8], right_arm=[9..14], right_grip=[15]
        self.arm_act_start = 2  # first arm actuator
        self.arm_act_end = 16   # exclusive end (16 total, first 2 are eye)

        # --- Eye state ---
        self.eye_pan = 0.0
        self.eye_tilt = 0.0
        self.prev_eye_vel = np.zeros(2)
        self.eye_ema_alpha = eye_ema_alpha
        self.eye_vel_scale = eye_vel_scale

        # --- Action buffer ---
        self.action_buffer = ActionBuffer(
            chunk_size=agent_config.action_chunk_size,
            action_dim=14,
            agent_fps=agent_fps,
        )

        # --- Timing ---
        self.sim_time = 0.0
        self.agent_step_count = 0
        self.agent_fps = agent_fps
        self.agent_dt = 1.0 / agent_fps

        # --- Last rendered frame (cached for video saving) ---
        self.last_rgb: np.ndarray | None = None
        self.last_multicrop: np.ndarray | None = None  # (n_levels, H, W, 3) uint8

        # --- Optional external-view renderers (CPU, EGL) ---
        self.side_view_renderer: mujoco.Renderer | None = None
        self.side_view_cam: mujoco.MjvCamera | None = None
        self.side_view_scene_option: mujoco.MjvOption | None = None
        self.last_side_view_rgb: np.ndarray | None = None

        self.top_view_renderer: mujoco.Renderer | None = None
        self.top_view_cam: mujoco.MjvCamera | None = None
        self.top_view_scene_option: mujoco.MjvOption | None = None
        self.last_top_view_rgb: np.ndarray | None = None

        # --- Demo home position (mean_start_position from checkpoint) ---
        # This is the average starting joint config across all demonstrations,
        # used by robot_eye_demo_yam.py as the reset target.
        self.home_joints: np.ndarray | None = None
        if normalization_params is not None and "mean_start_position" in normalization_params:
            home = normalization_params["mean_start_position"]
            if isinstance(home, torch.Tensor):
                self.home_joints = home.float().cpu().numpy().flatten()
            else:
                self.home_joints = np.array(home, dtype=np.float32).flatten()
            print(f"Demo home position ({len(self.home_joints)} DOF): {self.home_joints}")

    def enable_side_view(self, width: int = 640, height: int = 480):
        """Enable a side-view CPU renderer for visualizing table contact."""
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
        """Render from the side camera. Returns (H, W, 3) uint8 or None."""
        if self.side_view_renderer is None:
            return None
        self.side_view_renderer.update_scene(
            self.mjd, self.side_view_cam, scene_option=self.side_view_scene_option,
        )
        self.side_view_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        rgb = self.side_view_renderer.render()
        self.last_side_view_rgb = rgb
        return rgb

    def enable_top_view(self, width: int = 640, height: int = 480):
        """Enable a top-down camera renderer for bird's-eye visualization."""
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
        """Render from the top-down camera. Returns (H, W, 3) uint8 or None."""
        if self.top_view_renderer is None:
            return None
        self.top_view_renderer.update_scene(
            self.mjd, self.top_view_cam, scene_option=self.top_view_scene_option,
        )
        self.top_view_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        rgb = self.top_view_renderer.render()
        self.last_top_view_rgb = rgb
        return rgb

    def reset(self, config: dict) -> dict[str, torch.Tensor]:
        """Reset environment for a new episode.

        Args:
            config: Task config dict from task.generate_eval_configs().

        Returns:
            Observation dict with shapes (T=1, B=1, ...).
        """
        # 1. Reset to home keyframe (sets eye + arm joints to XML defaults)
        mujoco.mj_resetDataKeyframe(self.mjm, self.mjd, 0)

        # 2. Override arm joints with demo home position (mean_start_position)
        if self.home_joints is not None:
            for i, jid in enumerate(self.joint_ids):
                val = self.home_joints[i]
                # Map gripper values from agent [0,1] to MuJoCo finger joint range
                if i == 6 or i == 13:  # gripper indices
                    val = _gripper_agent_to_mujoco(val)
                self.mjd.qpos[self.mjm.jnt_qposadr[jid]] = val
            # Also set ctrl so PD controllers target the home position
            # Left arm joints (actuators 2-7) + left gripper (8)
            self.mjd.ctrl[2:8] = self.home_joints[:6]
            self.mjd.ctrl[8] = _gripper_agent_to_mujoco(self.home_joints[6])
            # Right arm joints (actuators 9-14) + right gripper (15)
            self.mjd.ctrl[9:15] = self.home_joints[7:13]
            self.mjd.ctrl[15] = _gripper_agent_to_mujoco(self.home_joints[13])

        # 3. Task config application (object poses)
        self.task.apply_eval_config(self.mjm, self.mjd, config)

        # 4. Forward kinematics to populate all poses
        mujoco.mj_forward(self.mjm, self.mjd)

        # 5. Reset eye state, action buffer, timing
        if self.stereo:
            # Stereo: virtual eye state (az, el) — no MuJoCo joints involved.
            # [-π/2, -π/2] = forward horizontal (matches training init range center)
            self._eye_state[:] = torch.tensor(
                [[-np.pi / 2, -np.pi / 2]], device="cuda")
            self._fixation_depth[:] = 0.5
            self._prev_smoothed_eye_action.zero_()
        elif self.elp_mono:
            # ELP mono: same virtual eye state as stereo, no fixation depth
            self._eye_state[:] = torch.tensor(
                [[-np.pi / 2, -np.pi / 2]], device="cuda")
            self._prev_smoothed_eye_action.zero_()
        else:
            # Mono: physically move eye_camera via MuJoCo pan/tilt joints
            self.eye_pan = 0.0
            self.eye_tilt = np.deg2rad(30.0)  # positive = down
            self.prev_eye_vel = np.zeros(2)
            eye_pan_jid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_JOINT, "eye_pan")
            eye_tilt_jid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_JOINT, "eye_tilt")
            self.mjd.qpos[self.mjm.jnt_qposadr[eye_pan_jid]] = self.eye_pan
            self.mjd.qpos[self.mjm.jnt_qposadr[eye_tilt_jid]] = self.eye_tilt
            self.mjd.ctrl[self.eye_pan_act_id] = self.eye_pan
            self.mjd.ctrl[self.eye_tilt_act_id] = self.eye_tilt
        mujoco.mj_forward(self.mjm, self.mjd)  # update camera pose / body xpos

        # 6. Warmup physics steps so PD controllers settle (especially grippers).
        # Without this, the first few get_obs() frames see transient gripper
        # proprio values (~0.85) that don't match the real robot's stable start
        # (~0.96), potentially poisoning the KV cache for the whole episode.
        for _ in range(500):
            mujoco.mj_step(self.mjm, self.mjd)

        # 7. Post-warmup hook: let tasks re-place objects that shifted during warmup.
        self.task.post_warmup(self.mjm, self.mjd, config)
        mujoco.mj_forward(self.mjm, self.mjd)

        self.action_buffer.reset()
        self.sim_time = 0.0
        self.agent_step_count = 0

        return self.get_obs()

    def get_obs(self) -> dict[str, torch.Tensor]:
        """Build observation dict matching agent input format.

        All shapes are (T=1, B=1, ...) for single-step single-batch inference.
        """
        device = "cuda"

        if self.stereo:
            return self._get_obs_stereo(device)
        if self.elp_mono:
            return self._get_obs_elp_mono(device)
        return self._get_obs_mono(device)

    def _get_obs_mono(self, device: str) -> dict[str, torch.Tensor]:
        """Mono observation path (existing eye_camera pipeline)."""
        # 1. Render eye camera → multicrop on GPU (fisheye remap via grid_sample)
        img_gpu = self.renderer.render_torch(self.mjm, self.mjd, device=device)
        multicrop = self._render_multicrop_gpu(img_gpu)
        # Cache CPU fisheye frame for video saving (uses precomputed remap tables)
        self.last_rgb = self.renderer.last_cpu_rgb
        # Cache multicrop as uint8 numpy for visualization: (n_levels, H, W, 3)
        self.last_multicrop = (multicrop.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()

        # 2. Eye direction
        eye_direction = self._compute_eye_direction().to(device)

        # 3. Eye-to-base SE3
        eye_to_base_se3 = self._compute_eye_to_base_se3().to(device)

        # 4. Proprio (14 DOF)
        proprio = self._read_proprio().to(device)

        # 5. EE-to-world (20D bimanual)
        ee_to_world = self._compute_ee_to_world(proprio).to(device)

        # 6. EE-to-eye (20D bimanual)
        ee_to_eye = self._compute_ee_to_eye(ee_to_world, eye_to_base_se3).to(device)

        # 7. CLIP target embedding (static — oracle/selector swap happens in eval loop)
        target_clip_vec = self.task.get_clip_embedding(device).unsqueeze(0)  # (1, 512)

        # 8. Table frame SE3 and EE-to-table (for se3_table_rel action type)
        # Note: helpers already return (1, D) tensors (batch dim = 1)
        table_frame_se3 = compute_table_frame_se3(
            eye_to_base_se3,   # (1, 7)
            eye_direction,     # (1, 3)
        )  # (1, 7)
        ee_to_table = ee_to_table_frame(
            ee_to_world,       # (1, 20)
            table_frame_se3,   # (1, 7)
        )  # (1, 20)

        # Package all as (T=1, B=1, ...)
        obs = {
            "multicrop": multicrop.unsqueeze(0).unsqueeze(0),        # (1, 1, n_levels, 3, ws, ws)
            "eye_direction": eye_direction.unsqueeze(0),              # (1, 1, 3)
            "target_clip_vec": target_clip_vec.unsqueeze(0),          # (1, 1, 512)
            "proprio": proprio.unsqueeze(0),                          # (1, 1, 14)
            "ee_to_world": ee_to_world.unsqueeze(0),                  # (1, 1, 20)
            "ee_to_eye": ee_to_eye.unsqueeze(0),                      # (1, 1, 20)
            "eye_to_base_se3": eye_to_base_se3.unsqueeze(0),          # (1, 1, 7)
            "table_frame_se3": table_frame_se3.unsqueeze(0),          # (1, 1, 7)
            "ee_to_table": ee_to_table.unsqueeze(0),                  # (1, 1, 20)
        }
        if self.vignette_alpha is not None:
            obs["vignette_alpha"] = self.vignette_alpha
        return obs

    def _get_obs_stereo(self, device: str) -> dict[str, torch.Tensor]:
        """Stereo observation path (ELP cameras via equirect reprojection)."""
        # 1. Stereo geometry from fixation point
        fixation = compute_fixation_point(
            self._eye_state, self._fixation_depth)           # (1, 3)
        so3_left, so3_right = fixation_to_eye_so3s(fixation)  # per-eye SO3

        # 2. Render L/R multicrop via equirect sampling
        multicrop_left = self._render_one_eye(
            self._equirect_left, so3_left)                     # (n_levels, 3, ws, ws)
        multicrop_right = self._render_one_eye(
            self._equirect_right, so3_right)

        # Cache for video: L/R multicrop stacked as two columns
        mc_left_vis = (multicrop_left.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        mc_right_vis = (multicrop_right.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        self.last_multicrop = np.concatenate([mc_left_vis, mc_right_vis], axis=2)  # (n_levels, ws, ws*2, 3)
        # Main video frame: two rows (top=left eye levels, bottom=right eye levels)
        left_row = np.concatenate(list(mc_left_vis), axis=1)   # (ws, n_levels*ws, 3)
        right_row = np.concatenate(list(mc_right_vis), axis=1)  # (ws, n_levels*ws, 3)
        self.last_rgb = np.concatenate([left_row, right_row], axis=0)  # (2*ws, n_levels*ws, 3)

        # 3. Cyclopean eye direction + fixation depth (4D)
        cyclopean_dir = eye_state_to_so3(self._eye_state).as_matrix()[:, :, 2]  # (1, 3)
        eye_direction = torch.cat([cyclopean_dir, self._fixation_depth.unsqueeze(-1)], dim=-1).to(device)  # (1, 4)

        # 4. Cyclopean eye-to-base SE3
        eye_to_base_se3 = self._compute_stereo_eye_to_base_se3()  # (1, 7) on cuda

        # 5. Proprio (14 DOF)
        proprio = self._read_proprio().to(device)

        # 6. EE-to-world (20D bimanual)
        ee_to_world = self._compute_ee_to_world(proprio).to(device)

        # 7. EE-to-eye (20D bimanual)
        ee_to_eye = self._compute_ee_to_eye(ee_to_world, eye_to_base_se3).to(device)

        # 8. CLIP target embedding
        target_clip_vec = self.task.get_clip_embedding(device).unsqueeze(0)  # (1, 512)

        # 9. Table frame (uses cyclopean direction, same as training)
        table_frame_se3 = compute_table_frame_se3(
            eye_to_base_se3, cyclopean_dir)
        ee_to_table = ee_to_table_frame(ee_to_world, table_frame_se3)

        # 10. Gaze (fixation) frame — stereo only
        gaze_frame_se3 = compute_gaze_frame_se3(
            eye_to_base_se3, self._fixation_depth)
        ee_to_fixation = ee_to_gaze_frame(ee_to_world, gaze_frame_se3)

        # Package all as (T=1, B=1, ...)
        obs = {
            "multicrop_left": multicrop_left.unsqueeze(0).unsqueeze(0),   # (1, 1, n_levels, 3, ws, ws)
            "multicrop_right": multicrop_right.unsqueeze(0).unsqueeze(0),
            "eye_direction": eye_direction.unsqueeze(0),                   # (1, 1, 4)
            "target_clip_vec": target_clip_vec.unsqueeze(0),               # (1, 1, 512)
            "proprio": proprio.unsqueeze(0),                               # (1, 1, 14)
            "ee_to_world": ee_to_world.unsqueeze(0),                       # (1, 1, 20)
            "ee_to_eye": ee_to_eye.unsqueeze(0),                           # (1, 1, 20)
            "eye_to_base_se3": eye_to_base_se3.unsqueeze(0),               # (1, 1, 7)
            "table_frame_se3": table_frame_se3.unsqueeze(0),               # (1, 1, 7)
            "ee_to_table": ee_to_table.unsqueeze(0),                       # (1, 1, 20)
            "gaze_frame_se3": gaze_frame_se3.unsqueeze(0),                 # (1, 1, 7)
            "ee_to_fixation": ee_to_fixation.unsqueeze(0),                 # (1, 1, 20)
        }
        if self.vignette_alpha is not None:
            obs["vignette_alpha"] = self.vignette_alpha
        return obs

    def _get_obs_elp_mono(self, device: str) -> dict[str, torch.Tensor]:
        """ELP mono observation path (left/right camera via equirect reprojection).

        Uses the same mono obs format as center mode, but renders through a
        single ELP camera's equirect projection instead of the Insta360 fisheye.
        Eye frame calculations use the cyclopean virtual eye state, matching
        robot_gym's behavior for left/right video modes.
        """
        # 1. Cyclopean SO3 from virtual eye state
        eye_so3 = eye_state_to_so3(self._eye_state)

        # 2. Render multicrop via equirect sampling (reuses stereo's _render_one_eye)
        multicrop = self._render_one_eye(self._equirect_mono, eye_so3)

        # Cache for video: levels side by side
        mc_vis = (multicrop.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        self.last_multicrop = mc_vis
        self.last_rgb = np.concatenate(list(mc_vis), axis=1)

        # 3. Eye direction (cyclopean)
        eye_direction = eye_so3.as_matrix()[:, :, 2].to(device)  # (1, 3)

        # 4. Eye-to-base SE3 (elp_stereo body + cyclopean rotation)
        eye_to_base_se3 = self._compute_stereo_eye_to_base_se3()  # (1, 7) on cuda

        # 5-8. Same as center mono
        proprio = self._read_proprio().to(device)
        ee_to_world = self._compute_ee_to_world(proprio).to(device)
        ee_to_eye = self._compute_ee_to_eye(ee_to_world, eye_to_base_se3).to(device)
        target_clip_vec = self.task.get_clip_embedding(device).unsqueeze(0)
        table_frame_se3 = compute_table_frame_se3(eye_to_base_se3, eye_direction)
        ee_to_table = ee_to_table_frame(ee_to_world, table_frame_se3)

        obs = {
            "multicrop": multicrop.unsqueeze(0).unsqueeze(0),
            "eye_direction": eye_direction.unsqueeze(0),
            "target_clip_vec": target_clip_vec.unsqueeze(0),
            "proprio": proprio.unsqueeze(0),
            "ee_to_world": ee_to_world.unsqueeze(0),
            "ee_to_eye": ee_to_eye.unsqueeze(0),
            "eye_to_base_se3": eye_to_base_se3.unsqueeze(0),
            "table_frame_se3": table_frame_se3.unsqueeze(0),
            "ee_to_table": ee_to_table.unsqueeze(0),
        }
        if self.vignette_alpha is not None:
            obs["vignette_alpha"] = self.vignette_alpha
        return obs

    def step(
        self, eye_action: np.ndarray, joint_chunk: np.ndarray
    ) -> tuple[dict[str, torch.Tensor], bool, dict[str, bool]]:
        """Apply actions and step physics.

        Args:
            eye_action: (2,) for mono or (3,) for stereo — raw eye deltas from agent.
            joint_chunk: (chunk_size, 14) joint position targets from agent.

        Returns:
            Tuple of (observation dict, success bool, stage results dict).
        """
        # 1. Eye update (stereo/elp_mono: virtual state; center mono: MuJoCo joints)
        if self.stereo:
            action_t = torch.from_numpy(eye_action).float().cuda().unsqueeze(0)  # (1, 3)
            self._prev_smoothed_eye_action = (
                self.eye_ema_alpha * action_t
                + (1 - self.eye_ema_alpha) * self._prev_smoothed_eye_action
            )
            self._eye_state = self._eye_state + self._prev_smoothed_eye_action[:, :2]
            self._fixation_depth = (
                self._fixation_depth + self._prev_smoothed_eye_action[:, 2]
            ).clamp(DEPTH_MIN, DEPTH_MAX)
        elif self.elp_mono:
            action_t = torch.from_numpy(eye_action).float().cuda().unsqueeze(0)  # (1, 2)
            self._prev_smoothed_eye_action = (
                self.eye_ema_alpha * action_t
                + (1 - self.eye_ema_alpha) * self._prev_smoothed_eye_action
            )
            self._eye_state = self._eye_state + self._prev_smoothed_eye_action
        else:
            raw_vel = np.rad2deg(eye_action) * self.eye_vel_scale
            raw_vel[..., 0] *= -1
            raw_vel[..., 1] *= -1
            smoothed_vel = self.eye_ema_alpha * raw_vel + (1 - self.eye_ema_alpha) * self.prev_eye_vel
            self.prev_eye_vel = smoothed_vel

            self.eye_pan += np.deg2rad(smoothed_vel[0]) * self.agent_dt
            self.eye_tilt += np.deg2rad(smoothed_vel[1]) * self.agent_dt
            self.eye_pan = float(np.clip(self.eye_pan, -np.pi, np.pi))
            self.eye_tilt = float(np.clip(self.eye_tilt, -np.pi / 2, np.pi / 2))

            self.mjd.ctrl[self.eye_pan_act_id] = self.eye_pan
            self.mjd.ctrl[self.eye_tilt_act_id] = self.eye_tilt

        # 2. Add hand action chunk to buffer
        self.action_buffer.add_chunk(joint_chunk, self.sim_time)

        # 3. Step physics (multiple substeps per agent step)
        for _ in range(self.substeps):
            joint_targets = self.action_buffer.get_action(self.sim_time)
            if joint_targets is not None:
                # Left arm: actuators 2-7 (joints), 8 (gripper)
                self.mjd.ctrl[2:8] = joint_targets[:6]
                self.mjd.ctrl[8] = _gripper_agent_to_mujoco(joint_targets[6])
                # Right arm: actuators 9-14 (joints), 15 (gripper)
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
    # Stereo ELP camera setup
    # ──────────────────────────────────────────────

    def _init_stereo(self):
        """Initialize stereo ELP camera rendering via equirectangular reprojection.

        Creates per-eye Equirect360Renderer instances (single-camera mode with
        ELP distortion), precomputes multicrop rays from training intrinsics,
        and initializes stereo eye state (azimuth, elevation, fixation depth).
        """
        from eye.camera import get_default_video_config, get_stereo_video_config

        # ELP calibrated intrinsics (per-eye K + distortion)
        K_left, dist_left, K_right, dist_right, _, _, _ = get_stereo_video_config()

        # Equirect output: 2:1 spherical projection. Resolution controls
        # sampling quality — only the ELP's FOV will have content (rest is black).
        eq_w, eq_h = 2562, 1282

        # Per-eye equirect renderers (single-camera mode with ELP distortion).
        # Internally renders a wider pinhole, then remaps to equirect with the
        # ELP distortion model baked in — matching what the real camera sees.
        self._equirect_left = Equirect360Renderer(
            self.mjm, self.mjd,
            out_width=eq_w, out_height=eq_h,
            cam_name="elp_left",
            K=K_left, dist=dist_left, is_fisheye=False,
        )
        self._equirect_right = Equirect360Renderer(
            self.mjm, self.mjd,
            out_width=eq_w, out_height=eq_h,
            cam_name="elp_right",
            K=K_right, dist=dist_right, is_fisheye=False,
        )

        # Precompute multicrop rays on GPU using TRAINING camera intrinsics.
        # This preserves the same FOV structure / distortion pattern the agent
        # was trained on (Insta360 fisheye model at 1920×1200).
        train_w, train_h = 1920, 1200
        K, dist_coeffs, is_fisheye = get_default_video_config(train_w, train_h)
        K_t = torch.tensor(K).float().cuda()
        dist_t = dist_coeffs.float().cuda()
        local_rays = _get_local_cam_rays(K_t, dist_t, train_h, train_w, is_fisheye)
        self._stereo_multicrop_rays = _multicrop_rays(
            local_rays, self.crop_sizes, self.window_size, train_h, train_w,
        )  # (n_levels * ws * ws, 3) on cuda

        # elp_stereo body ID (for eye_to_base_se3 position)
        self._elp_body_id = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, "elp_stereo")

        # Stereo eye state (on GPU to match training gym convention)
        self._eye_state = torch.tensor(
            [[-np.pi / 2, -np.pi / 2]], dtype=torch.float32, device="cuda")
        self._fixation_depth = torch.tensor(
            [0.5], dtype=torch.float32, device="cuda")
        self._prev_smoothed_eye_action = torch.zeros(
            1, 3, dtype=torch.float32, device="cuda")

        print(f"Stereo ELP cameras initialized (equirect {eq_w}×{eq_h})")

    def _init_elp_mono(self, side: str):
        """Initialize single ELP camera for left/right mono mode.

        Same equirect reprojection pipeline as stereo, but with only one camera
        and no fixation depth. The model is mono (stereo=False) — it just sees
        through an ELP camera instead of the Insta360 fisheye.
        """
        from eye.camera import get_default_video_config, get_stereo_video_config

        K_left, dist_left, K_right, dist_right, _, _, _ = get_stereo_video_config()
        eq_w, eq_h = 2562, 1282

        if side == "left":
            self._equirect_mono = Equirect360Renderer(
                self.mjm, self.mjd, out_width=eq_w, out_height=eq_h,
                cam_name="elp_left", K=K_left, dist=dist_left, is_fisheye=False,
            )
        else:
            self._equirect_mono = Equirect360Renderer(
                self.mjm, self.mjd, out_width=eq_w, out_height=eq_h,
                cam_name="elp_right", K=K_right, dist=dist_right, is_fisheye=False,
            )

        # Multicrop rays (same as stereo — uses training camera intrinsics)
        train_w, train_h = 1920, 1200
        K, dist_coeffs, is_fisheye = get_default_video_config(train_w, train_h)
        K_t = torch.tensor(K).float().cuda()
        dist_t = dist_coeffs.float().cuda()
        local_rays = _get_local_cam_rays(K_t, dist_t, train_h, train_w, is_fisheye)
        self._stereo_multicrop_rays = _multicrop_rays(
            local_rays, self.crop_sizes, self.window_size, train_h, train_w,
        )

        self._elp_body_id = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, "elp_stereo")

        # Virtual eye state (same as stereo, but no fixation depth)
        self._eye_state = torch.tensor(
            [[-np.pi / 2, -np.pi / 2]], dtype=torch.float32, device="cuda")
        self._prev_smoothed_eye_action = torch.zeros(
            1, 2, dtype=torch.float32, device="cuda")

        print(f"ELP mono ({side}) camera initialized (equirect {eq_w}×{eq_h})")

    def _render_one_eye(
        self,
        equirect_renderer: Equirect360Renderer,
        so3_eye: SO3,
    ) -> torch.Tensor:
        """Render multicrop for one eye via equirect sampling.

        Matches the training pipeline exactly:
        1. Equirect360Renderer renders the ELP camera to equirect (with distortion)
        2. _sample_from_equiv samples the equirect with per-eye SO3 + multicrop rays

        Args:
            equirect_renderer: Per-eye Equirect360Renderer (single-camera mode).
            so3_eye: SO3 per-eye rotation from fixation_to_eye_so3s.

        Returns:
            (n_levels, 3, ws, ws) float32 tensor on CUDA.
        """
        # 1. Render equirect on GPU (includes ELP distortion in the remap)
        equirect_gpu = equirect_renderer.render_torch(
            self.mjm, self.mjd, device="cuda")  # (3, eq_H, eq_W)

        # 2. Sample multicrop from equirect using per-eye SO3
        R_eye = so3_eye.as_matrix().squeeze(0)  # (3, 3)
        raw = _sample_from_equiv(
            equirect_gpu, R_eye, self._stereo_multicrop_rays,
        )  # (3, N) where N = n_levels * ws * ws

        # 3. Reshape to (n_levels, 3, ws, ws)
        n_levels = len(self.crop_sizes)
        ws = self.window_size
        return torch.nan_to_num(
            raw.reshape(3, n_levels, ws, ws).permute(1, 0, 2, 3), nan=0.0)

    def _compute_stereo_eye_to_base_se3(self) -> torch.Tensor:
        """Compute cyclopean eye-to-base SE3 for stereo mode.

        Uses the elp_stereo body position (fixed) + cyclopean SO3 from eye_state.
        Matches training: neutral_eye_to_base_se3 @ SE3.from_rotation(eye_so3).

        Returns:
            (1, 7) wxyz_xyz tensor on CUDA.
        """
        eye_so3 = eye_state_to_so3(self._eye_state)  # SO3 on cuda
        body_pos = torch.from_numpy(
            self.mjd.xpos[self._elp_body_id].copy()
        ).float().unsqueeze(0).cuda()  # (1, 3)
        se3 = SE3.from_rotation_and_translation(eye_so3, body_pos)
        return se3.wxyz_xyz.float()  # (1, 7)

    # ──────────────────────────────────────────────
    # Private observation helpers
    # ──────────────────────────────────────────────

    def _render_multicrop(self, rgb: np.ndarray) -> torch.Tensor:
        """Convert raw camera render to multicrop foveated pyramid.

        Center-crops at each crop_size, resizes to window_size.

        Args:
            rgb: (H, W, 3) uint8 from renderer.

        Returns:
            (n_levels, 3, window_size, window_size) float32 in [0, 1].
        """
        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        img = img.unsqueeze(0)  # (1, 3, H, W)
        multicrop = create_foveated_batch(img, self.crop_sizes, self.window_size)
        # create_foveated_batch concatenates along batch dim: (n_levels, 3, ws, ws)
        return multicrop

    def _render_multicrop_gpu(self, img: torch.Tensor) -> torch.Tensor:
        """Convert a GPU tensor to multicrop foveated pyramid (stays on GPU).

        Args:
            img: (3, H, W) float32 in [0, 1] on GPU (from render_torch).

        Returns:
            (n_levels, 3, window_size, window_size) float32 on GPU.
        """
        multicrop = create_foveated_batch(img.unsqueeze(0), self.crop_sizes, self.window_size)
        return multicrop

    def _compute_eye_direction(self) -> torch.Tensor:
        """Compute eye gaze direction from the actual MuJoCo camera frame.

        MuJoCo cameras look along -Z. In the OpenCV/training convention, the
        gaze direction is +Z. So gaze_world = -cam_xmat[:, 2].

        Returns:
            (1, 3) unit vector.
        """
        cam_xmat = self.mjd.cam_xmat[self.eye_cam_id].reshape(3, 3)
        gaze_dir = -cam_xmat[:, 2]  # MuJoCo -Z = forward
        return torch.from_numpy(gaze_dir.copy()).float().unsqueeze(0)  # (1, 3)

    def _compute_eye_to_base_se3(self) -> torch.Tensor:
        """Compute eye-to-base transform from the actual MuJoCo camera frame.

        Reads cam_xpos/cam_xmat for the eye_camera, converts from MuJoCo
        camera convention (Y-up, -Z forward) to OpenCV (Y-down, +Z forward):
            R_opencv = R_mujoco @ diag(1, -1, -1)

        Returns:
            (1, 7) wxyz_xyz tensor.
        """
        cam_xpos = self.mjd.cam_xpos[self.eye_cam_id].copy()  # (3,)
        cam_xmat = self.mjd.cam_xmat[self.eye_cam_id].reshape(3, 3).copy()  # (3, 3)

        # MuJoCo → OpenCV: flip Y (up→down) and Z (backward→forward)
        R_opencv = cam_xmat @ np.diag([1.0, -1.0, -1.0])

        so3 = SO3.from_matrix(torch.from_numpy(R_opencv).float().unsqueeze(0))
        pos = torch.from_numpy(cam_xpos).float().unsqueeze(0)
        se3 = SE3.from_rotation_and_translation(so3, pos)
        return se3.wxyz_xyz.float()  # (1, 7)

    def _read_proprio(self) -> torch.Tensor:
        """Read 14-DOF joint positions from mjd.qpos.

        Gripper values are mapped from MuJoCo finger joint space back to
        agent [0, 1] space so the agent sees the same representation it
        was trained on.

        Returns:
            (1, 14) tensor of joint positions.
        """
        joints = np.array([
            self.mjd.qpos[self.mjm.jnt_qposadr[jid]]
            for jid in self.joint_ids
        ])
        # Map gripper from MuJoCo meters → agent [0, 1]
        joints[6] = np.clip((joints[6] - FINGER_JOINT_MIN) / FINGER_JOINT_RANGE, 0.0, 1.0)
        joints[13] = np.clip((joints[13] - FINGER_JOINT_MIN) / FINGER_JOINT_RANGE, 0.0, 1.0)
        return torch.from_numpy(joints).float().unsqueeze(0)  # (1, 14)

    def _compute_ee_to_world(self, proprio: torch.Tensor) -> torch.Tensor:
        """Get EE poses via PyRoKi FK on tool0 frames (matches training exactly).

        MuJoCo grasp_sites are at a different position/orientation than the
        URDF tool0 frames used during training (~7cm offset, wrong rotation).
        Using PyRoKi FK ensures the ee_to_world observations match training.

        Args:
            proprio: (1, 14) joint positions (gripper already in agent [0,1] space).

        Returns:
            (1, 20) tensor: [left_10vec, right_10vec].
        """
        import jax.numpy as jnp
        import numpy as onp
        from eye.sim.demo_data import compute_bimanual_fk_batch

        joints_np = proprio[0].cpu().numpy()  # (14,)
        joints_jnp = jnp.array(onp.array(joints_np[np.newaxis, :]))  # (1, 14)

        left_wxyz_xyz, right_wxyz_xyz = compute_bimanual_fk_batch(
            joints_jnp, self.pk_robot, self.left_ee_link_idx, self.right_ee_link_idx
        )
        left_wxyz_xyz = onp.array(left_wxyz_xyz[0])  # (7,)
        right_wxyz_xyz = onp.array(right_wxyz_xyz[0])  # (7,)

        def _wxyz_xyz_to_10vec(wxyz_xyz: np.ndarray, gripper_val: float) -> np.ndarray:
            quat_wxyz = torch.tensor(wxyz_xyz[:4], dtype=torch.float32).unsqueeze(0)
            pos = wxyz_xyz[4:]
            rotmat = SO3(quat_wxyz).as_matrix()[0].numpy()  # (3, 3)
            col1 = rotmat[:, 0]
            col2 = rotmat[:, 1]
            return np.concatenate([pos, col1, col2, [gripper_val]])

        left_gripper = proprio[0, 6].item()
        right_gripper = proprio[0, 13].item()

        left_10vec = _wxyz_xyz_to_10vec(left_wxyz_xyz, left_gripper)
        right_10vec = _wxyz_xyz_to_10vec(right_wxyz_xyz, right_gripper)

        ee_20vec = np.concatenate([left_10vec, right_10vec])
        return torch.from_numpy(ee_20vec).float().unsqueeze(0)  # (1, 20)

    def _compute_ee_to_eye(
        self, ee_to_world: torch.Tensor, eye_to_base_se3: torch.Tensor
    ) -> torch.Tensor:
        """Transform ee_to_world to eye frame.

        ee_to_eye = eye_to_base^-1 @ ee_to_world

        Mirrors robot_gym._transform_single_ee_to_eye_frame exactly.

        Args:
            ee_to_world: (1, 20) bimanual 10-vec pairs.
            eye_to_base_se3: (1, 7) wxyz_xyz.

        Returns:
            (1, 20) tensor in eye frame.
        """
        left_eye = self._transform_single_ee_to_eye_frame(
            ee_to_world[..., :10].float(), eye_to_base_se3.float()
        )
        right_eye = self._transform_single_ee_to_eye_frame(
            ee_to_world[..., 10:].float(), eye_to_base_se3.float()
        )
        return torch.cat([left_eye, right_eye], dim=-1).float()

    @staticmethod
    def _transform_single_ee_to_eye_frame(
        ee_10vec: torch.Tensor, eye_to_base_se3: torch.Tensor
    ) -> torch.Tensor:
        """Transform a single EE 10-vector from world to eye frame.

        Ported from robot_gym._transform_single_ee_to_eye_frame.

        Args:
            ee_10vec: (..., 10) [pos(3), col1(3), col2(3), gripper(1)].
            eye_to_base_se3: (..., 7) wxyz_xyz.

        Returns:
            (..., 10) in eye frame.
        """
        ee_position = ee_10vec[..., :3]
        ee_col1 = ee_10vec[..., 3:6]
        ee_col2 = ee_10vec[..., 6:9]
        ee_col3 = torch.cross(ee_col1, ee_col2, dim=-1)
        ee_rotation_matrix = torch.stack([ee_col1, ee_col2, ee_col3], dim=-1)
        ee_se3 = SE3.from_rotation_and_translation(SO3.from_matrix(ee_rotation_matrix), ee_position)

        ee_to_eye_se3 = SE3(eye_to_base_se3).inverse() @ ee_se3
        ee_to_eye_matrix = ee_to_eye_se3.as_matrix()

        return torch.cat([
            ee_to_eye_matrix[..., :3, 3],  # position
            ee_to_eye_matrix[..., :3, 0],  # rotation col 1
            ee_to_eye_matrix[..., :3, 1],  # rotation col 2
            ee_10vec[..., 9:10],           # gripper (unchanged)
        ], dim=-1)


# ──────────────────────────────────────────────
# SE3 helpers and IK solver — canonical implementations in eye.ik_utils
# ──────────────────────────────────────────────
from eye.ik_utils import (
    setup_bimanual_ik as _setup_ik_solver,
    solve_bimanual_ik_batch as _solve_bimanual_ik_batch,
    solve_ik_for_se3_chunk as _solve_ik_for_se3_chunk,
    se3_10vec_to_se3 as _10vec_to_se3,
    se3_to_10vec as _se3_to_10vec,
    process_action_chunk,
)


# ──────────────────────────────────────────────
# CLI eval script
# ──────────────────────────────────────────────

def make_eval_dir(ckpt_path: str, task_name: str, label: str | None = None) -> Path:
    """Create a unique eval output directory under evals/.

    Structure: evals/<ckpt_stem>_<task>[_<label>]_<YYYYMMDD_HHMMSS>/
    """
    ckpt_stem = Path(ckpt_path).stem  # e.g. "checkpoint_final"
    # Include parent dir name for context (e.g. "my_run")
    run_name = Path(ckpt_path).parent.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [run_name, ckpt_stem, task_name]
    if label:
        parts.append(label)
    parts.append(timestamp)
    eval_dir = Path("evals") / "_".join(parts)
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


def main():
    parser = argparse.ArgumentParser(description="Evaluate EyeRobotAgent in MuJoCo simulation")
    parser.add_argument("--ckpt", required=True, help="Path to agent checkpoint")
    parser.add_argument("--task", default="tape_handover_grid", choices=list(TASK_REGISTRY.keys()))
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=350, help="Max steps per episode (~10s at 30Hz)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-dir", type=str, default=None, help="Override eval output directory")
    parser.add_argument("--rank", type=int, default=0, help="Worker rank for distributed eval")
    parser.add_argument("--world-size", type=int, default=1, help="Total number of workers")
    parser.add_argument("--no-video", action="store_true", help="Skip saving episode videos")
    parser.add_argument("--video-fps", type=int, default=30, help="Video playback FPS")
    parser.add_argument("--freeze-eye", action="store_true", help="Zero out eye velocity (debug: test arm-only behavior)")
    parser.add_argument("--action-type", type=str, default=None,
                        help="Override action type (e.g. se3_eye_rel, joint_abs). Default: first from config.")
    parser.add_argument("--settle-time", type=int, default=0,
                        help="Number of eye-only steps before arm starts moving. The eye policy runs "
                             "and saccades, but arm holds initial position. Matches training time_pause.")
    # Oracle CLIP toggle is auto-detected from checkpoint (OracleTargetSelector).
    parser.add_argument("--use-ema-hand", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply EMA hand weights for eval (default: True). Use --no-use-ema-hand for raw (fast) hand weights.")
    parser.add_argument("--eye-ema-alpha", type=float, default=0.8,
                        help="EMA smoothing for eye actions (1.0=no smoothing, 0.5=heavy, 0.0=frozen)")
    parser.add_argument("--vignette-alpha", type=float, default=None,
                        help="Override vignette alpha for hand tokens (0.0=tunnel vision, 1.0=full FOV, None=disabled).")
    parser.add_argument("--no-fisheye", action="store_true",
                        help="Disable fisheye distortion (render raw pinhole instead)")
    parser.add_argument("--side-view", action="store_true",
                        help="Save side-by-side video with a side camera view")
    parser.add_argument("--top-view", action="store_true",
                        help="Save side-by-side video with a top-down camera view")
    parser.add_argument("--multicrop-view", action="store_true",
                        help="Show foveal multicrop pyramid as an extra panel in the video")
    parser.add_argument("--dump-obs", action="store_true",
                        help="Save per-episode .npz with obs tensors and actions for debugging")
    parser.add_argument("--real-video", type=str, default=None,
                        help="Path to a 360° demo video. Swaps MuJoCo multicrop with real "
                             "video crops starting at step 120 (handover phase).")
    parser.add_argument("--real-video-frame", type=int, default=None,
                        help="Frame index in the real video to hold on (default: middle)")
    parser.add_argument("--real-video-levels", type=str, default=None,
                        help="Comma-separated crop level indices to swap with real video "
                             "(0=fovea, ..., N-1=periphery). Default: swap all levels. "
                             "Example: --real-video-levels 0  (fovea only)")
    parser.add_argument("--chunk-publish-size", type=int, default=None,
                        help="Number of timesteps to publish from each predicted action chunk. "
                             "Slices the first N steps before IK, saving compute. "
                             "Default: None (use full action_chunk_size from checkpoint).")
    args = parser.parse_args()

    from eye.agent_configs import AgentConfigManager
    from eye.agents.eye_robot_agent import EyeRobotAgent

    # Create output directory
    if args.eval_dir:
        eval_dir = Path(args.eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
    else:
        eval_dir = make_eval_dir(args.ckpt, args.task)
    print(f"Eval output → {eval_dir}")

    # Load agent
    config = AgentConfigManager.load_config_from_checkpoint(args.ckpt)
    ckpt = torch.load(args.ckpt, map_location="cuda")
    agent = EyeRobotAgent(config=config, normalization_params=ckpt.get("normalization_params"))
    agent.load_state_dict(ckpt["model_state_dict"], strict=True)
    # Prefer EMA hand weights for eval (slow model = deployed model)
    ema_state = ckpt.get("hand_ema_state")
    if ema_state is not None and args.use_ema_hand:
        ema_params = ema_state["ema_params"]
        n_loaded = 0
        for name, p in agent.named_parameters():
            if name in ema_params:
                p.data.copy_(ema_params[name])
                n_loaded += 1
        print(f"Loaded EMA hand weights ({n_loaded} params, decay={ema_state['decay']})")
    agent.eval().cuda()

    # Auto-detect target selector from checkpoint (learned or oracle)
    selector = None
    target_clip_embeddings = None
    ts_data = ckpt.get("target_selector")
    if ts_data is not None:
        from eye.agents.target_selector import (
            TargetSelector, TargetSelectorConfig,
            OracleTargetSelector, OracleTargetSelectorConfig,
        )
        ts_config_dict = ts_data["config"]
        if ts_config_dict.get("is_oracle", False):
            selector = OracleTargetSelector(config=OracleTargetSelectorConfig.from_dict(ts_config_dict))
            print(f"Oracle target selector loaded: {list(selector.config.prompts)}")
        else:
            ts_cfg = TargetSelectorConfig.from_dict(ts_config_dict)
            selector = TargetSelector(config=ts_cfg)
            selector.load_state_dict(ts_data["state_dict"])
            print(f"Target selector loaded: {list(ts_cfg.prompts)}")
        selector.eval().cuda()
        target_clip_embeddings = ts_data["clip_embeddings"].cuda()

    # Determine video mode from checkpoint (center for old checkpoints)
    video_mode = ckpt.get("video_mode", "center")
    print(f"Video mode: {video_mode}")

    # Determine action type
    normalization_params = ckpt.get("normalization_params")
    action_mode = args.action_type or config.action_types[0]
    chunk_publish_size = args.chunk_publish_size  # None = use full chunk
    print(f"Action type: {action_mode} (available: {config.action_types})")
    if chunk_publish_size is not None:
        print(f"Chunk publish size: {chunk_publish_size} (model predicts {config.action_chunk_size})")

    # Set up IK solver if using SE3 action types
    pk_robot, left_link_name, right_link_name = None, "", ""
    if action_mode.startswith("se3"):
        print("Setting up PyRoKi IK solver for SE3 actions...")
        pk_robot, left_link_name, right_link_name = _setup_ik_solver()

    # Create task
    task = TASK_REGISTRY[args.task]()

    # Create environment
    env = EvalEnv(
        xml_path=XML_PATH,
        task=task,
        agent_config=config,
        normalization_params=ckpt.get("normalization_params"),
        fisheye=not args.no_fisheye,
        vignette_alpha=args.vignette_alpha,
        eye_ema_alpha=args.eye_ema_alpha,
        video_mode=video_mode,
    )

    if args.side_view:
        env.enable_side_view()
    if args.top_view:
        env.enable_top_view()

    # Optional real-video overlay for multicrop (swapped in at step 120)
    real_video = None
    if args.real_video:
        from eye.camera import get_default_video_config
        from eye.sim.spherical_video import SphericalVideo
        vid_path = Path(args.real_video)
        # Use the fisheye OUTPUT dimensions (out_height/out_width = 1200×1920),
        # NOT renderer.height/width which are the internal pinhole render dimensions
        # (scaled by pin_scale=1.5 → 1800×2880). SphericalVideo must match the
        # fisheye output so its ray grid covers the same FOV as the MuJoCo crops.
        rv_H = getattr(env.renderer, "out_height", env.renderer.height)
        rv_W = getattr(env.renderer, "out_width", env.renderer.width)
        K, dist_coeffs, is_fisheye = get_default_video_config(rv_W, rv_H)
        real_video = SphericalVideo(
            vid_path, K, dist_coeffs, rv_H, rv_W, is_fisheye, "cuda",
            env.crop_sizes, window_size=env.window_size, decoder_device="cuda",
        )
        real_video.reset(random_time=False)
        # Seek to requested frame
        target_frame = args.real_video_frame if args.real_video_frame is not None else real_video.total_frames // 2
        for _ in range(target_frame):
            real_video.advance()
        levels_str = "all" if args.real_video_levels is None else args.real_video_levels
        print(f"Real video loaded: {vid_path.name}, frame {target_frame}/{real_video.total_frames}, swap levels={levels_str}")

    # Parse which levels to swap (None = all)
    _real_levels: list[int] | None = None
    if args.real_video_levels is not None:
        _real_levels = [int(x) for x in args.real_video_levels.split(",")]

    def _swap_multicrop_real(obs: dict, env_obj) -> dict:
        """Replace MuJoCo multicrop with real video multicrop at current eye pose.

        If _real_levels is set, only those level indices are replaced; the rest
        keep their MuJoCo crops. Level 0 is the fovea (highest resolution),
        level N-1 is the most peripheral.
        """
        if real_video is None:
            return obs
        # Convert MuJoCo eye_pan/eye_tilt → SphericalVideo SO3
        azimuth = -torch.pi / 2 - env_obj.eye_pan
        elevation = -torch.pi / 2 - env_obj.eye_tilt
        rot = SO3.from_z_radians(torch.tensor(-azimuth - torch.pi, device="cuda")) @ \
              SO3.from_x_radians(torch.tensor(elevation, device="cuda"))
        real_multicrop = real_video.render_multicrop(rot)  # (n_levels, 3, ws, ws)

        if _real_levels is None:
            # Swap all levels
            obs["multicrop"] = real_multicrop.unsqueeze(0).unsqueeze(0)
            env_obj.last_multicrop = (real_multicrop.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        else:
            # Selective swap: only replace the requested levels
            merged = obs["multicrop"].clone()  # (1, 1, n_levels, 3, ws, ws)
            for lvl in _real_levels:
                merged[0, 0, lvl] = real_multicrop[lvl]
            obs["multicrop"] = merged
            # Visualization shows the merged result
            vis = merged[0, 0]  # (n_levels, 3, ws, ws)
            env_obj.last_multicrop = (vis.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        return obs

    # Generate all configs upfront, then slice for this rank
    all_configs = task.generate_eval_configs(args.n_episodes, args.seed)
    my_work = [(i, all_configs[i]) for i in range(args.rank, args.n_episodes, args.world_size)]

    successes = []
    episode_results = []
    save_video = not args.no_video

    # ─── DEBUG: per-step gripper state for overlay ───────────────────────────
    _dbg_gripper = {}  # populated each step when DEBUG_GRIPPER_OVERLAY is True
    # ─── END DEBUG ───────────────────────────────────────────────────────────

    rank_desc = f"rank {args.rank}" if args.world_size > 1 else "Episodes"
    ep_bar = tqdm(my_work, desc=rank_desc, unit="ep")
    for global_idx, ep_config in ep_bar:
        obs = env.reset(config=ep_config)
        kv_cache = None
        frame_offset = 0
        frames: list[np.ndarray] = []
        stages_achieved = {s: False for s in task.stages}

        # ─── Metric accumulators ──────────────────────────────────────────
        metric_values = []          # value estimate per step
        metric_eye_vels = []        # (2,) eye velocity commands
        metric_joint_chunks = []    # (chunk_size, 14) per step
        metric_ee_positions = []    # (6,) left_xyz + right_xyz per step
        metric_joint_positions = [] # (14,) joint positions per step
        stage_first_steps = {}      # {stage_name: first step_idx achieved}
        metric_gaze_angular_dists = []  # list of {target_name: angular_dist} per step
        metric_gaze_directions = []     # list of (3,) gaze direction per step
        # ──────────────────────────────────────────────────────────────────

        def _capture_frame():
            if not save_video or env.last_rgb is None:
                return
            from PIL import Image
            eye_rgb = env.last_rgb.copy()
            panels = [eye_rgb]
            for render_fn in (env.render_side_view, env.render_top_view):
                ext_rgb = render_fn()
                if ext_rgb is not None:
                    ext_img = Image.fromarray(ext_rgb)
                    scale = eye_rgb.shape[0] / ext_rgb.shape[0]
                    ext_img = ext_img.resize((int(ext_rgb.shape[1] * scale), eye_rgb.shape[0]))
                    panels.append(np.array(ext_img))
            # Multicrop visualization: stack foveal levels vertically, scale to match eye height
            if args.multicrop_view and env.last_multicrop is not None:
                crops = env.last_multicrop  # (n_levels, ws, ws, 3)
                crop_col = np.concatenate(list(crops), axis=0)  # (n_levels*ws, ws, 3)
                crop_img = Image.fromarray(crop_col)
                scale = eye_rgb.shape[0] / crop_col.shape[0]
                crop_img = crop_img.resize(
                    (int(crop_col.shape[1] * scale), eye_rgb.shape[0]),
                    Image.NEAREST,
                )
                panels.append(np.array(crop_img))
            combined = np.concatenate(panels, axis=1) if len(panels) > 1 else eye_rgb
            # ─── DEBUG: apply gripper overlay if available ────────────────
            if DEBUG_GRIPPER_OVERLAY and _dbg_gripper:
                combined = _debug_gripper_overlay(combined, **_dbg_gripper)
            # ─── END DEBUG ────────────────────────────────────────────────
            frames.append(combined)

        # Capture initial frame
        if DEBUG_GRIPPER_OVERLAY:
            _dbg_gripper = dict(step_idx=-1, stages_achieved=dict(stages_achieved))
        _capture_frame()

        # Obs dump: collect per-step tensors for debugging
        obs_log = {k: [] for k in [
            "proprio", "ee_to_world", "ee_to_eye", "eye_to_base_se3",
            "eye_direction", "agent_raw", "joint_chunk",
        ]} if args.dump_obs else None

        # ─── Settle time: eye-only steps before arm starts ─────────────
        # During settle steps, the eye policy runs (saccades to target) but
        # the arm holds its initial position. Matches training's time_pause.
        if args.settle_time > 0:
            init_joints = obs["proprio"][-1, 0].float().cpu().numpy()  # (14,)
            hold_chunk = np.tile(init_joints, (config.action_chunk_size, 1))  # (chunk_size, 14)
            for _settle_step in range(args.settle_time):
                with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    if selector is not None:
                        target_idx = selector(obs, deterministic=True)[0]
                        obs["target_clip_vec"] = target_clip_embeddings[target_idx]
                    result = agent.get_action_and_value(
                        obs, deterministic=True, kv_cache=kv_cache, frame_offset=frame_offset
                    )
                    eye_action = result[0]
                    kv_cache = result[-1]
                frame_offset += 1
                eye_vel = eye_action[-1, 0].float().cpu().numpy()
                if args.freeze_eye:
                    eye_vel = np.zeros_like(eye_vel)
                obs, _, _ = env.step(eye_vel, hold_chunk)
                _capture_frame()
        # ──────────────────────────────────────────────────────────────────

        success = False
        step_bar = tqdm(range(args.max_steps), desc=f"  Ep {global_idx}", unit="step", leave=False)
        for step_idx in step_bar:
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Target selector: swap CLIP embedding before agent forward
                if selector is not None:
                    target_idx = selector(obs, deterministic=True)[0]
                    obs["target_clip_vec"] = target_clip_embeddings[target_idx]

                result = agent.get_action_and_value(
                    obs, kv_cache=kv_cache, frame_offset=frame_offset
                )
                # Unpack: (action, log_prob, entropy, value, value_logits, joint_actions, kv_cache)
                eye_action = result[0]
                joint_actions = result[5]
                kv_cache = result[-1]

            frame_offset += 1


            # Extract eye action: (2,) for mono, (3,) for stereo
            eye_vel = eye_action[-1, 0].float().cpu().numpy()
            if args.freeze_eye:
                eye_vel = np.zeros_like(eye_vel)

            # Process hand action through the full action pipeline
            # (denormalization, frame transforms, IK for SE3 types)
            cur_joints = obs["proprio"][-1]  # (1, 14) from current obs
            ee_to_world_flat = obs["ee_to_world"][-1]  # (1, 20)
            eye_to_base_flat = obs["eye_to_base_se3"][-1]  # (1, 7)
            table_frame_flat = obs["table_frame_se3"][-1]  # (1, 7)

            # Slice action chunks before IK to save compute
            publish_actions = joint_actions
            if chunk_publish_size is not None:
                publish_actions = {
                    k: v[:, :, :chunk_publish_size] for k, v in joint_actions.items()
                }

            # Fixation frame for se3_fixation_rel (stereo only)
            fixation_frame_flat = (
                obs["gaze_frame_se3"][-1] if "gaze_frame_se3" in obs else None)

            joint_chunk = process_action_chunk(
                action_mode=action_mode,
                joint_actions_output=publish_actions,
                cur_joints=cur_joints,
                ee_to_world=ee_to_world_flat,
                eye_to_base_se3_tensor=eye_to_base_flat,
                normalization_params=normalization_params,
                normalization_type=config.normalization_type,
                pk_robot=pk_robot,
                left_link_name=left_link_name,
                right_link_name=right_link_name,
                table_frame_se3=table_frame_flat,
                fixation_frame_se3=fixation_frame_flat,
            )  # (chunk_publish_size or chunk_size, 14) numpy

            obs, success, stage_results = env.step(eye_vel, joint_chunk)
            for k, v in stage_results.items():
                stages_achieved[k] |= v

            # ─── Per-step metric accumulation ─────────────────────────────
            # Value estimate (result[3] is value)
            metric_values.append(result[3][-1, 0].float().cpu().item())
            # Eye velocity commands
            metric_eye_vels.append(eye_vel.copy())
            # Action chunks (for overlap comparison)
            metric_joint_chunks.append(joint_chunk.copy())
            # EE positions: first 3 of each 10-vec arm (xyz only)
            ee_flat = obs["ee_to_world"][-1, 0].float().cpu().numpy()  # (20,)
            metric_ee_positions.append(np.concatenate([ee_flat[:3], ee_flat[10:13]]))
            # Joint positions (from proprio)
            metric_joint_positions.append(obs["proprio"][-1, 0].float().cpu().numpy())
            # Stage timing: record first step each stage is achieved
            for k, v in stage_results.items():
                if v and k not in stage_first_steps:
                    stage_first_steps[k] = step_idx

            # Eye gaze metrics (use cyclopean direction for angular distance)
            gaze_targets = task.get_gaze_targets(env.mjm, env.mjd)
            if gaze_targets:
                if env.stereo:
                    eye_dir = eye_state_to_so3(env._eye_state).as_matrix()[0, :, 2].cpu().numpy()
                else:
                    eye_dir = obs["eye_direction"][-1, 0].float().cpu().numpy()  # (3,)
                eye_pos = obs["eye_to_base_se3"][-1, 0, 4:7].float().cpu().numpy()  # (3,)
                metric_gaze_directions.append(eye_dir.copy())
                step_dists = {}
                for name, obj_pos in gaze_targets.items():
                    to_obj = obj_pos - eye_pos
                    to_obj_unit = to_obj / (np.linalg.norm(to_obj) + 1e-8)
                    angle = np.arccos(np.clip(np.dot(eye_dir, to_obj_unit), -1, 1))
                    step_dists[name] = float(angle)
                metric_gaze_angular_dists.append(step_dists)
            # ──────────────────────────────────────────────────────────────

            # Swap multicrop to real video after step 120 (handover phase)
            if real_video is not None and step_idx >= 120:
                obs = _swap_multicrop_real(obs, env)

            # ─── DEBUG: populate gripper debug state after step ───────────
            if DEBUG_GRIPPER_OVERLAY:
                _dbg_gripper = dict(step_idx=step_idx, stages_achieved=dict(stages_achieved))
            # ─── END DEBUG ────────────────────────────────────────────────

            # Obs dump: record per-step data
            if obs_log is not None:
                obs_log["proprio"].append(obs["proprio"][-1, 0].float().cpu().numpy())
                obs_log["ee_to_world"].append(obs["ee_to_world"][-1, 0].float().cpu().numpy())
                obs_log["ee_to_eye"].append(obs["ee_to_eye"][-1, 0].float().cpu().numpy())
                obs_log["eye_to_base_se3"].append(obs["eye_to_base_se3"][-1, 0].float().cpu().numpy())
                obs_log["eye_direction"].append(obs["eye_direction"][-1, 0].float().cpu().numpy())
                raw_actions = joint_actions[action_mode][-1, 0, 0].float().cpu().numpy()
                obs_log["agent_raw"].append(raw_actions)
                obs_log["joint_chunk"].append(joint_chunk[0])  # first step of chunk

            # Capture frame after step
            _capture_frame()

            if success:
                break

        step_bar.close()

        # ─── Compute episode-level metrics from accumulators ──────────────
        # 1. Chunk overlap agreement
        chunk_diffs = []
        for i in range(1, len(metric_joint_chunks)):
            prev = metric_joint_chunks[i - 1][1:]      # steps [1, chunk_size)
            curr = metric_joint_chunks[i][:-1]          # steps [0, chunk_size-1)
            overlap_len = min(len(prev), len(curr))
            if overlap_len > 0:
                chunk_diffs.append(np.mean(np.linalg.norm(
                    prev[:overlap_len] - curr[:overlap_len], axis=-1)))
        chunk_overlap_mean = float(np.mean(chunk_diffs)) if chunk_diffs else 0.0

        # 2. Eye smoothness (angular velocity magnitude stats)
        eye_vel_mags = [np.linalg.norm(v) for v in metric_eye_vels]
        eye_angular_vel_mean = float(np.mean(eye_vel_mags)) if eye_vel_mags else 0.0
        eye_angular_vel_std = float(np.std(eye_vel_mags)) if eye_vel_mags else 0.0

        # 3. Joint acceleration (finite difference of velocity)
        if len(metric_joint_positions) >= 3:
            positions = np.array(metric_joint_positions)     # (T, 14)
            vel = np.diff(positions, axis=0)                 # (T-1, 14)
            accel = np.diff(vel, axis=0)                     # (T-2, 14)
            joint_accel_mean = float(np.mean(np.linalg.norm(accel, axis=-1)))
        else:
            joint_accel_mean = 0.0

        # 4. Value stats
        value_mean = float(np.mean(metric_values)) if metric_values else 0.0
        value_std = float(np.std(metric_values)) if metric_values else 0.0
        value_initial = float(metric_values[0]) if metric_values else 0.0
        value_max = float(np.max(metric_values)) if metric_values else 0.0

        # 5. EE idle fraction (both arms must be below threshold)
        EE_IDLE_THRESH = 0.001  # 1mm per step
        ee_arr = np.array(metric_ee_positions)  # (T, 6)
        if len(ee_arr) >= 2:
            left_deltas = np.linalg.norm(np.diff(ee_arr[:, :3], axis=0), axis=-1)
            right_deltas = np.linalg.norm(np.diff(ee_arr[:, 3:], axis=0), axis=-1)
            both_idle = (left_deltas < EE_IDLE_THRESH) & (right_deltas < EE_IDLE_THRESH)
            ee_idle_fraction = float(np.mean(both_idle))
        else:
            ee_idle_fraction = 0.0

        # 6. Time-to-stage (per stage)
        steps_to_stage = {}
        for s in task.stages:
            steps_to_stage[s] = stage_first_steps.get(s, step_idx + 1)

        # 7. Gaze metrics (angular distance to targets + fixation stability)
        gaze_metrics = {}
        if metric_gaze_angular_dists:
            # Per-episode mean angular distance to each target
            target_names = list(metric_gaze_angular_dists[0].keys())
            for tname in target_names:
                vals = [d[tname] for d in metric_gaze_angular_dists]
                gaze_metrics[f"gaze_mean_deg_{tname}"] = float(np.degrees(np.mean(vals)))

            # Per-stage mean angular distance
            if len(task.stages) > 1:
                # Build stage boundaries from stage_first_steps
                stage_list = list(task.stages)
                boundaries = []
                for s in stage_list:
                    boundaries.append(stage_first_steps.get(s, len(metric_gaze_angular_dists)))
                # Windows: [0, first_boundary), [first_boundary, second_boundary), ...
                window_starts = [0] + boundaries[:-1]
                window_ends = boundaries
                for si, stage_name in enumerate(stage_list):
                    ws = window_starts[si]
                    we = window_ends[si]
                    if we <= ws:
                        continue
                    for tname in target_names:
                        vals = [d[tname] for d in metric_gaze_angular_dists[ws:we]]
                        gaze_metrics[f"gaze_{stage_name}_mean_deg_{tname}"] = float(np.degrees(np.mean(vals)))

        # Fixation stability: std of angular changes between consecutive gaze dirs
        if len(metric_gaze_directions) >= 2:
            gaze_dirs = np.array(metric_gaze_directions)  # (T, 3)
            dot_products = np.sum(gaze_dirs[:-1] * gaze_dirs[1:], axis=-1)
            angular_changes = np.arccos(np.clip(dot_products, -1, 1))
            gaze_metrics["gaze_stability_deg"] = float(np.degrees(np.std(angular_changes)))
        # ──────────────────────────────────────────────────────────────────

        # Save episode video (full-res + small version for wandb upload)
        if save_video and len(frames) > 0:
            video_path = eval_dir / f"episode_{global_idx:03d}.mp4"
            imageio.mimwrite(
                str(video_path), frames, fps=args.video_fps, codec="h264",
                output_params=["-crf", "23", "-pix_fmt", "yuv420p"],
            )
            # Small version for wandb (480p height, maintains aspect ratio)
            from PIL import Image
            max_h = 480
            if frames[0].shape[0] > max_h:
                scale = max_h / frames[0].shape[0]
                new_w = int(frames[0].shape[1] * scale) // 2 * 2  # ensure even
                small_frames = [
                    np.array(Image.fromarray(f).resize((new_w, max_h), Image.LANCZOS))
                    for f in frames
                ]
            else:
                small_frames = frames
            small_path = eval_dir / f"episode_{global_idx:03d}_small.mp4"
            imageio.mimwrite(
                str(small_path), small_frames, fps=args.video_fps, codec="h264",
                output_params=["-crf", "28", "-pix_fmt", "yuv420p"],
            )
            print(f"  Video → {video_path} ({len(frames)} frames)")

        # Save obs dump
        if obs_log is not None and len(obs_log["proprio"]) > 0:
            obs_path = eval_dir / f"episode_{global_idx:03d}_obs.npz"
            np.savez_compressed(
                str(obs_path),
                **{k: np.stack(v) for k, v in obs_log.items()},
            )
            print(f"  Obs dump → {obs_path}")

        successes.append(success)
        episode_results.append({
            "episode": global_idx,
            "config": ep_config,
            "success": bool(success),
            "steps": step_idx + 1,
            "sim_time": float(env.sim_time),
            **{f"stage_{k}": bool(v) for k, v in stages_achieved.items()},
            # Scalar metrics (auto-forwarded to wandb as episode/*)
            "chunk_overlap_mean": chunk_overlap_mean,
            "eye_angular_vel_mean": eye_angular_vel_mean,
            "eye_angular_vel_std": eye_angular_vel_std,
            "joint_accel_mean": joint_accel_mean,
            "value_mean": value_mean,
            "value_std": value_std,
            "value_initial": value_initial,
            "value_max": value_max,
            "ee_idle_fraction": ee_idle_fraction,
            **{f"steps_to_{s}": steps_to_stage[s] for s in task.stages},
            # Gaze metrics (angular distance to targets + fixation stability)
            **gaze_metrics,
            # Trajectory data for value plot (list/dict — skipped by scalar forwarding)
            "value_trajectory": [round(v, 4) for v in metric_values],
            "stage_first_steps": stage_first_steps,
        })
        n_ok = sum(successes)
        ep_bar.set_postfix(success_rate=f"{n_ok}/{len(successes)}")
        tqdm.write(f"Episode {global_idx}: {'SUCCESS' if success else 'FAIL'} ({step_idx + 1} steps)")

        # Progress marker for launcher monitoring (also used for incremental wandb upload)
        with open(eval_dir / f"progress_rank{args.rank}.jsonl", "a") as pf:
            pf.write(json.dumps({
                "episode": global_idx,
                "success": bool(success),
                "steps": step_idx + 1,
                "sim_time": float(env.sim_time),
                **{f"stage_{k}": bool(v) for k, v in stages_achieved.items()},
                "chunk_overlap_mean": chunk_overlap_mean,
                "eye_angular_vel_mean": eye_angular_vel_mean,
                "eye_angular_vel_std": eye_angular_vel_std,
                "joint_accel_mean": joint_accel_mean,
                "value_mean": value_mean,
                "value_std": value_std,
                "value_initial": value_initial,
                "value_max": value_max,
                "ee_idle_fraction": ee_idle_fraction,
                **{f"steps_to_{s}": steps_to_stage[s] for s in task.stages},
                **gaze_metrics,
                "value_trajectory": [round(v, 4) for v in metric_values],
                "stage_first_steps": stage_first_steps,
            }) + "\n")

    # Compute per-stage success rates
    stage_rates = {}
    if len(task.stages) > 1 and episode_results:
        for s in task.stages:
            stage_rates[s] = float(np.mean([ep.get(f"stage_{s}", False) for ep in episode_results]))

    # Save results JSON
    results = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "task": args.task,
        "action_type": action_mode,
        "n_episodes": args.n_episodes,
        "base_seed": args.seed,
        "max_steps": args.max_steps,
        "freeze_eye": args.freeze_eye,
        "chunk_publish_size": chunk_publish_size,
        "rank": args.rank,
        "world_size": args.world_size,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "stage_rates": stage_rates,
        "episodes": episode_results,
    }
    if args.world_size > 1:
        results_path = eval_dir / f"results_rank{args.rank}.json"
    else:
        results_path = eval_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    n_total = len(successes) if successes else 0
    n_ok = sum(successes) if successes else 0
    rate = np.mean(successes) if successes else 0.0
    print(f"\nSuccess rate: {n_ok}/{n_total} = {rate:.1%}")
    if stage_rates:
        parts = "  ".join(f"{s}: {stage_rates[s]:.0%}" for s in task.stages)
        print(f"Stage rates:  {parts}")
    print(f"Results → {results_path}")



if __name__ == "__main__":
    main()
