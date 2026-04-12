"""Quest Pro VR teleoperation for MuJoCo bimanual simulation.

Provides QuestTeleopSim: a MuJoCo sim with stereo camera rendering,
IK-based arm control, and shared state for threaded physics/render.

Stereo cameras are on a mocap body ("vr_head") that tracks the VR headset's
full 6DOF pose (position + orientation), completely decoupled from the
robot's eye gimbal. The robot's existing eye_camera on the gimbal is untouched.
"""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from eye.mujoco.eval import (
    FINGER_JOINT_MAX,
    FINGER_JOINT_MIN,
    FINGER_JOINT_RANGE,
    JOINT_NAMES,
    XML_PATH,
    _gripper_agent_to_mujoco,
)
from eye.ik_utils import setup_bimanual_ik as _setup_ik_solver, solve_bimanual_ik_batch as _solve_bimanual_ik_batch
from eye.mujoco.tasks import Task


# Half inter-pupillary distance in meters (63mm total)
HALF_IPD = 0.0315

# Stereo render resolution per eye
STEREO_WIDTH = 1800
STEREO_HEIGHT = 1800

# VR → MuJoCo coordinate transform
# WebXR: +X right, +Y up, -Z forward
# MuJoCo: +X forward, +Y left, +Z up
R_VR_TO_MJ = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
], dtype=np.float64)

# Position offset applied to VR head pose after coordinate transform (MuJoCo frame).
# Maps VR origin to a useful viewpoint in the sim.
VR_HEAD_OFFSET = np.array([-0.8, 0.0, -0.6], dtype=np.float64)

# Load font once at module level
try:
    _HUD_FONT = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 22)
except OSError:
    _HUD_FONT = ImageFont.load_default(size=22)


_hud_cache: dict[str, tuple[np.ndarray, int, int]] = {}

# World position where the HUD is anchored (near back wall, above table)
HUD_WORLD_POS = np.array([0.8, 0.5, 0.55], dtype=np.float64)


def _render_hud_patch(text: str, recording: bool) -> tuple[np.ndarray, int, int]:
    """Render text to a small RGBA patch image, cached by content."""
    cache_key = f"{text}|{recording}"
    if cache_key not in _hud_cache:
        padding = 8
        bg_color = (200, 50, 50) if recording else (255, 255, 255)
        text_color = (255, 255, 255) if recording else (0, 0, 0)

        scratch = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(scratch)
        bbox = draw.multiline_textbbox((0, 0), text, font=_HUD_FONT)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        patch_w = text_w + 2 * padding
        patch_h = text_h + 2 * padding
        patch_img = Image.new("RGB", (patch_w, patch_h), color=bg_color)
        draw = ImageDraw.Draw(patch_img)
        draw.text((padding, padding), text, fill=text_color, font=_HUD_FONT)
        _hud_cache[cache_key] = (np.array(patch_img, dtype=np.uint8), patch_w, patch_h)

        # Keep cache small
        if len(_hud_cache) > 4:
            oldest = next(iter(_hud_cache))
            del _hud_cache[oldest]

    return _hud_cache[cache_key]


def _project_world_to_pixel(
    world_pos: np.ndarray,
    scene_cam,
    img_w: int,
    img_h: int,
    fovy_deg: float,
) -> tuple[int, int] | None:
    """Project a 3D world point to pixel coordinates using MuJoCo scene camera."""
    cam_pos = np.array(scene_cam.pos, dtype=np.float64)
    cam_fwd = np.array(scene_cam.forward, dtype=np.float64)
    cam_up = np.array(scene_cam.up, dtype=np.float64)
    cam_right = np.cross(cam_fwd, cam_up)
    rn = np.linalg.norm(cam_right)
    if rn < 1e-10:
        return None
    cam_right /= rn

    d = world_pos - cam_pos
    z_cam = np.dot(d, cam_fwd)
    if z_cam <= 0:
        return None  # Behind camera

    x_cam = np.dot(d, cam_right)
    y_cam = np.dot(d, cam_up)

    fovy_rad = np.radians(fovy_deg)
    aspect = img_w / img_h
    half_h = np.tan(fovy_rad / 2) * z_cam
    half_w = half_h * aspect

    px = int(round((x_cam / half_w + 1) / 2 * img_w))
    py = int(round((1 - y_cam / half_h) / 2 * img_h))
    return px, py


def _blit_patch(
    img: np.ndarray,
    patch: np.ndarray,
    cx: int,
    cy: int,
    x_offset: int = 0,
) -> None:
    """Blit a patch onto img centered at (cx + x_offset, cy), clamped to bounds."""
    ph, pw = patch.shape[:2]
    img_h, img_w = img.shape[:2]

    x = x_offset + cx - pw // 2
    y = cy - ph // 2

    # Compute overlap region
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + pw)
    y2 = min(img_h, y + ph)

    if x1 >= x2 or y1 >= y2:
        return

    # Corresponding patch region
    px1 = x1 - x
    py1 = y1 - y
    px2 = px1 + (x2 - x1)
    py2 = py1 + (y2 - y1)

    img[y1:y2, x1:x2] = patch[py1:py2, px1:px2]


@dataclasses.dataclass
class SharedState:
    """Thread-safe shared state between physics and render/control threads."""

    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    # Written by control thread, read by physics thread
    joint_targets: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(14, dtype=np.float64)
    )
    eye_pan: float = 0.0
    eye_tilt: float = np.deg2rad(30.0)

    # VR head pose (6DOF mocap tracking)
    head_pos: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([-0.3, 0.0, -0.4], dtype=np.float64)
    )
    head_quat: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # wxyz
    )
    head_quat_vr: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # xyzw
    )

    # Written by physics thread, read by control/render thread
    proprio: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(14, dtype=np.float64)
    )
    qpos_snapshot: np.ndarray | None = None
    mjdata_snapshot: dict[str, np.ndarray] | None = None

    # Set by control thread to request a reset; physics thread performs it
    pending_reset_seed: int | None = None

    # HUD overlay state (set from any thread, drawn onto stereo frames)
    hud_text: str | None = None
    hud_recording: bool = False


class QuestTeleopSim:
    """MuJoCo sim with stereo cameras for Quest Pro VR teleoperation.

    Adds left/right eye cameras to the eye_tilt body at runtime,
    provides stereo rendering and IK-based arm control.
    """

    def __init__(
        self,
        xml_path: str | Path = XML_PATH,
        task: Task | None = None,
        stereo_width: int = STEREO_WIDTH,
        stereo_height: int = STEREO_HEIGHT,
        stereo_fovy: float = 110.0,
        ipd_mm: float = 69.0,
        show_polygons: bool = False,
    ):
        self.stereo_width = stereo_width
        self.stereo_height = stereo_height
        self.stereo_fovy = stereo_fovy
        half_ipd = ipd_mm / 1440.0  # mm → meters, halved

        # Build model via MjSpec so we can add stereo cameras
        spec = mujoco.MjSpec.from_file(str(xml_path))
        if task is not None:
            task.configure_scene(spec)
            if show_polygons:
                task.configure_polygon_geoms(spec)

        # VR operator cameras on a mocap body (6DOF head tracking).
        # Completely decoupled from the robot's eye gimbal.
        vr_head = spec.worldbody.add_body()
        vr_head.name = "vr_head"
        vr_head.mocap = True
        vr_head.pos = [-0.3, 0.0, -0.4]  # initial: behind and below workspace

        # Quaternion matching eye_camera's xyaxes="0 -1 0 0 0 1"
        # Verified by reading compiled cam_quat from the existing eye_camera
        cam_quat = [-0.5, -0.5, 0.5, 0.5]  # wxyz

        left_cam = vr_head.add_camera()
        left_cam.name = "left_eye_camera"
        left_cam.pos = [0, half_ipd, 0]
        left_cam.quat = cam_quat
        left_cam.fovy = stereo_fovy

        right_cam = vr_head.add_camera()
        right_cam.name = "right_eye_camera"
        right_cam.pos = [0, -half_ipd, 0]
        right_cam.quat = cam_quat
        right_cam.fovy = stereo_fovy

        self.mjm = spec.compile()
        self.mjd = mujoco.MjData(self.mjm)

        # Cache mocap ID for the VR head body
        vr_head_body_id = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, "vr_head"
        )
        self.vr_head_mocap_id = self.mjm.body_mocapid[vr_head_body_id]

        if task is not None:
            task.setup(self.mjm, self.mjd)
        self.task = task

        self.physics_dt = self.mjm.opt.timestep

        # Cache joint/actuator IDs
        self.joint_ids = [
            mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]
        self.eye_pan_act = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, "eye_pan"
        )
        self.eye_tilt_act = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, "eye_tilt"
        )

        # Camera IDs for rendering
        self.left_cam_id = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_CAMERA, "left_eye_camera"
        )
        self.right_cam_id = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_CAMERA, "right_eye_camera"
        )

        # Renderers (EGL-backed)
        self.mjm.vis.global_.offwidth = max(
            self.mjm.vis.global_.offwidth, stereo_width
        )
        self.mjm.vis.global_.offheight = max(
            self.mjm.vis.global_.offheight, stereo_height
        )

        self._left_renderer = mujoco.Renderer(self.mjm, stereo_height, stereo_width)
        self._right_renderer = mujoco.Renderer(self.mjm, stereo_height, stereo_width)

        # Pre-allocated stereo buffer (avoids np.concatenate copy each frame)
        self._stereo_buf = np.empty((stereo_height, stereo_width * 2, 3), dtype=np.uint8)

        # Scene options: geom groups 0+2 (task objects + visual meshes)
        self._scene_option = mujoco.MjvOption()
        for g in range(len(self._scene_option.geomgroup)):
            self._scene_option.geomgroup[g] = 1 if g in (0, 2) else 0

        # Camera references
        self._left_cam_ref = mujoco.MjvCamera()
        self._left_cam_ref.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._left_cam_ref.fixedcamid = self.left_cam_id

        self._right_cam_ref = mujoco.MjvCamera()
        self._right_cam_ref.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._right_cam_ref.fixedcamid = self.right_cam_id

        # Render-local MjData for thread-safe rendering (physics owns self.mjd)
        self._render_mjd = mujoco.MjData(self.mjm)

        # IK solver (lazy init)
        self._pk_robot = None
        self._left_link_name = None
        self._right_link_name = None

        # Head pose tracking state (recentering support)
        self._head_offset = VR_HEAD_OFFSET.copy()
        self._head_rot_correction = np.eye(3, dtype=np.float64)
        self._prev_head_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Shared state for threading
        self.shared = SharedState()

    def update_head_pose(self, vr_pos: np.ndarray, vr_quat_xyzw: np.ndarray):
        """Convert VR head pose to MuJoCo frame and write to shared state.

        Applies the current head offset and rotation correction (set by
        reset_head_pose). Ensures quaternion sign continuity for smooth
        mocap interpolation.

        Args:
            vr_pos: (3,) head position in WebXR coords
            vr_quat_xyzw: (4,) head orientation in WebXR xyzw format
        """
        head_rotmat = quat_xyzw_to_rotmat(vr_quat_xyzw)
        mj_pos, mj_rotmat = vr_pose_to_mujoco(vr_pos, head_rotmat)
        mj_pos = self._head_rot_correction @ mj_pos + self._head_offset
        mj_rotmat = self._head_rot_correction @ mj_rotmat
        mj_quat = rotmat_to_wxyz(mj_rotmat)

        # Quaternion sign continuity (q and -q are the same rotation,
        # but sign flips cause jerky mocap interpolation in MuJoCo)
        if np.dot(mj_quat, self._prev_head_quat) < 0:
            mj_quat = -mj_quat
        self._prev_head_quat = mj_quat.copy()

        with self.shared.lock:
            self.shared.head_pos[:] = mj_pos
            self.shared.head_quat[:] = mj_quat
            self.shared.head_quat_vr[:] = vr_quat_xyzw.astype(np.float32)

    def reset_head_pose(self, vr_pos: np.ndarray, vr_quat_xyzw: np.ndarray):
        """Recenter head tracking: position + yaw only (preserves pitch/roll).

        Snaps the MuJoCo camera back to the default viewpoint and zeroes out
        the user's horizontal (yaw) rotation, but leaves pitch and roll
        untouched so the ground plane stays level.

        Args:
            vr_pos: (3,) current head position in WebXR coords
            vr_quat_xyzw: (4,) current head orientation in WebXR xyzw format
        """
        vr_rotmat = quat_xyzw_to_rotmat(vr_quat_xyzw)

        # Rotation: only correct yaw (rotation around MuJoCo Z axis).
        # Extract the current MuJoCo forward direction projected onto the XY plane,
        # then build a pure-yaw correction to rotate it back to +X (forward).
        current_mj_rotmat = R_VR_TO_MJ @ vr_rotmat @ R_VR_TO_MJ.T
        # Forward direction is first column of rotmat (MuJoCo +X = forward)
        fwd = current_mj_rotmat[:, 0]
        yaw = np.arctan2(fwd[1], fwd[0])  # angle from +X in XY plane
        # Build rotation matrix for -yaw around Z to cancel horizontal rotation
        c, s = np.cos(-yaw), np.sin(-yaw)
        self._head_rot_correction = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Position: preserve user's height, reset horizontal offset.
        # offset = target - rot_correction @ current_raw, so that
        # rot_correction @ current_raw + offset = target exactly at reset time.
        h = vr_pos[1]  # user height in VR (y-up)
        target_mj_pos = R_VR_TO_MJ @ np.array([0.0, h, 0.0]) + VR_HEAD_OFFSET
        current_raw_mj_pos = R_VR_TO_MJ @ vr_pos
        self._head_offset = target_mj_pos - self._head_rot_correction @ current_raw_mj_pos

        print(f"[Head Reset] Recentered (yaw correction: {np.degrees(yaw):.1f}°)")

    def correct_controller_pose(
        self, mj_pos: np.ndarray, mj_rotmat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the current head-reset yaw correction to a controller pose.

        Call this on MuJoCo-frame controller poses (after vr_pose_to_mujoco)
        so arm control stays consistent with the recentered head view.

        Returns:
            (corrected_pos, corrected_rotmat) in MuJoCo world frame.
        """
        corrected_pos = self._head_rot_correction @ mj_pos
        corrected_rotmat = self._head_rot_correction @ mj_rotmat
        return corrected_pos, corrected_rotmat

    def init_ik(self):
        """Initialize PyRoKi IK solver (JIT compiles on first call)."""
        self._pk_robot, self._left_link_name, self._right_link_name = (
            _setup_ik_solver()
        )
        # Warm up JIT with a dummy solve
        dummy_wxyz = np.array([[1.0, 0, 0, 0]], dtype=np.float32)
        dummy_pos = np.array([[0.3, 0.2, 0.3]], dtype=np.float32)
        dummy_q = np.zeros((1, 14), dtype=np.float32)
        _solve_bimanual_ik_batch(
            self._pk_robot,
            self._left_link_name,
            self._right_link_name,
            dummy_wxyz,
            dummy_pos,
            dummy_wxyz,
            dummy_pos,
            q_init=dummy_q,
        )
        print("[IK] Solver warmed up and ready")

    def reset(self, seed: int = 0, max_retries: int = 10):
        """Reset to keyframe, randomize task, run PD warmup, set eye pose.

        Retries with incremented seed if MuJoCo hits a collision/constraint error
        during warmup (e.g. objects spawned in bad configurations).
        """
        for attempt in range(max_retries):
            try:
                self._reset_inner(seed + attempt)
                return
            except mujoco.FatalError as e:
                print(f"[RESET] Seed {seed + attempt} failed ({e}), retrying...")
        # Last attempt — let it raise
        self._reset_inner(seed + max_retries)

    def _reset_inner(self, seed: int):
        mujoco.mj_resetDataKeyframe(self.mjm, self.mjd, 0)

        # Task randomization
        if self.task is not None:
            configs = self.task.generate_eval_configs(1, seed)
            self.task.apply_eval_config(self.mjm, self.mjd, configs[0])

        mujoco.mj_forward(self.mjm, self.mjd)

        # Set initial eye pose (30° tilt toward table)
        eye_pan_jid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_JOINT, "eye_pan"
        )
        eye_tilt_jid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_JOINT, "eye_tilt"
        )
        self.mjd.qpos[self.mjm.jnt_qposadr[eye_pan_jid]] = 0.0
        self.mjd.qpos[self.mjm.jnt_qposadr[eye_tilt_jid]] = np.deg2rad(30.0)
        self.mjd.ctrl[self.eye_pan_act] = 0.0
        self.mjd.ctrl[self.eye_tilt_act] = np.deg2rad(30.0)
        mujoco.mj_forward(self.mjm, self.mjd)

        # PD warmup
        for _ in range(500):
            mujoco.mj_step(self.mjm, self.mjd)

        # Re-place objects that may have shifted during warmup and zero velocities
        if self.task is not None and hasattr(self.task, 'post_warmup'):
            self.task.post_warmup(self.mjm, self.mjd, configs[0])
            mujoco.mj_forward(self.mjm, self.mjd)

        # Init shared state from current sim
        proprio = self.read_proprio()
        with self.shared.lock:
            self.shared.joint_targets[:] = proprio
            self.shared.proprio[:] = proprio
            self.shared.eye_pan = 0.0
            self.shared.eye_tilt = np.deg2rad(30.0)
            self.shared.qpos_snapshot = self.mjd.qpos.copy()

    def set_arm_ctrl(self, joints: np.ndarray):
        """Write 14 joint values to mjd.ctrl (same as MuJoCoTeleopSim)."""
        self.mjd.ctrl[2:8] = joints[:6]
        self.mjd.ctrl[8] = _gripper_agent_to_mujoco(joints[6])
        self.mjd.ctrl[9:15] = joints[7:13]
        self.mjd.ctrl[15] = _gripper_agent_to_mujoco(joints[13])

    def step(self, n_substeps: int = 1):
        """Step physics n_substeps times."""
        for _ in range(n_substeps):
            mujoco.mj_step(self.mjm, self.mjd)

    def read_proprio(self) -> np.ndarray:
        """Read 14-DOF from mjd.qpos with gripper mapped to [0, 1]."""
        joints = np.array([
            self.mjd.qpos[self.mjm.jnt_qposadr[jid]] for jid in self.joint_ids
        ])
        for idx in (6, 13):
            joints[idx] = np.clip(
                (joints[idx] - FINGER_JOINT_MIN) / FINGER_JOINT_RANGE, 0.0, 1.0
            )
        return joints

    def update_ipd(self, ipd_meters: float):
        """Update stereo camera spacing from headset-reported IPD."""
        half_ipd = ipd_meters / 2.0
        # cam_pos is relative to parent body; Y axis is the IPD axis
        self.mjm.cam_pos[self.left_cam_id, 1] = half_ipd
        self.mjm.cam_pos[self.right_cam_id, 1] = -half_ipd
        print(f"[IPD] Updated stereo cameras: {ipd_meters * 1000:.1f} mm")

    def render_stereo(self) -> np.ndarray:
        """Render side-by-side stereo image using render-local MjData.

        Returns:
            (H, 2*W, 3) uint8 RGB array (left|right).
        """
        import time as _time

        # Copy current qpos + mocap head pose to render mjd
        _t0 = _time.perf_counter()
        with self.shared.lock:
            if self.shared.qpos_snapshot is not None:
                self._render_mjd.qpos[:] = self.shared.qpos_snapshot
            self._render_mjd.mocap_pos[self.vr_head_mocap_id] = self.shared.head_pos
            self._render_mjd.mocap_quat[self.vr_head_mocap_id] = self.shared.head_quat
        mujoco.mj_forward(self.mjm, self._render_mjd)
        _t1 = _time.perf_counter()

        # Render left eye (shadows disabled for perf)
        self._left_renderer.update_scene(
            self._render_mjd, self._left_cam_ref, scene_option=self._scene_option
        )
        self._left_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        left_rgb = self._left_renderer.render()
        _t2 = _time.perf_counter()

        # Render right eye (shadows disabled for perf)
        self._right_renderer.update_scene(
            self._render_mjd, self._right_cam_ref, scene_option=self._scene_option
        )
        self._right_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        right_rgb = self._right_renderer.render()
        _t3 = _time.perf_counter()

        # Side-by-side into pre-allocated buffer (avoids concatenate copy)
        w = self.stereo_width
        self._stereo_buf[:, :w, :] = left_rgb
        self._stereo_buf[:, w:, :] = right_rgb
        stereo = self._stereo_buf
        _t4 = _time.perf_counter()

        # Accumulate and periodically print sub-timings
        if not hasattr(self, '_rt_count'):
            self._rt_count = 0
            self._rt_fwd = self._rt_left = self._rt_right = self._rt_concat = 0.0
            self._rt_last = _t0
        self._rt_count += 1
        self._rt_fwd += _t1 - _t0
        self._rt_left += _t2 - _t1
        self._rt_right += _t3 - _t2
        self._rt_concat += _t4 - _t3
        if _t4 - self._rt_last >= 3.0 and self._rt_count > 0:
            n = self._rt_count
            print(f"[Render] fwd {self._rt_fwd/n*1000:.1f}ms  "
                  f"left {self._rt_left/n*1000:.1f}ms  "
                  f"right {self._rt_right/n*1000:.1f}ms  "
                  f"concat {self._rt_concat/n*1000:.1f}ms")
            self._rt_count = 0
            self._rt_fwd = self._rt_left = self._rt_right = self._rt_concat = 0.0
            self._rt_last = _t4

        # Overlay HUD anchored to a world position
        with self.shared.lock:
            hud_text = self.shared.hud_text
            hud_recording = self.shared.hud_recording
        if hud_text is not None:
            patch, pw, ph = _render_hud_patch(hud_text, hud_recording)

            # Project world point through each eye camera
            left_px = _project_world_to_pixel(
                HUD_WORLD_POS,
                self._left_renderer.scene.camera[0],
                self.stereo_width,
                self.stereo_height,
                self.stereo_fovy,
            )
            right_px = _project_world_to_pixel(
                HUD_WORLD_POS,
                self._right_renderer.scene.camera[0],
                self.stereo_width,
                self.stereo_height,
                self.stereo_fovy,
            )

            if left_px is not None:
                _blit_patch(stereo, patch, left_px[0], left_px[1])
            if right_px is not None:
                _blit_patch(stereo, patch, right_px[0], right_px[1], x_offset=self.stereo_width)

        return stereo

    def solve_ik_single(
        self,
        left_target_pos: np.ndarray,
        left_target_wxyz: np.ndarray,
        right_target_pos: np.ndarray,
        right_target_wxyz: np.ndarray,
        current_proprio: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve bimanual IK for a single target pair.

        All inputs are 1D arrays. current_proprio is (14,) from read_proprio().
        Returns (1, 14) joint solution — caller extracts arm joints and sets grippers.
        """
        assert self._pk_robot is not None, "Call init_ik() first"

        q_init = None
        if current_proprio is not None:
            q_init = current_proprio[np.newaxis].astype(np.float32)

        result = _solve_bimanual_ik_batch(
            self._pk_robot,
            self._left_link_name,
            self._right_link_name,
            left_target_wxyz[np.newaxis],
            left_target_pos[np.newaxis],
            right_target_wxyz[np.newaxis],
            right_target_pos[np.newaxis],
            q_init=q_init,
        )
        # result shape is (1, 14) — unwrap batch dim
        return np.asarray(result)[0]

    def compute_fk_mujoco(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute EE poses using MuJoCo sites (approximate but fast, no JAX).

        Returns:
            (left_pos, left_rotmat, right_pos, right_rotmat)
            where pos is (3,) and rotmat is (3,3).
        """
        left_site = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_site"
        )
        right_site = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_site"
        )

        left_pos = self.mjd.site_xpos[left_site].copy()
        left_rotmat = self.mjd.site_xmat[left_site].reshape(3, 3).copy()
        right_pos = self.mjd.site_xpos[right_site].copy()
        right_rotmat = self.mjd.site_xmat[right_site].reshape(3, 3).copy()

        return left_pos, left_rotmat, right_pos, right_rotmat


def vr_pose_to_mujoco(
    vr_pos: np.ndarray, vr_rotmat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Transform VR controller pose to MuJoCo world frame.

    Args:
        vr_pos: (3,) position in WebXR coords
        vr_rotmat: (3,3) rotation matrix in WebXR coords

    Returns:
        (mj_pos, mj_rotmat): position and rotation in MuJoCo world frame
    """
    mj_pos = R_VR_TO_MJ @ vr_pos
    mj_rotmat = R_VR_TO_MJ @ vr_rotmat @ R_VR_TO_MJ.T
    return mj_pos, mj_rotmat


def rotmat_to_wxyz(rotmat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to wxyz quaternion."""
    # Using MuJoCo's quaternion convention
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotmat.flatten())
    return quat


def quat_xyzw_to_rotmat(xyzw: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion to 3x3 rotation matrix via MuJoCo."""
    x, y, z, w = xyzw
    wxyz = np.array([w, x, y, z], dtype=np.float64)
    rotmat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotmat, wxyz)
    return rotmat.reshape(3, 3)


def head_orientation_to_eye_angles(
    head_quat_xyzw: np.ndarray,
) -> tuple[float, float]:
    """Extract yaw/pitch from Quest head orientation for eye gimbal control.

    Args:
        head_quat_xyzw: (4,) quaternion in WebXR xyzw format

    Returns:
        (pan, tilt) in radians for MuJoCo eye actuators
    """
    # Convert xyzw to rotation matrix
    # WebXR quaternion is (x, y, z, w)
    x, y, z, w = head_quat_xyzw
    # Convert to wxyz for MuJoCo
    wxyz = np.array([w, x, y, z])
    rotmat = np.zeros(9)
    mujoco.mju_quat2Mat(rotmat, wxyz)
    rotmat = rotmat.reshape(3, 3)

    # Transform to MuJoCo frame
    mj_rotmat = R_VR_TO_MJ @ rotmat @ R_VR_TO_MJ.T

    # Extract yaw (pan) and pitch (tilt) from rotation matrix
    # MuJoCo eye: pan = rotation around Z, tilt = rotation around Y
    # Forward direction is the first column of rotmat (after frame transform)
    forward = mj_rotmat[:, 0]  # Forward in MuJoCo is +X

    pan = np.arctan2(forward[1], forward[0])  # yaw

    # eye_tilt axis="0 1 0": positive = look down.
    # WebXR look-up = positive rotation around X → mj forward Z > 0.
    # Negate Z so looking up gives negative tilt delta from the 30° default.
    tilt = np.arctan2(-forward[2], np.sqrt(forward[0] ** 2 + forward[1] ** 2))
    tilt = tilt + np.deg2rad(30.0)  # offset to default 30° table-viewing angle

    return float(pan), float(tilt)
