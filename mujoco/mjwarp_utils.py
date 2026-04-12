"""MuJoCo built-in renderer helpers.

Provides fisheye distortion (equidistant model matching the real eyeball camera)
and a reusable renderer class that wraps ``mujoco.Renderer`` (EGL-backed).
"""

import cv2
import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from eye.camera import (
    FISH_FOCAL, FISH_NATIVE_W, FISH_NATIVE_H, FISH_DIST_COEFFS,
)


# ──────────────────────────────────────────────
# Fisheye distortion (equidistant model)
# ──────────────────────────────────────────────


def _undistort_points_numpy(xd, yd, dist_coeffs, n_iter=50):
    """Newton's method undistortion, vectorized numpy.

    Matches the iterative solver in eye/camera.py exactly.
    Given distorted normalized coordinates (xd, yd), solves for undistorted
    (xu, yu) such that distort(xu, yu) = (xd, yd).
    """
    k1, k2, p1, p2, k3, k4 = dist_coeffs

    x = xd.copy()
    y = yd.copy()

    for _ in range(n_iter):
        r2 = x * x + y * y
        d = 1.0 + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)))

        fx = x * d + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) - xd
        fy = y * d + 2 * p2 * x * y + p1 * (r2 + 2 * y * y) - yd

        d_r = k1 + r2 * (2 * k2 + r2 * (3 * k3 + r2 * 4 * k4))
        d_x = 2 * x * d_r
        d_y = 2 * y * d_r

        fx_x = d + d_x * x + 2 * p1 * y + 6 * p2 * x
        fx_y = d_y * x + 2 * p1 * x + 2 * p2 * y
        fy_x = d_x * y + 2 * p2 * y + 2 * p1 * x
        fy_y = d + d_y * y + 2 * p2 * x + 6 * p1 * y

        # Matches camera.py sign convention exactly
        denominator = fy_x * fx_y - fx_x * fy_y
        x_num = fx * fy_y - fy * fx_y
        y_num = fy * fx_x - fx * fy_x

        mask = np.abs(denominator) > 1e-10
        x = np.where(mask, x + x_num / denominator, x)
        y = np.where(mask, y + y_num / denominator, y)

    return x, y


def build_fisheye_to_pinhole_remap(out_W, out_H, pin_fovy_deg, pin_W, pin_H):
    """Build remap tables: fisheye output pixel → pinhole render pixel.

    For each pixel in the output fisheye image, computes where to sample from
    the wide-angle pinhole render. Uses the equidistant fisheye model matching
    eye/sim/spherical_video.py ``_get_local_cam_rays()``.

    Args:
        out_W, out_H: Output fisheye image dimensions.
        pin_fovy_deg: Pinhole camera vertical FoV in degrees.
        pin_W, pin_H: Pinhole render resolution.

    Returns:
        map_x, map_y: ``(out_H, out_W)`` float32 arrays for ``cv2.remap``.
    """
    # Pinhole camera intrinsics (from MuJoCo fovy)
    f_pin = (pin_H / 2) / np.tan(np.radians(pin_fovy_deg / 2))
    cx_pin, cy_pin = pin_W / 2, pin_H / 2

    # Fisheye camera intrinsics (scaled from native 1920×1200 calibration)
    fx_fish = FISH_FOCAL * (out_W / FISH_NATIVE_W)
    fy_fish = FISH_FOCAL * (out_H / FISH_NATIVE_H)
    cx_fish, cy_fish = out_W / 2, out_H / 2

    # Create pixel grid for output fisheye image (pixel centers at +0.5,
    # matching the convention in spherical_video.py _get_local_cam_rays)
    u = np.arange(out_W, dtype=np.float64) + 0.5
    v = np.arange(out_H, dtype=np.float64) + 0.5
    u_grid, v_grid = np.meshgrid(u, v)  # (out_H, out_W)

    # Step 1: Normalize by fisheye K matrix
    xd = (u_grid - cx_fish) / fx_fish
    yd = (v_grid - cy_fish) / fy_fish

    # Step 2: Undistort (Newton's method — same as training pipeline)
    xu, yu = _undistort_points_numpy(xd, yd, FISH_DIST_COEFFS)

    # Step 3: Equidistant fisheye → 3D ray direction
    # In equidistant model: theta = ||(xu, yu)|| is the angle from optical axis
    theta = np.sqrt(xu**2 + yu**2)
    theta = np.clip(theta, 0, np.pi)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Numerically stable sin(theta)/theta (sinc function)
    eps = 1e-8
    safe_theta = np.where(theta < eps, 1.0, theta)
    factor = np.where(theta < eps, 1.0 - theta**2 / 6.0, sin_theta / safe_theta)

    # 3D ray: (xu * sinc(theta), yu * sinc(theta), cos(theta))
    ray_x = xu * factor
    ray_y = yu * factor
    ray_z = cos_theta

    # Step 4: Project 3D ray onto pinhole image plane
    valid = ray_z > 0.001
    map_x = np.where(valid, f_pin * ray_x / ray_z + cx_pin, -1).astype(np.float32)
    map_y = np.where(valid, f_pin * ray_y / ray_z + cy_pin, -1).astype(np.float32)

    return map_x, map_y


def build_elp_to_pinhole_remap(K_elp, dist_elp, out_W, out_H, pin_fovy_deg, pin_W, pin_H):
    """Build remap tables: ELP distorted output pixel → wide pinhole render pixel.

    Same pattern as ``build_fisheye_to_pinhole_remap`` but for the standard
    OpenCV radial-tangential distortion model (real ELP stereo camera).
    After undistortion the ray is simply ``[xu, yu, 1]`` — no equidistant step.

    Args:
        K_elp: (3, 3) ELP intrinsic matrix.
        dist_elp: (5,) or (6,) OpenCV distortion coefficients [k1,k2,p1,p2,k3(,k4)].
        out_W, out_H: Output (distorted) ELP image dimensions.
        pin_fovy_deg: Internal pinhole vertical FoV in degrees.
        pin_W, pin_H: Internal pinhole render resolution.

    Returns:
        map_x, map_y: ``(out_H, out_W)`` float32 arrays for ``cv2.remap``.
    """
    # Pinhole camera intrinsics (from MuJoCo fovy)
    f_pin = (pin_H / 2) / np.tan(np.radians(pin_fovy_deg / 2))
    cx_pin, cy_pin = pin_W / 2, pin_H / 2

    # ELP camera intrinsics
    fx, fy = K_elp[0, 0], K_elp[1, 1]
    cx, cy = K_elp[0, 2], K_elp[1, 2]

    # Pixel grid for output ELP image
    u = np.arange(out_W, dtype=np.float64) + 0.5
    v = np.arange(out_H, dtype=np.float64) + 0.5
    u_grid, v_grid = np.meshgrid(u, v)

    # Step 1: Normalize by ELP K
    xd = (u_grid - cx) / fx
    yd = (v_grid - cy) / fy

    # Step 2: Undistort (same Newton solver as fisheye path)
    dist_6 = np.zeros(6)
    dist_6[:len(dist_elp)] = dist_elp
    xu, yu = _undistort_points_numpy(xd, yd, dist_6)

    # Step 3: Project undistorted ray onto MuJoCo pinhole (standard model — no equidistant step)
    map_x = (f_pin * xu + cx_pin).astype(np.float32)
    map_y = (f_pin * yu + cy_pin).astype(np.float32)

    # Mask pixels outside the pinhole render
    valid = (map_x >= 0) & (map_x <= pin_W - 1) & (map_y >= 0) & (map_y <= pin_H - 1)
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0

    return map_x, map_y


# ──────────────────────────────────────────────
# MuJoCo Built-in Renderer
# ──────────────────────────────────────────────

class MujocoRenderer:
    """Wraps ``mujoco.Renderer`` (EGL-backed rasterizer) for a single camera.

    When ``K`` and ``dist`` are provided, renders a wider-FoV pinhole
    internally and applies a distortion remap to match the real camera.
    The distortion model is selected by ``is_fisheye``:

    - ``is_fisheye=True``:  equidistant fisheye (Insta360).
    - ``is_fisheye=False``: standard OpenCV radial-tangential (ELP stereo).

    When ``K`` and ``dist`` are not provided, renders a plain pinhole.

    Args:
        mjm: MuJoCo MjModel.
        cam_name: Name of the camera to render.
        width: Output image width in pixels.
        height: Output image height in pixels.
        K: (3, 3) intrinsic matrix of the real camera. If provided, enables
            distortion remap.
        dist: Distortion coefficients [k1, k2, p1, p2, k3, k4]. Can be a
            torch.Tensor or np.ndarray.
        is_fisheye: If True, use equidistant fisheye model; otherwise use
            standard OpenCV radtan model.
        fisheye: **Deprecated** — use ``K, dist, is_fisheye=True`` instead.
            When True and K/dist not provided, uses the default Insta360
            fisheye calibration from ``get_default_video_config()``.
        pin_fovy: Pinhole vertical FoV in degrees for the internal render.
            Default 125° covers the full fisheye diagonal with margin.
            Auto-computed for non-fisheye cameras.
        pin_scale: Resolution multiplier for the internal pinhole render.
            1.5× gives f_pin/f_fish ≈ 0.85 — good center quality for the fovea.
    """

    def __init__(
        self,
        mjm: mujoco.MjModel,
        cam_name: str,
        width: int,
        height: int,
        K: np.ndarray | None = None,
        dist: torch.Tensor | np.ndarray | None = None,
        is_fisheye: bool = False,
        fisheye: bool = False,
        pin_fovy: float = 125.0,
        pin_scale: float = 1.5,
    ):
        # Backward compat: fisheye=True without explicit K/dist → use default fisheye calib
        if K is None and fisheye:
            from eye.camera import get_default_video_config
            K, dist, is_fisheye = get_default_video_config(width, height)

        has_distortion = K is not None and dist is not None
        self.fisheye = is_fisheye if has_distortion else False

        # Convert dist to numpy if needed
        if has_distortion and isinstance(dist, torch.Tensor):
            dist = dist.numpy()

        self.cam_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

        if has_distortion:
            self.out_width = width
            self.out_height = height
            render_width = int(width * pin_scale)
            render_height = int(height * pin_scale)

            if is_fisheye:
                mjm.cam_fovy[self.cam_id] = pin_fovy
                self._remap_x, self._remap_y = build_fisheye_to_pinhole_remap(
                    width, height, pin_fovy, render_width, render_height,
                )
            else:
                # Standard OpenCV radtan — compute pin_fovy from K
                fy = K[1, 1]
                cam_vfov = 2 * np.degrees(np.arctan(height / (2 * fy)))
                pin_fovy = cam_vfov + 5.0
                mjm.cam_fovy[self.cam_id] = pin_fovy
                self._remap_x, self._remap_y = build_elp_to_pinhole_remap(
                    K, dist, width, height, pin_fovy, render_width, render_height,
                )

            # Precompute normalized grid for GPU grid_sample.
            grid_x = 2.0 * self._remap_x / render_width - 1.0
            grid_y = 2.0 * self._remap_y / render_height - 1.0
            grid = np.stack([grid_x, grid_y], axis=-1)  # (out_H, out_W, 2)
            self._grid_gpu = torch.from_numpy(grid).unsqueeze(0).cuda()  # (1, out_H, out_W, 2)
        else:
            render_width = width
            render_height = height
            self._grid_gpu = None

        self.width = render_width
        self.height = render_height

        # Ensure offscreen framebuffer is large enough for our render resolution.
        # MuJoCo's Renderer requires offwidth/offheight >= render dimensions.
        mjm.vis.global_.offwidth = max(mjm.vis.global_.offwidth, render_width)
        mjm.vis.global_.offheight = max(mjm.vis.global_.offheight, render_height)

        # Create EGL-backed renderer
        self._renderer = mujoco.Renderer(mjm, render_height, render_width)
        self.last_cpu_rgb: np.ndarray | None = None

        # Configure scene options: geom groups and shadows
        self._scene_option = mujoco.MjvOption()
        # Groups 0+2: task objects + visual meshes.
        # Group 3 (collision capsules) intentionally excluded.
        # MjvOption.geomgroup is a length-6 boolean array indexed by group number.
        for g in range(len(self._scene_option.geomgroup)):
            self._scene_option.geomgroup[g] = 1 if g in (0, 2) else 0

        # Create a fixed camera reference for update_scene
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._cam.fixedcamid = self.cam_id

    def render(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> np.ndarray:
        """Render the scene from the configured camera.

        Args:
            mjm: MuJoCo MjModel — same model used at init.
            mjd: MuJoCo MjData with current state.

        Returns:
            (H, W, 3) uint8 RGB array. If fisheye is enabled, this is the
            distorted fisheye image at (out_height, out_width, 3).
        """
        self._renderer.update_scene(mjd, self._cam, scene_option=self._scene_option)
        # Set shadow flag after update_scene — update_scene resets all scene flags.
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        rgb = self._renderer.render()

        if self._grid_gpu is not None:
            rgb = cv2.remap(
                rgb.astype(np.float32), self._remap_x, self._remap_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            ).astype(np.uint8)

        return rgb

    def render_torch(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, device: str = "cuda") -> torch.Tensor:
        """Render the scene and return a GPU tensor, applying fisheye via grid_sample.

        Unlike render() which stays on CPU numpy, this uploads the raw pinhole
        render to GPU once and does the fisheye remap there — saving the CPU
        remap + separate GPU upload.

        Args:
            mjm: MuJoCo MjModel — same model used at init.
            mjd: MuJoCo MjData with current state.
            device: Target device (default "cuda").

        Returns:
            (3, H, W) float32 tensor in [0, 1] on the target device.
        """
        self._renderer.update_scene(mjd, self._cam, scene_option=self._scene_option)
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        rgb = self._renderer.render()  # CPU numpy (H, W, 3) uint8

        img = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float() / 255.0  # (3, H, W)

        if self._grid_gpu is not None:
            # GPU remap (fisheye or ELP distortion).
            # (1, 3, pin_H, pin_W) → grid_sample → (1, 3, out_H, out_W)
            img = F.grid_sample(
                img.unsqueeze(0), self._grid_gpu,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            ).squeeze(0)  # (3, out_H, out_W)

        # Cache CPU uint8 for video saving (GPU→CPU transfer, no cv2.remap needed).
        self.last_cpu_rgb = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()

        return img


# Backwards compatibility alias
MJWarpRenderer = MujocoRenderer


# ──────────────────────────────────────────────
# 360° Equirectangular Rendering (cube-map)
# ──────────────────────────────────────────────

# Camera names in the 360° XML, ordered: +X, -X, +Y, -Y, +Z, -Z.
CUBE_CAM_NAMES = ["cube_px", "cube_nx", "cube_py", "cube_ny", "cube_pz", "cube_nz"]
CUBE_LEFT_CAM_NAMES = ["cube_left_px", "cube_left_nx", "cube_left_py",
                        "cube_left_ny", "cube_left_pz", "cube_left_nz"]
CUBE_RIGHT_CAM_NAMES = ["cube_right_px", "cube_right_nx", "cube_right_py",
                         "cube_right_ny", "cube_right_pz", "cube_right_nz"]

# Camera xyaxes from the XML — each row is (cam_X_world, cam_Y_world).
# Camera Z = cross(cam_X, cam_Y); camera looks along -Z_cam.
_CUBE_XYAXES = np.array([
    [0, -1, 0,   0, 0, 1],   # +X: looks along +X
    [0,  1, 0,   0, 0, 1],   # -X: looks along -X
    [1,  0, 0,   0, 0, 1],   # +Y: looks along +Y
    [-1, 0, 0,   0, 0, 1],   # -Y: looks along -Y
    [0, -1, 0,  -1, 0, 0],   # +Z: looks along +Z (up)
    [0, -1, 0,   1, 0, 0],   # -Z: looks along -Z (down)
], dtype=np.float64)


def _cube_rotation_matrices() -> np.ndarray:
    """Build world-to-camera rotation matrices for the 6 cube faces.

    Returns:
        (6, 3, 3) array where ``R[f] @ world_ray`` gives the ray in camera f's frame.
        Camera convention: +X = right, +Y = up, -Z = forward (MuJoCo default).
    """
    R = np.zeros((6, 3, 3), dtype=np.float64)
    for f in range(6):
        cam_x = _CUBE_XYAXES[f, :3]
        cam_y = _CUBE_XYAXES[f, 3:]
        cam_z = np.cross(cam_x, cam_y)
        # Rows of R are the camera axes expressed in world coords
        # R @ world_vec = cam_vec
        R[f, 0] = cam_x
        R[f, 1] = cam_y
        R[f, 2] = cam_z
    return R


def build_equirect360_to_pinhole_remap(
    out_W: int = 2562,
    out_H: int = 1282,
    pin_W: int = 1024,
    pin_H: int = 1024,
    pin_fovy: float = 100.0,
    blend_power: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build remap tables: equirectangular pixel -> cube-face pinhole pixel.

    Az/el convention matches ``spherical_video.py`` (lines 148-152):
        u = atan2(-y, x) / pi  in [-1, 1],  u=0 is +X direction
        v = atan2(xy_radius, z) / pi  in [0, 1],  v=0 is +Z (up)

    Args:
        out_W, out_H: Output equirectangular image dimensions.
        pin_W, pin_H: Per-face pinhole render resolution.
        pin_fovy: Pinhole vertical FoV in degrees (should be >= 90 for coverage).
        blend_power: Exponent for blending weights. Higher = sharper transitions.

    Returns:
        map_x:   (6, out_H, out_W) float32 — pixel x-coord in each face render.
        map_y:   (6, out_H, out_W) float32 — pixel y-coord in each face render.
        weights: (6, out_H, out_W) float32 — blending weights per face.
        valid:   (6, out_H, out_W) bool    — whether projection is valid for this face.
    """
    # Pinhole intrinsics
    f_pin = (pin_H / 2) / np.tan(np.radians(pin_fovy / 2))
    cx_pin, cy_pin = pin_W / 2, pin_H / 2

    # Equirectangular pixel grid -> 3D rays
    u_px = np.arange(out_W, dtype=np.float64) + 0.5
    v_px = np.arange(out_H, dtype=np.float64) + 0.5
    u_grid, v_grid = np.meshgrid(u_px, v_px)  # (out_H, out_W)

    # Normalized coords matching spherical_video.py convention
    u_norm = u_grid / out_W * 2 - 1   # [-1, 1], 0 = +X direction
    v_norm = v_grid / out_H           # [0, 1], 0 = +Z (up), 1 = -Z (down)
    az = u_norm * np.pi               # azimuth
    pol = v_norm * np.pi              # polar angle from +Z

    # 3D ray directions (matching atan2(-y, x) convention)
    sin_pol = np.sin(pol)
    ray_x = sin_pol * np.cos(az)
    ray_y = -sin_pol * np.sin(az)
    ray_z = np.cos(pol)
    rays = np.stack([ray_x, ray_y, ray_z], axis=-1)  # (out_H, out_W, 3)

    # Rotation matrices: world -> camera
    R = _cube_rotation_matrices()  # (6, 3, 3)

    # Project rays onto each cube face
    map_x = np.full((6, out_H, out_W), -1, dtype=np.float32)
    map_y = np.full((6, out_H, out_W), -1, dtype=np.float32)
    weights = np.zeros((6, out_H, out_W), dtype=np.float32)
    valid = np.zeros((6, out_H, out_W), dtype=bool)

    rays_flat = rays.reshape(-1, 3)  # (N, 3)

    for f in range(6):
        # Transform rays to camera frame: (N, 3) @ (3, 3).T = (N, 3)
        cam_rays = rays_flat @ R[f].T  # (N, 3) in camera coords

        cam_z = cam_rays[:, 2]

        # MuJoCo camera: X=right, Y=up, Z=cross(X,Y). Looks along -Z_cam.
        # Points in front have cam_z < 0, so proj_z = -cam_z > 0.
        #
        # Projection to pixel coords (MuJoCo render() flips rows so row 0 = top):
        #   px = f * cam_x / proj_z + cx    (cam_x > 0 = right = larger column)
        #   py = f * (-cam_y) / proj_z + cy (cam_y > 0 = up = smaller row, so negate)
        #
        # This matches the CV convention used by build_fisheye_to_pinhole_remap
        # where ray = (cam_x, -cam_y, -cam_z) i.e. Y-down, Z-forward.
        proj_z = -cam_z

        in_front = proj_z > 1e-6

        # Project onto image plane
        px = np.where(in_front, f_pin * cam_rays[:, 0] / proj_z + cx_pin, -1)
        py = np.where(in_front, f_pin * (-cam_rays[:, 1]) / proj_z + cy_pin, -1)

        # Check pixel bounds (with small margin for interpolation)
        in_bounds = in_front & (px >= -0.5) & (px < pin_W + 0.5) & (py >= -0.5) & (py < pin_H + 0.5)

        # Edge tapering: smoothly fade weights to 0 near face edges to avoid
        # black-bleed artifacts from bilinear interpolation at boundaries.
        margin = pin_W * 0.05  # fade over outer 5% of face
        edge_dist_x = np.minimum(px, pin_W - px)
        edge_dist_y = np.minimum(py, pin_H - py)
        edge_mask = np.clip(np.minimum(edge_dist_x, edge_dist_y) / margin, 0, 1)

        # Weight = cos(angle from face center) ^ blend_power, tapered at edges
        # cos(angle) = proj_z / ||ray|| = proj_z (rays are unit length)
        w = np.where(in_bounds, np.power(np.clip(proj_z, 0, 1), blend_power) * edge_mask, 0)

        map_x[f] = px.reshape(out_H, out_W).astype(np.float32)
        map_y[f] = py.reshape(out_H, out_W).astype(np.float32)
        weights[f] = w.reshape(out_H, out_W).astype(np.float32)
        valid[f] = in_bounds.reshape(out_H, out_W)

    # Error check: every output pixel should have nonzero total weight
    total_weight = (weights * valid).sum(axis=0)
    eps = 1e-8
    uncovered = (total_weight < eps).sum()
    if uncovered > 0:
        import warnings
        warnings.warn(
            f"build_equirect360_to_pinhole_remap: {uncovered} output pixels have "
            f"near-zero total weight (fovy={pin_fovy}° may be too narrow for full coverage)."
        )

    return map_x, map_y, weights, valid


class Equirect360Renderer:
    """Renders an equirectangular image from MuJoCo cameras.

    Two modes:

    - **Full 360°** (default): 6 cube-map cameras with overlap blending.
      Requires cameras named ``cube_px`` … ``cube_nz`` (or custom via
      ``cam_names``).
    - **Single-camera partial equirect**: Pass ``cam_name``, ``K``, and
      ``dist`` to render one camera with calibrated distortion into a partial
      equirect (only the camera's FOV is covered, rest is black).

    Args:
        mjm: MuJoCo MjModel.
        mjd: MuJoCo MjData (used once at init to read cam_xmat).
        out_width: Output equirectangular image width.
        out_height: Output equirectangular image height.
        cam_name: Single camera name for partial equirect mode. When
            provided, ``K`` and ``dist`` must also be set.
        K: (3, 3) intrinsic matrix for the single camera.
        dist: Distortion coefficients [k1, k2, p1, p2, k3, k4].
        is_fisheye: Distortion model (True=equidistant, False=OpenCV radtan).
        pin_fovy: Pinhole vertical FoV for cube faces (full 360° mode only).
        pin_resolution: Per-face render resolution (full 360° mode only).
        blend_power: Blending exponent (full 360° mode only).
        cam_names: Override cube-map camera names (full 360° mode only).
    """

    def __init__(
        self,
        mjm: mujoco.MjModel,
        mjd: mujoco.MjData,
        out_width: int = 2562,
        out_height: int = 1282,
        cam_name: str | None = None,
        K: np.ndarray | None = None,
        dist: torch.Tensor | np.ndarray | None = None,
        is_fisheye: bool = False,
        pin_fovy: float = 100.0,
        pin_resolution: int | None = None,
        blend_power: float = 6.0,
        cam_names: list[str] | None = None,
    ):
        self.out_width = out_width
        self.out_height = out_height

        if cam_name is not None:
            # Single-camera partial equirect mode
            assert K is not None and dist is not None, \
                "K and dist must be provided for single-camera equirect mode"
            self.cam_name = cam_name
            self._init_single_cam(mjm, mjd, cam_name, K, dist, is_fisheye)
        else:
            # Full 360° cube-map mode
            self.cam_name = "equirect360"
            self._init_cubemap(mjm, mjd, pin_fovy, pin_resolution, blend_power, cam_names)

    def _init_single_cam(self, mjm, mjd, cam_name, K, dist, is_fisheye):
        """Single camera → partial equirect with calibrated distortion."""
        if isinstance(dist, torch.Tensor):
            dist = dist.numpy()

        cam_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            raise ValueError(f"Camera {cam_name!r} not found in model")

        # Internal pinhole: wider FOV so remap covers camera after distortion
        fy = K[1, 1]
        out_h_cam = int(2 * K[1, 2] + 1)  # approximate image height from cy
        cam_vfov = 2 * np.degrees(np.arctan(out_h_cam / (2 * fy)))
        pin_fovy = cam_vfov + 5.0
        pin_w = int(K[0, 2] * 2 * 1.5)  # ~1.5x camera width
        pin_h = int(K[1, 2] * 2 * 1.5)  # ~1.5x camera height
        mjm.cam_fovy[cam_id] = pin_fovy

        # Get camera orientation (fixed body — one forward pass sufficient)
        mujoco.mj_forward(mjm, mjd)
        cam_xmat = mjd.cam_xmat[cam_id].reshape(3, 3).copy()

        # Build equirect → distorted camera → wide pinhole remap
        map_x, map_y = _build_elp_equirect_remap(
            self.out_width, self.out_height, pin_w, pin_h, pin_fovy,
            cam_xmat, K, dist,
        )

        self._n_faces = 1
        self._map_x = map_x[np.newaxis]  # (1, H, W)
        self._map_y = map_y[np.newaxis]
        valid = ((map_x >= 0) & (map_y >= 0))[np.newaxis]  # (1, H, W)
        self._norm_weights = valid.astype(np.float32)
        self._face_winner = np.zeros((self.out_height, self.out_width), dtype=np.int32)
        self.pin_W = pin_w
        self.pin_H = pin_h

        # GPU grid
        grid_x = 2.0 * map_x / pin_w - 1.0
        grid_y = 2.0 * map_y / pin_h - 1.0
        inv = ~valid[0]
        grid_x[inv] = -2.0
        grid_y[inv] = -2.0
        grid = np.stack([grid_x, grid_y], axis=-1)
        self._gpu_grids = [torch.from_numpy(grid).unsqueeze(0).cuda()]
        self._gpu_norm_weights = [
            torch.from_numpy(self._norm_weights[0]).unsqueeze(0).unsqueeze(0).cuda()
        ]

        # Renderer
        mjm.vis.global_.offwidth = max(mjm.vis.global_.offwidth, pin_w)
        mjm.vis.global_.offheight = max(mjm.vis.global_.offheight, pin_h)
        self._renderer = mujoco.Renderer(mjm, height=pin_h, width=pin_w)
        self.last_cpu_rgb: np.ndarray | None = None

        self._scene_option = mujoco.MjvOption()
        for g in range(len(self._scene_option.geomgroup)):
            self._scene_option.geomgroup[g] = 1 if g in (0, 2) else 0

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cam_id
        self._cams = [cam]
        self.cam_ids = [cam_id]

    def _init_cubemap(self, mjm, mjd, pin_fovy, pin_resolution, blend_power, cam_names):
        """6 cube-map cameras → full 360° equirect with blending."""
        pin_res = pin_resolution or self.out_height
        self.pin_W = pin_res
        self.pin_H = pin_res

        cam_names = cam_names or CUBE_CAM_NAMES
        self._n_faces = len(cam_names)

        # Look up camera IDs and override fovy
        self.cam_ids = []
        for name in cam_names:
            cid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid < 0:
                raise ValueError(f"Camera {name!r} not found in MuJoCo model")
            self.cam_ids.append(cid)
            mjm.cam_fovy[cid] = pin_fovy

        # Build remap tables
        map_x, map_y, weights, valid = build_equirect360_to_pinhole_remap(
            self.out_width, self.out_height, pin_res, pin_res, pin_fovy, blend_power,
        )

        # Precompute normalized blend weights: w_norm[f] = w[f] * valid[f] / sum
        total = (weights * valid).sum(axis=0, keepdims=True)  # (1, H, W)
        eps = 1e-8
        self._norm_weights = (weights * valid) / (total + eps)  # (6, H, W)

        # CPU remap tables (for cv2.remap path)
        self._map_x = map_x  # (6, H, W)
        self._map_y = map_y  # (6, H, W)

        # Precompute face winner map for segmentation rendering
        self._face_winner = np.argmax(self._norm_weights, axis=0).astype(np.int32)

        # Precompute GPU grids and weights for grid_sample path
        grid_x = 2.0 * map_x / pin_res - 1.0
        grid_y = 2.0 * map_y / pin_res - 1.0
        for f in range(self._n_faces):
            grid_x[f][~valid[f]] = -2.0
            grid_y[f][~valid[f]] = -2.0

        self._gpu_grids = []
        self._gpu_norm_weights = []
        for f in range(self._n_faces):
            grid = np.stack([grid_x[f], grid_y[f]], axis=-1)
            self._gpu_grids.append(torch.from_numpy(grid).unsqueeze(0).cuda())
            self._gpu_norm_weights.append(
                torch.from_numpy(self._norm_weights[f]).unsqueeze(0).unsqueeze(0).cuda()
            )

        # Ensure offscreen framebuffer is large enough
        mjm.vis.global_.offwidth = max(mjm.vis.global_.offwidth, pin_res)
        mjm.vis.global_.offheight = max(mjm.vis.global_.offheight, pin_res)

        # Single MuJoCo renderer reused for all faces
        self._renderer = mujoco.Renderer(mjm, pin_res, pin_res)
        self.last_cpu_rgb: np.ndarray | None = None

        # Scene options: geom groups 0+2 (task objects + visual meshes)
        self._scene_option = mujoco.MjvOption()
        for g in range(len(self._scene_option.geomgroup)):
            self._scene_option.geomgroup[g] = 1 if g in (0, 2) else 0

        # Create MjvCamera objects (one per cube face)
        self._cams = []
        for cid in self.cam_ids:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = cid
            self._cams.append(cam)

    def render(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> np.ndarray:
        """Render equirectangular image (CPU path with cv2.remap).

        Returns:
            (out_H, out_W, 3) uint8 RGB array.
        """
        out = np.zeros((self.out_height, self.out_width, 3), dtype=np.float32)

        for f in range(self._n_faces):
            self._renderer.update_scene(mjd, self._cams[f], scene_option=self._scene_option)
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
            face_rgb = self._renderer.render()

            remapped = cv2.remap(
                face_rgb.astype(np.float32), self._map_x[f], self._map_y[f],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            w = self._norm_weights[f, :, :, np.newaxis]
            out += w * remapped

        result = np.clip(out, 0, 255).astype(np.uint8)
        self.last_cpu_rgb = result
        return result

    def render_torch(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, device: str = "cuda") -> torch.Tensor:
        """Render equirectangular image (GPU path with grid_sample).

        Returns:
            (3, out_H, out_W) float32 tensor in [0, 1] on the target device.
        """
        out = torch.zeros(1, 3, self.out_height, self.out_width, device=device)

        for f in range(self._n_faces):
            self._renderer.update_scene(mjd, self._cams[f], scene_option=self._scene_option)
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
            face_rgb = self._renderer.render()

            face_gpu = (
                torch.from_numpy(face_rgb).to(device)
                .permute(2, 0, 1).unsqueeze(0).float() / 255.0
            )

            sampled = F.grid_sample(
                face_gpu, self._gpu_grids[f],
                mode="bilinear", padding_mode="border", align_corners=False,
            )

            out += self._gpu_norm_weights[f] * sampled

        result = out.squeeze(0)

        self.last_cpu_rgb = (result.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()

        return result

    def render_segmentation(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> np.ndarray:
        """Render equirectangular segmentation map (geom IDs per pixel).

        Uses nearest-neighbor indexing and the face_winner map (argmax of blend
        weights) instead of blending, since blending would corrupt integer IDs.

        Returns:
            (out_H, out_W) int32 array of geom IDs. -1 for background.
        """
        self._renderer.enable_segmentation_rendering()

        face_segs = []
        for f in range(self._n_faces):
            self._renderer.update_scene(mjd, self._cams[f], scene_option=self._scene_option)
            seg = self._renderer.render()
            face_segs.append(seg[:, :, 0].astype(np.int32))

        self._renderer.disable_segmentation_rendering()

        out = np.full((self.out_height, self.out_width), -1, dtype=np.int32)

        for f in range(self._n_faces):
            ix = np.clip(np.round(self._map_x[f]).astype(np.intp), 0, self.pin_W - 1)
            iy = np.clip(np.round(self._map_y[f]).astype(np.intp), 0, self.pin_H - 1)
            remapped = face_segs[f][iy, ix]

            mask = self._face_winner == f
            out[mask] = remapped[mask]

        return out


def _build_elp_equirect_remap(
    out_w: int,
    out_h: int,
    pin_w: int,
    pin_h: int,
    pin_fovy_deg: float,
    cam_xmat: np.ndarray,
    K_elp: np.ndarray,
    dist_elp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build remap: equirect pixel → wide-pinhole pixel, accounting for ELP distortion.

    For each equirect output pixel:
    1. Compute world-space ray direction from equirect coordinates.
    2. Transform ray into camera frame using cam_xmat.
    3. Project onto the ELP sensor (apply K), then distort (OpenCV radtan model).
    4. Check if the distorted pixel is inside the ELP image (partial coverage).
    5. Undistort and project onto the wide internal pinhole for rendering.

    This means the equirect faithfully represents what the *distorted* real
    ELP camera sees, not an ideal pinhole.

    Returns:
        map_x, map_y: (out_h, out_w) float32 remap tables for cv2.remap.
    """
    f_pin = (pin_h / 2.0) / np.tan(np.radians(pin_fovy_deg / 2.0))
    cx_pin, cy_pin = pin_w / 2.0, pin_h / 2.0

    fx, fy = K_elp[0, 0], K_elp[1, 1]
    cx_elp, cy_elp = K_elp[0, 2], K_elp[1, 2]
    elp_w, elp_h = int(2 * cx_elp + 1), int(2 * cy_elp + 1)  # approximate image size

    # Distortion coefficients
    k1, k2, p1, p2, k3 = dist_elp[:5]

    # Equirect pixel grid → world rays
    u_grid, v_grid = np.meshgrid(
        np.arange(out_w, dtype=np.float64) + 0.5,
        np.arange(out_h, dtype=np.float64) + 0.5,
    )
    az = (u_grid / out_w * 2.0 - 1.0) * np.pi
    pol = v_grid / out_h * np.pi
    sin_pol = np.sin(pol)
    rays_world = np.stack(
        [sin_pol * np.cos(az), -sin_pol * np.sin(az), np.cos(pol)], axis=-1
    )

    # World → camera frame (cam_xmat is camera→world, so world→cam = cam_xmat.T)
    rays_cam = rays_world.reshape(-1, 3) @ cam_xmat
    proj_z = -rays_cam[:, 2]  # MuJoCo looks along -Z_cam
    in_front = proj_z > 1e-6

    # Undistorted normalized coords in camera frame
    xu = np.where(in_front, rays_cam[:, 0] / proj_z, 0.0)
    yu = np.where(in_front, -rays_cam[:, 1] / proj_z, 0.0)

    # Apply OpenCV distortion: undistorted → distorted
    r2 = xu**2 + yu**2
    radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = xu * radial + 2 * p1 * xu * yu + p2 * (r2 + 2 * xu**2)
    yd = yu * radial + 2 * p2 * xu * yu + p1 * (r2 + 2 * yu**2)

    # Distorted pixel coords in ELP image
    px_elp = fx * xd + cx_elp
    py_elp = fy * yd + cy_elp

    # Check if distorted pixel falls within sensor bounds (approximate from K)
    sensor_w = int(round(2 * cx_elp))
    sensor_h = int(round(2 * cy_elp))
    in_elp = in_front & (px_elp >= 0) & (px_elp < sensor_w) & (py_elp >= 0) & (py_elp < sensor_h)

    # Project the undistorted ray onto the wide internal pinhole
    px = np.where(in_elp, f_pin * xu + cx_pin, -1.0)
    py = np.where(in_elp, f_pin * yu + cy_pin, -1.0)

    in_bounds = in_elp & (px >= 0) & (px <= pin_w - 1) & (py >= 0) & (py <= pin_h - 1)
    map_x = px.reshape(out_h, out_w).astype(np.float32)
    map_y = py.reshape(out_h, out_w).astype(np.float32)
    inv = ~in_bounds.reshape(out_h, out_w)
    map_x[inv] = -1.0
    map_y[inv] = -1.0
    return map_x, map_y


# Keep backward-compat alias
StereoEquirectCamera = None  # Removed — use Equirect360Renderer(distortion="elp_left"/"elp_right")
