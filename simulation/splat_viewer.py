"""Gaussian splat viewer for the messy workbench scene.

Loads a standard 3DGS PLY file and renders it using gsplat.
Can sync camera pose with a MuJoCo viewer for composite rendering.
"""

import struct
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parent
SPLAT_PLY = SIM_DIR.parent / "worlds" / "Messy Workbench" / "Messy Workbench Room.ply"


def load_ply(path: Path) -> dict:
    """Load a standard 3D Gaussian Splatting PLY file.

    Expected properties: x, y, z, nx, ny, nz, f_dc_0..2,
    opacity, scale_0..2, rot_0..3.
    """
    with open(path, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_points = 0
        properties = []
        for line in header_lines:
            if line.startswith("element vertex"):
                n_points = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                properties.append((parts[1], parts[2]))

        print(f"Loading {n_points} Gaussians from {path.name} ...")

        # Read binary data
        dtype_map = {"float": "f", "double": "d", "uchar": "B", "int": "i"}
        fmt = "<" + "".join(dtype_map.get(p[0], "f") for p in properties)
        point_size = struct.calcsize(fmt)

        raw = f.read(n_points * point_size)

    # Parse into arrays
    data = np.frombuffer(raw, dtype=np.float32).reshape(n_points, -1)

    # Map property names to column indices
    prop_names = [p[1] for p in properties]

    def col(name):
        return prop_names.index(name)

    result = {
        "positions": data[:, [col("x"), col("y"), col("z")]],
        "normals": data[:, [col("nx"), col("ny"), col("nz")]],
        "sh_dc": data[:, [col("f_dc_0"), col("f_dc_1"), col("f_dc_2")]],
        "opacity": data[:, col("opacity")],
        "scales": data[:, [col("scale_0"), col("scale_1"), col("scale_2")]],
        "rotations": data[
            :, [col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")]
        ],
    }

    print(f"  Loaded: {n_points} Gaussians")
    print(f"  Position range: {result['positions'].min(0)} to {result['positions'].max(0)}")
    return result


def sh_to_rgb(sh_dc: np.ndarray) -> np.ndarray:
    """Convert SH DC coefficients to RGB colors."""
    C0 = 0.28209479177387814
    return np.clip(sh_dc * C0 + 0.5, 0.0, 1.0)


class GaussianSplatRenderer:
    """Renders Gaussian splats using gsplat."""

    def __init__(self, ply_path: Path = SPLAT_PLY, device: str = "cuda"):
        import torch

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.splat_data = load_ply(ply_path)
        self._to_torch()

    def _to_torch(self):
        """Move splat data to torch tensors on device."""
        import torch

        self.means = torch.tensor(
            self.splat_data["positions"], dtype=torch.float32, device=self.device
        )
        self.scales = torch.exp(
            torch.tensor(
                self.splat_data["scales"], dtype=torch.float32, device=self.device
            )
        )
        self.quats = torch.nn.functional.normalize(
            torch.tensor(
                self.splat_data["rotations"], dtype=torch.float32, device=self.device
            ),
            dim=-1,
        )
        self.opacities = torch.sigmoid(
            torch.tensor(
                self.splat_data["opacity"], dtype=torch.float32, device=self.device
            )
        )
        self.colors = torch.tensor(
            sh_to_rgb(self.splat_data["sh_dc"]),
            dtype=torch.float32,
            device=self.device,
        )
        self.n_gaussians = len(self.means)
        print(f"  {self.n_gaussians} Gaussians on {self.device}")

    def render(
        self,
        viewmat: np.ndarray,
        K: np.ndarray,
        width: int = 640,
        height: int = 480,
    ) -> np.ndarray:
        """Render splats from a given camera viewpoint.

        Args:
            viewmat: 4x4 world-to-camera transform.
            K: 3x3 camera intrinsic matrix.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            RGB image as (H, W, 3) uint8 numpy array.
        """
        import torch
        from gsplat import rasterization

        viewmat_t = torch.tensor(viewmat, dtype=torch.float32, device=self.device)
        K_t = torch.tensor(K, dtype=torch.float32, device=self.device)

        renders, _, _ = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
            viewmats=viewmat_t[None],
            Ks=K_t[None],
            width=width,
            height=height,
            packed=False,
        )

        img = renders[0].clamp(0, 1).cpu().numpy()
        return (img * 255).astype(np.uint8)


def mujoco_cam_to_viewmat(cam_pos, cam_forward, cam_up) -> np.ndarray:
    """Convert MuJoCo camera state to a 4x4 world-to-camera matrix."""
    forward = np.array(cam_forward, dtype=np.float64)
    up = np.array(cam_up, dtype=np.float64)
    pos = np.array(cam_pos, dtype=np.float64)

    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    R = np.eye(4)
    R[:3, 0] = right
    R[:3, 1] = -up
    R[:3, 2] = forward
    R[:3, 3] = pos

    return np.linalg.inv(R).astype(np.float32)


def make_intrinsics(
    width: int = 640, height: int = 480, fov_y: float = 45.0
) -> np.ndarray:
    """Create a camera intrinsic matrix from image dimensions and vertical FOV."""
    fy = height / (2.0 * np.tan(np.radians(fov_y) / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
