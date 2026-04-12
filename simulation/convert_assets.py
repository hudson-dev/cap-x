"""Convert the World Labs textured mesh GLB to OBJ + texture for MuJoCo.

The textured GLB already contains UV-mapped geometry and a baked PNG texture.
We just need to:
  1. Extract the texture image from the GLB binary.
  2. Load the mesh with UVs via trimesh.
  3. Convert OpenGL Y-up to MuJoCo Z-up.
  4. Export OBJ + MTL + PNG.
"""

import json
import struct
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
TEXTURED_MESH_PATH = (
    ROOT / "worlds" / "Messy Workbench" / "Messy Workbench Room_texture_mesh.glb"
)
ASSETS_DIR = ROOT / "simulation" / "assets"


def extract_texture_from_glb(glb_path: Path, out_path: Path):
    """Extract the embedded PNG texture from the GLB binary."""
    with open(glb_path, "rb") as f:
        _magic, _version, _length = struct.unpack("<III", f.read(12))

        # JSON chunk
        chunk_len, _ = struct.unpack("<II", f.read(8))
        gltf = json.loads(f.read(chunk_len))

        # BIN chunk
        _chunk_len2, _ = struct.unpack("<II", f.read(8))
        bin_data = f.read(_chunk_len2)

    # Find the image's bufferView
    img_info = gltf["images"][0]
    bv = gltf["bufferViews"][img_info["bufferView"]]
    offset = bv.get("byteOffset", 0)
    length = bv["byteLength"]

    png_bytes = bin_data[offset : offset + length]

    with open(out_path, "wb") as f:
        f.write(png_bytes)

    # Get image dimensions
    img = Image.open(out_path)
    print(f"  Texture: {img.size[0]}x{img.size[1]}, {length / 1e6:.1f}MB")
    return img


def opengl_to_mujoco(vertices: np.ndarray) -> np.ndarray:
    """Convert World Labs OpenGL export to MuJoCo Z-up.

    The textured mesh raw axes:
      X: width (~4.3m), Y: depth (~8.4m), Z: height (~2.8m)
    MuJoCo Z-up is already correct — just pass through.
    Negate Z if upside-down.
    """
    return vertices.copy()


def convert():
    """Load textured GLB, fix coords, export OBJ + MTL + PNG for MuJoCo."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Extract the baked texture ---
    print(f"Extracting texture from {TEXTURED_MESH_PATH.name} ...")
    tex_path = ASSETS_DIR / "workbench_visual.png"
    extract_texture_from_glb(TEXTURED_MESH_PATH, tex_path)

    # --- Load mesh with UVs ---
    print(f"Loading mesh ...")
    scene = trimesh.load(str(TEXTURED_MESH_PATH))

    # Get the geometry (there's only one mesh in this GLB)
    geom_name = list(scene.geometry.keys())[0]
    mesh = scene.geometry[geom_name]
    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    # Extract UV coords from texture visual
    uvs = None
    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        uvs = mesh.visual.uv
    elif hasattr(mesh.visual, "vertex_attributes"):
        uvs = mesh.visual.vertex_attributes.get("TEXCOORD_0")

    if uvs is None:
        raise RuntimeError("No UV coordinates found in mesh!")
    print(f"  UVs: {uvs.shape}")

    # --- Convert coordinates ---
    vertices = opengl_to_mujoco(mesh.vertices)
    print(f"  After OpenGL→MuJoCo bounds: {vertices.min(0)} to {vertices.max(0)}")

    # --- Export OBJ + MTL ---
    obj_path = ASSETS_DIR / "workbench_visual.obj"
    mtl_path = ASSETS_DIR / "workbench_visual.mtl"

    with open(mtl_path, "w") as f:
        f.write("newmtl workbench_material\n")
        f.write("Ka 0.2 0.2 0.2\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write(f"map_Kd {tex_path.name}\n")
    print(f"  Saved material: {mtl_path}")

    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_path.name}\n")
        f.write("usemtl workbench_material\n")

        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for face in mesh.faces:
            i, j, k = face[0] + 1, face[1] + 1, face[2] + 1
            f.write(f"f {i}/{i} {j}/{j} {k}/{k}\n")

    print(f"  Saved mesh: {obj_path} ({len(mesh.faces)} faces)")
    print(f"  Extents: {vertices.max(0) - vertices.min(0)}")
    return mesh


if __name__ == "__main__":
    convert()
