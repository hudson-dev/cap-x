"""MuJoCo simulation environment: bimanual YAM arms on a messy workbench."""

from pathlib import Path

import mujoco
import numpy as np

SIM_DIR = Path(__file__).resolve().parent
SCENE_XML = SIM_DIR / "scene.xml"
YAM_XML = (
    SIM_DIR.parent
    / "mujoco"
    / "robot_xmls"
    / "i2rt_yam"
    / "yam.xml"
)

_LIBERO = SIM_DIR.parent / "LIBERO-PRO" / "libero" / "libero" / "assets" / "stable_hope_objects"
TOMATO_SAUCE_XML = _LIBERO / "tomato_sauce" / "tomato_sauce.xml"
BBQ_SAUCE_XML    = _LIBERO / "bbq_sauce"    / "bbq_sauce.xml"
CREAM_CHEESE_XML = _LIBERO / "cream_cheese" / "cream_cheese.xml"

_LIBERO_SCANNED = SIM_DIR.parent / "LIBERO-PRO" / "libero" / "libero" / "assets" / "stable_scanned_objects"
BASKET_XML = _LIBERO_SCANNED / "basket" / "basket.xml"

# Basket position on the table, next to the scanned object grid.
# Adjust x/y to move it; z=-0.35 puts the basket bottom flush with the table surface (z=-0.41).
BASKET_POS = [0.875, -0.125, -0.44]

# (xml_path, prefix, [x, y, z], quat or None)
# z = -0.41 (table surface). Tomato sauce needs 90-deg rotation around X to stand upright.
OBJECT_PLACEMENTS = [
    (TOMATO_SAUCE_XML, "tomato_",       [0.2,  0.55, -0.39], [0.7071, 0.7071, 0, 0]),
    (BBQ_SAUCE_XML,    "bbq_",          [0.2,  0.40, -0.385], None),
    (CREAM_CHEESE_XML, "cream_cheese_", [0.2,  0.25, -0.43], None),
]

_SCANNED = SIM_DIR.parent / "objects"

# 2x3 grid on the table for scanned objects. Per-object z places the object's
# bottom on the table surface (z = -0.41), computed from each mesh's bounding box.
# quat [0.7071, 0.7071, 0, 0] = 90° around X: converts Y-up scan meshes to MuJoCo Z-up world.
# Per-object z is computed so the object's bottom rests on the table surface (z = -0.41).
# Global offset applied to all objects and arms: [x, y, 0]
# +x = further back on table, -y = further right
SCENE_OFFSET = [-0.75, 0.35, 0.0]

# Object-only offset, applied on top of SCENE_OFFSET: [x, y, 0]
# +x = further back, -x = further forward, +y = further left, -y = further right
OBJECT_OFFSET = [0.15, 0.0, 0.0]

# 2x3 grid origin (top-left object position) and spacing
# GRID_SPACING_X: distance between the 2 rows (back-to-front)
# GRID_SPACING_Y: distance between the 3 columns (left-right)
GRID_BASE_X   = 0.628
GRID_BASE_Y   = 0.023
GRID_SPACING_X = 0.2
GRID_SPACING_Y = 0.175

_RX90 = [0.7071, 0.7071, 0, 0]
_RX90_RY90 = [0.5, 0.5, 0.5, 0.5]   # RX90 + 90deg around Y

# Per-object z keeps each object's bottom flush with the table surface
_SCANNED_Z = {
    "aspirin":        -0.3652,
    "hot_sauce":      -0.3115,
    "jello":          -0.3738,
    "mustard":        -0.3138,
    "tomato_sauce":   -0.3603,
    "toothpaste_box": -0.3867,
}

def _scanned_placements(low: bool):
    gx, gy = GRID_BASE_X, GRID_BASE_Y
    sx, sy = GRID_SPACING_X, GRID_SPACING_Y
    items = [
        ("aspirin",        "aspirin_low.xml"        if low else "aspirin.xml",        "aspirin_",        0, 0, _RX90),
        ("hot_sauce",      "hot_sauce_low.xml"      if low else "hot_sauce.xml",      "hot_sauce_",      0, 1, _RX90),
        ("jello",          "jello_low.xml"          if low else "jello.xml",          "jello_",          0, 2, _RX90_RY90),
        ("mustard",        "mustard_low.xml"        if low else "mustard.xml",        "mustard_",        1, 0, _RX90),
        ("tomato_sauce",   "tomato_sauce_low.xml"   if low else "tomato_sauce.xml",   "scan_tomato_",    1, 1, _RX90),
        ("toothpaste_box", "toothpaste_box_low.xml" if low else "toothpaste_box.xml", "toothpaste_box_", 1, 2, _RX90),
    ]
    return [
        (_SCANNED / name / xml, prefix, [gx + row * sx, gy + col * sy, _SCANNED_Z[name]], quat)
        for name, xml, prefix, row, col, quat in items
    ]

# Arm base positions: 12 inches (0.3048 m) left/right of center, at table surface (z = -0.43)
# Y center shifted +0.15 m so both arms land on the visible workbench table surface.
LEFT_ARM_POS = [0.508, 0.4778, -0.43]
RIGHT_ARM_POS = [0.508, -0.1318, -0.43]

# Per-arm home configuration (from yam.xml keyframe)
_HOME_QPOS = [0, 1.047, 1.047, 0, 0, 0, 0, 0]  # joint1..6, left_finger, right_finger
_HOME_CTRL = [0, 1.047, 1.047, 0, 0, 0, 0]       # joint1..6, gripper


def _name_meshes(spec: mujoco.MjSpec):
    """Give explicit names to meshes that only have file paths.

    MjSpec's attach_body prefixes geom mesh references but not unnamed mesh
    assets, causing compilation failures. Setting names explicitly fixes this.
    """
    for m in spec.mesh:
        if not m.name:
            m.name = Path(m.file).stem


def _attach_object(scene_spec: mujoco.MjSpec, xml_path: Path, prefix: str, pos: list,
                   quat: list = None):
    """Attach a static LIBERO object into the scene at the given position.

    Resolves mesh/texture file paths to absolute so the scene's meshdir compiler
    directive does not incorrectly prepend its own assets/ path.
    quat: optional [w, x, y, z] rotation for the mount frame.
    """
    obj_spec = mujoco.MjSpec()
    obj_spec.from_file(str(xml_path))
    asset_dir = xml_path.parent

    for m in obj_spec.mesh:
        m.file = str((asset_dir / m.file).resolve())
    for t in obj_spec.texture:
        if t.file:
            t.file = str((asset_dir / t.file).resolve())

    mount = scene_spec.worldbody.add_frame()
    mount.pos = pos
    if quat is not None:
        mount.quat = quat
    mount.attach_body(obj_spec.worldbody.first_body(), prefix, "")


def build_model(scanned_objects: bool = False, low_poly: bool = False) -> mujoco.MjModel:
    """Compose the scene by attaching two YAM arms and tabletop objects to the workbench world."""
    scene_spec = mujoco.MjSpec()
    scene_spec.from_file(str(SCENE_XML))

    # Load yam.xml twice — attach_body detaches from source, so we need two copies
    left_spec = mujoco.MjSpec()
    left_spec.from_file(str(YAM_XML))
    _name_meshes(left_spec)

    right_spec = mujoco.MjSpec()
    right_spec.from_file(str(YAM_XML))
    _name_meshes(right_spec)

    ox, oy, oz = SCENE_OFFSET

    # Attach left arm with "left_" prefix
    left_mount = scene_spec.worldbody.add_frame()
    left_mount.pos = [LEFT_ARM_POS[0] + ox, LEFT_ARM_POS[1] + oy, LEFT_ARM_POS[2] + oz]
    left_mount.attach_body(left_spec.worldbody.first_body(), "left_", "")

    # Attach right arm with "right_" prefix
    right_mount = scene_spec.worldbody.add_frame()
    right_mount.pos = [RIGHT_ARM_POS[0] + ox, RIGHT_ARM_POS[1] + oy, RIGHT_ARM_POS[2] + oz]
    right_mount.attach_body(right_spec.worldbody.first_body(), "right_", "")

    # Add static tabletop objects
    if scanned_objects:
        placements = _scanned_placements(low=low_poly)
    else:
        placements = OBJECT_PLACEMENTS
    px, py, pz = OBJECT_OFFSET
    for xml_path, prefix, pos, quat in placements:
        _attach_object(scene_spec, xml_path, prefix,
                       [pos[0] + ox + px, pos[1] + oy + py, pos[2] + oz + pz], quat)

    # Basket (SCENE_OFFSET applied, but not OBJECT_OFFSET)
    _attach_object(scene_spec, BASKET_XML, "basket_",
                   [BASKET_POS[0] + ox, BASKET_POS[1] + oy, BASKET_POS[2] + oz])

    return scene_spec.compile()


class BimanualYamWorkbenchEnv:
    """Two YAM robot arms on a messy workbench."""

    def __init__(self, scanned_objects: bool = False, low_poly: bool = False):
        self.model = build_model(scanned_objects=scanned_objects, low_poly=low_poly)
        self.data = mujoco.MjData(self.model)

        self.n_actuators = self.model.nu

        # Cache end-effector body IDs
        self._left_ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_link_6"
        )
        self._right_ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_link_6"
        )

        self.reset()

    def reset(self) -> dict:
        """Reset simulation to home configuration."""
        mujoco.mj_resetData(self.model, self.data)

        # Set home qpos for both arms
        # qpos layout: [left_joint1..6, left_left_finger, left_right_finger,
        #               right_joint1..6, right_left_finger, right_right_finger]
        self.data.qpos[:8] = _HOME_QPOS
        self.data.qpos[8:16] = _HOME_QPOS

        # Set home ctrl for both arms
        # ctrl layout: [left_joint1..6, left_gripper,
        #               right_joint1..6, right_gripper]
        self.data.ctrl[:7] = _HOME_CTRL
        self.data.ctrl[7:14] = _HOME_CTRL

        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def step(self, ctrl: np.ndarray) -> dict:
        """Apply control and step the simulation.

        Args:
            ctrl: Control vector of shape (n_actuators,).
                  Left arm  [0:7]:  joint1-6 positions (rad) + gripper (0.0-0.041 m).
                  Right arm [7:14]: joint1-6 positions (rad) + gripper (0.0-0.041 m).
        """
        np.copyto(self.data.ctrl, ctrl[: self.n_actuators])
        mujoco.mj_step(self.model, self.data)
        return self.get_observation()

    def get_observation(self) -> dict:
        """Get current robot state."""
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "time": self.data.time,
        }

    def get_ee_pose(self, side: str = "left") -> tuple[np.ndarray, np.ndarray]:
        """Get end-effector position and orientation (quaternion).

        Args:
            side: "left" or "right".
        """
        ee_id = self._left_ee_id if side == "left" else self._right_ee_id
        pos = self.data.xpos[ee_id].copy()
        quat = self.data.xquat[ee_id].copy()
        return pos, quat
