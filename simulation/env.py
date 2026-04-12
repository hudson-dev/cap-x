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

# (xml_path, prefix, [x, y, z], quat or None)
# z = -0.41 (table surface). Tomato sauce needs 90-deg rotation around X to stand upright.
OBJECT_PLACEMENTS = [
    (TOMATO_SAUCE_XML, "tomato_",       [0.2,  0.55, -0.39], [0.7071, 0.7071, 0, 0]),
    (BBQ_SAUCE_XML,    "bbq_",          [0.2,  0.40, -0.385], None),
    (CREAM_CHEESE_XML, "cream_cheese_", [0.2,  0.25, -0.43], None),
]

# Arm base positions: 12 inches (0.3048 m) left/right of center, at table surface (z = -0.43)
# Y center shifted +0.15 m so both arms land on the visible workbench table surface.
LEFT_ARM_POS = [0.0, 0.7048, -0.43]
RIGHT_ARM_POS = [0.0, 0.0952, -0.43]

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


def build_model() -> mujoco.MjModel:
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

    # Attach left arm with "left_" prefix
    left_mount = scene_spec.worldbody.add_frame()
    left_mount.pos = LEFT_ARM_POS
    left_mount.attach_body(left_spec.worldbody.first_body(), "left_", "")

    # Attach right arm with "right_" prefix
    right_mount = scene_spec.worldbody.add_frame()
    right_mount.pos = RIGHT_ARM_POS
    right_mount.attach_body(right_spec.worldbody.first_body(), "right_", "")

    # Add static tabletop objects
    for xml_path, prefix, pos, quat in OBJECT_PLACEMENTS:
        _attach_object(scene_spec, xml_path, prefix, pos, quat)

    return scene_spec.compile()


class BimanualYamWorkbenchEnv:
    """Two YAM robot arms on a messy workbench."""

    def __init__(self):
        self.model = build_model()
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
