"""Run the bimanual YAM sine demo and stream it to a browser via viser.

Usage (remote host):
    python simulation/viser_demo.py --port 8080

Client (laptop): forward the port and open http://localhost:8080
    ssh -L 8080:localhost:8080 <remote-host>
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "third_party" / "robots_realtime"))

import mujoco  # noqa: E402
import viser  # noqa: E402
from robots_realtime.runtime.sim._mujoco_viser import (  # noqa: E402
    _get_body_name,
    _is_fixed_body,
    _merge_geoms,
)

sys.path.insert(0, str(_HERE))
from env import BimanualYamWorkbenchEnv  # noqa: E402


VISIBLE_GROUPS = (1, 2)  # drop group 0 (LIBERO collision hulls) and group 3 (yam collision)
SINE_INDICES = np.array([1, 2, 8, 9])
SINE_AMP = 0.3
SINE_FREQ_HZ = 0.25


def build_scene(server: viser.ViserServer, model, data) -> dict[int, object]:
    """Populate the viser scene from a compiled MjModel.

    Caller must run mj_forward(model, data) before this so data.xpos/xquat
    hold world-frame poses for fixed bodies (nested LIBERO objects have
    geom-owning children at local pos=0, so only world pose is correct).

    Returns a dict mapping movable body_id → mesh handle; update
    handle.position/wxyz each frame from data.xpos/xquat.
    """
    body_visual: dict[int, list[int]] = {}
    for gid in range(model.ngeom):
        if int(model.geom_group[gid]) in VISIBLE_GROUPS:
            body_visual.setdefault(int(model.geom_bodyid[gid]), []).append(gid)

    mesh_handles: dict[int, object] = {}
    server.scene.add_frame("/fixed", show_axes=False)

    for body_id, geom_ids in body_visual.items():
        body_name = _get_body_name(model, body_id)

        if _is_fixed_body(model, body_id):
            plane_ids = [g for g in geom_ids if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
            nonplane_ids = [g for g in geom_ids if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
            for gid in plane_ids:
                gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
                server.scene.add_grid(
                    f"/fixed/{body_name}/{gname}",
                    width=20.0,
                    height=20.0,
                    position=model.geom_pos[gid],
                    wxyz=model.geom_quat[gid],
                )
            if nonplane_ids:
                merged = _merge_geoms(model, nonplane_ids)
                server.scene.add_mesh_trimesh(
                    f"/fixed/{body_name}",
                    merged,
                    position=tuple(data.xpos[body_id]),
                    wxyz=tuple(data.xquat[body_id]),
                )
        else:
            merged = _merge_geoms(model, geom_ids)
            handle = server.scene.add_mesh_trimesh(f"/bodies/{body_name}", merged)
            mesh_handles[body_id] = handle

    return mesh_handles


def main():
    parser = argparse.ArgumentParser(description="Viser-streamed YAM sine demo")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--fps", type=float, default=30.0, help="Visual update rate")
    parser.add_argument(
        "--no-scanned-objects", action="store_true",
        help="Use the LIBERO hope-object defaults instead of the scanned cap-x/objects grid",
    )
    parser.add_argument("--low-poly", action="store_true", help="Use decimated 10k-face scanned meshes")
    args = parser.parse_args()

    print("Building env ...")
    env = BimanualYamWorkbenchEnv(
        scanned_objects=not args.no_scanned_objects,
        low_poly=args.low_poly,
    )
    print(f"  model: nu={env.model.nu} nq={env.model.nq} nbody={env.model.nbody}")

    print("Starting viser ...")
    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"  Viser: http://{args.host}:{args.port}  (use SSH port-forwarding)")

    # mj_forward populates data.xpos/xquat so fixed bodies with nested
    # geom-owning children get placed in world frame (local pos = 0).
    mujoco.mj_forward(env.model, env.data)

    print("Building scene ...")
    mesh_handles = build_scene(server, env.model, env.data)
    print(f"  {len(mesh_handles)} movable bodies")

    for body_id, handle in mesh_handles.items():
        handle.position = tuple(env.data.xpos[body_id])
        handle.wxyz = tuple(env.data.xquat[body_id])

    # Aim the default client camera at the center of the two arms in *this*
    # scene (the robots_realtime default was calibrated for a different frame).
    left_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "left_arm")
    right_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "right_arm")
    left_pos = np.asarray(env.data.xpos[left_id])
    right_pos = np.asarray(env.data.xpos[right_id])
    target = (left_pos + right_pos) / 2.0 + np.array([0.0, 0.0, 0.2])
    cam_pos = target + np.array([1.2, 0.0, 0.6])
    print(f"  left_arm at  {left_pos.round(3).tolist()}")
    print(f"  right_arm at {right_pos.round(3).tolist()}")
    print(f"  camera target {target.round(3).tolist()}, pos {cam_pos.round(3).tolist()}")

    @server.on_client_connect
    def _set_camera(client):
        client.camera.position = tuple(cam_pos)
        client.camera.look_at = tuple(target)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    # Simple GUI
    paused = {"v": False}
    with server.gui.add_folder("Playback"):
        btn_pause = server.gui.add_button("Pause")
        btn_reset = server.gui.add_button("Reset")

    @btn_pause.on_click
    def _(_):
        paused["v"] = not paused["v"]
        btn_pause.label = "Resume" if paused["v"] else "Pause"

    @btn_reset.on_click
    def _(_):
        env.reset()

    home = env.get_state_14d()
    step_dt = 1.0 / args.fps
    n_physics = max(1, int(round(step_dt / env.model.opt.timestep)))

    print(f"Running: {args.fps:.0f} Hz visual, {n_physics} mj_step per frame.")
    try:
        while True:
            frame_start = time.time()

            if not paused["v"]:
                t = env.data.time
                action = home.copy()
                action[SINE_INDICES] += SINE_AMP * np.sin(2.0 * np.pi * SINE_FREQ_HZ * t)
                env.step(action, decimation=n_physics)

            # Push body poses to viser
            for body_id, handle in mesh_handles.items():
                handle.position = tuple(env.data.xpos[body_id])
                handle.wxyz = tuple(env.data.xquat[body_id])

            dt = step_dt - (time.time() - frame_start)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
