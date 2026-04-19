"""Launch the bimanual YAM workbench simulation.

Usage:
    conda activate data_gen
    python simulation/run.py          # MuJoCo viewer only
    python simulation/run.py --splat  # With Gaussian splat rendering
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from env import BimanualYamWorkbenchEnv


def main():
    parser = argparse.ArgumentParser(description="Bimanual YAM Workbench Simulation")
    parser.add_argument(
        "--splat", action="store_true", help="Enable Gaussian splat rendering"
    )
    parser.add_argument(
        "--splat-width", type=int, default=640, help="Splat render width"
    )
    parser.add_argument(
        "--splat-height", type=int, default=480, help="Splat render height"
    )
    parser.add_argument(
        "--scanned-objects", action="store_true",
        help="Replace default tabletop objects with the 6 scanned objects in a 2x3 grid",
    )
    parser.add_argument(
        "--static", action="store_true",
        help="Disable physics stepping — just view the scene (much faster for checking positions)",
    )
    parser.add_argument(
        "--low-poly", action="store_true",
        help="Use decimated 10k-face meshes for scanned objects (much faster rendering)",
    )
    parser.add_argument(
        "--sine-demo", action="store_true",
        help="Drive the arms with a scripted sine motion through the 14-D action API",
    )
    args = parser.parse_args()

    if args.sine_demo and args.static:
        parser.error("--sine-demo and --static are mutually exclusive")

    print("Building simulation environment ...")
    env = BimanualYamWorkbenchEnv(scanned_objects=args.scanned_objects, low_poly=args.low_poly)
    print(f"  Model: {env.model.nq} qpos, {env.model.nv} qvel, {env.model.nu} actuators")

    # Initialize splat renderer if requested
    splat_renderer = None
    if args.splat:
        try:
            from splat_viewer import GaussianSplatRenderer, make_intrinsics

            print("Loading Gaussian splat renderer ...")
            splat_renderer = GaussianSplatRenderer()
            splat_K = make_intrinsics(args.splat_width, args.splat_height)
            print("  Splat renderer ready.")
        except ImportError as e:
            print(f"  Warning: Could not load splat renderer: {e}")
            print("  Install torch and gsplat for splat rendering.")

    # Launch interactive MuJoCo viewer
    print("Launching viewer ... (close window to exit)")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Set home position
        env.reset()
        viewer.sync()

        if args.static:
            # No physics — just render the scene for inspection
            while viewer.is_running():
                viewer.sync()
                time.sleep(1.0 / 10.0)
        elif args.sine_demo:
            # Drive the 14-D action API with a sine perturbation around home.
            n_steps = max(1, int(1.0 / 60.0 / env.model.opt.timestep))
            home = env.get_state_14d()

            # Shoulder-ish joints: left_joint2, left_joint3, right_joint2, right_joint3
            # in the 14-D layout [left_j1..6, left_grip, right_j1..6, right_grip].
            sine_idx = np.array([1, 2, 8, 9])
            amplitude = 0.3
            freq_hz = 0.25

            print(
                f"[sine-demo] home = {np.round(home, 3).tolist()}"
                f"\n[sine-demo] perturbing indices {sine_idx.tolist()} "
                f"at {freq_hz} Hz, amplitude {amplitude} rad"
            )

            while viewer.is_running():
                step_start = time.time()

                action = home.copy()
                t = env.data.time
                action[sine_idx] += amplitude * np.sin(2.0 * np.pi * freq_hz * t)

                env.step(action, decimation=n_steps)

                viewer.sync()

                dt = n_steps * env.model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)
        else:
            # Step multiple times per viewer frame to maintain real-time with fine timestep
            n_steps = max(1, int(1.0 / 60.0 / env.model.opt.timestep))

            while viewer.is_running():
                step_start = time.time()

                for _ in range(n_steps):
                    env._set_ctrl_raw(env.data.ctrl)

                # Sync viewer
                viewer.sync()

                # Maintain real-time
                dt = n_steps * env.model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)


if __name__ == "__main__":
    main()
