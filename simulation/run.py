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
    args = parser.parse_args()

    print("Building simulation environment ...")
    env = BimanualYamWorkbenchEnv()
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

        # Step multiple times per viewer frame to maintain real-time with fine timestep
        n_steps = max(1, int(1.0 / 60.0 / env.model.opt.timestep))

        while viewer.is_running():
            step_start = time.time()

            for _ in range(n_steps):
                env.step(env.data.ctrl)

            # Sync viewer
            viewer.sync()

            # Maintain real-time
            dt = n_steps * env.model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
