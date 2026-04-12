# eye/mujoco — MuJoCo Simulation Environment

## Overview

MuJoCo-based simulation for bimanual robot manipulation. Originally built for **policy evaluation** (running trained checkpoints in sim), but the real-to-sim visual gap was too high for reliable eval. Now being extended for **teleop data collection** to investigate training dynamics, reward shaping, and action representations without needing the physical robot.

## Architecture

### Scene Pipeline (MjSpec)
1. Load base XML (`robot_xmls/eyeball_bimanual_scene.xml`) — robot + table + eye camera
2. Task adds objects via `configure_scene(spec)` — meshes, textures, collision geoms
3. Compile to `MjModel` with all objects
4. Task caches body/joint/site IDs via `setup(mjm, mjd)`

### Mesh & Collision Handling (`decompose_mesh.py`)
- GLB/OBJ models are preprocessed into visual + collision assets
- **Visual**: original mesh + extracted per-material textures (PNG)
- **Collision**: convex decomposition into 25-30 hull pieces for realistic physics (via `coacd`)
- Cached in `assets/<object>/collision_config.json` — run once, reuse forever
- CLI: `python -m eye.mujoco.decompose_mesh <path.glb>`

### Rendering & Fisheye (`mjwarp_utils.py`)
- `MujocoRenderer`: EGL-backed renderer with optional fisheye distortion
- When `fisheye=True`: renders a wider pinhole internally, then applies equidistant fisheye remap (precomputed `cv2.remap` tables) to match the real Insta360 camera
- Calibration constants match the 1920x1200 Insta360 sensor (`FISH_FOCAL`, `FISH_DIST_COEFFS`)
- `render_torch()` returns GPU tensor directly for foveated multicrop pipeline

### Tasks (`tasks.py`)
- Abstract `Task` base class with: `configure_scene`, `setup`, `generate_eval_configs`, `apply_eval_config`, `check_success`, `get_clip_embedding`
- `TapeHandoverTask`: bimanual tape handover with randomized object placement
- Configs are deterministic (seeded) for reproducible evaluation

**Task Stages**: Tasks can define ordered sub-goals for granular progress metrics beyond binary success. Override `stages` (tuple of names) and `check_stages(mjm, mjd)` (single contact scan returning `{name: bool}`). The eval loop tracks "ever achieved" per stage with sticky OR — once a stage is reached, it stays True for the episode. Stage metrics auto-forward to wandb and appear in the video debug overlay.

```python
class MyTask(Task):
    @property
    def stages(self) -> tuple[str, ...]:
        return ("reach", "grasp", "place")

    def check_stages(self, mjm, mjd) -> dict[str, bool]:
        return {"reach": ..., "grasp": ..., "place": ...}

    def check_success(self, mjm, mjd) -> bool:
        return self.check_stages(mjm, mjd)["place"]
```

Default `stages` is `("success",)` wrapping `check_success()`, so single-stage tasks need no changes. `TapeHandoverTask` defines: `pick_yellow` (lifted off table), `handover` (both right fingers touching, no left fingers), `place_yellow` (on grey tape, flat, released).

### Action Buffer (`action_buffer.py`)
- Temporal action ensembling: blends overlapping action chunks with exponential weighting
- Newer chunks weighted higher (decay=0.007), linear interpolation for sub-frame smoothness
- Auto-expires old chunks

## Teleop Vision (WIP)

Two-stage pipeline for collecting simulation demonstrations:
1. **Teleop collection**: Record joint trajectories + eye movements in MuJoCo (rendering live for the operator)
2. **Equirect rendering**: Separate offline script replays trajectories and renders 360° equirectangular video matching the training format

Stage 2 is not yet built — currently investigating the best approach for offline equirect rendering from recorded trajectories.

## Evaluation

### Running Evals
```bash
# Single GPU, direct
python -m eye.mujoco.eval --ckpt runs/my_run/checkpoint_final.pt --n-episodes 10

# Multi-GPU via launcher
python scripts/eval_launcher.py --ckpts runs/my_run/ckpt.pt --nproc 4 \
    -- --n-episodes 50 --task tape_handover --seed 42

# Multiple checkpoints
python scripts/eval_launcher.py --ckpts runs/run_a/ckpt.pt runs/run_b/ckpt.pt --nproc 4 \
    -- --n-episodes 50 --task tape_handover
```

### W&B Logging
All wandb code lives in `scripts/eval_launcher.py` — workers (`eval.py`) never touch wandb.

```bash
# Enable wandb
python scripts/eval_launcher.py --ckpts ckpt.pt --nproc 4 --wandb \
    -- --n-episodes 50 --task tape_handover

# A/B comparison: use --wandb-group to put runs in the same group
python scripts/eval_launcher.py --ckpts runs/run_a/ckpt.pt --wandb --wandb-group "my_comparison" -- ...
python scripts/eval_launcher.py --ckpts runs/run_b/ckpt.pt --wandb --wandb-group "my_comparison" -- ...
```

**How it works**:
- Launcher inits a wandb run per checkpoint (project: `eyeball-eval`)
- Episodes are uploaded **incrementally** as workers complete them (via the poll loop)
- Videos are saved at 480p (`_small.mp4`) for wandb, full-res kept on disk
- Default group = training run name (parent dir of checkpoint)
- `--wandb-group` overrides for controlled A/B comparisons

**Adding new metrics**: Add numeric keys to the episode results dict in `eval.py` — they auto-forward to wandb as `episode/<key>` with no launcher changes needed:
```python
# In eval.py's episode_results and progress jsonl:
episode_results.append({
    ...,
    "max_ee_force": float(max_force),  # → episode/max_ee_force in wandb
})
```

**Flags**: `--wandb-project` (default: `eyeball-eval`), `--wandb-entity`, `--wandb-group`

### Oracle CLIP Toggle
Models trained with `--oracle-clip-toggle` use an `OracleTargetSelector` that switches the CLIP prompt based on gripper state: gripper open → "yellow tape", gripper closed → "grey tape". The oracle selector is saved in the checkpoint and **auto-detected at eval time** — no special flags needed.

```bash
# Oracle is detected from checkpoint automatically
python -m eye.mujoco.eval --ckpt ... --n-episodes 10
```

Note: Pre-refactor oracle checkpoints (EXP-011 through EXP-031) do not contain the `target_selector` key and cannot be correctly evaluated with the current eval.py.

### Eye Settle Time
`--settle-time N` runs N steps of eye-only saccading before the arm starts moving. During settle steps, the eye policy runs normally but the arm holds its initial joint position. This matches training's `time_pause` parameter and gives the eye time to lock onto the target before manipulation begins. Default: 0 (no settle time).

### Eval Video Options
- `--side-view` / `--top-view`: extra camera panels
- `--multicrop-view`: foveal pyramid visualization
- `--no-video`: skip video saving entirely
- `--no-fisheye`: raw pinhole render (no distortion)
