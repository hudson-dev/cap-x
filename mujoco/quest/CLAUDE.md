# Quest Pro VR Teleoperation Guide

Teleoperate the bimanual MuJoCo simulation using a Meta Quest Pro headset. The Quest's stereo displays show the robot's eye-camera view, and the 6DOF controllers drive the two arms. No native APK build required — runs entirely in the Quest browser via WebXR.

## Architecture Overview

```
┌─────────────────────────────────────┐        USB (ADB reverse)        ┌──────────────────────────┐
│         Python Server (Mac)         │◄──────── WebSocket ────────────►│  Quest Pro Browser       │
│                                     │         port 8080               │  (WebXR app)             │
│  Physics Thread (200 Hz)            │                                 │                          │
│    read joint targets → mj_step()   │  ── stereo JPEG frames ──►     │  Display on L/R eyes     │
│    write proprio state              │                                 │                          │
│                                     │  ◄── controller poses ──       │  Read 6DOF controllers   │
│  Main Thread (asyncio, ~40 Hz)      │      + buttons (72 Hz)         │  Read head tracking      │
│    aiohttp: serve HTML + WebSocket  │                                 │                          │
│    receive controller data          │  ── mode/status JSON ──►       │  Show status overlay     │
│    IK solve (PyRoKi) → joint targets│                                 │                          │
│    render stereo → JPEG encode      │                                 │                          │
└─────────────────────────────────────┘                                 └──────────────────────────┘
```

## 1. Install Dependencies

### System packages (macOS)

```bash
# libturbojpeg for fast JPEG encoding (~5x faster than OpenCV)
brew install jpeg-turbo

# Android Debug Bridge for USB connection to Quest
brew install android-platform-tools
```

### Python packages

From within your `eyeball` conda environment:

```bash
conda activate eyeball

# aiohttp — async web server + WebSocket
pip install aiohttp

# pynvjpeg — GPU-accelerated JPEG encoding via nvJPEG (recommended, ~8x faster)
# Requires CUDA toolkit. Falls back to PyTurboJPEG → cv2 if not available.
pip install pynvjpeg

# PyTurboJPEG — CPU JPEG encoding fallback
# Note: PyTurboJPEG 2.x requires libjpeg-turbo 3.0+; use 'PyTurboJPEG<2' with older system libs
pip install 'PyTurboJPEG<2'
```

All other dependencies (`mujoco`, `numpy`, `jax`, `pyroki`, `h5py`, etc.) are already installed via `pip install -e .` from the main project setup.

### Verify installation

```bash
python -c "
import aiohttp; print(f'aiohttp {aiohttp.__version__}')
import mujoco; print(f'mujoco {mujoco.__version__}')
try:
    import nvjpeg; nvjpeg.NvJpeg()
    print('nvjpeg OK (GPU)')
except Exception:
    try:
        from turbojpeg import TurboJPEG; TurboJPEG()
        print('turbojpeg OK (CPU)')
    except Exception as e:
        print(f'no fast JPEG encoder (will use cv2): {e}')
"
```

## 2. Quest Pro Setup

### Enable Developer Mode

1. Install the **Meta Quest Developer Hub** (MQDH) on your Mac: https://developer.oculus.com/meta-quest-developer-hub/
2. Create a developer organization at https://developer.oculus.com/manage/ if you haven't already
3. In the Quest headset: **Settings → System → Developer → Developer Mode → ON**
4. Restart the headset

### USB Connection

1. Connect Quest Pro to Mac via USB-C cable
2. Put on the headset — accept the "Allow USB debugging" prompt
3. Verify the connection:

```bash
adb devices
# Should show something like:
# 1WMHH815P10234    device
```

If you see `unauthorized`, put the headset back on and accept the debugging prompt.

### ADB Port Forwarding

This makes the Quest's `localhost:8080` forward to your Mac's port 8080 over USB — lower latency and more reliable than WiFi:

```bash
adb reverse tcp:8080 tcp:8080
```

You need to re-run this each time you reconnect the USB cable.

## 3. Running the Server

### Basic launch

```bash
conda activate eyeball
python scripts/mujoco_quest_teleop.py --task tape_handover
```

First launch takes ~30 seconds for IK solver JIT compilation. Subsequent launches within the same session are faster.

### Full options

```bash
python scripts/mujoco_quest_teleop.py \
    --task tape_handover \       # Scene: "tape_handover" or "none"
    --port 8080 \                # Server port (match ADB reverse)
    --seed 0 \                   # Initial random seed for object placement
    --control-rate 200 \         # Physics thread Hz
    --render-fps 40 \            # Stereo frame rate
    --jpeg-quality 75 \          # JPEG quality (lower = faster, lower quality)
    --translation-scale 1.5 \    # VR movement → sim movement multiplier
    --output-dir data/demos/mujoco_quest_teleop  # Recording save dir
```

### Head-only mode (no arm control)

Useful for testing stereo view + head tracking without IK overhead:

```bash
python scripts/mujoco_quest_teleop.py --no-ik
```

### No recording

```bash
python scripts/mujoco_quest_teleop.py --no-record
```

### Expected startup output

```
[Init] Creating MuJoCo sim with stereo cameras...
[Init] Warming up IK solver (JIT compile, ~30s first time)...
[IK] Compiled bimanual IK solver for (left_tool0, right_tool0)
[IK] Solver warmed up and ready
[Init] Recording enabled → data/demos/mujoco_quest_teleop
[Init] Physics thread started at 200 Hz

[Ready] Server at http://0.0.0.0:8080
  ADB: adb reverse tcp:8080 tcp:8080
  Quest browser: http://localhost:8080
  Mode: full control
```

## 4. Connecting from Quest Pro

1. Make sure ADB reverse is set up: `adb reverse tcp:8080 tcp:8080`
2. On Quest Pro, open the **Meta Quest Browser**
3. Navigate to `http://localhost:8080`
4. You should see the "Enter VR" button with an overlay showing "Connected"
5. Click **Enter VR** — the browser enters immersive VR mode
6. You should see the robot simulation in stereo 3D

### Flat-screen preview (no headset needed)

For testing without a Quest, open `http://localhost:8080` in any desktop browser and click "Enter VR". If WebXR isn't available, it falls back to a flat side-by-side stereo preview.

## 5. Controls

### Arm Control (Relative/Clutch Mode — default)

| Input | Action |
|-------|--------|
| **Grip button** (per hand) | **Clutch**: hold to control arm, release to freeze |
| **Trigger** (analog) | **Gripper**: squeeze to close, release to open |

How clutch mode works:
1. Hold the grip button — saves your current controller pose as reference
2. Move the controller while holding grip — the robot arm follows the delta
3. Release grip — the arm freezes in place
4. Reposition your controller freely, then grip again to continue

Each arm is independent: left grip controls left arm, right grip controls right arm.

### Arm Control (Direct Mode)

Press **Y** to switch to direct mapping mode. Controller position maps directly to EE position (with a calibration offset computed on mode switch).

### Head Tracking

Look around in VR — the robot's eye camera follows your head orientation. The stereo cameras are mounted on the eye gimbal, so you see what the robot would see.

### Buttons

| Button | Action |
|--------|--------|
| **Y** (left controller) | Toggle relative/direct control mode |
| **A** (right controller) | Reset head position/orientation |
| **B** (right controller) | Discard current recording |
| **X** (left controller) | Reset scene (new random seed) |

### Head Recentering

Press **A** to recenter the VR viewpoint. Your current physical position and orientation become the new "center" — the MuJoCo camera snaps to the default viewpoint. Subsequent head movements are tracked relative to this new reference. Useful when you've physically moved in your room and the sim view has drifted.

### Recording Workflow

Recording is controlled via the **Y** button cycle (relative → direct+record → home+save → reset). Press **B** to discard instead of saving. Press **X** to reset the scene (auto-saves if recording).

Each saved trajectory contains:
- `joints.h5` — joint positions (training-compatible format)
- `mujoco_state.h5` — full MuJoCo state snapshots

## 6. Tuning & Troubleshooting

### Performance tuning

| Parameter | Effect | Tradeoff |
|-----------|--------|----------|
| `--render-fps 40` | Frame rate sent to Quest | Higher = smoother but more CPU/bandwidth |
| `--jpeg-quality 75` | JPEG compression level | Lower = faster encode + less bandwidth, worse quality |
| `--translation-scale 1.5` | Movement multiplier | Higher = bigger robot movements per hand movement |

For lower latency at the cost of visual quality:
```bash
python scripts/mujoco_quest_teleop.py --render-fps 30 --jpeg-quality 50
```

### Connection issues

**"Disconnected" in overlay:**
- Check USB cable is connected
- Re-run `adb reverse tcp:8080 tcp:8080`
- Check server is running and showing "Ready"

**ADB device not found:**
```bash
adb kill-server
adb start-server
adb devices
```

**Quest shows blank/black in VR:**
- Check the flat-screen preview first (`http://localhost:8080` in desktop browser)
- If flat preview works but VR doesn't, the WebXR session may need HTTPS — try WiFi mode instead (see below)

### WiFi mode (no USB)

If USB isn't available, you can use WiFi instead (higher latency, ~5-10ms more):

1. Find your Mac's local IP: `ipconfig getifaddr en0`
2. Make sure Quest is on the same WiFi network
3. Start server: `python scripts/mujoco_quest_teleop.py --host 0.0.0.0`
4. On Quest browser, navigate to `http://<mac-ip>:8080`

Note: WebXR requires a "secure context" (HTTPS or localhost). Over WiFi with plain HTTP, the Quest browser may block WebXR. Workarounds:
- Use Chrome flags: navigate to `chrome://flags` on Quest browser, enable "Insecure origins treated as secure", add `http://<mac-ip>:8080`
- Or use ADB wireless: `adb tcpip 5555 && adb connect <quest-ip>:5555 && adb reverse tcp:8080 tcp:8080`

### IK failures

If arms jitter or don't move correctly:
- Check the terminal for `[IK] Failed` messages
- Try increasing `--translation-scale` if movements seem too small
- The IK solver occasionally fails for extreme poses — the controller falls back to holding current position

### Stereo view looks wrong

- **No parallax**: Check that both cameras were added (look for cam IDs in startup logs)
- **Reversed eyes**: The WebXR layer maps left/right views to left/right eye viewports automatically — if reversed, check that your IPD is set correctly in Quest settings

## 7. File Reference

| File | Purpose |
|------|---------|
| `scripts/mujoco_quest_teleop.py` | Entry point — aiohttp server, physics thread, control loop |
| `eye/mujoco/quest/quest_teleop.py` | `QuestTeleopSim` class — stereo cameras, rendering, IK, shared state |
| `eye/mujoco/quest/quest_client/index.html` | WebXR app served to Quest browser (self-contained, no build step) |

## 8. Quick-Start Checklist

```bash
# 1. Install deps (one-time)
brew install jpeg-turbo android-platform-tools
conda activate eyeball
pip install aiohttp PyTurboJPEG

# 2. Connect Quest via USB, accept debug prompt in headset

# 3. Set up port forwarding
adb reverse tcp:8080 tcp:8080

# 4. Start server
python scripts/mujoco_quest_teleop.py --task tape_handover

# 5. On Quest: open browser → http://localhost:8080 → Enter VR

# 6. Grip to control arms, trigger for gripper, look around freely
```
