"""Shared teleop utilities: data logging, mjData snapshots, JPEG encoding."""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np


# Fields to save from mjData. qpos alone is sufficient for full replay via
# mj_forward; the rest are kept for convenience.
MJDATA_SAVE_FIELDS = ("qpos", "qvel", "ctrl", "xpos", "xquat")


def snapshot_mjdata(mjd) -> dict[str, np.ndarray]:
    """Capture essential fields from mjData (~8 MB/episode vs ~1 GB for all)."""
    return {attr: getattr(mjd, attr).copy() for attr in MJDATA_SAVE_FIELDS}


class DataLogger:
    """Records per-timestep data and saves dual H5 files per episode.

    Supports three initialization modes via ``existing_data_policy``:
    - ``"backup"`` (default): move existing trajectory data to a sibling dir.
    - ``"append"``: count existing trajectories and continue numbering.
    - ``"overwrite"``: start from episode 0 regardless of existing data.
    """

    def __init__(self, output_dir: str | Path, *, existing_data_policy: str = "backup"):
        self.output_dir = Path(output_dir)
        if existing_data_policy == "backup":
            self._backup_existing_data()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if existing_data_policy == "append":
            self._episode_count = self._count_existing_trajectories()
        else:
            self._episode_count = 0
        self._gello_buf: list[np.ndarray] = []
        self._yam_buf: list[np.ndarray] = []
        self._ts_buf: list[float] = []
        self._mjstate_buf: list[dict[str, np.ndarray]] = []

    # -- Initialization helpers ------------------------------------------------

    def _backup_existing_data(self):
        """If output_dir has trajectory data, move it to a sibling _copy_N dir."""
        if not self.output_dir.exists():
            return
        has_trajectories = any(self.output_dir.glob("trajectory_*"))
        if not has_trajectories:
            return
        copy_idx = 0
        while True:
            backup_path = self.output_dir.parent / f"{self.output_dir.name}_copy_{copy_idx}"
            if not backup_path.exists():
                break
            copy_idx += 1
        shutil.move(str(self.output_dir), str(backup_path))
        print(f"[BACKUP] Moved {self.output_dir} → {backup_path}")

    def _count_existing_trajectories(self) -> int:
        """Count sequential trajectory_NNNN dirs to resume numbering."""
        count = 0
        while (self.output_dir / f"trajectory_{count:04d}").exists():
            count += 1
        return count

    # -- Recording API ---------------------------------------------------------

    def start_episode(self):
        """Clear buffers for a new episode."""
        self._gello_buf.clear()
        self._yam_buf.clear()
        self._ts_buf.clear()
        self._mjstate_buf.clear()

    def log_step(
        self,
        gello_position: np.ndarray,
        yam_position: np.ndarray,
        timestamp: float,
        mjdata_snapshot: dict[str, np.ndarray],
    ):
        """Append one timestep to buffers."""
        self._gello_buf.append(gello_position.copy())
        self._yam_buf.append(yam_position.copy())
        self._ts_buf.append(timestamp)
        self._mjstate_buf.append(mjdata_snapshot)

    def save_episode(self) -> Path:
        """Write both H5 files and return the trajectory directory path."""
        traj_dir = self.output_dir / f"trajectory_{self._episode_count:04d}"
        traj_dir.mkdir(parents=True, exist_ok=True)

        gello_arr = np.array(self._gello_buf, dtype=np.float32)
        yam_arr = np.array(self._yam_buf, dtype=np.float32)
        ts_arr = np.array(self._ts_buf, dtype=np.float64)

        # joints.h5 — training-compatible
        with h5py.File(traj_dir / "joints.h5", "w") as f:
            f.create_dataset("gello_position", data=gello_arr)
            f.create_dataset("yam_position", data=yam_arr)
            f.create_dataset("timestamps", data=ts_arr)

        # mujoco_state.h5 — exhaustive mjData snapshot
        with h5py.File(traj_dir / "mujoco_state.h5", "w") as f:
            f.create_dataset("gello_position", data=gello_arr)
            if self._mjstate_buf:
                for key in sorted(self._mjstate_buf[0].keys()):
                    try:
                        f.create_dataset(key, data=np.array([s[key] for s in self._mjstate_buf]))
                    except (ValueError, KeyError):
                        pass

        self._episode_count += 1
        self.start_episode()
        print(f"[SAVED] {traj_dir} ({len(gello_arr)} steps)")
        return traj_dir

    def delete_last_episode(self) -> Path | None:
        """Delete the last saved trajectory and decrement episode count."""
        if self._episode_count == 0:
            return None
        self._episode_count -= 1
        traj_dir = self.output_dir / f"trajectory_{self._episode_count:04d}"
        if traj_dir.exists():
            shutil.rmtree(traj_dir)
        return traj_dir

    def discard_episode(self):
        """Clear buffers without saving."""
        self.start_episode()
        print("[DISCARDED] Episode recording discarded.")

    @property
    def has_data(self) -> bool:
        return len(self._gello_buf) > 0


# -- JPEG encoding with nvjpeg/turbojpeg/cv2 fallback -------------------------

try:
    import nvjpeg as _nvjpeg

    _nv_encoder = _nvjpeg.NvJpeg()
    print("[JPEG] Using nvJPEG (GPU)")

    import cv2 as _cv2
    _nv_bgr_buf = [None]  # mutable holder for pre-allocated BGR buffer

    def jpeg_encode(rgb: np.ndarray, quality: int = 75) -> bytes:
        if _nv_bgr_buf[0] is None or _nv_bgr_buf[0].shape != rgb.shape:
            _nv_bgr_buf[0] = np.empty_like(rgb)
        _cv2.cvtColor(rgb, _cv2.COLOR_RGB2BGR, dst=_nv_bgr_buf[0])
        return bytes(_nv_encoder.encode(_nv_bgr_buf[0], quality))

except Exception:
    try:
        from turbojpeg import TurboJPEG, TJPF_RGB

        _tj = TurboJPEG()

        def jpeg_encode(rgb: np.ndarray, quality: int = 75) -> bytes:
            return _tj.encode(rgb, pixel_format=TJPF_RGB, quality=quality)

    except Exception:
        import cv2

        def jpeg_encode(rgb: np.ndarray, quality: int = 75) -> bytes:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buf.tobytes()


# -- H.264 I-frame encoding with PyAV + nvenc ---------------------------------

class H264Encoder:
    """Encodes RGB frames as standalone H.264 IDR I-frames using NVENC."""

    def __init__(self, width: int, height: int, qp: int = 23):
        import av
        import cv2
        self._av = av
        self._cv2 = cv2
        self.width = width
        self.height = height
        self.codec_ctx = av.CodecContext.create("h264_nvenc", "w")
        self.codec_ctx.width = width
        self.codec_ctx.height = height
        self.codec_ctx.pix_fmt = "yuv420p"
        self.codec_ctx.gop_size = 1
        self.codec_ctx.max_b_frames = 0
        self.codec_ctx.options = {
            "forced-idr": "1",
            "preset": "p1",
            "tune": "ull",
            "rc": "constqp",
            "qp": str(qp),
            "delay": "0",
        }
        self.codec_ctx.open()
        # Pre-allocate YUV buffer for cv2 colorspace conversion (SIMD-fast)
        self._yuv_buf = np.empty((height * 3 // 2, width), dtype=np.uint8)
        # Warm up with a dummy frame to avoid first-frame latency
        dummy = av.VideoFrame(width, height, "yuv420p")
        self.codec_ctx.encode(dummy)
        print(f"[H264] NVENC encoder ready: {width}x{height} qp={qp}")

    def encode(self, rgb: np.ndarray) -> bytes:
        self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2YUV_I420, dst=self._yuv_buf)
        frame = self._av.VideoFrame.from_ndarray(self._yuv_buf, format="yuv420p")
        packets = self.codec_ctx.encode(frame)
        if not packets:
            return b""
        return bytes(packets[0])


_h264_encoder: H264Encoder | None = None


def h264_encode(rgb: np.ndarray, qp: int = 23) -> bytes:
    global _h264_encoder
    h, w = rgb.shape[:2]
    if _h264_encoder is None or _h264_encoder.width != w or _h264_encoder.height != h:
        _h264_encoder = H264Encoder(w, h, qp)
    return _h264_encoder.encode(rgb)
