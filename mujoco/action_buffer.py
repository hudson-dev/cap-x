"""Temporal ensembling of overlapping action chunks.

Ported from robot_inference_yam.py RobotController.get_interpolated_action().
Blends overlapping action chunks with exponential weighting and linear
interpolation between adjacent timesteps for sub-frame smoothness.

Weighting follows the ACT/LeRobot convention: w_i = exp(-k * i) where i=0
is the oldest prediction. k>0 weights older predictions more (ACT default),
k<0 weights newer predictions more, k=0 is uniform averaging.
"""

import math

import numpy as np


class ActionBuffer:
    """Temporal ensembling of overlapping action chunks.

    Each call to add_chunk() stores an action trajectory with its timestamp.
    get_action() blends all active chunks at the current time using exponential
    weighting w_i = exp(-k * i) where i=0 is the oldest chunk.

    Args:
        chunk_size: Number of timesteps per action chunk.
        action_dim: Dimensionality of each action vector (14 for bimanual).
        agent_fps: Agent inference frequency in Hz (determines chunk timing).
        k: Ensemble weighting parameter. k>0 = older predictions weighted more
            (ACT-like), k=0 = uniform, k<0 = newer weighted more.
    """

    def __init__(
        self,
        chunk_size: int,
        action_dim: int,
        agent_fps: float,
        k: float = 0.01,
    ):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.agent_fps = agent_fps
        self.k = k
        self.action_chunks: list[np.ndarray] = []
        self.initial_timestamps: list[float] = []

    def add_chunk(self, chunk: np.ndarray, timestamp: float):
        """Add a new action chunk.

        Args:
            chunk: (chunk_size, action_dim) array of joint targets.
            timestamp: Simulation time when this chunk was generated.
        """
        self.action_chunks.append(chunk)
        self.initial_timestamps.append(timestamp)

    def get_action(self, current_time: float) -> np.ndarray | None:
        """Get temporally-blended action for the current time.

        Returns:
            (action_dim,) array of blended joint targets, or None if no
            active chunks are available.
        """
        if len(self.action_chunks) == 0:
            return None

        # Expire old chunks whose time index exceeds chunk_size
        while len(self.action_chunks) > 0:
            time_delta = current_time - self.initial_timestamps[0]
            index = int(time_delta * self.agent_fps)
            if index >= self.action_chunks[0].shape[0]:
                self.action_chunks.pop(0)
                self.initial_timestamps.pop(0)
            else:
                break

        if len(self.action_chunks) == 0:
            return None

        # Blend all active chunks with exponential weighting
        num_chunks = len(self.action_chunks)
        total_w = 0.0
        avg_action = np.zeros(self.action_dim)
        debug_weights = []  # for breakpoint inspection

        for i in range(num_chunks):
            time_delta = current_time - self.initial_timestamps[i]
            index = int(time_delta * self.agent_fps)
            index = min(index, self.action_chunks[i].shape[0] - 2)

            # Linear interpolation between adjacent timesteps
            alpha = (time_delta * self.agent_fps) - index
            alpha = max(0.0, min(1.0, alpha))
            state_1 = self.action_chunks[i][index]
            state_2 = self.action_chunks[i][index + 1]
            interpolated = state_1 * (1 - alpha) + state_2 * alpha

            # Exponential weighting: w_i = exp(-k * i), i=0 oldest
            w = math.exp(-self.k * i)
            total_w += w
            avg_action += interpolated[:self.action_dim] * w
            debug_weights.append((i, index, w))


        avg_action /= total_w
        return avg_action

    def reset(self):
        """Clear all chunks (new episode)."""
        self.action_chunks.clear()
        self.initial_timestamps.clear()
