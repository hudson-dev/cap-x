"""Shared W&B logging utilities for MuJoCo evaluation.

Used by both eval_launcher.py (standalone eval) and ppo_robot.py (inline
training eval). All functions import wandb lazily to avoid import-time errors
when wandb is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path


def merge_rank_results(eval_dir: Path, world_size: int) -> dict:
    """Merge per-rank result files into a single results.json.

    Sorts episodes by global index, aggregates success rate and per-stage rates.
    Returns the merged results dict.
    """
    all_episodes = []
    base_meta = None
    missing_ranks = []

    for rank in range(world_size):
        rank_path = eval_dir / f"results_rank{rank}.json"
        if not rank_path.exists():
            missing_ranks.append(rank)
            continue
        with open(rank_path) as f:
            data = json.load(f)
        if base_meta is None:
            base_meta = {k: v for k, v in data.items() if k != "episodes"}
        all_episodes.extend(data["episodes"])

    # Sort by global episode index
    all_episodes.sort(key=lambda e: e["episode"])

    successes = [e["success"] for e in all_episodes]
    merged = base_meta or {}
    merged["success_rate"] = sum(successes) / len(successes) if successes else 0.0

    # Aggregate per-stage rates from episode data (keys like "stage_pick_yellow")
    # Exclude non-bool stage keys (e.g. stage_first_steps is a dict)
    first_ep = all_episodes[0] if all_episodes else {}
    stage_keys = [k for k in first_ep if k.startswith("stage_") and isinstance(first_ep[k], bool)]
    if stage_keys:
        merged["stage_rates"] = {
            k.removeprefix("stage_"): sum(e.get(k, False) for e in all_episodes) / len(all_episodes)
            for k in stage_keys
        }
    merged["episodes"] = all_episodes
    merged["rank"] = "merged"
    merged["world_size"] = world_size
    if missing_ranks:
        merged["missing_ranks"] = missing_ranks

    merged_path = eval_dir / "results.json"
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)

    # Clean up rank files
    for rank in range(world_size):
        rank_path = eval_dir / f"results_rank{rank}.json"
        if rank_path.exists():
            rank_path.unlink()

    return merged


def wandb_upload_episode(eval_dir: Path, ep_entry: dict, prefix: str = "episode"):
    """Upload a single episode's metrics and video to the active wandb run.

    Reads whatever keys are in the episode dict and forwards numeric ones
    as {prefix}/* metrics. Attaches the video if it exists on disk.
    The ``prefix`` param allows namespacing (e.g. "episode" or "eval_episode").
    """
    import wandb

    ep_idx = ep_entry["episode"]
    success = ep_entry.get("success", False)

    ep_log = {
        f"{prefix}/index": ep_idx,
        f"{prefix}/success": int(success),
        f"{prefix}/steps": ep_entry.get("steps", 0),
        f"{prefix}/sim_time": ep_entry.get("sim_time", 0.0),
    }

    # Forward any extra keys from episode results (future-proofing).
    # Convert bools to ints so wandb treats them as numeric chart metrics.
    for k, v in ep_entry.items():
        if k not in ("episode", "config", "success", "steps", "sim_time"):
            if isinstance(v, bool):
                ep_log[f"{prefix}/{k}"] = int(v)
            elif isinstance(v, (int, float)):
                ep_log[f"{prefix}/{k}"] = v

    # Attach video if it exists (prefer small version for wandb)
    small_video = eval_dir / f"episode_{ep_idx:03d}_small.mp4"
    full_video = eval_dir / f"episode_{ep_idx:03d}.mp4"
    video_path = small_video if small_video.exists() else full_video
    if video_path.exists():
        caption = f"ep{ep_idx} {'OK' if success else 'FAIL'} ({ep_entry.get('steps', '?')}steps)"
        ep_log[f"{prefix}/video"] = wandb.Video(str(video_path), caption=caption, format="mp4")
    else:
        # Debug: list mp4 files in eval_dir to diagnose missing videos
        import os
        mp4s = sorted(f for f in os.listdir(eval_dir) if f.endswith(".mp4")) if eval_dir.is_dir() else []
        print(f"  [video] ep{ep_idx}: NOT FOUND at {small_video} or {full_video}  (mp4s in dir: {mp4s[:5]}{'...' if len(mp4s) > 5 else ''})")

    # Value trajectory plot (matplotlib figure with stage transition lines)
    if "value_trajectory" in ep_entry:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        values = ep_entry["value_trajectory"]
        stage_steps = ep_entry.get("stage_first_steps", {})

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(range(len(values)), values, linewidth=0.8, color="steelblue")
        for stage_name, step_val in stage_steps.items():
            ax.axvline(x=step_val, linestyle="--", alpha=0.7, label=stage_name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_title(f"Episode {ep_idx}")
        if stage_steps:
            ax.legend(fontsize=8)
        fig.tight_layout()
        plot_path = eval_dir / f"episode_{ep_idx:03d}_value.png"
        fig.savefig(plot_path, dpi=100)
        ep_log[f"{prefix}/value_plot"] = wandb.Image(fig)
        plt.close(fig)

    wandb.log(ep_log)


def wandb_log_eval_summary(merged: dict, step: int | None = None, prefix: str = ""):
    """Log aggregate eval metrics to wandb summary (and optionally as a step-indexed log).

    When ``step`` is provided, also calls ``wandb.log()`` so the metrics appear
    as a time-series point (useful for inline training eval curves).

    When ``prefix`` is set (e.g. "eval_0/tape_handover"), all logged keys are
    namespaced under it, and summary writes are skipped (appropriate for inline
    training evals that log multiple tasks under one training run).
    """
    import wandb
    import numpy as np

    rate = merged.get("success_rate", 0.0)
    episodes = merged.get("episodes", [])
    n_eps = len(episodes)

    # Aggregate scalar metrics across all episodes
    metric_keys = [
        "chunk_overlap_mean", "eye_angular_vel_mean", "joint_accel_mean",
        "value_mean", "ee_idle_fraction",
    ]
    # Auto-discover gaze_* keys from first episode (task-dependent target names)
    if episodes:
        metric_keys += [k for k in episodes[0] if k.startswith("gaze_") and isinstance(episodes[0][k], (int, float))]
    aggregated = {}
    for key in metric_keys:
        vals = [ep.get(key) for ep in episodes if ep.get(key) is not None]
        if vals:
            aggregated[f"mean_{key}"] = float(np.mean(vals))

    def _prefixed(k: str) -> str:
        return f"{prefix}/{k}" if prefix else k

    # Summary writes (skipped when prefix is set — inline evals shouldn't
    # overwrite top-level summary with per-task values).
    if not prefix:
        wandb.summary["success_rate"] = rate
        wandb.summary["n_episodes"] = n_eps
        wandb.summary["n_successes"] = int(round(rate * n_eps)) if n_eps else 0
        for stage, stage_rate in merged.get("stage_rates", {}).items():
            wandb.summary[f"stage_rate/{stage}"] = stage_rate
        for k, v in aggregated.items():
            wandb.summary[k] = v

    # Optionally log as a time-series point (for training eval curves)
    if step is not None or prefix:
        log_dict = {}
        if step is not None:
            log_dict["global_step"] = step
        log_dict[_prefixed("success_rate")] = rate
        for stage, stage_rate in merged.get("stage_rates", {}).items():
            log_dict[_prefixed(f"stage_{stage}")] = stage_rate
        for k, v in aggregated.items():
            log_dict[_prefixed(k)] = v
        wandb.log(log_dict)


def wandb_log_all_episodes(eval_dir: Path, merged: dict, prefix: str = "episode"):
    """Upload all episodes from merged results + log summary.

    Convenience wrapper: iterates over merged["episodes"], uploads each with
    ``wandb_upload_episode``, then logs the aggregate summary.
    """
    for ep in merged.get("episodes", []):
        wandb_upload_episode(eval_dir, ep, prefix=prefix)
    wandb_log_eval_summary(merged)
