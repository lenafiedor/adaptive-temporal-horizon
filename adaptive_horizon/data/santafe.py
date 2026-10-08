"""Measured Santa Fe laser data and offline neighbour-divergence scores."""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

SANTAFE = "santafe-laser"


def neighbour_divergence(
    states,
    reference,
    query_start=0,
    window=5,
    neighbours=5,
    theiler_window=50,
    epsilon=1e-8,
):
    """Estimate finite-window growth using complete training continuations only."""
    candidates = reference[:-window]
    if len(candidates) < neighbours or len(states) <= window:
        raise ValueError("Recording is too short for neighbour divergence")
    scores = np.empty(len(states) - window, dtype=np.float64)
    reference_indices = np.arange(len(candidates))
    for start in range(0, len(scores), 64):
        stop = min(start + 64, len(scores))
        query_indices = np.arange(start, stop)
        distances = np.linalg.norm(
            states[start:stop, None, :] - candidates[None, :, :], axis=2
        )
        excluded = (
            np.abs(query_indices[:, None] + query_start - reference_indices)
            <= theiler_window
        ) | (distances == 0)
        distances[excluded] = np.inf
        if np.any(np.isfinite(distances).sum(axis=1) < neighbours):
            raise ValueError(
                "Not enough distinct neighbours outside the Theiler window"
            )
        indices = np.argpartition(distances, neighbours - 1, axis=1)[:, :neighbours]
        initial = np.take_along_axis(distances, indices, axis=1)
        future = np.linalg.norm(
            states[query_indices + window, None, :] - reference[indices + window],
            axis=2,
        )
        scores[start:stop] = (
            np.log((future + epsilon) / (initial + epsilon)).mean(axis=1) / window
        )
    return scores


@lru_cache(maxsize=2)
def _load_santafe(
    path,
    modified,
    size,
    history_length,
    window,
    neighbours,
    theiler_window,
    epsilon,
):
    # File identity is part of the cache key so replacement never reuses old data.
    raw = np.loadtxt(path, ndmin=2)
    if raw.shape[1] != 1 or not np.isfinite(raw).all():
        raise ValueError("Santa Fe data must contain one finite intensity per sample")
    raw = raw[:, 0]
    train_end, val_end = int(0.7 * len(raw)), int(0.85 * len(raw))
    bounds = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, len(raw)),
    }
    for split, (start, end) in bounds.items():
        if end - start <= history_length:
            raise ValueError(f"Santa Fe {split} split is too short for history windows")
    mean, std = float(raw[:train_end].mean()), float(raw[:train_end].std(ddof=1))
    mean, std = float(np.float32(mean)), float(np.float32(std))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Santa Fe training data must have nonzero finite variance")
    trajectories = {}
    for split, (start, end) in bounds.items():
        states = np.lib.stride_tricks.sliding_window_view(
            raw[start:end], history_length
        ).copy()
        trajectories[split] = torch.tensor(states, dtype=torch.float32)
    normalization = {"mean": [mean] * history_length, "std": [std] * history_length}
    metadata = {
        "path": path,
        "checksum": sha256(Path(path).read_bytes()).hexdigest(),
        "samples": len(raw),
        "split_bounds": bounds,
        "history_length": history_length,
        "normalization_stats": normalization,
        "sample_time": "one recorded sample",
        "estimator": {
            "name": "neighbour-divergence",
            "window": window,
            "neighbours": neighbours,
            "theiler_window": theiler_window,
            "epsilon": epsilon,
        },
    }
    return {"trajectories": trajectories, "metadata": metadata, "scores": {}}


def load_santafe(data_path, max_horizon=10, compute_scores=False, metadata=None):
    """Load once per file/settings and validate checkpoint identity if supplied."""
    if data_path is None:
        raise ValueError("Santa Fe requires --data-path pointing to a local recording")
    if max_horizon < 1:
        raise ValueError("Santa Fe prediction horizon must be positive")
    path = Path(data_path).expanduser().resolve()
    identity = path.stat()
    metadata = metadata or {}
    estimator = metadata.get("estimator", {})
    data = _load_santafe(
        str(path),
        identity.st_mtime_ns,
        identity.st_size,
        metadata.get("history_length", 8),
        estimator.get("window", 5),
        estimator.get("neighbours", 5),
        estimator.get("theiler_window", 50),
        estimator.get("epsilon", 1e-8),
    )
    for split, trajectory in data["trajectories"].items():
        if len(trajectory) <= max_horizon:
            raise ValueError(
                f"Santa Fe {split} split is too short for horizon {max_horizon}"
            )
    for name in (
        "checksum",
        "split_bounds",
        "normalization_stats",
        "samples",
        "sample_time",
    ):
        if name in metadata and metadata[name] != data["metadata"][name]:
            raise ValueError(f"Santa Fe {name} does not match checkpoint metadata")
    if compute_scores and not data["scores"]:
        settings = data["metadata"]["estimator"]
        stats = data["metadata"]["normalization_stats"]
        normalized = {
            split: (states.numpy().astype(np.float64) - stats["mean"][0])
            / (stats["std"][0] + 1e-8)
            for split, states in data["trajectories"].items()
            if split != "test"
        }
        scores = {
            split: neighbour_divergence(
                normalized[split],
                normalized["train"],
                data["metadata"]["split_bounds"][split][0],
                window=settings["window"],
                neighbours=settings["neighbours"],
                theiler_window=settings["theiler_window"],
                epsilon=settings["epsilon"],
            )
            for split in ("train", "val")
        }
        data["scores"] = scores
        data["score_stats"] = {
            "mean": float(scores["train"].mean()),
            "std": float(scores["train"].std()),
        }
        data["metadata"]["score_stats"] = data["score_stats"]
    return data


def observed_trajectory(dataset, data, split):
    """Attach an already split observed trajectory to an existing dataset."""
    dataset.trajectory_path = data["metadata"]["path"]
    dataset.split_bounds = data["metadata"]["split_bounds"][split]
    return data["trajectories"][split]
