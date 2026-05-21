from typing import Optional

import numpy as np
import torch


def sample_lorenz_initial_state(rng=None):
    """Sample an initial state from the repository's Lorenz initialization box."""
    uniform = np.random.uniform if rng is None else rng.uniform
    return np.array(
        [
            uniform(-20, 20),
            uniform(-20, 20),
            uniform(0, 50),
        ],
        dtype=np.float64,
    )


def apply_normalization(dataset, normalization_stats: Optional[dict]):
    """Apply z-score normalization to a dataset object with trajectories."""
    if dataset.normalize:
        if normalization_stats is None:
            dataset.mean = dataset.trajectories.mean(dim=(0, 1))
            dataset.std = dataset.trajectories.std(dim=(0, 1))
        else:
            dataset.mean = torch.as_tensor(
                normalization_stats["mean"], dtype=torch.float32
            )
            dataset.std = torch.as_tensor(
                normalization_stats["std"], dtype=torch.float32
            )
        dataset.trajectories = (dataset.trajectories - dataset.mean) / (
            dataset.std + 1e-8
        )
    else:
        dataset.mean = None
        dataset.std = None


def normalization_stats(dataset):
    if dataset.mean is None or dataset.std is None:
        return None
    return {
        "mean": dataset.mean.detach().cpu().tolist(),
        "std": dataset.std.detach().cpu().tolist(),
    }
