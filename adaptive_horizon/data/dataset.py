import torch
from torch.utils.data import Dataset
from typing import Optional

import adaptive_horizon.config as config
from adaptive_horizon.data.utils import (
    apply_normalization,
    default_lorenz_trajectory_path,
    get_lorenz_trajectory,
    NormalizationStats,
    split_trajectory,
)


class LorenzDataset(NormalizationStats, Dataset):
    """Lorenz dataset sliced from one shared long trajectory."""

    def __init__(
        self,
        T: int = 1,
        dt: float = config.DT,
        normalize: bool = True,
        seed: Optional[int] = config.TRAJECTORY_SEED,
        burn_in: Optional[int] = None,
        history_window: int = config.HISTORY_WINDOW,
        normalization_stats: Optional[dict] = None,
        split: str = "train",
        trajectory_steps: int = config.TRAJECTORY_STEPS,
        train_fraction: float = config.TRAIN_FRACTION,
        split_gap: int = 0,
        trajectory_path: Optional[str] = None,
    ):
        """
        Args:
            T: Temporal horizon (number of prediction steps)
            dt: Time step for simulation
            normalize: Whether to normalize the data
            seed: Random seed for the shared Lorenz trajectory
            burn_in: Number of initial steps to discard
            history_window: Number of past trajectory states included in each input
            normalization_stats: Optional mean/std values from a training dataset
            split: Trajectory split to use ("train" or "val")
            trajectory_steps: Length of the cached post-burn-in trajectory
            train_fraction: Fraction assigned to the training split
            split_gap: Guard gap between train and validation splits
            trajectory_path: Optional path for the cached trajectory file
        """
        self.T = T
        self.normalize = normalize
        self.burn_in = config.resolve_burn_in_steps(dt, burn_in)
        self.history_window = int(history_window)
        self.split = split
        self.trajectory_path = trajectory_path or default_lorenz_trajectory_path(
            config.DATA_DIR, dt, trajectory_steps, seed
        )

        full_trajectory = get_lorenz_trajectory(
            dt=dt,
            steps=trajectory_steps,
            burn_in=self.burn_in,
            seed=seed,
            path=self.trajectory_path,
        )
        trajectory, self.split_bounds = split_trajectory(
            full_trajectory,
            split=split,
            train_fraction=train_fraction,
            gap=split_gap,
        )
        self.trajectories = trajectory.unsqueeze(0)
        apply_normalization(self, normalization_stats)

        self.samples = self._create_samples()

    def _create_samples(self):
        """Create (input, targets) pairs for all valid starting points."""
        samples = []
        num_traj, seq_len, _ = self.trajectories.shape

        for traj_idx in range(num_traj):
            for m in range(self.history_window - 1, seq_len - self.T):
                input_state = self.trajectories[
                    traj_idx, m - self.history_window + 1 : m + 1
                ].flatten()
                targets = self.trajectories[traj_idx, m + 1 : m + self.T + 1]
                samples.append((input_state, targets))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            input_state: (history_window * 3,) tensor - flattened history window
            targets: (T, 3) tensor - next T states to predict
        """
        return self.samples[idx]


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])  # [batch size, T, 3]
    return inputs, targets
