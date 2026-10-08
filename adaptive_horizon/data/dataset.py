from typing import Optional

import torch
from torch.utils.data import Dataset

import adaptive_horizon.config as config
from adaptive_horizon.data.santafe import observed_trajectory
from adaptive_horizon.data.utils import (
    NormalizationStats,
    apply_normalization,
    default_trajectory_path,
    get_trajectory,
    split_trajectory,
)
from adaptive_horizon.dynamics.systems import get_system
from adaptive_horizon.training.utils import resolve_burn_in_steps


class TrajectoryDataset(NormalizationStats, Dataset):
    """Dataset sliced from one shared long dynamical-system trajectory."""

    def __init__(
        self,
        T: int = 1,
        dt: float = config.DT,
        system: str = config.DEFAULT_SYSTEM,
        normalize: bool = True,
        seed: Optional[int] = config.RANDOM_SEED,
        burn_in: Optional[int] = None,
        normalization_stats: Optional[dict] = None,
        split: str = "train",
        trajectory_steps: int = config.TRAJECTORY_STEPS,
        train_fraction: float = config.TRAIN_FRACTION,
        split_gap: int = 0,
        trajectory_path: Optional[str] = None,
        observed_data=None,
    ):
        """
        Args:
            T: Temporal horizon (number of prediction steps)
            dt: Time step for simulation
            system: Dynamical system name
            normalize: Whether to normalize the data
            seed: Random seed for the shared trajectory
            burn_in: Number of initial steps to discard
            normalization_stats: Optional mean/std values from a training dataset
            split: Trajectory split to use ("train" or "val")
            trajectory_steps: Length of the cached post-burn-in trajectory
            train_fraction: Fraction assigned to the training split
            split_gap: Guard gap between train and validation splits
            trajectory_path: Optional path for the cached trajectory file
        """
        self.T = T
        self.system = None if observed_data is not None else get_system(system)
        self.system_name = system if self.system is None else self.system.name
        self.normalize = normalize
        self.burn_in = (
            0 if observed_data is not None else resolve_burn_in_steps(dt, burn_in)
        )
        self.split = split
        self.trajectory_path = (
            observed_data["metadata"]["path"]
            if observed_data is not None
            else trajectory_path
        ) or default_trajectory_path(
            self.system_name,
            config.system_path(config.DATA_DIR, self.system_name),
            dt,
            trajectory_steps,
            seed,
        )

        if observed_data is not None:
            trajectory = observed_trajectory(self, observed_data, split)
            normalization_stats = observed_data["metadata"]["normalization_stats"]
        else:
            full_trajectory = get_trajectory(
                self.system,
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
            for m in range(seq_len - self.T):
                input_state = self.trajectories[traj_idx, m]
                targets = self.trajectories[traj_idx, m + 1 : m + self.T + 1]
                samples.append((input_state, targets))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            input_state: (state_dim,) tensor - current state
            targets: (T, state_dim) tensor - next T states to predict
        """
        return self.samples[idx]


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return inputs, targets
