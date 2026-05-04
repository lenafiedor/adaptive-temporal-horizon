import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional

import adaptive_horizon.config as config
from adaptive_horizon.dynamics.lorenz import simulate_lorenz


class LorenzDataset(Dataset):
    """PyTorch Dataset for Lorenz attractor trajectories with temporal horizon support."""

    def __init__(
        self,
        num_trajectories: int = config.NUM_TRAJECTORIES,
        steps_per_trajectory: int = config.STEPS_PER_TRAJECTORY,
        T: int = 1,
        dt: float = config.DT,
        normalize: bool = True,
        seed: Optional[int] = None,
        burn_in: Optional[int] = None,
        normalization_stats: Optional[dict] = None,
    ):
        """
        Args:
            num_trajectories: Number of trajectories to generate
            steps_per_trajectory: Length of each trajectory
            T: Temporal horizon (number of prediction steps)
            dt: Time step for simulation
            normalize: Whether to normalize the data
            seed: Random seed for reproducibility
            burn_in: Number of initial steps to discard
            normalization_stats: Optional mean/std values from a training dataset
        """
        self.T = T
        self.normalize = normalize
        self.burn_in = config.resolve_burn_in_steps(dt, burn_in)

        if seed is not None:
            np.random.seed(seed)

        trajectories = []
        for _ in range(num_trajectories):
            initial_state = [
                np.random.uniform(-20, 20),
                np.random.uniform(-20, 20),
                np.random.uniform(0, 50),
            ]
            traj = simulate_lorenz(
                initial_state=initial_state,
                dt=dt,
                steps=steps_per_trajectory,
                burn_in=self.burn_in,
            )
            trajectories.append(traj)

        self.trajectories = torch.tensor(np.array(trajectories), dtype=torch.float32)

        # Z-score normalization
        if self.normalize:
            if normalization_stats is None:
                self.mean = self.trajectories.mean(dim=(0, 1))
                self.std = self.trajectories.std(dim=(0, 1))
            else:
                self.mean = torch.as_tensor(
                    normalization_stats["mean"], dtype=torch.float32
                )
                self.std = torch.as_tensor(
                    normalization_stats["std"], dtype=torch.float32
                )
            self.trajectories = (self.trajectories - self.mean) / (self.std + 1e-8)
        else:
            self.mean = None
            self.std = None

        self.samples = self._create_samples()

    @property
    def normalization_stats(self):
        if self.mean is None or self.std is None:
            return None
        return {
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
        }

    def _create_samples(self):
        """Create (input, targets) pairs for all valid starting points."""
        samples = []
        num_traj, seq_len, _ = self.trajectories.shape

        for traj_idx in range(num_traj):
            for m in range(seq_len - self.T):
                input_state = self.trajectories[traj_idx, m]  # [3,]
                targets = self.trajectories[traj_idx, m + 1 : m + self.T + 1]  # [T, 3]
                samples.append((input_state, targets))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            input_state: (3,) tensor - starting state
            targets: (T, 3) tensor - next T states to predict
        """
        return self.samples[idx]


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    inputs = torch.stack([item[0] for item in batch])  # [batch size, 3]
    targets = torch.stack([item[1] for item in batch])  # [batch size, T, 3]
    return inputs, targets
