import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional

from adaptive_horizon.dynamics.lorenz import simulate_lorenz
from adaptive_horizon.dynamics.lyapunov import smooth_lle, compute_local_lyapunov


class AdaptiveLorenzDataset(Dataset):
    """PyTorch Dataset for Lorenz attractor trajectories with temporal horizon support."""

    def __init__(
        self,
        num_trajectories: int = 10,
        steps_per_trajectory: int = 10000,
        T: int = 1,
        dt: float = 0.01,
        normalize: bool = True,
        seed: Optional[int] = None
    ):
        """
        Args:
            num_trajectories: Number of trajectories to generate
            steps_per_trajectory: Length of each trajectory
            dt: Time step for simulation
            normalize: Whether to normalize the data
            seed: Random seed for reproducibility
        """
        self.normalize = normalize

        if seed is not None:
            np.random.seed(seed)

        trajectories = []
        self.lles = []
        self.horizons = []

        for _ in range(num_trajectories):
            initial_state = [
                np.random.uniform(-20, 20),
                np.random.uniform(-20, 20),
                np.random.uniform(0, 50)
            ]
            traj = simulate_lorenz(
                initial_state=initial_state,
                dt=dt,
                steps=steps_per_trajectory
            ) #  [steps_per_trajectory, 3]
            trajectories.append(traj)

            # Compute local Lyapunov exponents for this trajectory
            lles = smooth_lle(compute_local_lyapunov(traj, dt=dt), window=5)
            lle_max = lles[:, 0]  # Take the largest exponents

            self.lles.append(lle_max)
            self.horizons.append(self._lle_to_horizon(lle_max, dt))
            print(f"Initialized a trajectory with LLE max: {np.max(lle_max)}")

        self.trajectories = torch.tensor(np.array(trajectories), dtype=torch.float32)  # [num_trajectories, steps_per_trajectory, 3]

        # Z-score normalization
        if self.normalize:
            self.mean = self.trajectories.mean(dim=(0, 1))  # scalar
            self.std = self.trajectories.std(dim=(0, 1))  # scalar
            self.trajectories = (self.trajectories - self.mean) / (self.std + 1e-8)

        self.samples = self._create_samples()

    def _create_samples(self):
        """Create (input, targets) pairs for all valid starting points."""
        samples = []
        num_traj, seq_len, _ = self.trajectories.shape

        for traj_idx in range(num_traj):
            traj = self.trajectories[traj_idx]
            horizon = self.horizons[traj_idx]

            for m in range(len(horizon)):
                T = horizon[m]
                if m + T < seq_len:
                    input_state = traj[m]  # [3,]
                    target_state = traj[m+1: m+T+1]  # [T, 3]
                    samples.append((input_state, target_state, T))

        return samples

    def _lle_to_horizon(self, lambda_max, dt, C=2.0, min_T=1, max_T=16):
        tau = np.log(C) / (lambda_max + 1e-6)
        T = np.clip((tau / dt), min_T, max_T)
        return T.astype(int)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            input_state: (3,) tensor - starting state
            targets: (T, 3) tensor - next T states to predict
        """
        input_state, target, T = self.samples[idx]
        return input_state, target, torch.tensor(T, dtype=torch.float32)


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    inputs = torch.stack([item[0] for item in batch])  # [batch size, 3]
    T = torch.stack([item[2] for item in batch])  # [B]
    
    # Pad targets to max_T in batch
    max_T = int(T.max().item())
    padded_targets = []
    for item in batch:
        target = item[1]  # [T_i, 3]
        T_i = target.shape[0]
        if T_i < max_T:
            padding = torch.zeros(max_T - T_i, 3)
            target = torch.cat([target, padding], dim=0)
        padded_targets.append(target)
    
    targets = torch.stack(padded_targets)  # [batch size, max_T, 3]
    return inputs, targets, T
