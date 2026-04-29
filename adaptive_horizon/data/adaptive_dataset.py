import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional
import json
from pathlib import Path
from datetime import datetime

from adaptive_horizon.config import DT, EVAL_DIR
from adaptive_horizon.dynamics.lorenz import simulate_lorenz
from adaptive_horizon.dynamics.lyapunov import smooth_lle, compute_local_lyapunov

DEFAULT_HORIZON = 48


class AdaptiveLorenzDataset(Dataset):
    """PyTorch Dataset for Lorenz attractor trajectories with temporal horizon support."""

    def __init__(
        self,
        num_trajectories: int = 10,
        steps_per_trajectory: int = 10000,
        dt: float = DT,
        normalize: bool = True,
        seed: Optional[int] = None,
        burn_in: int = 0,
        horizon_prior_path: Optional[str] = None,
        base_T: Optional[int] = None,
        min_T: Optional[int] = None,
        max_T: Optional[int] = None,
        alpha: float = 1.0,
        debug: bool = False,
    ):
        """
        Args:
            num_trajectories: Number of trajectories to generate
            steps_per_trajectory: Length of each trajectory
            dt: Time step for simulation
            normalize: Whether to normalize the data
            seed: Random seed for reproducibility
            burn_in: Number of initial steps to discard (transient period)
            horizon_prior_path: Path to the cross-validation-derived horizon prior
            base_T: Horizon center used for adaptive mapping
            min_T: Lower clip bound for adaptive horizons
            max_T: Upper clip bound for adaptive horizons
            alpha: Scale factor for horizon adaptation around base_T
            debug: Whether to write T values to a file
        """
        self.normalize = normalize
        self.alpha = alpha

        prior = self._load_horizon_prior(horizon_prior_path, dt)
        self.base_T, self.min_T, self.max_T = self._resolve_horizon_params(
            dt, prior, base_T, min_T, max_T
        )

        if seed is not None:
            np.random.seed(seed)

        trajectories = []
        self.lles = []
        self.horizons = []

        for _ in range(num_trajectories):
            initial_state = [
                np.random.uniform(-20, 20),
                np.random.uniform(-20, 20),
                np.random.uniform(0, 50),
            ]
            traj = simulate_lorenz(
                initial_state=initial_state, dt=dt, steps=steps_per_trajectory + burn_in
            )
            lles = smooth_lle(compute_local_lyapunov(traj, dt=dt), window=5)

            # Discard the burn-in period from trajectory and LLEs
            traj, lles = traj[burn_in:], lles[burn_in:]
            trajectories.append(traj)

            lle_max = lles[:, 0]  # Take the largest exponents

            self.lles.append(lle_max)
            self.horizons.append(
                self._lle_to_horizon(
                    lle_max,
                    self.base_T,
                    self.min_T,
                    self.max_T,
                    alpha=self.alpha,
                )
            )

        self.trajectories = torch.tensor(
            np.array(trajectories), dtype=torch.float32
        )  # [num_trajectories, steps_per_trajectory, 3]

        # Z-score normalization
        if self.normalize:
            self.mean = self.trajectories.mean(dim=(0, 1))  # scalar
            self.std = self.trajectories.std(dim=(0, 1))  # scalar
            self.trajectories = (self.trajectories - self.mean) / (self.std + 1e-8)

        self.samples = self._create_samples()
        if debug:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._write_t_values(EVAL_DIR / f"t_values_{timestamp}.txt")

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
                    target_state = traj[m + 1 : m + T + 1]  # [T, 3]
                    samples.append((input_state, target_state, T))

        return samples

    def _write_t_values(self, output_path: Optional[str]):
        if output_path is None:
            return

        path = Path(output_path)
        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)

        t_values = [int(sample[2]) for sample in self.samples]
        unique_t_values, counts = np.unique(t_values, return_counts=True)

        with path.open("w", encoding="ascii") as file:
            file.write("# Unique T values and counts\n")
            for value, count in zip(unique_t_values, counts):
                file.write(f"T={int(value)} count={int(count)}\n")

            file.write("\n# T value per sample\n")
            for value in t_values:
                file.write(f"{value}\n")

    @classmethod
    def _load_horizon_prior(cls, horizon_prior_path: Optional[str], dt: float):
        path = (
            Path(horizon_prior_path)
            if horizon_prior_path is not None
            else EVAL_DIR / f"horizon_prior_dt_{str(dt).split(".")[1]}.json"
        )

        if not path.exists():
            print(f"Horizon prior file not found: {path}")
            return None

        with path.open("r", encoding="ascii") as file:
            return json.load(file)

    @staticmethod
    def _resolve_horizon_params(dt, prior, base_T, min_T, max_T):
        if base_T is None:
            if prior is not None:
                base_T = int(prior["best_train_T"])
            else:
                base_T = int(DEFAULT_HORIZON / (100 * dt))

        if min_T is None:
            min_T = int(prior["recommended_min_T"]) if prior is not None else max(1, base_T - 2)

        if max_T is None:
            max_T = int(prior["recommended_max_T"]) if prior is not None else base_T + 2

        return int(base_T), int(min_T), int(max_T)

    @staticmethod
    def _lle_to_horizon(lambda_max, base_T, min_T, max_T, alpha=1.0):
        lambda_mean = float(np.mean(lambda_max))
        lambda_std = float(np.std(lambda_max)) + 1e-8
        z_scores = (lambda_max - lambda_mean) / lambda_std

        half_range = max(1.0, (max_T - min_T) / 2.0)
        T = base_T - alpha * z_scores * half_range
        T = np.clip(np.round(T), min_T, max_T)
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


def collate_fn_adaptive(batch):
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
