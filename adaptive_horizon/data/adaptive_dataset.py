from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import adaptive_horizon.config as config
from adaptive_horizon.data.utils import (
    apply_normalization,
    default_lorenz_trajectory_path,
    get_lorenz_trajectory,
    NormalizationStats,
    split_trajectory,
)
from adaptive_horizon.dynamics.lyapunov import (
    compute_forward_ftle,
    compute_local_lyapunov,
)
from adaptive_horizon.training.utils import resolve_burn_in_steps
from adaptive_horizon.utils import time_to_steps


def default_adaptive_T_max(dt: float) -> int:
    """Default fixed rollout for weighted-loss training."""
    return time_to_steps(config.DEFAULT_ADAPTIVE_HORIZON, dt)


class AdaptiveHorizonLorenzDataset(NormalizationStats, Dataset):
    """Adaptive-horizon Lorenz dataset sliced from one shared long trajectory."""

    def __init__(
        self,
        dt: float = config.DT,
        normalize: bool = True,
        seed: int = config.TRAJECTORY_SEED,
        burn_in: int = 0,
        var: int = config.VARIANCE,
        history_window: int = config.HISTORY_WINDOW,
        normalization_stats: Optional[dict] = None,
        debug: bool = False,
        split: str = "train",
        trajectory_steps: int = config.TRAJECTORY_STEPS,
        train_fraction: float = config.TRAIN_FRACTION,
        split_gap: int = 0,
        trajectory_path: Optional[str] = None,
    ):
        self.normalize = normalize
        self.burn_in = resolve_burn_in_steps(dt, burn_in)
        self.var = var
        self.history_window = int(history_window)
        self.split = split
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

        self.base_T = default_adaptive_T_max(dt)
        self.min_T = max(1, self.base_T - self.var)
        self.max_T = min(self.base_T + self.var, config.MAX_TRAIN_T)

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

        traj_np = trajectory.numpy()
        self.lles = []
        self.horizons = []

        lles = compute_local_lyapunov(traj_np, dt=dt)
        lle_max = lles[:, 0]
        self.lles.append(lle_max)
        self.horizons.append(
            self._lle_to_horizon(lle_max, self.base_T, self.min_T, self.max_T)
        )

        self.trajectories = trajectory.unsqueeze(0)
        apply_normalization(self, normalization_stats)

        self.samples = self._create_samples()
        if debug:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._write_t_values(config.EVAL_DIR / f"t_values_{timestamp}.txt")

    def _create_samples(self):
        samples = []
        num_traj, seq_len, _ = self.trajectories.shape

        for traj_idx in range(num_traj):
            traj = self.trajectories[traj_idx]
            horizon = self.horizons[traj_idx]

            for m in range(self.history_window - 1, len(horizon)):
                T = horizon[m]
                if m + T < seq_len:
                    input_state = traj[m - self.history_window + 1 : m + 1].flatten()
                    target_state = traj[m + 1 : m + T + 1]
                    samples.append((input_state, target_state, T))

        return samples

    def _write_t_values(self, output_path: Optional[Path]):
        if output_path is None:
            return

        if output_path.parent != Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)

        t_values = [int(sample[2]) for sample in self.samples]
        unique_t_values, counts = np.unique(t_values, return_counts=True)

        with output_path.open("w", encoding="ascii") as file:
            file.write("# Unique T values and counts\n")
            for value, count in zip(unique_t_values, counts):
                file.write(f"T={int(value)} count={int(count)}\n")

            file.write("\n# LLE and T value per sample\n")
            for traj_idx in range(len(self.horizons)):
                for i in range(len(self.horizons[traj_idx])):
                    file.write(
                        f"{self.lles[traj_idx][i]},{self.horizons[traj_idx][i]}\n"
                    )

    @staticmethod
    def _lle_to_horizon(lambda_max, base_T, min_T, max_T):
        lambda_mean = float(np.mean(lambda_max))
        lambda_std = float(np.std(lambda_max)) + 1e-8
        z_scores = (lambda_max - lambda_mean) / lambda_std

        half_range = max(1.0, (max_T - min_T) / 2.0)
        T = base_T - z_scores * half_range
        T = np.clip(np.round(T), min_T, max_T)
        return T.astype(int)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_state, target, T = self.samples[idx]
        return input_state, target, torch.tensor(T, dtype=torch.float32)


class WeightedLossLorenzDataset(NormalizationStats, Dataset):
    """Weighted-loss Lorenz dataset sliced from one shared long trajectory."""

    def __init__(
        self,
        dt: float = config.DT,
        T_max: Optional[int] = None,
        ftle_window: int = config.FTLE_WINDOW,
        history_window: int = config.HISTORY_WINDOW,
        normalize: bool = True,
        seed: int = config.TRAJECTORY_SEED,
        burn_in: int = 0,
        normalization_stats: Optional[dict] = None,
        debug: bool = False,
        split: str = "train",
        trajectory_steps: int = config.TRAJECTORY_STEPS,
        train_fraction: float = config.TRAIN_FRACTION,
        split_gap: int = 0,
        trajectory_path: Optional[str] = None,
    ):
        if T_max is None:
            T_max = default_adaptive_T_max(dt)
        if T_max < 1:
            raise ValueError(f"T_max must be at least 1, got {T_max}")

        self.T_max = int(T_max)
        self.dt = dt
        self.ftle_window = int(ftle_window)
        self.history_window = int(history_window)
        self.normalize = normalize
        self.burn_in = resolve_burn_in_steps(dt, burn_in)
        self.split = split
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

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

        traj_np = trajectory.numpy()
        self.lambda_scores = [
            compute_forward_ftle(traj_np, dt=dt, window=self.ftle_window)
        ]

        self.trajectories = trajectory.unsqueeze(0)
        apply_normalization(self, normalization_stats)

        self.samples = self._create_samples()
        if debug:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._write_lambda_values(
                config.EVAL_DIR / f"lambda_values_{timestamp}.txt"
            )

    def _create_samples(self):
        samples = []
        num_traj, seq_len, _ = self.trajectories.shape

        for traj_idx in range(num_traj):
            traj = self.trajectories[traj_idx]
            lambda_scores = self.lambda_scores[traj_idx]
            max_start = min(seq_len - self.T_max, len(lambda_scores))

            for m in range(self.history_window - 1, max_start):
                input_state = traj[m - self.history_window + 1 : m + 1].flatten()
                targets = traj[m + 1 : m + self.T_max + 1]
                lambda_score = float(lambda_scores[m])
                samples.append((input_state, targets, lambda_score))

        return samples

    def _write_lambda_values(self, output_path: Path):
        values = np.array([sample[2] for sample in self.samples])
        if output_path.parent != Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="ascii") as file:
            file.write("# Lambda score diagnostics\n")
            file.write(f"count={len(values)}\n")
            file.write(f"mean={float(np.mean(values))}\n")
            file.write(f"std={float(np.std(values))}\n")
            file.write(f"min={float(np.min(values))}\n")
            file.write(f"max={float(np.max(values))}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_state, targets, lambda_score = self.samples[idx]
        return input_state, targets, torch.tensor(lambda_score, dtype=torch.float32)


def collate_fn_adaptive_horizon(batch):
    inputs = torch.stack([item[0] for item in batch])
    T = torch.stack([item[2] for item in batch])

    max_T = int(T.max().item())
    padded_targets = []
    for item in batch:
        target = item[1]
        T_i = target.shape[0]
        if T_i < max_T:
            padding = torch.zeros(max_T - T_i, 3)
            target = torch.cat([target, padding], dim=0)
        padded_targets.append(target)

    targets = torch.stack(padded_targets)
    return inputs, targets, T


def collate_fn_weighted_loss(batch):
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    lambda_scores = torch.stack([item[2] for item in batch])
    return inputs, targets, lambda_scores


AdaptiveLorenzDataset = AdaptiveHorizonLorenzDataset
collate_fn_adaptive = collate_fn_adaptive_horizon
