from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from adaptive_horizon import config
from adaptive_horizon.data.utils import (
    NormalizationStats,
    apply_normalization,
    default_trajectory_path,
    get_trajectory,
    split_trajectory,
)
from adaptive_horizon.dynamics.lyapunov import (
    compute_forward_ftle,
    compute_local_lyapunov,
    compute_step_tangent_maps,
    lyapunov_center_horizon,
    mean_horizon,
    time_horizon,
)
from adaptive_horizon.dynamics.systems import get_system
from adaptive_horizon.training.methods import LYAPUNOV_MEAN, LYAPUNOV_TIME
from adaptive_horizon.training.utils import resolve_burn_in_steps


def lyapunov_center_and_ftle(
    trajectory, dt, system, lambda_estimate, lyapunov_fraction, max_T
):
    tangent_maps = compute_step_tangent_maps(
        trajectory, dt=dt, system=system, integration_dt=config.INTEGRATION_DT
    )
    if lambda_estimate is None:
        lles = compute_local_lyapunov(
            trajectory, dt=dt, system=system, tangent_maps=tangent_maps
        )
        lambda_estimate = np.mean(lles[:, 0])
    lambda_estimate = float(lambda_estimate)
    base_T = lyapunov_center_horizon(lambda_estimate, dt, lyapunov_fraction, max_T)
    ftle = compute_forward_ftle(
        trajectory, dt=dt, window=base_T, tangent_maps=tangent_maps
    )
    return lambda_estimate, base_T, ftle


class LyapunovBasedDataset(NormalizationStats, Dataset):
    def __init__(
        self,
        dt: float = config.DT,
        system: str = config.DEFAULT_SYSTEM,
        normalize: bool = True,
        seed: int = config.RANDOM_SEED,
        burn_in: int | None = None,
        max_T: int = config.MAX_TRAIN_T,
        var: int = config.VARIANCE,
        adaptive_method: str = LYAPUNOV_MEAN,
        normalization_stats: dict | None = None,
        debug: bool = False,
        split: str = "train",
        trajectory_steps: int = config.TRAJECTORY_STEPS,
        train_fraction: float = config.TRAIN_FRACTION,
        split_gap: int = 0,
        trajectory_path: str | None = None,
        lyapunov_fraction: float = config.LYAPUNOV_FRACTION,
        lambda_estimate: float | None = None,
    ):
        self.system = get_system(system)
        self.system_name = self.system.name
        self.normalize = normalize
        self.burn_in: int = resolve_burn_in_steps(dt, burn_in_steps=burn_in)
        self.dt = dt
        self.split = split
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.var = var
        self.lyapunov_fraction = float(lyapunov_fraction)

        self.trajectory_path = trajectory_path or default_trajectory_path(
            self.system.name,
            config.system_path(config.DATA_DIR, self.system.name),
            dt,
            trajectory_steps,
            seed,
        )
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

        traj_np = trajectory.numpy()
        self.scores = []
        self.horizons = []

        self.lambda_estimate, self.base_T, sample_scores = lyapunov_center_and_ftle(
            traj_np, dt, self.system, lambda_estimate, self.lyapunov_fraction, max_T
        )
        self.ftle_window = self.base_T
        self.scores.append(sample_scores)

        if adaptive_method == LYAPUNOV_TIME:
            self.min_T = 1
            self.max_T = max_T
            horizons = time_horizon(sample_scores, self.dt, self.min_T, self.max_T)
        else:
            self.min_T = max(1, self.base_T - self.var)
            self.max_T = min(self.base_T + self.var, max_T)
            horizons = mean_horizon(sample_scores, self.base_T, self.min_T, self.max_T)
        self.horizons.append(horizons)
        self.mean_T = float(np.mean(horizons))

        self.trajectories = trajectory.unsqueeze(0)
        apply_normalization(self, normalization_stats)

        self.samples = self._create_samples()
        if debug:
            timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            self._write_t_values(
                config.system_path(config.EVAL_DIR, self.system.name)
                / f"t_values_{timestamp}.txt"
            )

    def _create_samples(self):
        samples = []
        num_traj, seq_len, _ = self.trajectories.shape

        for traj_idx in range(num_traj):
            traj = self.trajectories[traj_idx]
            horizon = self.horizons[traj_idx]

            for m in range(len(horizon)):
                T = horizon[m]
                if m + T < seq_len:
                    input_state = traj[m]
                    target_state = traj[m + 1 : m + T + 1]
                    samples.append((input_state, target_state, T))

        return samples

    def _write_t_values(self, output_path: Path | None):
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

            file.write("\n# FTLE score and T value per sample\n")
            for traj_idx in range(len(self.horizons)):
                for i in range(len(self.horizons[traj_idx])):
                    file.write(
                        f"{self.scores[traj_idx][i]},{self.horizons[traj_idx][i]}\n"
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        input_state, target, T = self.samples[index]
        return input_state, target, torch.tensor(T, dtype=torch.float32)


class WeightedLossDataset(NormalizationStats, Dataset):
    """Weighted-loss dataset sliced from one shared long trajectory."""

    def __init__(
        self,
        dt: float = config.DT,
        system: str = config.DEFAULT_SYSTEM,
        normalize: bool = True,
        seed: int = config.RANDOM_SEED,
        burn_in: int | None = None,
        normalization_stats: dict | None = None,
        debug: bool = False,
        split: str = "train",
        trajectory_steps: int = config.TRAJECTORY_STEPS,
        train_fraction: float = config.TRAIN_FRACTION,
        split_gap: int = 0,
        trajectory_path: str | None = None,
        lyapunov_fraction: float = config.LYAPUNOV_FRACTION,
        lambda_estimate: float | None = None,
    ):
        """
        Every sample is rolled out for base_T steps, and its forward FTLE over
        the same base_T steps scales the per-step loss weights. base_T follows
        the same Lyapunov-fraction rule as LyapunovBasedDataset. Pass the
        training split's lambda_estimate to validation datasets.
        """
        self.system = get_system(system)
        self.system_name = self.system.name
        self.dt = dt
        self.lyapunov_fraction = float(lyapunov_fraction)
        self.normalize = normalize
        self.burn_in: int = resolve_burn_in_steps(dt, burn_in_steps=burn_in)
        self.split = split
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

        self.trajectory_path = trajectory_path or default_trajectory_path(
            self.system.name,
            config.system_path(config.DATA_DIR, self.system.name),
            dt,
            trajectory_steps,
            seed,
        )
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

        traj_np = trajectory.numpy()
        self.lambda_estimate, self.base_T, ftle = lyapunov_center_and_ftle(
            traj_np,
            dt,
            self.system,
            lambda_estimate,
            self.lyapunov_fraction,
            config.MAX_TRAIN_T,
        )
        self.ftle_window = self.base_T
        self.T_max = self.base_T
        self.lambda_scores = [ftle]

        self.trajectories = trajectory.unsqueeze(0)
        apply_normalization(self, normalization_stats)

        self.samples = self._create_samples()
        if debug:
            timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            self._write_lambda_values(
                config.system_path(config.EVAL_DIR, self.system.name)
                / f"lambda_values_{timestamp}.txt"
            )

    def _create_samples(self):
        samples = []
        num_traj, seq_len, _ = self.trajectories.shape

        for traj_idx in range(num_traj):
            traj = self.trajectories[traj_idx]
            lambda_scores = self.lambda_scores[traj_idx]
            max_start = min(seq_len - self.T_max, len(lambda_scores))

            for m in range(max_start):
                input_state = traj[m]
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

    def __getitem__(self, index):
        input_state, targets, lambda_score = self.samples[index]
        return input_state, targets, torch.tensor(lambda_score, dtype=torch.float32)


def collate_fn_lyapunov_based(batch):
    inputs = torch.stack([item[0] for item in batch])
    T = torch.stack([item[2] for item in batch])

    max_T = int(T.max().item())
    padded_targets = []
    for item in batch:
        target = item[1]
        T_i = target.shape[0]
        if T_i < max_T:
            padding = torch.zeros(max_T - T_i, target.shape[-1])
            target = torch.cat([target, padding], dim=0)
        padded_targets.append(target)

    targets = torch.stack(padded_targets)
    return inputs, targets, T


def collate_fn_weighted_loss(batch):
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    lambda_scores = torch.stack([item[2] for item in batch])
    return inputs, targets, lambda_scores
