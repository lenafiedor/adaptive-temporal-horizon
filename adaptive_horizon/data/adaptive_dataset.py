from pathlib import Path

import numpy as np
import torch

from adaptive_horizon import config
from adaptive_horizon.data.dataset import BaseTrajectoryDataset
from adaptive_horizon.dynamics.lyapunov import (
    compute_forward_ftle,
    compute_local_lyapunov,
    compute_step_tangent_maps,
    lyapunov_center_horizon,
    mean_horizon,
)


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


class LyapunovBasedDataset(BaseTrajectoryDataset):
    def __init__(
        self,
        max_T: int = config.MAX_TRAIN_T,
        var: int = config.VARIANCE,
        normalization_stats: dict | None = None,
        debug: bool = False,
        lyapunov_fraction: float = config.LYAPUNOV_FRACTION,
        lambda_estimate: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.var = var
        self.lyapunov_fraction = float(lyapunov_fraction)

        traj_np = self.raw_trajectory.numpy()
        self.scores = []
        self.horizons = []

        self.lambda_estimate, self.base_T, sample_scores = lyapunov_center_and_ftle(
            traj_np,
            self.dt,
            self.system,
            lambda_estimate,
            self.lyapunov_fraction,
            max_T,
        )
        self.ftle_window = self.base_T
        self.scores.append(sample_scores)

        self.min_T = max(1, self.base_T - self.var)
        self.max_T = min(self.base_T + self.var, max_T)
        horizons = mean_horizon(sample_scores, self.base_T, self.min_T, self.max_T)
        self.horizons.append(horizons)
        self.mean_T = float(np.mean(horizons))

        self._finalize(normalization_stats)
        if debug:
            self._write_t_values(self._debug_path("t_values", self.system_name))

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

    def _write_t_values(self, output_path: Path):
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

    def __getitem__(self, index):
        input_state, target, T = self.samples[index]
        return input_state, target, torch.tensor(T, dtype=torch.float32)


class WeightedLossDataset(BaseTrajectoryDataset):
    """Weighted-loss dataset sliced from one shared long trajectory."""

    def __init__(
        self,
        normalization_stats: dict | None = None,
        debug: bool = False,
        lyapunov_fraction: float = config.LYAPUNOV_FRACTION,
        lambda_estimate: float | None = None,
        **kwargs,
    ):
        """
        Every sample is rolled out for base_T steps, and its forward FTLE over
        the same base_T steps scales the per-step loss weights. base_T follows
        the same Lyapunov-fraction rule as LyapunovBasedDataset. Pass the
        training split's lambda_estimate to validation datasets.
        """
        super().__init__(**kwargs)
        self.lyapunov_fraction = float(lyapunov_fraction)

        self.lambda_estimate, self.base_T, ftle = lyapunov_center_and_ftle(
            self.raw_trajectory.numpy(),
            self.dt,
            self.system,
            lambda_estimate,
            self.lyapunov_fraction,
            config.MAX_TRAIN_T,
        )
        self.ftle_window = self.base_T
        self.T_max = self.base_T
        self.lambda_scores = [ftle]

        self._finalize(normalization_stats)
        if debug:
            self._write_lambda_values(
                self._debug_path("lambda_values", self.system_name)
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
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="ascii") as file:
            file.write("# Lambda score diagnostics\n")
            file.write(f"count={len(values)}\n")
            file.write(f"mean={float(np.mean(values))}\n")
            file.write(f"std={float(np.std(values))}\n")
            file.write(f"min={float(np.min(values))}\n")
            file.write(f"max={float(np.max(values))}\n")

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
