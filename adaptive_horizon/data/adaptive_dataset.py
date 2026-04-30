from datetime import datetime
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import adaptive_horizon.config as config
from adaptive_horizon.dynamics.lorenz import simulate_lorenz
from adaptive_horizon.dynamics.lyapunov import (
    compute_forward_ftle,
    compute_local_lyapunov,
    smooth_lle,
)


def default_adaptive_T_max(dt: float) -> int:
    """Default fixed rollout for weighted-loss training."""
    return max(1, int(round(config.DEFAULT_ADAPTIVE_HORIZON / dt)))


def _apply_normalization(dataset, normalization_stats):
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


def _normalization_stats(dataset):
    if dataset.mean is None or dataset.std is None:
        return None
    return {
        "mean": dataset.mean.detach().cpu().tolist(),
        "std": dataset.std.detach().cpu().tolist(),
    }


class AdaptiveHorizonLorenzDataset(Dataset):
    """Lorenz dataset with sample-specific mutable temporal horizons."""

    def __init__(
        self,
        num_trajectories: int = config.NUM_TRAJECTORIES,
        steps_per_trajectory: int = config.STEPS_PER_TRAJECTORY,
        dt: float = config.DT,
        normalize: bool = True,
        seed: Optional[int] = None,
        burn_in: int = 0,
        horizon_prior_path: Optional[str] = None,
        base_T: Optional[int] = None,
        min_T: Optional[int] = None,
        max_T: Optional[int] = None,
        alpha: float = 1.0,
        normalization_stats: Optional[dict] = None,
        debug: bool = False,
    ):
        self.normalize = normalize
        self.alpha = alpha
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

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

            traj, lles = traj[burn_in:], lles[burn_in:]
            trajectories.append(traj)

            lle_max = lles[:, 0]
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

        self.trajectories = torch.tensor(np.array(trajectories), dtype=torch.float32)
        _apply_normalization(self, normalization_stats)

        self.samples = self._create_samples()
        if debug:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._write_t_values(config.EVAL_DIR / f"t_values_{timestamp}.txt")

    def _apply_normalization(self, normalization_stats):
        _apply_normalization(self, normalization_stats)

    @property
    def normalization_stats(self):
        return _normalization_stats(self)

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

            file.write("\n# T value per sample\n")
            for value in t_values:
                file.write(f"{value}\n")

    @classmethod
    def _load_horizon_prior(cls, horizon_prior_path: Optional[str], dt: float):
        path = (
            Path(horizon_prior_path)
            if horizon_prior_path is not None
            else config.EVAL_DIR / f"horizon_prior_dt_{str(dt).split('.')[1]}.json"
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
                base_T = int(config.DEFAULT_ADAPTIVE_HORIZON / dt)

        if min_T is None:
            min_T = (
                int(prior["recommended_min_T"])
                if prior is not None
                else max(1, base_T - 2)
            )

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
        input_state, target, T = self.samples[idx]
        return input_state, target, torch.tensor(T, dtype=torch.float32)


class WeightedLossLorenzDataset(Dataset):
    """Lorenz dataset with fixed rollouts and aligned FTLE scores."""

    def __init__(
        self,
        num_trajectories: int = config.NUM_TRAJECTORIES,
        steps_per_trajectory: int = config.STEPS_PER_TRAJECTORY,
        dt: float = config.DT,
        T_max: Optional[int] = None,
        ftle_window: int = config.WINDOW_SIZE,
        normalize: bool = True,
        seed: Optional[int] = None,
        burn_in: int = 0,
        normalization_stats: Optional[dict] = None,
        debug: bool = False,
    ):
        if T_max is None:
            T_max = default_adaptive_T_max(dt)
        if T_max < 1:
            raise ValueError(f"T_max must be at least 1, got {T_max}")

        self.T_max = int(T_max)
        self.dt = dt
        self.ftle_window = int(ftle_window)
        self.normalize = normalize
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

        if seed is not None:
            np.random.seed(seed)

        trajectories = []
        self.lambda_scores = []

        for _ in range(num_trajectories):
            initial_state = [
                np.random.uniform(-20, 20),
                np.random.uniform(-20, 20),
                np.random.uniform(0, 50),
            ]
            traj = np.array(
                simulate_lorenz(
                    initial_state=initial_state,
                    dt=dt,
                    steps=steps_per_trajectory + burn_in,
                ),
                dtype=np.float32,
            )
            traj = traj[burn_in:]
            trajectories.append(traj)
            self.lambda_scores.append(
                compute_forward_ftle(traj, dt=dt, window=self.ftle_window)
            )

        self.trajectories = torch.tensor(np.array(trajectories), dtype=torch.float32)
        _apply_normalization(self, normalization_stats)

        self.samples = self._create_samples()
        if debug:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._write_lambda_values(
                config.EVAL_DIR / f"lambda_values_{timestamp}.txt"
            )

    def _apply_normalization(self, normalization_stats):
        _apply_normalization(self, normalization_stats)

    @property
    def normalization_stats(self):
        return _normalization_stats(self)

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
