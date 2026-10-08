from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from adaptive_horizon import config
from adaptive_horizon.config import resolve_burn_in_steps
from adaptive_horizon.data.utils import (
    NormalizationStats,
    apply_normalization,
    default_split_gap,
    default_trajectory_path,
    get_trajectory,
    split_trajectory,
)
from adaptive_horizon.data.santafe import observed_trajectory
from adaptive_horizon.dynamics.systems import get_system
from adaptive_horizon.utils import timestamp


class BaseTrajectoryDataset(NormalizationStats, Dataset):
    def __init__(
        self,
        dt: float = config.DT,
        system: str = config.DEFAULT_SYSTEM,
        normalize: bool = True,
        seed: int | None = config.RANDOM_SEED,
        burn_in: int | None = None,
        split: str = "train",
        trajectory_steps: int = config.TRAJECTORY_STEPS,
        train_fraction: float = config.TRAIN_FRACTION,
        split_gap: int = 0,
        trajectory_path: str | None = None,
        observed_data: dict | None = None,
    ):
        """``observed_data`` (from ``load_santafe``) replaces the simulated
        trajectory with a measured, already split delay-embedded recording."""
        self.observed_data = observed_data
        self.dt = dt
        self.normalize = normalize
        self.split = split
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        if observed_data is not None:
            self.system = None
            self.system_name = system
            self.burn_in = 0
            self.raw_trajectory = observed_trajectory(self, observed_data, split)
            return

        self.system = get_system(system)
        self.system_name = self.system.name
        self.burn_in: int = resolve_burn_in_steps(dt, burn_in_steps=burn_in)
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
        self.raw_trajectory, self.split_bounds = split_trajectory(
            full_trajectory,
            split=split,
            train_fraction=train_fraction,
            gap=split_gap,
        )

    def _finalize(self, normalization_stats: dict | None):
        if self.observed_data is not None:
            # Measured data is always normalized with training-split statistics.
            normalization_stats = self.observed_data["metadata"]["normalization_stats"]
        self.trajectories = self.raw_trajectory.unsqueeze(0)
        apply_normalization(self, normalization_stats)
        self.samples = self._create_samples()

    def _create_samples(self):
        raise NotImplementedError

    @staticmethod
    def _debug_path(prefix: str, system_name: str) -> Path:
        return config.system_path(config.EVAL_DIR, system_name) / (
            f"{prefix}_{timestamp()}.txt"
        )

    def __len__(self):
        return len(self.samples)


class TrajectoryDataset(BaseTrajectoryDataset):
    def __init__(self, T: int = 1, normalization_stats: dict | None = None, **kwargs):
        self.T = T
        super().__init__(**kwargs)
        self._finalize(normalization_stats)

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

    def __getitem__(self, idx):
        """
        Returns:
            input_state: (state_dim,) tensor - current state
            targets: (T, state_dim) tensor - next T states to predict
        """
        return self.samples[idx]


def collate_fn(batch):
    """Stack every field of the samples; works for any fixed-size sample tuple."""
    return tuple(torch.stack(column) for column in zip(*batch))


def make_val_loader(
    system_name,
    dt,
    T,
    normalization_stats=None,
    batch_size=config.BATCH_SIZE,
    observed_data=None,
    split="val",
):
    dataset = TrajectoryDataset(
        T=T,
        dt=dt,
        system=system_name,
        normalize=True,
        seed=config.RANDOM_SEED,
        burn_in=None if observed_data is not None else resolve_burn_in_steps(dt),
        split=split,
        split_gap=default_split_gap(T),
        normalization_stats=normalization_stats,
        observed_data=observed_data,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
