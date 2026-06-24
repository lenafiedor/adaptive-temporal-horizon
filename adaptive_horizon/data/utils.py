from typing import Optional
from pathlib import Path

import numpy as np
import torch

from adaptive_horizon.dynamics.integrators import rk4_step
from adaptive_horizon.dynamics.systems import DynamicsSystem, get_system
from adaptive_horizon.utils import format_dt
from adaptive_horizon.config import DEFAULT_SYSTEM


def simulate_trajectory(
    system: str | DynamicsSystem, initial_state, dt, steps, burn_in=0
):
    """Simulate a trajectory for the selected dynamical system."""
    system = get_system(system)
    states = [initial_state]
    current_state = np.array(initial_state, dtype=np.float64)

    for _ in range(steps + burn_in):
        current_state = rk4_step(system.rhs, current_state, dt)
        states.append(current_state)

    return states[burn_in:]


def default_trajectory_path(system_name, data_dir, dt, steps, seed):
    """Return the cache path for a shared system rollout."""
    return (
        Path(data_dir) / f"{system_name}_dt{format_dt(dt)}_seed{seed}_steps{steps}.pt"
    )


def get_trajectory(
    system: str | DynamicsSystem,
    dt: float,
    steps,
    burn_in: int,
    seed: int,
    path,
    regenerate=False,
):
    """Load a cached long trajectory or generate it once."""
    system = get_system(system)
    path = Path(path)
    if path.exists() and not regenerate:
        bundle = torch.load(path, map_location="cpu")
        trajectory = bundle["trajectory"]
        trajectory = trajectory.float()

        if isinstance(bundle, dict):
            saved_steps = int(bundle.get("steps", len(trajectory) - 1))
            saved_burn_in = int(bundle.get("burn_in", burn_in))
            saved_seed = int(bundle.get("seed", seed))
            saved_dt = float(bundle.get("dt", dt))
            saved_system = bundle.get("system", DEFAULT_SYSTEM)
            metadata_matches = (
                saved_steps >= steps
                and saved_burn_in == int(burn_in)
                and saved_seed == int(seed)
                and np.isclose(saved_dt, dt)
                and saved_system == system.name
            )
            if metadata_matches:
                return trajectory[: steps + 1]
            print("Cached trajectory metadata does not match; regenerating")
        elif len(trajectory) >= steps + 1:
            return trajectory[: steps + 1]

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    trajectory = simulate_trajectory(
        system,
        initial_state=system.sample_initial_state(rng),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
    )
    trajectory = torch.tensor(np.array(trajectory), dtype=torch.float32)
    torch.save(
        {
            "trajectory": trajectory,
            "system": system.name,
            "system_parameters": dict(system.parameters),
            "dt": dt,
            "steps": steps,
            "burn_in": burn_in,
            "seed": seed,
        },
        path,
    )
    return trajectory


def split_trajectory(trajectory, split, train_fraction=0.8, gap=0):
    """Slice the shared trajectory into train/validation blocks."""
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    n_steps = len(trajectory)
    train_end = int(train_fraction * n_steps)
    gap = int(gap)
    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")

    if split == "train":
        start, end = 0, train_end
    elif split == "val":
        start, end = train_end + gap, n_steps
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    if end <= start:
        raise ValueError(
            f"Empty {split} split for trajectory length {n_steps}, "
            f"train_fraction={train_fraction}, gap={gap}"
        )

    return trajectory[start:end], (start, end)


def apply_normalization(dataset, normalization_stats: Optional[dict]):
    """Apply z-score normalization to a dataset object with trajectories."""
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


class NormalizationStats:
    @property
    def normalization_stats(self):
        if self.mean is None or self.std is None:
            return None
        return {
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
        }
