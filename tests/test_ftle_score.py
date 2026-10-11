import numpy as np
import pytest

from adaptive_horizon.data.adaptive_dataset import LyapunovBasedDataset
from adaptive_horizon.data.utils import get_trajectory
from adaptive_horizon.dynamics.integrators import rk4_step_coupled
from adaptive_horizon.dynamics.lyapunov import (
    compute_forward_ftle,
    compute_local_lyapunov,
    compute_step_tangent_maps,
)
from adaptive_horizon.dynamics.systems import get_system
from adaptive_horizon.training.utils import resolve_burn_in_steps


def _naive_forward_ftle(trajectory, dt, window, system_name):
    """Reference implementation: integrate each window from scratch."""
    system = get_system(system_name)
    out = []
    for start in range(len(trajectory) - window):
        tangent_map = np.eye(system.dim)
        for offset in range(window):
            x = trajectory[start + offset]
            x, tangent_map = rk4_step_coupled(
                x, tangent_map, dt, system.rhs, system.jacobian
            )
        sigma_max = float(np.linalg.svd(tangent_map, compute_uv=False)[0])
        out.append(np.log(sigma_max + 1e-12) / (window * dt))
    return np.array(out)


@pytest.fixture
def lorenz_trajectory(tmp_path):
    return get_trajectory(
        "lorenz", dt=0.01, steps=60, burn_in=100, seed=7, path=tmp_path / "l.pt"
    ).numpy()


@pytest.mark.parametrize("window", [1, 3, 6])
def test_batched_ftle_matches_naive(lorenz_trajectory, window):
    fast = compute_forward_ftle(
        lorenz_trajectory, dt=0.01, window=window, system="lorenz"
    )
    slow = _naive_forward_ftle(lorenz_trajectory, 0.01, window, "lorenz")
    assert fast.shape == (len(lorenz_trajectory) - window,)
    np.testing.assert_allclose(fast, slow, rtol=1e-10, atol=1e-10)


def test_precomputed_maps_give_same_results(lorenz_trajectory):
    maps = compute_step_tangent_maps(lorenz_trajectory, dt=0.01, system="lorenz")
    np.testing.assert_allclose(
        compute_local_lyapunov(lorenz_trajectory, dt=0.01, system="lorenz"),
        compute_local_lyapunov(lorenz_trajectory, dt=0.01, tangent_maps=maps),
    )
    np.testing.assert_allclose(
        compute_forward_ftle(lorenz_trajectory, dt=0.01, window=4, system="lorenz"),
        compute_forward_ftle(lorenz_trajectory, dt=0.01, window=4, tangent_maps=maps),
    )


def _dataset(tmp_path, **kwargs):
    return LyapunovBasedDataset(
        dt=0.08,
        system="lorenz",
        burn_in=100,
        trajectory_steps=600,
        trajectory_path=str(tmp_path / "traj.pt"),
        max_T=10,
        **kwargs,
    )


def test_dataset_ftle_window_is_base_T(tmp_path):
    ds = _dataset(tmp_path)
    n = ds.trajectories.shape[1]
    assert ds.ftle_window == ds.base_T
    assert len(ds.scores[0]) == n - ds.base_T
    assert ds.horizons[0].min() >= ds.min_T and ds.horizons[0].max() <= ds.max_T


def test_validation_split_reuses_train_lambda(tmp_path):
    train = _dataset(tmp_path)
    val = _dataset(tmp_path, split="val", lambda_estimate=train.lambda_estimate)
    assert val.base_T == train.base_T
    assert val.ftle_window == train.ftle_window


def test_burn_in_is_interpreted_as_steps(tmp_path):
    assert _dataset(tmp_path).burn_in == 100
    default = LyapunovBasedDataset(
        dt=0.08,
        system="lorenz",
        trajectory_steps=600,
        trajectory_path=str(tmp_path / "default.pt"),
        max_T=10,
    )
    assert default.burn_in == resolve_burn_in_steps(0.08)


def test_weighted_loss_dataset_uses_base_T_window(tmp_path):
    from adaptive_horizon.data.adaptive_dataset import WeightedLossDataset

    common = {
        "dt": 0.08,
        "system": "lorenz",
        "burn_in": 100,
        "trajectory_steps": 600,
        "trajectory_path": str(tmp_path / "traj.pt"),
    }
    lm = _dataset(tmp_path)
    wl = WeightedLossDataset(**common)
    assert wl.ftle_window == wl.base_T == lm.base_T
    assert wl.T_max == wl.base_T
    assert all(sample[1].shape[0] == wl.base_T for sample in wl.samples)
    assert len(wl.lambda_scores[0]) == wl.trajectories.shape[1] - wl.base_T
    val = WeightedLossDataset(**common, split="val", lambda_estimate=wl.lambda_estimate)
    assert val.ftle_window == wl.ftle_window
