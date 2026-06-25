import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from adaptive_horizon.data.dataset import TrajectoryDataset, collate_fn
from adaptive_horizon.data.utils import get_trajectory
from adaptive_horizon.dynamics.lyapunov import (
    compute_forward_ftle,
    compute_global_lyapunov,
    compute_local_lyapunov,
)
from adaptive_horizon.dynamics.systems import get_system


def test_system_registry_contains_lorenz_and_rossler():
    lorenz = get_system("lorenz")
    rossler = get_system("rossler")

    assert lorenz.name == "lorenz"
    assert rossler.name == "rossler"
    assert lorenz.dim == 3
    assert rossler.dim == 3


def test_rossler_rhs_and_jacobian_shapes():
    system = get_system("rossler")
    state = np.array([1.0, 2.0, 3.0])

    assert system.rhs(state).shape == (3,)
    assert system.jacobian(*state).shape == (3, 3)
    np.testing.assert_allclose(
        system.jacobian(*state),
        np.array(
            [
                [0.0, -1.0, -1.0],
                [1.0, 0.37, 0.0],
                [3.0, 0.0, -4.7],
            ]
        ),
    )


def test_unknown_system_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown dynamical system"):
        get_system("not-a-system")


def test_short_rossler_trajectory_cache(tmp_path):
    path = tmp_path / "rossler.pt"

    trajectory = get_trajectory(
        "rossler",
        dt=0.01,
        steps=6,
        burn_in=0,
        seed=123,
        path=path,
    )

    assert trajectory.shape == (7, 3)
    bundle = torch.load(path, map_location="cpu")
    assert bundle["system"] == "rossler"


def test_rossler_fixed_horizon_loader_shapes(tmp_path):
    dataset = TrajectoryDataset(
        T=2,
        dt=0.01,
        system="rossler",
        seed=123,
        burn_in=0,
        trajectory_steps=12,
        trajectory_path=str(tmp_path / "rossler_dataset.pt"),
    )
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    inputs, targets = next(iter(loader))

    assert inputs.shape == (4, 3)
    assert targets.shape == (4, 2, 3)


@pytest.mark.parametrize("system_name", ["lorenz", "rossler"])
def test_lyapunov_utilities_accept_registered_systems(system_name, tmp_path):
    trajectory = get_trajectory(
        system_name,
        dt=0.01,
        steps=6,
        burn_in=0,
        seed=123,
        path=tmp_path / f"{system_name}.pt",
    ).numpy()

    local_lle = compute_local_lyapunov(trajectory, dt=0.01, system=system_name)
    forward_ftle = compute_forward_ftle(
        trajectory,
        dt=0.01,
        window=2,
        system=system_name,
    )
    global_lle = compute_global_lyapunov(dt=0.01, steps=3, system=system_name)

    assert local_lle.shape == (6, 3)
    assert forward_ftle.shape == (5,)
    assert np.isfinite(global_lle)
