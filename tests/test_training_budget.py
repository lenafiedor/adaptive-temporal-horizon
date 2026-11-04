import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from adaptive_horizon.data.dataset import TrajectoryDataset
from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.training.methods import LYAPUNOV_MEAN
from adaptive_horizon.training.train import train
from adaptive_horizon.training.utils import (
    find_fixed_models,
    fixed_sweep_rollout_steps,
    resolve_dirs,
)


def test_find_fixed_models_searches_recursively_and_skips_adaptive(tmp_path):
    nested = tmp_path / "run" / "fixed"
    nested.mkdir(parents=True)
    (nested / "mlp_T2_seed1_x.pt").touch()
    (nested / "mlp_T1_seed0_x.pt").touch()
    (nested / "adaptive_mlp_seed0_x.pt").touch()

    models = find_fixed_models(tmp_path)

    assert [(path.name, T, seed) for path, T, seed in models] == [
        ("mlp_T1_seed0_x.pt", 1, 0),
        ("mlp_T2_seed1_x.pt", 2, 1),
    ]


def test_resolve_dirs_appends_when_output_dir_exists(tmp_path):
    output_dir = tmp_path / "run"
    result = resolve_dirs(
        dt=0.08,
        max_train_T=3,
        debug=False,
        budget_based=True,
        output_dir=output_dir,
    )
    model_root, fixed_dir, adaptive_dir, _, append = result

    assert model_root == output_dir.resolve()
    assert fixed_dir == model_root / "fixed"
    assert adaptive_dir == model_root / "adaptive"
    assert append is False

    result = resolve_dirs(
        dt=0.08,
        max_train_T=3,
        debug=False,
        budget_based=True,
        output_dir=output_dir,
    )

    assert result[-1] is True


def test_fixed_sweep_rollout_steps_matches_fixed_datasets(tmp_path):
    steps, max_T, epochs_per_T = 600, 4, 3
    expected = 0
    for T in range(1, max_T + 1):
        dataset = TrajectoryDataset(
            T=T,
            dt=0.08,
            system="lorenz",
            burn_in=100,
            trajectory_steps=steps,
            trajectory_path=str(tmp_path / "traj.pt"),
            split_gap=10,
        )
        expected += epochs_per_T * len(dataset) * T

    assert fixed_sweep_rollout_steps(max_T, epochs_per_T, steps) == expected


def _lyapunov_mean_loader(n_samples=50, batch_size=8):
    torch.manual_seed(0)
    T = torch.randint(2, 5, (n_samples,)).float()
    targets = torch.randn(n_samples, 4, 3)
    dataset = TensorDataset(torch.randn(n_samples, 3), targets, T)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_rollout_step_budget_stops_training_within_budget():
    loader = _lyapunov_mean_loader()
    full_epoch_steps = sum(
        inputs.shape[0] * int(T.max().item()) for inputs, _, T in loader
    )
    budget = int(2.5 * full_epoch_steps)
    model = MLP(MLPConfig(3, 3, [10, 10, 10], residual_connections=True, k=1), 0)
    metadata = {}

    train_losses, _ = train(
        model,
        loader,
        loader,
        torch.optim.Adam(model.parameters()),
        epochs=math.inf,
        adaptive=True,
        adaptive_method=LYAPUNOV_MEAN,
        metadata=metadata,
        max_rollout_steps=budget,
    )

    assert len(train_losses) == 3
    assert 2 * full_epoch_steps < metadata["rollout_steps"] <= budget
    assert metadata["rollout_steps_budget_result"] == {
        "epochs_ran": 3,
        "stopped_by_rollout_steps": True,
    }
