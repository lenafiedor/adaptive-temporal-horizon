import pytest
import torch

from adaptive_horizon.training.utils import (
    find_fixed_models,
    fixed_budget_wall_time,
    resolve_dirs,
)


def save_checkpoint(path, wall_time):
    torch.save({"metadata": {"wall_time_seconds": wall_time}}, path)


def test_fixed_budget_wall_time_sums_mean_time_by_T(tmp_path):
    save_checkpoint(tmp_path / "mlp_T1_seed0_x.pt", 10.0)
    save_checkpoint(tmp_path / "mlp_T1_seed1_x.pt", 14.0)
    save_checkpoint(tmp_path / "mlp_T2_seed0_x.pt", 20.0)
    save_checkpoint(tmp_path / "mlp_T2_seed1_x.pt", 24.0)

    total = fixed_budget_wall_time(tmp_path, 2)

    assert total == 34.0


def test_fixed_budget_wall_time_searches_subdirectories(tmp_path):
    fixed_subdir = tmp_path / "fixed"
    fixed_subdir.mkdir()
    save_checkpoint(fixed_subdir / "mlp_T1_seed0_x.pt", 10.0)
    save_checkpoint(fixed_subdir / "mlp_T2_seed0_x.pt", 20.0)

    total = fixed_budget_wall_time(tmp_path, 2)

    assert total == 30.0


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


def test_fixed_budget_wall_time_requires_each_T(tmp_path):
    save_checkpoint(tmp_path / "mlp_T1_seed0_x.pt", 10.0)

    with pytest.raises(FileNotFoundError, match=r"T values: \[2\]"):
        fixed_budget_wall_time(tmp_path, 2)


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
