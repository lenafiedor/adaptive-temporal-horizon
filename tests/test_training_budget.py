import pytest
import torch

from adaptive_horizon.training.utils import fixed_budget_wall_time, resolve_dirs


def save_checkpoint(path, wall_time):
    torch.save({"metadata": {"wall_time_seconds": wall_time}}, path)


def test_fixed_budget_wall_time_sums_mean_time_by_T(tmp_path):
    save_checkpoint(tmp_path / "mlp_T1_seed0_x.pt", 10.0)
    save_checkpoint(tmp_path / "mlp_T1_seed1_x.pt", 14.0)
    save_checkpoint(tmp_path / "mlp_T2_seed0_x.pt", 20.0)
    save_checkpoint(tmp_path / "mlp_T2_seed1_x.pt", 24.0)

    total = fixed_budget_wall_time(tmp_path, 2)

    assert total == 34.0


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
    model_root, fixed_dir, adaptive_dir, _, _, append = result

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
