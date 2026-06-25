from pathlib import Path

import pytest

from adaptive_horizon.evaluation.utils import get_dt_from_model_dir
from adaptive_horizon.evaluation import cross_validation


def test_get_dt_from_model_dir_accepts_dt_segment_with_or_without_suffix():
    assert get_dt_from_model_dir(Path("experiments/lorenz/models/dt_08")) == 0.08
    assert (
        get_dt_from_model_dir(
            Path("experiments/lorenz/models/budget_based_dt_08_T10/fixed")
        )
        == 0.08
    )


def test_get_dt_from_model_dir_rejects_path_without_dt_segment():
    with pytest.raises(ValueError, match="Could not infer dt"):
        get_dt_from_model_dir(Path("experiments/lorenz/models/latest/fixed"))


def test_cross_validate_models_falls_back_to_requested_system(monkeypatch):
    seen = {}

    class DummyModel:
        def to(self, device):
            return self

    monkeypatch.setattr(
        cross_validation,
        "load_model",
        lambda path: (
            DummyModel(),
            {
                "seed": 1,
                "metadata": {"wall_time_seconds": 0.0},
            },
        ),
    )
    monkeypatch.setattr(cross_validation, "validation_loss", lambda *args: 0.0)

    def fake_make_eval_loader(max_val_T, dt, normalization_stats, system_name):
        seen["system_name"] = system_name
        return object()

    monkeypatch.setattr(cross_validation, "make_eval_loader", fake_make_eval_loader)

    cross_validation.cross_validate_models(
        fixed_paths={1: [Path("mlp_T1_seed1.pt")]},
        adaptive_paths=[],
        dt=0.08,
        val_Ts=[1],
        system_name="rossler",
    )

    assert seen["system_name"] == "rossler"
