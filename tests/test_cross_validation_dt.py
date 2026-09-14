from pathlib import Path
from unittest.mock import Mock

from adaptive_horizon.evaluation.utils import summarize_cross_validation
from adaptive_horizon.evaluation import cross_validation


def test_cross_validate_models_falls_back_to_requested_system(monkeypatch):
    model = Mock()
    model.to.return_value = model

    monkeypatch.setattr(
        cross_validation,
        "load_model",
        Mock(
            return_value=(
                model,
                {
                    "seed": 1,
                    "metadata": {"wall_time_seconds": 0.0},
                },
            )
        ),
    )
    monkeypatch.setattr(cross_validation, "validation_loss", Mock(return_value=0.0))

    make_eval_loader = Mock(return_value=object())
    monkeypatch.setattr(cross_validation, "make_eval_loader", make_eval_loader)

    cross_validation.cross_validate_models(
        fixed_paths={1: [Path("mlp_T1_seed1.pt")]},
        adaptive_paths=[],
        dt=0.08,
        val_Ts=[1],
        system_name="rossler",
    )

    assert make_eval_loader.call_args.kwargs["system_name"] == "rossler"


def test_summarize_cross_validation_allows_fixed_only_records():
    summary = summarize_cross_validation(
        evaluation_records=[
            {
                "model_type": "fixed",
                "seed": 0,
                "train_T": 1,
                "val_T": 1,
                "mse": 0.1,
            }
        ],
        train_Ts=[1],
        val_Ts=[1],
    )

    assert summary["adaptive"] is None
    assert "deltas" not in summary
    assert summary["fixed"][0]["overall"]["mean"] == 0.1
