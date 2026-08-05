import pytest

from adaptive_horizon.training import train as train_module
from adaptive_horizon.training.methods import CURRICULUM_HORIZON


def test_curriculum_boundary_reached_for_linear_schedule():
    assert not train_module.curriculum_boundary_reached(8, 20, 1, 2)
    assert train_module.curriculum_boundary_reached(9, 20, 1, 2)
    assert train_module.curriculum_boundary_reached(19, 20, 2, 2)


def test_cross_validation_median_loss(monkeypatch):
    monkeypatch.setattr(
        train_module,
        "validation_loss",
        lambda _model, _loader, val_T, _device: float(val_T),
    )

    median_loss, losses = train_module.cross_validation_median_loss(
        model=None,
        val_loader=None,
        val_Ts=[1, 2, 3],
        device="cpu",
    )

    assert median_loss == 2.0
    assert losses == {1: 1.0, 2: 2.0, 3: 3.0}


def test_curriculum_stopping_methods_are_mutually_exclusive():
    with pytest.raises(ValueError, match="only one"):
        train_module.train(
            model=None,
            train_loader=None,
            val_loader=None,
            optimizer=None,
            epochs=0,
            T=2,
            adaptive=True,
            adaptive_method=CURRICULUM_HORIZON,
            early_stopping=True,
            cross_validation_early_stopping=True,
        )
