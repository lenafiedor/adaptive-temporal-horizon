import numpy as np
import pytest
import torch

from adaptive_horizon import config
from adaptive_horizon.data.santafe import SANTAFE
from adaptive_horizon.evaluation.metrics import evaluate_metrics
from adaptive_horizon.training.methods import (
    ERROR_TRUNCATED,
    LYAPUNOV_MEAN,
    WEIGHTED_LOSS,
)
from adaptive_horizon.training.setup import create_model_and_loaders
from adaptive_horizon.utils import format_dt


@pytest.fixture
def recording(tmp_path, monkeypatch):
    t = np.arange(1500)
    rng = np.random.default_rng(0)
    signal = np.sin(0.3 * t) * (1 + 0.5 * np.sin(0.011 * t)) + 0.05 * rng.normal(
        size=len(t)
    )
    path = tmp_path / "laser.dat"
    np.savetxt(path, signal)
    monkeypatch.setattr(config, "SANTAFE_DATA_PATH", path)
    return path


@pytest.mark.parametrize(
    "method", [None, LYAPUNOV_MEAN, WEIGHTED_LOSS, ERROR_TRUNCATED]
)
def test_santafe_setup_builds_delay_window_model(recording, method):
    model, train_loader, _, _, mlp_config, metadata = create_model_and_loaders(
        seed=0,
        adaptive=method is not None,
        device="cpu",
        dt=1.0,
        T=3,
        adaptive_method=method,
        system_name=SANTAFE,
    )
    history = metadata["observed_data"]["history_length"]
    assert mlp_config.delay_window and mlp_config.output_size == 1
    inputs, *_ = next(iter(train_loader))
    assert inputs.shape[1] == history
    prediction = model(inputs)
    # The model shifts the history and appends one new observation.
    assert torch.equal(prediction[:, :-1], inputs[:, 1:])
    assert metadata["system"] == SANTAFE
    if method == LYAPUNOV_MEAN:
        adaptive = metadata["adaptive"]
        assert adaptive["lambda_estimate"] == pytest.approx(
            metadata["observed_data"]["score_stats"]["mean"]
        )
        assert 1 <= adaptive["min_T"] <= adaptive["max_T"] <= 3


def test_santafe_metrics_score_only_new_observation(recording):
    model, _, val_loader, *_ = create_model_and_loaders(
        seed=0, adaptive=False, device="cpu", dt=1.0, T=2, system_name=SANTAFE
    )
    results = evaluate_metrics(model, val_loader, [1, 2], device="cpu")
    std = val_loader.dataset.normalization_stats["std"][-1]
    assert results[1]["physical_mse"] == pytest.approx(
        results[1]["mse"] * (std + 1e-8) ** 2, rel=1e-4
    )


def test_format_dt_keeps_small_steps_and_distinguishes_unit_step():
    assert format_dt(0.08) == "08"
    assert format_dt(0.1) == "1"
    assert format_dt(1.0) == "1_0"
