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


def test_observed_fsle_recovers_exponential_growth_rate():
    from adaptive_horizon.dynamics.fsle import observed_fsle

    rate, steps = 0.1, np.arange(301)[:, None]
    initial = np.random.default_rng(0).uniform(0.01, 0.02, size=500)
    # Exponential growth that saturates at 1, like a chaotic separation.
    curves = np.minimum(initial * np.exp(rate * steps), 1.0)
    thresholds = 0.03 * np.sqrt(2) ** np.arange(10)
    lam, valid = observed_fsle(curves, thresholds)
    assert valid[0] == 500
    finite = lam[np.isfinite(lam)]
    assert finite == pytest.approx(rate, rel=0.15)


def test_tolerance_from_fsle_uses_recording(recording, tmp_path, monkeypatch):
    from adaptive_horizon.dynamics.fsle import observed_fsle_tolerance

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    info = observed_fsle_tolerance(recording, n=200, t_max=100)
    assert info["source"] in ("fsle_knee", "fsle_fallback")
    assert info["tolerance"] > 0 and info["flops"] > 0
    assert SANTAFE in info["cache"]
    # Second call reads the per-checksum cache.
    assert observed_fsle_tolerance(recording, n=200, t_max=100) == info
