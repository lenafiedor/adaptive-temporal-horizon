import torch

from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.training.loss import (
    batch_loss,
    error_truncated_batch_loss,
    rollout_predictions,
)


def _model():
    return MLP(
        MLPConfig(3, 3, [10, 10, 10], residual_connections=True, k=2), random_seed=0
    )


def reference_loss(model, inputs, targets, tolerance):
    preds = rollout_predictions(model, inputs, targets.shape[1])
    losses, horizons = [], []
    for i in range(len(inputs)):
        step_mse = (preds[i] - targets[i]).square().mean(dim=1)
        exceeded = (step_mse.sqrt() > tolerance).nonzero()
        t = int(exceeded[0]) + 1 if len(exceeded) else targets.shape[1]
        losses.append(step_mse[:t].mean())
        horizons.append(t)
    return torch.stack(losses).mean(), sum(horizons) / len(horizons)


def test_matches_per_sample_truncation_and_gradients():
    torch.manual_seed(0)
    model = _model()
    inputs = torch.randn(64, 3)
    targets = torch.randn(64, 8, 3)
    expected, expected_T = reference_loss(model, inputs, targets, tolerance=1.2)
    actual, actual_T = error_truncated_batch_loss(
        model, inputs, targets, tolerance=1.2, return_horizon=True
    )
    torch.testing.assert_close(actual, expected)
    assert abs(actual_T - expected_T) < 1e-6
    assert 1 < actual_T < 8

    params = list(model.parameters())
    for a, e in zip(
        torch.autograd.grad(actual, params), torch.autograd.grad(expected, params)
    ):
        torch.testing.assert_close(a, e)


def test_limits_match_fixed_horizons():
    torch.manual_seed(1)
    model = _model()
    inputs = torch.randn(32, 3)
    targets = torch.randn(32, 6, 3)
    torch.testing.assert_close(
        error_truncated_batch_loss(model, inputs, targets, tolerance=1e9),
        batch_loss(model, inputs, targets, 6),
    )
    torch.testing.assert_close(
        error_truncated_batch_loss(model, inputs, targets, tolerance=1e-9),
        batch_loss(model, inputs, targets[:, :1], 1),
    )


def test_growth_threshold_is_relative_to_median_one_step_error():
    torch.manual_seed(2)
    model = _model()
    inputs = torch.randn(64, 3)
    targets = torch.randn(64, 8, 3)
    one_step = (model(inputs) - targets[:, 0]).square().mean(dim=1).sqrt()
    tolerance = 1.5 * one_step.median().item()

    expected, expected_T = error_truncated_batch_loss(
        model, inputs, targets, tolerance=tolerance, return_horizon=True
    )
    actual, actual_T = error_truncated_batch_loss(
        model, inputs, targets, tolerance=123.0, return_horizon=True, growth=1.5
    )
    torch.testing.assert_close(actual, expected)
    assert actual_T == expected_T
    assert 1 < actual_T < 8
