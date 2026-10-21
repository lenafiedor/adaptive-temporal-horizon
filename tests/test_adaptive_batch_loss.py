import torch

from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.training.loss import adaptive_batch_loss, rollout_predictions


def reference_adaptive_batch_loss(model, inputs, targets, T):
    preds = rollout_predictions(model, inputs, int(T.max().item()))
    losses = [
        torch.nn.functional.mse_loss(preds[i, :t], targets[i, :t])
        for i, t in enumerate(T.long().tolist())
    ]
    return sum(losses) / len(losses)


def test_adaptive_batch_loss_matches_per_sample_loop():
    torch.manual_seed(0)
    model = MLP(
        MLPConfig(3, 3, [10, 10, 10], residual_connections=True, k=2), random_seed=0
    )
    inputs = torch.randn(64, 3)
    T = torch.randint(1, 8, (64,)).float()
    targets = torch.randn(64, int(T.max().item()), 3)
    for i, t in enumerate(T.long().tolist()):
        targets[i, t:] = 0.0

    expected = reference_adaptive_batch_loss(model, inputs, targets, T)
    actual = adaptive_batch_loss(model, inputs, targets, T)
    torch.testing.assert_close(actual, expected)

    expected_grads = torch.autograd.grad(expected, list(model.parameters()))
    actual_grads = torch.autograd.grad(actual, list(model.parameters()))
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e)
