from typing import ClassVar

import torch
from torch.utils.data import DataLoader, TensorDataset

from adaptive_horizon.evaluation.metrics import evaluate_metrics
from adaptive_horizon.training.loss import validation_loss


class ShiftModel(torch.nn.Module):
    """Adds a constant each step, so the rollout error grows linearly."""

    def __init__(self, dim, shift):
        super().__init__()
        self.embedding = torch.nn.Linear(dim, dim)
        self.shift = shift

    def forward(self, x):
        return x + self.shift


class Dataset(TensorDataset):
    normalization_stats: ClassVar[dict] = {"mean": [0.0] * 3, "std": [1.0] * 3}


def make_loader(n_samples=6, max_T=5, dim=3):
    inputs = torch.ones(n_samples, dim)
    targets = torch.ones(n_samples, max_T, dim)
    return DataLoader(Dataset(inputs, targets), batch_size=4)


def test_vpt_counts_steps_before_first_nrmse_crossing():
    # Step t has normalized error 0.3 * t in every coordinate, so NRMSE = 0.3 * t.
    results = evaluate_metrics(ShiftModel(3, 0.3), make_loader(), [2, 5], [1.0], "cpu")

    vpt = results[5]["vpt"]["1.0"]
    assert vpt["median"] == 3  # 0.9 is valid, 1.2 exceeds 1.0
    assert vpt["censored_fraction"] == 0
    assert "vpt" not in results[2]


def test_vpt_is_censored_at_longest_prefix():
    results = evaluate_metrics(ShiftModel(3, 0.3), make_loader(), [2], [1.0], "cpu")

    assert results[2]["vpt"]["1.0"] == {"median": 2, "censored_fraction": 1}


def test_mse_matches_validation_loss():
    loader = make_loader()
    model = ShiftModel(3, 0.3)
    results = evaluate_metrics(model, loader, [1, 4], device="cpu")

    for T in (1, 4):
        expected = validation_loss(
            model,
            [(inputs, targets[:, :T]) for inputs, targets in loader],
            T,
            "cpu",
        )
        assert abs(results[T]["mse"] - expected) < 1e-6
