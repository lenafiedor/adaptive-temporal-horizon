"""Rollout errors and valid prediction times (VPT) of trajectory forecasts."""

import math
from typing import Any

import torch

from adaptive_horizon import config
from adaptive_horizon.training.loss import rollout_predictions

ERROR_METRICS = ("mse", "nrmse", "smape")
DEFAULT_VPT_THRESHOLDS = (0.1, 0.3, 0.5, 1.0)


def evaluate_metrics(
    model,
    loader,
    val_Ts,
    vpt_thresholds=DEFAULT_VPT_THRESHOLDS,
    device=config.DEVICE,
):
    """Evaluate each prefix of one max(val_Ts)-step rollout per batch.

    Errors per step:
    - MSE keeps the training loss's normalized coordinates and batch averaging.
    - NRMSE is the per-sample RMSE in normalized coordinates, i.e., scaled by
      each coordinate's standard deviation over the attractor.
    - SMAPE (0-200%) uses physical coordinates, for comparison with Horizon Forcing.

    A trajectory's VPT is the number of steps before its NRMSE first exceeds
    a threshold; rollouts without a crossing are censored at ``max(val_Ts)``.
    VPT is only reported for the longest prefix, where it is least censored.
    """
    max_T = int(max(val_Ts))
    stats = loader.dataset.normalization_stats
    std = torch.as_tensor(stats["std"], device=device) + 1e-8
    mean = torch.as_tensor(stats["mean"], device=device)
    mse_sum = torch.zeros(max_T, dtype=torch.float64)
    nrmse_sum = torch.zeros(max_T, dtype=torch.float64)
    smape_sum = torch.zeros(max_T, dtype=torch.float64)
    nrmse_rows = []
    n_batches = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets[:, :max_T].to(device)
            predictions = rollout_predictions(model, inputs, max_T)
            # Diverged/non-finite forecasts must never appear valid again.
            squared = torch.nan_to_num(
                (predictions - targets).square().mean(dim=2), nan=math.inf
            )
            nrmse = squared.sqrt()
            predictions, targets = predictions * std + mean, targets * std + mean
            smape = torch.nan_to_num(
                200
                * (predictions - targets).abs()
                / (predictions.abs() + targets.abs()).clamp_min(1e-8),
                nan=200.0,
            ).mean(dim=2)
            mse_sum += squared.mean(dim=0).double().cpu()
            nrmse_sum += nrmse.sum(dim=0).double().cpu()
            smape_sum += smape.sum(dim=0).double().cpu()
            nrmse_rows.append(nrmse.cpu())
            n_batches += 1

    if n_batches == 0:
        raise ValueError("The evaluation loader is empty")
    nrmse = torch.cat(nrmse_rows)
    curves = {
        "mse": mse_sum / n_batches,
        "nrmse": nrmse_sum / len(nrmse),
        "smape": smape_sum / len(nrmse),
    }
    results: dict[int, dict[str, Any]] = {
        T: {name: curve[:T].mean().item() for name, curve in curves.items()}
        for T in val_Ts
    }
    vpt = {}
    for threshold in vpt_thresholds:
        invalid = nrmse > threshold
        steps = torch.where(invalid.any(dim=1), invalid.int().argmax(dim=1), max_T)
        vpt[str(threshold)] = {
            "median": steps.double().quantile(0.5).item(),
            "censored_fraction": (steps == max_T).double().mean().item(),
        }
    results[max_T]["vpt"] = vpt
    return results
