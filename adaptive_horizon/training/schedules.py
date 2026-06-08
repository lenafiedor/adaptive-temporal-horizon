import numpy as np

import adaptive_horizon.config as config


def curriculum_horizon(
    epoch: int,
    num_epochs: int,
    val_loss: float,
    current_T: int,
    success_count: int,
    T_max: int = config.MAX_TRAIN_T,
    loss_threshold: float = 0.01,
    patience: int = 4,
) -> tuple[int, int]:
    if epoch < int(num_epochs / 10):
        return current_T, 0

    if val_loss <= loss_threshold:
        success_count += 1
    elif val_loss >= loss_threshold * 5:
        success_count = 0
    else:
        success_count = 0

    if success_count >= patience and current_T < T_max:
        return current_T + 1, 0

    return current_T, success_count


def summarize_gradient_scaling(g_values):
    """Summarize per-batch g(T) values with robust statistics."""
    summary = {}
    for T, values in g_values.items():
        if values.size == 0:
            summary[int(T)] = {"median": float("inf"), "p90": float("inf")}
            continue
        summary[int(T)] = {
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90.0)),
        }
    return summary


def select_gradient_scaling_horizon(
    current_T: int,
    T_max: int,
    g_summary: dict,
    median_threshold: float = config.GRADIENT_SCALING_MEDIAN_THRESHOLD,
    p90_threshold: float = config.GRADIENT_SCALING_P90_THRESHOLD,
) -> tuple[int, int]:
    """Select the next horizon from robust gradient-scaling statistics."""
    safe_horizons = [
        T
        for T in range(1, T_max + 1)
        if g_summary.get(T, {}).get("median", float("inf")) <= median_threshold
        and g_summary.get(T, {}).get("p90", float("inf")) <= p90_threshold
    ]
    safe_T = max(safe_horizons) if safe_horizons else 1

    if safe_T < current_T:
        next_T = safe_T
    elif safe_T > current_T:
        next_T = current_T + 1
    else:
        next_T = current_T

    return max(1, min(T_max, next_T)), safe_T
