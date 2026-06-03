import numpy as np

import adaptive_horizon.config as config


def scheduled_sampling_probability(epoch: int, epochs: int) -> float:
    """Linear decay teacher forcing from 1 to 0 over the training run."""
    if epochs <= 1:
        return 0.0
    return max(0.0, 1.0 - epoch / float(epochs - 1))


def curriculum_horizon(
    epoch: int,
    epochs: int,
    T_max: int,
) -> int:
    """Split training epochs as evenly as possible across T=1..T_max."""
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    if T_max < 1:
        raise ValueError(f"T_max must be at least 1, got {T_max}")

    return min(T_max, (epoch * T_max) // epochs + 1)


def summarize_gradient_scaling(g_values):
    """Summarize per-batch g(T) values with robust statistics."""
    summary = {}
    for T, values in g_values.items():
        values = np.asarray(values, dtype=np.float64)
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
