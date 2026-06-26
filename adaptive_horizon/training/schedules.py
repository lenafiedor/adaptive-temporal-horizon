import adaptive_horizon.config as config


def curriculum_horizon(
    epoch: int,
    total_epochs: int,
    T_max: int = config.MAX_TRAIN_T,
    T_min: int = 1,
) -> int:
    """Compute a deterministic linear curriculum horizon.

    Each horizon receives the same number of epochs when total_epochs is
    divisible by the number of horizons. Any remainder is assigned to T_max.
    """
    num_horizons = T_max - T_min + 1
    epochs_per_horizon = max(1, total_epochs // num_horizons)

    return T_min + min(epoch // epochs_per_horizon, num_horizons - 1)
