import adaptive_horizon.config as config


def stepped_scheduler(
    epoch: int,
    T_max: int = config.MAX_TRAIN_T,
    T_min: int = 1,
    T_step: int = 2,
    epochs_per_step: int = 10,
) -> int:
    return min(T_min + epoch // epochs_per_step * T_step, T_max)
