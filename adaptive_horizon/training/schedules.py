import numpy as np

import adaptive_horizon.config as config


def curriculum_horizon(
    epoch: int,
    val_losses: list[float],
    current_T: int,
    T_max: int = config.MAX_TRAIN_T,
    loss_threshold: float = config.CURRICULUM_LOSS_THRESHOLD,
    update_frequency: int = config.CURRICULUM_UPDATE_FREQUENCY,
) -> tuple[int, float | None]:
    if update_frequency <= 0:
        raise ValueError("update_frequency must be positive")

    if (epoch + 1) % update_frequency != 0:
        return current_T, None

    recent_val_losses = val_losses[-update_frequency:]
    if len(recent_val_losses) < update_frequency:
        return current_T, None

    mean_val_loss = float(np.mean(recent_val_losses))
    if mean_val_loss <= loss_threshold and current_T < T_max:
        return current_T + 1, mean_val_loss

    return current_T, mean_val_loss
