import adaptive_horizon.config as config


def linear_scheduler(
    epoch: int,
    total_epochs: int,
    T_max: int = config.MAX_TRAIN_T,
    T_min: int = 1,
) -> int:
    num_horizons = T_max - T_min + 1
    epochs_per_horizon = max(1, total_epochs // num_horizons)

    return T_min + min(epoch // epochs_per_horizon, num_horizons - 1)


def proportional_horizon_epochs(
    total_epochs: int,
    T_max: int = config.MAX_TRAIN_T,
    T_min: int = 1,
) -> dict[int, int]:
    horizons = list(range(T_min, T_max + 1))
    if total_epochs < len(horizons):
        raise ValueError("total_epochs must allow at least one epoch per horizon")

    total_weight = sum(horizons)
    allocations = {
        T: total_epochs * T // total_weight
        for T in horizons
    }
    remaining = total_epochs - sum(allocations.values())
    remainders = sorted(
        horizons,
        key=lambda T: total_epochs * T % total_weight,
        reverse=True,
    )
    for T in remainders[:remaining]:
        allocations[T] += 1

    for T in horizons:
        if allocations[T] == 0:
            donor = max(horizons, key=allocations.get)
            allocations[donor] -= 1
            allocations[T] = 1

    return allocations


def proportional_scheduler(
    epoch: int,
    total_epochs: int,
    T_max: int = config.MAX_TRAIN_T,
    T_min: int = 1,
) -> int:
    epochs_by_T = proportional_horizon_epochs(total_epochs, T_max, T_min)
    boundary = 0
    for T, horizon_epochs in epochs_by_T.items():
        boundary += horizon_epochs
        if epoch < boundary:
            return T
    return T_max
