ADAPTIVE_METHODS = {
    "lyapunov-mean": "lm",
    "weighted-loss": "wl",
    "linear-scheduler": "ls",
    "cross-validation": "cv",
    "error-truncated": "et",
}

(
    LYAPUNOV_MEAN,
    WEIGHTED_LOSS,
    LINEAR_SCHEDULER,
    CROSS_VALIDATION,
    ERROR_TRUNCATED,
) = ADAPTIVE_METHODS.keys()
ADAPTIVE_METHOD_CHOICES = list(ADAPTIVE_METHODS)

LYAPUNOV_METHODS = (LYAPUNOV_MEAN,)
SCHEDULED_METHODS = (LINEAR_SCHEDULER, CROSS_VALIDATION)


def takes_max_T(method: str | None) -> bool:
    return method in (*SCHEDULED_METHODS, ERROR_TRUNCATED)


def adaptive_method_abbreviation(
    method: str | None, random_horizons: bool = False, invert_horizons: bool = False
) -> str | None:
    if method is None:
        return None
    if method not in ADAPTIVE_METHODS:
        raise ValueError(f"Unknown adaptive method: {method}")
    # "lmir" permutes the inverted horizons: the matched control for "lmi".
    suffix = ("i" if invert_horizons else "") + ("r" if random_horizons else "")
    return ADAPTIVE_METHODS[method] + suffix
