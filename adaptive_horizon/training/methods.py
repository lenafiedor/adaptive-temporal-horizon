ADAPTIVE_METHODS = {
    "lyapunov-mean": "lm",
    "weighted-loss": "wl",
    "linear-scheduler": "ls",
    "early-stopping": "es",
    "cross-validation": "cv",
}

(
    LYAPUNOV_MEAN,
    WEIGHTED_LOSS,
    LINEAR_SCHEDULER,
    EARLY_STOPPING,
    CROSS_VALIDATION,
) = ADAPTIVE_METHODS.keys()
ADAPTIVE_METHOD_CHOICES = list(ADAPTIVE_METHODS)

LYAPUNOV_METHODS = (LYAPUNOV_MEAN,)
SCHEDULED_METHODS = (LINEAR_SCHEDULER, EARLY_STOPPING, CROSS_VALIDATION)


def takes_max_T(method: str | None) -> bool:
    return method in SCHEDULED_METHODS


def adaptive_method_abbreviation(method: str | None) -> str | None:
    """Return a short filename-safe abbreviation for an adaptive method."""
    if method is None:
        return None
    if method not in ADAPTIVE_METHODS:
        raise ValueError(f"Unknown adaptive method: {method}")
    return ADAPTIVE_METHODS[method]
