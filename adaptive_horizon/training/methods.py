from typing import Optional

ADAPTIVE_METHODS = {
    "lyapunov-based": "lb",
    "weighted-loss": "wl",
    "linear-scheduler": "ls",
    "early-stopping": "es",
    "cross-validation": "cv",
}

LYAPUNOV_BASED, WEIGHTED_LOSS, LINEAR_SCHEDULER, EARLY_STOPPING, CROSS_VALIDATION = (
    ADAPTIVE_METHODS.keys()
)
ADAPTIVE_METHOD_CHOICES = list(ADAPTIVE_METHODS)


def adaptive_method_abbreviation(method: Optional[str]) -> Optional[str]:
    """Return a short filename-safe abbreviation for an adaptive method."""
    if method is None:
        return None
    if method not in ADAPTIVE_METHODS:
        raise ValueError(f"Unknown adaptive method: {method}")
    return ADAPTIVE_METHODS[method]
