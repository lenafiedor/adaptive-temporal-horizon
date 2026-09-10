from typing import Optional

ADAPTIVE_METHODS = {
    "lyapunov-based": "lb",
    "weighted-loss": "wl",
    "linear-scheduler": "ls",
}

LYAPUNOV_BASED, WEIGHTED_LOSS, LINEAR_SCHEDULER = ADAPTIVE_METHODS.keys()
ADAPTIVE_METHOD_CHOICES = list(ADAPTIVE_METHODS)


def adaptive_method_abbreviation(method: Optional[str]) -> Optional[str]:
    """Return a short filename-safe abbreviation for an adaptive method."""
    if method is None:
        return None
    if method not in ADAPTIVE_METHODS:
        raise ValueError(f"Unknown adaptive method: {method}")
    return ADAPTIVE_METHODS[method]
