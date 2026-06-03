from typing import Optional

ADAPTIVE_METHODS = {
    "adaptive-horizon": "ah",
    "weighted-loss": "wl",
    "curriculum-horizon": "ch",
    "gradient-scaling-horizon": "gs",
}

ADAPTIVE_HORIZON, WEIGHTED_LOSS, CURRICULUM_HORIZON, GRADIENT_SCALING_HORIZON = (
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
