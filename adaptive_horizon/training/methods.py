from typing import Optional

ADAPTIVE_HORIZON = "adaptive-horizon"
WEIGHTED_LOSS = "weighted-loss"
CURRICULUM_HORIZON = "curriculum-horizon"
GRADIENT_SCALING_HORIZON = "gradient-scaling-horizon"

ADAPTIVE_METHOD_CHOICES = [
    ADAPTIVE_HORIZON,
    WEIGHTED_LOSS,
    CURRICULUM_HORIZON,
    GRADIENT_SCALING_HORIZON,
]

ADAPTIVE_METHODS = {
    ADAPTIVE_HORIZON: "ah",
    WEIGHTED_LOSS: "wl",
    CURRICULUM_HORIZON: "ch",
    GRADIENT_SCALING_HORIZON: "gs",
}


def adaptive_method_abbreviation(method: Optional[str]) -> Optional[str]:
    """Return a short filename-safe abbreviation for an adaptive method."""
    if method is None:
        return None
    if method not in ADAPTIVE_METHODS:
        raise ValueError(f"Unknown adaptive method: {method}")
    return ADAPTIVE_METHODS[method]
