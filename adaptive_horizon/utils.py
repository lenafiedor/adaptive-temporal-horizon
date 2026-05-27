from typing import Optional

ADAPTIVE_METHODS = {
    "adaptive-horizon": "ah",
    "weighted-loss": "wl",
    "curriculum-horizon": "ch",
    "gradient-scaling-horizon": "gs",
}


def adaptive_method_abbreviation(method: Optional[str]) -> Optional[str]:
    """Return a short filename-safe abbreviation for an adaptive method."""
    if method is None:
        return None
    if method not in ADAPTIVE_METHODS:
        raise ValueError(f"Unknown adaptive method: {method}")
    return ADAPTIVE_METHODS[method]


def time_to_steps(time_value: float, dt: float, min_steps: int = 1) -> int:
    """Convert physical time to an integer number of simulation steps."""
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if min_steps < 0:
        raise ValueError(f"min_steps must be non-negative, got {min_steps}")

    return max(min_steps, int(round(time_value / dt)))


def format_dt(dt: float) -> str:
    """Format a dt value for filenames."""
    return str(dt).split(".")[1]
