from typing import Optional


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


def resolve_T_val(T_val: Optional[int], tau: Optional[float], dt: float) -> int:
    """Resolve an explicit horizon or physical time horizon to step count."""
    if T_val is not None:
        if T_val < 1:
            raise ValueError(f"T_val must be at least 1, got {T_val}")
        return int(T_val)

    if tau is None:
        raise ValueError("Provide either T_val or tau")

    return time_to_steps(tau, dt)
