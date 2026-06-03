def time_to_steps(time_value: float, dt: float, min_steps: int = 1) -> int:
    """Convert physical time to an integer number of simulation steps."""
    return max(min_steps, int(round(time_value / dt)))


def format_dt(dt: float) -> str:
    """Format a dt value for filenames."""
    return str(dt).split(".")[1]
