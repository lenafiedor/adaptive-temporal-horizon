def format_dt(dt: float) -> str:
    """Format a dt value for filenames."""
    return str(dt).split(".")[1]
