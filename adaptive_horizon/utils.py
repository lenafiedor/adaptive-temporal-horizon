from datetime import datetime


def format_dt(dt: float) -> str:
    """Format a dt value for filenames."""
    return str(dt).split(".")[1]


def timestamp() -> str:
    """Local-time timestamp used in generated file names."""
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
