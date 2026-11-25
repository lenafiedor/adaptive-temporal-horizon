from datetime import datetime


def format_dt(dt: float) -> str:
    """Format a dt value for filenames: 0.08 -> "08", 1.0 -> "1_0", 1.5 -> "1_5"."""
    whole, fraction = f"{float(dt)}".split(".")
    return fraction if whole == "0" else f"{whole}_{fraction}"


def timestamp() -> str:
    """Local-time timestamp used in generated file names."""
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
