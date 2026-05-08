import re
from pathlib import Path


ADAPTIVE_HORIZON = "adaptive-horizon"
WEIGHTED_LOSS = "weighted-loss"
MIXED = "mixed"
FIXED = "fixed"

TRAINING_MODES = {
    ADAPTIVE_HORIZON: "ah",
    WEIGHTED_LOSS: "wl",
    MIXED: "mixed",
    FIXED: "fixed",
}


def get_adaptive_method(model_path: str | Path) -> str:
    """Infer the adaptive method from the model filename."""
    model_path = Path(model_path)
    match = re.search(r"adaptive_mlp_([a-z]+)_.+$", model_path.name)
    if not match:
        raise ValueError(f"Unsupported adaptive model filename: {model_path.name}")
    abbreviation = match.group(1)

    try:
        return next(k for k, v in TRAINING_MODES.items() if v == abbreviation)
    except KeyError as exc:
        raise ValueError(
            f"Unsupported adaptive method abbreviation '{abbreviation}' in {model_path.name}"
        ) from exc
