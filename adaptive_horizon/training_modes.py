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


def get_mode_abbreviation(mode: str | None) -> str | None:
    if mode is None:
        return None
    try:
        return TRAINING_MODES[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode: {mode}") from exc


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


def resolve_adaptive_method(evaluation_records, adaptive_method: str) -> str:
    """Resolve adaptive method from evaluation records or directly from name."""
    if adaptive_method is not None:
        return adaptive_method

    adaptive_methods = {
        record["adaptive_method"]
        for record in evaluation_records
        if record.get("model_type") == "adaptive" and record.get("adaptive_method")
    }
    if len(adaptive_methods) > 1:
        return MIXED

    return FIXED
