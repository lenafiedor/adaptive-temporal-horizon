import re
from pathlib import Path


ADAPTIVE_HORIZON_METHOD = "adaptive-horizon"
WEIGHTED_LOSS_METHOD = "weighted-loss"

ADAPTIVE_METHOD_ABBREVIATIONS = {
    ADAPTIVE_HORIZON_METHOD: "ah",
    WEIGHTED_LOSS_METHOD: "wl",
}

ADAPTIVE_METHODS_BY_ABBREVIATION = {
    abbreviation: method
    for method, abbreviation in ADAPTIVE_METHOD_ABBREVIATIONS.items()
}


def get_adaptive_method_abbreviation(method: str | None) -> str:
    if method is None:
        raise ValueError("Adaptive method is required when saving adaptive outputs")

    try:
        return ADAPTIVE_METHOD_ABBREVIATIONS[method]
    except KeyError as exc:
        raise ValueError(f"Unsupported adaptive method: {method}") from exc


def get_adaptive_method(model_path: str | Path) -> str:
    model_path = Path(model_path)
    match = re.search(r"adaptive_mlp(?:_([a-z]+))?_seed\d+", model_path.name)
    if not match:
        raise ValueError(f"Unsupported adaptive model filename: {model_path.name}")

    abbreviation = match.group(1)
    if abbreviation is None:
        return ADAPTIVE_HORIZON_METHOD

    try:
        return ADAPTIVE_METHODS_BY_ABBREVIATION[abbreviation]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported adaptive method abbreviation '{abbreviation}' in {model_path.name}"
        ) from exc
