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


def get_method_short(method: str | None) -> str:
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


def filter_adaptive_model_paths(adaptive_paths, adaptive_method: str | None):
    if adaptive_method is None:
        return adaptive_paths

    return [
        model_path
        for model_path in adaptive_paths
        if get_adaptive_method(model_path) == adaptive_method
    ]


def resolve_adaptive_method(
    evaluation_records=None, adaptive_method: str | None = None
) -> str | None:
    if adaptive_method is not None:
        return adaptive_method

    if evaluation_records is None:
        return None

    adaptive_methods = {
        record["adaptive_method"]
        for record in evaluation_records
        if record.get("model_type") == "adaptive" and record.get("adaptive_method")
    }
    if len(adaptive_methods) == 1:
        return next(iter(adaptive_methods))

    return None


def get_evaluation_method_abbreviation(
    evaluation_records=None, adaptive_method: str | None = None
) -> str:
    resolved_method = resolve_adaptive_method(evaluation_records, adaptive_method)
    if resolved_method is not None:
        return get_method_short(resolved_method)

    has_adaptive = bool(
        evaluation_records
        and any(record.get("model_type") == "adaptive" for record in evaluation_records)
    )
    return "mixed" if has_adaptive else "fixed"
