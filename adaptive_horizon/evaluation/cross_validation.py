import argparse
import json
from pathlib import Path

import numpy as np

from adaptive_horizon import config
from adaptive_horizon.data.dataset import make_val_loader
from adaptive_horizon.dynamics.lyapunov import compute_global_lyapunov
from adaptive_horizon.dynamics.systems import SYSTEM_CHOICES
from adaptive_horizon.evaluation.metrics import (
    DEFAULT_VPT_THRESHOLDS,
    ERROR_METRICS,
    evaluate_metrics,
)
from adaptive_horizon.evaluation.utils import (
    get_checkpoint_normalization_stats,
    get_dt_from_model_dir,
    load_model,
    save_cross_validation_results,
    summarize_cross_validation,
)
from adaptive_horizon.training.utils import find_fixed_models
from adaptive_horizon.visualization.plotting import plot_mse, plot_vpt


def get_fixed_paths(
    train_Ts: list[int] | None = None,
    model_dir: Path = config.MODEL_DIR,
) -> dict[int, list[Path]]:
    model_paths: dict[int, list[Path]] = {}
    for model_path, train_T, _ in find_fixed_models(model_dir):
        if train_Ts is None or train_T in train_Ts:
            model_paths.setdefault(train_T, []).append(model_path)

    if train_Ts is None:
        return dict(sorted(model_paths.items()))
    return {train_T: model_paths.get(train_T, []) for train_T in train_Ts}


def get_adaptive_paths(model_dir: Path = config.MODEL_DIR) -> list[Path]:
    return sorted(model_dir.glob("adaptive_mlp*.pt"))


def eval_loader_cache_key(normalization_stats):
    if normalization_stats is None:
        return None
    return tuple(
        tuple(float(value) for value in normalization_stats[name])
        for name in ("mean", "std")
    )


def cross_validate_models(
    fixed_paths: dict[int, list[Path]],
    adaptive_paths: list[Path],
    dt=config.DT,
    device=config.DEVICE,
    val_Ts: list[int] | None = None,
    system_name: str = config.DEFAULT_SYSTEM,
    vpt_thresholds=DEFAULT_VPT_THRESHOLDS,
):
    train_Ts = list(fixed_paths)
    val_Ts = list(val_Ts) if val_Ts is not None else train_Ts
    eval_loaders = {}

    def get_eval_loader(checkpoint):
        normalization_stats = get_checkpoint_normalization_stats(checkpoint)
        checkpoint_system = checkpoint.get("metadata", {}).get("system") or system_name
        key = (checkpoint_system, eval_loader_cache_key(normalization_stats))
        if key not in eval_loaders:
            eval_loaders[key] = make_val_loader(
                checkpoint_system, dt, max(val_Ts), normalization_stats
            )
        return eval_loaders[key]

    def evaluate(model_path, model_type, train_T, model_val_Ts):
        model, checkpoint = load_model(model_path)
        model = model.to(device)
        eval_loader = get_eval_loader(checkpoint)
        metrics = evaluate_metrics(
            model, eval_loader, model_val_Ts, vpt_thresholds, device
        )
        records = [
            {
                "model_type": model_type,
                "seed": checkpoint.get("seed"),
                "train_T": train_T,
                "val_T": val_T,
                **metrics[val_T],
            }
            for val_T in model_val_Ts
        ]
        print(
            f"  Model {model_path.name}: mean MSE = "
            f"{np.mean([record['mse'] for record in records]):.6f}"
        )
        return records

    evaluation_records = []
    if any(fixed_paths.values()):
        print("\nEvaluating fixed models")
        for train_T in train_Ts:
            for model_path in fixed_paths[train_T]:
                evaluation_records.extend(
                    evaluate(model_path, "fixed", train_T, val_Ts)
                )
    if adaptive_paths:
        print("\nEvaluating adaptive models")
        for model_path in adaptive_paths:
            evaluation_records.extend(evaluate(model_path, "adaptive", None, val_Ts))
    return evaluation_records


def load_cross_validation_results(cached: Path):
    results_file = Path(cached)
    if not results_file.exists():
        raise FileNotFoundError(
            f"Cached cross-validation results not found: {results_file}"
        )
    with results_file.open("r") as file:
        return json.load(file)


def reusable_fixed_records(payload, fixed_paths, fixed_dir, val_Ts, run_metadata):
    """Split fixed models into cached complete evaluations and paths to evaluate."""
    metadata = payload["metadata"]
    if any(metadata.get(key) != value for key, value in run_metadata.items()):
        return [], fixed_paths
    records = [
        record
        for record in payload["evaluation_records"]
        if record["model_type"] == "fixed"
        and record["train_T"] in fixed_paths
        and record["val_T"] in val_Ts
    ]
    cached_keys = {(r["train_T"], r["seed"], r["val_T"]) for r in records}
    reusable_models = {
        (train_T, seed)
        for train_T, seed, _ in cached_keys
        if all((train_T, seed, T) in cached_keys for T in val_Ts)
    }
    records = [r for r in records if (r["train_T"], r["seed"]) in reusable_models]
    seeds_by_path = {path: seed for path, _, seed in find_fixed_models(fixed_dir)}
    remaining_paths = {
        T: [path for path in paths if (T, seeds_by_path[path]) not in reusable_models]
        for T, paths in fixed_paths.items()
    }
    return records, remaining_paths


def cross_validation(
    model_dir,
    fixed_dir=None,
    output_dir=None,
    max_train_T=None,
    max_eval_T=config.MAX_EVAL_T,
    cached=None,
    device=config.DEVICE,
    metric="median",
    system=config.DEFAULT_SYSTEM,
    error_metric="mse",
    vpt_thresholds=DEFAULT_VPT_THRESHOLDS,
):
    model_dir = Path(model_dir)
    cached = Path(cached) if cached is not None else None

    if cached:
        payload = load_cross_validation_results(cached)
        metadata = payload["metadata"]
        dt = float(metadata["dt"])
        output_dir = output_dir or Path(
            config.system_path(config.EVAL_DIR, metadata.get("system", system))
        )
        budget_based = cached.name.startswith("budget")
        train_Ts = list(range(1, metadata["max_train_T"] + 1))
        if max_train_T is not None:
            train_Ts = [train_T for train_T in train_Ts if train_T <= max_train_T]
        if max_eval_T is None:
            max_eval_T = config.MAX_EVAL_T
        val_Ts = list(range(1, max_eval_T + 1))
        evaluation_records = [
            record
            for record in payload["evaluation_records"]
            if record["val_T"] in val_Ts
            and (record["model_type"] == "adaptive" or record["train_T"] in train_Ts)
        ]
        lyapunov_exponent = metadata.get("lyapunov_exponent")
    else:
        output_dir = output_dir or Path(config.system_path(config.EVAL_DIR, system))
        fixed_dir = Path(fixed_dir) if fixed_dir is not None else model_dir / "fixed"
        adaptive_dir = model_dir / "adaptive"
        dt = get_dt_from_model_dir(model_dir)
        if dt is None:
            dt = get_dt_from_model_dir(fixed_dir) or config.DT
        budget_based = model_dir.name.startswith("budget")
        if lyapunov_exponent is None:
            lyapunov_exponent = float(
                compute_global_lyapunov(dt=config.INTEGRATION_DT, system=system)
            )
        run_metadata = {
            "dt": dt,
            "system": system,
            "vpt_thresholds": list(vpt_thresholds),
            "lyapunov_exponent": lyapunov_exponent,
        }
        fixed_paths = get_fixed_paths(model_dir=fixed_dir)
        train_Ts = sorted(fixed_paths)
        if max_train_T is not None:
            train_Ts = [train_T for train_T in train_Ts if train_T <= max_train_T]
            fixed_paths = {train_T: fixed_paths[train_T] for train_T in train_Ts}
        if max_eval_T is None:
            max_eval_T = config.MAX_EVAL_T
        val_Ts = list(range(1, max_eval_T + 1))
        cached_fixed_records = []
        if budget_based and max_train_T is not None and max_train_T > 1:
            previous_results = sorted(
                Path(output_dir).glob(f"budget_mse_results_*_T{max_train_T - 1}_*.json")
            )
            if previous_results:
                previous_result = previous_results[-1]
                payload = load_cross_validation_results(previous_result)
                cached_fixed_records, fixed_paths = reusable_fixed_records(
                    payload, fixed_paths, fixed_dir, val_Ts, run_metadata
                )
                print(
                    f"Reusing {len(cached_fixed_records)} fixed-model evaluation "
                    f"records from {previous_result}"
                )

        evaluation_records = cached_fixed_records + cross_validate_models(
            fixed_paths,
            get_adaptive_paths(adaptive_dir),
            dt=dt,
            device=device,
            val_Ts=val_Ts,
            system_name=system,
            vpt_thresholds=vpt_thresholds,
        )

    effective_max_train_T = max_train_T or max(train_Ts, default=0)
    summary = summarize_cross_validation(evaluation_records, train_Ts, val_Ts)
    if cached is None:
        save_cross_validation_results(
            evaluation_records,
            summary,
            effective_max_train_T,
            fixed_dir,
            output_dir,
            budget_based,
            run_metadata,
        )
    if not summary["fixed"]:
        print("No fixed models evaluated; skipping plots")
        return
    plot_mse(
        summary,
        output_dir,
        dt,
        effective_max_train_T,
        budget_based,
        metric,
        error_metric,
    )
    if lyapunov_exponent is not None:
        plot_vpt(summary, output_dir, dt, lyapunov_exponent, metric)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixed-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-train-T", type=int, default=None)
    parser.add_argument("--max-eval-T", type=int, default=config.MAX_EVAL_T)
    parser.add_argument("--cached", type=Path, default=None)
    parser.add_argument("--metric", choices=("mean", "median"), default="median")
    parser.add_argument("--error-metric", choices=ERROR_METRICS, default="mse")
    parser.add_argument(
        "--system", choices=SYSTEM_CHOICES, default=config.DEFAULT_SYSTEM
    )
    args = parser.parse_args()
    cross_validation(**vars(args))


if __name__ == "__main__":
    main()
