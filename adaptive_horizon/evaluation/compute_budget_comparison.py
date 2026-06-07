import argparse
from datetime import datetime
from math import sqrt
from pathlib import Path
from statistics import mean, median, stdev
import torch

import adaptive_horizon.config as config
from adaptive_horizon.evaluation.cross_validation import (
    cross_validate_models,
    filter_adaptive_paths,
    get_adaptive_paths,
    get_model_paths,
    get_T_values,
    compute_statistics,
)
from adaptive_horizon.evaluation.utils import (
    load_model,
    save_cross_validation_results,
    summarize_cross_validation,
)
from adaptive_horizon.training.methods import CURRICULUM_HORIZON
from adaptive_horizon.training.train import train_adaptive_models, train_fixed_models
from adaptive_horizon.utils import format_dt
from adaptive_horizon.visualization.plotting import plot_mse


def infer_dt_from_models(model_dir: Path, fallback_dt: float):
    for model_path in sorted(model_dir.rglob("*.pt")):
        _, checkpoint = load_model(model_path)
        metadata = checkpoint.get("metadata", {})
        if "dt" in metadata:
            return float(metadata["dt"])
    return float(fallback_dt)


def confidence_summary(values):
    values = [float(value) for value in values]
    mean_value = float(mean(values))
    std_value = float(stdev(values))
    margin = float(1.96 * std_value / sqrt(len(values)))
    return {
        "mean": mean_value,
        "std": std_value,
        "median": float(median(values)),
        "ci95_low": mean_value - margin,
        "ci95_high": mean_value + margin,
    }


def train_compute_budget_models(
    dt,
    max_train_T,
    epochs_per_T,
    n_seeds,
    device,
    batch_size,
    timestamp,
):
    model_root = (
        config.MODEL_DIR
        / f"compute_budget_dt_{format_dt(dt)}_T{max_train_T}_{timestamp}"
    )
    fixed_dir = model_root / "fixed"
    adaptive_dir = model_root / "adaptive"

    train_fixed_models(
        train_Ts=list(range(1, max_train_T + 1)),
        n_seeds=n_seeds,
        epochs=epochs_per_T,
        device=device,
        model_save_dir=fixed_dir,
        dt=dt,
        batch_size=batch_size,
    )

    train_adaptive_models(
        n_seeds=n_seeds,
        epochs=epochs_per_T * max_train_T,
        device=device,
        model_save_dir=adaptive_dir,
        dt=dt,
        batch_size=batch_size,
        adaptive_method=CURRICULUM_HORIZON,
        max_T=max_train_T,
    )

    return fixed_dir, adaptive_dir


def get_clock_seconds(model_path):
    _, checkpoint = load_model(model_path)
    metadata = checkpoint.get("metadata", {})
    return float(metadata.get("train_wall_clock_seconds", 0.0)), checkpoint


def compute_budget_comparison(
    dt,
    epochs_per_T,
    max_train_T=None,
    n_seeds=config.NUM_SEEDS,
    max_eval_T=config.MAX_EVAL_T,
    batch_size=config.BATCH_SIZE,
    device=config.DEVICE,
    save_dir=config.EVAL_DIR,
    cached=None,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if cached is not None:
        model_root = Path(cached)
        if not model_root.exists():
            raise FileNotFoundError(f"Cached model directory not found: {model_root}")

        fixed_dir = model_root / "fixed"
        adaptive_dir = model_root / "adaptive"
        dt = infer_dt_from_models(model_root, dt)
        available_train_Ts = get_T_values(fixed_dir)
        if not available_train_Ts:
            raise ValueError(f"No fixed-horizon models found in {fixed_dir}")

        if max_train_T is None:
            max_train_T = max(available_train_Ts)
        else:
            max_train_T = min(max_train_T, max(available_train_Ts))

        if max_eval_T is None:
            max_eval_T = max_train_T
        else:
            max_eval_T = min(max_eval_T, max_train_T)

        print(f"Using cached models from: {model_root}")
    else:
        if max_train_T is None:
            max_train_T = int(config.MAX_TRAIN_T)
        if max_eval_T is None:
            max_eval_T = min(max_train_T, config.MAX_EVAL_T)

        fixed_dir, adaptive_dir = train_compute_budget_models(
            dt=dt,
            max_train_T=max_train_T,
            epochs_per_T=epochs_per_T,
            n_seeds=n_seeds,
            device=device,
            batch_size=batch_size,
            timestamp=timestamp,
        )

    val_Ts = list(range(1, max_eval_T + 1))
    fixed_paths = get_model_paths(val_Ts, fixed_dir)
    adaptive_paths = filter_adaptive_paths(
        get_adaptive_paths(adaptive_dir), CURRICULUM_HORIZON
    )
    if not adaptive_paths:
        raise ValueError(
            f"No curriculum-horizon adaptive models found in {adaptive_dir}"
        )

    print(f"Cross-validating T values: {val_Ts}")
    records = cross_validate_models(
        fixed_paths=fixed_paths,
        adaptive_paths=adaptive_paths,
        T_values=val_Ts,
        dt=dt,
        device=device,
    )
    best_T, mean_fixed_mse, mean_adaptive_mse = compute_statistics(records, val_Ts)
    summary = summarize_cross_validation(records, val_Ts, val_Ts)

    results_path = save_cross_validation_results(
        records,
        summary,
        max_train_T,
        best_T,
        mean_fixed_mse[best_T],
        mean_adaptive_mse,
        dt,
        adaptive_dir,
        fixed_dir,
        save_dir=save_dir,
    )
    plot_mse(val_Ts, records, save_dir, dt, summary_mode="mean-ci")

    return results_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=config.DT)
    parser.add_argument(
        "--max-train-T",
        type=int,
        default=None,
        help="Maximum training horizon",
    )
    parser.add_argument(
        "--max-eval-T",
        type=int,
        default=None,
        help="Maximum validation horizon for the cross-validation",
    )
    parser.add_argument("--epochs-per-T", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=config.NUM_SEEDS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--save-dir", type=Path, default=config.EVAL_DIR)
    parser.add_argument(
        "--cached",
        type=Path,
        default=None,
        help="Skip training and cross-validate an existing model directory",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else config.DEVICE

    print(f"Using device: {device}")
    print(f"dt: {args.dt}")
    print(
        f"max_train_T: {args.max_train_T if args.max_train_T is not None else 'auto'}"
    )
    print(f"max_eval_T: {args.max_eval_T if args.max_eval_T is not None else 'auto'}")
    print(f"epochs_per_T: {args.epochs_per_T}")
    print(f"n_seeds: {args.n_seeds}\n")

    compute_budget_comparison(
        dt=args.dt,
        epochs_per_T=args.epochs_per_T,
        max_train_T=args.max_train_T,
        n_seeds=args.n_seeds,
        max_eval_T=args.max_eval_T,
        batch_size=args.batch_size,
        device=device,
        save_dir=args.save_dir,
        cached=args.cached,
    )


if __name__ == "__main__":
    main()
