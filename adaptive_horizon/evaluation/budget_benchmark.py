import argparse
from datetime import datetime
from pathlib import Path
import torch

import adaptive_horizon.config as config
from adaptive_horizon.evaluation.cross_validation import (
    cross_validate_models,
    filter_adaptive_paths,
    get_adaptive_paths,
    get_model_paths,
    get_T_values,
)
from adaptive_horizon.evaluation.utils import (
    save_cross_validation_results,
    summarize_cross_validation,
    get_dt_from_model_dir,
)
from adaptive_horizon.training.methods import CURRICULUM_HORIZON
from adaptive_horizon.training.train import train_adaptive_models, train_fixed_models
from adaptive_horizon.utils import format_dt
from adaptive_horizon.visualization.plotting import plot_mse


def train_compute_budget_models(
    dt, max_train_T, epochs_per_T, n_seeds, device, batch_size, adaptive_only=False
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_root = (
        config.MODEL_DIR / f"budget_based_dt_{format_dt(dt)}_T{max_train_T}_{timestamp}"
    )
    fixed_dir = model_root / "fixed"
    adaptive_dir = model_root / "adaptive"

    if not adaptive_only:
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
    fixed=None,
    adaptive_only=False,
):
    fixed = Path(fixed) if fixed is not None else None
    if adaptive_only and fixed is None and cached is None:
        raise ValueError("Adaptive-only budget evaluation requires --fixed or --cached")

    if cached is not None:
        model_root = Path(cached)
        if not model_root.exists():
            raise FileNotFoundError(f"Cached model directory not found: {model_root}")

        fixed_dir = fixed or model_root / "fixed"
        if not fixed_dir.exists():
            raise FileNotFoundError(f"Fixed model directory not found: {fixed_dir}")
        adaptive_dir = model_root / "adaptive"
        dt = get_dt_from_model_dir(model_root)
        available_train_Ts = get_T_values(fixed_dir)
        if not available_train_Ts:
            raise ValueError(f"No fixed-horizon models found in {fixed_dir}")

        if max_train_T is None:
            max_train_T = max(available_train_Ts)
        else:
            max_train_T = min(max_train_T, max(available_train_Ts))
        train_Ts = [T for T in available_train_Ts if T <= max_train_T]

        if max_eval_T is None:
            max_eval_T = int(config.MAX_EVAL_T)

        print(f"Using cached models from: {model_root}")
    else:
        if max_train_T is None:
            max_train_T = int(config.MAX_TRAIN_T)
        if max_eval_T is None:
            max_eval_T = int(config.MAX_EVAL_T)

        if fixed is not None:
            if not fixed.exists():
                raise FileNotFoundError(f"Fixed model directory not found: {fixed}")
            available_train_Ts = get_T_values(fixed)
            if not available_train_Ts:
                raise ValueError(f"No fixed-horizon models found in {fixed}")
            max_train_T = min(max_train_T, max(available_train_Ts))
            train_Ts = [T for T in available_train_Ts if T <= max_train_T]
        else:
            train_Ts = list(range(1, max_train_T + 1))

        fixed_dir, adaptive_dir = train_compute_budget_models(
            dt=dt,
            max_train_T=max_train_T,
            epochs_per_T=epochs_per_T,
            n_seeds=n_seeds,
            device=device,
            batch_size=batch_size,
            adaptive_only=adaptive_only or fixed is not None,
        )
        if fixed is not None:
            fixed_dir = fixed

    val_Ts = list(range(1, max_eval_T + 1))
    fixed_paths = get_model_paths(train_Ts, fixed_dir)
    missing_train_Ts = [T for T, paths in fixed_paths.items() if not paths]
    if missing_train_Ts:
        raise ValueError(
            f"No fixed-horizon models found for train T values {missing_train_Ts} "
            f"in {fixed_dir}"
        )
    adaptive_paths = filter_adaptive_paths(
        get_adaptive_paths(adaptive_dir), CURRICULUM_HORIZON
    )
    if not adaptive_paths:
        raise ValueError(
            f"No curriculum-horizon adaptive models found in {adaptive_dir}"
        )

    print(f"\nCross-validating train T values: {train_Ts}")
    print(f"Cross-validating eval T values: {val_Ts}\n")
    records = cross_validate_models(
        fixed_paths=fixed_paths,
        adaptive_paths=adaptive_paths,
        dt=dt,
        device=device,
        val_Ts=val_Ts,
    )
    summary = summarize_cross_validation(records, train_Ts, val_Ts, n_seeds)
    results_path = save_cross_validation_results(
        records,
        summary,
        max_train_T,
        dt,
        adaptive_dir,
        fixed_dir,
        save_dir=save_dir,
        budget_based=True,
    )
    plot_mse(summary, save_dir, dt, max_train_T, budget_based=True)

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
        default=config.MAX_EVAL_T,
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
    parser.add_argument(
        "--fixed",
        type=Path,
        default=None,
        help="Directory containing existing fixed-horizon models for evaluation",
    )
    parser.add_argument(
        "--adaptive-only",
        "--adaptive",
        action="store_true",
        dest="adaptive_only",
        help="Train adaptive models only",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else config.DEVICE

    print(f"Using device: {device}")
    print(f"dt: {args.dt}")
    print(
        f"max_train_T: {args.max_train_T if args.max_train_T is not None else 'auto'}"
    )
    print(f"max_eval_T: {args.max_eval_T}")
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
        fixed=args.fixed,
        adaptive_only=args.adaptive_only,
    )


if __name__ == "__main__":
    main()
