import argparse
from datetime import datetime
from math import sqrt
from pathlib import Path
from statistics import mean, median, stdev
import torch
import json

import adaptive_horizon.config as config
from adaptive_horizon.evaluation.cross_validation import (
    cross_validate_models,
    filter_adaptive_paths,
    get_adaptive_paths,
    get_model_paths,
    get_T_values,
)
from adaptive_horizon.evaluation.utils import load_model
from adaptive_horizon.training.methods import CURRICULUM_HORIZON
from adaptive_horizon.training.train import train_adaptive_models, train_fixed_models
from adaptive_horizon.utils import format_dt
from adaptive_horizon.visualization.plotting import plot_mse, plot_paired_deltas

RESOURCE_METRIC = "rollout_model_calls"


def infer_dt_from_models(model_dir: Path, fallback_dt: float):
    for model_path in sorted(model_dir.rglob("*.pt")):
        _, checkpoint = load_model(model_path)
        metadata = checkpoint.get("metadata", {})
        if "dt" in metadata:
            return float(metadata["dt"])
    return float(fallback_dt)


def confidence_summary(values):
    values = [float(value) for value in values]
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    mean_value = float(mean(values))
    std_value = float(stdev(values)) if n > 1 else 0.0
    margin = float(1.96 * std_value / sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean_value,
        "std": std_value,
        "median": float(median(values)),
        "ci95_low": mean_value - margin,
        "ci95_high": mean_value + margin,
    }


def metadata_budget(checkpoint):
    metadata = checkpoint.get("metadata", {})
    return {
        "train_num_batches": int(metadata.get("train_num_batches", 0)),
        "train_optimizer_updates": int(metadata.get("train_optimizer_updates", 0)),
        "train_rollout_model_calls": int(metadata.get("train_rollout_model_calls", 0)),
        "train_wall_clock_seconds": float(
            metadata.get("train_wall_clock_seconds", 0.0)
        ),
        "T_schedule": metadata.get("T_schedule", []),
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
    model_root = config.MODEL_DIR / f"compute_budget_dt_{format_dt(dt)}_{timestamp}"
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


def add_budget_metadata(records):
    budget_cache = {}
    for record in records:
        model_path = record["model_path"]
        if model_path not in budget_cache:
            _, checkpoint = load_model(model_path)
            budget_cache[model_path] = metadata_budget(checkpoint)
        record.update(budget_cache[model_path])

    return records


def build_paired_deltas(records, seeds, val_Ts):
    paired = []
    for seed in seeds:
        for val_T in val_Ts:
            fixed_records = [
                record
                for record in records
                if record["model_type"] == "fixed"
                and record["seed"] == seed
                and record["val_T"] == val_T
            ]
            adaptive_records = [
                record
                for record in records
                if record["model_type"] == "adaptive"
                and record.get("adaptive_method") == CURRICULUM_HORIZON
                and record["seed"] == seed
                and record["val_T"] == val_T
            ]
            if not fixed_records or not adaptive_records:
                continue

            best_fixed = min(fixed_records, key=lambda record: record["mse"])
            adaptive = adaptive_records[0]
            paired.append(
                {
                    "seed": int(seed),
                    "val_T": int(val_T),
                    "val_time": float(val_T * records[0]["dt"])
                    if "dt" in records[0]
                    else None,
                    "best_fixed_train_T": int(best_fixed["train_T"]),
                    "best_fixed_mse": float(best_fixed["mse"]),
                    "adaptive_mse": float(adaptive["mse"]),
                    "delta_mse": float(best_fixed["mse"] - adaptive["mse"]),
                    "fixed_grid_rollout_model_calls": int(
                        sum(
                            record["train_rollout_model_calls"]
                            for record in fixed_records
                        )
                    ),
                    "adaptive_rollout_model_calls": int(
                        adaptive["train_rollout_model_calls"]
                    ),
                }
            )

    return paired


def summarize_paired_deltas(paired, val_Ts, primary_val_T):
    by_val_T = {}
    for val_T in val_Ts:
        values = [record["delta_mse"] for record in paired if record["val_T"] == val_T]
        by_val_T[str(val_T)] = confidence_summary(values)

    primary_values = [
        record["delta_mse"] for record in paired if record["val_T"] == primary_val_T
    ]

    seeds = sorted({record["seed"] for record in paired})
    mean_over_val_T = []
    for seed in seeds:
        seed_values = [
            record["delta_mse"] for record in paired if record["seed"] == seed
        ]
        if seed_values:
            mean_over_val_T.append(float(mean(seed_values)))

    return {
        "by_val_T": by_val_T,
        "primary_val_T": int(primary_val_T),
        "primary": confidence_summary(primary_values),
        "mean_over_val_T": confidence_summary(mean_over_val_T),
    }


def save_results(
    records,
    paired,
    summary,
    metadata,
    save_dir,
    timestamp,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    results_path = (
        save_dir
        / f"compute_budget_results_dt_{format_dt(metadata['dt'])}_{timestamp}.json"
    )
    payload = {
        "metadata": metadata,
        "evaluation_records": records,
        "paired_deltas": paired,
        "summary": summary,
    }

    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Compute-budget comparison saved to {results_path}")
    return results_path


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
        print(f"Using cached models from: {model_root}")
    else:
        if max_train_T is None:
            max_train_T = config.MAX_TRAIN_T
        fixed_dir, adaptive_dir = train_compute_budget_models(
            dt=dt,
            max_train_T=max_train_T,
            epochs_per_T=epochs_per_T,
            n_seeds=n_seeds,
            device=device,
            batch_size=batch_size,
            timestamp=timestamp,
        )

    max_train_T = int(max_train_T)
    eval_limit = min(int(max_eval_T), max_train_T)
    val_Ts = list(range(1, eval_limit + 1))
    model_paths = get_model_paths(val_Ts, fixed_dir)
    adaptive_paths = filter_adaptive_paths(
        get_adaptive_paths(adaptive_dir), CURRICULUM_HORIZON
    )
    if not adaptive_paths:
        raise ValueError(
            f"No curriculum-horizon adaptive models found in {adaptive_dir}"
        )

    print(f"Cross-validating T values: {val_Ts}")
    records = cross_validate_models(
        model_paths=model_paths,
        adaptive_paths=adaptive_paths,
        T_values=val_Ts,
        dt=dt,
        device=device,
    )
    records = add_budget_metadata(records)
    for record in records:
        record["dt"] = float(dt)

    seeds = sorted(
        {
            int(record["seed"])
            for record in records
            if record.get("seed") is not None
        }
    )
    paired = build_paired_deltas(records, seeds, val_Ts)
    summary = summarize_paired_deltas(paired, val_Ts, primary_val_T=eval_limit)

    fixed_budget_by_seed = {}
    adaptive_budget_by_seed = {}
    for seed in seeds:
        fixed_budget_by_seed[str(seed)] = int(
            sum(
                record["train_rollout_model_calls"]
                for record in records
                if record["model_type"] == "fixed"
                and record["seed"] == seed
                and record["val_T"] == val_Ts[0]
            )
        )
        adaptive_records = [
            record
            for record in records
            if record["model_type"] == "adaptive"
            and record.get("adaptive_method") == CURRICULUM_HORIZON
            and record["seed"] == seed
            and record["val_T"] == val_Ts[0]
        ]
        adaptive_budget_by_seed[str(seed)] = int(
            adaptive_records[0]["train_rollout_model_calls"] if adaptive_records else 0
        )

    metadata = {
        "created_at": timestamp,
        "dt": float(dt),
        "max_train_T": int(max_train_T),
        "max_eval_T": int(max_eval_T),
        "epochs_per_T": int(epochs_per_T),
        "adaptive_epochs": int(epochs_per_T * max_train_T),
        "n_seeds": int(len(seeds)),
        "batch_size": int(batch_size),
        "resource_metric": RESOURCE_METRIC,
        "fixed_dir": str(fixed_dir),
        "adaptive_dir": str(adaptive_dir),
        "fixed_grid_rollout_model_calls_by_seed": fixed_budget_by_seed,
        "adaptive_rollout_model_calls_by_seed": adaptive_budget_by_seed,
    }

    results_path = save_results(records, paired, summary, metadata, save_dir, timestamp)
    plot_mse(val_Ts, records, save_dir, dt, summary_mode="mean-ci")
    plot_paired_deltas(summary, val_Ts, dt, save_dir, timestamp)
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
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else config.DEVICE

    print(f"Using device: {device}")
    print(f"dt: {args.dt}")
    print(f"max_train_T: {args.max_train_T if args.max_train_T is not None else 'auto'}")
    print(f"max_eval_T: {args.max_eval_T}")
    print(f"epochs_per_T: {args.epochs_per_T}")
    print(f"n_seeds: {args.n_seeds}")
    print(f"resource metric: {RESOURCE_METRIC}\n")

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
