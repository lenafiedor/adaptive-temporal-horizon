import argparse
import json
from datetime import datetime
from math import sqrt
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import adaptive_horizon.config as config
from adaptive_horizon.evaluation.cross_validation import (
    cross_validate_models,
    filter_adaptive_paths,
    get_adaptive_paths,
    get_model_paths,
)
from adaptive_horizon.evaluation.utils import load_model
from adaptive_horizon.training.methods import CURRICULUM_HORIZON
from adaptive_horizon.training.train import train_adaptive_models, train_fixed_models
from adaptive_horizon.utils import format_dt
from adaptive_horizon.visualization.plotting import COLOR_EVAL, COLOR_TRAIN


RESOURCE_METRIC = "rollout_model_calls"


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
    loss_root = config.LOSS_DIR / f"compute_budget_dt_{format_dt(dt)}_{timestamp}"
    fixed_dir = model_root / "fixed"
    adaptive_dir = model_root / "adaptive"
    fixed_loss_dir = loss_root / "fixed"
    adaptive_loss_dir = loss_root / "adaptive"
    loss_root.mkdir(parents=True, exist_ok=True)

    train_fixed_models(
        train_Ts=list(range(1, max_train_T + 1)),
        n_seeds=n_seeds,
        epochs=epochs_per_T,
        device=device,
        model_save_dir=fixed_dir,
        loss_save_dir=fixed_loss_dir,
        dt=dt,
        batch_size=batch_size,
    )

    train_adaptive_models(
        n_seeds=n_seeds,
        epochs=epochs_per_T * max_train_T,
        device=device,
        model_save_dir=adaptive_dir,
        loss_save_dir=adaptive_loss_dir,
        dt=dt,
        batch_size=batch_size,
        adaptive_method=CURRICULUM_HORIZON,
        max_T=max_train_T,
    )

    return model_root, loss_root, fixed_dir, adaptive_dir


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
                    "delta_mse": float(adaptive["mse"] - best_fixed["mse"]),
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


def plot_paired_deltas(summary, val_Ts, dt, save_dir, timestamp):
    save_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray([T * dt for T in val_Ts], dtype=np.float64)
    means = np.asarray(
        [summary["by_val_T"][str(T)]["mean"] for T in val_Ts], dtype=np.float64
    )
    lows = np.asarray(
        [summary["by_val_T"][str(T)]["ci95_low"] for T in val_Ts], dtype=np.float64
    )
    highs = np.asarray(
        [summary["by_val_T"][str(T)]["ci95_high"] for T in val_Ts], dtype=np.float64
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.plot(
        x,
        means,
        color=COLOR_TRAIN,
        linewidth=2.0,
        marker="o",
        label="Curriculum - best fixed grid",
    )
    ax.fill_between(x, lows, highs, color=COLOR_EVAL, alpha=0.35, linewidth=0)
    ax.set_xlabel("Validation horizon time")
    ax.set_ylabel("Paired MSE delta")
    ax.set_title("Compute-budget horizon search: paired validation deltas")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    plot_path = save_dir / f"compute_budget_deltas_dt_{format_dt(dt)}_{timestamp}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Paired delta plot saved to {plot_path}")
    return plot_path


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
    max_train_T,
    epochs_per_T,
    n_seeds,
    max_eval_T=config.MAX_EVAL_T,
    batch_size=config.BATCH_SIZE,
    device=config.DEVICE,
    save_dir=config.EVAL_DIR,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seeds = list(range(n_seeds))
    train_Ts = list(range(1, max_train_T + 1))
    val_Ts = list(range(1, max_eval_T + 1))

    model_root, loss_root, fixed_dir, adaptive_dir = train_compute_budget_models(
        dt=dt,
        max_train_T=max_train_T,
        epochs_per_T=epochs_per_T,
        n_seeds=n_seeds,
        device=device,
        batch_size=batch_size,
        timestamp=timestamp,
    )

    model_paths = get_model_paths(val_Ts, fixed_dir)
    adaptive_paths = filter_adaptive_paths(
        get_adaptive_paths(adaptive_dir), CURRICULUM_HORIZON
    )
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

    paired = build_paired_deltas(records, seeds, val_Ts)
    summary = summarize_paired_deltas(paired, val_Ts, primary_val_T=max_train_T)

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
        "train_Ts": train_Ts,
        "val_Ts": val_Ts,
        "epochs_per_T": int(epochs_per_T),
        "adaptive_epochs": int(epochs_per_T * max_train_T),
        "seeds": seeds,
        "n_seeds": int(n_seeds),
        "batch_size": int(batch_size),
        "resource_metric": RESOURCE_METRIC,
        "resource_formula": "sum_over_epochs(num_batches * T_epoch)",
        "model_dir": str(model_root),
        "loss_dir": str(loss_root),
        "fixed_grid_rollout_model_calls_by_seed": fixed_budget_by_seed,
        "adaptive_rollout_model_calls_by_seed": adaptive_budget_by_seed,
    }

    results_path = save_results(records, paired, summary, metadata, save_dir, timestamp)
    plot_path = plot_paired_deltas(summary, val_Ts, dt, save_dir, timestamp)
    return results_path, plot_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=config.DT)
    parser.add_argument(
        "--max-T",
        type=int,
        default=config.MAX_TRAIN_T,
        help="Maximum training/search horizon",
    )
    parser.add_argument("--epochs-per-T", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=config.NUM_SEEDS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--save-dir", type=Path, default=config.EVAL_DIR)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else config.DEVICE

    print(f"Using device: {device}")
    print(f"dt: {args.dt}")
    print(f"max_train_T: {args.max_T}")
    print(f"epochs_per_T: {args.epochs_per_T}")
    print(f"n_seeds: {args.n_seeds}")
    print(f"resource metric: {RESOURCE_METRIC}")

    compute_budget_comparison(
        dt=args.dt,
        max_train_T=args.max_T,
        epochs_per_T=args.epochs_per_T,
        n_seeds=args.n_seeds,
        batch_size=args.batch_size,
        device=device,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
