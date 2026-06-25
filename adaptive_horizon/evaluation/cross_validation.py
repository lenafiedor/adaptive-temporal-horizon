from torch.utils.data import DataLoader
import argparse
import json
import re
import numpy as np
from pathlib import Path

import adaptive_horizon.config as config
from adaptive_horizon.data.dataset import TrajectoryDataset, collate_fn
from adaptive_horizon.dynamics.systems import SYSTEM_CHOICES
from adaptive_horizon.training.loss import validation_loss
from adaptive_horizon.training.utils import resolve_burn_in_steps
from adaptive_horizon.visualization.plotting import (
    plot_mse,
    plot_mse_subplots,
    plot_paired_deltas,
)
from adaptive_horizon.evaluation.utils import (
    load_model,
    save_cross_validation_results,
    get_last_run,
    summarize_cross_validation,
    get_dt_from_model_dir,
    get_checkpoint_normalization_stats,
)


def get_T_values(model_dir: Path):
    """Get unique train_T values from model files matching mlp_T{T}*.pt"""
    model_files = list(model_dir.glob("mlp_T*.pt"))
    train_Ts = set()
    for f in model_files:
        match = re.search(r"mlp_T(\d+)", f.name)
        if match:
            train_Ts.add(int(match.group(1)))
    return sorted(train_Ts)


def get_fixed_paths(train_Ts, model_dir=config.MODEL_DIR):
    """Get all model paths for each train_T."""
    model_paths = {T: [] for T in train_Ts}
    for f in sorted(model_dir.glob("mlp_T*.pt")):
        match = re.search(r"mlp_T(\d+)", f.name)
        if match:
            T = int(match.group(1))
            if T in model_paths:
                model_paths[T].append(f)
    return model_paths


def get_adaptive_paths(model_dir=config.MODEL_DIR):
    """Get all adaptive model paths."""
    return sorted(model_dir.glob("adaptive_mlp*.pt"))


def get_adaptive_method(checkpoint):
    metadata = checkpoint.get("metadata", {})
    adaptive_metadata = metadata.get("adaptive", {})
    return adaptive_metadata.get("method")


def get_training_wall_time(checkpoint):
    metadata = checkpoint.get("metadata", {})
    if "wall_time_seconds" not in metadata:
        return {"wall_time_seconds": float(metadata["train_wall_clock_seconds"])}
    return {"wall_time_seconds": float(metadata["wall_time_seconds"])}


def filter_adaptive_paths(adaptive_paths, adaptive_method=None):
    if adaptive_method is None:
        return adaptive_paths

    filtered_paths = []
    for model_path in adaptive_paths:
        _, checkpoint = load_model(model_path)
        if get_adaptive_method(checkpoint) == adaptive_method:
            filtered_paths.append(model_path)
    return filtered_paths


def make_eval_loader(
    max_val_T, dt, normalization_stats=None, system_name=config.DEFAULT_SYSTEM
):
    burn_in_steps = resolve_burn_in_steps(dt)
    split_gap = max(
        config.MAX_TRAIN_T,
        config.MAX_EVAL_T,
        max_val_T,
    )
    eval_dataset = TrajectoryDataset(
        T=max_val_T,
        dt=dt,
        system=system_name,
        normalize=True,
        seed=config.RANDOM_SEED,
        burn_in=burn_in_steps,
        split="val",
        split_gap=split_gap,
        normalization_stats=normalization_stats,
    )
    return DataLoader(
        eval_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )


def eval_loader_cache_key(normalization_stats):
    if normalization_stats is None:
        return None
    mean = tuple(float(value) for value in normalization_stats["mean"])
    std = tuple(float(value) for value in normalization_stats["std"])
    return mean, std


def cross_validate_models(
    fixed_paths,
    adaptive_paths,
    dt=config.DT,
    device=config.DEVICE,
    val_Ts=None,
    system_name=config.DEFAULT_SYSTEM,
):
    """
    Evaluate models across different validation horizons.

    Args:
        fixed_paths: dict of {train_T: [list of model paths]}
        adaptive_paths: list of adaptive model paths
        dt: time step for simulation
        device: CPU or GPU
        val_Ts: list of validation horizon values

    Returns:
        evaluation_records: list of per-model, per-validation-horizon MSE records.
    """
    train_Ts = list(fixed_paths.keys())
    if val_Ts is None:
        val_Ts = train_Ts
    else:
        val_Ts = list(val_Ts)
    eval_loaders = {}

    def get_eval_loader(checkpoint):
        normalization_stats = get_checkpoint_normalization_stats(checkpoint)
        checkpoint_system = checkpoint.get("metadata", {}).get("system") or system_name
        key = (checkpoint_system, eval_loader_cache_key(normalization_stats))
        if key not in eval_loaders:
            eval_loaders[key] = make_eval_loader(
                max(val_Ts),
                dt,
                normalization_stats,
                system_name=checkpoint_system,
            )
        return eval_loaders[key]

    evaluation_records = []

    for train_T in train_Ts:
        model_paths = fixed_paths.get(train_T, [])
        if model_paths:
            print(f"\nEvaluating fixed models trained with T={train_T}")
        for model_path in model_paths:
            model, checkpoint = load_model(model_path)
            model = model.to(device)
            eval_loader = get_eval_loader(checkpoint)
            seed = checkpoint.get("seed")
            wall_time = get_training_wall_time(checkpoint)
            model_records = []

            for val_T in val_Ts:
                mse = validation_loss(model, eval_loader, val_T, device)
                record = {
                    "model_type": "fixed",
                    "seed": seed,
                    "train_T": train_T,
                    "val_T": val_T,
                    "mse": mse,
                    **wall_time,
                }
                evaluation_records.append(record)
                model_records.append(record)
            print(
                f"  Model {model_path.name}: mean MSE = {np.mean([record['mse'] for record in model_records]):.6f}"
            )

    if adaptive_paths:
        print("\nEvaluating adaptive models")
        for model_path in adaptive_paths:
            model, checkpoint = load_model(model_path)
            model = model.to(device)
            eval_loader = get_eval_loader(checkpoint)
            seed = checkpoint.get("seed")
            method = get_adaptive_method(checkpoint)
            wall_time = get_training_wall_time(checkpoint)
            model_records = []

            for val_T in val_Ts:
                mse = validation_loss(model, eval_loader, val_T, device)
                record = {
                    "model_type": "adaptive",
                    "adaptive_method": method,
                    "seed": seed,
                    "train_T": None,
                    "val_T": val_T,
                    "mse": mse,
                    **wall_time,
                }
                evaluation_records.append(record)
                model_records.append(record)
            print(
                f"  Model {model_path.name}: mean MSE = {np.mean([record['mse'] for record in model_records]):.6f}"
            )

    return evaluation_records


def load_cross_validation_results(cached: Path):
    results_file = Path(cached)
    if not results_file.exists():
        raise FileNotFoundError(
            f"Cached cross-validation results not found: {results_file}"
        )

    with open(results_file, "r") as f:
        return json.load(f)


def cross_validation(
    model_dir=None,
    fixed_dir=None,
    max_train_T=None,
    max_eval_T=config.MAX_EVAL_T,
    cached=None,
    device=config.DEVICE,
    metric="median",
    system_name=config.DEFAULT_SYSTEM,
):
    model_dir = Path(model_dir) if model_dir is not None else None
    fixed_dir = Path(fixed_dir) if fixed_dir is not None else None
    cached = Path(cached) if cached is not None else None

    if cached:
        payload = load_cross_validation_results(cached)
        dt = float(payload["metadata"]["dt"])
        save_dir = Path(
            config.system_path(
                config.EVAL_DIR, payload["metadata"].get("system", system_name)
            )
        )
        budget_based = cached.name.startswith("budget")

        train_Ts = list(range(1, payload["metadata"]["max_train_T"] + 1))
        if max_train_T is not None:
            train_Ts = [T for T in train_Ts if T <= max_train_T]
        if max_eval_T is None or budget_based:
            max_eval_T = config.MAX_EVAL_T
        val_Ts = list(range(1, max_eval_T + 1))

        fixed_records = [
            record
            for record in payload["evaluation_records"]
            if record["model_type"] == "fixed"
            and record["train_T"] in train_Ts
            and record["val_T"] in val_Ts
        ]
        adaptive_records = [
            record
            for record in payload["evaluation_records"]
            if record["model_type"] == "adaptive" and record["val_T"] in val_Ts
        ]

        adaptive_dir = payload["metadata"].get("adaptive_dir", "cached")
        fixed_dir = payload["metadata"].get("fixed_dir", "cached")
        evaluation_records = fixed_records + adaptive_records

    else:
        save_dir = Path(config.system_path(config.EVAL_DIR, system_name))
        model_dir = model_dir or get_last_run(
            config.system_path(config.MODEL_DIR, system_name)
        )
        fixed_dir = fixed_dir or model_dir / "fixed"
        adaptive_dir = model_dir / "adaptive"
        dt = get_dt_from_model_dir(model_dir)
        budget_based = model_dir.name.startswith("budget")

        train_Ts = get_T_values(fixed_dir)
        if max_train_T is not None:
            train_Ts = [T for T in train_Ts if T <= max_train_T]
        if max_eval_T is None or budget_based:
            max_eval_T = int(config.MAX_EVAL_T)
        val_Ts = list(range(1, max_eval_T + 1))

        fixed_paths = get_fixed_paths(train_Ts, fixed_dir)
        adaptive_paths = get_adaptive_paths(adaptive_dir)

        evaluation_records = cross_validate_models(
            fixed_paths,
            adaptive_paths,
            dt=dt,
            device=device,
            val_Ts=val_Ts,
            system_name=system_name,
        )

    effective_max_train_T = max_train_T if max_train_T is not None else max(train_Ts)
    summary = summarize_cross_validation(evaluation_records, train_Ts, val_Ts)
    if not cached:
        save_cross_validation_results(
            evaluation_records,
            summary,
            effective_max_train_T,
            dt,
            adaptive_dir,
            fixed_dir,
            save_dir,
            budget_based,
            system_name,
        )
    plot_mse(summary, save_dir, dt, effective_max_train_T, budget_based, metric)
    plot_mse_subplots(
        evaluation_records,
        summary,
        save_dir,
        dt,
        effective_max_train_T,
        budget_based,
        metric,
    )
    plot_paired_deltas(
        summary["deltas"],
        val_Ts,
        dt,
        save_dir,
        effective_max_train_T,
        budget_based,
        metric,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Run directory containing fixed/ and adaptive/ subdirectories (default: reads from models/last_run.txt)",
    )
    parser.add_argument(
        "--fixed-dir",
        type=str,
        default=None,
        help="Fixed model directory (default: reads from model_dir)",
    )
    parser.add_argument(
        "--max-train-T",
        type=int,
        default=None,
        help="Maximum fixed-model training horizon to include (default: all fixed T values found)",
    )
    parser.add_argument(
        "--max-eval-T",
        type=int,
        default=config.MAX_EVAL_T,
        help="Maximum validation horizon for cross-validation",
    )
    parser.add_argument(
        "--cached",
        type=str,
        default=None,
        help="Reuse records from cached cross-validation results",
    )
    parser.add_argument(
        "--metric",
        choices=("mean", "median"),
        default="median",
        help="Statistic to plot with 95%% CI intervals",
    )
    parser.add_argument(
        "--system",
        choices=SYSTEM_CHOICES,
        default=config.DEFAULT_SYSTEM,
        help="Dynamical system to evaluate",
    )
    args = parser.parse_args()

    cross_validation(
        model_dir=args.model_dir,
        fixed_dir=args.fixed_dir,
        max_train_T=args.max_train_T,
        max_eval_T=args.max_eval_T,
        cached=args.cached,
        metric=args.metric,
        system_name=args.system,
    )


if __name__ == "__main__":
    main()
