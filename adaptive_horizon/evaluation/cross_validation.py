from torch.utils.data import DataLoader
import argparse
import json
import re
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

from adaptive_horizon.adaptive_methods import (
    ADAPTIVE_HORIZON_METHOD,
    WEIGHTED_LOSS_METHOD,
    filter_adaptive_model_paths,
    get_evaluation_method_abbreviation,
    get_adaptive_method,
    resolve_adaptive_method,
)
import adaptive_horizon.config as config
from adaptive_horizon.data.dataset import LorenzDataset, collate_fn
from adaptive_horizon.training.loss import validation_loss
from adaptive_horizon.visualization.plotting import plot_mse
from adaptive_horizon.evaluation.utils import load_model

LAST_RUN_FILE = "last_run.txt"


def get_last_run(save_dir):
    last_run_file = Path(save_dir) / LAST_RUN_FILE
    if not last_run_file.exists():
        raise FileNotFoundError(
            "No last run found. Run training / cross-validation first."
        )

    with open(last_run_file, "r") as f:
        return Path(f.read().strip())


def get_dt_from_model_dir(model_dir: Path):
    match = re.search(r"dt_(\d+)_.+$", model_dir.name)
    if not match:
        raise ValueError(
            "Could not infer dt from model directory name. "
            f"Expected format 'dt_{{dt}}_{{timestamp}}', got: {model_dir.name}"
        )
    digits = match.group(1)
    return float(digits) / (10 ** len(digits))


def get_fixed_model_dir(dt: float):
    return config.MODEL_DIR / f"fixed_dt_{str(dt).split('.')[1]}"


def get_T_values(model_dir: Path):
    """Get unique train_T values from fixed model files matching mlp_T{T}*.pt."""
    train_Ts = set()
    for model_path in model_dir.glob("mlp_T*.pt"):
        match = re.search(r"mlp_T(\d+)", model_path.name)
        if match:
            train_Ts.add(int(match.group(1)))
    return sorted(train_Ts)


def get_model_paths(train_Ts, fixed_model_dir: Path):
    """Get all fixed model paths for each train_T."""
    model_paths = {T: [] for T in train_Ts}
    for model_path in sorted(fixed_model_dir.glob("mlp_T*.pt")):
        match = re.search(r"mlp_T(\d+)", model_path.name)
        if match:
            T = int(match.group(1))
            if T in model_paths:
                model_paths[T].append(model_path)
    return model_paths


def get_fixed_model_paths(model_dir: Path):
    """Get fixed model paths from the current run directory and legacy fixed dir."""
    fixed_dirs = [model_dir]
    legacy_fixed_dir = get_fixed_model_dir(get_dt_from_model_dir(model_dir))
    if legacy_fixed_dir != model_dir and legacy_fixed_dir.exists():
        fixed_dirs.append(legacy_fixed_dir)

    T_values = sorted({T for fixed_dir in fixed_dirs for T in get_T_values(fixed_dir)})
    model_paths = {T: [] for T in T_values}
    used_dirs = []

    for fixed_dir in fixed_dirs:
        dir_model_paths = get_model_paths(T_values, fixed_dir)
        if any(dir_model_paths.values()):
            used_dirs.append(fixed_dir)
        for T, paths in dir_model_paths.items():
            model_paths[T].extend(paths)

    return T_values, model_paths, used_dirs


def get_adaptive_paths(model_dir=config.MODEL_DIR):
    """Get all adaptive model paths."""
    return sorted(model_dir.glob("adaptive_mlp*.pt"))


def get_adaptive_eval_T_values(adaptive_paths):
    """Infer evaluation horizons from adaptive checkpoint metadata."""
    max_supported_T = 0

    for model_path in adaptive_paths:
        checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
        metadata = checkpoint.get("metadata", {})
        adaptive_metadata = metadata.get("adaptive", {})
        supported_T = adaptive_metadata.get("T_max") or adaptive_metadata.get("max_T")
        if supported_T is not None:
            max_supported_T = max(max_supported_T, int(supported_T))

    return list(range(1, max_supported_T + 1))


def get_normalization_stats(checkpoint):
    metadata = checkpoint.get("metadata", {})
    return metadata.get("normalization_stats")


def make_eval_loader(max_val_T, dt, normalization_stats=None):
    burn_in_steps = config.resolve_burn_in_steps(dt)
    eval_dataset = LorenzDataset(
        num_trajectories=config.NUM_TRAJECTORIES,
        steps_per_trajectory=config.STEPS_PER_TRAJECTORY,
        T=max_val_T,
        dt=dt,
        normalize=True,
        seed=config.EVAL_SEED,
        burn_in=burn_in_steps,
        normalization_stats=normalization_stats,
    )
    return DataLoader(
        eval_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )


def loader_cache_key(normalization_stats):
    if normalization_stats is None:
        return None
    mean = tuple(float(value) for value in normalization_stats["mean"])
    std = tuple(float(value) for value in normalization_stats["std"])
    return mean, std


def cross_validate_models(
    model_paths,
    adaptive_paths,
    T_values,
    dt,
    device=config.DEVICE,
):
    """
    Evaluate models across different validation horizons.

    Args:
        model_paths: dict of {train_T: [list of model paths]}
        adaptive_paths: list of adaptive model paths
        T_values: list of horizon values
        dt: time step for simulation
        device: CPU or GPU

    Returns:
        evaluation_records: list of per-model, per-validation-horizon MSE records.
    """
    eval_loaders = {}

    def get_eval_loader(checkpoint):
        normalization_stats = get_normalization_stats(checkpoint)
        key = loader_cache_key(normalization_stats)
        if key not in eval_loaders:
            eval_loaders[key] = make_eval_loader(max(T_values), dt, normalization_stats)
        return eval_loaders[key]

    evaluation_records = []

    for T in T_values:
        print(f"\nEvaluating models for T={T}")
        for model_path in model_paths[T]:
            model, checkpoint = load_model(model_path)
            model = model.to(device)
            eval_loader = get_eval_loader(checkpoint)
            seed = checkpoint.get("seed")
            model_records = []

            for val_T in T_values:
                mse = validation_loss(model, eval_loader, val_T, device)
                record = {
                    "model_type": "fixed",
                    "train_T": int(T),
                    "seed": int(seed) if seed is not None else None,
                    "val_T": int(val_T),
                    "mse": float(mse),
                    "model_file": model_path.name,
                    "model_path": str(model_path),
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
            model_records = []

            for val_T in T_values:
                mse = validation_loss(model, eval_loader, val_T, device)
                record = {
                    "model_type": "adaptive",
                    "adaptive_method": get_adaptive_method(model_path),
                    "train_T": None,
                    "seed": int(seed) if seed is not None else None,
                    "val_T": int(val_T),
                    "mse": float(mse),
                    "model_file": model_path.name,
                    "model_path": str(model_path),
                }
                evaluation_records.append(record)
                model_records.append(record)
            print(
                f"  Model {model_path.name}: mean MSE = {np.mean([record['mse'] for record in model_records]):.6f}"
            )

    return evaluation_records


def compute_statistics(evaluation_records, T_values):
    """
    Compute mean and std for each (train_T, val_T) combination.
    If only one model, std is 0.

    Returns:
        stats: dict of {train_T: {val_T: (mean, std)}}
        adaptive_stats: dict of {val_T: (mean, std)}
    """
    stats = {T: {} for T in T_values}

    for train_T in T_values:
        for val_T in T_values:
            values = np.array(
                [
                    record["mse"]
                    for record in evaluation_records
                    if record["model_type"] == "fixed"
                    and record["train_T"] == train_T
                    and record["val_T"] == val_T
                ]
            )
            if len(values) > 0:
                stats[train_T][val_T] = (float(np.mean(values)), float(np.std(values)))
            else:
                stats[train_T][val_T] = (float("nan"), float("nan"))

    adaptive_stats = {}
    for val_T in T_values:
        values = np.array(
            [
                record["mse"]
                for record in evaluation_records
                if record["model_type"] == "adaptive" and record["val_T"] == val_T
            ]
        )
        if len(values) > 0:
            adaptive_stats[val_T] = (float(np.mean(values)), float(np.std(values)))

    return stats, adaptive_stats


def save_cross_validation_results(
    evaluation_records,
    T_values,
    best_train_T,
    dt,
    model_dir,
    adaptive_method=None,
    save_dir=config.EVAL_DIR,
):
    """Save cross-validation summaries to a JSON file."""
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_adaptive_method = resolve_adaptive_method(
        evaluation_records, adaptive_method
    )
    method = get_evaluation_method_abbreviation(
        evaluation_records, resolved_adaptive_method
    )
    results_file = save_dir / (
        f"mse_results_dt_{str(dt).split('.')[1]}_{method}_{timestamp}.json"
    )
    payload = {
        "metadata": {
            "created_at": timestamp,
            "dt": float(dt),
            "burn_in_time": config.BURN_IN_TIME,
            "burn_in_steps": config.resolve_burn_in_steps(dt),
            "model_dir": str(model_dir),
            "T_values": [int(T) for T in T_values],
            "T_max": max(T_values),
            "best_train_T": int(best_train_T) if best_train_T is not None else None,
            "adaptive_method": resolved_adaptive_method,
        },
        "evaluation_records": evaluation_records,
    }

    with open(results_file, "w") as f:
        json.dump(payload, f, indent=2)

    with open(save_dir / LAST_RUN_FILE, "w") as f:
        f.write(str(results_file))

    print(f"Cross-validation results saved to {results_file}")
    return results_file


def load_cross_validation_results(cached, save_dir=config.EVAL_DIR):
    save_dir = Path(save_dir)
    if cached == "__last__":
        results_file = get_last_run(save_dir)
    else:
        results_file = Path(cached)

    if not results_file.exists():
        raise FileNotFoundError(
            f"Cached cross-validation results not found: {results_file}"
        )

    with open(results_file, "r") as f:
        payload = json.load(f)

    if "evaluation_records" not in payload:
        raise ValueError(
            "Cached results file does not contain raw evaluation records. "
            "Re-run cross-validation to regenerate it."
        )

    return results_file, payload


def cross_validation(
    model_dir=None,
    max_T=None,
    adaptive_method=None,
    plot_summary_mode="mean-std",
    cached=None,
    save_dir=config.EVAL_DIR,
    device=config.DEVICE,
):
    save_dir = Path(save_dir)

    if cached is not None:
        results_file, payload = load_cross_validation_results(cached, save_dir)
        metadata = payload["metadata"]
        evaluation_records = payload["evaluation_records"]
        T_values = [int(T) for T in metadata["T_values"]]
        dt = float(metadata["dt"])
        cached_adaptive_method = metadata.get("adaptive_method")

        print(f"Using cached cross-validation results: {results_file}")
        plot_mse(
            T_values,
            evaluation_records,
            save_dir,
            dt,
            summary_mode=plot_summary_mode,
            adaptive_method=cached_adaptive_method,
        )
        return

    if model_dir is None:
        model_dir = get_last_run(config.MODEL_DIR)
    else:
        model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    dt = get_dt_from_model_dir(model_dir)
    print(f"Using model directory: {model_dir}")
    fixed_T_values, model_paths, fixed_model_dirs = get_fixed_model_paths(model_dir)
    adaptive_paths = filter_adaptive_model_paths(
        get_adaptive_paths(model_dir), adaptive_method
    )

    T_values = fixed_T_values or get_adaptive_eval_T_values(adaptive_paths)
    if not T_values:
        print("No models found to evaluate")
        return

    if max_T is not None:
        T_values = [T for T in T_values if T <= max_T]
        if not T_values:
            print(f"No models found with T <= {max_T}")
            return

    if fixed_T_values:
        model_paths = {T: paths for T, paths in model_paths.items() if T in T_values}
    else:
        model_paths = {}

    print(f"T values: {T_values}")
    if fixed_model_dirs:
        print(
            "Fixed model directories: "
            + ", ".join(str(fixed_model_dir) for fixed_model_dir in fixed_model_dirs)
        )
    else:
        print("Fixed model directory: none found, evaluating adaptive models only")
    if adaptive_method is None:
        print(f"Adaptive models included: {len(adaptive_paths)} (all methods)")
    else:
        print(
            f"Adaptive models included: {len(adaptive_paths)} (method={adaptive_method})"
        )

    evaluation_records = cross_validate_models(
        model_paths, adaptive_paths, T_values, dt, device
    )
    if not evaluation_records:
        print("No models found to evaluate")
        return

    stats, adaptive_stats = compute_statistics(evaluation_records, T_values)

    fixed_records_present = any(
        record["model_type"] == "fixed" for record in evaluation_records
    )
    best_train_T = None
    if fixed_records_present:
        mean_across_val_Ts = {
            train_T: np.mean([stats[train_T][val_T][0] for val_T in T_values])
            for train_T in T_values
        }
        best_train_T = min(mean_across_val_Ts, key=mean_across_val_Ts.get)
        print(
            f"Best train_T: {best_train_T} with mean MSE {mean_across_val_Ts[best_train_T]:.6f}"
        )
    if adaptive_stats:
        print(
            f"Mean MSE for adaptive models: {np.mean([stats[0] for stats in adaptive_stats.values()]):.6f}"
        )

    save_cross_validation_results(
        evaluation_records,
        T_values,
        best_train_T,
        dt,
        model_dir,
        adaptive_method=adaptive_method,
        save_dir=save_dir,
    )
    plot_mse(
        T_values,
        evaluation_records,
        save_dir,
        dt,
        summary_mode=plot_summary_mode,
        adaptive_method=adaptive_method,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing models (default: reads from last_run.txt)",
    )
    parser.add_argument(
        "--max-T",
        type=int,
        default=None,
        help="Maximum T for evaluation (default: all T values found in model_dir)",
    )
    parser.add_argument(
        "--adaptive-method",
        choices=[ADAPTIVE_HORIZON_METHOD, WEIGHTED_LOSS_METHOD],
        default=None,
        help="Evaluate only adaptive models trained with the selected method",
    )
    parser.add_argument(
        "--cached",
        nargs="?",
        const="__last__",
        default=None,
        help="Plot cached cross-validation results from JSON without re-running evaluation",
    )
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot",
        choices=["mean-std", "mean-ci", "median-iqr"],
        default="mean-ci",
        help="Plot fixed-horizon mean / median MSE with 95%% confidence intervals / interquartile ranges",
    )
    args = parser.parse_args()

    cross_validation(
        model_dir=args.model_dir,
        max_T=args.max_T,
        adaptive_method=args.adaptive_method,
        plot_summary_mode=args.plot,
        cached=args.cached,
    )


if __name__ == "__main__":
    main()
