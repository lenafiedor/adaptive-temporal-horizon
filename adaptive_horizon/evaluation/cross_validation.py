from torch.utils.data import DataLoader
import argparse
import json
import re
import numpy as np
from datetime import datetime
from pathlib import Path

from adaptive_horizon.training.train import ADAPTIVE_HORIZON, WEIGHTED_LOSS
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


def get_T_values(model_dir: Path):
    """Get unique train_T values from model files matching mlp_T{T}*.pt"""
    model_files = list(model_dir.glob("mlp_T*.pt"))
    train_Ts = set()
    for f in model_files:
        match = re.search(r"mlp_T(\d+)", f.name)
        if match:
            train_Ts.add(int(match.group(1)))
    return sorted(train_Ts)


def get_model_paths(train_Ts, model_dir=config.MODEL_DIR):
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
    T_values=None,
    dt=config.DT,
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
            stats[train_T][val_T] = (float(np.mean(values)), float(np.std(values)))

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
    mean_adaptive_mse,
    dt,
    model_dir,
    save_dir=config.EVAL_DIR,
):
    """Save cross-validation summaries to a JSON file."""
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = save_dir / (
        f"mse_results_dt_{str(dt).split('.')[1]}_{timestamp}.json"
    )
    payload = {
        "metadata": {
            "created_at": timestamp,
            "dt": float(dt),
            "burn_in_time": config.BURN_IN_TIME,
            "burn_in_steps": config.resolve_burn_in_steps(dt),
            "model_dir": str(model_dir),
            "T_max": max(T_values),
            "best_train_T": int(best_train_T),
            "mean adaptive MSE": mean_adaptive_mse,
        },
        "evaluation_records": evaluation_records,
    }

    with open(results_file, "w") as f:
        json.dump(payload, f, indent=2)

    with open(save_dir / LAST_RUN_FILE, "w") as f:
        f.write(str(results_file))

    print(f"Cross-validation results saved to {results_file}")
    return results_file


def load_cross_validation_results(
    cached: Path = None, save_dir: Path = config.EVAL_DIR
):
    save_dir = Path(save_dir)
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
        print(f"Using cached cross-validation results: {results_file}")

        plot_mse(
            T_values=list(range(1, metadata["T_max"] + 1)),
            evaluation_records=[
                record
                for record in payload["evaluation_records"]
                if record["model_type"] == "fixed"
            ],
            save_dir=save_dir,
            dt=float(metadata["dt"]),
            summary_mode=plot_summary_mode,
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

    T_values = get_T_values(model_dir)
    if not T_values:
        print("No fixed models found to evaluate")
        return

    if max_T is not None:
        T_values = [T for T in T_values if T <= max_T]
        if not T_values:
            print(f"No models found with T <= {max_T}")
            return

    model_paths = get_model_paths(T_values, model_dir)
    adaptive_paths = get_adaptive_paths(model_dir)
    print(f"T values: {T_values}")

    evaluation_records = cross_validate_models(
        model_paths, adaptive_paths, T_values, dt, device
    )
    stats, adaptive_stats = compute_statistics(evaluation_records, T_values)

    mean_fixed_mse = {
        train_T: np.mean([stats[train_T][val_T][0] for val_T in T_values])
        for train_T in T_values
    }
    best_train_T = min(mean_fixed_mse, key=mean_fixed_mse.get)
    mean_adaptive_mse = np.mean([stats[0] for stats in adaptive_stats.values()])

    print(
        f"Best train_T: {best_train_T} with mean MSE {mean_fixed_mse[best_train_T]:.6f}"
    )
    if adaptive_stats:
        print(f"Mean MSE for adaptive models: {mean_adaptive_mse:.6f}")

    save_cross_validation_results(
        evaluation_records,
        T_values,
        best_train_T,
        mean_adaptive_mse,
        dt,
        model_dir,
        save_dir=save_dir,
    )
    plot_mse(
        T_values,
        evaluation_records,
        save_dir,
        dt,
        summary_mode=plot_summary_mode,
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
        choices=[ADAPTIVE_HORIZON, WEIGHTED_LOSS],
        default=None,
        help="Evaluate only adaptive models trained with the selected method",
    )
    parser.add_argument(
        "--cached",
        type=str,
        default=None,
        help="Plot with cached cross-validation results for fixed T values",
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
