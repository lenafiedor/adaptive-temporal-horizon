from torch.utils.data import DataLoader
import argparse
import json
import re
import numpy as np
from datetime import datetime
from pathlib import Path

import adaptive_horizon.config as config
from adaptive_horizon.data.dataset import LorenzDataset, collate_fn
from adaptive_horizon.training.loss import validation_loss
from adaptive_horizon.visualization.plotting import plot_mse
from adaptive_horizon.evaluation.utils import load_model

EVAL_SEED = 12345


def get_last_run():
    """Read last_run.txt and return the corresponding model directory."""
    last_run_file = config.MODEL_DIR / "last_run.txt"
    if not last_run_file.exists():
        return config.MODEL_DIR
    with open(last_run_file, "r") as f:
        return Path(f.read().strip())


def get_T_values(model_dir):
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
    eval_dataset = LorenzDataset(
        num_trajectories=config.NUM_TRAJECTORIES,
        steps_per_trajectory=config.STEPS_PER_TRAJECTORY,
        T=max_val_T,
        dt=dt,
        normalize=True,
        seed=EVAL_SEED,
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
    model_paths, adaptive_paths, T_values, dt, device=config.DEVICE
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


def build_summary_results(stats, adaptive_stats, T_values, evaluation_records):
    """Build cross-validation summaries."""
    fixed_horizon = []
    for train_T in T_values:
        validation = []
        for val_T in T_values:
            mean, std = stats[train_T][val_T]
            validation.append(
                {
                    "val_T": int(val_T),
                    "mean": float(mean),
                    "std": float(std),
                }
            )
        fixed_horizon.append({"train_T": int(train_T), "validation": validation})

    adaptive_validation = []
    for val_T in T_values:
        if val_T not in adaptive_stats:
            continue
        mean, std = adaptive_stats[val_T]
        adaptive_validation.append(
            {
                "val_T": int(val_T),
                "mean": float(mean),
                "std": float(std),
            }
        )
    adaptive_horizon = []
    if adaptive_validation:
        adaptive_horizon.append({"validation": adaptive_validation})

    return {
        "fixed_horizon": fixed_horizon,
        "adaptive_horizon": adaptive_horizon,
    }


def save_cross_validation_results(
    results,
    T_values,
    best_train_T,
    dt,
    model_dir,
    save_dir=config.EVAL_DIR,
):
    """Save cross-validation summaries to a JSON file."""
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = save_dir / f"mse_results_dt_{str(dt).split('.')[1]}_{timestamp}.json"
    payload = {
        "metadata": {
            "created_at": timestamp,
            "dt": float(dt),
            "model_dir": str(model_dir),
            "T_values": [int(T) for T in T_values],
            "best_train_T": int(best_train_T),
        },
        "results": results,
    }

    with open(results_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Cross-validation results saved to {results_file}")
    return results_file


def cross_validation(
    dt,
    model_dir=None,
    max_T=None,
    save_dir=config.EVAL_DIR,
    device=config.DEVICE,
):
    if model_dir is None:
        model_dir = get_last_run()
    else:
        model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Using model directory: {model_dir}")

    T_values = get_T_values(model_dir)
    if not T_values:
        print("No models found to evaluate")
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

    mean_across_val_Ts = {
        train_T: np.mean([stats[train_T][val_T][0] for val_T in T_values])
        for train_T in T_values
    }
    best_train_T = min(mean_across_val_Ts, key=mean_across_val_Ts.get)

    print(
        f"Best train_T: {best_train_T} with mean MSE {mean_across_val_Ts[best_train_T]:.6f}"
    )

    results = build_summary_results(
        stats, adaptive_stats, T_values, evaluation_records
    )

    save_cross_validation_results(
        results,
        T_values,
        best_train_T,
        dt,
        model_dir,
        save_dir,
    )
    plot_mse(T_values, stats, adaptive_stats, save_dir, dt)


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
        "--dt", type=float, default=config.DT, help="Time step for simulation"
    )
    args = parser.parse_args()

    cross_validation(args.dt, model_dir=args.model_dir, max_T=args.max_T)


if __name__ == "__main__":
    main()
