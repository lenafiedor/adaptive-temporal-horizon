from torch.utils.data import DataLoader
import argparse
import json
import re
import numpy as np
from datetime import datetime
from pathlib import Path

from adaptive_horizon.config import (
    MODEL_DIR,
    EVAL_DIR,
    NUM_TRAJECTORIES,
    STEPS_PER_TRAJECTORY,
    BATCH_SIZE,
    DT,
)
from adaptive_horizon.data.dataset import LorenzDataset, collate_fn
from adaptive_horizon.training.loss import validation_loss
from adaptive_horizon.visualization.plotting import plot_mse
from adaptive_horizon.evaluation.utils import load_model


def get_last_run():
    """Read last_run.txt and return the corresponding model directory."""
    last_run_file = MODEL_DIR / "last_run.txt"
    if not last_run_file.exists():
        return MODEL_DIR
    with open(last_run_file, "r") as f:
        return Path(f.read().strip())


def get_train_Ts(model_dir=MODEL_DIR):
    """Get unique train_T values from model files matching mlp_T{T}*.pt"""
    model_files = list(model_dir.glob("mlp_T*.pt"))
    train_Ts = set()
    for f in model_files:
        match = re.search(r"mlp_T(\d+)", f.name)
        if match:
            train_Ts.add(int(match.group(1)))
    return sorted(train_Ts)


def get_model_paths(train_Ts, model_dir=MODEL_DIR):
    """Get all model paths for each train_T."""
    model_paths = {T: [] for T in train_Ts}
    for f in model_dir.glob("mlp_T*.pt"):
        match = re.search(r"mlp_T(\d+)", f.name)
        if match:
            T = int(match.group(1))
            if T in model_paths:
                model_paths[T].append(f)
    return model_paths


def get_adaptive_paths(model_dir=MODEL_DIR):
    """Get all adaptive model paths."""
    return list(model_dir.glob("adaptive_mlp*.pt"))


def get_val_Ts(train_Ts, max_val_T):
    val_Ts = [T for T in train_Ts if T <= max_val_T]
    if not val_Ts:
        val_Ts = [max_val_T]
        return val_Ts

    max_train_T = max(val_Ts)
    if max_val_T > max_train_T:
        next_val = ((max_train_T // 10) + 1) * 10
        while next_val <= max_val_T:
            if next_val not in val_Ts:
                val_Ts.append(next_val)
            next_val += 10
    return val_Ts


def cross_validate_models(
    model_paths, adaptive_paths, train_Ts, val_Ts, max_val_T, dt, device="cpu"
):
    """
    Evaluate models across different validation horizons.

    Args:
        model_paths: dict of {train_T: [list of model paths]}
        adaptive_paths: list of adaptive model paths
        train_Ts: list of training horizons
        val_Ts: list of validation horizons
        max_val_T: maximum validation T (for dataset)
        dt: time step for simulation
        device: CPU or GPU

    Returns:
        mse_matrix: dict of {train_T: {val_T: [list of MSE values]}}
        adaptive_mse: dict of {val_T: [list of MSE values]}
    """
    eval_dataset = LorenzDataset(
        num_trajectories=NUM_TRAJECTORIES,
        steps_per_trajectory=STEPS_PER_TRAJECTORY,
        T=max_val_T,
        dt=dt,
        normalize=True,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    mse_matrix = {T: {vT: [] for vT in val_Ts} for T in train_Ts}

    for T in train_Ts:
        print(f"\nEvaluating models for T={T}")
        for model_path in model_paths[T]:
            model, _ = load_model(model_path)
            model = model.to(device)

            for val_T in val_Ts:
                mse = validation_loss(model, eval_loader, val_T, device)
                mse_matrix[T][val_T].append(mse)
                print(f"    Validation T={val_T}: MSE = {mse:.6f}")
            print(
                f"  Model {model_path.name}: mean MSE = {np.mean([mse_matrix[T][vT][-1] for vT in val_Ts]):.6f}"
            )

    adaptive_mse = {vT: [] for vT in val_Ts}

    if adaptive_paths:
        print("\nEvaluating adaptive models")
        for model_path in adaptive_paths:
            model, _ = load_model(model_path)
            model = model.to(device)

            for val_T in val_Ts:
                mse = validation_loss(model, eval_loader, val_T, device)
                adaptive_mse[val_T].append(mse)
                print(f"    Validation T={val_T}: MSE = {mse:.6f}")
            print(
                f"  Model {model_path.name}: mean MSE = {np.mean([adaptive_mse[vT][-1] for vT in val_Ts]):.6f}"
            )

    return mse_matrix, adaptive_mse


def compute_statistics(mse_matrix, train_Ts, val_Ts, adaptive_mse):
    """
    Compute mean and std for each (train_T, val_T) combination.
    If only one model, std is 0.

    Returns:
        stats: dict of {train_T: {val_T: (mean, std)}}
        adaptive_stats: dict of {val_T: (mean, std)}
    """
    stats = {T: {} for T in train_Ts}

    for T in train_Ts:
        for val_T in val_Ts:
            values = np.array(mse_matrix[T][val_T])
            stats[T][val_T] = (np.mean(values), np.std(values))

    adaptive_stats = {}
    for val_T in val_Ts:
        values = np.array(adaptive_mse[val_T])
        if len(values) > 0:
            adaptive_stats[val_T] = (np.mean(values), np.std(values))

    return stats, adaptive_stats


def save_mse_results(stats, adaptive_stats, train_Ts, val_Ts, save_dir=EVAL_DIR):
    """
    Save MSE results to a CSV file.

    Format: train_T,val_T,mean,std
    Adaptive models have train_T = "adaptive"
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = save_dir / f"mse_results_{timestamp}.csv"
    with open(results_file, "w") as f:
        f.write("train_T,val_T,mean,std\n")
        for train_T in train_Ts:
            for val_T in val_Ts:
                mean, std = stats[train_T][val_T]
                f.write(f"{train_T},{val_T},{mean},{std}\n")

        for val_T in val_Ts:
            if val_T in adaptive_stats:
                mean, std = adaptive_stats[val_T]
                f.write(f"adaptive,{val_T},{mean},{std}\n")

    print(f"MSE results saved to {results_file}")


def save_horizon_prior(stats, train_Ts, val_Ts, best_train_T, dt, save_dir=EVAL_DIR):
    """Save a cross-validation-derived horizon prior for adaptive datasets."""
    save_dir.mkdir(parents=True, exist_ok=True)

    mean_across_val_Ts = {
        train_T: np.mean([stats[train_T][val_T][0] for val_T in val_Ts])
        for train_T in train_Ts
    }
    std_across_val_Ts = {
        train_T: np.mean([stats[train_T][val_T][1] for val_T in val_Ts])
        for train_T in train_Ts
    }

    best_score = mean_across_val_Ts[best_train_T]
    best_score_std = std_across_val_Ts[best_train_T]
    threshold = best_score + best_score_std
    valid_Ts = [T for T in train_Ts if mean_across_val_Ts[T] <= threshold]

    if not valid_Ts:
        valid_Ts = [best_train_T]

    prior = {
        "dt": dt,
        "best_train_T": int(best_train_T),
        "recommended_min_T": int(min(valid_Ts)),
        "recommended_max_T": int(max(valid_Ts)),
        "best_score": float(best_score),
        "best_score_std": float(best_score_std),
        "selection_threshold": float(threshold),
    }

    prior_file = save_dir / f"horizon_prior_dt_{str(dt).split(".")[1]}.json"
    with open(prior_file, "w") as f:
        json.dump(prior, f, indent=2)

    print(f"Horizon prior saved to {prior_file}")


def cross_validation(
    max_val_T, dt, model_dir=None, max_train_T=None, save_dir=EVAL_DIR, device="cpu"
):
    if model_dir is None:
        model_dir = get_last_run()
    else:
        model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Using model directory: {model_dir}")

    train_Ts = get_train_Ts(model_dir)
    if not train_Ts:
        print("No models found to evaluate")
        return

    if max_train_T is not None:
        train_Ts = [T for T in train_Ts if T <= max_train_T]
        if not train_Ts:
            print(f"No models found with T <= {max_train_T}")
            return

    if max_val_T is None:
        max_val_T = max(train_Ts)

    val_Ts = get_val_Ts(train_Ts, max_val_T)
    model_paths = get_model_paths(train_Ts, model_dir)
    adaptive_paths = get_adaptive_paths(model_dir)

    print(f"Training T values: {train_Ts}")
    print(f"Validation T values: {val_Ts}")

    mse_matrix, adaptive_mse = cross_validate_models(
        model_paths, adaptive_paths, train_Ts, val_Ts, max_val_T, dt, device
    )

    stats, adaptive_stats = compute_statistics(
        mse_matrix, train_Ts, val_Ts, adaptive_mse
    )

    save_mse_results(stats, adaptive_stats, train_Ts, val_Ts, save_dir)

    mean_across_val_Ts = {
        train_T: np.mean([stats[train_T][val_T][0] for val_T in val_Ts])
        for train_T in train_Ts
    }
    best_train_T = min(mean_across_val_Ts, key=mean_across_val_Ts.get)

    print(
        f"Best train_T: {best_train_T} with mean MSE {mean_across_val_Ts[best_train_T]:.6f}"
    )

    save_horizon_prior(stats, train_Ts, val_Ts, best_train_T, dt, save_dir)
    plot_mse(train_Ts, val_Ts, stats, adaptive_stats, save_dir, dt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing models (default: reads from last_run.txt)",
    )
    parser.add_argument(
        "--max-train-T",
        type=int,
        default=None,
        help="Include only models trained with T <= this value",
    )
    parser.add_argument(
        "--max-eval-T",
        type=int,
        default=None,
        help="Maximum T for evaluation (default: max trained T in the evaluated model dir)",
    )
    parser.add_argument("--dt", type=float, default=DT, help="Time step for simulation")
    args = parser.parse_args()

    cross_validation(
        args.max_eval_T, args.dt, model_dir=args.model_dir, max_train_T=args.max_train_T
    )


if __name__ == "__main__":
    main()
