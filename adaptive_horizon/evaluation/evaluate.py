import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import re
import numpy as np

from adaptive_horizon.config import MODEL_DIR, EVAL_DIR
from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.data.dataset import LorenzDataset, collate_fn
from adaptive_horizon.training.loss import compute_g_T, validation_loss
from adaptive_horizon.visualization.plotting import plot_g_T, plot_mse


def load_model(model_path):
    checkpoint = torch.load(model_path, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    cfg = checkpoint["config"]
    config = MLPConfig(
        input_size=cfg["input_size"],
        output_size=cfg["output_size"],
        layer_widths=cfg["layer_widths"],
        residual_connections=cfg["residual_connections"],
        k=cfg.get("k"),
        activation=nn.ReLU(),
    )

    model = MLP(config, random_seed=42)
    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint


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
    model_paths, adaptive_paths, train_Ts, val_Ts, max_val_T, device="cpu"
):
    """
    Evaluate models across different validation horizons.

    Args:
        model_paths: dict of {train_T: [list of model paths]}
        adaptive_paths: list of adaptive model paths
        train_Ts: list of training horizons
        val_Ts: list of validation horizons
        max_val_T: maximum validation T (for dataset)
        device: CPU or GPU

    Returns:
        mse_matrix: dict of {train_T: {val_T: [list of MSE values]}}
        adaptive_mse: dict of {val_T: [list of MSE values]}
    """
    eval_dataset = LorenzDataset(
        num_trajectories=100, steps_per_trajectory=1000, T=max_val_T, normalize=True
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
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

            min_mse = min(mse_matrix[T][vT][-1] for vT in val_Ts)
            print(f"  Model {model_path.name}: min MSE = {min_mse:.6f}")

    adaptive_mse = {vT: [] for vT in val_Ts}

    if adaptive_paths:
        print("\nEvaluating adaptive models")
        for model_path in adaptive_paths:
            model, _ = load_model(model_path)
            model = model.to(device)

            for val_T in val_Ts:
                mse = validation_loss(model, eval_loader, val_T, device)
                adaptive_mse[val_T].append(mse)

            min_mse = min(adaptive_mse[vT][-1] for vT in val_Ts)
            print(f"  Model {model_path.name}: min MSE = {min_mse:.6f}")

    return mse_matrix, adaptive_mse


def gradient_scaling(model_path, max_T):
    model, checkpoint = load_model(model_path)
    print(f"Loaded model from {model_path}")

    adaptive = "adaptive" in model_path
    train_T = checkpoint.get("train_T") if not adaptive else None

    eval_dataset = LorenzDataset(
        num_trajectories=100, steps_per_trajectory=1000, T=max_T, normalize=True
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    T_vals = list(range(1, max_T + 1))
    g_vals = compute_g_T(model, eval_loader, T_vals)
    plot_g_T(g_vals, train_T=train_T, adaptive=adaptive)


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


def load_mse_results(save_dir=EVAL_DIR):
    """Load MSE results from CSV and reconstruct stats dicts."""
    results_file = save_dir / "mse_results.csv"

    stats = {}
    adaptive_stats = {}
    train_Ts = set()
    val_Ts = set()

    with open(results_file, "r") as f:
        next(f)
        for line in f:
            train_T, val_T, mean, std = line.strip().split(",")
            val_T = int(val_T)
            val_Ts.add(val_T)

            if train_T == "adaptive":
                adaptive_stats[val_T] = (float(mean), float(std))
            else:
                train_T = int(train_T)
                train_Ts.add(train_T)
                if train_T not in stats:
                    stats[train_T] = {}
                stats[train_T][val_T] = (float(mean), float(std))

    return stats, adaptive_stats, sorted(train_Ts), sorted(val_Ts)


def save_mse_results(stats, adaptive_stats, train_Ts, val_Ts, save_dir=EVAL_DIR):
    """
    Save MSE results to a CSV file for later plotting without re-evaluation.

    Format: train_T,val_T,mean,std
    Adaptive models have train_T = "adaptive"
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    results_file = save_dir / "mse_results.csv"
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


def cross_validation(max_val_T, save_dir=EVAL_DIR, device="cpu"):
    train_Ts = get_train_Ts()
    if not train_Ts:
        print("No models found to evaluate")
        return

    val_Ts = get_val_Ts(train_Ts, max_val_T)
    model_paths = get_model_paths(train_Ts)
    adaptive_paths = get_adaptive_paths()

    print(f"Training T values: {train_Ts}")
    print(f"Validation T values: {val_Ts}")

    mse_matrix, adaptive_mse = cross_validate_models(
        model_paths, adaptive_paths, train_Ts, val_Ts, max_val_T, device
    )

    stats, adaptive_stats = compute_statistics(
        mse_matrix, train_Ts, val_Ts, adaptive_mse
    )

    save_mse_results(stats, adaptive_stats, train_Ts, val_Ts, save_dir)
    plot_mse(train_Ts, val_Ts, stats, adaptive_stats, save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["grad-scaling", "cross-val", "plot-mse"],
        default="grad-scaling",
        help="Evaluation mode: 'grad-scaling' to compute g(T), 'cross-val' to validate multiple models",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Path to saved model (only needed for grad-scaling mode)",
    )
    parser.add_argument(
        "--max-eval-T", type=int, default=20, help="Maximum T for evaluation"
    )
    args = parser.parse_args()

    if args.mode == "cross-val":
        cross_validation(args.max_eval_T)
    elif args.mode == "plot-mse":
        stats, adaptive_stats, train_Ts, val_Ts = load_mse_results()
        plot_mse(train_Ts, val_Ts, stats, adaptive_stats, EVAL_DIR)
    elif args.mode == "grad-scaling":
        gradient_scaling(args.model, args.max_eval_T)


if __name__ == "__main__":
    main()
