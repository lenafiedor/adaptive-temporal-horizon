import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import numpy as np

from adaptive_horizon.data.dataset import LorenzDataset, collate_fn
from adaptive_horizon.training.train import train, create_model_and_loaders
from adaptive_horizon.training.loss import validation_loss
from adaptive_horizon.training.evaluate import load_model, get_val_Ts
from adaptive_horizon.visualization.plotting import plot_aggregate_mse, save_model


SAVE_DIR = Path("experiments/lorenz")
MODELS_DIR = Path("experiments/lorenz/models")
TRAIN_TS = [1, 2, 4, 8, 12, 16, 20]
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]


def train_single_model(seed, epochs, device, T=None, adaptive=False):
    model, train_loader, val_loader, optimizer, config = create_model_and_loaders(
        seed, adaptive, device, T
    )

    train(model, train_loader, val_loader, optimizer, epochs=epochs, device=device, T=T, adaptive=adaptive)

    return save_model(model, config, seed, T=T, adaptive=adaptive)


def train_adaptive_models(seeds, epochs, device):
    adaptive_paths = []

    print(f"\n{'='*50}")
    print("Training adaptive models")
    print(f"{'='*50}")

    for seed in seeds:
        print(f"\n--- Adaptive Seed {seed} ---")
        model_path = train_single_model(seed, epochs, device, adaptive=True)
        adaptive_paths.append(model_path)

    return adaptive_paths


def train_all_models(train_Ts, seeds, epochs, device):
    model_paths = {T: [] for T in train_Ts}

    for T in train_Ts:
        print(f"\n{'='*50}")
        print(f"Training models for T={T}")
        print(f"{'='*50}")

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            model_path = train_single_model(T, seed, epochs, device)
            model_paths[T].append(model_path)

    adaptive_paths = train_adaptive_models(seeds, epochs, device)

    return model_paths, adaptive_paths


def evaluate_all_models(model_paths, adaptive_paths, train_Ts, max_val_T, device):
    val_Ts = get_val_Ts(train_Ts, max_val_T)

    eval_dataset = LorenzDataset(
        num_trajectories=100, steps_per_trajectory=1000, T=max_val_T, normalize=True
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    # mse_matrix[train_T][val_T] = list of MSE values across seeds
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

    # Evaluate adaptive models
    # adaptive_mse[val_T] = list of MSE values across seeds
    adaptive_mse = {vT: [] for vT in val_Ts}

    print("\nEvaluating adaptive models")
    for model_path in adaptive_paths:
        model, _ = load_model(model_path)
        model = model.to(device)

        for val_T in val_Ts:
            mse = validation_loss(model, eval_loader, val_T, device)
            adaptive_mse[val_T].append(mse)

        min_mse = min(adaptive_mse[vT][-1] for vT in val_Ts)
        print(f"  Model {model_path.name}: min MSE = {min_mse:.6f}")

    return mse_matrix, val_Ts, adaptive_mse


def compute_statistics(mse_matrix, train_Ts, val_Ts, adaptive_mse):
    """
    Compute mean and std for each (train_T, val_T) combination.

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
        adaptive_stats[val_T] = (np.mean(values), np.std(values))

    return stats, adaptive_stats


def evaluate_adaptive_models(adaptive_paths, train_Ts, max_val_T, device):
    val_Ts = get_val_Ts(train_Ts, max_val_T)

    eval_dataset = LorenzDataset(
        num_trajectories=100, steps_per_trajectory=1000, T=max_val_T, normalize=True
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    adaptive_mse = {vT: [] for vT in val_Ts}

    print("\nEvaluating adaptive models")
    for model_path in adaptive_paths:
        model, _ = load_model(model_path)
        model = model.to(device)

        for val_T in val_Ts:
            mse = validation_loss(model, eval_loader, val_T, device)
            adaptive_mse[val_T].append(mse)

        min_mse = min(adaptive_mse[vT][-1] for vT in val_Ts)
        print(f"  Model {model_path.name}: min MSE = {min_mse:.6f}")

    return adaptive_mse, val_Ts


def save_adaptive_results(adaptive_stats, val_Ts, save_dir):
    """Save adaptive-only MSE results to CSV."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results_file = save_dir / "adaptive_mse_results.csv"
    with open(results_file, "w") as f:
        f.write("val_T,mean,std\n")
        for val_T in val_Ts:
            mean, std = adaptive_stats[val_T]
            f.write(f"{val_T},{mean},{std}\n")

    print(f"Adaptive MSE results saved to {results_file}")


def save_mse_results(stats, adaptive_stats, train_Ts, val_Ts, save_dir):
    """
    Save MSE results to a CSV file for later plotting without re-evaluation.

    Format: train_T,val_T,mean,std
    Adaptive models have train_T = "adaptive"
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results_file = save_dir / "mse_results.csv"
    with open(results_file, "w") as f:
        f.write("train_T,val_T,mean,std\n")
        for train_T in train_Ts:
            for val_T in val_Ts:
                mean, std = stats[train_T][val_T]
                f.write(f"{train_T},{val_T},{mean},{std}\n")

        for val_T in val_Ts:
            mean, std = adaptive_stats[val_T]
            f.write(f"adaptive,{val_T},{mean},{std}\n")

    print(f"MSE results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "both"],
        default="both",
        help="Mode: train models, evaluate existing models, or both",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--max-eval-T", type=int, default=100, help="Maximum T for evaluation"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory with trained models (for evaluate mode)",
    )
    parser.add_argument(
        "--adaptive-only",
        action="store_true",
        help="Train and evaluate only adaptive models (skip fixed-T models)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    save_dir = SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ["train", "both"]:
        if args.adaptive_only:
            model_paths = {T: [] for T in TRAIN_TS}
            adaptive_paths = train_adaptive_models(SEEDS, args.epochs, device)
        else:
            model_paths, adaptive_paths = train_all_models(
                TRAIN_TS, SEEDS, args.epochs, device
            )

            paths_file = MODELS_DIR / "model_paths.txt"
            with open(paths_file, "w") as f:
                for T in TRAIN_TS:
                    for path in model_paths[T]:
                        f.write(f"{T},{path.name}\n")

        adaptive_paths_file = MODELS_DIR / "adaptive_paths.txt"
        with open(adaptive_paths_file, "w") as f:
            for path in adaptive_paths:
                f.write(f"{path.name}\n")

    if args.mode in ["evaluate", "both"]:
        if args.mode == "evaluate":
            models_dir = Path(args.models_dir) if args.models_dir else MODELS_DIR
            model_paths = {T: [] for T in TRAIN_TS}
            paths_file = models_dir / "model_paths.txt"
            if paths_file.exists():
                with open(paths_file, "r") as f:
                    for line in f:
                        T, filename = line.strip().split(",", 1)
                        model_paths[int(T)].append(models_dir / filename)

            adaptive_paths = []
            adaptive_paths_file = models_dir / "adaptive_paths.txt"
            if adaptive_paths_file.exists():
                with open(adaptive_paths_file, "r") as f:
                    for line in f:
                        adaptive_paths.append(models_dir / line.strip())

        if args.adaptive_only:
            adaptive_mse, val_Ts = evaluate_adaptive_models(
                adaptive_paths, TRAIN_TS, args.max_eval_T, device
            )
            adaptive_stats = {}
            for val_T in val_Ts:
                values = np.array(adaptive_mse[val_T])
                adaptive_stats[val_T] = (np.mean(values), np.std(values))

            # Save adaptive-only results
            save_adaptive_results(adaptive_stats, val_Ts, save_dir)

            print("\n" + "=" * 50)
            print("Adaptive Results Summary")
            print("=" * 50)
            for val_T in val_Ts:
                mean, std = adaptive_stats[val_T]
                print(f"val_T={val_T:3d}: mean={mean:.6f}, std={std:.6f}")

            adaptive_min_mean = min(adaptive_stats[vT][0] for vT in val_Ts)
            print(f"Adaptive min mean MSE = {adaptive_min_mean:.6f}")
        else:
            mse_matrix, val_Ts, adaptive_mse = evaluate_all_models(
                model_paths, adaptive_paths, TRAIN_TS, args.max_eval_T, device
            )
            stats, adaptive_stats = compute_statistics(
                mse_matrix, TRAIN_TS, val_Ts, adaptive_mse
            )

            # Save MSE results to CSV for later plotting
            save_mse_results(stats, adaptive_stats, TRAIN_TS, val_Ts, save_dir)

            print("\n" + "=" * 50)
            print("Results Summary")
            print("=" * 50)
            for T in TRAIN_TS:
                min_mse_mean = min(stats[T][vT][0] for vT in val_Ts)
                print(f"T={T:2d}: min mean MSE = {min_mse_mean:.6f}")

            adaptive_min_mean = min(adaptive_stats[vT][0] for vT in val_Ts)
            print(f"Adaptive: min mean MSE = {adaptive_min_mean:.6f}")

            plot_aggregate_mse(TRAIN_TS, val_Ts, stats, adaptive_stats, save_dir)


if __name__ == "__main__":
    main()
