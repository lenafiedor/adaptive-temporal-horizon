import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.data.dataset import LorenzDataset, collate_fn
from adaptive_horizon.training.loss import compute_g_T, validation_loss
from adaptive_horizon.visualization.plotting import plot_g_T, plot_mse_cross_validation


SAVE_DIR = Path("experiments/lorenz/evaluation")
MODELS_DIR = Path("experiments/lorenz/models")


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


def cross_validate_models(train_Ts, val_Ts, models_dir=MODELS_DIR, device="cpu"):
    max_val_T = max(val_Ts)
    eval_dataset = LorenzDataset(
        num_trajectories=100, steps_per_trajectory=1000, T=max_val_T, normalize=True
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    mse_matrix = {}

    for train_T in train_Ts:
        model_path = models_dir / f"mlp_T{train_T}.pt"
        if not model_path.exists():
            print(f"Warning: Model {model_path} not found, skipping")
            continue

        model, _ = load_model(model_path)
        model = model.to(device)
        print(f"Evaluating model trained at T={train_T}")

        mse_matrix[train_T] = {}
        for val_T in val_Ts:
            mse = validation_loss(model, eval_loader, val_T, device)
            mse_matrix[train_T][val_T] = mse

        print(f"  Min MSE: {min(mse_matrix[train_T].values()):.6f}")

    return mse_matrix


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
    plot_g_T(g_vals, SAVE_DIR, train_T=train_T, adaptive=adaptive)


def cross_validation(max_train_T, max_val_T, save_dir=SAVE_DIR):
    train_Ts = list(range(1, max_train_T + 1))
    val_Ts = list(range(1, max_val_T + 1))
    mse_matrix = cross_validate_models(train_Ts, val_Ts)
    valid_train_Ts = sorted(mse_matrix.keys())
    if valid_train_Ts:
        valid_val_Ts = [T for T in val_Ts if T in valid_train_Ts]
        plot_mse_cross_validation(mse_matrix, valid_train_Ts, valid_val_Ts, save_dir)
    else:
        print("No models found to evaluate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["grad-scaling", "cross-val"],
        default="grad-scaling",
        help="Evaluation mode: 'grad-scaling' to compute g(T), 'cross-val' to validate multiple models",
    )
    parser.add_argument(
        "--model", "-m", type=str, help="Path to saved model (only needed for g_T mode)"
    )
    parser.add_argument(
        "--max-train-T",
        type=int,
        default=20,
        help="Maximum training T (only needed for cross-eval mode)",
    )
    parser.add_argument(
        "--max-eval-T", type=int, default=20, help="Maximum T for evaluation"
    )
    args = parser.parse_args()

    if args.mode == "cross-val":
        cross_validation(args.max_train_T, args.max_eval_T, save_dir=SAVE_DIR)
    elif args.mode == "grad-scaling":
        gradient_scaling(args.model, args.max_eval_T)


if __name__ == "__main__":
    main()
