import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.data.dataset import LorenzDataset, collate_fn
from adaptive_horizon.training.loss import compute_g_T
from adaptive_horizon.visualization.plotting import plot_g_T

SAVE_DIR = Path("experiments/lorenz/evaluation")


def load_model(model_path, config):
    model = MLP(config, random_seed=42)
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to saved model (.pt file)")
    parser.add_argument("--max-T", type=int, default=64, help="Maximum T for evaluation")
    parser.add_argument("--adaptive", action="store_true", help="Mark as adaptive model (for output naming)")
    args = parser.parse_args()

    config = MLPConfig(
        input_size=3,
        output_size=3,
        layer_widths=[64, 64, 64],
        residual_connections=False,
        activation=nn.ReLU()
    )

    model, checkpoint = load_model(args.model, config)
    print(f"Loaded model from {args.model}")

    adaptive = 'T_schedule' in checkpoint
    train_T = checkpoint.get('train_T') if not adaptive else checkpoint['T_schedule'][0]

    eval_dataset = LorenzDataset(num_trajectories=100, steps_per_trajectory=1000, T=args.max_T, normalize=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    T_vals = list(range(1, args.max_T + 1))
    g_vals = compute_g_T(model, eval_loader, T_vals)
    plot_g_T(g_vals, SAVE_DIR, adaptive=adaptive, train_T=train_T)


if __name__ == "__main__":
    main()
