import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

from model.mlp import MLP, MLPConfig
from data.dataset import LorenzDataset, collate_fn
from utils import compute_g_T, plot_g_T

SAVE_DIR = Path("experiments/lorenz")


def load_model(model_path, config):
    model = MLP(config, random_seed=42)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to saved model (.pt file)")
    parser.add_argument("--max_T", type=int, default=64, help="Maximum T for evaluation")
    args = parser.parse_args()

    config = MLPConfig(
        input_size=3,
        output_size=3,
        layer_widths=[64, 64, 64],
        residual_connections=False,
        activation=nn.ReLU()
    )

    model = load_model(args.model, config)
    print(f"Loaded model from {args.model}")

    eval_dataset = LorenzDataset(
        num_trajectories=100,
        steps_per_trajectory=1000,
        T=args.max_T,
        normalize=True
    )
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    T_vals = list(range(1, args.max_T + 1))
    print(f"Evaluating g(T) for T values: {T_vals}")

    g_vals = compute_g_T(model, eval_loader, T_vals)
    plot_g_T(g_vals, SAVE_DIR)
