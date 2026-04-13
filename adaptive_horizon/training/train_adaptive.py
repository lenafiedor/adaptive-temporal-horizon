import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

from adaptive_horizon.data.adaptive_dataset import AdaptiveLorenzDataset, collate_fn
from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.training.loss import adaptive_batch_loss, adaptive_validation_loss
from adaptive_horizon.visualization.plotting import save_losses, save_model


LAYER_WIDTH = 10


SAVE_DIR = Path("experiments/lorenz")


def train_adaptive(model, train_loader, val_loader, optimizer, epochs, device="cpu"):
    """
    Train the MLP model with adaptive temporal horizon scheduling based on local Lyapunov exponents.

    Args:
        model: MLP model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: PyTorch optimizer
        epochs: Total epochs
        device: CPU or GPU

    Returns:
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, targets, T in train_loader:
            inputs, targets, T = inputs.to(device), targets.to(device), T.to(device)
            optimizer.zero_grad()
            loss = adaptive_batch_loss(model, inputs, targets, T)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        val_loss = adaptive_validation_loss(model, val_loader, device)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")

    return train_losses, val_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of training_results epochs")
    parser.add_argument("--max-T", type=int, default=16, help="Maximum temporal horizon")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    config = MLPConfig(
        input_size=3,
        output_size=3,
        layer_widths=[64, 64, 64],
        residual_connections=True,
        k=1,
        activation=nn.ReLU()
    )
    model = MLP(config, random_seed=42).to(device)

    train_dataset = AdaptiveLorenzDataset(num_trajectories=100, steps_per_trajectory=1000, normalize=True)
    val_dataset = AdaptiveLorenzDataset(num_trajectories=20, steps_per_trajectory=1000, normalize=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses, val_losses = train_adaptive(model, train_loader, val_loader, optimizer, epochs=args.epochs, device=device)

    save_losses(train_losses, val_losses, SAVE_DIR, adaptive=True)
    save_model(model, config, SAVE_DIR, adaptive=True)


if __name__ == "__main__":
    main()
