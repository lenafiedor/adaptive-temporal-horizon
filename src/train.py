import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from model.mlp import MLP, MLPConfig
from data.dataset import LorenzDataset, collate_fn


def compute_loss(model, inputs, targets, T):
    """Compute multi-step autoregressive loss per paper Equation 3."""
    x_pred = inputs
    total_loss = 0.0

    for tau in range(T):
        x_pred = model(x_pred)
        total_loss += torch.norm(x_pred - targets[:, tau], dim=1).mean()

    return total_loss / T

    # for trajectory in trajectories:
    #     M = len(trajectory)
    #     for m in range(M - T):
    #         x_pred = trajectory[m]
    #         for tau in range(1, T + 1):
    #             x_pred = model(x_pred)
    #             x_true = trajectory[m + tau]
    #             total_loss += torch.norm(x_true - x_pred)
    #         count += T
    #
    # return total_loss / count


def train(model, loader, optimizer, T, epochs, device="cpu"):
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model, inputs, targets, T)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses


def plot_losses(losses, T, save_dir="experiments/lorenz"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = save_path / f"loss_T{T}_{timestamp}.png"
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss (T={T})")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150)
    plt.close()
    
    print(f"Loss plot saved to {filename}")


if __name__ == "__main__":
    T = 4
    config = MLPConfig(
        input_size=3,
        output_size=3,
        layer_widths=[64, 64, 64],
        residual_connections=False,
        activation=nn.ReLU()
    )
    model = MLP(config, random_seed=42)
    dataset = LorenzDataset(num_trajectories=100, steps_per_trajectory=1000, T=T, normalize=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = train(model, loader, optimizer, T, epochs=100)
    plot_losses(losses, T)
