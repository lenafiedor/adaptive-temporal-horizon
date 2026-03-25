import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import argparse

from model.mlp import MLP, MLPConfig
from data.dataset import LorenzDataset, collate_fn
from utils import compute_loss

SAVE_DIR = Path("experiments/lorenz")


def train(model, loader, optimizer, T, epochs, device="cpu"):
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device) # shapes: [batch_size, input_size], [batch_size, T, input_size]
            optimizer.zero_grad()
            loss = compute_loss(model, inputs, targets, T)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses


def save_losses(losses, T, save_dir):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = save_path / f"loss_T{T}_{timestamp}"
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss (T={T})")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{filename}.png", dpi=150)
    plt.close()
    
    print(f"Loss plot saved to {filename}")

    with open(f"{filename}.txt", "w") as f:
        f.write('\n'.join(map(str, losses)))
        print(f"Loss values saved to {filename}.txt")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=4, help="Temporal horizon")
    args = parser.parse_args()
    T = args.T

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
    save_losses(losses, T, SAVE_DIR)

    model_dir = SAVE_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"mlp_T{T}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_T': T
    }, model_path)
    print(f"Model saved to {model_path}")
