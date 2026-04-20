import torch
from torch.utils.data import DataLoader
import argparse

from adaptive_horizon.config import SEEDS, LAYER_WIDTH, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY
from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.data.dataset import LorenzDataset, collate_fn
from adaptive_horizon.data.adaptive_dataset import (
    AdaptiveLorenzDataset,
    collate_fn_adaptive,
)
from adaptive_horizon.training.loss import (
    batch_loss,
    validation_loss,
    adaptive_batch_loss,
    adaptive_validation_loss,
)
from adaptive_horizon.visualization.plotting import save_losses, save_model


def create_model_and_loaders(seed, adaptive, device, T=None):
    """
    Create model, data loaders, optimizer, and config for training.

    Args:
        seed: Random seed
        adaptive: Whether to use adaptive temporal horizon
        device: CPU or GPU
        T: Temporal horizon (ignored if adaptive=True)

    Returns:
        model, train_loader, val_loader, optimizer, config
    """
    config = MLPConfig(
        input_size=3,
        output_size=3,
        layer_widths=[LAYER_WIDTH, LAYER_WIDTH, LAYER_WIDTH],
        residual_connections=True,
        k=1,
        activation=torch.nn.ReLU(),
    )
    model = MLP(config, random_seed=seed).to(device)

    if adaptive:
        train_dataset = AdaptiveLorenzDataset(
            num_trajectories=100, steps_per_trajectory=1000, normalize=True, seed=seed
        )
        val_dataset = AdaptiveLorenzDataset(
            num_trajectories=20,
            steps_per_trajectory=1000,
            normalize=True,
            seed=seed + 1000,
        )
        collate_function = collate_fn_adaptive
    else:
        train_dataset = LorenzDataset(
            num_trajectories=100,
            steps_per_trajectory=1000,
            T=T,
            dt=0.04,
            normalize=True,
            seed=seed,
        )
        val_dataset = LorenzDataset(
            num_trajectories=20,
            steps_per_trajectory=1000,
            T=T,
            normalize=True,
            seed=seed + 1000,
        )
        collate_function = collate_fn

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_function
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_function
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    return model, train_loader, val_loader, optimizer, config


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs,
    device="cpu",
    T=None,
    adaptive=False,
):
    """
    Train model with fixed temporal horizon T.

    Args:
        model: MLP model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: PyTorch optimizer
        epochs: Total epochs
        device: CPU or GPU
        T: Temporal horizon (only if non-adaptive)
        adaptive: Whether to use the adaptive temporal horizon

    Returns:
        losses: List of training_results losses
        val_losses: List of validation losses
    """
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            inputs, targets, *rest = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if adaptive:
                T_values = rest[0].to(device) if rest else None
                loss = adaptive_batch_loss(model, inputs, targets, T_values)
            else:
                loss = batch_loss(model, inputs, targets, T)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        val_loss = (
            validation_loss(model, val_loader, T, device)
            if not adaptive
            else adaptive_validation_loss(model, val_loader, device)
        )
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")

    return train_losses, val_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adaptive", action="store_true", help="Use adaptive temporal horizon"
    )
    parser.add_argument("-T", type=int, default=1, help="Temporal horizon")
    parser.add_argument(
        "--epochs", "-e", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--seed", "-s", type=int, default=SEEDS[0], help="Random seed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    model, train_loader, val_loader, optimizer, config = create_model_and_loaders(
        args.seed, args.adaptive, device, args.T
    )

    train_losses, val_losses = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=args.epochs,
        device=device,
        T=args.T,
        adaptive=args.adaptive,
    )

    save_losses(train_losses, val_losses, T=args.T, adaptive=args.adaptive)
    save_model(model, config, args.seed, T=args.T, adaptive=args.adaptive)


if __name__ == "__main__":
    main()
