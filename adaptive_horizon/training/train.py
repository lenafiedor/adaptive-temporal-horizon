import torch
from torch.utils.data import DataLoader
import argparse
from datetime import datetime

from adaptive_horizon.config import (
    LAYER_WIDTH,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_TRAJECTORIES,
    STEPS_PER_TRAJECTORY,
    MODEL_DIR,
    LOSS_DIR,
    TRAIN_TS,
    EPOCHS,
)
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
            num_trajectories=NUM_TRAJECTORIES,
            steps_per_trajectory=STEPS_PER_TRAJECTORY,
            normalize=True,
            seed=seed,
        )
        val_dataset = AdaptiveLorenzDataset(
            num_trajectories=NUM_TRAJECTORIES / 5,
            steps_per_trajectory=STEPS_PER_TRAJECTORY,
            normalize=True,
            seed=seed + 1000,
        )
        collate_function = collate_fn_adaptive
    else:
        train_dataset = LorenzDataset(
            num_trajectories=NUM_TRAJECTORIES,
            steps_per_trajectory=STEPS_PER_TRAJECTORY,
            T=T,
            dt=0.04,
            normalize=True,
            seed=seed,
        )
        val_dataset = LorenzDataset(
            num_trajectories=20,
            steps_per_trajectory=STEPS_PER_TRAJECTORY,
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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

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

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

    return train_losses, val_losses


def train_single_model(
    seed, epochs, device, model_save_dir, loss_save_dir, T=None, adaptive=False
):
    model, train_loader, val_loader, optimizer, config = create_model_and_loaders(
        seed, adaptive, device, T
    )

    train_losses, val_losses = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=epochs,
        device=device,
        T=T,
        adaptive=adaptive,
    )

    save_losses(
        train_losses, val_losses, save_dir=loss_save_dir, T=T, adaptive=adaptive
    )
    save_model(model, config, seed, save_dir=model_save_dir, T=T, adaptive=adaptive)


def train_fixed_models(
    train_Ts, n_seeds, epochs, device, model_save_dir, loss_save_dir
):
    for T in train_Ts:
        print(f"\n{'=' * 50}")
        print(f"Training models for T={T}")
        print(f"{'=' * 50}")

        for seed in range(n_seeds):
            print(f"\n--- Seed {seed} ---")
            train_single_model(seed, epochs, device, model_save_dir, loss_save_dir, T=T)


def train_adaptive_models(n_seeds, epochs, device, model_save_dir, loss_save_dir):
    print(f"\n{'=' * 50}")
    print("Training adaptive models")
    print(f"{'=' * 50}")

    for seed in range(n_seeds):
        print(f"\n--- Adaptive Seed {seed} ---")
        train_single_model(
            seed, epochs, device, model_save_dir, loss_save_dir, adaptive=True
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", "-e", type=int, default=EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--fixed", "-f", action="store_true", help="Train only fixed T models"
    )
    parser.add_argument(
        "--adaptive", "-a", action="store_true", help="Train only adaptive models"
    )
    parser.add_argument("--n-seeds", "-s", type=int, default=10, help="Number of seeds")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = MODEL_DIR / timestamp
    loss_save_dir = LOSS_DIR / timestamp
    model_save_dir.mkdir(parents=True, exist_ok=True)
    loss_save_dir.mkdir(parents=True, exist_ok=True)

    last_run_file = MODEL_DIR / "last_run.txt"
    with open(last_run_file, "w") as f:
        f.write(timestamp)

    if args.fixed:
        train_fixed_models(
            TRAIN_TS, args.n_seeds, args.epochs, device, model_save_dir, loss_save_dir
        )
    elif args.adaptive:
        train_adaptive_models(
            args.n_seeds, args.epochs, device, model_save_dir, loss_save_dir
        )
    else:
        train_fixed_models(
            TRAIN_TS, args.n_seeds, args.epochs, device, model_save_dir, loss_save_dir
        )
        train_adaptive_models(
            args.n_seeds, args.epochs, device, model_save_dir, loss_save_dir
        )

    print("\n" + "=" * 50)
    print("Training Complete")
    print("=" * 50)
    print(f"Models saved to {model_save_dir}")
    print(f"Losses saved to {loss_save_dir}")


if __name__ == "__main__":
    main()
