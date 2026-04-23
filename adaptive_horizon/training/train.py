import torch
from torch.utils.data import DataLoader
import argparse
from datetime import datetime
from pathlib import Path
import re

from adaptive_horizon.config import (
    LAYER_WIDTH,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_TRAJECTORIES,
    STEPS_PER_TRAJECTORY,
    DT,
    MODEL_DIR,
    LOSS_DIR,
    MAX_T,
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


def get_train_Ts(max_T: int):
    if max_T < 1:
        raise ValueError(f"--max-T must be at least 1, got {max_T}")
    return list(range(1, max_T + 1))


def get_existing_fixed_model_seeds(model_dir: Path):
    model_seeds = {}
    for model_path in model_dir.glob("mlp_T*.pt"):
        match = re.search(r"mlp_T(\d+)_seed(\d+)", model_path.name)
        if match:
            train_T = int(match.group(1))
            seed = int(match.group(2))
            model_seeds.setdefault(train_T, set()).add(seed)
    return model_seeds


def get_existing_adaptive_model_seeds(model_dir: Path):
    model_seeds = set()
    for model_path in model_dir.glob("adaptive_mlp*.pt"):
        match = re.search(r"adaptive_mlp_seed(\d+)", model_path.name)
        if match:
            model_seeds.add(int(match.group(1)))
    return model_seeds


def resolve_dirs(append: bool):
    last_run_file = MODEL_DIR / "last_run.txt"

    if append:
        if not last_run_file.exists():
            raise FileNotFoundError(
                f"Cannot append: {last_run_file} does not exist. Run training without --append first."
            )

        model_save_dir = Path(last_run_file.read_text().strip()).resolve()
        timestamp = model_save_dir.name
        loss_save_dir = LOSS_DIR / timestamp
        if not model_save_dir.exists():
            raise FileNotFoundError(
                "Cannot append: model directory referenced by last_run.txt was not found."
            )

        return timestamp, model_save_dir, loss_save_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = MODEL_DIR / timestamp
    loss_save_dir = LOSS_DIR / timestamp
    model_save_dir.mkdir(parents=True, exist_ok=True)
    loss_save_dir.mkdir(parents=True, exist_ok=True)
    last_run_file.write_text(str(model_save_dir))
    return timestamp, model_save_dir, loss_save_dir


def create_model_and_loaders(seed, adaptive, device, dt, T=None):
    """
    Create model, data loaders, optimizer, and config for training.

    Args:
        seed: Random seed
        adaptive: Whether to use adaptive temporal horizon
        device: CPU or GPU
        dt: Time step for simulation
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
            dt=dt,
            normalize=True,
            seed=seed,
        )
        val_dataset = AdaptiveLorenzDataset(
            num_trajectories=NUM_TRAJECTORIES // 5,
            steps_per_trajectory=STEPS_PER_TRAJECTORY,
            dt=dt,
            normalize=True,
            seed=seed + 1000,
        )
        collate_function = collate_fn_adaptive
    else:
        train_dataset = LorenzDataset(
            num_trajectories=NUM_TRAJECTORIES,
            steps_per_trajectory=STEPS_PER_TRAJECTORY,
            T=T,
            dt=dt,
            normalize=True,
            seed=seed,
        )
        val_dataset = LorenzDataset(
            num_trajectories=20,
            steps_per_trajectory=STEPS_PER_TRAJECTORY,
            T=T,
            dt=dt,
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
    seed,
    epochs,
    device,
    model_save_dir,
    loss_save_dir,
    dt,
    T=None,
    adaptive=False,
    save_loss_history=True,
):
    model, train_loader, val_loader, optimizer, config = create_model_and_loaders(
        seed, adaptive, device, dt, T
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

    if save_loss_history:
        save_losses(
            train_losses, val_losses, save_dir=loss_save_dir, T=T, adaptive=adaptive
        )
    save_model(model, config, seed, save_dir=model_save_dir, T=T, adaptive=adaptive)
    return train_losses, val_losses


def train_fixed_models(
    train_Ts, n_seeds, epochs, device, model_save_dir, loss_save_dir, dt, append=False
):
    seed_range = range(n_seeds)
    existing_model_seeds = (
        get_existing_fixed_model_seeds(model_save_dir) if append else {}
    )
    missing_seeds_by_T = {
        T: [
            seed
            for seed in seed_range
            if seed not in existing_model_seeds.get(T, set())
        ]
        for T in train_Ts
    }

    skipped_Ts = [
        T for T, missing_seeds in missing_seeds_by_T.items() if not missing_seeds
    ]
    if skipped_Ts:
        print(f"Skipping fixed T values with all seeds present: {skipped_Ts}")

    train_Ts = [T for T in train_Ts if missing_seeds_by_T[T]]
    if not train_Ts:
        print("No new fixed-horizon models to train")
        return

    for T in train_Ts:
        print(f"\n{'=' * 50}")
        print(f"Training models for T={T}")
        print(f"{'=' * 50}")

        train_losses = []
        val_losses = []

        missing_seeds = missing_seeds_by_T[T]
        if append and existing_model_seeds.get(T):
            existing_seeds = sorted(existing_model_seeds[T])
            print(f"Existing seeds for T={T}: {existing_seeds}")
            print(f"Training missing seeds for T={T}: {missing_seeds}")

        for seed in missing_seeds:
            print(f"\n--- Seed {seed} ---")
            train_loss, val_loss = train_single_model(
                seed,
                epochs,
                device,
                model_save_dir,
                loss_save_dir,
                dt=dt,
                T=T,
                save_loss_history=False,
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        save_losses(
            torch.tensor(train_losses, dtype=torch.float32).mean(dim=0),
            torch.tensor(val_losses, dtype=torch.float32).mean(dim=0),
            save_dir=loss_save_dir,
            T=T,
        )


def train_adaptive_models(
    n_seeds, epochs, device, model_save_dir, loss_save_dir, dt, append=False
):
    print(f"\n{'=' * 50}")
    print("Training adaptive models")
    print(f"{'=' * 50}")

    train_losses = []
    val_losses = []

    seed_range = range(n_seeds)
    existing_seeds = (
        get_existing_adaptive_model_seeds(model_save_dir) if append else set()
    )
    missing_seeds = [seed for seed in seed_range if seed not in existing_seeds]

    if append and existing_seeds:
        print(f"Existing adaptive seeds: {sorted(existing_seeds)}")

    if not missing_seeds:
        print("No new adaptive models to train")
        return

    if append:
        print(f"Training missing adaptive seeds: {missing_seeds}")

    for seed in missing_seeds:
        print(f"\n--- Adaptive Seed {seed} ---")
        train_loss, val_loss = train_single_model(
            seed,
            epochs,
            device,
            model_save_dir,
            loss_save_dir,
            dt,
            adaptive=True,
            save_loss_history=False,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    save_losses(
        torch.tensor(train_losses, dtype=torch.float32).mean(dim=0),
        torch.tensor(val_losses, dtype=torch.float32).mean(dim=0),
        save_dir=loss_save_dir,
        adaptive=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", "-e", type=int, default=EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Train a single fixed-horizon model",
    )
    parser.add_argument(
        "-T",
        type=int,
        default=1,
        help="Training horizon for --single mode",
    )
    parser.add_argument(
        "--fixed", "-f", action="store_true", help="Train only fixed T models"
    )
    parser.add_argument(
        "--adaptive", "-a", action="store_true", help="Train only adaptive models"
    )
    parser.add_argument(
        "--max-T",
        type=int,
        default=MAX_T,
        help="Train fixed-horizon models for T from 1 to this value",
    )
    parser.add_argument("--n-seeds", "-s", type=int, default=10, help="Number of seeds")
    parser.add_argument("--dt", type=float, default=DT, help="Time step for simulation")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append outputs to the run referenced by models/last_run.txt",
    )

    args = parser.parse_args()
    train_Ts = get_train_Ts(args.max_T)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Time step: {args.dt}")
    print(f"Append mode: {args.append}")

    timestamp, model_save_dir, loss_save_dir = resolve_dirs(args.append)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    loss_save_dir.mkdir(parents=True, exist_ok=True)

    if args.single:
        print(f"\n{'=' * 50}")
        print(f"Training single model for T={args.T}")
        print(f"{'=' * 50}")
        train_single_model(
            seed=0,
            epochs=args.epochs,
            device=device,
            model_save_dir=model_save_dir,
            loss_save_dir=loss_save_dir,
            dt=args.dt,
            T=args.T,
        )
    elif args.fixed:
        train_fixed_models(
            train_Ts,
            args.n_seeds,
            args.epochs,
            device,
            model_save_dir,
            loss_save_dir,
            args.dt,
            append=args.append,
        )
    elif args.adaptive:
        train_adaptive_models(
            args.n_seeds,
            args.epochs,
            device,
            model_save_dir,
            loss_save_dir,
            args.dt,
            append=args.append,
        )
    else:
        train_fixed_models(
            train_Ts,
            args.n_seeds,
            args.epochs,
            device,
            model_save_dir,
            loss_save_dir,
            args.dt,
            append=args.append,
        )
        train_adaptive_models(
            args.n_seeds,
            args.epochs,
            device,
            model_save_dir,
            loss_save_dir,
            args.dt,
            append=args.append,
        )

    print("\n" + "=" * 50)
    print("Training Complete")
    print("=" * 50)
    print(f"Models saved to {model_save_dir}")
    print(f"Losses saved to {loss_save_dir}")


if __name__ == "__main__":
    main()
