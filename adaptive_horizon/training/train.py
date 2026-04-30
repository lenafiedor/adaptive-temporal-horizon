import torch
from torch.utils.data import DataLoader
import argparse
from datetime import datetime
from pathlib import Path
import re

import adaptive_horizon.config as config
from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.data.dataset import LorenzDataset, collate_fn
from adaptive_horizon.data.adaptive_dataset import (
    AdaptiveHorizonLorenzDataset,
    WeightedLossLorenzDataset,
    collate_fn_adaptive_horizon,
    collate_fn_weighted_loss,
    default_adaptive_T_max,
)
from adaptive_horizon.training.loss import (
    adaptive_batch_loss,
    adaptive_validation_loss,
    batch_loss,
    lle_weighted_batch_loss,
    lle_weighted_validation_loss,
    validation_loss,
)
from adaptive_horizon.visualization.plotting import save_losses, save_model

ADAPTIVE_HORIZON_METHOD = "adaptive-horizon"
WEIGHTED_LOSS_METHOD = "weighted-loss"


def create_optimizer(optimizer_name, model):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


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


def get_existing_adaptive_model_seeds(model_dir: Path, method=ADAPTIVE_HORIZON_METHOD):
    model_seeds = set()
    for model_path in model_dir.glob("adaptive_mlp*.pt"):
        match = re.search(r"adaptive_mlp_seed(\d+)", model_path.name)
        if match:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            checkpoint_method = (
                checkpoint.get("metadata", {})
                .get("adaptive", {})
                .get("method", ADAPTIVE_HORIZON_METHOD)
            )
            if checkpoint_method == method:
                model_seeds.add(int(match.group(1)))
    return model_seeds


def resolve_dirs(dt, append: bool):
    last_run_file = config.MODEL_DIR / "last_run.txt"

    if append:
        if not last_run_file.exists():
            raise FileNotFoundError(
                f"Cannot append: {last_run_file} does not exist. Run training without --append first."
            )

        model_save_dir = Path(last_run_file.read_text().strip()).resolve()
        timestamp = model_save_dir.name
        loss_save_dir = config.LOSS_DIR / timestamp
        if not model_save_dir.exists():
            raise FileNotFoundError(
                "Cannot append: model directory referenced by last_run.txt was not found."
            )

        return timestamp, model_save_dir, loss_save_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dt_formatted = str(dt).split(".")[1]
    model_save_dir = config.MODEL_DIR / f"dt_{dt_formatted}_{timestamp}"
    loss_save_dir = config.LOSS_DIR / f"dt_{dt_formatted}_{timestamp}"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    loss_save_dir.mkdir(parents=True, exist_ok=True)
    last_run_file.write_text(str(model_save_dir))

    return timestamp, model_save_dir, loss_save_dir


def create_model_and_loaders(
    seed,
    adaptive,
    device,
    dt,
    T=None,
    method=ADAPTIVE_HORIZON_METHOD,
    optimizer_name=config.OPTIMIZER,
    batch_size=config.BATCH_SIZE,
    ftle_window=config.WINDOW_SIZE,
):
    """
    Create model, data loaders, optimizer, and config for training.

    Args:
        seed: Random seed
        adaptive: Whether to use adaptive temporal horizon
        device: CPU or GPU
        dt: Time step for simulation
        T: Temporal horizon (ignored if adaptive=True)
        method: Adaptive training method
        optimizer_name: Optimizer name
        batch_size: Batch size for data loaders
        ftle_window: Forward FTLE window for adaptive training

    Returns:
        model, train_loader, val_loader, optimizer, config, metadata
    """
    mlp_config = MLPConfig(
        input_size=config.INPUT_DIM,
        output_size=config.INPUT_DIM,
        layer_widths=[config.LAYER_WIDTH, config.LAYER_WIDTH, config.LAYER_WIDTH],
        residual_connections=True,
        k=1,
        activation=torch.nn.ReLU(),
    )
    model = MLP(mlp_config, random_seed=seed).to(device)
    metadata = {}

    if adaptive:
        if method == ADAPTIVE_HORIZON_METHOD:
            train_dataset = AdaptiveHorizonLorenzDataset(dt=dt, seed=seed)
            val_dataset = AdaptiveHorizonLorenzDataset(
                num_trajectories=config.NUM_TRAJECTORIES // 5,
                dt=dt,
                seed=seed + 1000,
                normalization_stats=train_dataset.normalization_stats,
            )
            collate_function = collate_fn_adaptive_horizon
        elif method == WEIGHTED_LOSS_METHOD:
            T = T if T is not None else default_adaptive_T_max(dt)
            train_dataset = WeightedLossLorenzDataset(
                dt=dt,
                T_max=T,
                ftle_window=ftle_window,
                seed=seed,
            )
            val_dataset = WeightedLossLorenzDataset(
                num_trajectories=config.NUM_TRAJECTORIES // 5,
                dt=dt,
                T_max=T,
                ftle_window=ftle_window,
                seed=seed + 1000,
                normalization_stats=train_dataset.normalization_stats,
            )
            collate_function = collate_fn_weighted_loss
        else:
            raise ValueError(f"Unsupported adaptive method: {method}")

        metadata["adaptive"] = {
            "method": method,
            "dt": dt,
        }
        if method == WEIGHTED_LOSS_METHOD:
            metadata["adaptive"]["T_max"] = T
            metadata["adaptive"]["ftle_window"] = ftle_window
        else:
            metadata["adaptive"].update(
                {
                    "base_T": train_dataset.base_T,
                    "min_T": train_dataset.min_T,
                    "max_T": train_dataset.max_T,
                    "alpha": train_dataset.alpha,
                }
            )
    else:
        train_dataset = LorenzDataset(
            T=T,
            dt=dt,
            seed=seed,
        )
        val_dataset = LorenzDataset(
            num_trajectories=config.NUM_TRAJECTORIES // 5,
            T=T,
            dt=dt,
            seed=seed + 1000,
            normalization_stats=train_dataset.normalization_stats,
        )
        collate_function = collate_fn

    metadata["normalization_stats"] = train_dataset.normalization_stats

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_function
    )
    optimizer = create_optimizer(optimizer_name, model)

    return model, train_loader, val_loader, optimizer, mlp_config, metadata


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs,
    device=config.DEVICE,
    T=None,
    adaptive=False,
    method=ADAPTIVE_HORIZON_METHOD,
    dt=config.DT,
    rho=config.RHO,
    temperature=config.TEMPERATURE,
    weight_floor=config.WEIGHT_FLOOR,
    anchor_alpha=config.ANCHOR_ALPHA,
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
        method: Adaptive training method
        dt: Time step used by adaptive predictability weights
        rho: Predictability budget threshold
        temperature: Sigmoid softness for adaptive weights
        weight_floor: Minimum unnormalized adaptive weight
        anchor_alpha: Weight of the one-step anchor loss

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
                if method == ADAPTIVE_HORIZON_METHOD:
                    T_values = rest[0].to(device) if rest else None
                    loss = adaptive_batch_loss(model, inputs, targets, T_values)
                elif method == WEIGHTED_LOSS_METHOD:
                    lambda_scores = rest[0].to(device) if rest else None
                    loss = lle_weighted_batch_loss(
                        model,
                        inputs,
                        targets,
                        lambda_scores,
                        dt=dt,
                        rho=rho,
                        temperature=temperature,
                        floor=weight_floor,
                        anchor_alpha=anchor_alpha,
                    )
                else:
                    raise ValueError(f"Unsupported adaptive method: {method}")
            else:
                loss = batch_loss(model, inputs, targets, T)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if not adaptive:
            val_loss = validation_loss(model, val_loader, T, device)
        elif method == ADAPTIVE_HORIZON_METHOD:
            val_loss = adaptive_validation_loss(model, val_loader, device)
        elif method == WEIGHTED_LOSS_METHOD:
            val_loss = lle_weighted_validation_loss(
                model,
                val_loader,
                dt=dt,
                device=device,
                rho=rho,
                temperature=temperature,
                floor=weight_floor,
                anchor_alpha=anchor_alpha,
            )
        else:
            raise ValueError(f"Unsupported adaptive method: {method}")
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
    method=ADAPTIVE_HORIZON_METHOD,
    save_loss_history=True,
    optimizer_name=config.OPTIMIZER,
    batch_size=config.BATCH_SIZE,
    ftle_window=config.WINDOW_SIZE,
    rho=config.RHO,
    temperature=config.TEMPERATURE,
    weight_floor=config.WEIGHT_FLOOR,
    anchor_alpha=config.ANCHOR_ALPHA,
):
    model, train_loader, val_loader, optimizer, mlp_config, metadata = (
        create_model_and_loaders(
            seed,
            adaptive,
            device,
            dt,
            T,
            method,
            optimizer_name,
            batch_size,
            ftle_window,
        )
    )
    if adaptive:
        if method == WEIGHTED_LOSS_METHOD:
            T = metadata["adaptive"]["T_max"]
            metadata["adaptive"].update(
                {
                    "rho": rho,
                    "temperature": temperature,
                    "weight_floor": weight_floor,
                    "anchor_alpha": anchor_alpha,
                }
            )
        else:
            T = None

    train_losses, val_losses = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=epochs,
        device=device,
        T=T,
        adaptive=adaptive,
        method=method,
        dt=dt,
        rho=rho,
        temperature=temperature,
        weight_floor=weight_floor,
        anchor_alpha=anchor_alpha,
    )

    if save_loss_history:
        save_losses(
            train_losses, val_losses, save_dir=loss_save_dir, T=T, adaptive=adaptive
        )
    save_model(
        model,
        config,
        seed,
        save_dir=model_save_dir,
        T=T,
        adaptive=adaptive,
        metadata=metadata,
    )
    return train_losses, val_losses


def train_fixed_models(
    train_Ts,
    n_seeds,
    epochs,
    device,
    model_save_dir,
    loss_save_dir,
    dt=config.DT,
    optimizer_name=config.OPTIMIZER,
    batch_size=config.BATCH_SIZE,
    append=False,
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
                optimizer_name=optimizer_name,
                batch_size=batch_size,
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
    n_seeds,
    epochs,
    device,
    model_save_dir,
    loss_save_dir,
    dt=config.DT,
    optimizer_name=config.OPTIMIZER,
    batch_size=config.BATCH_SIZE,
    method=ADAPTIVE_HORIZON_METHOD,
    T_max=None,
    ftle_window=config.WINDOW_SIZE,
    rho=config.RHO,
    temperature=config.TEMPERATURE,
    weight_floor=config.WEIGHT_FLOOR,
    anchor_alpha=config.ANCHOR_ALPHA,
    append=False,
):
    print(f"\n{'=' * 50}")
    print(f"Training adaptive models ({method})")
    print(f"{'=' * 50}")

    train_losses = []
    val_losses = []

    seed_range = range(n_seeds)
    existing_seeds = (
        get_existing_adaptive_model_seeds(model_save_dir, method) if append else set()
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
            T=T_max,
            adaptive=True,
            method=method,
            save_loss_history=False,
            optimizer_name=optimizer_name,
            batch_size=batch_size,
            ftle_window=ftle_window,
            rho=rho,
            temperature=temperature,
            weight_floor=weight_floor,
            anchor_alpha=anchor_alpha,
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
        "--epochs",
        "-e",
        type=int,
        default=config.EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Train a single model; combine with --adaptive for adaptive training",
    )
    parser.add_argument(
        "-T",
        type=int,
        default=1,
        help="Training horizon for fixed --single mode",
    )
    parser.add_argument(
        "--fixed", "-f", action="store_true", help="Train only fixed T models"
    )
    parser.add_argument(
        "--adaptive", "-a", action="store_true", help="Train only adaptive models"
    )
    parser.add_argument(
        "--method",
        choices=[ADAPTIVE_HORIZON_METHOD, WEIGHTED_LOSS_METHOD],
        default=ADAPTIVE_HORIZON_METHOD,
        help="Adaptive training method used with --adaptive",
    )
    parser.add_argument(
        "--max-T",
        type=int,
        default=config.MAX_T,
        help="Train fixed-horizon models for T from 1 to this value",
    )
    parser.add_argument(
        "--n-seeds", "-s", type=int, default=config.NUM_SEEDS, help="Number of seeds"
    )
    parser.add_argument(
        "--dt", type=float, default=config.DT, help="Time step for simulation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size for training and validation loaders",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=config.OPTIMIZER,
        choices=["sgd", "adam", "adamw"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append outputs to the run referenced by models/last_run.txt",
    )
    parser.add_argument(
        "--adaptive-T-max",
        type=int,
        default=None,
        help="Shared rollout horizon for LLE-weighted adaptive training",
    )
    parser.add_argument(
        "--ftle-window",
        type=int,
        default=config.WINDOW_SIZE,
        help="Forward FTLE window for adaptive lambda scores",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=config.RHO,
        help="Predictability budget threshold for adaptive weights",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.TEMPERATURE,
        help="Sigmoid temperature for adaptive weights",
    )
    parser.add_argument(
        "--weight-floor",
        type=float,
        default=config.WEIGHT_FLOOR,
        help="Minimum unnormalized adaptive rollout weight",
    )
    parser.add_argument(
        "--anchor-alpha",
        type=float,
        default=config.ANCHOR_ALPHA,
        help="One-step anchor fraction in the adaptive loss",
    )

    args = parser.parse_args()
    train_Ts = get_train_Ts(args.max_T)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Time step: {args.dt}")
    print(f"Batch size: {args.batch_size}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Adaptive method: {args.method}")
    print(f"Append mode: {args.append}")

    timestamp, model_save_dir, loss_save_dir = resolve_dirs(args.dt, args.append)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    loss_save_dir.mkdir(parents=True, exist_ok=True)

    if args.single:
        print(f"\n{'=' * 50}")
        if args.adaptive:
            print("Training single adaptive model")
        else:
            print(f"Training single model for T={args.T}")
        print(f"{'=' * 50}")
        train_single_model(
            seed=0,
            epochs=args.epochs,
            device=device,
            model_save_dir=model_save_dir,
            loss_save_dir=loss_save_dir,
            dt=args.dt,
            T=args.adaptive_T_max if args.adaptive else args.T,
            adaptive=args.adaptive,
            method=args.method,
            optimizer_name=args.optimizer,
            batch_size=args.batch_size,
            ftle_window=args.ftle_window,
            rho=args.rho,
            temperature=args.temperature,
            weight_floor=args.weight_floor,
            anchor_alpha=args.anchor_alpha,
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
            args.optimizer,
            args.batch_size,
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
            args.optimizer,
            args.batch_size,
            method=args.method,
            T_max=args.adaptive_T_max,
            ftle_window=args.ftle_window,
            rho=args.rho,
            temperature=args.temperature,
            weight_floor=args.weight_floor,
            anchor_alpha=args.anchor_alpha,
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
            args.optimizer,
            args.batch_size,
            append=args.append,
        )
        train_adaptive_models(
            args.n_seeds,
            args.epochs,
            device,
            model_save_dir,
            loss_save_dir,
            args.dt,
            args.optimizer,
            args.batch_size,
            method=args.method,
            T_max=args.adaptive_T_max,
            ftle_window=args.ftle_window,
            rho=args.rho,
            temperature=args.temperature,
            weight_floor=args.weight_floor,
            anchor_alpha=args.anchor_alpha,
            append=args.append,
        )

    print("\n" + "=" * 50)
    print("Training Complete")
    print("=" * 50)
    print(f"Models saved to {model_save_dir}")
    print(f"Losses saved to {loss_save_dir}")


if __name__ == "__main__":
    main()
