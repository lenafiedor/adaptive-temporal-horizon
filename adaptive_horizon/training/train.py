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
    compute_g_T,
    lle_weighted_batch_loss,
    lle_weighted_validation_loss,
    validation_loss,
)
from adaptive_horizon.visualization.plotting import (
    save_gradient_history,
    save_gradients_histogram,
    save_losses,
    save_model,
)
from adaptive_horizon.utils import format_dt

ADAPTIVE_HORIZON = "adaptive-horizon"
WEIGHTED_LOSS = "weighted-loss"


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


def get_existing_adaptive_model_seeds(model_dir: Path):
    model_seeds = set()
    for model_path in model_dir.glob("adaptive_mlp*.pt"):
        match = re.search(r"adaptive_mlp_seed(\d+)", model_path.name)
        if match:
            model_seeds.add(int(match.group(1)))
    return model_seeds


def resolve_dirs(dt, append: bool, debug: bool):
    last_run_file = config.MODEL_DIR / "last_run.txt"

    if append:
        if not last_run_file.exists():
            raise FileNotFoundError(
                f"Cannot append: {last_run_file} does not exist. Run training without --append first."
            )

        model_save_dir = Path(last_run_file.read_text().strip()).resolve()
        timestamp = model_save_dir.name
        loss_save_dir = config.LOSS_DIR / timestamp
        if debug:
            loss_save_dir.mkdir(parents=True, exist_ok=True)
        if not model_save_dir.exists():
            raise FileNotFoundError(
                "Cannot append: model directory referenced by last_run.txt was not found."
            )
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_dir = config.MODEL_DIR / f"dt_{format_dt(dt)}_{timestamp}"
        model_save_dir.mkdir(parents=True, exist_ok=True)
        loss_save_dir = config.LOSS_DIR / f"dt_{format_dt(dt)}_{timestamp}"
        if debug:
            loss_save_dir.mkdir(parents=True, exist_ok=True)
        last_run_file.write_text(str(model_save_dir))

    return timestamp, model_save_dir, loss_save_dir


def create_model_and_loaders(
    seed,
    adaptive,
    device,
    dt,
    T=None,
    adaptive_method=ADAPTIVE_HORIZON,
    optimizer_name=config.OPTIMIZER,
    batch_size=config.BATCH_SIZE,
    history_window=config.HISTORY_WINDOW,
    ftle_window=config.FTLE_WINDOW,
    var=config.VARIANCE,
    debug=False,
):
    """
    Create model, data loaders, optimizer, and config for training.

    Args:
        seed: Random seed
        adaptive: Whether to use adaptive temporal horizon
        device: CPU or GPU
        dt: Time step for simulation
        T: Temporal horizon (ignored if adaptive=True)
        adaptive_method: Adaptive training method
        optimizer_name: Optimizer name
        batch_size: Batch size for data loaders
        history_window: Number of past trajectory states in each model input
        ftle_window: Forward FTLE window for weighted-loss training
        var: Variance of the adaptive horizon
        debug: Whether adaptive datasets should write T values and Lyapunov exponents

    Returns:
        model, train_loader, val_loader, optimizer, config, metadata
    """
    mlp_config = MLPConfig(
        input_size=config.INPUT_DIM * history_window,
        output_size=config.INPUT_DIM,
        layer_widths=[config.LAYER_WIDTH, config.LAYER_WIDTH, config.LAYER_WIDTH],
        residual_connections=True,
        k=1,
        activation=torch.nn.ReLU(),
    )
    model = MLP(mlp_config, random_seed=seed).to(device)
    burn_in_steps = config.resolve_burn_in_steps(dt)
    metadata = {
        "dt": dt,
        "burn_in_time": config.BURN_IN_TIME,
        "burn_in_steps": burn_in_steps,
    }

    if adaptive:
        if adaptive_method == ADAPTIVE_HORIZON:
            train_dataset = AdaptiveHorizonLorenzDataset(
                dt=dt,
                seed=seed,
                burn_in=burn_in_steps,
                var=var,
                history_window=history_window,
                debug=debug,
            )
            val_dataset = AdaptiveHorizonLorenzDataset(
                num_trajectories=config.NUM_TRAJECTORIES // 5,
                dt=dt,
                seed=seed + 1000,
                burn_in=burn_in_steps,
                var=var,
                history_window=history_window,
                normalization_stats=train_dataset.normalization_stats,
                debug=debug,
            )
            collate_function = collate_fn_adaptive_horizon
        elif adaptive_method == WEIGHTED_LOSS:
            T = T if T is not None else default_adaptive_T_max(dt)
            train_dataset = WeightedLossLorenzDataset(
                dt=dt,
                T_max=T,
                ftle_window=ftle_window,
                history_window=history_window,
                seed=seed,
                burn_in=burn_in_steps,
                debug=debug,
            )
            val_dataset = WeightedLossLorenzDataset(
                num_trajectories=config.NUM_TRAJECTORIES // 5,
                dt=dt,
                T_max=T,
                ftle_window=ftle_window,
                history_window=history_window,
                seed=seed + 1000,
                burn_in=burn_in_steps,
                normalization_stats=train_dataset.normalization_stats,
                debug=debug,
            )
            collate_function = collate_fn_weighted_loss
        else:
            raise ValueError(f"Unsupported adaptive method: {adaptive_method}")

        metadata["adaptive"] = {
            "method": adaptive_method,
        }
        if adaptive_method == WEIGHTED_LOSS:
            metadata["adaptive"]["T_max"] = T
            metadata["adaptive"]["ftle_window"] = ftle_window
        else:
            metadata["adaptive"].update(
                {
                    "variance": var,
                    "base_T": train_dataset.base_T,
                    "min_T": train_dataset.min_T,
                    "max_T": train_dataset.max_T,
                }
            )
    else:
        train_dataset = LorenzDataset(
            T=T,
            dt=dt,
            seed=seed,
            burn_in=burn_in_steps,
            history_window=history_window,
        )
        val_dataset = LorenzDataset(
            num_trajectories=config.NUM_TRAJECTORIES // 5,
            T=T,
            dt=dt,
            seed=seed + 1000,
            burn_in=burn_in_steps,
            history_window=history_window,
            normalization_stats=train_dataset.normalization_stats,
        )
        collate_function = collate_fn

    metadata["history_window"] = history_window
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
    adaptive_method=ADAPTIVE_HORIZON,
    dt=config.DT,
    debug=False,
    save_dir=config.LOSS_DIR,
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
        adaptive_method: Adaptive training method
        dt: Time step used by adaptive predictability weights
        debug: Whether to save g(T) histograms during training
        save_dir: Directory to save gradient histograms

    Returns:
        losses: List of training_results losses
        val_losses: List of validation losses
    """
    train_losses = []
    val_losses = []
    gradient_history = []
    if debug:
        debug_dataset = LorenzDataset(
            num_trajectories=config.NUM_TRAJECTORIES,
            steps_per_trajectory=config.STEPS_PER_TRAJECTORY,
            T=config.MAX_EVAL_T,
            dt=dt,
            normalize=True,
            seed=config.EVAL_SEED,
            burn_in=config.resolve_burn_in_steps(dt),
            history_window=getattr(model, "history_window", config.HISTORY_WINDOW),
            normalization_stats=getattr(model, "normalization_stats", None),
        )
        debug_loader = DataLoader(
            debug_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
        )
        debug_T_vals = [2, 4, 6, 8, 10, 15, 20]

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            inputs, targets, *rest = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if adaptive:
                if adaptive_method == ADAPTIVE_HORIZON:
                    T_values = rest[0].to(device) if rest else None
                    loss = adaptive_batch_loss(model, inputs, targets, T_values)
                elif adaptive_method == WEIGHTED_LOSS:
                    lambda_scores = rest[0].to(device) if rest else None
                    loss = lle_weighted_batch_loss(
                        model, inputs, targets, lambda_scores, dt=dt
                    )
                else:
                    raise ValueError(f"Unsupported adaptive method: {adaptive_method}")
            else:
                loss = batch_loss(model, inputs, targets, T)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if not adaptive:
            val_loss = validation_loss(model, val_loader, T, device)
        elif adaptive_method == ADAPTIVE_HORIZON:
            val_loss = adaptive_validation_loss(model, val_loader, device)
        elif adaptive_method == WEIGHTED_LOSS:
            val_loss = lle_weighted_validation_loss(
                model, val_loader, dt=dt, device=device
            )
        else:
            raise ValueError(f"Unsupported adaptive method: {adaptive_method}")
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}"
            )
            if debug:
                gradients = compute_g_T(
                    model, debug_loader, debug_T_vals, device=device, per_batch=True
                )
                gradient_history.append((epoch, gradients))
                save_gradients_histogram(
                    gradients,
                    save_dir=save_dir,
                    epoch=epoch,
                    train_T=T,
                    dt=dt,
                    adaptive=adaptive,
                )
    if debug:
        save_gradient_history(
            gradient_history, save_dir=save_dir, train_T=T, dt=dt, adaptive=adaptive
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
    adaptive_method=ADAPTIVE_HORIZON,
    optimizer_name=config.OPTIMIZER,
    batch_size=config.BATCH_SIZE,
    history_window=config.HISTORY_WINDOW,
    ftle_window=config.FTLE_WINDOW,
    var=config.VARIANCE,
    debug=False,
):
    model, train_loader, val_loader, optimizer, mlp_config, metadata = (
        create_model_and_loaders(
            seed,
            adaptive,
            device,
            dt,
            T,
            adaptive_method,
            optimizer_name,
            batch_size,
            history_window,
            ftle_window,
            var,
            debug,
        )
    )
    if adaptive:
        if adaptive_method == WEIGHTED_LOSS:
            T = metadata["adaptive"]["T_max"]
            metadata["adaptive"].update(
                {
                    "ftle_window": ftle_window,
                    "rho": config.RHO,
                    "temperature": config.TEMPERATURE,
                    "weight_floor": config.WEIGHT_FLOOR,
                    "anchor_alpha": config.ANCHOR_ALPHA,
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
        adaptive_method=adaptive_method,
        dt=dt,
        debug=debug,
        save_dir=loss_save_dir,
    )

    save_model(
        model,
        mlp_config,
        seed,
        save_dir=model_save_dir,
        T=T,
        adaptive=adaptive,
        metadata=metadata,
        var=var if adaptive_method == ADAPTIVE_HORIZON else None,
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
    history_window=config.HISTORY_WINDOW,
    append=False,
    debug=False,
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
                optimizer_name=optimizer_name,
                batch_size=batch_size,
                history_window=history_window,
                debug=debug,
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        if debug:
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
    adaptive_method=ADAPTIVE_HORIZON,
    T_max=None,
    history_window=config.HISTORY_WINDOW,
    ftle_window=config.FTLE_WINDOW,
    var=config.VARIANCE,
    append=False,
    debug=False,
):
    print(f"\n{'=' * 50}")
    print(f"Training adaptive models ({adaptive_method})")
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
            T=T_max,
            adaptive=True,
            adaptive_method=adaptive_method,
            optimizer_name=optimizer_name,
            batch_size=batch_size,
            history_window=history_window,
            ftle_window=ftle_window,
            var=var,
            debug=debug,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    if debug:
        save_losses(
            torch.tensor(train_losses, dtype=torch.float32).mean(dim=0),
            torch.tensor(val_losses, dtype=torch.float32).mean(dim=0),
            save_dir=loss_save_dir,
            adaptive=True,
            var=var if adaptive_method == ADAPTIVE_HORIZON else None,
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
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--fixed", "-f", action="store_true", help="Train only fixed T models"
    )
    mode_group.add_argument(
        "--adaptive", "-a", action="store_true", help="Train only adaptive models"
    )
    parser.add_argument(
        "--adaptive-method",
        choices=[ADAPTIVE_HORIZON, WEIGHTED_LOSS],
        default=ADAPTIVE_HORIZON,
        help="Adaptive training method used with --adaptive",
    )
    parser.add_argument(
        "--max-T",
        type=int,
        default=config.MAX_TRAIN_T,
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
        "--history-window",
        type=int,
        default=config.HISTORY_WINDOW,
        help="Number of past trajectory states included in each model input",
    )
    parser.add_argument(
        "--ftle-window",
        type=int,
        default=config.FTLE_WINDOW,
        help="Forward FTLE window for weighted-loss adaptive training",
    )
    parser.add_argument(
        "--variance",
        type=int,
        default=config.VARIANCE,
        help="Adaptive horizon variance around the default horizon",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug training losses, adaptive dataset T values and Lyapunov exponents",
    )

    args = parser.parse_args()
    train_Ts = get_train_Ts(args.max_T)

    device = "cuda" if torch.cuda.is_available() else config.DEVICE
    print(f"Using device: {device}")
    print(f"Time step: {args.dt}")
    print(
        f"Burn-in: {config.resolve_burn_in_steps(args.dt)} steps "
        f"({config.BURN_IN_TIME:g} time units)"
    )
    print(f"Batch size: {args.batch_size}")
    print(f"Optimizer: {args.optimizer}")

    timestamp, model_save_dir, loss_save_dir = resolve_dirs(
        args.dt, args.append, args.debug
    )

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
            T=args.T,
            adaptive=args.adaptive,
            adaptive_method=args.adaptive_method,
            optimizer_name=args.optimizer,
            batch_size=args.batch_size,
            history_window=args.history_window,
            ftle_window=args.ftle_window,
            var=args.variance,
            debug=args.debug,
        )
    else:
        if args.fixed or not args.adaptive:
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
                history_window=args.history_window,
                append=args.append,
                debug=args.debug,
            )
        if args.adaptive or not args.fixed:
            train_adaptive_models(
                args.n_seeds,
                args.epochs,
                device,
                model_save_dir,
                loss_save_dir,
                args.dt,
                args.optimizer,
                args.batch_size,
                adaptive_method=args.adaptive_method,
                T_max=args.adaptive_T_max,
                history_window=args.history_window,
                ftle_window=args.ftle_window,
                var=args.variance,
                append=args.append,
                debug=args.debug,
            )

    print("\n" + "=" * 50)
    print("Training Complete")
    print("=" * 50)
    print(f"Models saved to {model_save_dir}")
    print(f"Losses saved to {loss_save_dir}")


if __name__ == "__main__":
    main()
