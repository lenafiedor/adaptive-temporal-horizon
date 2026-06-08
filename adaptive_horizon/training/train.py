import torch
from torch.utils.data import DataLoader
import argparse
from time import perf_counter

import adaptive_horizon.config as config
from adaptive_horizon.data.dataset import LorenzDataset, collate_fn
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
from adaptive_horizon.training.methods import (
    ADAPTIVE_HORIZON,
    ADAPTIVE_METHOD_CHOICES,
    CURRICULUM_HORIZON,
    GRADIENT_SCALING_HORIZON,
    WEIGHTED_LOSS,
)
from adaptive_horizon.training.run_utils import (
    get_existing_adaptive_model_seeds,
    get_existing_fixed_model_seeds,
    get_train_Ts,
    resolve_dirs,
)
from adaptive_horizon.training.schedules import (
    curriculum_horizon,
    select_gradient_scaling_horizon,
    summarize_gradient_scaling,
)
from adaptive_horizon.training.setup import create_model_and_loaders


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
    save_dir=None,
    metadata=None,
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
        metadata: Optional checkpoint metadata to update with adaptive schedules

    Returns:
        losses: List of training_results losses
        val_losses: List of validation losses
    """
    train_losses = []
    val_losses = []
    gradient_history = []
    train_wall_clock_seconds = 0.0
    if debug:
        if save_dir is None:
            save_dir = config.LOSS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
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

    curriculum_T, gradient_scaling_T = 1, 1
    success_count = 0
    if adaptive and adaptive_method == GRADIENT_SCALING_HORIZON:
        probe_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            shuffle=False,
            collate_fn=train_loader.collate_fn,
        )
        probe_T_vals = list(range(1, T + 1))

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        if adaptive and adaptive_method == GRADIENT_SCALING_HORIZON:
            current_T = gradient_scaling_T
        elif adaptive and adaptive_method == CURRICULUM_HORIZON:
            current_T = curriculum_T
        else:
            current_T = T

        for batch in train_loader:
            batch_start = perf_counter()
            inputs, targets, *rest = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if adaptive:
                if adaptive_method == ADAPTIVE_HORIZON:
                    T_values = rest[0].to(device) if rest else None
                    loss = adaptive_batch_loss(
                        model,
                        inputs,
                        targets,
                        T_values,
                    )
                elif adaptive_method == WEIGHTED_LOSS:
                    lambda_scores = rest[0].to(device) if rest else None
                    loss = lle_weighted_batch_loss(
                        model,
                        inputs,
                        targets,
                        lambda_scores,
                        dt=dt,
                    )
                elif adaptive_method in (CURRICULUM_HORIZON, GRADIENT_SCALING_HORIZON):
                    loss = batch_loss(
                        model,
                        inputs,
                        targets[:, :current_T],
                        current_T,
                    )
                else:
                    raise ValueError(f"Unsupported adaptive method: {adaptive_method}")
            else:
                loss = batch_loss(
                    model,
                    inputs,
                    targets,
                    T,
                )
            loss.backward()
            optimizer.step()
            train_wall_clock_seconds += perf_counter() - batch_start
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
        elif adaptive_method in (CURRICULUM_HORIZON, GRADIENT_SCALING_HORIZON):
            val_loss = validation_loss(model, val_loader, current_T, device)
        else:
            raise ValueError(f"Unsupported adaptive method: {adaptive_method}")
        val_losses.append(val_loss)

        if adaptive and adaptive_method == GRADIENT_SCALING_HORIZON:
            gradients = compute_g_T(
                model,
                probe_loader,
                probe_T_vals,
                device=device,
                per_batch=True,
            )
            g_summary = summarize_gradient_scaling(gradients)
            next_T, safe_T = select_gradient_scaling_horizon(
                current_T=current_T,
                T_max=T,
                g_summary=g_summary,
            )
            model.zero_grad(set_to_none=True)
            gradient_scaling_T = next_T
        elif adaptive and adaptive_method == CURRICULUM_HORIZON:
            current_T, success_count = curriculum_horizon(
                epoch, epochs, val_loss, curriculum_T, success_count, T
            )
            if current_T != curriculum_T:
                print(
                    f"\tEpoch {epoch + 1}/{epochs}, updating T: {curriculum_T} -> {current_T}"
                )
                curriculum_T = current_T

        if (epoch + 1) % 10 == 0:
            message = (
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}"
            )
            if adaptive and adaptive_method in (
                CURRICULUM_HORIZON,
                GRADIENT_SCALING_HORIZON,
            ):
                message += f", T={current_T}/{T}"
            print(message)
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

    if metadata is not None:
        metadata["train_wall_clock_seconds"] = float(train_wall_clock_seconds)

    return train_losses, val_losses


def train_single_model(
    seed,
    epochs,
    device,
    model_save_dir,
    loss_save_dir=None,
    dt=config.DT,
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
            (
                T
                if adaptive_method in (CURRICULUM_HORIZON, GRADIENT_SCALING_HORIZON)
                else None
                if adaptive
                else T
            ),
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
        elif adaptive_method == CURRICULUM_HORIZON:
            T = metadata["adaptive"]["T_max"]
            metadata["adaptive"]["epochs_per_horizon"] = epochs / T
        elif adaptive_method == GRADIENT_SCALING_HORIZON:
            T = metadata["adaptive"]["T_max"]
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
        metadata=metadata,
    )

    model_path = save_model(
        model,
        mlp_config,
        seed,
        save_dir=model_save_dir,
        T=T,
        adaptive=adaptive,
        metadata=metadata,
        var=var if adaptive_method == ADAPTIVE_HORIZON else None,
        adaptive_method=adaptive_method if adaptive else None,
    )
    return train_losses, val_losses, model_path


def train_fixed_models(
    train_Ts,
    n_seeds,
    epochs,
    device,
    model_save_dir,
    loss_save_dir=None,
    dt=config.DT,
    optimizer_name=config.OPTIMIZER,
    batch_size=config.BATCH_SIZE,
    history_window=config.HISTORY_WINDOW,
    append=False,
    debug=False,
):
    if debug and loss_save_dir is None:
        loss_save_dir = config.LOSS_DIR
        loss_save_dir.mkdir(parents=True, exist_ok=True)

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
            train_loss, val_loss, _ = train_single_model(
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
    loss_save_dir=None,
    dt=config.DT,
    optimizer_name=config.OPTIMIZER,
    batch_size=config.BATCH_SIZE,
    adaptive_method=ADAPTIVE_HORIZON,
    max_T=config.MAX_TRAIN_T,
    history_window=config.HISTORY_WINDOW,
    ftle_window=config.FTLE_WINDOW,
    var=config.VARIANCE,
    append=False,
    debug=False,
):
    if debug and loss_save_dir is None:
        loss_save_dir = config.LOSS_DIR
        loss_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"Training adaptive models ({adaptive_method})")
    print(f"{'=' * 50}")

    train_losses = []
    val_losses = []

    seed_range = range(n_seeds)
    existing_seeds = (
        get_existing_adaptive_model_seeds(model_save_dir, adaptive_method)
        if append
        else set()
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
        train_loss, val_loss, _ = train_single_model(
            seed,
            epochs,
            device,
            model_save_dir,
            loss_save_dir,
            dt,
            T=(
                max_T
                if adaptive_method in (CURRICULUM_HORIZON, GRADIENT_SCALING_HORIZON)
                else None
            ),
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
        choices=ADAPTIVE_METHOD_CHOICES,
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
        "--append",
        action="store_true",
        help="Append outputs to the run referenced by models/last_run.txt",
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
    print(f"Optimizer: {config.OPTIMIZER}")

    timestamp, model_save_dir, loss_save_dir, last_run_file = resolve_dirs(
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
            T=(
                args.max_T
                if args.adaptive and args.adaptive_method == GRADIENT_SCALING_HORIZON
                else None
                if args.adaptive
                else args.T
            ),
            adaptive=args.adaptive,
            adaptive_method=args.adaptive_method,
            batch_size=args.batch_size,
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
                dt=args.dt,
                batch_size=args.batch_size,
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
                dt=args.dt,
                batch_size=args.batch_size,
                adaptive_method=args.adaptive_method,
                max_T=args.max_T,
                append=args.append,
                debug=args.debug,
            )

    print("\n" + "=" * 50)
    print("Training Complete")
    print("=" * 50)
    print(f"Models saved to {model_save_dir}")
    if loss_save_dir is not None:
        print(f"Losses saved to {loss_save_dir}")
    if not args.append:
        last_run_file.write_text(str(model_save_dir))


if __name__ == "__main__":
    main()
