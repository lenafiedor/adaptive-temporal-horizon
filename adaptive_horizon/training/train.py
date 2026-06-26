import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import adaptive_horizon.config as config
from adaptive_horizon.data.dataset import TrajectoryDataset, collate_fn
from adaptive_horizon.dynamics.systems import SYSTEM_CHOICES, get_system
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
    plot_gradient_history,
    plot_gradients_histogram,
    save_losses,
)
from adaptive_horizon.training.methods import (
    ADAPTIVE_HORIZON,
    ADAPTIVE_METHOD_CHOICES,
    CURRICULUM_HORIZON,
    WEIGHTED_LOSS,
)
from adaptive_horizon.training.utils import (
    resolve_burn_in_steps,
    get_existing_adaptive_model_seeds,
    get_existing_fixed_model_seeds,
    get_train_Ts,
    resolve_dirs,
    save_model,
)
from adaptive_horizon.training.schedules import curriculum_horizon
from adaptive_horizon.training.setup import create_model_and_loaders


def padded_mean_loss(losses_by_seed):
    max_len = max(len(losses) for losses in losses_by_seed)
    padded = torch.full((len(losses_by_seed), max_len), float("nan"))
    for idx, losses in enumerate(losses_by_seed):
        padded[idx, : len(losses)] = torch.tensor(losses, dtype=torch.float32)
    return torch.nanmean(padded, dim=0)


def clone_model_state_dict(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def restore_model_state_dict(model, state_dict):
    model.load_state_dict(state_dict)


def curriculum_boundary_reached(epoch, total_epochs, current_T, T_max):
    if epoch + 1 >= total_epochs:
        return True
    next_T = curriculum_horizon(epoch + 1, total_epochs, T_max)
    return next_T > current_T


def cross_validation_median_loss(model, val_loader, val_Ts, device):
    losses_by_T = {
        int(val_T): float(validation_loss(model, val_loader, val_T, device))
        for val_T in val_Ts
    }
    return float(median(losses_by_T.values())), losses_by_T


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
    early_stopping=False,
    system_name=config.DEFAULT_SYSTEM,
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
        early_stopping: Enable early stopping based on validation loss

    Returns:
        losses: List of training_results losses
        val_losses: List of validation losses
    """
    train_losses = []
    val_losses = []
    gradient_history = []
    wall_time_start = perf_counter()

    if early_stopping:
        cv_val_Ts = list(range(1, config.MAX_EVAL_T + 1))
        cv_cached_state = None
        cv_cached_T = None
        cv_cached_epoch = None
        cv_cached_median = None
        if metadata is not None:
            metadata["early_stopping"] = {
                "enabled": True,
                "metric": "median_validation_loss",
                "val_Ts": cv_val_Ts,
                "max_T": T,
            }

    if debug:
        if save_dir is None:
            save_dir = config.system_path(config.LOSS_DIR, system_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        split_gap = max(
            config.MAX_TRAIN_T,
            config.MAX_EVAL_T,
        )
        debug_dataset = TrajectoryDataset(
            T=config.MAX_EVAL_T,
            dt=dt,
            system=system_name,
            normalize=True,
            seed=config.RANDOM_SEED,
            burn_in=resolve_burn_in_steps(dt),
            split="val",
            split_gap=split_gap,
            normalization_stats=getattr(model, "normalization_stats", None),
        )
        debug_loader = DataLoader(
            debug_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
        )
        debug_T_vals = [2, 4, 6, 8, 10]

    curriculum_T = 1

    epoch = 0
    while epoch < epochs:
        model.train()
        epoch_loss = 0.0
        if adaptive and adaptive_method == CURRICULUM_HORIZON:
            current_T = curriculum_horizon(epoch, epochs, T)
            if current_T != curriculum_T:
                print(
                    f"\tEpoch {epoch + 1}/{epochs}, updating T: {curriculum_T} -> {current_T}"
                )
                curriculum_T = current_T
        else:
            current_T = T

        for batch in train_loader:
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
                elif adaptive_method == CURRICULUM_HORIZON:
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
        elif adaptive_method == CURRICULUM_HORIZON:
            val_loss = validation_loss(model, val_loader, current_T, device)
        val_losses.append(val_loss)

        if (
            early_stopping
            and adaptive
            and adaptive_method == CURRICULUM_HORIZON
            and curriculum_boundary_reached(epoch, epochs, current_T, T)
        ):
            cv_median, _ = cross_validation_median_loss(
                model,
                val_loader,
                cv_val_Ts,
                device,
            )
            print(
                f"\tCross-validation check at epoch {epoch + 1}, "
                f"T={current_T}: median MSE={cv_median:.6f}"
            )

            if current_T > 1 and cv_cached_median < cv_median:
                restore_model_state_dict(model, cv_cached_state)
                print(
                    f"\tCross-validation early stopping selected cached "
                    f"T={cv_cached_T} from epoch {cv_cached_epoch} "
                    f"(median MSE={cv_cached_median:.6f})"
                )
                break

            cv_cached_state = clone_model_state_dict(model)
            cv_cached_T = int(current_T)
            cv_cached_epoch = epoch + 1
            cv_cached_median = cv_median

        if (epoch + 1) % 10 == 0:
            message = (
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}"
            )
            if adaptive and adaptive_method in (CURRICULUM_HORIZON,):
                message += f", T={current_T}/{T}"
            print(message)
            if debug:
                gradients = compute_g_T(
                    model, debug_loader, debug_T_vals, device=device, per_batch=True
                )
                gradient_history.append((epoch, gradients))
                plot_gradients_histogram(gradients, save_dir, epoch, T, dt, adaptive)
        epoch += 1
    if debug:
        plot_gradient_history(gradient_history, save_dir, T, dt, adaptive)

    if metadata is not None:
        metadata["wall_time_seconds"] = float(perf_counter() - wall_time_start)
        if early_stopping:
            metadata["early_stopping"].update(
                {
                    "selected_T": cv_cached_T,
                    "selected_epoch": cv_cached_epoch,
                    "selected_median": cv_cached_median,
                    "epochs_ran": len(train_losses),
                }
            )

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
    ftle_window=config.FTLE_WINDOW,
    var=config.VARIANCE,
    debug=False,
    early_stopping=False,
    system_name=config.DEFAULT_SYSTEM,
):
    model, train_loader, val_loader, optimizer, mlp_config, metadata = (
        create_model_and_loaders(
            seed,
            adaptive,
            device,
            dt,
            T if adaptive_method == CURRICULUM_HORIZON else None if adaptive else T,
            adaptive_method,
            optimizer_name,
            batch_size,
            ftle_window,
            var,
            debug,
            system_name,
        )
    )
    if adaptive:
        if adaptive_method in (
            CURRICULUM_HORIZON,
            WEIGHTED_LOSS,
        ):
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
        early_stopping=early_stopping,
        system_name=system_name,
    )

    model_path = save_model(
        model,
        mlp_config,
        seed,
        save_dir=model_save_dir,
        T=T,
        adaptive=adaptive,
        metadata=metadata,
        adaptive_method=adaptive_method if adaptive else None,
    )
    if debug:
        save_losses(
            torch.tensor(train_losses, dtype=torch.float32),
            torch.tensor(val_losses, dtype=torch.float32),
            loss_save_dir,
            T=T,
            adaptive=adaptive,
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
    append=False,
    debug=False,
    system_name=config.DEFAULT_SYSTEM,
):
    if debug and loss_save_dir is None:
        loss_save_dir = config.system_path(config.LOSS_DIR, system_name)
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
                debug=debug,
                system_name=system_name,
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
    ftle_window=config.FTLE_WINDOW,
    var=config.VARIANCE,
    append=False,
    debug=False,
    early_stopping=False,
    system_name=config.DEFAULT_SYSTEM,
):
    if debug and loss_save_dir is None:
        loss_save_dir = config.system_path(config.LOSS_DIR, system_name)
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
            T=max_T if adaptive_method == CURRICULUM_HORIZON else None,
            adaptive=True,
            adaptive_method=adaptive_method,
            optimizer_name=optimizer_name,
            batch_size=batch_size,
            ftle_window=ftle_window,
            var=var,
            debug=debug,
            early_stopping=early_stopping,
            system_name=system_name,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    if debug:
        save_losses(
            padded_mean_loss(train_losses),
            padded_mean_loss(val_losses),
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
        "--budget-based",
        action="store_true",
        help="Train fixed and curriculum-horizon adaptive models under the same epoch budget",
    )
    parser.add_argument(
        "--epochs-per-T",
        type=int,
        default=20,
        help="Budget mode epochs for each fixed horizon",
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
        "--system",
        choices=SYSTEM_CHOICES,
        default=config.DEFAULT_SYSTEM,
        help="Dynamical system to train on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size for training and validation loaders",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Early stop curriculum-horizon training using median validation loss over all horizons",
    )
    parser.add_argument(
        "--append",
        nargs="?",
        const="",
        default=None,
        metavar="MODEL_DIR",
        help="Append outputs to MODEL_DIR, or to the run referenced by models/last_run.txt when no value is provided",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the trained models to",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug training losses, adaptive dataset T values and Lyapunov exponents",
    )

    args = parser.parse_args()
    append = args.append is not None
    append_model_dir = Path(args.append) if args.append else None
    train_Ts = get_train_Ts(args.max_T)
    system = get_system(args.system)
    effective_adaptive_method = (
        CURRICULUM_HORIZON if args.budget_based else args.adaptive_method
    )
    if args.early_stopping:
        if args.fixed or (args.single and not args.adaptive):
            parser.error("--early-stopping requires adaptive training")
        if effective_adaptive_method != CURRICULUM_HORIZON:
            parser.error(
                "--early-stopping requires "
                "--adaptive-method curriculum-horizon or --budget-based"
            )

    device = "cuda" if torch.cuda.is_available() else config.DEVICE
    print(f"Using device: {device}")
    print(f"Dynamical system: {system.label}")
    print(f"Time step: {args.dt}")
    print(
        f"Burn-in: {resolve_burn_in_steps(args.dt)} steps "
        f"({config.BURN_IN_TIME:g} time units)"
    )
    print(f"Batch size: {args.batch_size}")
    print(f"Optimizer: {config.OPTIMIZER}")

    model_root, fixed_dir, adaptive_dir, loss_dir, last_run_file = resolve_dirs(
        args.dt,
        append,
        args.max_T,
        args.debug,
        args.budget_based,
        append_model_dir,
        args.system,
        args.output_dir,
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
            model_save_dir=adaptive_dir if args.adaptive else fixed_dir,
            loss_save_dir=loss_dir,
            dt=args.dt,
            T=(
                args.max_T
                if args.adaptive and args.adaptive_method == CURRICULUM_HORIZON
                else None
                if args.adaptive
                else args.T
            ),
            adaptive=args.adaptive,
            adaptive_method=args.adaptive_method,
            batch_size=args.batch_size,
            debug=args.debug,
            early_stopping=args.early_stopping,
            system_name=args.system,
        )
    else:
        if args.fixed or not args.adaptive:
            train_fixed_models(
                train_Ts,
                args.n_seeds,
                args.epochs_per_T if args.budget_based else args.epochs,
                device,
                fixed_dir,
                loss_dir,
                dt=args.dt,
                batch_size=args.batch_size,
                append=append,
                debug=args.debug,
                system_name=args.system,
            )
        if args.adaptive or not args.fixed:
            train_adaptive_models(
                args.n_seeds,
                args.epochs_per_T * args.max_T if args.budget_based else args.epochs,
                device,
                adaptive_dir,
                loss_dir,
                dt=args.dt,
                batch_size=args.batch_size,
                adaptive_method=CURRICULUM_HORIZON
                if args.budget_based
                else args.adaptive_method,
                max_T=args.max_T,
                append=append,
                debug=args.debug,
                early_stopping=args.early_stopping,
                system_name=args.system,
            )

    print("\n" + "=" * 50)
    print("Training Complete")
    print("=" * 50)
    print(f"Models saved to {model_root}")
    if loss_dir is not None:
        print(f"Losses saved to {loss_dir}")
    if not append:
        last_run_file.write_text(str(model_root))


if __name__ == "__main__":
    main()
