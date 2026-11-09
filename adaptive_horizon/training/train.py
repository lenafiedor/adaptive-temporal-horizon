import argparse
import math
from pathlib import Path
from time import perf_counter

import torch

from adaptive_horizon import config
from adaptive_horizon.config import resolve_burn_in_steps
from adaptive_horizon.data.dataset import make_val_loader
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
from adaptive_horizon.training.methods import (
    ADAPTIVE_METHOD_CHOICES,
    CROSS_VALIDATION,
    LINEAR_SCHEDULER,
    LYAPUNOV_MEAN,
    LYAPUNOV_METHODS,
    SCHEDULED_METHODS,
    WEIGHTED_LOSS,
    takes_max_T,
)
from adaptive_horizon.training.schedules import HorizonScheduler
from adaptive_horizon.training.setup import create_model_and_loaders
from adaptive_horizon.training.utils import (
    fixed_sweep_rollout_steps,
    get_existing_adaptive_model_seeds,
    get_existing_fixed_model_seeds,
    get_train_Ts,
    resolve_dirs,
    save_model,
)
from adaptive_horizon.visualization.plotting import (
    plot_gradient_history,
    plot_gradients_histogram,
    save_losses,
)


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


class HorizonSelector:
    """Selects a checkpoint among schedule boundaries, preferring larger T.

    The chosen checkpoint has the largest T whose loss is within ``tolerance``
    (relative) of the best loss; ties at that T go to the lowest loss.
    Cross-validation adds one candidate per T (the end of its curriculum
    stage), so no T is favoured by extra stages of training.
    """

    def __init__(self, tolerance=config.CV_TOLERANCE):
        self.tolerance = tolerance
        self.candidates = []
        self.selected = None
        self.best_loss = None

    def add(self, model, loss, T, epoch):
        self.candidates.append(
            {
                "state": clone_model_state_dict(model),
                "loss": float(loss),
                "T": int(T),
                "epoch": epoch,
            }
        )

    def select(self):
        if not self.candidates:
            return None
        best_loss = min(c["loss"] for c in self.candidates)
        threshold = best_loss * (1.0 + self.tolerance)
        eligible = [c for c in self.candidates if c["loss"] <= threshold]
        self.selected = max(eligible, key=lambda c: (c["T"], -c["loss"]))
        self.best_loss = best_loss
        return self.selected

    def restore(self, model):
        """Load the selected state into ``model``; returns whether one exists."""
        if self.select() is None:
            return False
        model.load_state_dict(self.selected["state"])
        return True


def method_batch_loss(
    model, inputs, targets, extra, adaptive, adaptive_method, current_T, dt
):
    """Training loss for one batch under the selected (adaptive) method."""
    if not adaptive:
        return batch_loss(model, inputs, targets, current_T)
    if adaptive_method in LYAPUNOV_METHODS:
        return adaptive_batch_loss(model, inputs, targets, *extra)
    if adaptive_method == WEIGHTED_LOSS:
        return lle_weighted_batch_loss(model, inputs, targets, *extra, dt=dt)
    if adaptive_method in SCHEDULED_METHODS:
        return batch_loss(model, inputs, targets[:, :current_T], current_T)
    raise ValueError(f"Unsupported adaptive method: {adaptive_method}")


def method_rollout_length(targets, extra, adaptive, adaptive_method, current_T):
    """Number of autoregressive model steps one training batch rolls out."""
    if adaptive and adaptive_method in LYAPUNOV_METHODS:
        return int(extra[0].max().item())
    if adaptive and adaptive_method == WEIGHTED_LOSS:
        return targets.shape[1]
    return current_T


def method_validation_loss(model, val_loader, T, adaptive, adaptive_method, dt, device):
    """Validation loss matching the training objective of the selected method."""
    if not adaptive:
        return validation_loss(model, val_loader, T, device)
    if adaptive_method in LYAPUNOV_METHODS:
        return adaptive_validation_loss(model, val_loader, device)
    if adaptive_method == WEIGHTED_LOSS:
        return lle_weighted_validation_loss(model, val_loader, dt=dt, device=device)
    if adaptive_method in SCHEDULED_METHODS:
        return validation_loss(model, val_loader, 1, device)
    raise ValueError(f"Unsupported adaptive method: {adaptive_method}")


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs,
    device=config.DEVICE,
    T=None,
    adaptive=False,
    adaptive_method=LYAPUNOV_MEAN,
    dt=config.DT,
    debug=False,
    save_dir=None,
    metadata=None,
    max_rollout_steps=None,
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
        max_rollout_steps: Optional training compute budget in rollout steps
            (samples x autoregressive steps per training batch). Every method
            uses the same model, so this is proportional to training FLOPs.

    Returns:
        losses: List of training_results losses
        val_losses: List of validation losses
    """
    train_losses = []
    val_losses = []
    gradient_history = []
    wall_time_start = perf_counter()
    rollout_steps = 0
    out_of_steps = False
    if max_rollout_steps is not None and metadata is not None:
        metadata["rollout_steps_budget"] = int(max_rollout_steps)

    scheduled = adaptive and adaptive_method in SCHEDULED_METHODS
    cross_validation = adaptive and adaptive_method == CROSS_VALIDATION
    # Cross-validation: curriculum T=1..T_max with one candidate per stage, then
    # training continues from the selected checkpoint at the selected T.
    selecting = cross_validation
    selection_epochs = None
    selector = HorizonSelector()
    if scheduled:
        horizon_scheduler = (
            HorizonScheduler.stall(T)
            if cross_validation
            else HorizonScheduler.fixed(T, epochs)
        )

    if metadata is not None and scheduled:
        metadata["schedule"] = {
            "type": "stall_then_selected_T" if cross_validation else "fixed",
            "max_T": T,
            **horizon_scheduler.describe(),
        }
    if metadata is not None and cross_validation:
        metadata["selection"] = {
            "metric": f"validation_loss_T1-{config.MAX_EVAL_T}",
            "tolerance": selector.tolerance,
        }

    if debug:
        if save_dir is None:
            save_dir = config.system_path(config.LOSS_DIR, system_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        debug_loader = make_val_loader(
            system_name,
            dt,
            config.MAX_EVAL_T,
            normalization_stats=getattr(model, "normalization_stats", None),
            batch_size=1,
        )
        debug_T_vals = [2, 4, 6, 8, 10]

    epoch = 0
    final_T = None
    while epoch < epochs:
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        current_T = horizon_scheduler.T if scheduled else T
        final_T = current_T

        for batch in train_loader:
            inputs, targets, *extra = [tensor.to(device) for tensor in batch]
            batch_steps = inputs.shape[0] * method_rollout_length(
                targets, extra, adaptive, adaptive_method, current_T
            )
            if (
                max_rollout_steps is not None
                and rollout_steps + batch_steps > max_rollout_steps
            ):
                out_of_steps = True
                break
            rollout_steps += batch_steps
            num_batches += 1
            optimizer.zero_grad()
            loss = method_batch_loss(
                model, inputs, targets, extra, adaptive, adaptive_method, current_T, dt
            )
            loss.backward()
            optimizer.step()
            if scheduled:
                with torch.no_grad():
                    loss = batch_loss(model, inputs, targets[:, :1], 1)
            epoch_loss += loss.item()

        if num_batches == 0:
            print(
                f"\tStopping before epoch {epoch + 1}: "
                f"rollout-step budget {max_rollout_steps} reached"
            )
            break
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        val_loss = method_validation_loss(
            model, val_loader, T, adaptive, adaptive_method, dt, device
        )
        val_losses.append(val_loss)

        if scheduled and (selecting or not cross_validation):
            selection_loss = (
                validation_loss(model, val_loader, config.MAX_EVAL_T, device)
                if selecting
                else None
            )
            training_ends = out_of_steps or epoch + 1 >= epochs
            stage_ended = horizon_scheduler.step(
                selection_loss, force_boundary=training_ends
            )
            if stage_ended and selecting:
                score = horizon_scheduler.stage_score
                selector.add(model, score, current_T, epoch + 1)
                print(
                    f"\tStage end at epoch {epoch + 1}, T={current_T}: "
                    f"T=1..{config.MAX_EVAL_T} MSE={score:.6f}"
                )
                if current_T == T or training_ends:
                    selector.restore(model)
                    selecting = False
                    selection_epochs = epoch + 1
                    final_T = horizon_scheduler.T = selector.selected["T"]
                    print(
                        f"\tCross-validation selected T={final_T} from epoch "
                        f"{selector.selected['epoch']} "
                        f"(T=1..{config.MAX_EVAL_T} MSE="
                        f"{selector.selected['loss']:.6f}, "
                        f"best={selector.best_loss:.6f})"
                    )
            if horizon_scheduler.T != current_T and not training_ends:
                print(
                    f"\tAfter epoch {epoch + 1}/{epochs}, updating T: "
                    f"{current_T} -> {horizon_scheduler.T}"
                )

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}"
            )
            if debug:
                gradients = compute_g_T(
                    model, debug_loader, debug_T_vals, device=device, per_batch=True
                )
                gradient_history.append((epoch, gradients))
                plot_gradients_histogram(gradients, save_dir, epoch, T, dt, adaptive)

        if out_of_steps:
            print(
                f"\tStopping after epoch {epoch + 1}: "
                f"rollout-step budget {max_rollout_steps} reached"
            )
            epoch += 1
            break
        epoch += 1

    if debug:
        plot_gradient_history(gradient_history, save_dir, T, dt, adaptive)

    if metadata is not None:
        metadata["wall_time_seconds"] = float(perf_counter() - wall_time_start)
        metadata["epochs_ran"] = len(train_losses)
        metadata["rollout_steps"] = rollout_steps
        if cross_validation:
            selected = selector.selected or {}
            metadata["selection"].update(
                {
                    "selected_T": selected.get("T"),
                    "selected_epoch": selected.get("epoch"),
                    "selected_loss": selected.get("loss"),
                    "best_loss": selector.best_loss,
                    "candidates": [
                        {k: c[k] for k in ("T", "epoch", "loss")}
                        for c in selector.candidates
                    ],
                    "selection_epochs": selection_epochs,
                    "epochs_ran": len(train_losses),
                    "final_T": final_T,
                }
            )
        if max_rollout_steps is not None:
            metadata["rollout_steps_budget_result"] = {
                "epochs_ran": len(train_losses),
                "stopped_by_rollout_steps": out_of_steps,
            }

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
    adaptive_method=LYAPUNOV_MEAN,
    optimizer_name=config.OPTIMIZER,
    batch_size=config.BATCH_SIZE,
    debug=False,
    max_rollout_steps=None,
    budget_metadata=None,
    system_name=config.DEFAULT_SYSTEM,
    base_T=None,
    random_horizons=False,
):
    model, train_loader, val_loader, optimizer, mlp_config, metadata = (
        create_model_and_loaders(
            seed,
            adaptive,
            device,
            dt,
            T if not adaptive or takes_max_T(adaptive_method) else None,
            adaptive_method,
            optimizer_name,
            batch_size,
            debug,
            system_name,
            base_T=base_T,
            random_horizons=random_horizons,
        )
    )
    if adaptive:
        if budget_metadata is not None:
            metadata["budget"] = budget_metadata
        if adaptive_method in (*SCHEDULED_METHODS, WEIGHTED_LOSS):
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
        max_rollout_steps=max_rollout_steps,
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
        random_horizons=random_horizons,
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
    adaptive_method=LYAPUNOV_MEAN,
    max_T=config.MAX_TRAIN_T,
    append=False,
    debug=False,
    max_rollout_steps=None,
    budget_metadata=None,
    system_name=config.DEFAULT_SYSTEM,
    base_T=None,
    random_horizons=False,
):
    if debug and loss_save_dir is None:
        loss_save_dir = config.system_path(config.LOSS_DIR, system_name)
        loss_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 50}")
    horizons_label = ", random horizons" if random_horizons else ""
    print(f"Training adaptive models ({adaptive_method}{horizons_label})")
    print(f"{'=' * 50}")

    train_losses = []
    val_losses = []

    seed_range = range(n_seeds)
    existing_seeds = (
        get_existing_adaptive_model_seeds(
            model_save_dir, adaptive_method, random_horizons
        )
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
            T=max_T if takes_max_T(adaptive_method) else None,
            adaptive=True,
            adaptive_method=adaptive_method,
            optimizer_name=optimizer_name,
            batch_size=batch_size,
            debug=debug,
            max_rollout_steps=max_rollout_steps,
            budget_metadata=budget_metadata,
            system_name=system_name,
            base_T=base_T,
            random_horizons=random_horizons,
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
        help="Train fixed and adaptive models under the same budget",
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
        "--method",
        choices=ADAPTIVE_METHOD_CHOICES,
        default=None,
        help="Adaptive method",
    )
    parser.add_argument(
        "--base-T",
        type=int,
        default=None,
        help="Override the Lyapunov-based center horizon (1..config.MAX_TRAIN_T)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help=(
            "Lyapunov-mean ablation: randomly permute the FTLE-based horizons "
            "across samples (same base T and variance, no FTLE -> T mapping)"
        ),
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
        "--dt", type=float, default=config.DT, help="Model prediction step"
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
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save models to; existing directories are reused and missing models are appended",
    )
    arch_group = parser.add_mutually_exclusive_group()
    arch_group.add_argument(
        "--layer-widths",
        type=int,
        nargs="+",
        default=None,
        help=f"Hidden block widths (default {config.LAYER_WIDTHS})",
    )
    arch_group.add_argument(
        "--small",
        action="store_true",
        help=f"Use the reduced architecture {config.SMALL_LAYER_WIDTHS}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug training losses, adaptive dataset T values and Lyapunov exponents",
    )

    args = parser.parse_args()
    if args.small:
        config.LAYER_WIDTHS = list(config.SMALL_LAYER_WIDTHS)
    elif args.layer_widths is not None:
        config.LAYER_WIDTHS = list(args.layer_widths)
    train_Ts = get_train_Ts(args.max_T)
    system = get_system(args.system)
    if args.method is None:
        train_fixed = args.fixed or not args.adaptive
        train_adaptive = args.adaptive or not args.fixed
        adaptive_method = LINEAR_SCHEDULER
    else:
        train_fixed = False
        train_adaptive = True
        adaptive_method = args.method

    if args.base_T is not None:
        if not train_adaptive or adaptive_method not in (
            *LYAPUNOV_METHODS,
            WEIGHTED_LOSS,
        ):
            parser.error("--base-T requires --method lyapunov-mean or weighted-loss")
        if not 1 <= args.base_T <= config.MAX_TRAIN_T:
            parser.error(f"--base-T must be between 1 and {config.MAX_TRAIN_T}")

    if args.random and (not train_adaptive or adaptive_method not in LYAPUNOV_METHODS):
        parser.error("--random requires --method lyapunov-mean")

    device = "cuda" if torch.cuda.is_available() else config.DEVICE
    print(f"\nUsing device: {device}")
    print(f"Dynamical system: {system.label}")
    print(f"Prediction step: {args.dt}")
    print(f"Integration step: {config.INTEGRATION_DT}")
    print(
        f"Burn-in: {resolve_burn_in_steps(args.dt)} steps "
        f"({config.BURN_IN_TIME:g} time units)"
    )
    print(f"Layer widths: {config.LAYER_WIDTHS}")
    print(f"Batch size: {args.batch_size}")
    print(f"Optimizer: {config.OPTIMIZER}")

    model_root, fixed_dir, adaptive_dir, loss_dir, append = resolve_dirs(
        args.dt,
        args.T if args.single and not train_adaptive else args.max_T,
        args.debug,
        args.budget_based,
        args.system,
        args.output_dir,
    )

    if args.single:
        print(f"\n{'=' * 50}")
        if train_adaptive:
            print("Training single adaptive model")
        else:
            print(f"Training single model for T={args.T}")
        print(f"{'=' * 50}")
        train_single_model(
            seed=0,
            epochs=args.epochs,
            device=device,
            model_save_dir=adaptive_dir if train_adaptive else fixed_dir,
            loss_save_dir=loss_dir,
            dt=args.dt,
            T=(
                (args.max_T if takes_max_T(adaptive_method) else None)
                if train_adaptive
                else args.T
            ),
            adaptive=train_adaptive,
            adaptive_method=adaptive_method,
            base_T=args.base_T,
            random_horizons=args.random,
            batch_size=args.batch_size,
            debug=args.debug,
            system_name=args.system,
        )
    else:
        if train_fixed:
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
        if train_adaptive:
            rollout_steps_budget = None
            budget_metadata = None
            adaptive_epochs = (
                args.epochs_per_T * args.max_T if args.budget_based else args.epochs
            )
            if args.budget_based and adaptive_method in (
                *LYAPUNOV_METHODS,
                WEIGHTED_LOSS,
            ):
                # Rollout steps (FLOP-proportional) of the fixed sweep T=1..max_T,
                # computed from the config; epochs are not capped.
                rollout_steps_budget = fixed_sweep_rollout_steps(
                    args.max_T, args.epochs_per_T
                )
                adaptive_epochs = math.inf
                budget_metadata = {
                    "unit": "rollout_steps",
                    "max_T": args.max_T,
                    "epochs_per_T": args.epochs_per_T,
                    "max_epochs": None,
                }
                print(
                    f"Adaptive rollout-step budget from fixed sweep: "
                    f"{rollout_steps_budget}"
                )
            train_adaptive_models(
                args.n_seeds,
                adaptive_epochs,
                device,
                adaptive_dir,
                loss_dir,
                dt=args.dt,
                batch_size=args.batch_size,
                adaptive_method=adaptive_method,
                max_T=args.max_T,
                append=append,
                debug=args.debug,
                max_rollout_steps=rollout_steps_budget,
                budget_metadata=budget_metadata,
                system_name=args.system,
                base_T=args.base_T,
                random_horizons=args.random,
            )

    print(f"\nModels saved to {model_root}")
    print("\n" + "=" * 50)
    print("Training Complete")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
