import torch
from torch.utils.data import DataLoader
import argparse
from datetime import datetime
from pathlib import Path
import re
import numpy as np

import adaptive_horizon.config as config
from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.data.dataset import LorenzDataset, collate_fn
from adaptive_horizon.data.adaptive_dataset import (
    AdaptiveHorizonLorenzDataset,
    WeightedLossLorenzDataset,
    collate_fn_adaptive_horizon,
    collate_fn_weighted_loss,
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
from adaptive_horizon.utils import (
    adaptive_method_abbreviation,
    format_dt,
    time_to_steps,
)

ADAPTIVE_HORIZON = "adaptive-horizon"
WEIGHTED_LOSS = "weighted-loss"
CURRICULUM_HORIZON = "curriculum-horizon"
GRADIENT_SCALING_HORIZON = "gradient-scaling-horizon"
CURRICULUM_RAMP_FRACTION = 0.7


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


def get_existing_adaptive_model_seeds(
    model_dir: Path, adaptive_method: str | None = None
):
    model_seeds = set()
    method_short = adaptive_method_abbreviation(adaptive_method)
    for model_path in model_dir.glob("adaptive_mlp*.pt"):
        match = re.search(r"adaptive_mlp(?:_([a-z]+))?_seed(\d+)", model_path.name)
        if not match:
            continue

        method_abbr = match.group(1)
        if method_abbr != method_short:
            continue

        model_seeds.add(int(match.group(2)))
    return model_seeds


def scheduled_sampling_probability(epoch: int, epochs: int) -> float:
    """Linearly decay teacher forcing from 1 to 0 over the training run."""
    if epochs <= 1:
        return 0.0
    return max(0.0, 1.0 - epoch / float(epochs - 1))


def curriculum_horizon(
    epoch: int,
    epochs: int,
    T_max: int,
    ramp_fraction: float = CURRICULUM_RAMP_FRACTION,
) -> int:
    """Linearly increase T from 1 to T_max over the ramp portion of training."""
    if not 0 < ramp_fraction <= 1:
        raise ValueError(f"ramp_fraction must be in (0, 1], got {ramp_fraction}")

    ramp_epochs = max(1, int(epochs * ramp_fraction))
    progress = min(1.0, epoch / float(ramp_epochs))
    return 1 + int((T_max - 1) * progress)


def summarize_gradient_scaling(g_values):
    """Summarize per-batch g(T) values with robust statistics."""
    summary = {}
    for T, values in g_values.items():
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            summary[int(T)] = {"median": float("inf"), "p90": float("inf")}
            continue
        summary[int(T)] = {
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90.0)),
        }
    return summary


def select_gradient_scaling_horizon(
    current_T: int,
    T_max: int,
    g_summary: dict,
    median_threshold: float = config.GRADIENT_SCALING_MEDIAN_THRESHOLD,
    p90_threshold: float = config.GRADIENT_SCALING_P90_THRESHOLD,
) -> tuple[int, int]:
    """Select next horizon from robust gradient-scaling statistics."""
    safe_horizons = [
        T
        for T in range(1, T_max + 1)
        if g_summary.get(T, {}).get("median", float("inf")) <= median_threshold
        and g_summary.get(T, {}).get("p90", float("inf")) <= p90_threshold
    ]
    safe_T = max(safe_horizons) if safe_horizons else 1

    if safe_T < current_T:
        next_T = safe_T
    elif safe_T > current_T:
        next_T = current_T + 1
    else:
        next_T = current_T

    return max(1, min(T_max, next_T)), safe_T


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

    return timestamp, model_save_dir, loss_save_dir, last_run_file


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
            train_dataset = WeightedLossLorenzDataset(
                dt=dt,
                ftle_window=ftle_window,
                history_window=history_window,
                seed=seed,
                burn_in=burn_in_steps,
                debug=debug,
            )
            val_dataset = WeightedLossLorenzDataset(
                num_trajectories=config.NUM_TRAJECTORIES // 5,
                dt=dt,
                ftle_window=ftle_window,
                history_window=history_window,
                seed=seed + 1000,
                burn_in=burn_in_steps,
                normalization_stats=train_dataset.normalization_stats,
                debug=debug,
            )
            collate_function = collate_fn_weighted_loss
        elif adaptive_method in (CURRICULUM_HORIZON, GRADIENT_SCALING_HORIZON):
            if adaptive_method == CURRICULUM_HORIZON:
                T = time_to_steps(config.DEFAULT_ADAPTIVE_HORIZON, dt)
            elif T is None:
                T = config.MAX_TRAIN_T
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
        else:
            raise ValueError(f"Unsupported adaptive method: {adaptive_method}")

        metadata["adaptive"] = {
            "method": adaptive_method,
        }
        if adaptive_method == WEIGHTED_LOSS:
            metadata["adaptive"]["T_max"] = train_dataset.T_max
            metadata["adaptive"]["ftle_window"] = ftle_window
        elif adaptive_method == CURRICULUM_HORIZON:
            metadata["adaptive"].update(
                {
                    "T_max": T,
                    "ramp_fraction": CURRICULUM_RAMP_FRACTION,
                }
            )
        elif adaptive_method == GRADIENT_SCALING_HORIZON:
            metadata["adaptive"].update(
                {
                    "T_max": T,
                    "g_median_threshold": config.GRADIENT_SCALING_MEDIAN_THRESHOLD,
                    "g_p90_threshold": config.GRADIENT_SCALING_P90_THRESHOLD,
                }
            )
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
    model.history_window = history_window
    model.normalization_stats = train_dataset.normalization_stats

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
    scheduled_sampling=False,
    debug=False,
    save_dir=config.LOSS_DIR,
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
        scheduled_sampling: Whether to use scheduled sampling during training
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

    previous_curriculum_T = None
    gradient_scaling_T = 1
    gradient_scaling_schedule = []
    gradient_scaling_history = []
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
        if adaptive and adaptive_method == CURRICULUM_HORIZON:
            current_T = curriculum_horizon(epoch, epochs, T)
        elif adaptive and adaptive_method == GRADIENT_SCALING_HORIZON:
            current_T = gradient_scaling_T
            gradient_scaling_schedule.append(int(current_T))
        else:
            current_T = T

        if (
            adaptive
            and adaptive_method == CURRICULUM_HORIZON
            and current_T != previous_curriculum_T
        ):
            print(f"Curriculum horizon: epoch {epoch + 1}/{epochs}, T={current_T}/{T}")
            previous_curriculum_T = current_T
        teacher_forcing_prob = (
            scheduled_sampling_probability(epoch, epochs) if scheduled_sampling else 0.0
        )
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
                        scheduled_sampling=scheduled_sampling,
                        teacher_forcing_prob=teacher_forcing_prob,
                    )
                elif adaptive_method == WEIGHTED_LOSS:
                    lambda_scores = rest[0].to(device) if rest else None
                    loss = lle_weighted_batch_loss(
                        model,
                        inputs,
                        targets,
                        lambda_scores,
                        dt=dt,
                        scheduled_sampling=scheduled_sampling,
                        teacher_forcing_prob=teacher_forcing_prob,
                    )
                elif adaptive_method in (CURRICULUM_HORIZON, GRADIENT_SCALING_HORIZON):
                    loss = batch_loss(
                        model,
                        inputs,
                        targets[:, :current_T],
                        current_T,
                        scheduled_sampling=scheduled_sampling,
                        teacher_forcing_prob=teacher_forcing_prob,
                    )
                else:
                    raise ValueError(f"Unsupported adaptive method: {adaptive_method}")
            else:
                loss = batch_loss(
                    model,
                    inputs,
                    targets,
                    T,
                    scheduled_sampling=scheduled_sampling,
                    teacher_forcing_prob=teacher_forcing_prob,
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
            gradient_scaling_history.append(
                {
                    "epoch": epoch + 1,
                    "current_T": int(current_T),
                    "safe_T": int(safe_T),
                    "next_T": int(next_T),
                    "g_summary": g_summary,
                }
            )
            model.zero_grad(set_to_none=True)
            gradient_scaling_T = next_T

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

    if adaptive and adaptive_method == GRADIENT_SCALING_HORIZON and metadata:
        metadata["adaptive"]["T_schedule"] = gradient_scaling_schedule
        metadata["adaptive"]["g_history"] = gradient_scaling_history

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
    scheduled_sampling=False,
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
                if adaptive_method == GRADIENT_SCALING_HORIZON
                else None if adaptive else T
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
            metadata["adaptive"]["ramp_epochs"] = max(
                1, int(epochs * CURRICULUM_RAMP_FRACTION)
            )
        elif adaptive_method == GRADIENT_SCALING_HORIZON:
            T = metadata["adaptive"]["T_max"]
        else:
            T = None

    metadata["scheduled_sampling"] = {
        "enabled": bool(scheduled_sampling),
        "schedule": "linear",
        "start_probability": 1.0,
        "end_probability": 0.0,
    }

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
        scheduled_sampling=scheduled_sampling,
        debug=debug,
        save_dir=loss_save_dir,
        metadata=metadata,
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
        adaptive_method=adaptive_method if adaptive else None,
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
    scheduled_sampling=False,
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
                scheduled_sampling=scheduled_sampling,
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
    max_T=config.MAX_TRAIN_T,
    history_window=config.HISTORY_WINDOW,
    ftle_window=config.FTLE_WINDOW,
    var=config.VARIANCE,
    append=False,
    scheduled_sampling=False,
    debug=False,
):
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
        train_loss, val_loss = train_single_model(
            seed,
            epochs,
            device,
            model_save_dir,
            loss_save_dir,
            dt,
            T=max_T if adaptive_method == GRADIENT_SCALING_HORIZON else None,
            adaptive=True,
            adaptive_method=adaptive_method,
            optimizer_name=optimizer_name,
            batch_size=batch_size,
            history_window=history_window,
            ftle_window=ftle_window,
            var=var,
            scheduled_sampling=scheduled_sampling,
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
        choices=[
            ADAPTIVE_HORIZON,
            WEIGHTED_LOSS,
            CURRICULUM_HORIZON,
            GRADIENT_SCALING_HORIZON,
        ],
        default=ADAPTIVE_HORIZON,
        help="Adaptive training method used with --adaptive",
    )
    parser.add_argument(
        "--scheduled-sampling",
        action="store_true",
        help="Use linear scheduled sampling during training",
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
    print(f"History window: {args.history_window}")

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
                else None if args.adaptive else args.T
            ),
            adaptive=args.adaptive,
            adaptive_method=args.adaptive_method,
            optimizer_name=args.optimizer,
            batch_size=args.batch_size,
            history_window=args.history_window,
            ftle_window=args.ftle_window,
            var=args.variance,
            scheduled_sampling=args.scheduled_sampling,
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
                scheduled_sampling=args.scheduled_sampling,
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
                max_T=args.max_T,
                history_window=args.history_window,
                ftle_window=args.ftle_window,
                var=args.variance,
                append=args.append,
                scheduled_sampling=args.scheduled_sampling,
                debug=args.debug,
            )

    print("\n" + "=" * 50)
    print("Training Complete")
    print("=" * 50)
    print(f"Models saved to {model_save_dir}")
    print(f"Losses saved to {loss_save_dir}")
    if not args.append:
        last_run_file.write_text(str(model_save_dir))


if __name__ == "__main__":
    main()
