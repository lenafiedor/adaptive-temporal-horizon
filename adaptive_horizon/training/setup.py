import torch
from torch.utils.data import DataLoader

import adaptive_horizon.config as config
from adaptive_horizon.data.adaptive_dataset import (
    AdaptiveHorizonDataset,
    WeightedLossDataset,
    collate_fn_adaptive_horizon,
    collate_fn_weighted_loss,
)
from adaptive_horizon.data.dataset import TrajectoryDataset, collate_fn
from adaptive_horizon.dynamics.systems import get_system
from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.training.methods import (
    ADAPTIVE_HORIZON,
    CURRICULUM_HORIZON,
    WEIGHTED_LOSS,
)
from adaptive_horizon.training.utils import resolve_burn_in_steps
from adaptive_horizon.utils import time_to_steps


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


def create_model_and_loaders(
    seed,
    adaptive,
    device,
    dt,
    T=None,
    adaptive_method=ADAPTIVE_HORIZON,
    optimizer_name=config.OPTIMIZER,
    batch_size=config.BATCH_SIZE,
    ftle_window=config.FTLE_WINDOW,
    var=config.VARIANCE,
    debug=False,
    system_name=config.DEFAULT_SYSTEM,
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
        ftle_window: Forward FTLE window for weighted-loss training
        var: Variance of the adaptive horizon
        debug: Whether adaptive datasets should write T values and Lyapunov exponents
        system_name: Name of the dynamical system

    Returns:
        model, train_loader, val_loader, optimizer, config, metadata
    """
    system = get_system(system_name)
    mlp_config = MLPConfig(
        input_size=system.dim,
        output_size=system.dim,
        layer_widths=[config.LAYER_WIDTH, config.LAYER_WIDTH, config.LAYER_WIDTH],
        residual_connections=True,
        k=1,
        activation=torch.nn.ReLU(),
    )
    model = MLP(mlp_config, random_seed=seed).to(device)
    burn_in_steps = resolve_burn_in_steps(dt)
    split_gap = max(config.MAX_TRAIN_T, config.MAX_EVAL_T, ftle_window)
    metadata = {
        "dt": dt,
        "system": system.name,
        "system_parameters": dict(system.parameters),
        "burn_in_time": config.BURN_IN_TIME,
        "trajectory": {
            "steps": config.TRAJECTORY_STEPS,
            "seed": config.RANDOM_SEED,
            "train_fraction": config.TRAIN_FRACTION,
            "split_gap": split_gap,
        },
    }

    if adaptive:
        if adaptive_method == ADAPTIVE_HORIZON:
            train_dataset = AdaptiveHorizonDataset(
                dt=dt,
                system=system.name,
                seed=config.RANDOM_SEED,
                burn_in=burn_in_steps,
                var=var,
                split="train",
                split_gap=split_gap,
                debug=debug,
            )
            val_dataset = AdaptiveHorizonDataset(
                dt=dt,
                system=system.name,
                seed=config.RANDOM_SEED,
                burn_in=burn_in_steps,
                var=var,
                split="val",
                split_gap=split_gap,
                normalization_stats=train_dataset.normalization_stats,
                debug=debug,
            )
            collate_function = collate_fn_adaptive_horizon
        elif adaptive_method == WEIGHTED_LOSS:
            train_dataset = WeightedLossDataset(
                dt=dt,
                system=system.name,
                ftle_window=ftle_window,
                seed=config.RANDOM_SEED,
                burn_in=burn_in_steps,
                split="train",
                split_gap=split_gap,
                debug=debug,
            )
            val_dataset = WeightedLossDataset(
                dt=dt,
                system=system.name,
                ftle_window=ftle_window,
                seed=config.RANDOM_SEED,
                burn_in=burn_in_steps,
                split="val",
                split_gap=split_gap,
                normalization_stats=train_dataset.normalization_stats,
                debug=debug,
            )
            collate_function = collate_fn_weighted_loss
        elif adaptive_method == CURRICULUM_HORIZON:
            if T is None:
                T = time_to_steps(config.DEFAULT_ADAPTIVE_HORIZON, dt)
            train_dataset = TrajectoryDataset(
                T=T,
                dt=dt,
                system=system.name,
                seed=config.RANDOM_SEED,
                burn_in=burn_in_steps,
                split="train",
                split_gap=split_gap,
            )
            val_dataset = TrajectoryDataset(
                T=T,
                dt=dt,
                system=system.name,
                seed=config.RANDOM_SEED,
                burn_in=burn_in_steps,
                split="val",
                split_gap=split_gap,
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
        train_dataset = TrajectoryDataset(
            T=T,
            dt=dt,
            system=system.name,
            seed=config.RANDOM_SEED,
            burn_in=burn_in_steps,
            split="train",
            split_gap=split_gap,
        )
        val_dataset = TrajectoryDataset(
            T=T,
            dt=dt,
            system=system.name,
            seed=config.RANDOM_SEED,
            burn_in=burn_in_steps,
            split="val",
            split_gap=split_gap,
            normalization_stats=train_dataset.normalization_stats,
        )
        collate_function = collate_fn

    metadata["normalization_stats"] = train_dataset.normalization_stats
    metadata["trajectory"].update(
        {
            "path": str(train_dataset.trajectory_path),
            "train_split_bounds": tuple(
                int(value) for value in train_dataset.split_bounds
            ),
            "val_split_bounds": tuple(int(value) for value in val_dataset.split_bounds),
        }
    )
    model.normalization_stats = train_dataset.normalization_stats

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_function
    )
    optimizer = create_optimizer(optimizer_name, model)

    return model, train_loader, val_loader, optimizer, mlp_config, metadata
