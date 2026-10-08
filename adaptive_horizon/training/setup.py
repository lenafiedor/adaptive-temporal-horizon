import torch
from torch.utils.data import DataLoader

import adaptive_horizon.config as config
from adaptive_horizon.data.adaptive_dataset import (
    LyapunovBasedDataset,
    WeightedLossDataset,
    collate_fn_lyapunov_based,
    collate_fn_weighted_loss,
)
from adaptive_horizon.data.dataset import TrajectoryDataset, collate_fn
from adaptive_horizon.data.santafe import SANTAFE, load_santafe
from adaptive_horizon.dynamics.systems import get_system
from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.training.methods import (
    CROSS_VALIDATION,
    EARLY_STOPPING,
    LINEAR_SCHEDULER,
    LYAPUNOV_MEAN,
    LYAPUNOV_TIME,
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
    adaptive_method=LYAPUNOV_MEAN,
    optimizer_name=config.OPTIMIZER,
    batch_size=config.BATCH_SIZE,
    ftle_window=config.FTLE_WINDOW,
    debug=False,
    system_name=config.DEFAULT_SYSTEM,
    data_path=None,
):
    """
    Create model, data loaders, optimizer, and config for training.

    Args:
        seed: Random seed
        adaptive: Whether to use adaptive temporal horizon
        device: CPU or GPU
        dt: Time step for simulation
        T: Fixed or maximum rollout horizon
        adaptive_method: Adaptive training method
        optimizer_name: Optimizer name
        batch_size: Batch size for data loaders
        ftle_window: Forward FTLE window for weighted-loss training
        debug: Whether adaptive datasets should write T values and Lyapunov exponents
        system_name: Name of the dynamical system

    Returns:
        model, train_loader, val_loader, optimizer, config, metadata
    """
    observed_data = None
    if system_name == SANTAFE:
        if dt != 1:
            raise ValueError("Santa Fe dt must be 1 (one recorded sample)")
        observed_data = load_santafe(
            data_path,
            max(T or 10, config.MAX_EVAL_T),
            compute_scores=adaptive
            and adaptive_method in (LYAPUNOV_MEAN, LYAPUNOV_TIME, WEIGHTED_LOSS),
        )
    system = None if observed_data is not None else get_system(system_name)
    dimension = (
        observed_data["metadata"]["history_length"]
        if observed_data is not None
        else system.dim
    )
    observed_kwargs = (
        {"observed_data": observed_data} if observed_data is not None else {}
    )
    mlp_config = MLPConfig(
        input_size=dimension,
        output_size=1 if observed_data is not None else dimension,
        delay_window=observed_data is not None,
        layer_widths=[config.LAYER_WIDTH, config.LAYER_WIDTH, config.LAYER_WIDTH],
        residual_connections=True,
        k=1,
        activation=torch.nn.ReLU(),
    )
    model = MLP(mlp_config, random_seed=seed).to(device)
    burn_in_steps = 0 if observed_data is not None else resolve_burn_in_steps(dt)
    split_gap = max(config.MAX_TRAIN_T, config.MAX_EVAL_T, ftle_window, T or 0)
    metadata = {
        "dt": dt,
        "integration_dt": None if observed_data is not None else config.INTEGRATION_DT,
        "system": system_name,
        "system_parameters": {}
        if observed_data is not None
        else dict(system.parameters),
        "burn_in_time": 0 if observed_data is not None else config.BURN_IN_TIME,
        "trajectory": {
            "steps": config.TRAJECTORY_STEPS,
            "seed": config.RANDOM_SEED,
            "train_fraction": config.TRAIN_FRACTION,
            "split_gap": split_gap,
        },
    }
    if observed_data is not None:
        metadata["observed_data"] = observed_data["metadata"]
        metadata["trajectory"] = {
            "steps": observed_data["metadata"]["samples"] - 1,
            "train_fraction": 0.7,
            "split_gap": 0,
        }

    if adaptive:
        if adaptive_method in (LYAPUNOV_MEAN, LYAPUNOV_TIME):
            train_dataset = LyapunovBasedDataset(
                dt=dt,
                system=system_name,
                **observed_kwargs,
                seed=config.RANDOM_SEED,
                burn_in=burn_in_steps,
                max_T=T or config.MAX_TRAIN_T,
                adaptive_method=adaptive_method,
                split="train",
                split_gap=split_gap,
                debug=debug,
            )
            val_dataset = LyapunovBasedDataset(
                dt=dt,
                system=system_name,
                **observed_kwargs,
                seed=config.RANDOM_SEED,
                burn_in=burn_in_steps,
                max_T=T or config.MAX_TRAIN_T,
                adaptive_method=adaptive_method,
                split="val",
                split_gap=split_gap,
                normalization_stats=train_dataset.normalization_stats,
                debug=debug,
            )
            collate_function = collate_fn_lyapunov_based
        elif adaptive_method == WEIGHTED_LOSS:
            train_dataset = WeightedLossDataset(
                dt=dt,
                system=system_name,
                **observed_kwargs,
                T_max=(T or 10) if observed_data is not None else None,
                ftle_window=ftle_window,
                seed=config.RANDOM_SEED,
                burn_in=burn_in_steps,
                split="train",
                split_gap=split_gap,
                debug=debug,
            )
            val_dataset = WeightedLossDataset(
                dt=dt,
                system=system_name,
                **observed_kwargs,
                T_max=(T or 10) if observed_data is not None else None,
                ftle_window=ftle_window,
                seed=config.RANDOM_SEED,
                burn_in=burn_in_steps,
                split="val",
                split_gap=split_gap,
                normalization_stats=train_dataset.normalization_stats,
                debug=debug,
            )
            collate_function = collate_fn_weighted_loss
        elif adaptive_method in (LINEAR_SCHEDULER, EARLY_STOPPING, CROSS_VALIDATION):
            if T is None:
                T = (
                    5
                    if observed_data is not None
                    else time_to_steps(config.DEFAULT_HORIZON, dt)
                )
            train_dataset = TrajectoryDataset(
                T=T,
                dt=dt,
                system=system_name,
                **observed_kwargs,
                seed=config.RANDOM_SEED,
                burn_in=burn_in_steps,
                split="train",
                split_gap=split_gap,
            )
            val_dataset = TrajectoryDataset(
                T=config.MAX_EVAL_T,
                dt=dt,
                system=system_name,
                **observed_kwargs,
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
            metadata["adaptive"]["ftle_window"] = (
                observed_data["metadata"]["estimator"]["window"]
                if observed_data is not None
                else ftle_window
            )
        elif adaptive_method in (LINEAR_SCHEDULER, EARLY_STOPPING, CROSS_VALIDATION):
            metadata["adaptive"].update(
                {
                    "T_max": T,
                }
            )
        else:
            metadata["adaptive"].update(
                {
                    "min_T": train_dataset.min_T,
                    "max_T": train_dataset.max_T,
                    "horizon_mapping": (
                        "lyapunov_time"
                        if adaptive_method == LYAPUNOV_TIME
                        else "lyapunov_mean"
                    ),
                }
            )
            if adaptive_method == LYAPUNOV_MEAN:
                metadata["adaptive"].update(
                    {"variance": train_dataset.var, "base_T": train_dataset.base_T}
                )
    else:
        train_dataset = TrajectoryDataset(
            T=T,
            dt=dt,
            system=system_name,
            **observed_kwargs,
            seed=config.RANDOM_SEED,
            burn_in=burn_in_steps,
            split="train",
            split_gap=split_gap,
        )
        val_dataset = TrajectoryDataset(
            T=T,
            dt=dt,
            system=system_name,
            **observed_kwargs,
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
