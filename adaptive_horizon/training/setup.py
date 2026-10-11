import torch
from torch.utils.data import DataLoader

from adaptive_horizon import config
from adaptive_horizon.config import resolve_burn_in_steps
from adaptive_horizon.data.adaptive_dataset import (
    LyapunovBasedDataset,
    WeightedLossDataset,
    collate_fn_lyapunov_based,
)
from adaptive_horizon.data.dataset import TrajectoryDataset, collate_fn
from adaptive_horizon.data.utils import default_split_gap
from adaptive_horizon.dynamics.systems import get_system
from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.training.methods import (
    LYAPUNOV_MEAN,
    LYAPUNOV_METHODS,
    LYAPUNOV_TIME,
    SCHEDULED_METHODS,
    WEIGHTED_LOSS,
)


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


def _make_pair(dataset_cls, common, val_overrides=None, carry=()):
    train_dataset = dataset_cls(split="train", **common)
    val_kwargs = {**common, **(val_overrides or {})}
    val_kwargs.update({name: getattr(train_dataset, name) for name in carry})
    val_dataset = dataset_cls(
        split="val",
        normalization_stats=train_dataset.normalization_stats,
        **val_kwargs,
    )
    return train_dataset, val_dataset


def _build_datasets(adaptive, adaptive_method, T, common, debug):
    if not adaptive:
        return *_make_pair(TrajectoryDataset, {**common, "T": T}), collate_fn, T

    if adaptive_method in LYAPUNOV_METHODS:
        lyapunov_common = {
            **common,
            "max_T": T or config.MAX_TRAIN_T,
            "adaptive_method": adaptive_method,
            "debug": debug,
        }
        pair = _make_pair(
            LyapunovBasedDataset, lyapunov_common, carry=("lambda_estimate",)
        )
        return *pair, collate_fn_lyapunov_based, T

    if adaptive_method == WEIGHTED_LOSS:
        pair = _make_pair(
            WeightedLossDataset,
            {**common, "debug": debug},
            carry=("lambda_estimate",),
        )
        return *pair, collate_fn, T

    if adaptive_method in SCHEDULED_METHODS:
        T = T or config.MAX_TRAIN_T
        pair = _make_pair(
            TrajectoryDataset,
            {**common, "T": T},
            val_overrides={"T": config.MAX_EVAL_T},
        )
        return *pair, collate_fn, T

    raise ValueError(f"Unsupported adaptive method: {adaptive_method}")


def _adaptive_metadata(adaptive_method, train_dataset, T, dt):
    metadata = {"method": adaptive_method}
    if adaptive_method in (*LYAPUNOV_METHODS, WEIGHTED_LOSS):
        metadata["ftle_window"] = train_dataset.ftle_window

    if adaptive_method == WEIGHTED_LOSS:
        metadata["T_max"] = train_dataset.T_max
        metadata["lambda_estimate"] = train_dataset.lambda_estimate
    elif adaptive_method in SCHEDULED_METHODS:
        metadata["T_max"] = T
    else:
        metadata["min_T"] = train_dataset.min_T
        metadata["max_T"] = train_dataset.max_T
        metadata["horizon_mapping"] = (
            "lyapunov_time" if adaptive_method == LYAPUNOV_TIME else "lyapunov_mean"
        )
        if adaptive_method == LYAPUNOV_MEAN:
            metadata.update(
                {
                    "variance": train_dataset.var,
                    "base_T": train_dataset.base_T,
                    "lyapunov_fraction": train_dataset.lyapunov_fraction,
                    "lambda_estimate": train_dataset.lambda_estimate,
                    "mean_train_T": train_dataset.mean_T,
                    "mean_train_T_lyapunov_times": (
                        train_dataset.lambda_estimate * dt * train_dataset.mean_T
                    ),
                }
            )
    return metadata


def create_model_and_loaders(
    seed,
    adaptive,
    device,
    dt,
    T=None,
    adaptive_method=LYAPUNOV_MEAN,
    optimizer_name=config.OPTIMIZER,
    batch_size=config.BATCH_SIZE,
    debug=False,
    system_name=config.DEFAULT_SYSTEM,
):
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
    split_gap = default_split_gap(T or 0)
    metadata = {
        "dt": dt,
        "integration_dt": config.INTEGRATION_DT,
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

    common = {
        "dt": dt,
        "system": system.name,
        "seed": config.RANDOM_SEED,
        "burn_in": resolve_burn_in_steps(dt),
        "split_gap": split_gap,
    }
    train_dataset, val_dataset, collate_function, T = _build_datasets(
        adaptive, adaptive_method, T, common, debug
    )
    if adaptive:
        metadata["adaptive"] = _adaptive_metadata(adaptive_method, train_dataset, T, dt)

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
