from datetime import datetime
from pathlib import Path
import re
import torch
import math
from statistics import mean

import adaptive_horizon.config as config
from adaptive_horizon.training.methods import adaptive_method_abbreviation
from adaptive_horizon.utils import format_dt


def resolve_burn_in_steps(dt=config.DT, burn_in_time=config.BURN_IN_TIME):
    """Resolve transient burn-in steps from physical time."""
    return int(math.ceil(burn_in_time / dt))


def get_train_Ts(max_T: int):
    if max_T < 1:
        raise ValueError(f"--max-T must be at least 1, got {max_T}")
    return list(range(1, max_T + 1))


def model_info(model_path: Path):
    match = re.search(r"mlp_T(\d+)_seed(\d+)(?:_|$)", model_path.name)
    if match:
        return int(match.group(1)), int(match.group(2))

    match = re.search(r"_seed(\d+)(?:_|$)", model_path.name)
    if not match:
        return None
    return None, int(match.group(1))


def get_existing_fixed_model_seeds(model_dir: Path):
    model_seeds = {}
    for model_path in model_dir.glob("mlp_T*.pt"):
        info = model_info(model_path)
        if info is None or info[0] is None:
            continue
        train_T, seed = info
        model_seeds.setdefault(train_T, set()).add(seed)
    return model_seeds


def get_existing_adaptive_model_seeds(
    model_dir: Path, adaptive_method: str | None = None
):
    model_seeds = set()
    method_short = adaptive_method_abbreviation(adaptive_method)
    for model_path in model_dir.glob("adaptive_mlp*.pt"):
        match = re.search(r"adaptive_mlp(?:_([a-z]+))?_seed\d+(?:_|$)", model_path.name)
        if not match:
            continue

        method_abbr = match.group(1)
        if method_abbr != method_short:
            continue

        info = model_info(model_path)
        if info is not None:
            model_seeds.add(info[1])
    return model_seeds


def resolve_dirs(
    dt,
    max_train_T: int,
    debug: bool,
    budget_based: bool,
    system_name=config.DEFAULT_SYSTEM,
    output_dir: Path | None = None,
):
    model_dir = config.system_path(config.MODEL_DIR, system_name)
    loss_root = config.system_path(config.LOSS_DIR, system_name)
    last_run_file = model_dir / "last_run.txt"

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "budget_" if budget_based else ""
        filename = f"{prefix}dt_{format_dt(dt)}_T{max_train_T}_{timestamp}"
        model_root = model_dir / filename
        append_existing = False
    else:
        model_root = output_dir.expanduser().resolve()
        filename = model_root.name
        append_existing = model_root.exists()
    model_root.mkdir(parents=True, exist_ok=True)

    if debug:
        loss_dir = loss_root / filename
        loss_dir.mkdir(parents=True, exist_ok=True)
    else:
        loss_dir = None

    fixed_dir = model_root / "fixed"
    adaptive_dir = model_root / "adaptive"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    adaptive_dir.mkdir(parents=True, exist_ok=True)

    return model_root, fixed_dir, adaptive_dir, loss_dir, last_run_file, append_existing


def checkpoint_wall_time(checkpoint):
    metadata = checkpoint.get("metadata", {})
    if "wall_time_seconds" in metadata:
        return float(metadata["wall_time_seconds"])
    return float(metadata["train_wall_clock_seconds"])


def fixed_budget_wall_time(fixed_dir: Path, max_T: int):
    wall_times_by_T = {T: [] for T in range(1, max_T + 1)}
    for model_path in Path(fixed_dir).glob("mlp_T*.pt"):
        info = model_info(model_path)
        if info is None or info[0] is None:
            continue
        train_T, _ = info
        if train_T not in wall_times_by_T:
            continue
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        wall_times_by_T[train_T].append(checkpoint_wall_time(checkpoint))

    missing_Ts = [T for T, values in wall_times_by_T.items() if not values]
    if missing_Ts:
        raise FileNotFoundError(
            f"Missing fixed model wall-time metadata for T values: {missing_Ts}"
        )

    return float(sum(mean(values) for values in wall_times_by_T.values()))


def save_model(
    model,
    cfg,
    seed,
    save_dir=config.MODEL_DIR,
    T=None,
    adaptive=False,
    metadata=None,
    adaptive_method: str | None = None,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if adaptive:
        method_suffix = f"{adaptive_method_abbreviation(adaptive_method)}_"
        filename = f"adaptive_mlp_{method_suffix}seed{seed}_{timestamp}.pt"
    else:
        filename = f"mlp_T{T}_seed{seed}_{timestamp}.pt"

    model_path = save_dir / filename

    save_dict = {
        "model_state_dict": model.state_dict(),
        "train_T": T,
        "seed": seed,
        "config": {
            "input_size": cfg.input_size,
            "output_size": cfg.output_size,
            "layer_widths": cfg.layer_widths,
            "residual_connections": cfg.residual_connections,
            "k": cfg.k,
        },
    }
    if metadata is not None:
        save_dict["metadata"] = metadata

    torch.save(save_dict, model_path)
    print(f"Model saved to {model_path}")

    return model_path
