from datetime import datetime
from pathlib import Path
import re
import torch

import adaptive_horizon.config as config
from adaptive_horizon.training.methods import adaptive_method_abbreviation
from adaptive_horizon.utils import format_dt


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


def resolve_dirs(dt, append: bool, max_train_T: int, debug: bool, budget_based: bool):
    last_run_file = config.MODEL_DIR / "last_run.txt"

    if append:
        if not last_run_file.exists():
            raise FileNotFoundError(
                f"Cannot append: {last_run_file} does not exist. Run training without --append first."
            )

        model_root = Path(last_run_file.read_text().strip()).resolve()
        filename = model_root.name
        if not model_root.exists():
            raise FileNotFoundError(
                "Cannot append: model directory referenced by last_run.txt was not found."
            )
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "budget_" if budget_based else ""
        filename = f"{prefix}dt_{format_dt(dt)}_T{max_train_T}_{timestamp}"
        model_root = config.MODEL_DIR / filename
        model_root.mkdir(parents=True, exist_ok=True)

    if debug:
        loss_dir = config.LOSS_DIR / filename
        loss_dir.mkdir(parents=True, exist_ok=True)
    else:
        loss_dir = None

    fixed_dir = model_root / "fixed"
    adaptive_dir = model_root / "adaptive"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    adaptive_dir.mkdir(parents=True, exist_ok=True)

    return model_root, fixed_dir, adaptive_dir, loss_dir, last_run_file


def save_model(
    model,
    cfg,
    seed,
    save_dir=config.MODEL_DIR,
    T=None,
    adaptive=False,
    metadata=None,
    var: int | None = None,
    adaptive_method: str | None = None,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if adaptive:
        method_suffix = f"{adaptive_method_abbreviation(adaptive_method)}_"
        var_suffix = f"var{var}_" if var is not None else ""
        filename = f"adaptive_mlp_{method_suffix}seed{seed}_{var_suffix}{timestamp}.pt"
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
