from datetime import datetime
from pathlib import Path
import re

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
