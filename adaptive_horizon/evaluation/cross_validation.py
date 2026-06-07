from torch.utils.data import DataLoader
import argparse
import json
import re
import numpy as np
from pathlib import Path

from adaptive_horizon.training.methods import (
    ADAPTIVE_HORIZON,
    CURRICULUM_HORIZON,
    GRADIENT_SCALING_HORIZON,
    WEIGHTED_LOSS,
)
import adaptive_horizon.config as config
from adaptive_horizon.data.dataset import LorenzDataset, collate_fn
from adaptive_horizon.training.loss import validation_loss
from adaptive_horizon.visualization.plotting import plot_mse
from adaptive_horizon.evaluation.utils import (
    load_model,
    save_cross_validation_results,
    get_last_run,
    summarize_cross_validation,
)


def get_dt_from_model_dir(model_dir: Path):
    match = re.search(r"dt_(\d+)_.+$", model_dir.name)
    if not match:
        raise ValueError(
            "Could not infer dt from model directory name. "
            f"Expected format 'dt_{{dt}}_{{timestamp}}', got: {model_dir.name}"
        )
    digits = match.group(1)
    return float(digits) / (10 ** len(digits))


def get_T_values(model_dir: Path):
    """Get unique train_T values from model files matching mlp_T{T}*.pt"""
    model_files = list(model_dir.glob("mlp_T*.pt"))
    train_Ts = set()
    for f in model_files:
        match = re.search(r"mlp_T(\d+)", f.name)
        if match:
            train_Ts.add(int(match.group(1)))
    return sorted(train_Ts)


def get_model_paths(train_Ts, model_dir=config.MODEL_DIR):
    """Get all model paths for each train_T."""
    model_paths = {T: [] for T in train_Ts}
    for f in sorted(model_dir.glob("mlp_T*.pt")):
        match = re.search(r"mlp_T(\d+)", f.name)
        if match:
            T = int(match.group(1))
            if T in model_paths:
                model_paths[T].append(f)
    return model_paths


def get_adaptive_paths(model_dir=config.MODEL_DIR):
    """Get all adaptive model paths."""
    return sorted(model_dir.glob("adaptive_mlp*.pt"))


def get_adaptive_method(checkpoint):
    metadata = checkpoint.get("metadata", {})
    adaptive_metadata = metadata.get("adaptive", {})
    return adaptive_metadata.get("method")


def filter_adaptive_paths(adaptive_paths, adaptive_method=None):
    if adaptive_method is None:
        return adaptive_paths

    filtered_paths = []
    for model_path in adaptive_paths:
        _, checkpoint = load_model(model_path)
        if get_adaptive_method(checkpoint) == adaptive_method:
            filtered_paths.append(model_path)
    return filtered_paths


def get_normalization_stats(checkpoint):
    metadata = checkpoint.get("metadata", {})
    return metadata.get("normalization_stats")


def get_history_window(checkpoint):
    metadata = checkpoint.get("metadata", {})
    if "history_window" in metadata:
        return int(metadata["history_window"])

    model_config = checkpoint.get("config", {})
    input_size = model_config.get("input_size")
    if input_size is not None:
        return int(input_size) // config.INPUT_DIM

    return config.HISTORY_WINDOW


def make_eval_loader(max_val_T, dt, normalization_stats=None, history_window=None):
    burn_in_steps = config.resolve_burn_in_steps(dt)
    eval_dataset = LorenzDataset(
        num_trajectories=config.NUM_TRAJECTORIES,
        steps_per_trajectory=config.STEPS_PER_TRAJECTORY,
        T=max_val_T,
        dt=dt,
        normalize=True,
        seed=config.EVAL_SEED,
        burn_in=burn_in_steps,
        history_window=history_window or config.HISTORY_WINDOW,
        normalization_stats=normalization_stats,
    )
    return DataLoader(
        eval_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )


def eval_loader_cache_key(normalization_stats, history_window):
    if normalization_stats is None:
        return None, history_window
    mean = tuple(float(value) for value in normalization_stats["mean"])
    std = tuple(float(value) for value in normalization_stats["std"])
    return mean, std, history_window


def cross_validate_models(
    fixed_paths,
    adaptive_paths,
    T_values=None,
    dt=config.DT,
    device=config.DEVICE,
):
    """
    Evaluate models across different validation horizons.

    Args:
        fixed_paths: dict of {train_T: [list of model paths]}
        adaptive_paths: list of adaptive model paths
        T_values: list of horizon values
        dt: time step for simulation
        device: CPU or GPU

    Returns:
        evaluation_records: list of per-model, per-validation-horizon MSE records.
    """
    eval_loaders = {}

    def get_eval_loader(checkpoint):
        normalization_stats = get_normalization_stats(checkpoint)
        history_window = get_history_window(checkpoint)
        key = eval_loader_cache_key(normalization_stats, history_window)
        if key not in eval_loaders:
            eval_loaders[key] = make_eval_loader(
                max(T_values), dt, normalization_stats, history_window
            )
        return eval_loaders[key]

    evaluation_records = []

    for T in T_values:
        print(f"\nEvaluating models for T={T}")
        for model_path in fixed_paths[T]:
            model, checkpoint = load_model(model_path)
            model = model.to(device)
            eval_loader = get_eval_loader(checkpoint)
            seed = checkpoint.get("seed")
            model_records = []

            for val_T in T_values:
                mse = validation_loss(model, eval_loader, val_T, device)
                record = {
                    "model_type": "fixed",
                    "seed": seed,
                    "train_T": T,
                    "val_T": val_T,
                    "mse": mse,
                    "model_file": model_path.name,
                    "model_path": str(model_path),
                }
                evaluation_records.append(record)
                model_records.append(record)
            print(
                f"  Model {model_path.name}: mean MSE = {np.mean([record['mse'] for record in model_records]):.6f}"
            )

    if adaptive_paths:
        print("\nEvaluating adaptive models")
        for model_path in adaptive_paths:
            model, checkpoint = load_model(model_path)
            model = model.to(device)
            eval_loader = get_eval_loader(checkpoint)
            seed = checkpoint.get("seed")
            method = get_adaptive_method(checkpoint)
            model_records = []

            for val_T in T_values:
                mse = validation_loss(model, eval_loader, val_T, device)
                record = {
                    "model_type": "adaptive",
                    "adaptive_method": method,
                    "seed": seed,
                    "train_T": None,
                    "val_T": val_T,
                    "mse": mse,
                    "model_file": model_path.name,
                    "model_path": str(model_path),
                }
                evaluation_records.append(record)
                model_records.append(record)
            print(
                f"  Model {model_path.name}: mean MSE = {np.mean([record['mse'] for record in model_records]):.6f}"
            )

    return evaluation_records


def load_cross_validation_results(
    cached: Path = None, save_dir: Path = config.EVAL_DIR
):
    save_dir = Path(save_dir)
    results_file = Path(cached)

    if not results_file.exists():
        raise FileNotFoundError(
            f"Cached cross-validation results not found: {results_file}"
        )

    with open(results_file, "r") as f:
        payload = json.load(f)

    if "evaluation_records" not in payload:
        raise ValueError(
            "Cached results file does not contain raw evaluation records. "
            "Re-run cross-validation to regenerate it."
        )

    return results_file, payload


def cached_fixed_records(payload, T_values):
    T_values = set(T_values)
    return [
        record
        for record in payload["evaluation_records"]
        if record["model_type"] == "fixed"
        and record["train_T"] in T_values
        and record["val_T"] in T_values
    ]


def cross_validation(
    model_dir=None,
    fixed_dir=None,
    max_T=None,
    adaptive_method=None,
    cached=None,
    save_dir=config.EVAL_DIR,
    device=config.DEVICE,
):
    save_dir = Path(save_dir)

    if model_dir is None:
        model_dir = get_last_run(config.MODEL_DIR)
    else:
        model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if cached is not None and fixed_dir is not None:
        raise ValueError(
            "--cached and --fixed-dir both provide fixed baselines; use one"
        )

    if cached is not None:
        results_file, payload = load_cross_validation_results(cached, save_dir)
        metadata = payload["metadata"]
        dt = float(metadata["dt"])
        model_dir_dt = get_dt_from_model_dir(model_dir)
        if not np.isclose(dt, model_dir_dt):
            raise ValueError(
                "Cached fixed results and adaptive model directory use different dt values: "
                f"cached dt={dt}, model-dir dt={model_dir_dt}"
            )

        T_values = list(range(1, int(metadata["T_max"]) + 1))
        if max_T is not None:
            T_values = [T for T in T_values if T <= max_T]
            if not T_values:
                print(f"No cached fixed records found with T <= {max_T}")
                return

        fixed_records = cached_fixed_records(payload, T_values)
        if not fixed_records:
            print("No fixed records found in cached cross-validation results")
            return

        adaptive_paths = filter_adaptive_paths(
            get_adaptive_paths(model_dir), adaptive_method
        )

        adaptive_records = cross_validate_models(
            {T: [] for T in T_values}, adaptive_paths, T_values, dt, device
        )
        evaluation_records = fixed_records + adaptive_records

    else:
        if fixed_dir is None:
            fixed_dir = model_dir
        else:
            fixed_dir = Path(fixed_dir)
        if not fixed_dir.exists():
            raise FileNotFoundError(f"Fixed model directory not found: {fixed_dir}")

        dt = get_dt_from_model_dir(model_dir)

        T_values = get_T_values(fixed_dir)
        if not T_values:
            print(f"No fixed models found to evaluate in {fixed_dir}")
            return

        if max_T is not None:
            T_values = [T for T in T_values if T <= max_T]
            if not T_values:
                print(f"No models found with T <= {max_T}")
                return

        fixed_paths = get_model_paths(T_values, fixed_dir)
        adaptive_paths = filter_adaptive_paths(
            get_adaptive_paths(model_dir), adaptive_method
        )

        evaluation_records = cross_validate_models(
            fixed_paths, adaptive_paths, T_values, dt, device
        )

    summary = summarize_cross_validation(evaluation_records, T_values, T_values)

    save_cross_validation_results(
        evaluation_records,
        summary,
        max_T,
        dt,
        model_dir,
        fixed_dir=f"cached:{results_file}" if cached else fixed_dir,
        save_dir=save_dir,
    )
    plot_mse(summary, save_dir, dt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing adaptive models, and fixed models unless --fixed-dir is set (default: reads from last_run.txt)",
    )
    parser.add_argument(
        "--fixed-dir",
        type=str,
        default=None,
        help="Directory containing fixed-horizon models; defaults to --model-dir",
    )
    parser.add_argument(
        "--max-T",
        type=int,
        default=None,
        help="Maximum T for evaluation (default: all T values found in model_dir)",
    )
    parser.add_argument(
        "--adaptive-method",
        choices=[
            ADAPTIVE_HORIZON,
            WEIGHTED_LOSS,
            CURRICULUM_HORIZON,
            GRADIENT_SCALING_HORIZON,
        ],
        default=None,
        help="Evaluate only adaptive models trained with the selected method",
    )
    parser.add_argument(
        "--cached",
        type=str,
        default=None,
        help="Reuse fixed-model records from cached cross-validation results and evaluate adaptive models from --model-dir",
    )
    args = parser.parse_args()

    cross_validation(
        model_dir=args.model_dir,
        fixed_dir=args.fixed_dir,
        max_T=args.max_T,
        adaptive_method=args.adaptive_method,
        cached=args.cached,
    )


if __name__ == "__main__":
    main()
