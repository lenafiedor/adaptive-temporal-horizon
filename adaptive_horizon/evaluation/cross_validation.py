import argparse
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

import adaptive_horizon.config as config
from adaptive_horizon.data.dataset import TrajectoryDataset, collate_fn
from adaptive_horizon.dynamics.systems import SYSTEM_CHOICES
from adaptive_horizon.evaluation.utils import (
    get_checkpoint_normalization_stats,
    get_dt_from_model_dir,
    load_model,
    save_cross_validation_results,
    summarize_cross_validation,
)
from adaptive_horizon.training.loss import validation_loss
from adaptive_horizon.training.utils import model_info, resolve_burn_in_steps
from adaptive_horizon.visualization.plotting import plot_mse


def get_fixed_paths(
    train_Ts: list[int] | None = None,
    model_dir: Path = config.MODEL_DIR,
) -> dict[int, list[Path]]:
    model_paths: dict[int, list[Path]] = {}
    for model_path in sorted(model_dir.glob("mlp_T*.pt")):
        info = model_info(model_path)
        if info is None or info[0] is None:
            continue
        train_T = info[0]
        if train_Ts is None or train_T in train_Ts:
            model_paths.setdefault(train_T, []).append(model_path)

    if train_Ts is None:
        return dict(sorted(model_paths.items()))
    return {train_T: model_paths.get(train_T, []) for train_T in train_Ts}


def get_adaptive_paths(model_dir: Path = config.MODEL_DIR) -> list[Path]:
    return sorted(model_dir.glob("adaptive_mlp*.pt"))


def get_adaptive_method(checkpoint):
    return checkpoint.get("metadata", {}).get("adaptive", {}).get("method")


def get_training_wall_time(checkpoint):
    metadata = checkpoint.get("metadata", {})
    key = "wall_time_seconds"
    if key not in metadata:
        key = "train_wall_clock_seconds"
    return {"wall_time_seconds": float(metadata[key])}


def make_eval_loader(
    max_val_T, dt, normalization_stats=None, system_name=config.DEFAULT_SYSTEM
):
    split_gap = max(config.MAX_TRAIN_T, config.MAX_EVAL_T, max_val_T)
    dataset = TrajectoryDataset(
        T=max_val_T,
        dt=dt,
        system=system_name,
        normalize=True,
        seed=config.RANDOM_SEED,
        burn_in=resolve_burn_in_steps(dt),
        split="val",
        split_gap=split_gap,
        normalization_stats=normalization_stats,
    )
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )


def eval_loader_cache_key(normalization_stats):
    if normalization_stats is None:
        return None
    return tuple(
        tuple(float(value) for value in normalization_stats[name])
        for name in ("mean", "std")
    )


def cross_validate_models(
    fixed_paths: dict[int, list[Path]],
    adaptive_paths: list[Path],
    dt=config.DT,
    device=config.DEVICE,
    val_Ts: list[int] | None = None,
    system_name: str = config.DEFAULT_SYSTEM,
):
    train_Ts = list(fixed_paths)
    val_Ts = list(val_Ts) if val_Ts is not None else train_Ts
    eval_loaders = {}

    def get_eval_loader(checkpoint):
        normalization_stats = get_checkpoint_normalization_stats(checkpoint)
        checkpoint_system = checkpoint.get("metadata", {}).get("system") or system_name
        key = (checkpoint_system, eval_loader_cache_key(normalization_stats))
        if key not in eval_loaders:
            eval_loaders[key] = make_eval_loader(
                max(val_Ts),
                dt,
                normalization_stats,
                system_name=checkpoint_system,
            )
        return eval_loaders[key]

    def evaluate(model_path, model_type, train_T, model_val_Ts):
        model, checkpoint = load_model(model_path)
        model = model.to(device)
        eval_loader = get_eval_loader(checkpoint)
        records = []
        for val_T in model_val_Ts:
            record = {
                "model_type": model_type,
                "seed": checkpoint.get("seed"),
                "train_T": train_T,
                "val_T": val_T,
                "mse": validation_loss(model, eval_loader, val_T, device),
                **get_training_wall_time(checkpoint),
            }
            if model_type == "adaptive":
                record["adaptive_method"] = get_adaptive_method(checkpoint)
            records.append(record)
        print(
            f"  Model {model_path.name}: mean MSE = "
            f"{np.mean([record['mse'] for record in records]):.6f}"
        )
        return records

    evaluation_records = []
    if any(fixed_paths.values()):
        print("\nEvaluating fixed models")
        for train_T in train_Ts:
            for model_path in fixed_paths[train_T]:
                evaluation_records.extend(
                    evaluate(model_path, "fixed", train_T, val_Ts)
                )
    if adaptive_paths:
        print("\nEvaluating adaptive models")
        for model_path in adaptive_paths:
            evaluation_records.extend(evaluate(model_path, "adaptive", None, val_Ts))
    return evaluation_records


def load_cross_validation_results(cached: Path):
    results_file = Path(cached)
    if not results_file.exists():
        raise FileNotFoundError(
            f"Cached cross-validation results not found: {results_file}"
        )
    with results_file.open("r") as file:
        return json.load(file)


def cross_validation(
    model_dir,
    fixed_dir=None,
    output_dir=None,
    max_train_T=None,
    max_eval_T=config.MAX_EVAL_T,
    cached=None,
    device=config.DEVICE,
    metric="median",
    system=config.DEFAULT_SYSTEM,
):
    model_dir = Path(model_dir)
    cached = Path(cached) if cached is not None else None

    if cached:
        payload = load_cross_validation_results(cached)
        metadata = payload["metadata"]
        dt = float(metadata["dt"])
        output_dir = output_dir or Path(
            config.system_path(config.EVAL_DIR, metadata.get("system", system))
        )
        budget_based = cached.name.startswith("budget")
        train_Ts = list(range(1, metadata["max_train_T"] + 1))
        if max_train_T is not None:
            train_Ts = [train_T for train_T in train_Ts if train_T <= max_train_T]
        if max_eval_T is None or budget_based:
            max_eval_T = config.MAX_EVAL_T
        val_Ts = list(range(1, max_eval_T + 1))
        evaluation_records = [
            record
            for record in payload["evaluation_records"]
            if record["val_T"] in val_Ts
            and (record["model_type"] == "adaptive" or record["train_T"] in train_Ts)
        ]
        adaptive_dir = metadata.get("adaptive_dir", "cached")
        fixed_dir = metadata.get("fixed_dir", "cached")
    else:
        output_dir = output_dir or Path(config.system_path(config.EVAL_DIR, system))
        fixed_dir = Path(fixed_dir) if fixed_dir is not None else model_dir / "fixed"
        adaptive_dir = model_dir / "adaptive"
        dt = get_dt_from_model_dir(model_dir)
        budget_based = model_dir.name.startswith("budget")
        fixed_paths = get_fixed_paths(model_dir=fixed_dir)
        train_Ts = sorted(fixed_paths)
        if max_train_T is not None:
            train_Ts = [train_T for train_T in train_Ts if train_T <= max_train_T]
            fixed_paths = {train_T: fixed_paths[train_T] for train_T in train_Ts}
        if max_eval_T is None or budget_based:
            max_eval_T = config.MAX_EVAL_T
        val_Ts = list(range(1, max_eval_T + 1))
        evaluation_records = cross_validate_models(
            fixed_paths,
            get_adaptive_paths(adaptive_dir),
            dt=dt,
            device=device,
            val_Ts=val_Ts,
            system_name=system,
        )

    effective_max_train_T = max_train_T if max_train_T is not None else max(train_Ts)
    summary = summarize_cross_validation(evaluation_records, train_Ts, val_Ts)
    if cached is None:
        save_cross_validation_results(
            evaluation_records,
            summary,
            effective_max_train_T,
            dt,
            adaptive_dir,
            fixed_dir,
            output_dir,
            budget_based,
            system,
        )
    plot_mse(summary, output_dir, dt, effective_max_train_T, budget_based, metric)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixed-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-train-T", type=int, default=None)
    parser.add_argument("--max-eval-T", type=int, default=config.MAX_EVAL_T)
    parser.add_argument("--cached", type=Path, default=None)
    parser.add_argument("--metric", choices=("mean", "median"), default="median")
    parser.add_argument(
        "--system", choices=SYSTEM_CHOICES, default=config.DEFAULT_SYSTEM
    )
    args = parser.parse_args()
    cross_validation(**vars(args))


if __name__ == "__main__":
    main()
