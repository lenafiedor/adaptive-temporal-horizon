import argparse
import json
from pathlib import Path

import numpy as np
import torch

from adaptive_horizon import config
from adaptive_horizon.data.dataset import make_val_loader
from adaptive_horizon.data.santafe import SANTAFE, load_santafe
from adaptive_horizon.dynamics.lyapunov import compute_global_lyapunov
from adaptive_horizon.dynamics.systems import SYSTEM_CHOICES
from adaptive_horizon.evaluation.metrics import (
    DEFAULT_VPT_THRESHOLDS,
    ERROR_METRICS,
    evaluate_metrics,
)
from adaptive_horizon.evaluation.utils import (
    get_checkpoint_normalization_stats,
    get_dt_from_model_dir,
    load_model,
    save_cross_validation_results,
    summarize_cross_validation,
)
from adaptive_horizon.training.utils import find_fixed_models
from adaptive_horizon.visualization.plotting import plot_mse, plot_vpt


def get_fixed_paths(
    train_Ts: list[int] | None = None,
    model_dir: Path = config.MODEL_DIR,
) -> dict[int, list[Path]]:
    model_paths: dict[int, list[Path]] = {}
    for model_path, train_T, _ in find_fixed_models(model_dir):
        if train_Ts is None or train_T in train_Ts:
            model_paths.setdefault(train_T, []).append(model_path)

    if train_Ts is None:
        return dict(sorted(model_paths.items()))
    return {train_T: model_paths.get(train_T, []) for train_T in train_Ts}


def get_adaptive_paths(model_dir: Path = config.MODEL_DIR) -> list[Path]:
    return sorted(model_dir.glob("adaptive_mlp*.pt"))


def eval_loader_cache_key(normalization_stats):
    if normalization_stats is None:
        return None
    return tuple(
        tuple(float(value) for value in normalization_stats[name])
        for name in ("mean", "std")
    )


OBSERVED_SIGNATURE_FIELDS = (
    "checksum",
    "history_length",
    "split_bounds",
    "normalization_stats",
    "sample_time",
)


def observed_metadata(checkpoint):
    metadata = checkpoint.get("metadata", {}).get("observed_data")
    if not metadata:
        raise ValueError("Santa Fe checkpoint is missing observed-data metadata")
    return metadata


def load_observed_test_data(checkpoint, max_T, data_path=None, compute_scores=False):
    """Reload the checkpoint's recording, verifying it is the same data."""
    metadata = observed_metadata(checkpoint)
    return load_santafe(
        data_path or metadata["path"],
        max_T,
        compute_scores=compute_scores,
        metadata=metadata,
    )


def observed_lyapunov_exponent(model_paths, data_path=None):
    """Mean training neighbour-divergence rate (per sample) of the recording."""
    for model_path in model_paths:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        data = load_observed_test_data(checkpoint, 1, data_path, compute_scores=True)
        return float(data["score_stats"]["mean"])
    return None


def cross_validate_models(
    fixed_paths: dict[int, list[Path]],
    adaptive_paths: list[Path],
    dt=config.DT,
    device=config.DEVICE,
    val_Ts: list[int] | None = None,
    system_name: str = config.DEFAULT_SYSTEM,
    vpt_thresholds=DEFAULT_VPT_THRESHOLDS,
    data_path=None,
):
    train_Ts = list(fixed_paths)
    val_Ts = list(val_Ts) if val_Ts is not None else train_Ts
    eval_loaders = {}
    observed_signature = None

    def get_eval_loader(checkpoint):
        nonlocal observed_signature
        checkpoint_dt = checkpoint.get("metadata", {}).get("dt")
        if checkpoint_dt is not None and float(checkpoint_dt) != float(dt):
            raise ValueError(
                f"Checkpoint dt={checkpoint_dt} does not match evaluation dt={dt}"
            )
        normalization_stats = get_checkpoint_normalization_stats(checkpoint)
        checkpoint_system = checkpoint.get("metadata", {}).get("system") or system_name
        key = (checkpoint_system, eval_loader_cache_key(normalization_stats))
        if checkpoint_system == SANTAFE:
            # All compared models must share data, splits and history windows.
            metadata = observed_metadata(checkpoint)
            signature = json.dumps(
                {name: metadata[name] for name in OBSERVED_SIGNATURE_FIELDS},
                sort_keys=True,
            )
            if observed_signature not in (None, signature):
                raise ValueError(
                    "Santa Fe models must use identical data, splits and "
                    "history windows"
                )
            observed_signature = signature
            key += (signature,)
        if key not in eval_loaders:
            if checkpoint_system == SANTAFE:
                observed_data = load_observed_test_data(
                    checkpoint, max(val_Ts), data_path
                )
                # Measured data has a held-out test split; report on it.
                eval_loaders[key] = make_val_loader(
                    checkpoint_system,
                    dt,
                    max(val_Ts),
                    normalization_stats,
                    observed_data=observed_data,
                    split="test",
                )
            else:
                eval_loaders[key] = make_val_loader(
                    checkpoint_system, dt, max(val_Ts), normalization_stats
                )
        return eval_loaders[key]

    def evaluate(model_path, model_type, train_T, model_val_Ts):
        model, checkpoint = load_model(model_path)
        model = model.to(device)
        eval_loader = get_eval_loader(checkpoint)
        metrics = evaluate_metrics(
            model, eval_loader, model_val_Ts, vpt_thresholds, device
        )
        metadata = checkpoint.get("metadata", {})
        model_system = metadata.get("system") or system_name
        split = "test" if model_system == SANTAFE else "val"
        extra = {"system": model_system, "split": split}
        if "epochs_requested" in metadata:
            extra["epochs_requested"] = metadata["epochs_requested"]
        if model_type == "adaptive":
            extra["adaptive_method"] = metadata.get("adaptive", {}).get("method")
        if model_system == SANTAFE:
            extra["data_checksum"] = metadata["observed_data"]["checksum"]
        records = [
            {
                "model_type": model_type,
                "seed": checkpoint.get("seed"),
                "train_T": train_T,
                "val_T": val_T,
                **extra,
                **metrics[val_T],
            }
            for val_T in model_val_Ts
        ]
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


def reusable_fixed_records(payload, fixed_paths, fixed_dir, val_Ts, run_metadata):
    """Split fixed models into cached complete evaluations and paths to evaluate."""
    metadata = payload["metadata"]
    if any(metadata.get(key) != value for key, value in run_metadata.items()):
        return [], fixed_paths
    records = [
        record
        for record in payload["evaluation_records"]
        if record["model_type"] == "fixed"
        and record["train_T"] in fixed_paths
        and record["val_T"] in val_Ts
    ]
    cached_keys = {(r["train_T"], r["seed"], r["val_T"]) for r in records}
    reusable_models = {
        (train_T, seed)
        for train_T, seed, _ in cached_keys
        if all((train_T, seed, T) in cached_keys for T in val_Ts)
    }
    records = [r for r in records if (r["train_T"], r["seed"]) in reusable_models]
    seeds_by_path = {path: seed for path, _, seed in find_fixed_models(fixed_dir)}
    remaining_paths = {
        T: [path for path in paths if (T, seeds_by_path[path]) not in reusable_models]
        for T, paths in fixed_paths.items()
    }
    return records, remaining_paths


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
    error_metric="mse",
    vpt_thresholds=DEFAULT_VPT_THRESHOLDS,
    data_path=None,
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
        if max_eval_T is None:
            max_eval_T = config.MAX_EVAL_T
        val_Ts = list(range(1, max_eval_T + 1))
        evaluation_records = [
            record
            for record in payload["evaluation_records"]
            if record["val_T"] in val_Ts
            and (record["model_type"] == "adaptive" or record["train_T"] in train_Ts)
        ]
        lyapunov_exponent = metadata.get("lyapunov_exponent")
    else:
        output_dir = output_dir or Path(config.system_path(config.EVAL_DIR, system))
        fixed_dir = Path(fixed_dir) if fixed_dir is not None else model_dir / "fixed"
        adaptive_dir = model_dir / "adaptive"
        dt = get_dt_from_model_dir(model_dir)
        if dt is None:
            dt = get_dt_from_model_dir(fixed_dir) or config.DT
        budget_based = model_dir.name.startswith("budget")
        if system == SANTAFE:
            lyapunov_exponent = observed_lyapunov_exponent(
                [
                    *(
                        path
                        for paths in get_fixed_paths(model_dir=fixed_dir).values()
                        for path in paths
                    ),
                    *get_adaptive_paths(adaptive_dir),
                ],
                data_path,
            )
        else:
            lyapunov_exponent = float(
                compute_global_lyapunov(dt=config.INTEGRATION_DT, system=system)
            )
        run_metadata = {
            "dt": dt,
            "system": system,
            "vpt_thresholds": list(vpt_thresholds),
            "lyapunov_exponent": lyapunov_exponent,
        }
        fixed_paths = get_fixed_paths(model_dir=fixed_dir)
        train_Ts = sorted(fixed_paths)
        if max_train_T is not None:
            train_Ts = [train_T for train_T in train_Ts if train_T <= max_train_T]
            fixed_paths = {train_T: fixed_paths[train_T] for train_T in train_Ts}
        if max_eval_T is None:
            max_eval_T = config.MAX_EVAL_T
        val_Ts = list(range(1, max_eval_T + 1))
        cached_fixed_records = []
        if budget_based and max_train_T is not None and max_train_T > 1:
            previous_results = sorted(
                Path(output_dir).glob(f"budget_mse_results_*_T{max_train_T - 1}_*.json")
            )
            if previous_results:
                previous_result = previous_results[-1]
                payload = load_cross_validation_results(previous_result)
                cached_fixed_records, fixed_paths = reusable_fixed_records(
                    payload, fixed_paths, fixed_dir, val_Ts, run_metadata
                )
                print(
                    f"Reusing {len(cached_fixed_records)} fixed-model evaluation "
                    f"records from {previous_result}"
                )

        evaluation_records = cached_fixed_records + cross_validate_models(
            fixed_paths,
            get_adaptive_paths(adaptive_dir),
            dt=dt,
            device=device,
            val_Ts=val_Ts,
            system_name=system,
            vpt_thresholds=vpt_thresholds,
            data_path=data_path,
        )
        systems = {record["system"] for record in evaluation_records}
        if systems - {system}:
            raise ValueError(
                f"Models in {model_dir} were trained on {sorted(systems)}, "
                f"not --system {system}"
            )

    effective_max_train_T = max_train_T or max(train_Ts, default=0)
    summary = summarize_cross_validation(evaluation_records, train_Ts, val_Ts)
    methods = sorted(
        {
            record.get("adaptive_method")
            for record in evaluation_records
            if record["model_type"] == "adaptive" and record.get("adaptive_method")
        }
    )
    if len(methods) > 1:
        # Several adaptive methods in one directory: compare each separately.
        fixed_records = [r for r in evaluation_records if r["model_type"] == "fixed"]
        summary["adaptive_by_method"] = {
            method: summarize_cross_validation(
                fixed_records
                + [r for r in evaluation_records if r.get("adaptive_method") == method],
                train_Ts,
                val_Ts,
            )
            for method in methods
        }
    evaluation_split = (
        "test"
        if any(record.get("split") == "test" for record in evaluation_records)
        else "val"
    )
    if cached is None:
        save_cross_validation_results(
            evaluation_records,
            summary,
            effective_max_train_T,
            fixed_dir,
            output_dir,
            budget_based,
            run_metadata,
        )
    if not summary["fixed"]:
        print("No fixed models evaluated; skipping plots")
        return
    for method, method_summary in summary.get(
        "adaptive_by_method", {None: summary}
    ).items():
        method_dir = Path(output_dir) / method if method else output_dir
        plot_mse(
            method_summary,
            method_dir,
            dt,
            effective_max_train_T,
            budget_based,
            metric,
            error_metric,
            evaluation_split=evaluation_split,
        )
        if lyapunov_exponent is not None:
            plot_vpt(method_summary, method_dir, dt, lyapunov_exponent, metric)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixed-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-train-T", type=int, default=None)
    parser.add_argument("--max-eval-T", type=int, default=config.MAX_EVAL_T)
    parser.add_argument("--cached", type=Path, default=None)
    parser.add_argument("--metric", choices=("mean", "median"), default="median")
    parser.add_argument("--error-metric", choices=ERROR_METRICS, default="mse")
    parser.add_argument(
        "--system", choices=(*SYSTEM_CHOICES, SANTAFE), default=config.DEFAULT_SYSTEM
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override the checkpoints' local Santa Fe recording path",
    )
    args = parser.parse_args()
    cross_validation(**vars(args))


if __name__ == "__main__":
    main()
