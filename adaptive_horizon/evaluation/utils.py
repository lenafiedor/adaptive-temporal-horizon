import json
from math import isfinite, sqrt
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import torch
from torch import nn

from adaptive_horizon import config
from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.utils import format_dt, timestamp


def get_dt_from_model_dir(model_dir: Path):
    model_dir = Path(model_dir)
    checkpoint_paths = []
    for directory in (model_dir, model_dir / "fixed", model_dir / "adaptive"):
        checkpoint_paths.extend(sorted(directory.glob("*.pt")))

    metadata_dts = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        dt = checkpoint.get("metadata", {}).get("dt")
        if dt is not None:
            metadata_dts.append(float(dt))

    if metadata_dts:
        unique_dts = set(metadata_dts)
        if len(unique_dts) > 1:
            raise ValueError(
                f"Conflicting dt values in model metadata for {model_dir}: "
                f"{sorted(unique_dts)}"
            )
        return metadata_dts[0]
    return None


def load_model(model_path):
    checkpoint = torch.load(model_path, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    cfg = checkpoint["config"]
    mlp_config = MLPConfig(
        input_size=cfg["input_size"],
        output_size=cfg["output_size"],
        layer_widths=cfg["layer_widths"],
        residual_connections=cfg["residual_connections"],
        k=cfg.get("k"),
        activation=nn.ReLU(),
    )

    model = MLP(mlp_config, random_seed=config.RANDOM_SEED)
    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint


def get_checkpoint_normalization_stats(checkpoint):
    metadata = checkpoint.get("metadata", {})
    return metadata.get("normalization_stats")


def save_cross_validation_results(
    evaluation_records,
    summary,
    max_train_T,
    fixed_dir,
    save_dir,
    budget_based,
    run_metadata,
):
    """Save cross-validation summaries to a JSON file.

    ``run_metadata`` holds the evaluation settings shared by all records
    (``dt``, ``system``, ``vpt_thresholds`` and ``lyapunov_exponent``).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    prefix = "budget_" if budget_based else ""
    results_file = save_dir / (
        f"{prefix}mse_results_dt_{format_dt(run_metadata['dt'])}"
        f"_T{max_train_T}_{timestamp()}.json"
    )
    metadata = {
        **run_metadata,
        "fixed_dir": str(fixed_dir),
        "max_train_T": max_train_T,
    }
    if summary["fixed"]:
        best_fixed = min(summary["fixed"], key=lambda item: item["overall"]["median"])
        metadata["best_train_T"] = best_fixed["train_T"]
        metadata["best_fixed_median_MSE"] = round(best_fixed["overall"]["median"], 6)
    if summary["adaptive"] is not None:
        metadata["adaptive_median_MSE"] = round(
            summary["adaptive"]["overall"]["median"], 6
        )
    payload = {
        "metadata": metadata,
        "summary": summary,
        "evaluation_records": evaluation_records,
    }

    with open(results_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nCross-validation results saved to {results_file}")
    return results_file


def calculate_stats(values):
    values = [float(value) for value in values]
    mean_value = float(mean(values))
    median_value = float(median(values))
    margin = (
        float(1.96 * stdev(values) / sqrt(len(values)))
        if len(values) > 1 and all(isfinite(value) for value in values)
        else 0.0
    )
    return {"mean": mean_value, "median": median_value, "ci95": margin}


def win_counts(deltas):
    adaptive_wins = sum(delta > 0 for delta in deltas)
    fixed_wins = sum(delta < 0 for delta in deltas)
    return {
        "n_pairs": len(deltas),
        "adaptive_wins": adaptive_wins,
        "fixed_wins": fixed_wins,
        "ties": len(deltas) - adaptive_wins - fixed_wins,
        "adaptive_win_rate": adaptive_wins / len(deltas),
    }


def summarize_paired_deltas(evaluation_records, train_Ts, val_Ts):
    """Compare adaptive models with same-seed best fixed models per ``val_T``.

    Only seeds evaluated for both model types are paired. Returns None when
    no fixed and adaptive model share a seed.
    """
    mse_by_key = {
        (
            record["model_type"],
            record.get("train_T"),
            record["val_T"],
            int(record["seed"]),
        ): float(record["mse"])
        for record in evaluation_records
        if record.get("seed") is not None
    }
    seeds = {seed for model_type, *_, seed in mse_by_key if model_type == "adaptive"}
    by_val_T = {}
    all_deltas = []

    for val_T in val_Ts:
        adaptive_mse = {
            seed: mse_by_key[key]
            for seed in seeds
            if (key := ("adaptive", None, val_T, seed)) in mse_by_key
        }
        fixed_mse = {
            int(train_T): {
                seed: mse_by_key[key]
                for seed in adaptive_mse
                if (key := ("fixed", int(train_T), val_T, seed)) in mse_by_key
            }
            for train_T in train_Ts
        }
        fixed_mse = {train_T: mse for train_T, mse in fixed_mse.items() if mse}
        if not fixed_mse:
            continue
        best_train_T = min(
            fixed_mse, key=lambda train_T: median(fixed_mse[train_T].values())
        )
        deltas = [
            mse - adaptive_mse[seed] for seed, mse in fixed_mse[best_train_T].items()
        ]
        all_deltas.extend(deltas)
        by_val_T[str(int(val_T))] = {
            "best_train_T": best_train_T,
            **win_counts(deltas),
            **calculate_stats(deltas),
        }

    if not all_deltas:
        return None
    return {"overall": win_counts(all_deltas), "by_val_T": by_val_T}


def summarize_by_eval_T(records, val_Ts):
    """MSE, NRMSE, SMAPE, and (at the longest horizon) VPT statistics."""
    by_val_T = []
    for val_T in val_Ts:
        records_at_T = [record for record in records if record["val_T"] == val_T]
        stats: dict[str, Any] = {
            "eval_T": int(val_T),
            **calculate_stats(record["mse"] for record in records_at_T),
        }
        # Results saved before these metrics were added only have MSE.
        if all("nrmse" in record for record in records_at_T):
            stats["metrics"] = {
                name: calculate_stats(record[name] for record in records_at_T)
                for name in ("nrmse", "smape")
            }
        if all("vpt" in record for record in records_at_T):
            stats["vpt"] = {
                threshold: {
                    field: calculate_stats(
                        record["vpt"][threshold][field] for record in records_at_T
                    )
                    for field in vpt
                }
                for threshold, vpt in records_at_T[0]["vpt"].items()
            }
        by_val_T.append(stats)

    return by_val_T


def summarize_cross_validation(evaluation_records, train_Ts, val_Ts):
    summary: dict[str, Any] = {"fixed": [], "adaptive": None}

    for train_T in train_Ts:
        records_for_train_T = [
            record
            for record in evaluation_records
            if record["model_type"] == "fixed" and record["train_T"] == train_T
        ]
        overall = calculate_stats(record["mse"] for record in records_for_train_T)

        summary["fixed"].append(
            {
                "train_T": int(train_T),
                "overall": overall,
                "by_eval_T": summarize_by_eval_T(records_for_train_T, val_Ts),
            }
        )

    adaptive_records = [
        record for record in evaluation_records if record["model_type"] == "adaptive"
    ]
    if not adaptive_records:
        return summary

    adaptive_overall = calculate_stats(record["mse"] for record in adaptive_records)

    summary["adaptive"] = {
        "overall": adaptive_overall,
        "by_eval_T": summarize_by_eval_T(adaptive_records, val_Ts),
    }
    deltas = summarize_paired_deltas(evaluation_records, train_Ts, val_Ts)
    if deltas is not None:
        summary["deltas"] = deltas

    return summary
