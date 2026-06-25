import torch
import torch.nn as nn
import json
import re
from datetime import datetime
from math import sqrt
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.utils import format_dt
import adaptive_horizon.config as config

LAST_RUN_FILE = "last_run.txt"


def get_dt_from_model_dir(model_dir: Path):
    for path in (model_dir, *model_dir.parents):
        match = re.search(r"dt_(\d+)(?:_|$)", path.name)
        if match:
            digits = match.group(1)
            return float(digits) / (10 ** len(digits))


def get_last_run(save_dir):
    last_run_file = Path(save_dir) / LAST_RUN_FILE
    if not last_run_file.exists():
        raise FileNotFoundError(
            "No last run found. Run training / cross-validation first."
        )

    with open(last_run_file, "r") as f:
        return Path(f.read().strip())


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

    model = MLP(mlp_config, random_seed=42)
    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint


def get_checkpoint_normalization_stats(checkpoint):
    metadata = checkpoint.get("metadata", {})
    return metadata.get("normalization_stats")


def summarize_wall_time(evaluation_records):
    totals = {
        "fixed": 0.0,
        "adaptive": 0.0,
    }
    for record in evaluation_records:
        model_type = record.get("model_type")
        wall_time = record.get("wall_time_seconds", "train_wall_clock_seconds")
        if model_type not in totals or wall_time is None:
            continue
        totals[model_type] += float(wall_time)

    return {
        "total_time_fixed": totals["fixed"],
        "total_time_adaptive": totals["adaptive"],
    }


def save_cross_validation_results(
    evaluation_records,
    summary,
    max_train_T,
    dt,
    adaptive_dir,
    fixed_dir=None,
    save_dir=config.EVAL_DIR,
    budget_based=False,
    system_name=config.DEFAULT_SYSTEM,
):
    """Save cross-validation summaries to a JSON file."""
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "budget_" if budget_based else ""
    results_file = (
        save_dir
        / f"{prefix}mse_results_dt_{format_dt(dt)}_T{max_train_T}_{timestamp}.json"
    )
    summary_metadata = summarize_metadata(summary)
    wall_time_metadata = summarize_wall_time(evaluation_records)

    payload = {
        "metadata": {
            "created_at": timestamp,
            "dt": dt,
            "system": system_name,
            "adaptive_dir": str(adaptive_dir),
            "fixed_dir": str(fixed_dir or adaptive_dir),
            "max_train_T": max_train_T,
            **summary_metadata,
            **wall_time_metadata,
        },
        "summary": summary,
        "evaluation_records": evaluation_records,
    }

    with open(results_file, "w") as f:
        json.dump(payload, f, indent=2)

    with open(save_dir / LAST_RUN_FILE, "w") as f:
        f.write(str(results_file))

    print(f"\nCross-validation results saved to {results_file}")
    return results_file


def calculate_stats(values):
    values = [float(value) for value in values]
    mean_value = float(mean(values))
    median_value = float(median(values))
    margin = float(1.96 * stdev(values) / sqrt(len(values))) if len(values) > 1 else 0.0
    return {
        "mean": mean_value,
        "mean_ci95_low": mean_value - margin,
        "mean_ci95_high": mean_value + margin,
        "median": median_value,
        "median_ci95_low": median_value - margin,
        "median_ci95_high": median_value + margin,
    }


def summarize_paired_deltas(
    evaluation_records, train_Ts, val_Ts, num_seeds=config.NUM_SEEDS
):
    seeds = range(num_seeds)
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
    by_val_T = {}
    all_deltas = []

    for val_T in val_Ts:
        fixed_candidates = [
            (
                int(train_T),
                [
                    mse_by_key[("fixed", int(train_T), val_T, seed)]
                    for seed in seeds
                    if ("fixed", int(train_T), val_T, seed) in mse_by_key
                ],
            )
            for train_T in train_Ts
        ]
        fixed_candidates = [
            (train_T, values) for train_T, values in fixed_candidates if values
        ]
        best_train_T, _ = min(
            fixed_candidates,
            key=lambda candidate: median(candidate[1]),
        )
        deltas = [
            mse_by_key[("fixed", best_train_T, val_T, seed)]
            - mse_by_key[("adaptive", None, val_T, seed)]
            for seed in seeds
            if ("fixed", best_train_T, val_T, seed) in mse_by_key
            and ("adaptive", None, val_T, seed) in mse_by_key
        ]
        adaptive_wins = sum(delta > 0 for delta in deltas)
        fixed_wins = sum(delta < 0 for delta in deltas)
        ties = len(deltas) - adaptive_wins - fixed_wins
        all_deltas.extend(deltas)

        by_val_T[str(int(val_T))] = {
            "best_train_T": best_train_T,
            "n_pairs": len(deltas),
            "adaptive_wins": adaptive_wins,
            "fixed_wins": fixed_wins,
            "ties": ties,
            "adaptive_win_rate": adaptive_wins / len(deltas),
            **calculate_stats(deltas),
        }

    overall_adaptive_wins = sum(delta > 0 for delta in all_deltas)
    overall_fixed_wins = sum(delta < 0 for delta in all_deltas)
    overall_ties = len(all_deltas) - overall_adaptive_wins - overall_fixed_wins
    return {
        "overall": {
            "n_pairs": len(all_deltas),
            "adaptive_wins": overall_adaptive_wins,
            "fixed_wins": overall_fixed_wins,
            "ties": overall_ties,
            "adaptive_win_rate": overall_adaptive_wins / len(all_deltas),
        },
        "by_val_T": by_val_T,
    }


def summarize_metadata(summary):
    fixed_summaries = summary.get("fixed", [])
    best_fixed = min(
        fixed_summaries,
        key=lambda train_summary: train_summary["overall"]["median"],
        default=None,
    )
    metadata = {}
    if best_fixed is not None:
        metadata["best_train_T"] = best_fixed["train_T"]
        metadata["best_fixed_median_MSE"] = round(best_fixed["overall"]["median"], 6)

    adaptive_summary = summary.get("adaptive")
    if adaptive_summary is not None:
        metadata["adaptive_median_MSE"] = round(
            adaptive_summary["overall"]["median"], 6
        )

    return metadata


def summarize_by_eval_T(records, val_Ts):
    by_val_T = []
    for val_T in val_Ts:
        values = [record["mse"] for record in records if record["val_T"] == val_T]
        by_val_T.append(
            {
                "eval_T": int(val_T),
                **calculate_stats(values),
            }
        )

    return by_val_T


def summarize_cross_validation(
    evaluation_records, train_Ts, val_Ts, num_seeds=config.NUM_SEEDS
):
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
    adaptive_overall = calculate_stats(record["mse"] for record in adaptive_records)

    summary["adaptive"] = {
        "overall": adaptive_overall,
        "by_eval_T": summarize_by_eval_T(adaptive_records, val_Ts),
    }
    summary["deltas"] = summarize_paired_deltas(
        evaluation_records, train_Ts, val_Ts, num_seeds
    )

    return summary
