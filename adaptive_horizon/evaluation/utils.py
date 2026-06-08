import torch
import torch.nn as nn
import json
import re
from datetime import datetime
from math import sqrt
from pathlib import Path
from statistics import median, stdev
from typing import Any

from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.utils import format_dt
import adaptive_horizon.config as config

LAST_RUN_FILE = "last_run.txt"


def get_dt_from_model_dir(model_dir: Path):
    match = re.search(r"dt_(\d+)_.+$", model_dir.name)
    if not match:
        raise ValueError(
            "Could not infer dt from model directory name. "
            f"Expected format 'dt_{{dt}}_{{timestamp}}', got: {model_dir.name}"
        )
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


def save_cross_validation_results(
    evaluation_records,
    summary,
    max_train_T,
    dt,
    adaptive_dir,
    fixed_dir=None,
    save_dir=config.EVAL_DIR,
    budget_based=False,
):
    """Save cross-validation summaries to a JSON file."""
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "budget_" if budget_based else ""
    results_file = save_dir / f"{prefix}mse_results_dt_{format_dt(dt)}_T{max_train_T}_{timestamp}.json"
    summary_metadata = summarize_metadata(summary)

    payload = {
        "metadata": {
            "created_at": timestamp,
            "dt": dt,
            "adaptive_dir": str(adaptive_dir),
            "fixed_dir": str(fixed_dir or adaptive_dir),
            "max_train_T": max_train_T,
            **summary_metadata,
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


def median_ci95(values):
    values = [float(value) for value in values]
    median_value = float(median(values))
    margin = float(1.96 * stdev(values) / sqrt(len(values)))
    return {
        "median_mse": median_value,
        "ci95_low": median_value - margin,
        "ci95_high": median_value + margin,
    }


def summarize_metadata(summary):
    fixed_summaries = summary.get("fixed", [])
    best_fixed = min(
        fixed_summaries,
        key=lambda train_summary: train_summary["overall"]["median_mse"],
        default=None,
    )
    metadata = {}
    if best_fixed is not None:
        metadata["best_train_T"] = best_fixed["train_T"]
        metadata["best_fixed_median_MSE"] = round(
            best_fixed["overall"]["median_mse"], 6
        )

    adaptive_summary = summary.get("adaptive")
    if adaptive_summary is not None:
        metadata["adaptive_median_MSE"] = round(
            adaptive_summary["overall"]["median_mse"], 6
        )

    return metadata


def summarize_by_eval_T(records, val_Ts):
    by_val_T = []
    for val_T in val_Ts:
        values = [record["mse"] for record in records if record["val_T"] == val_T]
        val_summary = median_ci95(values)
        by_val_T.append(
            {
                "eval_T": int(val_T),
                **val_summary,
            }
        )

    return by_val_T


def summarize_cross_validation(evaluation_records, train_Ts, val_Ts):
    summary: dict[str, Any] = {"fixed": [], "adaptive": None}

    for train_T in train_Ts:
        records_for_train_T = [
            record
            for record in evaluation_records
            if record["model_type"] == "fixed" and record["train_T"] == train_T
        ]
        overall = median_ci95(record["mse"] for record in records_for_train_T)

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
    adaptive_overall = median_ci95(record["mse"] for record in adaptive_records)

    summary["adaptive"] = {
        "overall": adaptive_overall,
        "by_eval_T": summarize_by_eval_T(adaptive_records, val_Ts),
    }

    return summary
