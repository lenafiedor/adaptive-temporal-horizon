import torch
import torch.nn as nn
import json
from datetime import datetime
import Path

from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.utils import format_dt
import adaptive_horizon.config as config

LAST_RUN_FILE = "last_run.txt"


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
    config = MLPConfig(
        input_size=cfg["input_size"],
        output_size=cfg["output_size"],
        layer_widths=cfg["layer_widths"],
        residual_connections=cfg["residual_connections"],
        k=cfg.get("k"),
        activation=nn.ReLU(),
    )

    model = MLP(config, random_seed=42)
    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint


def save_cross_validation_results(
    evaluation_records,
    max_T,
    best_train_T,
    best_fixed_mse,
    mean_adaptive_mse,
    dt,
    adaptive_dir,
    fixed_dir=None,
    save_dir=config.EVAL_DIR,
):
    """Save cross-validation summaries to a JSON file."""
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = save_dir / f"mse_results_dt_{format_dt(dt)}_{timestamp}.json"
    payload = {
        "metadata": {
            "created_at": timestamp,
            "dt": dt,
            "adaptive_dir": str(adaptive_dir),
            "fixed_dir": str(fixed_dir or adaptive_dir),
            "max_train_T": max_T,
            "best_train_T": best_train_T,
            "best_fixed_MSE": round(best_fixed_mse, 6),
            "mean_adaptive_MSE": round(mean_adaptive_mse, 6),
        },
        "evaluation_records": evaluation_records,
    }

    with open(results_file, "w") as f:
        json.dump(payload, f, indent=2)

    with open(save_dir / LAST_RUN_FILE, "w") as f:
        f.write(str(results_file))

    print(f"Cross-validation results saved to {results_file}")
    return results_file
