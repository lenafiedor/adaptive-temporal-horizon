import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from adaptive_horizon import config
from adaptive_horizon.visualization.plot_budget_resources import (
    EvalScope,
    aggregate_values,
    parse_eval_scope,
    select_fixed_result,
)
from adaptive_horizon.visualization.plotting import COLOR_TRAIN


@dataclass(frozen=True)
class OptimalHorizon:
    epochs: int
    train_T: int
    mse: float
    seed_train_Ts: dict[int, int]
    source_path: Path


def infer_epochs(fixed_dir: str, budget_epochs: int) -> int:
    match = re.search(r"_(\d+)_epochs(?:/|$)", fixed_dir)
    if match:
        return int(match.group(1))
    if "budget_based_" in fixed_dir:
        return budget_epochs
    raise ValueError(f"Could not infer epoch budget from fixed directory: {fixed_dir}")


def select_seed_train_T(records, seed: int, metric: str, eval_scope: EvalScope) -> int:
    values_by_train_T = {}
    for record in records:
        if record["model_type"] != "fixed" or int(record["seed"]) != seed:
            continue
        if eval_scope.mode == "single" and int(record["val_T"]) != eval_scope.eval_T:
            continue
        values_by_train_T.setdefault(int(record["train_T"]), []).append(
            float(record["mse"])
        )

    if not values_by_train_T:
        raise ValueError(f"No cross-validation records found for seed {seed}")

    return min(
        values_by_train_T,
        key=lambda train_T: aggregate_values(values_by_train_T[train_T], metric),
    )


def load_optimal_horizon(
    path: Path,
    metric: str,
    eval_scope: EvalScope,
    budget_epochs: int,
) -> OptimalHorizon:
    with path.open("r") as f:
        payload = json.load(f)

    metadata = payload.get("metadata", {})
    fixed_dir = str(metadata.get("fixed_dir", ""))
    epochs = infer_epochs(fixed_dir, budget_epochs)
    best = select_fixed_result(payload["summary"], metric, eval_scope)
    if eval_scope.mode == "single":
        mse = next(
            item[metric]
            for item in best["by_eval_T"]
            if int(item["eval_T"]) == eval_scope.eval_T
        )
    else:
        mse = best["overall"][metric]

    records = payload["evaluation_records"]
    seeds = sorted(
        {
            int(record["seed"])
            for record in records
            if record["model_type"] == "fixed" and record.get("seed") is not None
        }
    )

    return OptimalHorizon(
        epochs=epochs,
        train_T=int(best["train_T"]),
        mse=float(mse),
        seed_train_Ts={
            seed: select_seed_train_T(records, seed, metric, eval_scope)
            for seed in seeds
        },
        source_path=path,
    )


def load_results(results_dir, metric, eval_scope, budget_epochs):
    paths = sorted(Path(results_dir).glob("mse_results_*.json"))
    if not paths:
        raise FileNotFoundError(f"No MSE result files found in {results_dir}")

    by_epochs = {}
    for path in paths:
        result = load_optimal_horizon(path, metric, eval_scope, budget_epochs)
        by_epochs[result.epochs] = result
    return [by_epochs[epochs] for epochs in sorted(by_epochs)]


def plot_results(results, eval_scope, output_path):
    epochs = [result.epochs for result in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    expected_horizons = []
    horizon_std = []
    for result in results:
        seed_horizons = np.array(list(result.seed_train_Ts.values()))
        expected_horizons.append(np.mean(seed_horizons))
        horizon_std.append(np.std(seed_horizons))
    ax.errorbar(
        epochs,
        expected_horizons,
        yerr=horizon_std,
        fmt="none",
        ecolor=COLOR_TRAIN,
        elinewidth=1.5,
        capsize=4,
        zorder=3,
    )
    ax.scatter(
        epochs,
        expected_horizons,
        color=COLOR_TRAIN,
        label="Expected horizon",
        zorder=4,
    )
    if len(epochs) >= 2:
        fit_coefficients = np.polyfit(np.log2(epochs), expected_horizons, deg=1)
        fit_epochs = np.linspace(min(epochs), max(epochs), 100)
        fit_horizons = np.polyval(fit_coefficients, np.log2(fit_epochs))
        ax.plot(
            fit_epochs,
            fit_horizons,
            linestyle="--",
            linewidth=2,
            color=COLOR_TRAIN,
            alpha=0.8,
            label="Linear fit",
            zorder=3,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Training budget (epochs per horizon)")
    ax.set_ylabel("Optimal training horizon T")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(epochs)
    scope_label = (
        "overall"
        if eval_scope.mode == "overall"
        else f"single eval T={eval_scope.eval_T}"
    )
    ax.set_title(f"Optimal Horizon ({scope_label})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_csv(results, output_path):
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epochs_per_horizon", "optimal_train_T", "mse", "source"])
        for result in results:
            writer.writerow(
                [result.epochs, result.train_T, result.mse, result.source_path]
            )
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot the optimal fixed training horizon against epoch budget."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=config.EVAL_DIR / "budget_epochs",
        help="Directory containing fixed-model cross-validation JSON files",
    )
    parser.add_argument("--metric", choices=("mean", "median"), default="median")
    parser.add_argument(
        "--scope",
        nargs="+",
        default=["overall"],
        metavar=("MODE", "T"),
        help="Use 'overall' or 'single <validation horizon>'",
    )
    parser.add_argument(
        "--budget-epochs",
        type=int,
        default=20,
        help="Epochs per horizon for results stored under a budget_based directory",
    )
    args = parser.parse_args()

    try:
        eval_scope = parse_eval_scope(args.scope)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    results = load_results(
        args.results_dir, args.metric, eval_scope, args.budget_epochs
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.results_dir / f"optimal_horizon_{timestamp}.png"
    plot_results(results, eval_scope, output_path)
    csv_path = save_csv(results, output_path)
    print(f"Saved optimal horizon plot to {output_path}")
    print(f"Saved plotted values to {csv_path}")


if __name__ == "__main__":
    main()
