import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from adaptive_horizon import config
from adaptive_horizon.utils import format_dt
from adaptive_horizon.visualization.plotting import COLOR_EVAL, COLOR_TRAIN


@dataclass(frozen=True)
class BudgetComparison:
    max_train_T: int
    adaptive_mse: float
    best_fixed_mse: float
    best_train_T: int | None
    source_path: Path


def load_result(path: Path, metric: str):
    with path.open("r") as f:
        result = json.load(f)

    metadata = result.get("metadata", {})
    summary = result.get("summary", {})
    max_train_T = int(metadata["max_train_T"])

    adaptive_summary = summary.get("adaptive", {}).get("overall", {})
    adaptive_mse = adaptive_summary.get(metric)
    if adaptive_mse is None:
        adaptive_mse = metadata.get(f"adaptive_{metric}_MSE")

    fixed_runs = summary.get("fixed", [])
    fixed_candidates = [
        (int(run["train_T"]), float(run["overall"][metric]))
        for run in fixed_runs
        if metric in run.get("overall", {})
    ]
    if fixed_candidates:
        best_train_T, best_fixed_mse = min(fixed_candidates, key=lambda item: item[1])
    else:
        best_train_T = metadata.get("best_train_T")
        best_fixed_mse = metadata.get(f"best_fixed_{metric}_MSE")

    if adaptive_mse is None or best_fixed_mse is None:
        raise ValueError(f"{path} does not contain {metric} MSE comparison values")

    return BudgetComparison(
        max_train_T=max_train_T,
        adaptive_mse=float(adaptive_mse),
        best_fixed_mse=float(best_fixed_mse),
        best_train_T=int(best_train_T),
        source_path=path,
    )


def load_comparisons(results_dir, metric):
    paths = sorted(Path(results_dir).glob("budget_mse_results_*.json"))
    if not paths:
        raise FileNotFoundError(f"No result files matched in {Path(results_dir)}")

    comparisons = [load_result(path, metric) for path in paths]
    return sorted(comparisons, key=lambda item: item.max_train_T)


def save_csv(comparisons, output_path: Path):
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "max_train_T",
                "adaptive_mse",
                "best_fixed_mse",
                "best_train_T",
                "source_file",
            ]
        )
        for comparison in comparisons:
            writer.writerow(
                [
                    comparison.max_train_T,
                    comparison.adaptive_mse,
                    comparison.best_fixed_mse,
                    comparison.best_train_T,
                    comparison.source_path,
                ]
            )
    return csv_path


def plot_comparisons(comparisons, metric, output_path):
    x_values = [comparison.max_train_T for comparison in comparisons]
    adaptive_values = [comparison.adaptive_mse for comparison in comparisons]
    fixed_values = [comparison.best_fixed_mse for comparison in comparisons]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        x_values,
        adaptive_values,
        marker="o",
        linewidth=2,
        label="Adaptive",
        color=COLOR_EVAL,
    )
    ax.plot(
        x_values,
        fixed_values,
        marker="s",
        linewidth=2,
        label="Best fixed",
        color=COLOR_TRAIN,
    )

    ax.set_xlabel("Training budget (max train T)")
    ax.set_xticks(x_values)
    ax.set_ylabel(f"Overall {metric} MSE")
    ax.set_title("Budget-Based Compute Comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def default_output_path(results_dir, metric, comparisons):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dt_values = {
        json.loads(comparison.source_path.read_text()).get("metadata", {}).get("dt")
        for comparison in comparisons
    }
    dt_part = ""
    if len(dt_values) == 1:
        dt = next(iter(dt_values))
        if dt is not None:
            dt_part = f"_dt_{format_dt(dt)}"
    return (
        Path(results_dir) / f"budget_based_comparison{dt_part}_{metric}_{timestamp}.png"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot adaptive and best fixed MSE against budget-based training resources."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=config.EVAL_DIR,
        help="Directory containing budget_mse_results_*.json files",
    )
    parser.add_argument(
        "--metric",
        choices=("mean", "median"),
        default="median",
        help="Overall MSE summary statistic to compare",
    )
    parser.add_argument(
        "--epochs-per-T",
        type=int,
        default=20,
        help="Epochs per horizon used to compute cumulative_epochs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. A CSV with the same stem is also written.",
    )
    args = parser.parse_args()

    comparisons = load_comparisons(args.results_dir, args.metric)
    output_path = args.output or default_output_path(
        args.results_dir, args.metric, comparisons
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_comparisons(
        comparisons,
        args.metric,
        output_path,
    )
    csv_path = save_csv(comparisons, output_path)

    print(f"Saved compute budget comparison plot to {output_path}")
    print(f"Saved plotted values to {csv_path}")


if __name__ == "__main__":
    main()
