import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from adaptive_horizon import config
from adaptive_horizon.utils import format_dt
from adaptive_horizon.visualization.plotting import COLOR_EVAL, COLOR_TRAIN


@dataclass(frozen=True)
class BudgetComparison:
    max_train_T: int
    adaptive_mse: float
    adaptive_seed_mses: list[float]
    fixed_mse: float
    fixed_seed_mses: list[float]
    best_train_T: int | None
    source_path: Path


def per_seed_mses(records, model_type: str, eval_scope: str, train_T=None):
    matching_records = [
        record
        for record in records
        if record["model_type"] == model_type
        and record.get("seed") is not None
        and (train_T is None or record.get("train_T") == train_T)
        and (eval_scope != "T1" or int(record["val_T"]) == 1)
    ]

    if eval_scope == "T1":
        return [float(record["mse"]) for record in matching_records]

    values_by_seed = {}
    for record in matching_records:
        values_by_seed.setdefault(int(record["seed"]), []).append(float(record["mse"]))

    return [
        float(np.mean(seed_values))
        for _, seed_values in sorted(values_by_seed.items())
        if seed_values
    ]


def aggregate_mse(seed_mses, metric: str, fallback: float):
    if not seed_mses:
        return fallback
    if metric == "mean":
        return float(np.mean(seed_mses))
    return float(np.median(seed_mses))


def load_result(path: Path, metric: str, eval_scope: str):
    with path.open("r") as f:
        result = json.load(f)
    metadata = result.get("metadata", {})
    summary = result.get("summary", {})
    records = result.get("evaluation_records", [])
    best_train_T = metadata.get("best_train_T")
    fixed_result = next(
        item for item in summary["fixed"] if item["train_T"] == best_train_T
    )

    if eval_scope == "T1":
        fixed_summary = fixed_result["by_eval_T"][0]
        adaptive_summary = summary["adaptive"]["by_eval_T"][0]
    else:
        fixed_summary = fixed_result["overall"]
        adaptive_summary = summary["adaptive"]["overall"]

    adaptive_seed_mses = per_seed_mses(records, "adaptive", eval_scope)
    fixed_seed_mses = per_seed_mses(records, "fixed", eval_scope, best_train_T)

    return BudgetComparison(
        max_train_T=metadata["max_train_T"],
        adaptive_mse=aggregate_mse(
            adaptive_seed_mses, metric, adaptive_summary[metric]
        ),
        adaptive_seed_mses=adaptive_seed_mses,
        fixed_mse=aggregate_mse(fixed_seed_mses, metric, fixed_summary[metric]),
        fixed_seed_mses=fixed_seed_mses,
        best_train_T=best_train_T,
        source_path=path,
    )


def load_comparisons(results_dir, metric, eval_scope):
    paths = sorted(Path(results_dir).glob("budget_mse_results_*.json"))
    if not paths:
        raise FileNotFoundError(f"No result files matched in {Path(results_dir)}")

    comparisons = [load_result(path, metric, eval_scope) for path in paths]
    return sorted(comparisons, key=lambda item: item.max_train_T)


def save_csv(comparisons, output_path: Path):
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "max_train_T",
                "adaptive_mse",
                "adaptive_seed_mses",
                "fixed_mse",
                "fixed_seed_mses",
                "best_train_T",
                "source_file",
            ]
        )
        for comparison in comparisons:
            writer.writerow(
                [
                    comparison.max_train_T,
                    comparison.adaptive_mse,
                    json.dumps(comparison.adaptive_seed_mses),
                    comparison.fixed_mse,
                    json.dumps(comparison.fixed_seed_mses),
                    comparison.best_train_T,
                    comparison.source_path,
                ]
            )
    return csv_path


def plot_comparisons(comparisons, metric, eval_scope, output_path):
    x_values = [comparison.max_train_T for comparison in comparisons]
    adaptive_values = [comparison.adaptive_mse for comparison in comparisons]
    fixed_values = [comparison.fixed_mse for comparison in comparisons]
    x_spacing = min(np.diff(sorted(x_values))) if len(x_values) > 1 else 1.0
    group_offset = 0.12 * float(x_spacing)
    jitter_width = 0.04 * float(x_spacing)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, comparison in enumerate(comparisons):
        if comparison.adaptive_seed_mses:
            offsets = np.linspace(
                -jitter_width, jitter_width, len(comparison.adaptive_seed_mses)
            )
            ax.scatter(
                np.full(len(comparison.adaptive_seed_mses), x_values[i] + group_offset)
                + offsets,
                comparison.adaptive_seed_mses,
                color=COLOR_EVAL,
                alpha=0.8,
                s=22,
                linewidths=0,
                label="Adaptive seeds" if i == 0 else "_nolegend_",
                zorder=2,
            )
        if comparison.fixed_seed_mses:
            offsets = np.linspace(
                -jitter_width, jitter_width, len(comparison.fixed_seed_mses)
            )
            ax.scatter(
                np.full(len(comparison.fixed_seed_mses), x_values[i] - group_offset)
                + offsets,
                comparison.fixed_seed_mses,
                color=COLOR_TRAIN,
                alpha=0.8,
                s=22,
                linewidths=0,
                label="Best fixed seeds" if i == 0 else "_nolegend_",
                zorder=2,
            )

    ax.plot(
        x_values,
        adaptive_values,
        linewidth=2,
        label=f"Adaptive {metric}",
        color=COLOR_EVAL,
        zorder=3,
    )
    ax.plot(
        x_values,
        fixed_values,
        linewidth=2,
        label=f"Best fixed {metric}",
        color=COLOR_TRAIN,
        zorder=3,
    )

    ax.set_xlabel("Training budget (max train T)")
    ax.set_xticks(x_values)
    if eval_scope == "T1":
        ax.set_ylabel(rf"{metric} MSE at $T_{{val}}=1$")
        ax.set_title(r"Budget-Based Comparison at $T_{{val}}=1$")
    else:
        ax.set_ylabel(f"Overall {metric} MSE")
        ax.set_title("Budget-Based Comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def default_output_path(results_dir, metric, eval_scope, comparisons):
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
    scope_part = "_T1" if eval_scope == "T1" else ""
    return Path(results_dir) / (
        f"budget_based_comparison{dt_part}_{metric}{scope_part}_{timestamp}.png"
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
        type=str,
        choices=("mean", "median"),
        default="median",
        help="Overall MSE summary statistic to compare",
    )
    parser.add_argument(
        "--eval-scope",
        type=str,
        choices=("overall", "T1"),
        default="overall",
        help="Use all evaluation results or only those for T=1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. A CSV with the same stem is also written.",
    )
    args = parser.parse_args()

    comparisons = load_comparisons(args.results_dir, args.metric, args.eval_scope)
    output_path = args.output or default_output_path(
        args.results_dir, args.metric, args.eval_scope, comparisons
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_comparisons(
        comparisons,
        args.metric,
        args.eval_scope,
        output_path,
    )
    csv_path = save_csv(comparisons, output_path)

    print(f"Saved compute budget comparison plot to {output_path}")
    print(f"Saved plotted values to {csv_path}")


if __name__ == "__main__":
    main()
