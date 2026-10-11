import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adaptive_horizon import config
from adaptive_horizon.evaluation.results import (
    EvalScope,
    add_scope_argument,
    load_payload,
    metric_for_scope,
    parse_eval_scope,
    per_seed_mses,
    select_fixed_result,
)
from adaptive_horizon.utils import format_dt, timestamp
from adaptive_horizon.visualization.plotting import COLOR_EVAL, COLOR_TRAIN, save_figure


@dataclass(frozen=True)
class BudgetComparison:
    max_train_T: int
    adaptive_mse: float
    adaptive_seed_mses: list[float]
    fixed_mse: float
    fixed_seed_mses: list[float]
    best_train_T: int | None
    source_path: Path


def load_result(path: Path, metric: str, eval_scope: EvalScope):
    result = load_payload(path)
    metadata = result.get("metadata", {})
    summary = result.get("summary", {})
    records = result.get("evaluation_records", [])
    fixed_result = select_fixed_result(summary, metric, eval_scope)
    best_train_T = fixed_result["train_T"]

    adaptive_seed_mses = per_seed_mses(records, "adaptive", metric, eval_scope)
    fixed_seed_mses = per_seed_mses(records, "fixed", metric, eval_scope, best_train_T)

    return BudgetComparison(
        max_train_T=metadata["max_train_T"],
        adaptive_mse=metric_for_scope(summary["adaptive"], metric, eval_scope),
        adaptive_seed_mses=adaptive_seed_mses,
        fixed_mse=metric_for_scope(fixed_result, metric, eval_scope),
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
                "fixed_mse",
                "best_train_T",
            ]
        )
        for comparison in comparisons:
            writer.writerow(
                [
                    comparison.max_train_T,
                    comparison.adaptive_mse,
                    comparison.fixed_mse,
                    comparison.best_train_T,
                ]
            )
    return csv_path


def plot_comparisons(comparisons, metric, eval_scope, output_path):
    x_values = [comparison.max_train_T for comparison in comparisons]
    adaptive_values = [comparison.adaptive_mse for comparison in comparisons]
    fixed_values = [comparison.fixed_mse for comparison in comparisons]
    x_spacing = min(np.diff(sorted(set(x_values)))) if len(x_values) > 1 else 1.0
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
    if eval_scope.mode == "single":
        ax.set_ylabel(rf"{metric.title()} MSE at $T_{{val}}={eval_scope.eval_T}$")
        ax.set_title(rf"Budget-Based Comparison at $T_{{val}}={eval_scope.eval_T}$")
    else:
        ax.set_ylabel(f"Overall {metric} MSE")
        ax.set_title("Budget-Based Comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    save_figure(
        fig, output_path.name, output_path.parent, bbox_inches="tight", verbose=False
    )


def default_output_path(results_dir, metric, eval_scope, comparisons):
    stamp = timestamp()
    dt_values = {
        json.loads(comparison.source_path.read_text()).get("metadata", {}).get("dt")
        for comparison in comparisons
    }
    dt_part = ""
    if len(dt_values) == 1:
        dt = next(iter(dt_values))
        if dt is not None:
            dt_part = f"_dt_{format_dt(dt)}"
    scope_part = f"_T{eval_scope.eval_T}" if eval_scope.mode == "single" else ""
    return Path(results_dir) / (
        f"budget_based_comparison{dt_part}_{metric}{scope_part}_{stamp}.png"
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
    add_scope_argument(parser)
    args = parser.parse_args()
    try:
        eval_scope = parse_eval_scope(args.scope)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    comparisons = load_comparisons(args.results_dir, args.metric, eval_scope)
    output_path = default_output_path(
        args.results_dir, args.metric, eval_scope, comparisons
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_comparisons(
        comparisons,
        args.metric,
        eval_scope,
        output_path,
    )
    csv_path = save_csv(comparisons, output_path)

    print(f"Saved compute budget comparison plot to {output_path}")
    print(f"Saved plotted values to {csv_path}")


if __name__ == "__main__":
    main()
