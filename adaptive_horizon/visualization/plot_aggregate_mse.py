import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from adaptive_horizon.visualization.plot_budget_resources import (
    EvalScope,
    parse_eval_scope,
    summary_for_eval_T,
)
from adaptive_horizon.visualization.plotting import COLOR_EVAL, COLOR_TRAIN


@dataclass(frozen=True)
class MethodResult:
    method: str
    directory: Path
    source_path: Path
    payload: dict


def load_payload(path: Path):
    with path.open("r") as f:
        return json.load(f)


def select_max_budget_result(result_paths):
    payloads = [(path, load_payload(path)) for path in result_paths]
    return max(
        payloads,
        key=lambda item: (
            int(item[1]["metadata"]["max_train_T"]),
            item[0].name,
        ),
    )


def method_name(directory: Path):
    match = re.match(r"^budget_based_dt_[^_]+(?:_(.*))?$", directory.name)
    if match and match.group(1):
        return match.group(1)
    return directory.name


def load_method_results(results_dir: Path):
    paths_by_directory = {}
    for path in results_dir.rglob("budget_mse_results_*.json"):
        paths_by_directory.setdefault(path.parent, []).append(path)

    if not paths_by_directory:
        raise FileNotFoundError(
            f"No budget_mse_results_*.json files found below {results_dir}"
        )

    results = []
    for directory, paths in sorted(paths_by_directory.items()):
        source_path, payload = select_max_budget_result(paths)
        results.append(
            MethodResult(
                method=method_name(directory),
                directory=directory,
                source_path=source_path,
                payload=payload,
            )
        )
    return results


def median_for_scope(summary, eval_scope: EvalScope):
    if eval_scope.mode == "overall":
        return float(summary["overall"]["median"])
    return float(
        summary_for_eval_T(summary["by_eval_T"], eval_scope.eval_T)["median"]
    )


def adaptive_median(result: MethodResult, eval_scope: EvalScope):
    summary = result.payload.get("summary", {}).get("adaptive")
    if summary is None:
        return None
    if eval_scope.mode == "overall":
        metadata_value = result.payload.get("metadata", {}).get(
            "adaptive_median_MSE"
        )
        if metadata_value is not None:
            return float(metadata_value)
    return median_for_scope(summary, eval_scope)


def best_fixed_median(result: MethodResult, eval_scope: EvalScope):
    summary = result.payload.get("summary", {})
    fixed_summaries = summary.get("fixed", [])
    if not fixed_summaries:
        return None

    if eval_scope.mode == "overall":
        metadata_value = result.payload.get("metadata", {}).get(
            "best_fixed_median_MSE"
        )
        if metadata_value is not None:
            return float(metadata_value)

    return min(
        median_for_scope(fixed_summary, eval_scope)
        for fixed_summary in fixed_summaries
    )


def unique_method_labels(results):
    counts = {}
    for result in results:
        counts[result.method] = counts.get(result.method, 0) + 1

    labels = []
    for result in results:
        label = result.method
        if counts[label] > 1:
            dt = result.payload.get("metadata", {}).get("dt")
            label = f"{label} (dt={dt:g})" if dt is not None else result.directory.name
        labels.append(label)
    return labels


def plot_aggregate(results, eval_scope: EvalScope, output_path: Path):
    adaptive_results = []
    fixed_values = []
    for result in results:
        adaptive_value = adaptive_median(result, eval_scope)
        fixed_value = best_fixed_median(result, eval_scope)
        if adaptive_value is not None:
            adaptive_results.append((result, adaptive_value))
        if fixed_value is not None:
            fixed_values.append((result, fixed_value))

    if not fixed_values:
        raise ValueError("No fixed-model MSE values found in the selected results")

    fixed_only_results = [
        result
        for result in results
        if result.payload.get("summary", {}).get("adaptive") is None
    ]
    baseline_candidates = [
        (result, value)
        for result, value in fixed_values
        if result in fixed_only_results
    ] or fixed_values
    baseline = baseline_candidates[0][1]
    if not all(
        np.isclose(baseline, value) for _, value in baseline_candidates[1:]
    ):
        sources = ", ".join(
            f"{result.directory.name}={value:g}"
            for result, value in baseline_candidates
        )
        raise ValueError(
            "Selected fixed-only result directories do not share one best "
            f"fixed MSE ({sources})."
        )

    labels = unique_method_labels([result for result, _ in adaptive_results])
    values = [value for _, value in adaptive_results]
    labels.append("fixed")
    values.append(baseline)
    colors = [COLOR_EVAL] * len(adaptive_results) + [COLOR_TRAIN]

    fig_width = max(7.0, 1.25 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, fmt="%.4g", padding=3, fontsize=9)
    ax.set_ylabel("Median MSE")
    ax.set_xlabel("Method")
    if eval_scope.mode == "single":
        ax.set_title(
            rf"Aggregate MSE by method at validation $T_{{val}}={eval_scope.eval_T}$"
        )
    else:
        ax.set_title("Aggregate MSE by method")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def default_output_path(results_dir: Path, eval_scope: EvalScope):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scope_part = (
        "overall"
        if eval_scope.mode == "overall"
        else f"T{eval_scope.eval_T}"
    )
    return results_dir / f"aggregate_mse_{scope_part}_{timestamp}.png"


def main():
    parser = argparse.ArgumentParser(
        description="Plot adaptive and shared best-fixed median MSE by method."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory whose subdirectories contain budget_mse_results_*.json files",
    )
    parser.add_argument(
        "--scope",
        nargs="+",
        default=["overall"],
        metavar=("MODE", "T"),
        help="Use 'overall' or 'single <validation horizon>'",
    )
    args = parser.parse_args()

    try:
        eval_scope = parse_eval_scope(args.scope)
        results = load_method_results(args.results_dir)
        output_path = default_output_path(args.results_dir, eval_scope)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_aggregate(results, eval_scope, output_path)
    except (argparse.ArgumentTypeError, FileNotFoundError, KeyError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Saved aggregate MSE plot to {output_path}")


if __name__ == "__main__":
    main()
