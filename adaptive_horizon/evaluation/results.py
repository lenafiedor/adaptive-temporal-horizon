"""Loading and selecting from saved cross-validation / budget result files."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EvalScope:
    mode: str
    eval_T: int | None = None


def parse_eval_scope(scope_args):
    if scope_args == ["overall"]:
        return EvalScope("overall")
    if len(scope_args) == 2 and scope_args[0] == "single":
        try:
            eval_T = int(scope_args[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--scope single requires an integer validation horizon"
            ) from exc
        return EvalScope("single", eval_T)
    raise argparse.ArgumentTypeError(
        "--scope must be either 'overall' or 'single <validation horizon>'"
    )


def add_scope_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--scope",
        nargs="+",
        default=["overall"],
        metavar=("MODE", "T"),
        help="Use 'overall' or 'single <validation horizon>'",
    )


def load_payload(path: Path):
    with Path(path).open("r") as f:
        return json.load(f)


def aggregate_values(values, metric: str):
    if metric == "mean":
        return float(np.mean(values))
    return float(np.median(values))


def matching_records(records, model_type, eval_scope, train_T=None, seed=None):
    """Evaluation records of one model type restricted to the eval scope."""
    return [
        record
        for record in records
        if record["model_type"] == model_type
        and (train_T is None or record.get("train_T") == train_T)
        and (
            seed is None
            or (record.get("seed") is not None and int(record["seed"]) == seed)
        )
        and (eval_scope.mode != "single" or int(record["val_T"]) == eval_scope.eval_T)
    ]


def per_seed_mses(
    records, model_type: str, metric: str, eval_scope: EvalScope, train_T=None
):
    values_by_seed = {}
    for record in matching_records(records, model_type, eval_scope, train_T):
        if record.get("seed") is None:
            continue
        values_by_seed.setdefault(int(record["seed"]), []).append(float(record["mse"]))

    return [
        aggregate_values(seed_values, metric)
        for _, seed_values in sorted(values_by_seed.items())
        if seed_values
    ]


def summary_for_eval_T(summaries, eval_T: int):
    for summary in summaries:
        if int(summary["eval_T"]) == eval_T:
            return summary
    available = ", ".join(str(summary["eval_T"]) for summary in summaries)
    raise ValueError(f"Validation horizon T={eval_T} not found. Available: {available}")


def scope_summary(summary, eval_scope: EvalScope):
    """The stats dict (mean/median/CI) of a model summary for the eval scope."""
    if eval_scope.mode == "overall":
        return summary["overall"]
    if eval_scope.eval_T is None:
        raise ValueError("Single-horizon scope requires an evaluation horizon")
    return summary_for_eval_T(summary["by_eval_T"], eval_scope.eval_T)


def metric_for_scope(summary, metric: str, eval_scope: EvalScope):
    return float(scope_summary(summary, eval_scope)[metric])


def select_fixed_result(summary, metric: str, eval_scope: EvalScope):
    """The fixed-horizon summary with the lowest metric for the eval scope."""
    return min(
        summary["fixed"],
        key=lambda item: metric_for_scope(item, metric, eval_scope),
    )
