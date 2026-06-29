import argparse
import json
import re
import shutil
from pathlib import Path

import adaptive_horizon.config as config
from adaptive_horizon.dynamics.systems import SYSTEM_CHOICES
from adaptive_horizon.evaluation.cross_validation import (
    cross_validate_models,
    cross_validation,
    get_adaptive_paths,
    get_fixed_paths,
)
from adaptive_horizon.evaluation.utils import (
    get_dt_from_model_dir,
    save_cross_validation_results,
    summarize_cross_validation,
)
from adaptive_horizon.visualization.plotting import (
    plot_mse,
    plot_mse_subplots,
    plot_paired_deltas,
)


def get_catalog_runs(catalog_dir: Path):
    runs = []
    for child in sorted(catalog_dir.iterdir()):
        if child.is_dir() and (child / "adaptive").is_dir():
            match = re.search(r"_T(\d+)$", child.name)
            if match:
                runs.append((int(match.group(1)), child))
    return sorted(runs)


def infer_fixed_dir(catalog_dir: Path):
    direct = catalog_dir.parent / f"{catalog_dir.name}_fixed"
    if direct.is_dir():
        return direct

    match = re.match(r"(.+dt_\d+)(?:_.+)?$", catalog_dir.name)
    if match:
        sibling = catalog_dir.parent / f"{match.group(1)}_fixed"
        if sibling.is_dir():
            return sibling

    raise FileNotFoundError(
        "Could not infer fixed model directory. Pass --fixed-dir explicitly."
    )


def inferred_cache_dirs(catalog_dir: Path, output_dir: Path):
    cache_dirs = [output_dir]
    parts = list(catalog_dir.parts)
    if "models" in parts:
        parts[parts.index("models")] = "evaluation"
        evaluation_dir = Path(*parts)
        if evaluation_dir not in cache_dirs:
            cache_dirs.append(evaluation_dir)
        evaluation_root = evaluation_dir.parent
        if evaluation_root.is_dir():
            for child in sorted(evaluation_root.iterdir()):
                if child.is_dir() and child not in cache_dirs:
                    cache_dirs.append(child)
    return cache_dirs


def fixed_model_seeds(fixed_dir: Path, max_train_T: int):
    seeds_by_T = {T: set() for T in range(1, max_train_T + 1)}
    for path in fixed_dir.glob("mlp_T*.pt"):
        match = re.search(r"mlp_T(\d+)_seed(\d+)_", path.name)
        if match:
            train_T = int(match.group(1))
            if train_T in seeds_by_T:
                seeds_by_T[train_T].add(int(match.group(2)))
    return {train_T: seeds for train_T, seeds in seeds_by_T.items() if seeds}


def adaptive_model_seeds(adaptive_dir: Path):
    seeds = set()
    for path in adaptive_dir.glob("adaptive_mlp*.pt"):
        match = re.search(r"_seed(\d+)_", path.name)
        if match:
            seeds.add(int(match.group(1)))
    return seeds


def cached_fixed_seeds(records, max_train_T: int):
    seeds_by_T = {T: set() for T in range(1, max_train_T + 1)}
    for record in records:
        if record["model_type"] != "fixed":
            continue
        train_T = int(record["train_T"])
        if train_T in seeds_by_T and record.get("seed") is not None:
            seeds_by_T[train_T].add(int(record["seed"]))
    return {train_T: seeds for train_T, seeds in seeds_by_T.items() if seeds}


def cached_adaptive_seeds(records):
    return {
        int(record["seed"])
        for record in records
        if record["model_type"] == "adaptive" and record.get("seed") is not None
    }


def same_path(left, right):
    if left is None:
        return False
    return Path(left).resolve() == Path(right).resolve()


def cache_matches(
    cache_path, model_dir, fixed_dir, max_train_T, max_eval_T, system_name
):
    with cache_path.open("r") as f:
        payload = json.load(f)

    metadata = payload.get("metadata", {})
    records = payload.get("evaluation_records", [])
    if metadata.get("system", config.DEFAULT_SYSTEM) != system_name:
        return False
    if int(metadata.get("max_train_T", -1)) != int(max_train_T):
        return False
    if not same_path(metadata.get("adaptive_dir"), model_dir / "adaptive"):
        return False
    if not same_path(metadata.get("fixed_dir"), fixed_dir):
        return False

    val_Ts = {int(record["val_T"]) for record in records}
    if not set(range(1, max_eval_T + 1)).issubset(val_Ts):
        return False

    adaptive_dir = model_dir / "adaptive"
    return cached_fixed_seeds(records, max_train_T) == fixed_model_seeds(
        fixed_dir, max_train_T
    ) and cached_adaptive_seeds(records) == adaptive_model_seeds(adaptive_dir)


def find_cached_result(
    cache_dirs, model_dir, fixed_dir, max_train_T, max_eval_T, system_name
):
    candidates = []
    for cache_dir in cache_dirs:
        candidates.extend(
            sorted(Path(cache_dir).glob(f"budget_mse_results_*_T{max_train_T}_*.json"))
        )

    for cache_path in sorted(
        candidates, key=lambda path: path.stat().st_mtime, reverse=True
    ):
        if cache_matches(
            cache_path, model_dir, fixed_dir, max_train_T, max_eval_T, system_name
        ):
            return cache_path
    return None


def copy_cached_result(cache_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / cache_path.name
    if cache_path.resolve() != target.resolve() and not target.exists():
        shutil.copy2(cache_path, target)


def model_seed(path: Path):
    match = re.search(r"_seed(\d+)_", path.name)
    if match:
        return int(match.group(1))
    return None


def reusable_cached_records(cache_dirs, model_dir, fixed_dir, max_train_T, max_eval_T):
    fixed_records = {}
    adaptive_records = {}
    val_Ts = set(range(1, max_eval_T + 1))
    current_fixed_seeds = fixed_model_seeds(fixed_dir, max_train_T)
    current_adaptive_seeds = adaptive_model_seeds(model_dir / "adaptive")

    candidates = []
    for cache_dir in cache_dirs:
        candidates.extend(Path(cache_dir).glob("budget_mse_results_*.json"))

    for cache_path in sorted(candidates, key=lambda path: path.stat().st_mtime):
        with cache_path.open("r") as f:
            payload = json.load(f)
        metadata = payload.get("metadata", {})
        records = payload.get("evaluation_records", [])

        use_fixed = same_path(metadata.get("fixed_dir"), fixed_dir)
        use_adaptive = same_path(metadata.get("adaptive_dir"), model_dir / "adaptive")
        if not use_fixed and not use_adaptive:
            continue

        for record in records:
            seed = record.get("seed")
            val_T = int(record["val_T"])
            if seed is None or val_T not in val_Ts:
                continue

            seed = int(seed)
            if use_fixed and record["model_type"] == "fixed":
                train_T = int(record["train_T"])
                if seed in current_fixed_seeds.get(train_T, set()):
                    fixed_records[("fixed", train_T, seed, val_T)] = record
            elif use_adaptive and record["model_type"] == "adaptive":
                if seed in current_adaptive_seeds:
                    adaptive_records[("adaptive", seed, val_T)] = record

    return fixed_records, adaptive_records


def missing_fixed_paths(fixed_paths, cached_records, max_eval_T):
    val_Ts = range(1, max_eval_T + 1)
    missing = {}
    for train_T, paths in fixed_paths.items():
        missing_paths = []
        for path in paths:
            seed = model_seed(path)
            if seed is None:
                missing_paths.append(path)
                continue
            if any(
                ("fixed", int(train_T), seed, val_T) not in cached_records
                for val_T in val_Ts
            ):
                missing_paths.append(path)
        missing[int(train_T)] = missing_paths
    return missing


def missing_adaptive_paths(adaptive_paths, cached_records, max_eval_T):
    val_Ts = range(1, max_eval_T + 1)
    missing = []
    for path in adaptive_paths:
        seed = model_seed(path)
        if seed is None:
            missing.append(path)
            continue
        if any(("adaptive", seed, val_T) not in cached_records for val_T in val_Ts):
            missing.append(path)
    return missing


def run_with_partial_cache(
    model_dir,
    fixed_dir,
    output_dir,
    cache_dirs,
    max_train_T,
    max_eval_T,
    metric,
    system_name,
):
    adaptive_dir = model_dir / "adaptive"
    train_Ts = list(range(1, max_train_T + 1))
    val_Ts = list(range(1, max_eval_T + 1))
    fixed_paths = get_fixed_paths(train_Ts, fixed_dir)
    adaptive_paths = get_adaptive_paths(adaptive_dir)
    fixed_cache, adaptive_cache = reusable_cached_records(
        cache_dirs, model_dir, fixed_dir, max_train_T, max_eval_T
    )
    fixed_missing = missing_fixed_paths(fixed_paths, fixed_cache, max_eval_T)
    adaptive_missing = missing_adaptive_paths(
        adaptive_paths, adaptive_cache, max_eval_T
    )
    cached_records = list(fixed_cache.values()) + list(adaptive_cache.values())

    missing_fixed_count = sum(len(paths) for paths in fixed_missing.values())
    print(
        f"Reusing {len(cached_records)} cached records; "
        f"validating {missing_fixed_count} fixed and {len(adaptive_missing)} adaptive models."
    )

    new_records = []
    if missing_fixed_count or adaptive_missing:
        new_records = cross_validate_models(
            fixed_missing,
            adaptive_missing,
            dt=get_dt_from_model_dir(model_dir),
            val_Ts=val_Ts,
            system_name=system_name,
        )

    evaluation_records = cached_records + new_records
    summary = summarize_cross_validation(evaluation_records, train_Ts, val_Ts)
    budget_based = model_dir.name.startswith("budget")
    dt = get_dt_from_model_dir(model_dir)
    save_cross_validation_results(
        evaluation_records,
        summary,
        max_train_T,
        dt,
        adaptive_dir,
        fixed_dir,
        output_dir,
        budget_based,
        system_name,
    )
    plot_mse(summary, output_dir, dt, max_train_T, budget_based, metric)
    plot_mse_subplots(
        evaluation_records,
        summary,
        output_dir,
        dt,
        max_train_T,
        budget_based,
        metric,
    )
    plot_paired_deltas(
        summary["deltas"],
        val_Ts,
        dt,
        output_dir,
        max_train_T,
        budget_based,
        metric,
    )


def run_catalog(
    catalog_dir,
    output_dir,
    fixed_dir=None,
    cache_dirs=None,
    max_eval_T=config.MAX_EVAL_T,
    metric="median",
    system_name=config.DEFAULT_SYSTEM,
):
    catalog_dir = Path(catalog_dir)
    output_dir = Path(output_dir)
    fixed_dir = (
        Path(fixed_dir) if fixed_dir is not None else infer_fixed_dir(catalog_dir)
    )
    cache_dirs = (
        [Path(cache_dir) for cache_dir in cache_dirs]
        if cache_dirs is not None
        else inferred_cache_dirs(catalog_dir, output_dir)
    )

    runs = get_catalog_runs(catalog_dir)
    if not runs:
        raise FileNotFoundError(f"No budget run directories found in {catalog_dir}")

    for max_train_T, model_dir in runs:
        cached = find_cached_result(
            cache_dirs, model_dir, fixed_dir, max_train_T, max_eval_T, system_name
        )
        if cached is not None:
            print(f"\nReusing cached cross-validation results from {cached}")
            copy_cached_result(cached, output_dir)
            cross_validation(
                output_dir=output_dir,
                max_train_T=max_train_T,
                max_eval_T=max_eval_T,
                cached=cached,
                metric=metric,
                system_name=system_name,
            )
            continue

        print(
            f"\nRunning cross-validation for {model_dir} with max_train_T={max_train_T}"
        )
        run_with_partial_cache(
            model_dir,
            fixed_dir=fixed_dir,
            output_dir=output_dir,
            cache_dirs=cache_dirs,
            max_train_T=max_train_T,
            max_eval_T=max_eval_T,
            metric=metric,
            system_name=system_name,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "catalog_dir",
        nargs="?",
        type=str,
        help="Directory containing budget_dt_*_T*/adaptive model directories",
    )
    parser.add_argument(
        "--catalog-dir",
        dest="catalog_dir_arg",
        type=str,
        default=None,
        help="Directory containing budget_dt_*_T*/adaptive model directories",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory for all cross-validation JSON and plots",
    )
    parser.add_argument(
        "--fixed-dir",
        type=str,
        default=None,
        help="Fixed model directory (default: inferred from catalog_dir)",
    )
    parser.add_argument(
        "--cache-dir",
        action="append",
        default=None,
        help="Directory to search for reusable cross-validation JSON files (can be repeated)",
    )
    parser.add_argument(
        "--max-eval-T",
        type=int,
        default=config.MAX_EVAL_T,
        help="Maximum validation horizon for cross-validation",
    )
    parser.add_argument(
        "--metric",
        choices=("mean", "median"),
        default="median",
        help="Statistic to plot with 95%% CI intervals",
    )
    parser.add_argument(
        "--system",
        choices=SYSTEM_CHOICES,
        default=config.DEFAULT_SYSTEM,
        help="Dynamical system to evaluate",
    )
    args = parser.parse_args()
    catalog_dir = args.catalog_dir_arg or args.catalog_dir
    if catalog_dir is None:
        parser.error("catalog_dir is required")

    run_catalog(
        catalog_dir,
        args.output_dir,
        fixed_dir=args.fixed_dir,
        cache_dirs=args.cache_dir,
        max_eval_T=args.max_eval_T,
        metric=args.metric,
        system_name=args.system,
    )


if __name__ == "__main__":
    main()
