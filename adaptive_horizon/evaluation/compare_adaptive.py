"""Compare two adaptive methods evaluated against the same fixed models.

Used for the FTLE ablation: LM (FTLE-assigned horizons) vs. LM-random (the same
horizon distribution, randomly permuted across samples) vs. the best fixed T.
Both result files must come from cross_validation with the same fixed directory,
so the fixed records and validation horizons are identical; adaptive models are
paired by seed.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

from adaptive_horizon.evaluation.metrics import ERROR_METRICS
from adaptive_horizon.evaluation.utils import calculate_stats
from adaptive_horizon.utils import timestamp


def load_results(path: Path):
    with open(path) as f:
        return json.load(f)


def records_by_key(payload, model_type, error_metric):
    return {
        (record["seed"], record["train_T"], record["val_T"]): record[error_metric]
        for record in payload["evaluation_records"]
        if record["model_type"] == model_type
    }


def sign_test_p_value(wins: int, losses: int):
    """Two-sided exact sign test, ties excluded."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_comparison(candidate, baseline):
    """Candidate wins where its error is lower, per (seed, val_T) and per seed."""
    keys = sorted(candidate.keys() & baseline.keys())
    wins = sum(candidate[key] < baseline[key] for key in keys)

    seeds = sorted({key[0] for key in keys})
    seed_wins = seed_losses = 0
    log_ratios = []
    for seed in seeds:
        seed_keys = [key for key in keys if key[0] == seed]
        candidate_median = median(candidate[key] for key in seed_keys)
        baseline_median = median(baseline[key] for key in seed_keys)
        seed_wins += candidate_median < baseline_median
        seed_losses += candidate_median > baseline_median
        log_ratios.append(math.log(candidate_median / baseline_median))

    return {
        "n_pairs": len(keys),
        "candidate_win_rate": wins / len(keys) if keys else None,
        "n_seeds": len(seeds),
        "seed_wins": seed_wins,
        "seed_losses": seed_losses,
        "seed_sign_test_p": sign_test_p_value(seed_wins, seed_losses),
        "median_seed_error_ratio": math.exp(median(log_ratios)) if log_ratios else None,
    }


def median_by_val_T(records, val_Ts):
    return [
        median(value for key, value in records.items() if key[2] == val_T)
        for val_T in val_Ts
    ]


def compare_adaptive(
    baseline_results: Path,
    candidate_results: Path,
    output_dir: Path,
    baseline_label="LM",
    candidate_label="LM-random",
    error_metric="mse",
):
    baseline_payload = load_results(baseline_results)
    candidate_payload = load_results(candidate_results)
    for key in ("fixed_dir", "dt", "system"):
        if baseline_payload["metadata"].get(key) != candidate_payload["metadata"].get(
            key
        ):
            raise ValueError(
                f"Result files differ in {key}: "
                f"{baseline_payload['metadata'].get(key)} vs "
                f"{candidate_payload['metadata'].get(key)}"
            )

    fixed = records_by_key(baseline_payload, "fixed", error_metric)
    baseline = records_by_key(baseline_payload, "adaptive", error_metric)
    candidate = records_by_key(candidate_payload, "adaptive", error_metric)
    if not baseline or not candidate:
        raise ValueError("Both result files need adaptive evaluation records")
    if fixed != records_by_key(candidate_payload, "fixed", error_metric):
        raise ValueError("Fixed-model records differ between the result files")

    val_Ts = sorted({key[2] for key in baseline})
    train_Ts = sorted({key[1] for key in fixed})
    fixed_by_T = {
        train_T: {
            (seed, None, val_T): value
            for (seed, T, val_T), value in fixed.items()
            if T == train_T
        }
        for train_T in train_Ts
    }
    best_T = min(train_Ts, key=lambda T: median(fixed_by_T[T].values()))
    best_fixed = fixed_by_T[best_T]

    summary = {
        "metadata": {
            "system": baseline_payload["metadata"]["system"],
            "dt": baseline_payload["metadata"]["dt"],
            "fixed_dir": baseline_payload["metadata"]["fixed_dir"],
            "baseline_results": str(baseline_results),
            "candidate_results": str(candidate_results),
            "baseline_label": baseline_label,
            "candidate_label": candidate_label,
            "error_metric": error_metric,
            "best_fixed_train_T": best_T,
        },
        "overall": {
            f"fixed_T{best_T}": calculate_stats(best_fixed.values()),
            baseline_label: calculate_stats(baseline.values()),
            candidate_label: calculate_stats(candidate.values()),
        },
        "by_val_T": {
            "val_T": val_Ts,
            f"fixed_T{best_T}": median_by_val_T(best_fixed, val_Ts),
            baseline_label: median_by_val_T(baseline, val_Ts),
            candidate_label: median_by_val_T(candidate, val_Ts),
        },
        "paired": {
            f"{candidate_label}_vs_{baseline_label}": paired_comparison(
                candidate, baseline
            ),
            f"{baseline_label}_vs_fixed_T{best_T}": paired_comparison(
                baseline, best_fixed
            ),
            f"{candidate_label}_vs_fixed_T{best_T}": paired_comparison(
                candidate, best_fixed
            ),
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp()
    results_file = output_dir / f"adaptive_comparison_{error_metric}_{stamp}.json"
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label in (f"fixed_T{best_T}", baseline_label, candidate_label):
        ax.plot(val_Ts, summary["by_val_T"][label], marker="o", label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Validation horizon (steps)")
    ax.set_ylabel(f"Median {error_metric.upper()}")
    ax.set_title(
        f"{summary['metadata']['system']} dt={summary['metadata']['dt']}: "
        f"{candidate_label} vs {baseline_label}"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plot_file = output_dir / f"adaptive_comparison_{error_metric}_{stamp}.png"
    fig.savefig(plot_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    for name, stats in summary["overall"].items():
        print(f"{name:>14}: median {error_metric} = {stats['median']:.6f}")
    for name, stats in summary["paired"].items():
        print(
            f"{name}: win rate {stats['candidate_win_rate']:.3f}, "
            f"seeds {stats['seed_wins']}/{stats['n_seeds']} "
            f"(sign test p={stats['seed_sign_test_p']:.3g}), "
            f"median error ratio {stats['median_seed_error_ratio']:.3f}"
        )
    print(f"\nComparison saved to {results_file}")
    return results_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", default="LM")
    parser.add_argument("--candidate-label", default="LM-random")
    parser.add_argument("--error-metric", choices=ERROR_METRICS, default="mse")
    args = parser.parse_args()
    compare_adaptive(**vars(args))


if __name__ == "__main__":
    main()
