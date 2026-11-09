import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from adaptive_horizon import config
from adaptive_horizon.data.dataset import make_val_loader
from adaptive_horizon.dynamics.systems import SYSTEM_CHOICES
from adaptive_horizon.evaluation.utils import (
    get_checkpoint_normalization_stats,
    load_model,
)
from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.training.loss import compute_g_T
from adaptive_horizon.training.methods import adaptive_method_abbreviation
from adaptive_horizon.training.utils import find_fixed_models
from adaptive_horizon.utils import timestamp

N_BOOT = 2000


def find_models(model_dir):
    models = [
        (path, f"T{train_T}", seed)
        for path, train_T, seed in find_fixed_models(model_dir)
    ]
    for path in sorted(Path(model_dir).rglob("adaptive_mlp*.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        adaptive = checkpoint.get("metadata", {}).get("adaptive", {})
        method = adaptive_method_abbreviation(
            adaptive.get("method"), adaptive.get("random_horizons", False)
        )
        models.append((path, (method or "adaptive").upper(), checkpoint.get("seed")))
    return models


def initial_model(checkpoint, seed):
    cfg = checkpoint["config"]
    mlp_config = MLPConfig(
        input_size=cfg["input_size"],
        output_size=cfg["output_size"],
        layer_widths=cfg["layer_widths"],
        residual_connections=cfg["residual_connections"],
        k=cfg.get("k"),
        activation=torch.nn.ReLU(),
    )
    model = MLP(mlp_config, random_seed=int(seed))
    model.eval()
    return model


def bootstrap_median_ci(values, rng, n_boot=N_BOOT):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values[0]), float(values[0])
    idx = rng.integers(0, len(values), (n_boot, len(values)))
    medians = np.median(values[idx], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def summarize(records, T_vals):
    rng = np.random.default_rng(0)
    summary = {}
    for group in sorted({r["group"] for r in records}, key=group_sort_key):
        rows = [r for r in records if r["group"] == group]
        summary[group] = {"n_models": len(rows), "T": T_vals, "median": [], "ci": []}
        for T in T_vals:
            values = [r["g"][str(T)] for r in rows]
            summary[group]["median"].append(float(np.median(values)))
            summary[group]["ci"].append(bootstrap_median_ci(values, rng))
    return summary


def group_sort_key(group):
    if group == "INIT":
        return 0, 0, group
    if group.startswith("T") and group[1:].isdigit():
        return 1, int(group[1:]), group
    return 2, 0, group


def aggregate_gradient_scaling(
    model_dir,
    max_T=config.MAX_EVAL_T,
    dt=None,
    system_name=None,
    include_init=False,
    groups=None,
    device=config.DEVICE,
):
    model_dir = Path(model_dir)
    models = find_models(model_dir)
    if groups:
        models = [m for m in models if m[1] in groups]
    if not models:
        raise FileNotFoundError(f"No checkpoints found below {model_dir}")

    T_vals = list(range(1, max_T + 1))
    loaders = {}
    records = []
    init_seeds_done = set()
    for idx, (path, group, seed) in enumerate(models, start=1):
        model, checkpoint = load_model(path)
        metadata = checkpoint.get("metadata", {})
        model_dt = float(metadata.get("dt", dt if dt is not None else config.DT))
        model_system = metadata.get("system") or system_name or config.DEFAULT_SYSTEM
        if dt is not None and not np.isclose(model_dt, dt):
            raise ValueError(f"{path}: dt {model_dt} differs from --dt {dt}")
        dt, system_name = model_dt, model_system

        stats = get_checkpoint_normalization_stats(checkpoint)
        key = None
        if stats is not None:
            key = tuple(
                tuple(float(v) for v in stats[name]) for name in ("mean", "std")
            )
        if key not in loaders:
            loaders[key] = make_val_loader(
                system_name, dt, max_T, normalization_stats=stats
            )
        loader = loaders[key]

        g = compute_g_T(model.to(device), loader, T_vals, device=device)
        records.append(
            {
                "path": str(path),
                "group": group,
                "seed": seed,
                "g": {str(T): float(v) for T, v in g.items()},
            }
        )
        print(f"[{idx}/{len(models)}] {group} seed {seed}: g({max_T}) = {g[max_T]:.3g}")

        if include_init and seed not in init_seeds_done:
            init_seeds_done.add(seed)
            init = initial_model(checkpoint, seed).to(device)
            init.normalization_stats = getattr(model, "normalization_stats", None)
            g0 = compute_g_T(init, loader, T_vals, device=device)
            records.append(
                {
                    "path": None,
                    "group": "INIT",
                    "seed": seed,
                    "g": {str(T): float(v) for T, v in g0.items()},
                }
            )

    return {
        "metadata": {
            "model_dir": str(model_dir),
            "system": system_name,
            "dt": dt,
            "max_T": max_T,
            "num_batches": config.NUM_BATCHES,
            "statistic": "median over seeds, 95% seed-bootstrap CI",
        },
        "records": records,
        "summary": summarize(records, T_vals),
    }


def plot_aggregate(payload, output_path):
    summary = payload["summary"]
    meta = payload["metadata"]
    fixed = [g for g in summary if g.startswith("T") and g[1:].isdigit()]
    other = [g for g in summary if g not in fixed]
    cmap = plt.get_cmap("viridis", max(len(fixed), 1))
    other_colors = ["#e34948", "#8a5cc7", "#2a9d6f", "#eb6834", "#0b0b0b"]

    fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
    styles = [(g, {"color": cmap(i), "linestyle": "-"}) for i, g in enumerate(fixed)]
    styles += [
        (g, {"color": other_colors[i % len(other_colors)], "linestyle": "--"})
        for i, g in enumerate(other)
    ]
    for group, style in styles:
        s = summary[group]
        lo, hi = np.array(s["ci"]).T
        label = f"Fixed, $T$ = {group[1:]}" if group in fixed else group.title()
        if group == "INIT":
            label = "Initialization"
        ax.plot(
            s["T"],
            s["median"],
            linewidth=1.4,
            marker="o",
            markersize=2.5,
            label=f"{label} (n={s['n_models']})",
            **style,
        )
        ax.fill_between(s["T"], lo, hi, color=style["color"], alpha=0.15, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("Horizon $T$ of the loss")
    ax.set_ylabel("$g(T)$, median over seeds\n(shaded: 95% bootstrap CI)")
    ax.set_title(f"{meta['system']}, $\\Delta t$ = {meta['dt']}", loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def run_gradient_scaling(
    model_dir,
    output_dir=None,
    max_T=config.MAX_EVAL_T,
    dt=None,
    system_name=None,
    include_init=False,
    groups=None,
):
    """Aggregate g(T) over all checkpoints below `model_dir`, save JSON and plot.

    If `output_dir` already holds a `gradient_scaling_<dir name>_*.json`, the
    latest one is reused and only its plot is redrawn; delete it to recompute.
    `output_dir` defaults to `<eval dir of the system>/gradient_scaling`, with the
    system taken from `system_name`, else from the model path, else the config.
    Returns the payload (metadata, records, summary).
    """
    model_dir = Path(model_dir)
    if output_dir is None:
        system = system_name or next(
            (p for p in model_dir.parts if p in SYSTEM_CHOICES), config.DEFAULT_SYSTEM
        )
        output_dir = config.system_path(config.EVAL_DIR, system) / "gradient_scaling"
    output_dir = Path(output_dir)
    stem_prefix = f"gradient_scaling_{model_dir.name}_"

    cached = sorted(output_dir.glob(f"{stem_prefix}*.json"))
    if cached:
        print(f"Using cached {cached[-1]}")
        payload = json.loads(cached[-1].read_text())
        plot_aggregate(payload, cached[-1].with_suffix(".png"))
        return payload

    payload = aggregate_gradient_scaling(
        model_dir,
        max_T=max_T,
        dt=dt,
        system_name=system_name,
        include_init=include_init,
        groups=groups,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{stem_prefix}{timestamp()}"
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=1))
    print(f"Results saved to {json_path}")
    plot_aggregate(payload, output_dir / f"{stem}.png")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Directory with checkpoints, searched recursively (default: the models dir of --system)",
    )
    parser.add_argument(
        "--max-eval-T",
        type=int,
        default=config.MAX_EVAL_T,
        help="Maximum T for evaluation",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Model prediction step (default: checkpoint metadata, else config.DT)",
    )
    parser.add_argument(
        "--system",
        choices=SYSTEM_CHOICES,
        default=None,
        help="Dynamical system (default: checkpoint metadata, else config)",
    )
    parser.add_argument(
        "--include-init",
        action="store_true",
        help="Also evaluate untrained models with the same seeds",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Only these groups, e.g. T1 T5 T10 LM",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save JSON and plot",
    )
    args = parser.parse_args()

    model_dir = args.model_dir or config.system_path(
        config.MODEL_DIR, args.system or config.DEFAULT_SYSTEM
    )
    run_gradient_scaling(
        model_dir,
        output_dir=args.output_dir,
        max_T=args.max_eval_T,
        dt=args.dt,
        system_name=args.system,
        include_init=args.include_init,
        groups=args.groups,
    )


if __name__ == "__main__":
    main()
