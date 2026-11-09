"""Long-horizon evaluation (default 100 steps) of every model group of one system/dt.

For each group (fixed 20/200 epochs per training horizon T, and the adaptive
methods) every checkpoint is rolled out once for ``--max-eval-T`` steps on the
validation split. Per model it stores the per-step error curves (MSE, NRMSE,
SMAPE; see ``evaluation.metrics``) and the median valid prediction time (VPT)
at several NRMSE thresholds, so that tables at any validation window and VPT
figures can be built without re-evaluating.

Output: ``<output-dir>/long_horizon_<max_T>.json`` with ``metadata`` and
``records`` (one per model: group, train_T, seed, curves, vpt). Groups that
already appear in an existing output with all their checkpoints are skipped.

Usage:
    python -m adaptive_horizon.evaluation.long_horizon --system lorenz --dt-tag 08
"""

import argparse
import json
import math
from pathlib import Path

import torch

from adaptive_horizon import config
from adaptive_horizon.data.dataset import make_val_loader
from adaptive_horizon.dynamics.lyapunov import compute_global_lyapunov
from adaptive_horizon.dynamics.systems import SYSTEM_CHOICES
from adaptive_horizon.evaluation.metrics import DEFAULT_VPT_THRESHOLDS
from adaptive_horizon.evaluation.utils import (
    get_checkpoint_normalization_stats,
    load_model,
)
from adaptive_horizon.training.loss import rollout_predictions
from adaptive_horizon.training.utils import find_fixed_models

EXP = Path("experiments")
DTS = {"05": 0.05, "08": 0.08, "1": 0.1}


def model_dirs(system, dt_tag):
    """Model directory of every group, following scripts/rerun_audit.sh."""
    base = EXP / system / "models" / f"dt_{dt_tag}"
    fixed200 = (
        base / "dt_08_fixed_200"
        if (system, dt_tag) == ("rossler", "08")
        else base / f"budget_based_dt_{dt_tag}_fixed_200"
    )
    return {
        "fixed20": base / f"budget_based_dt_{dt_tag}_fixed_20",
        "fixed200": fixed200,
        "budget": base / f"budget_based_dt_{dt_tag}_lm_200",
        "lm": base / f"dt_{dt_tag}_lm_200",
        "lmr": base / f"dt_{dt_tag}_lmr_200",
        "ls": base / f"dt_{dt_tag}_ls_200",
        "cv": base / f"dt_{dt_tag}_cv_200",
        "wl": base / f"dt_{dt_tag}_wl_200",
    }


def group_paths(group, directory):
    """[(path, train_T, seed)] of one group; train_T is None for adaptive models."""
    if not directory.exists():
        return []
    if group.startswith("fixed"):
        return find_fixed_models(directory)
    return [(p, None, None) for p in sorted(directory.rglob("adaptive_mlp*.pt"))]


@torch.no_grad()
def evaluate(model, loader, max_T, thresholds, device):
    """Per-step mean MSE/NRMSE/SMAPE curves and median VPT per threshold."""
    stats = loader.dataset.normalization_stats
    std = torch.as_tensor(stats["std"], device=device) + 1e-8
    mean = torch.as_tensor(stats["mean"], device=device)
    sums = {name: torch.zeros(max_T, dtype=torch.float64) for name in ("mse", "nrmse", "smape")}
    nrmse_rows, n = [], 0
    model.eval()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets[:, :max_T].to(device)
        pred = rollout_predictions(model, inputs, max_T)
        squared = torch.nan_to_num((pred - targets).square().mean(dim=2), nan=math.inf)
        nrmse = squared.sqrt()
        pred_phys, target_phys = pred * std + mean, targets * std + mean
        smape = torch.nan_to_num(
            200 * (pred_phys - target_phys).abs()
            / (pred_phys.abs() + target_phys.abs()).clamp_min(1e-8),
            nan=200.0,
        ).mean(dim=2)
        sums["mse"] += squared.sum(dim=0).double().cpu()
        sums["nrmse"] += nrmse.sum(dim=0).double().cpu()
        sums["smape"] += smape.sum(dim=0).double().cpu()
        nrmse_rows.append(nrmse.cpu())
        n += len(inputs)
    nrmse = torch.cat(nrmse_rows)
    vpt = {}
    for threshold in thresholds:
        invalid = nrmse > threshold
        steps = torch.where(invalid.any(dim=1), invalid.int().argmax(dim=1), max_T)
        vpt[str(threshold)] = {
            "median": steps.double().quantile(0.5).item(),
            "mean": steps.double().mean().item(),
            "censored_fraction": (steps == max_T).double().mean().item(),
        }
    curves = {name: (s / n).tolist() for name, s in sums.items()}
    return curves, vpt


def run(system, dt_tag, max_T, output_dir, groups, batch_size, device):
    dt = DTS[dt_tag]
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"long_horizon_{max_T}.json"
    payload = json.loads(out.read_text()) if out.exists() else {"records": []}
    payload["metadata"] = {
        "system": system,
        "dt": dt,
        "max_T": max_T,
        "vpt_thresholds": list(DEFAULT_VPT_THRESHOLDS),
        "lyapunov_exponent": float(
            compute_global_lyapunov(dt=config.INTEGRATION_DT, system=system)
        ),
        "model_dirs": {g: str(d) for g, d in model_dirs(system, dt_tag).items()},
    }
    loaders = {}
    for group, directory in model_dirs(system, dt_tag).items():
        if groups and group not in groups:
            continue
        paths = group_paths(group, directory)
        done = {r["path"] for r in payload["records"] if r["group"] == group}
        todo = [p for p in paths if str(p[0]) not in done]
        print(f"{system} dt_{dt_tag} {group}: {len(paths)} models, {len(todo)} to evaluate")
        for path, train_T, seed in todo:
            model, checkpoint = load_model(path)
            stats = get_checkpoint_normalization_stats(checkpoint)
            key = tuple(tuple(map(float, stats[k])) for k in ("mean", "std"))
            if key not in loaders:
                loaders[key] = make_val_loader(system, dt, max_T, stats, batch_size=batch_size)
            curves, vpt = evaluate(model.to(device), loaders[key], max_T,
                                   DEFAULT_VPT_THRESHOLDS, device)
            payload["records"].append({
                "group": group,
                "path": str(path),
                "train_T": train_T if train_T is not None else checkpoint.get("train_T"),
                "seed": seed if seed is not None else checkpoint.get("seed"),
                "curves": curves,
                "vpt": vpt,
            })
        out.write_text(json.dumps(payload))
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--system", choices=SYSTEM_CHOICES, required=True)
    parser.add_argument("--dt-tag", choices=list(DTS), required=True)
    parser.add_argument("--max-eval-T", type=int, default=100)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or (
        EXP / args.system / "evaluation" / f"dt_{args.dt_tag}" / f"long_horizon_{args.max_eval_T}"
    )
    run(args.system, args.dt_tag, args.max_eval_T, output_dir, args.groups,
        args.batch_size, config.DEVICE)


if __name__ == "__main__":
    main()
