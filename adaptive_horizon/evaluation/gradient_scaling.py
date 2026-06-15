from torch.utils.data import DataLoader
import argparse

import adaptive_horizon.config as config
from adaptive_horizon.data.dataset import LorenzDataset, collate_fn
from adaptive_horizon.training.loss import compute_g_T
from adaptive_horizon.visualization.plotting import plot_g_T
from adaptive_horizon.evaluation.cross_validation import get_history_window
from adaptive_horizon.evaluation.utils import (
    get_checkpoint_normalization_stats,
    load_model,
)


def gradient_scaling(
    model_path,
    max_T=config.MAX_EVAL_T,
    dt=config.DT,
    per_batch=False,
):
    model, checkpoint = load_model(model_path)
    print(f"Loaded model from {model_path}")
    burn_in_steps = config.resolve_burn_in_steps(dt)
    print(f"Burn-in: {burn_in_steps} steps ({config.BURN_IN_TIME:g} time units)")

    adaptive = "adaptive" in str(model_path)
    train_T = checkpoint.get("train_T") if not adaptive else None
    history_window = get_history_window(checkpoint)
    split_gap = max(
        config.MAX_TRAIN_T,
        config.MAX_EVAL_T,
        history_window,
        max_T,
    )

    eval_dataset = LorenzDataset(
        dt=dt,
        T=max_T,
        normalize=True,
        seed=config.TRAJECTORY_SEED,
        burn_in=burn_in_steps,
        history_window=history_window,
        split="val",
        split_gap=split_gap,
        normalization_stats=get_checkpoint_normalization_stats(checkpoint),
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    T_vals = list(range(1, max_T + 1))
    g_vals = compute_g_T(model, eval_loader, T_vals, per_batch=per_batch)
    plot_g_T(g_vals, train_T=train_T, adaptive=adaptive, dt=dt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to saved model",
    )
    parser.add_argument(
        "--max-eval-T",
        type=int,
        default=config.MAX_EVAL_T,
        help="Maximum T for evaluation",
    )
    parser.add_argument(
        "--dt", type=float, default=config.DT, help="Time step for simulation"
    )
    parser.add_argument(
        "--per-batch",
        action="store_true",
        help="Compute per-batch gradient scaling ratios",
    )
    args = parser.parse_args()

    gradient_scaling(args.model, args.max_eval_T, args.dt, args.per_batch)


if __name__ == "__main__":
    main()
