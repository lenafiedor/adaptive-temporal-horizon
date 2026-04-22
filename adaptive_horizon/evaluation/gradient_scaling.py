from torch.utils.data import DataLoader
import argparse

from adaptive_horizon.config import BATCH_SIZE, NUM_TRAJECTORIES, STEPS_PER_TRAJECTORY
from adaptive_horizon.data.dataset import LorenzDataset, collate_fn
from adaptive_horizon.training.loss import compute_g_T
from adaptive_horizon.visualization.plotting import plot_g_T
from adaptive_horizon.evaluation.cross_validation import load_model


def gradient_scaling(model_path, max_T):
    model, checkpoint = load_model(model_path)
    print(f"Loaded model from {model_path}")

    adaptive = "adaptive" in model_path
    train_T = checkpoint.get("train_T") if not adaptive else None

    eval_dataset = LorenzDataset(
        num_trajectories=NUM_TRAJECTORIES,
        steps_per_trajectory=STEPS_PER_TRAJECTORY,
        T=max_T,
        normalize=True,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    T_vals = list(range(1, max_T + 1))
    g_vals = compute_g_T(model, eval_loader, T_vals)
    plot_g_T(g_vals, train_T=train_T, adaptive=adaptive)


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
        "--max-eval-T", type=int, default=200, help="Maximum T for evaluation"
    )
    args = parser.parse_args()

    gradient_scaling(args.model, args.max_eval_T)


if __name__ == "__main__":
    main()
