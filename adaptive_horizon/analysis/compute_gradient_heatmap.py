import argparse
from pathlib import Path
import numpy as np
import torch
from datetime import datetime

import adaptive_horizon.config as config
from adaptive_horizon.data.utils import (
    default_lorenz_trajectory_path,
    get_lorenz_trajectory,
)
from adaptive_horizon.evaluation.cross_validation import (
    get_history_window,
)
from adaptive_horizon.evaluation.utils import (
    get_checkpoint_normalization_stats,
    load_model,
)
from adaptive_horizon.training.loss import rollout_predictions, batch_loss
from adaptive_horizon.visualization.plotting import (
    plot_g_T_heatmap,
    plot_prediction_overlay,
)


def normalize_trajectory(trajectory, normalization_stats):
    if normalization_stats is None:
        mean = trajectory.mean(axis=0)
        std = trajectory.std(axis=0)
        print(
            "Checkpoint has no normalization stats; using diagnostic trajectory stats"
        )
    else:
        mean = np.asarray(normalization_stats["mean"], dtype=np.float64)
        std = np.asarray(normalization_stats["std"], dtype=np.float64)

    return (trajectory - mean) / (std + 1e-8), mean, std


def denormalize(values, mean, std):
    return values * (std + 1e-8) + mean


def build_diagnostic_samples(trajectory_normalized, history_window, T_val):
    inputs = []
    targets = []
    sample_indices = []
    seq_len = len(trajectory_normalized)

    for m in range(history_window - 1, seq_len - T_val):
        inputs.append(trajectory_normalized[m - history_window + 1 : m + 1].flatten())
        targets.append(trajectory_normalized[m + 1 : m + T_val + 1])
        sample_indices.append(m)

    if not inputs:
        raise ValueError(
            "No diagnostic samples could be built. Increase --steps or reduce --T-val."
        )

    return (
        torch.tensor(np.asarray(inputs), dtype=torch.float32),
        torch.tensor(np.asarray(targets), dtype=torch.float32),
        np.asarray(sample_indices, dtype=int),
    )


def gradient_norm(loss, params, device):
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    total = torch.zeros((), device=device)
    for grad in grads:
        if grad is not None:
            total += grad.norm().pow(2)
    return torch.sqrt(total)


def compute_local_g_values(
    model,
    inputs,
    targets,
    T_val,
    microbatch_size,
    device,
    epsilon=1e-12,
):
    was_training = model.training
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    g_values = np.empty(len(inputs), dtype=np.float64)
    g1_norms = np.empty(len(inputs), dtype=np.float64)
    gT_norms = np.empty(len(inputs), dtype=np.float64)

    try:
        with torch.enable_grad():
            for start in range(0, len(inputs), microbatch_size):
                end = min(start + microbatch_size, len(inputs))
                batch_inputs = inputs[start:end].to(device)
                batch_targets = targets[start:end].to(device)

                loss_1 = batch_loss(model, batch_inputs, batch_targets[:, :1], 1)
                norm_1 = gradient_norm(loss_1, params, device)

                loss_T = batch_loss(
                    model, batch_inputs, batch_targets[:, :T_val], T_val
                )
                norm_T = gradient_norm(loss_T, params, device)

                ratio = (norm_T / norm_1.clamp_min(epsilon)).item()
                g_values[start:end] = ratio
                g1_norms[start:end] = norm_1.item()
                gT_norms[start:end] = norm_T.item()
    finally:
        model.train(was_training)

    return g_values, g1_norms, gT_norms


def rollout_prediction(model, inputs, sample_index, T_val, mean, std, device):
    model.eval()
    with torch.no_grad():
        input_tensor = inputs[sample_index : sample_index + 1].to(device)
        preds = rollout_predictions(model, input_tensor, T_val)[0].cpu().numpy()
    return denormalize(preds, mean, std)


def compute_gradient_heatmap(args):
    device = "cuda" if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu"
    model, checkpoint = load_model(args.model)
    model = model.to(device)
    history_window = get_history_window(checkpoint)
    normalization_stats = get_checkpoint_normalization_stats(checkpoint)

    burn_in = config.resolve_burn_in_steps(args.dt)
    trajectory_path = default_lorenz_trajectory_path(
        config.DATA_DIR,
        dt=args.dt,
        steps=args.steps,
        seed=args.seed,
    )
    trajectory = get_lorenz_trajectory(
        dt=args.dt,
        steps=args.steps,
        burn_in=burn_in,
        seed=args.seed,
        path=trajectory_path,
        regenerate=args.regenerate,
    ).numpy()
    print(f"Using diagnostic trajectory from {trajectory_path}")

    trajectory_normalized, mean, std = normalize_trajectory(
        trajectory, normalization_stats
    )
    inputs, targets, sample_indices = build_diagnostic_samples(
        trajectory_normalized, history_window, args.T_val
    )

    g_values, g1_norms, gT_norms = compute_local_g_values(
        model=model,
        inputs=inputs,
        targets=targets,
        T_val=args.T_val,
        microbatch_size=args.microbatch_size,
        device=device,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(args.model).stem

    plot_g_T_heatmap(
        trajectory,
        g_values,
        sample_indices,
        T_val=args.T_val,
        dt=args.dt,
        filename=f"lorenz_gradient_heatmap_{model_name}_T{args.T_val}_{timestamp}.png",
    )

    overlay_position = int(np.nanargmax(g_values))
    m = int(sample_indices[overlay_position])
    prediction = rollout_prediction(
        model,
        inputs,
        overlay_position,
        args.T_val,
        mean,
        std,
        device,
    )
    prediction_path = np.vstack([trajectory[m], prediction])
    ground_truth_path = trajectory[m : m + args.T_val + 1]
    plot_prediction_overlay(
        ground_truth_path,
        prediction_path,
        T_val=args.T_val,
        dt=args.dt,
        filename=f"lorenz_prediction_{model_name}_T{args.T_val}_{timestamp}.png",
    )

    print(
        f"Computed {len(g_values)} local g(T) values "
        f"for T={args.T_val} ({args.T_val * args.dt:.4f} time units)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True, help="Model path")
    parser.add_argument(
        "--T-val", type=int, default=config.MAX_EVAL_T, help="Evaluation horizon"
    )
    parser.add_argument("--dt", type=float, default=config.DT, help="Simulation step")
    parser.add_argument(
        "--steps",
        type=int,
        default=config.TRAJECTORY_STEPS,
        help="Post-burn-in diagnostic trajectory length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.TRAJECTORY_SEED,
        help="Diagnostic trajectory seed",
    )
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=1,
        help="Samples per local gradient-scaling estimate",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate the cached diagnostic trajectory",
    )
    args = parser.parse_args()
    compute_gradient_heatmap(args)


if __name__ == "__main__":
    main()
