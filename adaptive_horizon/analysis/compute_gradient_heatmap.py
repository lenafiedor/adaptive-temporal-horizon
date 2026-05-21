import argparse
from pathlib import Path

import numpy as np
import torch

import adaptive_horizon.config as config
from adaptive_horizon.dynamics.lorenz import simulate_lorenz
from adaptive_horizon.evaluation.cross_validation import (
    get_history_window,
    get_normalization_stats,
)
from adaptive_horizon.evaluation.utils import load_model
from adaptive_horizon.training.loss import rollout_predictions, batch_loss
from adaptive_horizon.visualization.plotting import (
    plot_g_T_heatmap,
    plot_prediction_overlay,
)


def format_dt(dt: float) -> str:
    return str(dt).split(".")[1]


def trajectory_path(dt: float, seed: int):
    return config.ANALYSIS_DIR / f"diagnostic_trajectory_dt{format_dt(dt)}_seed{seed}.npz"


def generate_initial_state(seed: int):
    rng = np.random.default_rng(seed)
    return np.array(
        [
            rng.uniform(-20, 20),
            rng.uniform(-20, 20),
            rng.uniform(0, 50),
        ],
        dtype=np.float64,
    )


def load_or_generate_trajectory(
    dt: float,
    steps: int,
    seed: int,
    history_window: int,
    regenerate: bool = False,
):
    path = trajectory_path(dt, seed)
    burn_in = config.resolve_burn_in_steps(dt)

    if path.exists() and not regenerate:
        data = np.load(path)
        trajectory = data["trajectory"]
        saved_steps = int(data["steps"])
        saved_history_window = int(data["history_window"])
        if saved_steps >= steps and saved_history_window == history_window:
            print(f"Loaded diagnostic trajectory from {path}")
            return trajectory[: steps + 1], path
        print("Cached diagnostic trajectory metadata does not match; regenerating")

    initial_state = generate_initial_state(seed)
    trajectory = np.asarray(
        simulate_lorenz(
            initial_state=initial_state,
            dt=dt,
            steps=steps,
            burn_in=burn_in,
        ),
        dtype=np.float64,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        trajectory=trajectory,
        initial_state=initial_state,
        dt=dt,
        steps=steps,
        seed=seed,
        burn_in=burn_in,
        history_window=history_window,
    )
    print(f"Saved diagnostic trajectory to {path}")
    return trajectory, path


def resolve_T_val(T_val, tau, dt):
    if T_val is not None:
        print(f"Using T_val={T_val} ({T_val * dt:.4f} time units)")
        return int(T_val)
    if tau is None:
        raise ValueError("Provide either --T-val or --tau")
    resolved = max(1, int(round(tau / dt)))
    print(f"Resolved tau={tau:g} to T_val={resolved} ({resolved * dt:.4f} time units)")
    return resolved


def normalize_trajectory(trajectory, normalization_stats):
    if normalization_stats is None:
        mean = trajectory.mean(axis=0)
        std = trajectory.std(axis=0)
        print("Checkpoint has no normalization stats; using diagnostic trajectory stats")
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

                loss_T = batch_loss(model, batch_inputs, batch_targets[:, :T_val], T_val)
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
    preds_raw = denormalize(preds, mean, std)
    return preds_raw


def save_gradient_values(
    output_path,
    trajectory,
    sample_indices,
    g_values,
    g1_norms,
    gT_norms,
    metadata,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        trajectory=trajectory,
        sample_indices=sample_indices,
        g_values=g_values,
        g1_norms=g1_norms,
        gT_norms=gT_norms,
        **metadata,
    )
    print(f"Saved gradient heatmap values to {output_path}")


def compute_gradient_heatmap(args):
    T_val = resolve_T_val(args.T_val, args.tau, args.dt)
    if args.microbatch_size < 1:
        raise ValueError("--microbatch-size must be at least 1")

    device = "cuda" if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu"
    model, checkpoint = load_model(args.model)
    model = model.to(device)
    history_window = get_history_window(checkpoint)
    normalization_stats = get_normalization_stats(checkpoint)

    trajectory, diagnostic_path = load_or_generate_trajectory(
        dt=args.dt,
        steps=args.steps,
        seed=args.seed,
        history_window=history_window,
        regenerate=args.regenerate,
    )
    trajectory_normalized, mean, std = normalize_trajectory(
        trajectory, normalization_stats
    )
    inputs, targets, sample_indices = build_diagnostic_samples(
        trajectory_normalized, history_window, T_val
    )

    g_values, g1_norms, gT_norms = compute_local_g_values(
        model=model,
        inputs=inputs,
        targets=targets,
        T_val=T_val,
        microbatch_size=args.microbatch_size,
        device=device,
    )

    timestamp = args.output_suffix
    if timestamp is None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_name = Path(args.model).stem
    values_path = (
        config.ANALYSIS_DIR / f"gradient_heatmap_{model_name}_T{T_val}_{timestamp}.npz"
    )
    metadata = {
        "dt": args.dt,
        "T_val": T_val,
        "tau": T_val * args.dt,
        "steps": args.steps,
        "seed": args.seed,
        "microbatch_size": args.microbatch_size,
        "history_window": history_window,
        "diagnostic_trajectory_path": str(diagnostic_path),
    }
    save_gradient_values(
        values_path,
        trajectory,
        sample_indices,
        g_values,
        g1_norms,
        gT_norms,
        metadata,
    )

    plot_g_T_heatmap(trajectory, g_values, sample_indices, T_val=T_val, dt=args.dt,
                     filename=f"lorenz_gradient_heatmap_{model_name}_T{T_val}_{timestamp}.png")

    overlay_position = int(np.nanargmax(g_values))
    m = int(sample_indices[overlay_position])
    prediction = rollout_prediction(
        model,
        inputs,
        overlay_position,
        T_val,
        mean,
        std,
        device,
    )
    prediction_path = np.vstack([trajectory[m], prediction])
    ground_truth_path = trajectory[m : m + T_val + 1]
    plot_prediction_overlay(
        ground_truth_path,
        prediction_path,
        T_val=T_val,
        dt=args.dt,
        filename=f"lorenz_prediction_overlay_{model_name}_T{T_val}_{timestamp}.png",
    )

    print(
        f"Computed {len(g_values)} local g(T) values "
        f"for T={T_val} ({T_val * args.dt:.4f} time units)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True, help="Model path")
    parser.add_argument("--T-val", type=int, default=None, help="Evaluation horizon")
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Evaluation horizon in physical Lorenz time",
    )
    parser.add_argument("--dt", type=float, default=config.DT, help="Simulation step")
    parser.add_argument(
        "--steps",
        type=int,
        default=config.STEPS_PER_TRAJECTORY,
        help="Post-burn-in diagnostic trajectory length",
    )
    parser.add_argument(
        "--seed", type=int, default=config.EVAL_SEED, help="Diagnostic trajectory seed"
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
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Optional suffix for output filenames",
    )
    args = parser.parse_args()
    compute_gradient_heatmap(args)


if __name__ == "__main__":
    main()
