import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Sequence
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from numpy.typing import NDArray

from adaptive_horizon.adaptive_methods import get_adaptive_method_abbreviation
from adaptive_horizon.config import MODEL_DIR, LOSS_DIR, EVAL_DIR, ANALYSIS_DIR


def save_losses(
    train_losses: torch.Tensor,
    val_losses: torch.Tensor,
    save_dir: Path = LOSS_DIR,
    T: int = None,
    adaptive: bool = False,
    method: str = None,
):
    """Save training history"""
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"loss_T{T}_{timestamp}"
        if not adaptive
        else f"adaptive_loss_{get_adaptive_method_abbreviation(method)}_{timestamp}"
    )
    loss_path = save_dir / filename
    plot_title = f"Training Loss (T={T})" if not adaptive else "Adaptive Training Loss"

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(train_losses, label="Train Loss", linewidth=2)
    ax1.plot(val_losses, label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(plot_title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{loss_path}.png", dpi=150)
    plt.close()
    print(f"\nLoss plot saved to {loss_path}.png")

    with open(f"{loss_path}.txt", "w") as f:
        f.write("min_train_loss,min_val_loss\n")
        f.write(f"{min(train_losses)},{min(val_losses)}")
    print(f"Min loss values saved to {loss_path}.txt")


def save_model(
    model,
    config,
    seed,
    save_dir=MODEL_DIR,
    T=None,
    adaptive=False,
    method: str = None,
    metadata=None,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if adaptive:
        filename = f"adaptive_mlp_{get_adaptive_method_abbreviation(method)}_seed{seed}_{timestamp}.pt"
    else:
        filename = f"mlp_T{T}_seed{seed}_{timestamp}.pt"

    model_path = save_dir / filename

    save_dict = {
        "model_state_dict": model.state_dict(),
        "train_T": T,
        "seed": seed,
        "config": {
            "input_size": config.input_size,
            "output_size": config.output_size,
            "layer_widths": config.layer_widths,
            "residual_connections": config.residual_connections,
            "k": config.k,
        },
    }
    if metadata is not None:
        save_dict["metadata"] = metadata

    torch.save(save_dict, model_path)
    print(f"Model saved to {model_path}")
    return model_path


def plot_g_T(g_values, save_dir=EVAL_DIR, train_T=None, adaptive=False):
    Ts = list(g_values.keys())
    values = list(g_values.values())

    save_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"gradient_scaling_T{train_T}_{timestamp}.png"
        if not adaptive
        else f"gradient_scaling_adaptive_{timestamp}.png"
    )
    plot_path = save_dir / filename

    plt.figure()
    plt.plot(Ts, values, marker="o")
    plt.xlabel("T")
    plt.ylabel("g(T)")
    plt.title("Gradient Scaling")
    plt.grid(True)
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Gradient scaling plot saved to {plot_path}")


def summarize_values(values: Sequence[float] | NDArray[np.float64], summary_mode):
    values_array: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    if len(values_array) == 0:
        raise ValueError("Cannot summarize empty values")

    if summary_mode == "mean-std":
        center = float(values_array.mean())
        spread = float(values_array.std())
        return center, spread, spread, "mean +/- std"

    if summary_mode == "mean-ci":
        center = float(values_array.mean())
        std = float(values_array.std())
        sem = std / np.sqrt(len(values_array))
        half_width = 1.96 * sem
        return center, half_width, half_width, "mean +/- 95% CI"

    if summary_mode == "median-iqr":
        q1, median, q3 = np.percentile(values_array, [25, 50, 75])
        return float(median), float(median - q1), float(q3 - median), "median with IQR"

    raise ValueError(f"Unsupported summary mode: {summary_mode}")


def get_summary_mode_abbreviation(summary_mode: str) -> str:
    abbreviations = {
        "mean-std": "std",
        "mean-ci": "ci",
        "median-iqr": "iqr",
    }

    try:
        return abbreviations[summary_mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported summary mode: {summary_mode}") from exc


def get_evaluation_method_abbreviation(
    evaluation_records, adaptive_method: str | None = None
) -> str:
    if adaptive_method is not None:
        return get_adaptive_method_abbreviation(adaptive_method)

    adaptive_methods = {
        record["adaptive_method"]
        for record in evaluation_records
        if record["model_type"] == "adaptive"
    }
    if len(adaptive_methods) == 1:
        return get_adaptive_method_abbreviation(next(iter(adaptive_methods)))
    if len(adaptive_methods) > 1:
        return "mixed"
    return "fixed"


def plot_mse(
    T_values,
    evaluation_records,
    save_dir,
    dt,
    summary_mode="mean-std",
    adaptive_method: str | None = None,
):
    """
    Plot MSE summaries for each validation T as separate lines.

    Args:
        T_values: list of T values
        evaluation_records: list of per-model, per-validation-horizon records
        save_dir: directory to save plot
        dt: simulation time step
        summary_mode: one of mean-std, mean-ci, median-iqr
        adaptive_method: adaptive method used for training
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.tab20
    colors = [cmap(i / len(T_values)) for i in range(len(T_values))]
    train_times = [train_T * dt for train_T in T_values]
    summary_label = None

    for i, val_T in enumerate(T_values):
        centers = []
        lower_errors = []
        upper_errors = []
        val_time = val_T * dt

        for train_T in T_values:
            values = [
                record["mse"]
                for record in evaluation_records
                if record["model_type"] == "fixed"
                and record["train_T"] == train_T
                and record["val_T"] == val_T
            ]
            center, lower_error, upper_error, summary_label = summarize_values(
                values, summary_mode
            )
            centers.append(center)
            lower_errors.append(lower_error)
            upper_errors.append(upper_error)

        ax.errorbar(
            train_times,
            centers,
            yerr=np.array([lower_errors, upper_errors]),
            color=colors[i],
            label=f"$t_L={val_time:.2f}$",
            linewidth=1.5,
            marker=".",
            markersize=6,
            capsize=3,
        )

        adaptive_values = [
            record["mse"]
            for record in evaluation_records
            if record["model_type"] == "adaptive" and record["val_T"] == val_T
        ]
        if adaptive_values:
            adaptive_center, _, _, _ = summarize_values(adaptive_values, summary_mode)
            ax.axhline(
                y=adaptive_center,
                color=colors[i],
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
            )

    has_adaptive = any(
        record["model_type"] == "adaptive" for record in evaluation_records
    )

    ax.set_xlabel(r"Training Horizon ($T \cdot dt$)")
    ax.set_ylabel(f"Validation MSE ({summary_label})")
    suffix = " (dashed = adaptive model)" if has_adaptive else ""
    ax.set_title("Cross-Validation MSE" + suffix)
    ax.set_yscale("log")
    ax.set_xticks(train_times)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Validation Horizon", loc="lower right")

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_abbreviation = get_evaluation_method_abbreviation(
        evaluation_records, adaptive_method
    )
    summary_abbreviation = get_summary_mode_abbreviation(summary_mode)
    save_path = save_dir / (
        f"mse_dt_{str(dt).split('.')[1]}_{method_abbreviation}_{summary_abbreviation}_{timestamp}.png"
    )
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Cross-validation MSE plot saved to {save_path}")


def plot_lyapunov_exponents(exponents, window):
    """
    Plot histograms of all 3 Lyapunov exponents.

    Args:
        exponents: array of shape [N, 3] with local Lyapunov exponents
        window: window size used for computation
    """
    exponents = np.array(exponents)
    labels = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, ax in enumerate(axes):
        lle_i = np.asarray(exponents[:, i], dtype=np.float64)
        mean_lle = sum(lle_i.tolist()) / len(lle_i)
        ax.hist(lle_i, bins=50, edgecolor="black", alpha=0.7, density=True)
        ax.axvline(
            x=mean_lle,
            color="r",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {mean_lle:.3f}",
        )
        ax.set_xlabel("Local Lyapunov Exponent")
        ax.set_ylabel("Density")
        ax.set_yscale("log")
        ax.set_title(f"Distribution of {labels[i]} (window={window})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, f"lle_histogram_W{window}.png")


def plot_trajectory_heatmap(trajectory, exponents, window, burn_in=0):
    """
    Plot 3D Lorenz trajectory colored by each of the 3 local Lyapunov exponents.

    Args:
        trajectory: array of shape [N, 3] with trajectory states
        exponents: array of shape [N-window, 3] with local Lyapunov exponents
        window: window size used for computation
        burn_in: number of initial steps to ignore
    """
    trajectory = np.array(trajectory)
    exponents = np.array(exponents)
    n_lle = len(exponents)

    offset = burn_in + window
    traj_aligned = trajectory[offset : offset + n_lle]

    x = traj_aligned[:, 0]
    y = traj_aligned[:, 1]
    z = traj_aligned[:, 2]

    labels = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]

    fig = plt.figure(figsize=(18, 6))

    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")

        points = np.stack((x, y, z), axis=-1).reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lle_i = exponents[:, i]
        norm = plt.Normalize(lle_i.min(), lle_i.max())
        lc = Line3DCollection(segments, cmap="coolwarm", norm=norm)
        lc.set_array(lle_i[:-1])
        lc.set_linewidth(1)
        ax.add_collection3d(lc)

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_zlim(z.min(), z.max())
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Lorenz Attractor colored by {labels[i]}")

        cbar = fig.colorbar(lc, ax=ax, shrink=0.5, aspect=20, pad=0.1)
        cbar.set_label(f"Local {labels[i]}")

    plt.tight_layout()
    save_figure(fig, f"lorenz_lle_heatmap_W{window}.png")


def save_figure(fig, filename, save_dir=ANALYSIS_DIR, dpi=150):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    fig.savefig(save_path, dpi=dpi)
    print(f"Plot saved to {save_path}")
