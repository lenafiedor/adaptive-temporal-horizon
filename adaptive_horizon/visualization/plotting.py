import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from adaptive_horizon import config
from adaptive_horizon.config import LOSS_DIR, EVAL_DIR, ANALYSIS_DIR, DT
from adaptive_horizon.utils import format_dt

COLOR_TRAIN = "#8B87B0"
COLOR_EVAL = "#A0BAB5"


def plot_bounds(centers, lower_errors, upper_errors):
    centers_array = np.asarray(centers, dtype=np.float64)
    lower_array = np.asarray(lower_errors, dtype=np.float64)
    upper_array = np.asarray(upper_errors, dtype=np.float64)
    lower_bound = np.maximum(centers_array - lower_array, 0.0)
    upper_bound = centers_array + upper_array

    return centers_array, lower_bound, upper_bound


def save_losses(
    train_losses: torch.Tensor,
    val_losses: torch.Tensor,
    save_dir: Path = LOSS_DIR,
    T: int = None,
    adaptive: bool = False,
):
    """Save training history"""
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if adaptive:
        filename = f"adaptive_loss_{timestamp}"
    else:
        filename = f"loss_T{T}_{timestamp}"

    loss_path = save_dir / filename

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(train_losses, label="Train Loss", color=COLOR_TRAIN, linewidth=2)
    ax1.plot(val_losses, label="Val Loss", color=COLOR_EVAL, linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(
        f"Training Loss (T={T})" if not adaptive else "Adaptive Training Loss"
    )
    ax1.legend()

    plt.tight_layout()
    plt.savefig(f"{loss_path}.png", dpi=150)
    plt.close()
    print(f"\nLoss plot saved to {loss_path}.png")

    with open(f"{loss_path}.txt", "w") as f:
        f.write("mean_train_loss,mean_val_loss\n")
        f.write(f"{train_losses.mean().item()},{val_losses.mean().item()}")
    print(f"Loss values saved to {loss_path}.txt")


def plot_g_T(
    g_values,
    save_dir=EVAL_DIR,
    train_T=None,
    adaptive=False,
    dt=DT,
):
    Ts = list(g_values.keys())
    values = list(g_values.values())
    times = [T * dt for T in Ts]

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"gradient_scaling_T{train_T}_{timestamp}.png"
        if not adaptive
        else f"gradient_scaling_adaptive_{timestamp}.png"
    )
    plot_path = save_dir / filename

    fig, ax = plt.subplots()
    if values and isinstance(values[0], list):
        centers = []
        lower_errors = []
        upper_errors = []
        summary_label = None

        for T in Ts:
            center, lower_error, upper_error, summary_label = summarize_values(
                g_values[T]
            )
            centers.append(center)
            lower_errors.append(lower_error)
            upper_errors.append(upper_error)

        centers_array, lower_bound, upper_bound = plot_bounds(
            centers, lower_errors, upper_errors
        )

        ax.plot(times, centers_array, color=COLOR_TRAIN, linewidth=1.8)
        ax.fill_between(times, lower_bound, upper_bound, color=COLOR_TRAIN, alpha=0.2)
        ax.set_ylabel(f"g(T) ({summary_label})")
    else:
        ax.plot(times, values, color=COLOR_TRAIN, linewidth=1.8)
        ax.set_ylabel("g(T)")

    ax.set_xlabel(r"Validation Horizon ($\tau = T \cdot dt$)")
    ax.set_title("Gradient Scaling")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Gradient scaling plot saved to {plot_path}")


def plot_gradients_histogram(
    gradients: dict[int, list[float]],
    save_dir=LOSS_DIR,
    epoch=None,
    train_T=None,
    dt=DT,
    adaptive=False,
):
    """Save per-batch g(T) histogram diagnostics.

    Args:
        gradients: dictionary of g(T) values for each horizon
        save_dir: directory to save plot
        epoch: training epoch number
        train_T: training horizon
        dt: time step
        adaptive: whether the model is adaptive
    Returns:
        save_path: path to the saved histogram
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    gradient_items = sorted(gradients.items())
    num_plots = len(gradient_items)
    num_cols = min(3, num_plots)
    num_rows = int(np.ceil(num_plots / num_cols))

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(5 * num_cols, 4 * num_rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for ax, (T, values) in zip(flat_axes, gradient_items):
        ax.hist(
            values, bins=min(20, max(1, len(values))), alpha=0.85, color=COLOR_TRAIN
        )
        ax.set_xlabel("g(T)")
        ax.set_ylabel("batch count")
        ax.set_title(rf"$\tau = {T * dt:.2f}$")

    for ax in flat_axes[num_plots:]:
        ax.set_visible(False)

    train_time = train_T * dt if train_T is not None else None
    title = "g(T) batch histograms"
    if train_time is not None:
        title += rf" (train $\tau = {train_time:.2f}$)"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"T{train_T}" if not adaptive else "adaptive"
    epoch_part = f"_epoch{epoch + 1:03d}" if epoch is not None else ""
    save_path = save_dir / f"g_T_hist_{label}{epoch_part}_{timestamp}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\tg(T) histogram saved to {save_path}")
    return save_path


def gradient_history_quantiles(sorted_history, T):
    percentile_levels = (5.0, 25.0, 50.0, 75.0, 95.0, 99.0)
    percentile_rows = [
        [
            float(np.percentile([float(value) for value in gradients[T]], level))
            for level in percentile_levels
        ]
        for _, gradients in sorted_history
    ]
    p05, p25, median, p75, p95, p99 = np.asarray(percentile_rows, dtype=np.float64).T
    return p05, p25, median, p75, p95, p99


def plot_gradient_history(
    gradient_history: list[tuple[int, dict[int, list[float]]]],
    save_dir=LOSS_DIR,
    train_T=None,
    dt=DT,
    adaptive=False,
):
    if not gradient_history:
        raise ValueError("Cannot plot gradient scaling history for empty history")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sorted_history = sorted(gradient_history, key=lambda item: item[0])
    epochs = [epoch + 1 for epoch, _ in sorted_history]
    T_values = sorted(sorted_history[0][1].keys())
    T_times = [T * dt for T in T_values]
    cmap = plt.cm.tab10
    colors = [cmap(i / max(1, len(T_values) - 1)) for i in range(len(T_values))]

    num_plots = len(T_values)
    num_cols = min(3, num_plots)
    num_rows = int(np.ceil(num_plots / num_cols))
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(5 * num_cols, 3.8 * num_rows),
        squeeze=False,
        sharex=True,
    )
    flat_axes = axes.flatten()

    for ax, color, T, T_time in zip(flat_axes, colors, T_values, T_times):
        p05_values, p25_values, medians, p75_values, p95_values, p99_values = (
            gradient_history_quantiles(sorted_history, T)
        )

        ax.plot(
            epochs,
            medians,
            color=color,
            linewidth=1.8,
            label=rf"$\tau = {T_time:.2f}$",
        )
        ax.fill_between(
            epochs, p05_values, p95_values, color=color, alpha=0.12, linewidth=0
        )
        ax.fill_between(
            epochs, p25_values, p75_values, color=color, alpha=0.25, linewidth=0
        )
        ax.scatter(epochs, p99_values, color=color, alpha=0.35, s=14, linewidths=0)
        ax.set_title(rf"$\tau = {T_time:.2f}$")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    for ax in flat_axes[num_plots:]:
        ax.set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("g(T)")

    train_time = train_T * dt if train_T is not None else None
    title = "Gradient scaling over epochs"
    if train_time is not None:
        title += rf" (train $\tau = {train_time:.2f}$)"
    fig.suptitle(
        title + "\nmedian line, p25-p75 inner band, p05-p95 outer band, p99 markers"
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"T{train_T}" if not adaptive else "adaptive"
    save_path = save_dir / f"g_T_history_{label}_{timestamp}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"g(T) history plot saved to {save_path}")
    return save_path


def summarize_values(values):
    median = float(np.median(values))
    half_width = 1.96 * (float(np.std(values)) / np.sqrt(len(values)))
    return median, half_width, half_width, "median +/- 95% CI"


def plot_mse(
    summary,
    save_dir,
    dt,
    max_train_T=config.MAX_TRAIN_T,
    budget_based=False,
    metric="median",
):
    """
    Plot MSE summaries for each validation T as separate lines.

    Args:
        summary: output from summarize_cross_validation
        save_dir: directory to save plot
        dt: simulation time step
        max_train_T: maximum training time horizon
        budget_based: whether the model training is budget-based
        metric: median or mean
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fixed_summaries = summary["fixed"]
    train_Ts = [train_summary["train_T"] for train_summary in fixed_summaries]
    eval_summaries = fixed_summaries[0]["by_eval_T"]

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.tab20
    colors = [cmap(i / len(eval_summaries)) for i in range(len(eval_summaries))]
    train_times = [train_T * dt for train_T in train_Ts]

    for i, val_summary in enumerate(eval_summaries):
        fixed_by_train_T = [
            train_summary["by_eval_T"][i] for train_summary in fixed_summaries
        ]
        centers = [mse_summary[metric] for mse_summary in fixed_by_train_T]
        lower_errors = [
            mse_summary[metric] - mse_summary[f"{metric}_ci95_low"]
            for mse_summary in fixed_by_train_T
        ]
        upper_errors = [
            mse_summary[f"{metric}_ci95_high"] - mse_summary[metric]
            for mse_summary in fixed_by_train_T
        ]

        ax.errorbar(
            train_times,
            centers,
            yerr=np.array([lower_errors, upper_errors]),
            color=colors[i],
            label=rf"$\tau_{{val}}={val_summary['eval_T'] * dt:.2f}$",
            linewidth=1.5,
            marker=".",
            markersize=6,
            capsize=3,
        )

        ax.axhline(
            y=summary["adaptive"]["by_eval_T"][i][metric],
            color=colors[i],
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )

    ax.set_xlabel(r"Training horizon ($\tau = T \cdot dt$)")
    ax.set_ylabel(f"Validation MSE ({metric} +/- 95% CI)")
    ax.set_title("Cross-Validation MSE (dashed = adaptive model)")
    ax.set_yscale("log")
    ax.set_xticks(train_times)
    ax.legend(
        title="Validation Horizon",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "budget_" if budget_based else ""
    filename = f"{prefix}mse_dt_{format_dt(dt)}_T{max_train_T}_{metric}_{timestamp}.png"
    save_figure(fig, filename, save_dir)


def plot_mse_subplots(
    evaluation_records,
    summary,
    save_dir,
    dt,
    max_train_T=config.MAX_TRAIN_T,
    budget_based=False,
    metric="median",
):
    """
    Plot one subplot per validation horizon with per-seed dots and summary CIs.

    Fixed models are shown at their training-horizon columns. The adaptive model is
    shown as a horizontal line because it has no single fixed training T.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fixed_summaries = summary["fixed"]
    train_Ts = [train_summary["train_T"] for train_summary in fixed_summaries]
    eval_summaries = fixed_summaries[0]["by_eval_T"]
    num_plots = len(eval_summaries)
    ncols = 2 if num_plots > 1 else 1
    nrows = int(np.ceil(num_plots / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.5 * ncols, 3.6 * nrows),
        squeeze=False,
        sharex=True,
    )
    flat_axes = axes.ravel()

    x_positions = np.arange(len(train_Ts), dtype=np.float64)
    x_labels = [f"{train_T * dt:.2f}" for train_T in train_Ts]

    for i, val_summary in enumerate(eval_summaries):
        ax = flat_axes[i]
        val_T = int(val_summary["eval_T"])

        fixed_centers = []
        fixed_lows = []
        fixed_highs = []
        positive_values = []
        for train_index, train_summary in enumerate(fixed_summaries):
            train_T = int(train_summary["train_T"])
            seed_values = [
                float(record["mse"])
                for record in evaluation_records
                if record["model_type"] == "fixed"
                and int(record["train_T"]) == train_T
                and int(record["val_T"]) == val_T
            ]
            if seed_values:
                offsets = np.linspace(-0.16, 0.16, len(seed_values))
                label = "fixed seeds" if i == 0 and train_index == 0 else "_nolegend_"
                ax.scatter(
                    np.full(len(seed_values), x_positions[train_index]) + offsets,
                    seed_values,
                    color=COLOR_TRAIN,
                    alpha=0.45,
                    s=18,
                    linewidths=0,
                    label=label,
                    zorder=2,
                )
                positive_values.extend(value for value in seed_values if value > 0)

            mse_summary = train_summary["by_eval_T"][i]
            center = mse_summary[metric]
            fixed_centers.append(center)
            fixed_lows.append(mse_summary[f"{metric}_ci95_low"])
            fixed_highs.append(mse_summary[f"{metric}_ci95_high"])
            if center > 0:
                positive_values.append(center)

        adaptive_summary = summary["adaptive"]["by_eval_T"][i]
        adaptive_center = adaptive_summary[metric]
        adaptive_low = adaptive_summary[f"{metric}_ci95_low"]
        adaptive_high = adaptive_summary[f"{metric}_ci95_high"]
        if adaptive_center > 0:
            positive_values.append(adaptive_center)

        plot_floor = min(positive_values) * 0.5 if positive_values else 1e-12
        fixed_lows = [max(low, plot_floor) for low in fixed_lows]
        fixed_highs = [max(high, plot_floor) for high in fixed_highs]
        adaptive_low = max(adaptive_low, plot_floor)
        adaptive_high = max(adaptive_high, plot_floor)

        ax.fill_between(
            x_positions,
            fixed_lows,
            fixed_highs,
            color=COLOR_TRAIN,
            alpha=0.18,
            linewidth=0,
            label=f"fixed {metric} 95% CI" if i == 0 else "_nolegend_",
            zorder=1,
        )
        ax.plot(
            x_positions,
            fixed_centers,
            color=COLOR_TRAIN,
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=f"fixed {metric}" if i == 0 else "_nolegend_",
            zorder=3,
        )

        ax.fill_between(
            x_positions,
            [adaptive_low] * len(x_positions),
            [adaptive_high] * len(x_positions),
            color=COLOR_EVAL,
            alpha=0.22,
            linewidth=0,
            label=f"adaptive {metric} 95% CI" if i == 0 else "_nolegend_",
            zorder=1,
        )
        ax.axhline(
            adaptive_center,
            color=COLOR_EVAL,
            linestyle="--",
            linewidth=1.5,
            label=f"adaptive {metric}" if i == 0 else "_nolegend_",
            zorder=3,
        )

        delta_summary = (
            summary.get("deltas", {}).get("by_val_T", {}).get(str(val_T), {})
        )
        paired_count = delta_summary.get("n_pairs")
        title = rf"$\tau_{{val}}={val_T * dt:.2f}$ | adaptive wins {delta_summary['adaptive_wins']}/{paired_count}"
        ax.set_title(title)
        ax.set_yscale("log")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=35, ha="right")

    for ax in flat_axes[num_plots:]:
        ax.set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"Training horizon ($\tau = T \cdot dt$)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Validation MSE ({metric} +/- 95% CI)")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Cross-validation MSE by validation horizon")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "budget_" if budget_based else ""
    filename = f"{prefix}mse_subplots_dt_{format_dt(dt)}_T{max_train_T}_{metric}_{timestamp}.png"
    save_figure(fig, filename, save_dir)


def plot_paired_deltas(
    deltas,
    val_Ts,
    dt,
    save_dir,
    max_train_T,
    budget_based=False,
    metric="median",
):
    save_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray([T * dt for T in val_Ts], dtype=np.float64)
    centers = np.asarray(
        [deltas["by_val_T"][str(T)][metric] for T in val_Ts], dtype=np.float64
    )
    lows = np.asarray(
        [deltas["by_val_T"][str(T)][f"{metric}_ci95_low"] for T in val_Ts],
        dtype=np.float64,
    )
    highs = np.asarray(
        [deltas["by_val_T"][str(T)][f"{metric}_ci95_high"] for T in val_Ts],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.plot(
        x,
        centers,
        color=COLOR_TRAIN,
        linewidth=2.0,
        marker="o",
        label="Best fixed MSE - adaptive MSE",
    )
    ax.fill_between(x, lows, highs, color=COLOR_EVAL, alpha=0.35, linewidth=0)
    ax.set_xlabel(r"Validation horizon ($\tau = T \cdot dt$)")
    ax.set_ylabel(f"Paired MSE deltas ({metric})")
    ax.set_title("Horizon search: validation deltas")
    ax.legend()
    plt.tight_layout()

    prefix = "budget_" if budget_based else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"deltas_{prefix}dt_{format_dt(dt)}_T{max_train_T}_{metric}_{timestamp}.png"
    )
    save_figure(fig, filename, save_dir)


def plot_lyapunov_exponents(
    exponents, system_name=config.DEFAULT_SYSTEM, save_dir=ANALYSIS_DIR
):
    """
    Plot histograms for all Lyapunov exponents.

    Args:
        exponents: array of shape [N, 3] with local Lyapunov exponents
        system_name: name of the dynamical system
        save_dir: directory to save the figure
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
        ax.set_title(f"Distribution of {labels[i]}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, f"{system_name.lower()}_lle_histogram.png", save_dir=save_dir)


def plot_lle_heatmap(
    trajectory,
    exponents,
    burn_in=0,
    system_name=config.DEFAULT_SYSTEM,
    save_dir=ANALYSIS_DIR,
):
    """
    Plot 3D trajectory colored by each local Lyapunov exponent.

    Args:
        trajectory: array of shape [N, 3] with trajectory states
        exponents: array of shape [N, 3] with local Lyapunov exponents
        burn_in: number of initial steps to ignore
        system_name: name of the dynamical system
        save_dir: directory to save the figure
    """
    trajectory = np.array(trajectory)
    exponents = np.array(exponents)
    n_lle = len(exponents)
    traj_aligned = trajectory[burn_in : burn_in + n_lle]

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
        ax.set_title(f"{system_name} attractor colored by {labels[i]}")

        cbar = fig.colorbar(lc, ax=ax, shrink=0.5, aspect=20, pad=0.1)
        cbar.set_label(f"Local {labels[i]}")

    plt.tight_layout()
    save_figure(fig, f"{system_name.lower()}_lle_heatmap.png", save_dir=save_dir)


def plot_g_T_heatmap(
    trajectory,
    g_values,
    sample_indices,
    T_val,
    dt=DT,
    save_dir=ANALYSIS_DIR,
    system_name=config.DEFAULT_SYSTEM,
):
    """Plot a trajectory colored by local gradient scaling values."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    g_values = np.asarray(g_values, dtype=np.float64)
    sample_indices = np.asarray(sample_indices, dtype=int)

    if len(g_values) == 0:
        raise ValueError("Cannot plot gradient heatmap with no g(T) values")
    if len(g_values) != len(sample_indices):
        raise ValueError(
            f"g_values length {len(g_values)} does not match sample_indices length "
            f"{len(sample_indices)}"
        )

    valid_mask = sample_indices + 1 < len(trajectory)
    valid_indices = sample_indices[valid_mask]
    if len(valid_indices) == 0:
        raise ValueError("No valid trajectory segments for gradient heatmap")

    segments = np.stack(
        [trajectory[valid_indices], trajectory[valid_indices + 1]], axis=1
    )
    segment_values = g_values[valid_mask]
    values_for_percentiles = segment_values.astype(float).tolist()
    vmin = float(np.percentile(values_for_percentiles, 5.0))
    vmax = float(np.percentile(values_for_percentiles, 95.0))
    if np.isclose(vmin, vmax):
        vmin = float(segment_values.min())
        vmax = float(segment_values.max())

    norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    lc = Line3DCollection(segments, cmap="viridis", norm=norm)
    lc.set_array(segment_values)
    lc.set_linewidth(1.1)
    ax.add_collection3d(lc)

    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        rf"{system_name} attractor colored by local $g(T)$, $\tau_{{val}}={T_val * dt:.2f}$"
    )

    cbar = fig.colorbar(lc, ax=ax, shrink=0.65, aspect=20, pad=0.1)
    cbar.set_label(rf"$g(T)$ at $T={T_val}$, $\tau={T_val * dt:.2f}$")

    fig.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{system_name}_gradient_heatmap_T{T_val}_{timestamp}.png"
    save_figure(fig, filename, save_dir=save_dir)
    plt.close(fig)


def plot_prediction_overlay(
    ground_truth,
    prediction,
    T_val,
    dt=DT,
    save_dir=ANALYSIS_DIR,
    system_name=config.DEFAULT_SYSTEM,
):
    """Plot one ground-truth trajectory segment and the matching model rollout."""
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        ground_truth[:, 0],
        ground_truth[:, 1],
        ground_truth[:, 2],
        color=COLOR_EVAL,
        linewidth=2,
        label="Ground truth",
    )
    ax.plot(
        prediction[:, 0],
        prediction[:, 1],
        prediction[:, 2],
        color=COLOR_TRAIN,
        linewidth=2,
        linestyle="--",
        label="Prediction",
    )
    ax.scatter(
        ground_truth[0, 0],
        ground_truth[0, 1],
        ground_truth[0, 2],
        color="black",
        s=25,
        label="Start",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(rf"Prediction rollout overlay, $\tau={T_val * dt:.2f}$")
    ax.legend()
    fig.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{system_name}_prediction_T{T_val}_{timestamp}.png"
    save_figure(fig, filename, save_dir=save_dir)
    plt.close(fig)


def save_figure(fig, filename, save_dir=ANALYSIS_DIR, dpi=150):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    fig.savefig(save_path, dpi=dpi)
    print(f"Plot saved to {save_path}")
