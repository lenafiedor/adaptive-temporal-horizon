from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adaptive_horizon import config
from adaptive_horizon.config import resolve_burn_in_steps
from adaptive_horizon.data.utils import simulate_trajectory
from adaptive_horizon.dynamics.systems import get_system
from adaptive_horizon.visualization.style import (
    BLUE,
    INK,
    INK_2,
    MUTED,
    ORANGE,
    RED,
    TEXT_WIDTH,
    apply_rc,
    style_axes,
    style_axes_3d,
)

LAP_DT = 0.01
LAP_STEPS = config.SIMULATION_STEPS
WINDOW_TIME = 12.0
OUT = Path(__file__).resolve().parents[2] / "experiments" / "thesis_results"


def lorenz_lobe_centers(rho=28, beta=8 / 3):
    center_radius = np.sqrt(beta * (rho - 1))
    z_center = rho - 1
    return {
        -1: np.array([-center_radius, -center_radius, z_center]),
        1: np.array([center_radius, center_radius, z_center]),
    }


def find_wing_crossings(trajectory):
    x = np.asarray(trajectory)[:, 0]
    signs = np.sign(x)
    signs[signs == 0] = 1
    return np.where(np.diff(signs) != 0)[0]


def same_wing_segments(trajectory, crossings, min_segment_steps=5):
    trajectory = np.asarray(trajectory)
    boundaries = [0, *[int(index) + 1 for index in crossings], len(trajectory)]
    segments = []

    for start, end in pairwise(boundaries):
        if end - start < min_segment_steps:
            continue

        segment_x = trajectory[start:end, 0]
        wing_sign = 1 if np.median(segment_x) >= 0 else -1
        segments.append((start, end, wing_sign))

    return segments


def detect_lobe_laps(
    trajectory,
    crossings,
    rho=28,
    beta=8 / 3,
    min_segment_steps=5,
):
    centers = lorenz_lobe_centers(rho=rho, beta=beta)
    lap_indices = []
    lap_durations = []

    for start, end, wing_sign in same_wing_segments(
        trajectory, crossings, min_segment_steps=min_segment_steps
    ):
        segment = trajectory[start:end]
        center = centers[wing_sign]
        theta = np.unwrap(
            np.arctan2(segment[:, 2] - center[2], segment[:, 0] - center[0])
        )
        direction = np.sign(np.median(np.diff(theta)))
        if direction == 0:
            continue

        directed_theta = direction * theta
        phase_progress = directed_theta - directed_theta[0]
        next_threshold = 2 * np.pi
        previous_lap_index = start

        while phase_progress[-1] >= next_threshold:
            crossed_threshold = np.flatnonzero(phase_progress >= next_threshold)
            if len(crossed_threshold) == 0:
                break
            local_lap_index = int(crossed_threshold[0])
            lap_index = start + local_lap_index
            lap_indices.append(lap_index)
            lap_durations.append(lap_index - previous_lap_index)

            previous_lap_index = lap_index
            next_threshold += 2 * np.pi

    return np.array(lap_indices, dtype=int), np.array(lap_durations, dtype=int)


def duration_stats(durations, dt):
    """Summary statistics of durations given in steps, converted to time units."""
    times = np.asarray(durations, dtype=np.float64) * dt
    if len(times) == 0:
        return {"count": 0}
    p05, p25, median, p75, p95 = np.percentile(times, [5, 25, 50, 75, 95])
    return {
        "count": len(times),
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "median": float(median),
        "p05": float(p05),
        "p25": float(p25),
        "p75": float(p75),
        "p95": float(p95),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }


def lorenz_trajectory(dt=LAP_DT, steps=LAP_STEPS):
    """Lorenz-63 trajectory with the same seed and burn-in as the LLE figures."""
    system = get_system("lorenz")
    rng = np.random.default_rng(config.RANDOM_SEED)
    return np.array(
        simulate_trajectory(
            system,
            initial_state=system.sample_initial_state(rng),
            dt=dt,
            steps=steps,
            burn_in=resolve_burn_in_steps(dt),
        )
    )


def busiest_window_start(crossings, length, window):
    """Start index of the window containing the most wing crossings."""
    if len(crossings) == 0 or window >= length:
        return 0
    starts = np.arange(0, length - window + 1, max(1, window // 20))
    counts = np.searchsorted(crossings, starts + window) - np.searchsorted(
        crossings, starts
    )
    return int(starts[np.argmax(counts)])


def plot_lorenz_with_crossings(
    traj, crossings, lap_indices, lap_durations, dt, window_time=WINDOW_TIME
):
    """Attractor and x(tau) with wing crossings and lap completions marked,
    plus the distribution of the lap duration.

    The first two panels show only a window of `window_time` time units
    (the one with the most wing crossings) so that
    individual laps stay readable; the histogram uses the whole trajectory.
    """
    window = min(len(traj), round(window_time / dt))
    start = busiest_window_start(crossings, len(traj), window)
    win_traj = traj[start : start + window]
    win_cross = crossings[(crossings >= start) & (crossings < start + window)] - start
    win_laps = lap_indices[(lap_indices >= start) & (lap_indices < start + window)]
    win_laps = win_laps - start
    tau = np.arange(window) * dt

    fig = plt.figure(figsize=(TEXT_WIDTH, 2.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.45, 1.0])

    # (a) attractor
    ax = fig.add_subplot(grid[0], projection="3d")
    ax.plot(*win_traj.T, color=MUTED, linewidth=0.4, alpha=0.8)
    ax.scatter(*win_traj[win_cross].T, color=ORANGE, s=5, linewidths=0, alpha=1)
    ax.scatter(*win_traj[win_laps].T, color=BLUE, s=5, linewidths=0, alpha=1)
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect(None, zoom=0.88)
    style_axes_3d(ax)
    ax.set_title("Lorenz-63 attractor", fontsize=8, color=INK, y=0.92)

    # (b) x coordinate over time
    ax = fig.add_subplot(grid[1])
    ax.plot(tau, win_traj[:, 0], color=MUTED, linewidth=0.7)
    ax.axhline(0.0, color=INK_2, linewidth=0.6, linestyle="--")
    for index in win_cross:
        ax.axvline(index * dt, color=ORANGE, linewidth=0.5, alpha=0.35)
    ax.scatter(
        win_cross * dt,
        win_traj[win_cross, 0],
        color=ORANGE,
        s=7,
        linewidths=0,
        zorder=3,
    )
    ax.scatter(
        win_laps * dt, win_traj[win_laps, 0], color=BLUE, s=7, linewidths=0, zorder=3
    )
    style_axes(ax)
    ax.set_xlim(0, window_time)
    ax.set_xlabel("Time $\\tau$ (from window start)", fontsize=7, color=INK_2)
    ax.set_ylabel("$x$", fontsize=7, color=INK_2)
    ax.set_title("Wing crossings and laps", fontsize=8, color=INK, loc="left")

    # (c) lap duration distribution
    ax = fig.add_subplot(grid[2])
    durations = np.asarray(lap_durations) * dt
    bins = (np.arange(lap_durations.min(), lap_durations.max() + 2) - 0.5) * dt
    ax.hist(durations, bins=bins, density=True, color=BLUE, alpha=0.85, linewidth=0)
    mean = float(np.mean(durations))
    ax.axvline(mean, color=RED, linewidth=1.2)
    ax.text(
        0.97,
        0.95,
        f"mean = {mean:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color=RED,
    )
    style_axes(ax)
    ax.set_xlabel("Lap duration $\\tau$", fontsize=7, color=INK_2)
    ax.set_ylabel("Density", fontsize=7, color=INK_2)
    ax.set_title(f"{len(durations)} laps", fontsize=8, color=INK, loc="left")

    handles = [
        plt.Line2D(
            [],
            [],
            color=ORANGE,
            marker="o",
            markersize=3,
            linestyle="",
            label="wing crossing ($x = 0$)",
        ),
        plt.Line2D(
            [],
            [],
            color=BLUE,
            marker="o",
            markersize=3,
            linestyle="",
            label="completed lap around a lobe center",
        ),
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=2, fontsize=7, frameon=False
    )
    return fig


def lorenz_lap_figure(save_path=OUT / "lorenz_laps.pdf", preview_dir=None):
    """Compute the lap statistics, save the thesis figure and return the stats."""
    trajectory = lorenz_trajectory()
    crossings = find_wing_crossings(trajectory)
    lap_indices, lap_durations = detect_lobe_laps(
        trajectory, crossings, min_segment_steps=5
    )

    fig = plot_lorenz_with_crossings(
        trajectory, crossings, lap_indices, lap_durations, LAP_DT
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    if preview_dir is not None:
        Path(preview_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(preview_dir) / f"{save_path.stem}.png", dpi=150)
    plt.close(fig)

    return {
        "dt": LAP_DT,
        "steps": LAP_STEPS,
        "wing_residence": duration_stats(np.diff(crossings), LAP_DT),
        "lap_duration": duration_stats(lap_durations, LAP_DT),
    }


def main():
    apply_rc()
    stats = lorenz_lap_figure(OUT / "lorenz_laps.pdf", OUT / "preview")
    for name, key in (
        ("Wing residence time", "wing_residence"),
        ("Lobe lap duration", "lap_duration"),
    ):
        s = stats[key]
        print(
            f"\n{name}: n={s['count']}, mean={s['mean']:.4f}, std={s['std']:.4f}, "
            f"median={s['median']:.4f}, P05-P95={s['p05']:.3f}-{s['p95']:.3f}, "
            f"min-max={s['min']:.3f}-{s['max']:.3f}"
        )
    print(f"\nSaved {OUT / 'lorenz_laps.pdf'}")


if __name__ == "__main__":
    main()
