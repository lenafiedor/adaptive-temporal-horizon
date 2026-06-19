import numpy as np
import matplotlib.pyplot as plt

from adaptive_horizon.dynamics.lorenz import simulate_lorenz
from adaptive_horizon.training.utils import resolve_burn_in_steps


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

    for start, end in zip(boundaries[:-1], boundaries[1:]):
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


def summarize_duration(name, durations, dt):
    durations = np.asarray(durations, dtype=np.float64)
    if len(durations) == 0:
        print(f"\n{name}: no durations found")
        return

    duration_times = durations * dt
    p05, p25, median, p75, p95 = np.percentile(duration_times, [5, 25, 50, 75, 95])
    steps_p05, steps_p25, steps_median, steps_p75, steps_p95 = np.percentile(
        durations, [5, 25, 50, 75, 95]
    )

    print(f"\n{name}")
    print(f"Count: {len(durations)}")
    print(f"Mean: {np.mean(durations):.1f} steps ({np.mean(duration_times):.4f} time)")
    print(f"Std: {np.std(durations):.1f} steps ({np.std(duration_times):.4f} time)")
    print(f"Median: {steps_median:.1f} steps ({median:.4f} time)")
    print(
        f"P25 / P75: {steps_p25:.1f} / {steps_p75:.1f} steps "
        f"({p25:.4f} / {p75:.4f} time)"
    )
    print(
        f"P05 / P95: {steps_p05:.1f} / {steps_p95:.1f} steps "
        f"({p05:.4f} / {p95:.4f} time)"
    )
    print(
        f"Min / Max: {np.min(durations):.0f} / {np.max(durations):.0f} steps "
        f"({np.min(duration_times):.4f} / {np.max(duration_times):.4f} time)"
    )


def plot_lorenz_with_crossings(traj, crossings, lap_indices):
    fig = plt.figure(figsize=(13, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.5, alpha=0.7)
    ax1.scatter(
        traj[crossings, 0],
        traj[crossings, 1],
        traj[crossings, 2],
        c="red",
        s=20,
        zorder=5,
        label="x=0 crossings",
    )
    if len(lap_indices) > 0:
        ax1.scatter(
            traj[lap_indices, 0],
            traj[lap_indices, 1],
            traj[lap_indices, 2],
            c="gold",
            edgecolors="black",
            linewidths=0.4,
            s=28,
            zorder=6,
            label="lobe laps",
        )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("Lorenz Attractor with Wing Crossings and Laps")
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(traj[:, 0], lw=0.5, alpha=0.7)
    ax2.scatter(crossings, traj[crossings, 0], c="red", s=20, zorder=5)
    if len(lap_indices) > 0:
        ax2.scatter(
            lap_indices,
            traj[lap_indices, 0],
            c="gold",
            edgecolors="black",
            linewidths=0.4,
            s=28,
            zorder=6,
        )
    ax2.axhline(y=0, color="gray", linestyle="--", lw=0.5)
    ax2.vlines(
        crossings,
        ymin=traj[:, 0].min(),
        ymax=traj[:, 0].max(),
        color="red",
        alpha=0.08,
    )
    if len(lap_indices) > 0:
        ax2.vlines(
            lap_indices,
            ymin=traj[:, 0].min(),
            ymax=traj[:, 0].max(),
            color="gold",
            alpha=0.12,
        )
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("X")
    ax2.set_title("X coordinate with crossings and lap completions")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dt = 0.01
    burn_in = resolve_burn_in_steps(dt)
    trajectory = np.array(simulate_lorenz(dt=dt, burn_in=burn_in))

    crossings = find_wing_crossings(trajectory)
    wing_residence_steps = np.diff(crossings)
    lap_indices, lap_duration_steps = detect_lobe_laps(
        trajectory, crossings, min_segment_steps=5
    )

    summarize_duration("Wing residence time", wing_residence_steps, dt)
    summarize_duration("Lobe lap duration", lap_duration_steps, dt)

    plot_lorenz_with_crossings(trajectory, crossings, lap_indices)
