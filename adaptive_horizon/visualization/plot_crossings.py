import numpy as np
import matplotlib.pyplot as plt
import adaptive_horizon.config as config
from adaptive_horizon.dynamics.lorenz import simulate_lorenz


def plot_lorenz_with_crossings(traj, cross):
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.5, alpha=0.7)
    ax1.scatter(
        traj[cross, 0],
        traj[cross, 1],
        traj[cross, 2],
        c="red",
        s=20,
        zorder=5,
        label="x=0 crossings",
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("Lorenz Attractor with Wing Crossings")
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(traj[:, 0], lw=0.5, alpha=0.7)
    ax2.scatter(cross, traj[cross, 0], c="red", s=20, zorder=5)
    ax2.axhline(y=0, color="gray", linestyle="--", lw=0.5)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("X")
    ax2.set_title("X coordinate with crossings")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dt = 0.01
    burn_in = config.resolve_burn_in_steps(dt)
    trajectory = np.array(simulate_lorenz(dt=dt, burn_in=burn_in))

    x = trajectory[:, 0]
    crossings = np.where(np.diff(np.sign(x)))[0]
    periods = np.diff(crossings)

    print(f"Average half-period (one wing): {np.mean(periods):.1f} timesteps")
    print(f"Std: {np.std(periods):.1f}")
    print(f"Range: {periods.min()} - {periods.max()}")

    plot_lorenz_with_crossings(trajectory, crossings)
