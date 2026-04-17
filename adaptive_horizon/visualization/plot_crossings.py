import numpy as np
import matplotlib.pyplot as plt
from adaptive_horizon.dynamics.lorenz import simulate_lorenz


def plot_lorenz_with_crossings(traj, crossings):
    fig = plt.figure(figsize=(12, 5))

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
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("Lorenz Attractor with Wing Crossings")
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(traj[:, 0], lw=0.5, alpha=0.7)
    ax2.scatter(crossings, traj[crossings, 0], c="red", s=20, zorder=5)
    ax2.axhline(y=0, color="gray", linestyle="--", lw=0.5)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("X")
    ax2.set_title("X coordinate with crossings")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    burn_in = 1000
    traj = np.array(
        simulate_lorenz(initial_state=[1.0, 1.0, 1.0], dt=0.01, steps=10000)
    )
    traj = traj[burn_in:]

    x = traj[:, 0]
    crossings = np.where(np.diff(np.sign(x)))[0]
    periods = np.diff(crossings)

    print(f"Average half-period (one wing): {np.mean(periods):.1f} timesteps")
    print(f"Std: {np.std(periods):.1f}")
    print(f"Range: {periods.min()} - {periods.max()}")

    plot_lorenz_with_crossings(traj, crossings)
