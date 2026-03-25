import numpy as np
import matplotlib.pyplot as plt

from data.lorenz import simulate_lorenz

WINDOW_SIZE = 1000


def jacobian_lorenz(x, y, z, sigma=10, rho=28, beta=8/3):
    return [[-sigma, sigma, 0], [rho-z, -1, -x], [y, x, -beta]]

def compute_lle(trajectory, dt=0.01, window=10):
    """
    Args:
        trajectory: (N, 3) array of states
        dt: time step
        window: number of steps to compute local exponent over
    Returns:
        lles: (N - window,) array of local Lyapunov exponents
    """

    N = len(trajectory)
    lles = []

    for i in range(N - window):
        Q = np.eye(3)
        sum_log = 0

        for j in range(window):
            J = jacobian_lorenz(*trajectory[i + j])
            Q = Q + dt * (J @ Q)

            Q, R = np.linalg.qr(Q)
            sum_log += np.log(np.abs(np.diag(R)))

        lles.append(np.max(sum_log / (window * dt)))

    return np.array(lles)


if __name__ == "__main__":
    trajectory = simulate_lorenz(
        initial_state=[1.0, 1.0, 1.0],
        dt=0.01,
        steps=10000
    )

    lles = compute_lle(trajectory, window=WINDOW_SIZE)

    print(f"Mean LLE: {np.mean(lles):.4f}")
    print(f"Std LLE: {np.std(lles):.4f}")
    print(f"Min LLE: {np.min(lles):.4f}")
    print(f"Max LLE: {np.max(lles):.4f}")

    plt.figure(figsize=(12, 4))
    plt.plot(lles, linewidth=0.5)
    plt.axhline(y=np.mean(lles), color='r', linestyle='--', label=f'Mean: {np.mean(lles):.3f}')
    plt.xlabel("Time step")
    plt.ylabel("Local Lyapunov Exponent")
    plt.title(f"Local Lyapunov Exponents along Lorenz Trajectory (window={WINDOW_SIZE})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
