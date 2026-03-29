import numpy as np

from adaptive_horizon.dynamics.integrators import rk4_step_coupled_lorenz
from adaptive_horizon.dynamics.lorenz import jacobian_lorenz


def compute_global_lyapunov(dt=0.01, steps=50000, burn_in=2000):
    """
    Compute largest Lyapunov exponent using RK4-consistent QR method.
    """
    x = np.array([1.0, 1.0, 1.0])
    Q = np.eye(3)
    sum_log = 0.0

    for _ in range(burn_in):
        x, Q = rk4_step_coupled_lorenz(x, Q, dt)
        Q, R = np.linalg.qr(Q)

    for _ in range(steps):
        x, Q = rk4_step_coupled_lorenz(x, Q, dt)
        Q, R = np.linalg.qr(Q)
        sum_log += np.log(np.abs(R[0, 0]) + 1e-12)

    lyap = sum_log / (steps * dt)
    return lyap


def compute_local_lyapunov(trajectory, dt=0.01, window=10):
    """
    Args:
        trajectory (array [N, 3]): array of states
        dt (float): time step
        window (int): number of steps to compute local exponent over
    Returns:
        LLEs (array [N - window,]): array of local Lyapunov exponents
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
