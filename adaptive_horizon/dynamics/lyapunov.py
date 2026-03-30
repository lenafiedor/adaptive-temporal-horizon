import numpy as np

from adaptive_horizon.dynamics.integrators import rk4_step_coupled
from adaptive_horizon.dynamics.lorenz import lorenz_f, jacobian_lorenz


def compute_global_lyapunov(dt=0.01, steps=50000, burn_in=2000):
    """
    Compute largest Lyapunov exponent using RK4-consistent QR method.
    """
    x = np.array([1.0, 1.0, 1.0])
    Q = np.eye(3)
    sum_log = 0.0

    for _ in range(burn_in):
        x, Q = rk4_step_coupled(x, Q, dt, lorenz_f, jacobian_lorenz)
        Q, R = np.linalg.qr(Q)

    for _ in range(steps):
        x, Q = rk4_step_coupled(x, Q, dt, lorenz_f, jacobian_lorenz)
        Q, R = np.linalg.qr(Q)
        sum_log += np.log(np.abs(R[0, 0]) + 1e-12)

    lyap = sum_log / (steps * dt)
    return lyap


def compute_local_lyapunov(trajectory, dt=0.01, window=10):
    """
    Compute local Lyapunov exponents using RK4-consistent tangent space evolution.
    
    Args:
        trajectory (array [N, 3]): array of states (used only for initial conditions)
        dt (float): time step
        window (int): number of steps to compute local exponent over
    Returns:
        LLEs (array [N - window,]): array of local Lyapunov exponents
    """
    trajectory = np.array(trajectory)
    N = len(trajectory)
    lles = []

    for i in range(N - window):
        x = trajectory[i].copy()
        Q = np.eye(3)
        sum_log = 0.0

        for _ in range(window):
            x, Q = rk4_step_coupled(x, Q, dt, lorenz_f, jacobian_lorenz)
            Q, R = np.linalg.qr(Q)
            sum_log += np.log(np.abs(R[0, 0]) + 1e-12)

        lles.append(sum_log / (window * dt))

    return np.array(lles)
