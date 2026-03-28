import numpy as np

from src.numerics.integrators import rk4_step_coupled_lorenz


def compute_global_lyapunov(dt=0.01, steps=50000, burn_in=2000):
    """
    Compute largest Lyapunov exponent using RK4-consistent QR method.
    """

    x = np.array([1.0, 1.0, 1.0])
    Q = np.eye(3)
    sum_log = 0.0

    for _ in range(burn_in):
        x, Q = rk4_step_coupled_lorenz(x, Q, dt)
        Q, R = np.linalg.qr(Q)  # keep it stable during burn-in

    for _ in range(steps):
        x, Q = rk4_step_coupled_lorenz(x, Q, dt)
        Q, R = np.linalg.qr(Q)  # QR decomposition
        sum_log += np.log(np.abs(R[0, 0]) + 1e-12)  # accumulate largest exponent

    lyap = sum_log / (steps * dt)
    return lyap
