import numpy as np

from src.model.lorenz import jacobian_lorenz


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

        lles.append(np.max(sum_log / (window * dt)))  # take max exponent

    return np.array(lles)
