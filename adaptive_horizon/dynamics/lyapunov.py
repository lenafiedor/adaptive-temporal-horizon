import numpy as np

from adaptive_horizon.dynamics.integrators import rk4_step_coupled
from adaptive_horizon.dynamics.lorenz import lorenz_f, jacobian_lorenz
from adaptive_horizon.config import WINDOW_SIZE, resolve_burn_in_steps


def compute_global_lyapunov(dt=0.01, steps=10000, burn_in=None):
    """
    Compute the largest Lyapunov exponent using the RK4-consistent QR method.
    """
    x = np.array([1.0, 1.0, 1.0])
    Q = np.eye(3)
    sum_log = 0.0
    burn_in = resolve_burn_in_steps(dt, burn_in)

    for _ in range(burn_in):
        x, Q = rk4_step_coupled(x, Q, dt, lorenz_f, jacobian_lorenz)
        Q, R = np.linalg.qr(Q)

    for _ in range(steps):
        x, Q = rk4_step_coupled(x, Q, dt, lorenz_f, jacobian_lorenz)
        Q, R = np.linalg.qr(Q)
        sum_log += np.log(np.abs(R[0, 0]) + 1e-12)

    lyap = sum_log / (steps * dt)
    return lyap


def compute_local_lyapunov(trajectory, burn_in=None, dt=0.01):
    """
    Compute local Lyapunov exponents using RK4-consistent tangent space evolution.

    Args:
        trajectory (array [N, 3]): array of states
        burn_in (int): number of initial steps to ignore
        dt (float): time step
    Returns:
        LLEs (array [N - 1, 3]): array of local Lyapunov exponents
    """
    trajectory = np.array(trajectory)
    N = len(trajectory)
    Q = np.eye(3)
    lles = []
    burn_in = resolve_burn_in_steps(dt, burn_in)

    for i in range(N - 1):
        x = trajectory[i]
        x_next, Q = rk4_step_coupled(x, Q, dt, lorenz_f, jacobian_lorenz)
        Q, R = np.linalg.qr(Q)

        if i >= burn_in:
            lles.append(np.log(np.abs(np.diag(R)) + 1e-12) / dt)

    return np.array(lles)


def compute_forward_ftle(trajectory, dt=0.01, window=WINDOW_SIZE):
    """
    Compute an aligned forward finite-time Lyapunov score for each start state.

    The returned value at index i estimates the largest tangent-space expansion
    over trajectory[i : i + window + 1], so it can be used directly as a
    training-time predictability score for samples that start at i.

    Args:
        trajectory (array [N, 3]): states along a trajectory
        dt (float): time step
        window (int): number of forward integration steps
    Returns:
        array [N - window]: largest forward FTLE for each valid start index
    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")

    trajectory = np.array(trajectory)
    scores = []

    for start in range(len(trajectory) - window):
        tangent_map = np.eye(3)
        for offset in range(window):
            x = trajectory[start + offset]
            _, tangent_map = rk4_step_coupled(
                x, tangent_map, dt, lorenz_f, jacobian_lorenz
            )

        sigma_max = np.linalg.svd(tangent_map, compute_uv=False)[0]
        scores.append(np.log(sigma_max + 1e-12) / (window * dt))

    return np.array(scores)


def smooth_lle(lles, window):
    smoothed = []
    for i in range(len(lles) - window):
        smoothed.append(np.mean(lles[i : i + window], axis=0))
    return np.array(smoothed)
