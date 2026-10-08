import numpy as np

from adaptive_horizon import config
from adaptive_horizon.config import DEFAULT_SYSTEM, FTLE_WINDOW, SIMULATION_STEPS
from adaptive_horizon.dynamics.integrators import (
    resolve_integration_substeps,
    rk4_step_coupled,
)
from adaptive_horizon.dynamics.systems import get_system
from adaptive_horizon.training.utils import resolve_burn_in_steps


def mean_horizon(lambda_max, base_T, min_T, max_T, mean=None, std=None):
    """Map standardized Lyapunov exponents to horizons around the base horizon."""
    lambda_mean = float(np.mean(lambda_max)) if mean is None else mean
    lambda_std = (float(np.std(lambda_max)) if std is None else std) + 1e-8
    z_scores = (lambda_max - lambda_mean) / lambda_std

    half_range = max(1.0, (max_T - min_T) / 2.0)
    T = base_T - z_scores * half_range
    T = np.clip(np.round(T), min_T, max_T)
    return T.astype(int)


def time_horizon(
    lambda_max,
    dt,
    min_T=1,
    max_T=config.MAX_TRAIN_T,
):
    """Map Lyapunov exponents to one Lyapunov time in rollout steps."""
    lambda_max = np.asarray(lambda_max)
    T = np.full(lambda_max.shape, max_T, dtype=float)
    expanding = lambda_max > 0
    T[expanding] = np.floor(1.0 / (lambda_max[expanding] * dt))
    T = np.clip(T, min_T, max_T).astype(int)
    return int(T) if T.ndim == 0 else T


def compute_global_lyapunov(
    dt=0.01,
    steps=SIMULATION_STEPS,
    system=DEFAULT_SYSTEM,
):
    """
    Compute the largest Lyapunov exponent using the RK4-consistent QR method.
    """
    system = get_system(system)
    x = system.sample_initial_state(np.random.default_rng(config.RANDOM_SEED))
    Q = np.eye(system.dim)
    sum_log = 0.0
    burn_in = resolve_burn_in_steps(dt)

    for _ in range(burn_in):
        x, Q = rk4_step_coupled(x, Q, dt, system.rhs, system.jacobian)
        Q, R = np.linalg.qr(Q)

    for _ in range(steps):
        x, Q = rk4_step_coupled(x, Q, dt, system.rhs, system.jacobian)
        Q, R = np.linalg.qr(Q)
        sum_log += np.log(np.abs(R[0, 0]) + 1e-12)

    lyap = sum_log / (steps * dt)
    return lyap


def compute_local_lyapunov(
    trajectory,
    dt=0.01,
    system=config.DEFAULT_SYSTEM,
    integration_dt=None,
):
    """
    Compute local Lyapunov exponents using RK4-consistent tangent space evolution.

    Args:
        trajectory (array [N, 3]): array of states
        dt (float): time step
        system (str): name of the dynamical system
    Returns:
        LLEs (array [N - 1, 3]): array of local Lyapunov exponents
    """
    system = get_system(system)
    integration_dt = dt if integration_dt is None else integration_dt
    substeps = resolve_integration_substeps(dt, integration_dt)
    trajectory = np.array(trajectory)
    Q = np.eye(system.dim)
    lles = []

    for i in range(len(trajectory) - 1):
        x = trajectory[i]
        for _ in range(substeps):
            x, Q = rk4_step_coupled(x, Q, integration_dt, system.rhs, system.jacobian)
        Q, R = np.linalg.qr(Q)
        lles.append(np.log(np.abs(np.diag(R)) + 1e-12) / dt)

    return np.array(lles)


def compute_forward_ftle(
    trajectory,
    dt=0.01,
    window=FTLE_WINDOW,
    system=config.DEFAULT_SYSTEM,
    integration_dt=None,
):
    """
    Compute an aligned forward finite-time Lyapunov score for each start state.

    The returned value at index i estimates the largest tangent-space expansion
    over trajectory[i : i + window + 1], so it can be used directly as a
    training-time predictability score for samples that start at i.

    Args:
        trajectory (array [N, 3]): states along a trajectory
        dt (float): time step
        window (int): number of forward integration steps
        system (str): name of the dynamical system
    Returns:
        array [N - window]: largest forward FTLE for each valid start index
    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")

    system = get_system(system)
    integration_dt = dt if integration_dt is None else integration_dt
    substeps = resolve_integration_substeps(dt, integration_dt)
    trajectory = np.array(trajectory)
    ftles = []

    for start in range(len(trajectory) - window):
        tangent_map = np.eye(system.dim)
        for offset in range(window):
            x = trajectory[start + offset]
            for _ in range(substeps):
                x, tangent_map = rk4_step_coupled(
                    x,
                    tangent_map,
                    integration_dt,
                    system.rhs,
                    system.jacobian,
                )

        sigma_max = float(
            np.linalg.norm(np.asarray(tangent_map, dtype=np.float64), ord=2)
        )
        ftles.append(np.log(sigma_max + 1e-12) / (window * dt))

    return np.array(ftles)
