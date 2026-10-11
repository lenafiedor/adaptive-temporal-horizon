import numpy as np

from adaptive_horizon import config
from adaptive_horizon.config import (
    DEFAULT_SYSTEM,
    SIMULATION_STEPS,
    resolve_burn_in_steps,
)
from adaptive_horizon.dynamics.integrators import (
    resolve_integration_substeps,
    rk4_step_coupled,
)
from adaptive_horizon.dynamics.systems import get_system


def lyapunov_center_horizon(lambda_estimate, dt, fraction, max_T, min_T=1):
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if fraction <= 0:
        raise ValueError(f"fraction must be positive, got {fraction}")
    if min_T > max_T:
        raise ValueError(f"min_T ({min_T}) must not exceed max_T ({max_T})")

    lambda_estimate = float(lambda_estimate)
    if not np.isfinite(lambda_estimate):
        raise ValueError(f"lambda_estimate must be finite, got {lambda_estimate}")
    if lambda_estimate <= 0:
        return int(max_T)

    T = round(fraction / (lambda_estimate * dt))
    return int(min(max(T, min_T), max_T))


def mean_horizon(lambda_max, base_T, min_T, max_T):
    """Map standardized Lyapunov exponents to horizons around the base horizon."""
    lambda_mean = float(np.mean(lambda_max))
    lambda_std = float(np.std(lambda_max)) + 1e-8
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


def compute_step_tangent_maps(
    trajectory,
    dt=0.01,
    system=config.DEFAULT_SYSTEM,
    integration_dt=None,
):
    """
    One-step tangent maps along a sampled trajectory.

    maps[i] is the Jacobian of the dt-flow evaluated at trajectory[i], i.e. it
    propagates a perturbation from sample i to sample i + 1.

    Returns:
        array [N - 1, dim, dim]
    """
    system = get_system(system)
    integration_dt = dt if integration_dt is None else integration_dt
    substeps = resolve_integration_substeps(dt, integration_dt)
    trajectory = np.array(trajectory)
    maps = np.empty((len(trajectory) - 1, system.dim, system.dim))

    for i in range(len(trajectory) - 1):
        x = trajectory[i]
        tangent_map = np.eye(system.dim)
        for _ in range(substeps):
            x, tangent_map = rk4_step_coupled(
                x, tangent_map, integration_dt, system.rhs, system.jacobian
            )
        maps[i] = tangent_map

    return maps


def compute_local_lyapunov(
    trajectory,
    dt=0.01,
    system=config.DEFAULT_SYSTEM,
    integration_dt=None,
    tangent_maps=None,
):
    """
    Compute local Lyapunov exponents using RK4-consistent tangent space evolution.

    Args:
        trajectory (array [N, 3]): array of states
        dt (float): simulation time step
        system (str): name of the dynamical system
        integration_dt (float): integration time step
        tangent_maps (array [N - 1, d, d], optional): precomputed one-step maps
    Returns:
        LLEs (array [N - 1, 3]): array of local Lyapunov exponents
    """
    if tangent_maps is None:
        tangent_maps = compute_step_tangent_maps(
            trajectory, dt=dt, system=system, integration_dt=integration_dt
        )
    Q = np.eye(tangent_maps.shape[-1])
    lles = []

    for tangent_map in tangent_maps:
        Q, R = np.linalg.qr(tangent_map @ Q)
        lles.append(np.log(np.abs(np.diag(R)) + 1e-12) / dt)

    return np.array(lles)


def compute_forward_ftle(
    trajectory,
    window,
    dt=0.01,
    system=config.DEFAULT_SYSTEM,
    integration_dt=None,
    tangent_maps=None,
):
    """
    Compute an aligned forward finite-time Lyapunov score for each start state.

    The returned value at index i estimates the largest tangent-space expansion
    over trajectory[i : i + window + 1], so it can be used directly as a
    training-time predictability score for samples that start at i.

    Args:
        trajectory (array [N, 3]): states along a trajectory
        dt (float): simulation time step
        window (int): number of forward integration steps
        system (str): name of the dynamical system
        integration_dt (float): integration time step
        tangent_maps (array [N - 1, d, d], optional): precomputed one-step maps
    Returns:
        array [N - window]: largest forward FTLE for each valid start index
    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")

    if tangent_maps is None:
        tangent_maps = compute_step_tangent_maps(
            trajectory, dt=dt, system=system, integration_dt=integration_dt
        )
    num_starts = len(tangent_maps) - window + 1
    if num_starts < 1:
        return np.empty(0)

    products = tangent_maps[:num_starts].copy()
    for offset in range(1, window):
        products = tangent_maps[offset : offset + num_starts] @ products

    sigma_max = np.linalg.svd(products, compute_uv=False)[:, 0]
    return np.log(sigma_max + 1e-12) / (window * dt)
