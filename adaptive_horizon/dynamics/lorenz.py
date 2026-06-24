import numpy as np

from adaptive_horizon.dynamics.integrators import rk4_step

LORENZ_PARAMETERS = {"sigma": 10, "rho": 28, "beta": 8 / 3}


def lorenz_f(x, sigma=LORENZ_PARAMETERS["sigma"], rho=LORENZ_PARAMETERS["rho"], beta=LORENZ_PARAMETERS["beta"]):
    return np.array(
        [sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]]
    )


def jacobian_lorenz(x, y, z, sigma=LORENZ_PARAMETERS["sigma"], rho=LORENZ_PARAMETERS["rho"], beta=LORENZ_PARAMETERS["beta"]):
    return np.array([[-sigma, sigma, 0], [rho - z, -1, -x], [y, x, -beta]])


def sample_lorenz_initial_state(rng=None):
    """Sample an initial state from the repository's Lorenz initialization box."""
    uniform = np.random.uniform if rng is None else rng.uniform
    return np.array(
        [
            uniform(-20, 20),
            uniform(-20, 20),
            uniform(0, 50),
        ],
        dtype=np.float64,
    )


def simulate_lorenz(
    initial_state=None,
    dt=0.01,
    steps=10000,
    burn_in=0,
    sigma=LORENZ_PARAMETERS["sigma"],
    rho=LORENZ_PARAMETERS["rho"],
    beta=LORENZ_PARAMETERS["beta"],
):
    """Simulate a Lorenz trajectory and discard an optional transient burn-in."""
    if initial_state is None:
        initial_state = np.array([1.0, 1.0, 1.0])

    states = [initial_state]
    current_state = np.array(initial_state)

    for _ in range(steps + burn_in):
        current_state = rk4_step(lorenz_f, current_state, dt, sigma, rho, beta)
        states.append(current_state)

    return states[burn_in:]
