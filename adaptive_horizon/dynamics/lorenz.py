import numpy as np

from adaptive_horizon.dynamics.integrators import rk4_step


def lorenz_f(x, sigma=10, rho=28, beta=8 / 3):
    return np.array(
        [sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]]
    )


def jacobian_lorenz(x, y, z, sigma=10, rho=28, beta=8 / 3):
    return np.array([[-sigma, sigma, 0], [rho - z, -1, -x], [y, x, -beta]])


def simulate_lorenz(
    initial_state=None, dt=0.01, steps=100000, sigma=10, rho=28, beta=8 / 3
):
    if initial_state is None:
        initial_state = np.array([1.0, 1.0, 1.0])

    states = [initial_state]
    current_state = np.array(initial_state)

    for _ in range(steps):
        current_state = rk4_step(lorenz_f, current_state, dt, sigma, rho, beta)
        states.append(current_state)

    return states
