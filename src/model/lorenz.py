import numpy as np


def lorenz_f(x, sigma=10, rho=28, beta=8/3):
    return np.array([
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2]
    ])

def jacobian_lorenz(x, y, z, sigma=10, rho=28, beta=8/3):
    return np.array([[-sigma, sigma, 0], [rho-z, -1, -x], [y, x, -beta]])


def _rk4_step(f, x, dt, *args):
    k1 = f(x, *args)
    k2 = f(x + 0.5 * dt * k1, *args)
    k3 = f(x + 0.5 * dt * k2, *args)
    k4 = f(x + dt * k3, *args)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_lorenz(initial_state=None, dt=0.01, steps=10000, sigma=10, rho=28, beta=8/3):
    if initial_state is None:
        initial_state = np.array([1.0, 1.0, 1.0])

    states = [initial_state]
    current_state = np.array(initial_state)

    for _ in range(steps):
        current_state = _rk4_step(lorenz_f, current_state, dt, sigma, rho, beta)
        states.append(current_state)

    return states