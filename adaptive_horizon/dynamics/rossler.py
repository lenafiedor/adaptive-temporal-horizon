import numpy as np

from adaptive_horizon.dynamics.integrators import rk4_step

ROSSLER_PARAMETERS = {"a": 0.37, "b": 0.2, "c": 5.7}


def rossler_f(
    x, a=ROSSLER_PARAMETERS["a"], b=ROSSLER_PARAMETERS["b"], c=ROSSLER_PARAMETERS["c"]
):

    return np.array(
        [
            -x[1] - x[2],
            x[0] + a * x[1],
            b + x[2] * (x[0] - c),
        ],
        dtype=np.float64,
    )


def jacobian_rossler(
    x,
    y,
    z,
    a=ROSSLER_PARAMETERS["a"],
    c=ROSSLER_PARAMETERS["c"],
):
    return np.array(
        [
            [0, -1, -1],
            [1, a, 0],
            [z, 0, x - c],
        ],
        dtype=np.float64,
    )


def sample_rossler_initial_state(rng=None):
    """Sample an initial state near the standard Rossler attractor."""
    uniform = np.random.uniform if rng is None else rng.uniform
    return np.array(
        [
            uniform(-5, 5),
            uniform(-5, 5),
            uniform(0, 10),
        ],
        dtype=np.float64,
    )


def simulate_rossler(
    initial_state=None,
    dt=0.01,
    steps=10000,
    burn_in=0,
    a=ROSSLER_PARAMETERS["a"],
    b=ROSSLER_PARAMETERS["b"],
    c=ROSSLER_PARAMETERS["c"],
):
    """Simulate a Rossler trajectory and discard an optional transient burn-in."""
    if initial_state is None:
        initial_state = np.array([1.0, 1.0, 1.0])

    states = [initial_state]
    current_state = np.array(initial_state)

    for _ in range(steps + burn_in):
        current_state = rk4_step(rossler_f, current_state, dt, a, b, c)
        states.append(current_state)

    return states[burn_in:]


if __name__ == "__main__":
    simulate_rossler()
