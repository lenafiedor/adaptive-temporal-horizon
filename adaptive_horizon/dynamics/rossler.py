import numpy as np

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
    _y,
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
