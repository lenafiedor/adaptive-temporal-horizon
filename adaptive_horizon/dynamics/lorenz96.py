import numpy as np


LORENZ96_PARAMETERS = {
    "dimension": 10,
    "forcing": 8.0,
}


def lorenz96_f(x, forcing=LORENZ96_PARAMETERS["forcing"]):
    """Lorenz-96 vector field with cyclic indices."""
    state = np.asarray(x, dtype=np.float64)
    if state.ndim != 1 or state.size < 4:
        raise ValueError(
            "Lorenz-96 requires a one-dimensional state with at least 4 variables"
        )

    return (
        (np.roll(state, -1) - np.roll(state, 2)) * np.roll(state, 1) - state + forcing
    )


def jacobian_lorenz96(
    *state,
    forcing=LORENZ96_PARAMETERS["forcing"],
):
    """Analytic Lorenz-96 Jacobian for the coupled Lyapunov integrator."""
    del forcing  # The constant forcing has zero derivative.
    x = np.asarray(state, dtype=np.float64)
    if x.ndim != 1 or x.size < 4:
        raise ValueError("Lorenz-96 requires at least 4 state variables")

    dimension = x.size
    jacobian = -np.eye(dimension, dtype=np.float64)
    for index in range(dimension):
        jacobian[index, (index + 1) % dimension] += x[(index - 1) % dimension]
        jacobian[index, (index - 2) % dimension] -= x[(index - 1) % dimension]
        jacobian[index, (index - 1) % dimension] += (
            x[(index + 1) % dimension] - x[(index - 2) % dimension]
        )
    return jacobian


def sample_lorenz96_initial_state(
    rng=None,
    dimension=LORENZ96_PARAMETERS["dimension"],
    forcing=LORENZ96_PARAMETERS["forcing"],
):
    """Sample a small perturbation around the standard uniform state."""
    if dimension < 4:
        raise ValueError("Lorenz-96 requires at least 4 variables")
    generator = np.random.default_rng() if rng is None else rng
    return generator.normal(forcing, 0.01, size=dimension).astype(np.float64)


def make_lorenz96_functions(dimension, forcing=LORENZ96_PARAMETERS["forcing"]):
    """Create dimension-specific RHS, Jacobian, and initial-state functions."""
    if dimension < 4:
        raise ValueError("Lorenz-96 requires at least 4 variables")

    def rhs(state):
        if len(state) != dimension:
            raise ValueError(
                f"Expected a Lorenz-96 state with {dimension} variables, got {len(state)}"
            )
        return lorenz96_f(state, forcing=forcing)

    def jacobian(*state):
        if len(state) != dimension:
            raise ValueError(
                f"Expected a Lorenz-96 state with {dimension} variables, got {len(state)}"
            )
        return jacobian_lorenz96(*state, forcing=forcing)

    def sample_initial_state(rng=None):
        return sample_lorenz96_initial_state(
            rng=rng,
            dimension=dimension,
            forcing=forcing,
        )

    return rhs, jacobian, sample_initial_state
