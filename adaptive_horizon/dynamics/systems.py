from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from adaptive_horizon.dynamics.lorenz import (
    LORENZ_PARAMETERS,
    jacobian_lorenz,
    lorenz_f,
    sample_lorenz_initial_state,
)
from adaptive_horizon.dynamics.rossler import (
    ROSSLER_PARAMETERS,
    jacobian_rossler,
    rossler_f,
    sample_rossler_initial_state,
)
from adaptive_horizon.dynamics.lorenz96 import make_lorenz96_functions
from adaptive_horizon import config


@dataclass(frozen=True)
class DynamicsSystem:
    name: str
    dim: int
    rhs: Callable[[np.ndarray], np.ndarray]
    jacobian: Callable[..., np.ndarray]
    sample_initial_state: Callable[..., np.ndarray]
    parameters: Mapping[str, float] = field(default_factory=dict)
    display_name: str | None = None

    @property
    def label(self):
        return self.display_name or self.name.capitalize()


SYSTEMS = {
    "lorenz": DynamicsSystem(
        name="lorenz",
        display_name="Lorenz",
        dim=3,
        rhs=lorenz_f,
        jacobian=jacobian_lorenz,
        sample_initial_state=sample_lorenz_initial_state,
        parameters=LORENZ_PARAMETERS,
    ),
    "rossler": DynamicsSystem(
        name="rossler",
        display_name="Rossler",
        dim=3,
        rhs=rossler_f,
        jacobian=jacobian_rossler,
        sample_initial_state=sample_rossler_initial_state,
        parameters=ROSSLER_PARAMETERS,
    ),
}

lorenz96_rhs, lorenz96_jacobian, lorenz96_sample_initial_state = (
    make_lorenz96_functions(
        config.LORENZ96_DIMENSION,
        config.LORENZ96_FORCING,
    )
)
SYSTEMS["lorenz96"] = DynamicsSystem(
    name="lorenz96",
    display_name="Lorenz-96",
    dim=config.LORENZ96_DIMENSION,
    rhs=lorenz96_rhs,
    jacobian=lorenz96_jacobian,
    sample_initial_state=lorenz96_sample_initial_state,
    parameters={
        "dimension": config.LORENZ96_DIMENSION,
        "forcing": config.LORENZ96_FORCING,
    },
)

SYSTEM_CHOICES = tuple(sorted(SYSTEMS))


def get_system(system: str | DynamicsSystem):
    if isinstance(system, DynamicsSystem):
        return system

    key = system.lower()
    try:
        return SYSTEMS[key]
    except KeyError as exc:
        choices = ", ".join(SYSTEM_CHOICES)
        raise ValueError(
            f"Unknown dynamical system {system!r}; choose one of: {choices}"
        ) from exc
