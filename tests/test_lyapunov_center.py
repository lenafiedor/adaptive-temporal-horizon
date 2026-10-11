import math

import pytest

from adaptive_horizon.dynamics.lyapunov import lyapunov_center_horizon


@pytest.mark.parametrize(
    "lambda_estimate, dt, expected",
    [
        (0.907, 0.05, 10),  # Lorenz: 0.45 / (0.907 * 0.05) = 9.92
        (0.907, 0.08, 6),  # 6.20
        (0.907, 0.1, 5),  # 4.96
        (1.197, 0.05, 8),  # Lorenz96 (N=10, F=8): 7.52
        (1.197, 0.1, 4),  # 3.76
    ],
)
def test_center_covers_fraction_of_lyapunov_time(lambda_estimate, dt, expected):
    assert lyapunov_center_horizon(lambda_estimate, dt, 0.45, max_T=50) == expected


def test_center_is_clipped_to_cap():
    # Rossler (a=0.37): 0.45 / (0.197 * 0.08) = 28.6 steps, above a cap of 10
    assert lyapunov_center_horizon(0.197, 0.08, 0.45, max_T=10) == 10
    assert lyapunov_center_horizon(0.197, 0.08, 0.45, max_T=50) == 29


def test_center_is_at_least_min_T():
    assert lyapunov_center_horizon(100.0, 0.1, 0.45, max_T=10) == 1
    assert lyapunov_center_horizon(100.0, 0.1, 0.45, max_T=10, min_T=2) == 2


@pytest.mark.parametrize("lambda_estimate", [0.0, -0.3])
def test_no_exponential_growth_returns_cap(lambda_estimate):
    assert lyapunov_center_horizon(lambda_estimate, 0.08, 0.45, max_T=10) == 10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0, "fraction": 0.45, "max_T": 10},
        {"dt": 0.08, "fraction": 0.0, "max_T": 10},
        {"dt": 0.08, "fraction": 0.45, "max_T": 1, "min_T": 2},
    ],
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        lyapunov_center_horizon(0.9, **kwargs)


@pytest.mark.parametrize("lambda_estimate", [math.nan, math.inf])
def test_non_finite_lambda_raises(lambda_estimate):
    with pytest.raises(ValueError):
        lyapunov_center_horizon(lambda_estimate, 0.08, 0.45, max_T=10)
