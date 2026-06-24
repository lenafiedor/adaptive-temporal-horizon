import sys
import numpy as np

from adaptive_horizon.analysis import compute_lyapunov


def test_global_cli_resolves_system_before_computing(monkeypatch, capsys):
    seen = {}

    def fake_compute_global_lyapunov(dt, steps, system):
        seen["dt"] = dt
        seen["steps"] = steps
        seen["system_name"] = system.name
        return 0.1959

    monkeypatch.setattr(
        compute_lyapunov,
        "compute_global_lyapunov",
        fake_compute_global_lyapunov,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute-lyapunov",
            "--system",
            "rossler",
            "--mode",
            "global",
            "--dt",
            "0.01",
            "--steps",
            "16",
        ],
    )

    compute_lyapunov.main()

    assert seen == {
        "dt": 0.01,
        "steps": 16,
        "system_name": "rossler",
    }
    assert "Rossler largest Lyapunov Exponent: 0.1959" in capsys.readouterr().out


def test_local_cli_resolves_system_for_trajectory_and_lle(monkeypatch, capsys):
    seen = {}

    def fake_simulate_trajectory(
        system,
        initial_state,
        dt,
        steps,
        burn_in,
    ):
        seen["trajectory_system_name"] = system.name
        seen["initial_state_shape"] = initial_state.shape
        seen["trajectory_dt"] = dt
        seen["trajectory_steps"] = steps
        seen["burn_in"] = burn_in
        return np.zeros((9, system.dim), dtype=np.float64)

    def fake_compute_local_lyapunov(
        trajectory,
        dt,
        system,
    ):
        seen["lle_system_name"] = system.name
        seen["lle_dt"] = dt
        return np.array([[1.0, 0.0, -1.0], [3.0, 2.0, -3.0]])

    monkeypatch.setattr(compute_lyapunov, "resolve_burn_in_steps", lambda dt: 5)
    monkeypatch.setattr(
        compute_lyapunov,
        "simulate_trajectory",
        fake_simulate_trajectory,
    )
    monkeypatch.setattr(
        compute_lyapunov,
        "compute_local_lyapunov",
        fake_compute_local_lyapunov,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute-lyapunov",
            "--system",
            "rossler",
            "--mode",
            "local",
            "--dt",
            "0.01",
            "--steps",
            "8",
        ],
    )

    compute_lyapunov.main()

    assert seen == {
        "trajectory_system_name": "rossler",
        "initial_state_shape": (3,),
        "trajectory_dt": 0.01,
        "trajectory_steps": 8,
        "burn_in": 5,
        "lle_system_name": "rossler",
        "lle_dt": 0.01,
    }
    output = capsys.readouterr().out
    assert "Burn-in: 5 steps" in output
    assert "Mean LLEs: 2.0000, 1.0000, -2.0000" in output
    assert "Std LLEs: 1.0000, 1.0000, 1.0000" in output


def test_local_cli_passes_system_specific_save_dir_to_plots(monkeypatch, tmp_path):
    plot_calls = []

    monkeypatch.setattr(compute_lyapunov, "resolve_burn_in_steps", lambda dt: 0)
    monkeypatch.setattr(
        compute_lyapunov,
        "simulate_trajectory",
        lambda system, initial_state, dt, steps, burn_in: np.zeros((9, system.dim)),
    )
    monkeypatch.setattr(
        compute_lyapunov,
        "compute_local_lyapunov",
        lambda trajectory, dt, system: np.ones((2, system.dim)),
    )
    monkeypatch.setattr(
        compute_lyapunov.config,
        "ANALYSIS_DIR",
        tmp_path / "experiments" / "lorenz" / "analysis",
    )
    monkeypatch.setattr(
        compute_lyapunov,
        "plot_lyapunov_exponents",
        lambda lles, system_name, save_dir: plot_calls.append(
            ("hist", system_name, save_dir)
        ),
    )
    monkeypatch.setattr(
        compute_lyapunov,
        "plot_lle_heatmap",
        lambda trajectory, lles, system_name, save_dir: plot_calls.append(
            ("heatmap", system_name, save_dir)
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute-lyapunov",
            "--system",
            "rossler",
            "--mode",
            "local",
            "--plot",
            "--steps",
            "8",
        ],
    )

    compute_lyapunov.main()

    expected_save_dir = tmp_path / "experiments" / "rossler" / "analysis"
    assert plot_calls == [
        ("hist", "Rossler", expected_save_dir),
        ("heatmap", "Rossler", expected_save_dir),
    ]
