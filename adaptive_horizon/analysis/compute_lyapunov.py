import argparse
import numpy as np

from adaptive_horizon.data.utils import simulate_trajectory
from adaptive_horizon.dynamics.lyapunov import (
    compute_global_lyapunov,
    compute_local_lyapunov,
)
from adaptive_horizon.dynamics.systems import SYSTEM_CHOICES, get_system
from adaptive_horizon.visualization.plotting import (
    plot_lyapunov_exponents,
    plot_lle_heatmap,
)
from adaptive_horizon.training.utils import resolve_burn_in_steps
import adaptive_horizon.config as config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="global",
        choices=["global", "local"],
        help="LLE computation mode (global/local)",
    )
    parser.add_argument("--plot", "-p", action="store_true", help="Plot the trajectory")
    parser.add_argument(
        "--system",
        choices=SYSTEM_CHOICES,
        default=config.DEFAULT_SYSTEM,
        help="Dynamical system to analyze",
    )
    parser.add_argument("--dt", type=float, default=config.DT, help="Simulation step")
    parser.add_argument(
        "--steps",
        type=int,
        default=config.SIMULATION_STEPS,
        help="Trajectory length",
    )
    args = parser.parse_args()
    system = get_system(args.system)

    if args.mode == "global":
        lle = compute_global_lyapunov(dt=args.dt, steps=args.steps, system=system)
        print(f"{system.label} largest Lyapunov Exponent: {lle:.4f}")

    elif args.mode == "local":
        burn_in = resolve_burn_in_steps(args.dt)
        rng = np.random.default_rng(config.EVAL_SEED)
        trajectory = np.array(
            simulate_trajectory(
                system,
                initial_state=system.sample_initial_state(rng),
                dt=args.dt,
                steps=args.steps,
                burn_in=burn_in,
            )
        )
        print(f"Burn-in: {burn_in} steps ({config.BURN_IN_TIME:g} time units)")

        lles = compute_local_lyapunov(
            trajectory,
            dt=args.dt,
            system=system,
        )

        print(
            "Mean LLEs: " + ", ".join(f"{value:.4f}" for value in np.mean(lles, axis=0))
        )
        print(
            "Std LLEs: " + ", ".join(f"{value:.4f}" for value in np.std(lles, axis=0))
        )

        if args.plot:
            save_dir = config.system_path(config.ANALYSIS_DIR, system.name)
            plot_lyapunov_exponents(lles, system_name=system.label, save_dir=save_dir)
            plot_lle_heatmap(
                trajectory,
                lles,
                system_name=system.label,
                save_dir=save_dir,
            )

    else:
        raise ValueError("Invalid mode. Choose 'global' or 'local'.")


if __name__ == "__main__":
    main()
