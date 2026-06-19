import argparse
import numpy as np

from adaptive_horizon.dynamics.lorenz import simulate_lorenz
from adaptive_horizon.dynamics.lyapunov import (
    compute_global_lyapunov,
    compute_local_lyapunov,
)
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
    parser.add_argument(
        "--plot", "-p", action="store_true", help="Plot the Lorenz trajectory"
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation step")
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Trajectory length",
    )
    args = parser.parse_args()

    if args.mode == "global":
        lle = compute_global_lyapunov(dt=args.dt)
        print(f"Largest Lyapunov Exponent: {lle:.4f}")

    elif args.mode == "local":
        burn_in = resolve_burn_in_steps(args.dt)
        lorenz_trajectory = np.array(
            simulate_lorenz(
                dt=args.dt,
                steps=args.steps,
                burn_in=burn_in,
            )
        )
        print(f"Burn-in: {burn_in} steps ({config.BURN_IN_TIME:g} time units)")

        lles = compute_local_lyapunov(lorenz_trajectory, dt=args.dt)

        print(f"Mean 1st LLE: {np.mean(lles[:, 0]):.4f}")
        print(f"Std 1st LLE: {np.std(lles[:, 0]):.4f}")

        if args.plot:
            plot_lyapunov_exponents(lles)
            plot_lle_heatmap(lorenz_trajectory, lles)

    else:
        raise ValueError("Invalid mode. Choose 'global' or 'local'.")


if __name__ == "__main__":
    main()
