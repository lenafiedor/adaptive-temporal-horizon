import argparse
import numpy as np

from adaptive_horizon.dynamics.lorenz import simulate_lorenz
from adaptive_horizon.dynamics.lyapunov import (
    compute_global_lyapunov,
    compute_local_lyapunov,
    smooth_lle,
)
from adaptive_horizon.visualization.plotting import (
    plot_lyapunov_exponents,
    plot_trajectory_heatmap,
)
import adaptive_horizon.config as config
from adaptive_horizon.config import WINDOW_SIZE


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
    parser.add_argument(
        "--window",
        "-w",
        type=int,
        default=WINDOW_SIZE,
        help="Window size for LLE computation (only local mode)",
    )
    parser.add_argument("--dt", type=float, default=config.DT, help="Simulation step")
    parser.add_argument(
        "--steps",
        type=int,
        default=config.STEPS_PER_TRAJECTORY,
        help="Trajectory length",
    )
    args = parser.parse_args()
    burn_in = config.resolve_burn_in_steps(args.dt)

    if args.mode == "global":
        lle = compute_global_lyapunov(dt=args.dt, burn_in=burn_in)
        print(f"Largest Lyapunov Exponent: {lle:.4f}")

    elif args.mode == "local":
        lorenz_trajectory = np.array(
            simulate_lorenz(
                dt=args.dt,
                steps=args.steps,
                burn_in=burn_in,
            )
        )
        print(f"Burn-in: {burn_in} steps ({config.BURN_IN_TIME:g} time units)")

        lles = smooth_lle(
            compute_local_lyapunov(lorenz_trajectory, burn_in=burn_in, dt=args.dt),
            window=args.window,
        )

        print(f"Mean 1st LLE: {np.mean(lles[:, 0]):.4f}")
        print(f"Std 1st LLE: {np.std(lles[:, 0]):.4f}")

        if args.plot:
            plot_lyapunov_exponents(lles, window=args.window)
            plot_trajectory_heatmap(lorenz_trajectory, lles, args.window)

    else:
        raise ValueError("Invalid mode. Choose 'global' or 'local'.")


if __name__ == "__main__":
    main()
