import argparse
import numpy as np

from adaptive_horizon.dynamics.lorenz import simulate_lorenz
from adaptive_horizon.dynamics.lyapunov import compute_global_lyapunov, compute_local_lyapunov, smooth_lle
from adaptive_horizon.visualization.plotting import plot_lyapunov_exponents, plot_trajectory_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, default="global", choices=["global", "local"], help="LLE computation mode (global/local)")
    parser.add_argument("--plot", "-p", action="store_true", help="Plot the Lorenz trajectory")
    parser.add_argument("--window", "-w", type=int, default=10, help="Window size for LLE computation (only local mode)")
    args = parser.parse_args()

    lorenz_trajectory = simulate_lorenz()
    burn_in = int(0.01 * len(lorenz_trajectory))

    if args.mode == "global":
        lyap = compute_global_lyapunov()
        print(f"Largest Lyapunov Exponent: {lyap:.4f}")

    elif args.mode == "local":
        lles = smooth_lle(compute_local_lyapunov(lorenz_trajectory, burn_in), window=args.window)

        print(f"Mean 1st LLE: {np.mean(lles[:, 0]):.4f}")
        print(f"Std 1st LLE: {np.std(lles[:, 0]):.4f}")

        if args.plot:
            plot_lyapunov_exponents(lles, window=args.window)
            plot_trajectory_heatmap(lorenz_trajectory, lles, args.window, burn_in)

    else:
        raise ValueError("Invalid mode. Choose 'global' or 'local'.")


if __name__ == "__main__":
    main()
