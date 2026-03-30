import argparse

from adaptive_horizon.dynamics.lorenz import simulate_lorenz
from adaptive_horizon.dynamics.lyapunov import compute_global_lyapunov, compute_local_lyapunov
from adaptive_horizon.visualization.plotting import plot_trajectory, plot_lyapunov_exponent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, default="global", choices=["global", "local"], help="LLE computation mode (global/local)")
    parser.add_argument("--plot", "-p", action="store_true", help="Plot the Lorenz trajectory")
    args = parser.parse_args()

    lorenz_trajectory = simulate_lorenz()
    if args.plot:
        plot_trajectory(lorenz_trajectory)

    if args.mode == "global":
        lyap = compute_global_lyapunov()
        print(f"Largest Lyapunov Exponent: {lyap:.4f}")

    elif args.mode == "local":
        lyap = compute_local_lyapunov(lorenz_trajectory)
        plot_lyapunov_exponent(lyap)

    else:
        raise ValueError("Invalid mode. Choose 'global' or 'local'.")


if __name__ == "__main__":
    main()
