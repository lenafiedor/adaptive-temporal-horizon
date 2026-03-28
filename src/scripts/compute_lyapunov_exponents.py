import matplotlib.pyplot as plt
import argparse
import numpy as np

from src.model.lorenz import simulate_lorenz


def plot_trajectory(trajectory):
    x = [state[0] for state in trajectory]
    y = [state[1] for state in trajectory]
    z = [state[2] for state in trajectory]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z)
    ax.set_title("Lorenz Attractor")
    plt.show()

def plot_lyapunov_exponent(exponents):
    print(f"Mean LLE: {np.mean(exponents):.4f}")
    print(f"Std LLE: {np.std(exponents):.4f}")
    print(f"Min LLE: {np.min(exponents):.4f}")
    print(f"Max LLE: {np.max(exponents):.4f}")

    plt.figure(figsize=(12, 4))
    plt.plot(exponents, linewidth=0.5)
    plt.axhline(y=np.mean(exponents), color='r', linestyle='--', label=f'Mean: {np.mean(exponents):.3f}')
    plt.xlabel("Time step")
    plt.ylabel("Local Lyapunov Exponent")
    plt.title(f"Local Lyapunov Exponents along Lorenz Trajectory")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, default="global", choices=["global", "local"], help="LLE computation mode (global/local)")
    parser.add_argument("--plot", "-p", default=False, help="Plot the Lorenz trajectory")
    args = parser.parse_args()

    lorenz_trajectory = simulate_lorenz()
    if args.plot:
        plot_trajectory(lorenz_trajectory)

    if args.mode == "global":
        from src.numerics.global_lyapunov import compute_global_lyapunov
        lyap = compute_global_lyapunov()
        print(f"Largest Lyapunov Exponent: {lyap:.4f}")

    elif args.mode == "local":
        from src.numerics.local_lyapunov import compute_local_lyapunov
        lyap = compute_local_lyapunov(lorenz_trajectory, dt=0.01, window=10)
        plot_lyapunov_exponent(lyap)

    else:
        raise ValueError("Invalid mode. Choose 'global' or 'local'.")
