import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path


def save_results(losses, val_losses, T, save_dir, T_schedule=None):
    """Save training_results history and optionally T schedule."""
    save_path = Path(save_dir) / "training_results"
    save_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    adaptive = T_schedule is not None
    prefix = "adaptive_" if adaptive else ""
    filename = save_path / f"{prefix}loss_T{T}_{timestamp}"
    
    if adaptive:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Training Loss (T={T})" if not adaptive else "Training with Adaptive Temporal Horizon")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if adaptive:
        ax2.plot(T_schedule, marker='o', linewidth=2, markersize=4)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Temporal Horizon T")
        ax2.set_title("Adaptive Temporal Horizon Schedule")
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=150)
    plt.close()
    print(f"Loss plot saved to {filename}.png")

    with open(f"{filename}.txt", "w") as f:
        if adaptive:
            f.write("epoch,train_loss,val_loss,T\n")
            for i, (tl, vl, t) in enumerate(zip(losses, val_losses, T_schedule)):
                f.write(f"{i},{tl:.6f},{vl:.6f},{t}\n")
        else:
            f.write("epoch,train_loss,val_loss\n")
            for i, (tl, vl) in enumerate(zip(losses, val_losses)):
                f.write(f"{i},{tl:.6f},{vl:.6f}\n")
    print(f"Loss values saved to {filename}.txt")


def plot_g_T(g_values, save_path, adaptive=False, train_T=None):
    Ts = list(g_values.keys())
    values = list(g_values.values())

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_T{train_T}" if train_T is not None else ""
    if adaptive:
        filename = save_path / f"adaptive_gradient_scaling{suffix}_{timestamp}.png"
    else:
        filename = save_path / f"gradient_scaling{suffix}_{timestamp}.png"

    plt.figure()
    plt.plot(Ts, values, marker='o')
    plt.xlabel("T")
    plt.ylabel("g(T)")
    plt.title("Gradient Scaling")
    plt.grid(True)
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Gradient scaling plot saved to {filename}")


def plot_trajectory(trajectory, save_dir="experiments/lorenz/trajectories"):
    x = [state[0] for state in trajectory]
    y = [state[1] for state in trajectory]
    z = [state[2] for state in trajectory]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z)
    ax.set_title("Lorenz Attractor")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "lorenz_system.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_lyapunov_exponent(exponents, save_dir="experiments/lorenz/analysis"):
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

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "local_lyapunov.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")
