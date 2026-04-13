import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def save_losses(train_losses: torch.Tensor, val_losses: torch.Tensor, save_dir: Path, T=None, adaptive=False):
    """Save training history"""
    loss_dir = save_dir / "training_results"
    loss_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"loss_T{T}_{timestamp}" if not adaptive else f"adaptive_loss_{timestamp}"
    loss_path = loss_dir / filename

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Training Loss (T={T})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{loss_path}.png", dpi=150)
    plt.close()
    print(f"Loss plot saved to {loss_path}.png")

    with open(f"{loss_path}.txt", "w") as f:
        f.write("min_train_loss,min_val_loss\n")
        f.write(f"{min(train_losses)},{min(val_losses)}")
    print(f"Min loss values saved to {loss_path}.txt")


def save_model(model, config, save_dir, T=None, adaptive=False):
    model_dir = save_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    filename = f"mlp_T{T}.pt" if not adaptive else f"adaptive_mlp.pt"
    model_path = model_dir / filename

    save_dict = {
        'model_state_dict': model.state_dict(),
        'train_T': T,
        'config': {
            'input_size': config.input_size,
            'output_size': config.output_size,
            'layer_widths': config.layer_widths,
            'residual_connections': config.residual_connections,
            'k': config.k
        }
    }

    torch.save(save_dict, model_path)
    print(f"Model saved to {model_path}")


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


def plot_lyapunov_exponents(exponents, window, save_dir="experiments/lorenz/analysis"):
    """
    Plot histograms of all 3 Lyapunov exponents.
    
    Args:
        exponents: array of shape [N, 3] with local Lyapunov exponents
        window: window size used for computation
        save_dir: directory to save plot
    """
    exponents = np.array(exponents)
    labels = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, ax in enumerate(axes):
        lle_i = exponents[:, i]
        mean_lle = np.mean(lle_i)
        ax.hist(lle_i, bins=50, edgecolor='black', alpha=0.7, density=True)
        ax.axvline(x=mean_lle, color='r', linestyle='-', linewidth=2, label=f'Mean: {mean_lle:.3f}')
        ax.set_xlabel("Local Lyapunov Exponent")
        ax.set_ylabel("Density")
        ax.set_yscale('log')
        ax.set_title(f"Distribution of {labels[i]} (window={window})")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"lle_histogram_W{window}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_trajectory_heatmap(trajectory, exponents, window, burn_in, save_dir="experiments/lorenz/analysis"):
    """
    Plot 3D Lorenz trajectory colored by each of the 3 local Lyapunov exponents.
    
    Args:
        trajectory: array of shape [N, 3] with trajectory states
        exponents: array of shape [N-window, 3] with local Lyapunov exponents
        window: window size used for computation
        burn_in: number of initial steps to ignore
        save_dir: directory to save plot
    """
    trajectory = np.array(trajectory)
    exponents = np.array(exponents)
    n_lle = len(exponents)

    offset = burn_in + window
    traj_aligned = trajectory[offset:offset + n_lle]

    x = traj_aligned[:, 0]
    y = traj_aligned[:, 1]
    z = traj_aligned[:, 2]
    
    labels = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]
    
    fig = plt.figure(figsize=(18, 6))
    
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lle_i = exponents[:, i]
        norm = plt.Normalize(lle_i.min(), lle_i.max())
        lc = Line3DCollection(segments, cmap='coolwarm', norm=norm)
        lc.set_array(lle_i[:-1])
        lc.set_linewidth(1)
        ax.add_collection3d(lc)
        
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_zlim(z.min(), z.max())
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Lorenz Attractor colored by {labels[i]}")
        
        cbar = fig.colorbar(lc, ax=ax, shrink=0.5, aspect=20, pad=0.1)
        cbar.set_label(f"Local {labels[i]}")
    
    plt.tight_layout()
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"lorenz_lle_heatmap_W{window}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")
