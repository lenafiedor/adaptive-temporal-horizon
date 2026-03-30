import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

from adaptive_horizon.model.mlp import MLP, MLPConfig
from adaptive_horizon.model.horizon_scheduler import AdaptiveHorizonScheduler
from adaptive_horizon.data.dataset import LorenzDataset, collate_fn
from adaptive_horizon.training.loss import compute_loss, compute_validation_loss
from adaptive_horizon.visualization.plotting import save_results

SAVE_DIR = Path("experiments/lorenz")


def train(model, train_loader, val_loader, optimizer, T, epochs, device="cpu", scheduler=None):
    """
    Train model with optional adaptive temporal horizon scheduling.
    
    Args:
        model: MLP model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: PyTorch optimizer
        T: Temporal horizon (used as fixed T if scheduler is None)
        epochs: Total epochs
        device: CPU or GPU
        scheduler: AdaptiveHorizonScheduler for training_results with an adaptive horizon (optional)
        
    Returns:
        losses: List of training_results losses
        val_losses: List of validation losses
        T_schedule: List of T values used at each epoch (None if not adaptive)
    """
    losses = []
    val_losses = []
    T_schedule = [] if scheduler else None
    best_val_loss = float('inf')
    current_T = scheduler.current_T if scheduler else T
    
    for epoch in range(epochs):
        if scheduler:
            if epoch > 0 and scheduler.should_increase_T(epoch, val_losses[-1], best_val_loss):
                old_T = scheduler.current_T
                new_T = scheduler.increase_T()
                train_loader.dataset.update_horizon(new_T)
                val_loader.dataset.update_horizon(new_T)
                print(f"\n>>> Epoch {epoch}: Increasing T from {old_T} to {new_T}\n")
            current_T = scheduler.current_T
            T_schedule.append(current_T)
        
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model, inputs, targets, current_T)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        val_loss = compute_validation_loss(model, val_loader, current_T, device)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return losses, val_losses, T_schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", type=int, default=1, help="Temporal horizon (or initial T if adaptive)")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of training_results epochs")
    parser.add_argument("--adaptive", "-a", action="store_true", help="Enable adaptive horizon training_results")
    parser.add_argument("--max-T", type=int, default=16, help="Maximum temporal horizon (adaptive only)")
    parser.add_argument("--warmup", "-w", type=int, default=10, help="Warmup epochs (adaptive only)")
    parser.add_argument("--update-freq", "-u", type=int, default=5, help="T increase check frequency (adaptive only)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    scheduler = None
    if args.adaptive:
        scheduler = AdaptiveHorizonScheduler(
            initial_T=args.T,
            max_T=args.max_T,
            warmup_epochs=args.warmup,
            update_frequency=args.update_freq
        )

    config = MLPConfig(
        input_size=3,
        output_size=3,
        layer_widths=[64, 64, 64],
        residual_connections=True,
        k = 1,
        activation=nn.ReLU()
    )
    model = MLP(config, random_seed=42).to(device)

    train_dataset = LorenzDataset(num_trajectories=100, steps_per_trajectory=1000, T=args.T, normalize=True)
    val_dataset = LorenzDataset(num_trajectories=20, steps_per_trajectory=1000, T=args.T, normalize=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses, val_losses, T_schedule = train(
        model, train_loader, val_loader, optimizer, args.T,
        epochs=args.epochs, device=device, scheduler=scheduler
    )

    initial_T = args.T
    save_results(losses, val_losses, initial_T, SAVE_DIR, T_schedule)

    model_dir = SAVE_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    prefix = "adaptive_" if args.adaptive else ""
    model_path = model_dir / f"{prefix}mlp_T{args.T}.pt"

    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': {
            'input_size': config.input_size,
            'output_size': config.output_size,
            'layer_widths': config.layer_widths,
            'residual_connections': config.residual_connections,
            'k': config.k,
        }
    }
    if args.adaptive:
        save_dict['final_T'] = scheduler.current_T
        save_dict['T_schedule'] = T_schedule
    else:
        save_dict['train_T'] = args.T
    
    torch.save(save_dict, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
