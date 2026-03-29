import torch

from adaptive_horizon.training.loss import compute_loss


def compute_validation_loss(model, val_loader, T, device="cpu"):
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss = compute_loss(model, inputs, targets, T)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)
