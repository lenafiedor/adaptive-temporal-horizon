import torch

from adaptive_horizon.model.mlp import MLP


def batch_loss(model, inputs, targets, T):
    """Compute per-sample loss with a fixed horizon.
    Args:
        model: MLP model
        inputs: (batch_size, input_size) tensor
        targets: (batch_size, T, input_size) tensor
        T: prediction horizon (scalar)
    Returns:
        float: average loss
    """
    x_pred = inputs
    total_loss = 0.0

    for tau in range(T):
        x_pred = model(x_pred)
        total_loss += torch.nn.functional.mse_loss(x_pred, targets[:, tau])

    return total_loss / T


def adaptive_batch_loss(model: MLP, inputs: torch.Tensor, targets: torch.Tensor, T: torch.Tensor):
    """Compute per-sample loss with sample-specific horizons.
    Args:
        model: MLP model
        inputs: (batch_size, input_size)
        targets: (batch_size, max_T, input_size) - padded to max_T
        T: (batch_size,) - horizon per sample
    Returns:
        float: average loss
    """
    batch_size = inputs.shape[0]
    max_T = int(T.max().item())

    # Autoregressive predictions up to max_T
    x_pred = inputs
    all_preds = []
    for tau in range(max_T):
        x_pred = model(x_pred)
        all_preds.append(x_pred)
    preds = torch.stack(all_preds, dim=1)  # [batch_size, max_T, input_size]

    # Per-sample MSE with masking based on each sample's T
    total_loss = 0.0
    for i in range(batch_size):
        t_i = int(T[i].item())
        sample_loss = torch.nn.functional.mse_loss(preds[i, :t_i], targets[i, :t_i])
        total_loss += sample_loss

    return total_loss / batch_size


def validation_loss(model, val_loader, T, device="cpu"):
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss = batch_loss(model, inputs, targets, T)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def adaptive_validation_loss(model, val_loader, device="cpu"):
    """Compute adaptive validation loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets, T in val_loader:
            inputs, targets, T = inputs.to(device), targets.to(device), T.to(device)
            loss = adaptive_batch_loss(model, inputs, targets, T)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def compute_gradient_norm(model, inputs, targets, T):
    """Compute gradient norm using summed loss per paper Eq. 3."""
    # Handle tensor T (from batched dataloader)
    if isinstance(T, torch.Tensor):
        T = int(T.min().item())

    model.zero_grad()

    x_pred = inputs
    total_loss = 0.0
    for tau in range(T):
        x_pred = model(x_pred)
        total_loss += torch.nn.functional.mse_loss(x_pred, targets[:, tau])

    total_loss = total_loss / T
    total_loss.backward()

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm()**2

    return torch.sqrt(total_norm)


def compute_g_T(model, loader, T_vals, device="cpu"):
    model.eval()

    g_vals = {T: 0.0 for T in T_vals}
    num_batches = 5

    for i, (inputs, targets) in enumerate(loader):
        if i >= num_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        g1 = compute_gradient_norm(model, inputs, targets[:, :1], T=1).detach()

        for T in T_vals:
            grad_norm = compute_gradient_norm(model, inputs, targets[:, :T], T=T)
            g_vals[T] += (grad_norm / g1).item()

    for T in g_vals:
        g_vals[T] /= num_batches

    return g_vals
