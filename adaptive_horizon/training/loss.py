import torch


def compute_loss(model, inputs, targets, T):
    """Compute multi-step autoregressive loss per paper Equation 3.
    Args:
        model: MLP model
        inputs: (batch_size, input_size) tensor
        targets: (batch_size, T, input_size) tensor
        T: prediction horizon (int or tensor)
    Returns:
        float: average loss
    """
    # Handle tensor T (from batched dataloader)
    if isinstance(T, torch.Tensor):
        T = int(T.min().item())  # Use minimum T in batch for safety

    x_pred = inputs
    total_loss = 0.0

    for tau in range(T):
        x_pred = model(x_pred)
        total_loss += torch.nn.functional.mse_loss(x_pred, targets[:, tau])

    return total_loss / T


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


def compute_adaptive_loss(model, val_loader, device="cpu"):
    """Compute adaptive validation loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets, T in val_loader:
            inputs, targets, T = inputs.to(device), targets.to(device), T.to(device)
            loss = compute_loss(model, inputs, targets, T)
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
