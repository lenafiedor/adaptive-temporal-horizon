import torch


def compute_loss(model, inputs, targets, T):
    """Compute multi-step autoregressive loss per paper Equation 3.
    Args:
        model: MLP model
        inputs: (batch_size, input_size) tensor
        targets: (batch_size, T, input_size) tensor
        T: prediction horizon
    Returns:
        float: average loss
    """
    x_pred = inputs
    total_loss = 0.0

    for tau in range(T):
        x_pred = model(x_pred)
        total_loss += torch.nn.functional.mse_loss(x_pred, targets[:, tau])

    return total_loss / T


def compute_gradient_norm(model, inputs, targets, T):
    model.zero_grad()

    loss = compute_loss(model, inputs, targets, T)
    loss.backward()

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm()**2

    return torch.sqrt(total_norm)


def compute_g_T(model, loader, T_vals, device="cpu"):
    model.eval()

    input, targets = next(iter(loader))
    input, targets = input.to(device), targets.to(device)

    g_vals = {}
    g1 = compute_gradient_norm(model, input, targets[:, :1], T=1)

    for T in T_vals:
        grad_norm = compute_gradient_norm(model, input, targets[:, :T], T=T)
        g_T = (grad_norm / g1).item()
        g_vals[T] = g_T

        print(f"T={T}, g(T)={g_T:.4f}")

    return g_vals
