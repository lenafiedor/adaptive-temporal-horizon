import torch

import adaptive_horizon.config as config
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


def adaptive_batch_loss(
    model: MLP, inputs: torch.Tensor, targets: torch.Tensor, T: torch.Tensor
):
    """Compute loss for sample-specific adaptive temporal horizons."""
    batch_size = inputs.shape[0]
    max_T = int(T.max().item())

    x_pred = inputs
    all_preds = []
    for _ in range(max_T):
        x_pred = model(x_pred)
        all_preds.append(x_pred)
    preds = torch.stack(all_preds, dim=1)

    total_loss = 0.0
    for i in range(batch_size):
        t_i = int(T[i].item())
        sample_loss = torch.nn.functional.mse_loss(preds[i, :t_i], targets[i, :t_i])
        total_loss += sample_loss

    return total_loss / batch_size


def lle_predictability_weights(
    lambda_scores: torch.Tensor,
    T: int,
    dt: float = config.DT,
    rho: float = config.RHO,
    temperature: float = config.TEMPERATURE,
    floor: float = config.WEIGHT_FLOOR,
):
    """Build normalized rollout-step weights from per-sample FTLE scores."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if floor < 0:
        raise ValueError(f"floor must be non-negative, got {floor}")

    taus = torch.arange(
        1, T + 1, dtype=lambda_scores.dtype, device=lambda_scores.device
    )
    positive_lambda = torch.clamp(lambda_scores, min=0.0).unsqueeze(1)
    predictability_budget = positive_lambda * taus.unsqueeze(0) * dt
    weights = torch.sigmoid((rho - predictability_budget) / temperature)
    weights = torch.clamp(weights, min=floor)

    return weights / weights.sum(dim=1, keepdim=True)


def lle_weighted_batch_loss(
    model: MLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    lambda_scores: torch.Tensor,
    dt: float = config.DT,
    rho: float = config.RHO,
    temperature: float = config.TEMPERATURE,
    floor: float = config.WEIGHT_FLOOR,
    anchor_alpha: float = config.ANCHOR_ALPHA,
):
    """Compute an LLE-weighted fixed-horizon autoregressive loss.

    Args:
        model: MLP model
        inputs: (batch_size, input_size)
        targets: (batch_size, T_max, input_size)
        lambda_scores: (batch_size,) aligned forward FTLE scores
        dt: simulation time step
        rho: predictability budget threshold
        temperature: sigmoid softness for the budget threshold
        floor: minimum unnormalized rollout weight
        anchor_alpha: one-step MSE weight in the final loss
    Returns:
        float: average weighted rollout loss
    """
    if not 0 <= anchor_alpha <= 1:
        raise ValueError(f"anchor_alpha must be in [0, 1], got {anchor_alpha}")

    T_max = targets.shape[1]
    x_pred = inputs
    all_preds = []
    for _ in range(T_max):
        x_pred = model(x_pred)
        all_preds.append(x_pred)
    preds = torch.stack(all_preds, dim=1)

    step_mse = torch.nn.functional.mse_loss(preds, targets, reduction="none").mean(
        dim=2
    )
    weights = lle_predictability_weights(
        lambda_scores=lambda_scores,
        T=T_max,
        dt=dt,
        rho=rho,
        temperature=temperature,
        floor=floor,
    )

    weighted_rollout_loss = (step_mse * weights).sum(dim=1).mean()
    one_step_loss = torch.nn.functional.mse_loss(preds[:, 0], targets[:, 0])

    return anchor_alpha * one_step_loss + (1.0 - anchor_alpha) * weighted_rollout_loss


def validation_loss(model, val_loader, T, device=config.DEVICE):
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
    """Compute validation loss for sample-specific hard temporal horizons."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets, T in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            T = T.to(device)
            loss = adaptive_batch_loss(model, inputs, targets, T)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def lle_weighted_validation_loss(
    model,
    val_loader,
    dt=config.DT,
    device=config.DEVICE,
    rho: float = config.RHO,
    temperature: float = config.TEMPERATURE,
    floor: float = config.WEIGHT_FLOOR,
    anchor_alpha: float = config.ANCHOR_ALPHA,
):
    """Compute validation loss for the LLE-weighted training objective."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets, lambda_scores in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            lambda_scores = lambda_scores.to(device)
            loss = lle_weighted_batch_loss(
                model,
                inputs,
                targets,
                lambda_scores,
                dt=dt,
                rho=rho,
                temperature=temperature,
                floor=floor,
                anchor_alpha=anchor_alpha,
            )
            total_loss += loss.item()

    return total_loss / len(val_loader)


def compute_gradient_norm(model, loader, T, max_batches=5, device=config.DEVICE):
    """Compute gradient norm using loss per paper Eq. 3."""
    model.zero_grad()
    total_loss = 0.0
    batch_count = 0

    for i, (inputs, targets) in enumerate(loader):
        if i >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        total_loss += batch_loss(model, inputs, targets[:, :T], T)
        batch_count += 1

    total_loss /= batch_count
    total_loss.backward()

    total_norm = torch.zeros((), device=device)
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().pow(2)

    return torch.sqrt(total_norm)


def compute_g_T(model, loader, T_vals, max_batches=5, device=config.DEVICE):
    model.eval()

    g1 = compute_gradient_norm(
        model, loader, T=1, device=device, max_batches=max_batches
    )
    g_vals = {}

    for T in T_vals:
        gT = compute_gradient_norm(
            model, loader, T=T, device=device, max_batches=max_batches
        )
        g_vals[T] = (gT / g1).item()

    return g_vals
