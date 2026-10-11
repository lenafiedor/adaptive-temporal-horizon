import torch

from adaptive_horizon import config
from adaptive_horizon.model.mlp import MLP


def rollout_predictions(
    model: MLP,
    inputs: torch.Tensor,
    steps: int,
):
    """Predict future states autoregressively from the current state."""
    expected_dim = model.embedding.in_features
    if inputs.shape[1] != expected_dim:
        raise ValueError(
            f"Input feature size {inputs.shape[1]} does not match "
            f"state dimension {expected_dim}"
        )
    current_state = inputs
    preds = []

    for _ in range(steps):
        next_state = model(current_state)
        preds.append(next_state)
        current_state = next_state

    return torch.stack(preds, dim=1)


def batch_loss(
    model,
    inputs,
    targets,
    T,
):
    """Compute per-sample loss with a fixed horizon.
    Args:
        model: MLP model
        inputs: (batch_size, input_size) tensor
        targets: (batch_size, T, input_size) tensor
        T: prediction horizon (scalar)
    Returns:
        float: average loss
    """
    preds = rollout_predictions(model, inputs, T)
    total_loss = 0.0

    for tau in range(T):
        total_loss += torch.nn.functional.mse_loss(preds[:, tau], targets[:, tau])

    return total_loss / T


def adaptive_batch_loss(
    model: MLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    T: torch.Tensor,
):
    """Compute loss for sample-specific adaptive temporal horizons."""
    batch_size = inputs.shape[0]
    max_T = int(T.max().item())

    preds = rollout_predictions(model, inputs, max_T)

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

    T_max = targets.shape[1]
    preds = rollout_predictions(model, inputs, T_max)

    step_mse = torch.nn.functional.mse_loss(preds, targets, reduction="none").mean(
        dim=2
    )
    weights = lle_predictability_weights(lambda_scores=lambda_scores, T=T_max, dt=dt)

    weighted_rollout_loss = (step_mse * weights).sum(dim=1).mean()
    one_step_loss = torch.nn.functional.mse_loss(preds[:, 0], targets[:, 0])

    return anchor_alpha * one_step_loss + (1.0 - anchor_alpha) * weighted_rollout_loss


def evaluate_loss(model, loader, loss_fn, device=config.DEVICE):
    """Average ``loss_fn(model, *batch)`` over a loader without gradients."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = [tensor.to(device) for tensor in batch]
            total_loss += loss_fn(model, *batch).item()

    return total_loss / len(loader)


def validation_loss(model, val_loader, T, device=config.DEVICE):
    """Compute validation loss at a fixed horizon T."""
    return evaluate_loss(
        model,
        val_loader,
        lambda model, inputs, targets: batch_loss(model, inputs, targets, T),
        device,
    )


def adaptive_validation_loss(model, val_loader, device="cpu"):
    """Compute validation loss for sample-specific hard temporal horizons."""
    return evaluate_loss(model, val_loader, adaptive_batch_loss, device)


def lle_weighted_validation_loss(model, val_loader, dt=config.DT, device=config.DEVICE):
    """Compute validation loss for the LLE-weighted training objective."""
    return evaluate_loss(
        model,
        val_loader,
        lambda model, inputs, targets, lambda_scores: lle_weighted_batch_loss(
            model, inputs, targets, lambda_scores, dt=dt
        ),
        device,
    )


def grads_l2_norm(grads, device=config.DEVICE):
    """L2 norm over a collection of gradient tensors."""
    total_norm = torch.zeros((), device=device)
    for grad in grads:
        if grad is not None:
            total_norm += grad.norm().pow(2)
    return torch.sqrt(total_norm)


def compute_gradient_norm(
    model,
    loader,
    T,
    max_batches=config.NUM_BATCHES,
    device=config.DEVICE,
    per_batch=False,
):
    """Compute gradient norm using loss per paper Eq. 3."""

    def parameter_grads():
        return (p.grad for p in model.parameters())

    def batches():
        for i, (inputs, targets) in enumerate(loader):
            if i >= max_batches:
                break
            yield inputs.to(device), targets.to(device)

    if per_batch:
        gradient_norms = []

        for inputs, targets in batches():
            model.zero_grad()
            batch_loss(model, inputs, targets[:, :T], T).backward()
            gradient_norms.append(grads_l2_norm(parameter_grads(), device).item())

        return gradient_norms

    model.zero_grad()
    losses = [
        batch_loss(model, inputs, targets[:, :T], T) for inputs, targets in batches()
    ]
    total_loss = sum(losses) / len(losses)
    total_loss.backward()

    return grads_l2_norm(parameter_grads(), device)


def compute_g_T(
    model,
    loader,
    T_vals,
    max_batches=config.NUM_BATCHES,
    device=config.DEVICE,
    per_batch=False,
):
    """Compute g(T) function per paper Eq. 4.

    Args:
        model: MLP model
        loader: DataLoader
        T_vals: list of horizon values to evaluate
        max_batches: maximum number of batches to use for gradient norm estimation
        device: CPU or GPU
        per_batch: whether to return per-batch g(T) ratios instead of aggregated values
    Returns:
        dict: mapping from evaluation horizon to g(T) value or list of per-batch values
    """
    model.eval()

    def norm_at(T):
        return compute_gradient_norm(
            model,
            loader,
            T=T,
            device=device,
            max_batches=max_batches,
            per_batch=per_batch,
        )

    g1 = norm_at(1)
    g_T = {T: norm_at(T) for T in T_vals}

    if per_batch:
        return {
            T: [gT / g1_i for gT, g1_i in zip(g_T[T], g1) if g1_i != 0] for T in T_vals
        }
    return {T: (g_T[T] / g1).item() for T in T_vals}
