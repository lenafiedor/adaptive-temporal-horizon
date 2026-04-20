import torch
import torch.nn as nn

from adaptive_horizon.model.mlp import MLP, MLPConfig


def load_model(model_path):
    checkpoint = torch.load(model_path, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    cfg = checkpoint["config"]
    config = MLPConfig(
        input_size=cfg["input_size"],
        output_size=cfg["output_size"],
        layer_widths=cfg["layer_widths"],
        residual_connections=cfg["residual_connections"],
        k=cfg.get("k"),
        activation=nn.ReLU(),
    )

    model = MLP(config, random_seed=42)
    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint
