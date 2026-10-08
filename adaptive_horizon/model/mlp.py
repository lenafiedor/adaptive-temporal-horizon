from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from adaptive_horizon.model.dense_block import DenseBlock
from adaptive_horizon.model.residual_block import ResidualBlock


@dataclass(frozen=True)
class MLPConfig:
    input_size: int
    output_size: int
    layer_widths: list[int]
    residual_connections: bool = False
    activation: nn.Module = nn.ReLU()
    k: Union[int, None] = None
    delay_window: bool = False


class MLP(torch.nn.Module):
    def __init__(self, config: MLPConfig, random_seed: int):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        super(MLP, self).__init__()
        self.delay_window = config.delay_window
        if self.delay_window and config.output_size != 1:
            raise ValueError("Delay-window models must predict one new intensity")
        self.layer_widths = config.layer_widths
        self.embedding = nn.Linear(
            config.input_size, config.layer_widths[0], bias=False
        )
        self.unembedding = nn.Linear(
            config.layer_widths[-1], config.output_size, bias=False
        )
        self.blocks = nn.ModuleList()

        if config.residual_connections:
            if config.k is None:
                raise ValueError("Residual connections require a k value.")
            for i in range(len(self.layer_widths)):
                self.blocks.append(
                    ResidualBlock(self.layer_widths[i], config.k, config.activation)
                )
        else:
            for i in range(len(self.layer_widths)):
                self.blocks.append(DenseBlock(self.layer_widths[i], config.activation))

    def forward(self, x):
        history = x
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.unembedding(x)
        return torch.cat((history[..., 1:], x), dim=-1) if self.delay_window else x
