import torch


class DenseBlock(torch.nn.Module):

    def __init__(self, width: int, activation: torch.nn.Module) -> None:
        super(DenseBlock, self).__init__()

        self.activation = activation
        self.layer_norm = torch.nn.LayerNorm(width)

        self.linear1 = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer_norm(x)
        out = self.linear1(out)
        out = self.activation(out)
        return out
