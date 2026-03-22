import torch


class ResidualBlock(torch.nn.Module):

    def __init__(self, width: int, k: int, activation: torch.nn.Module) -> None:
        super(ResidualBlock, self).__init__()

        self.activation = activation
        self.layer_norm = torch.nn.LayerNorm(width)

        self.linear1 = torch.nn.Linear(width, k * width)
        self.linear2 = torch.nn.Linear(width * k, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.layer_norm(x)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.linear2(out)
        out += identity
        return out
