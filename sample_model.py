import torch
import torch.nn as nn


class SampleModel(nn.Module):
    """Two-layer MLP with configurable input and hidden dimensions.

    Architecture:
      input (dim input_dim)
        -> Linear(input_dim, hidden_dim)
        -> ReLU
        -> Linear(hidden_dim, hidden_dim)
        -> ReLU

    Parameters
    ----------
    input_dim : int
        Dimension of the input vector.  Defaults to 4 for backward
        compatibility with the original 4×4 architecture.
    hidden_dim : int
        Width of the hidden layer (and the output).  Defaults to 4.
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 4):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        return x


__all__ = ["SampleModel"]
