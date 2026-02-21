import torch
import torch.nn as nn


class SampleModel(nn.Module):
    """Two-layer MLP matching the sample ONNX/sample.png architecture.

    Architecture:
      input (dim 4)
        -> Linear(4, 4)
        -> ReLU
        -> Linear(4, 4)
        -> ReLU
    """

    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(4, 4, bias=True)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(4, 4, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        return x


__all__ = ["SampleModel"]
