import torch
from torch import nn


class ParamTanhWS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        y = torch.tanh(alpha * x) * (1.0 + beta * (x ** 2))
        return y


class LUTWaveshaper(nn.Module):
    def __init__(self, table_size: int = 256, x_range: float = 3.0):
        super().__init__()
        self.table = nn.Parameter(torch.linspace(-1, 1, table_size))
        self.x_range = x_range
        self.table_size = table_size

    def forward(self, x: torch.Tensor):
        xi = (x.clamp(-self.x_range, self.x_range) + self.x_range) / (2 * self.x_range)
        idx = xi * (self.table_size - 1)
        idx0 = torch.floor(idx).long().clamp(0, self.table_size - 2)
        frac = (idx - idx0.float())
        v0 = self.table[idx0]
        v1 = self.table[idx0 + 1]
        return v0 + frac * (v1 - v0)
