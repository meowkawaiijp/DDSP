import torch
from torch import nn


class TransientSeparator(nn.Module):
    def __init__(self, kernel_size: int = 15):
        super().__init__()
        self.conv = nn.Conv1d(1, 2, kernel_size, padding=kernel_size // 2, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        h = self.conv(x.unsqueeze(1))  # (B,2,N)
        gate = self.gate(h[:, :1, :])  # (B,1,N)
        attack = gate * x.unsqueeze(1)
        sustain = (1 - gate) * x.unsqueeze(1)
        return attack.squeeze(1), sustain.squeeze(1), gate.squeeze(1)
