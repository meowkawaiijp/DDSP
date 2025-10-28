import torch
from torch import nn


def bilinear_low_shelf(fc, gain_db, Q, fs):
    A = 10 ** (gain_db / 40)
    w0 = 2 * torch.pi * fc / fs
    alpha = torch.sin(w0) / (2 * Q)
    cosw = torch.cos(w0)
    b0 = A * ((A + 1) - (A - 1) * cosw + 2 * torch.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cosw)
    b2 = A * ((A + 1) - (A - 1) * cosw - 2 * torch.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * cosw + 2 * torch.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cosw)
    a2 = (A + 1) + (A - 1) * cosw - 2 * torch.sqrt(A) * alpha
    b0, b1, b2, a0, a1, a2 = [x / a0 for x in (b0, b1, b2, a0, a1, a2)]
    return b0, b1, b2, a1, a2


class Biquad(nn.Module):
    def __init__(self, fs=48000):
        super().__init__()
        self.fs = fs
        self.register_buffer("z1", torch.zeros(1))
        self.register_buffer("z2", torch.zeros(1))

    def forward(self, x: torch.Tensor, b0, b1, b2, a1, a2):
        if self.z1.numel() != x.numel():
            self.z1 = torch.zeros_like(x)
            self.z2 = torch.zeros_like(x)
        y = torch.zeros_like(x)
        for n in range(x.shape[-1]):
            xn = x[..., n]
            yn = b0 * xn + self.z1
            self.z1 = b1 * xn - a1 * yn + self.z2
            self.z2 = b2 * xn - a2 * yn
            y[..., n] = yn
        return y


class ToneStack(nn.Module):
    def __init__(self, fs=48000):
        super().__init__()
        self.fs = fs
        self.low = Biquad(fs)
        self.mid = Biquad(fs)
        self.high = Biquad(fs)

    def forward(self, x: torch.Tensor, low_gain_db, mid_gain_db, mid_fc, mid_Q, high_gain_db):
        b0, b1, b2, a1, a2 = bilinear_low_shelf(
            torch.tensor(120.0, device=x.device), low_gain_db, torch.tensor(0.707, device=x.device), self.fs
        )
        y = self.low(x, b0, b1, b2, a1, a2)
        b0, b1, b2, a1, a2 = bilinear_low_shelf(mid_fc, mid_gain_db, mid_Q, self.fs)
        y = self.mid(y, b0, b1, b2, a1, a2)
        b0, b1, b2, a1, a2 = bilinear_low_shelf(
            torch.tensor(4000.0, device=x.device), high_gain_db, torch.tensor(0.707, device=x.device), self.fs
        )
        y = self.high(y, b0, b1, b2, a1, a2)
        return y
