import torch
from torch import nn


def rbj_low_shelf(fc, gain_db, Q, fs):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * torch.pi * fc / fs
    alpha = torch.sin(w0) / (2.0 * Q)
    cosw = torch.cos(w0)
    sqrtA = torch.sqrt(A)
    b0 = A * ((A + 1.0) - (A - 1.0) * cosw + 2.0 * sqrtA * alpha)
    b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cosw)
    b2 = A * ((A + 1.0) - (A - 1.0) * cosw - 2.0 * sqrtA * alpha)
    a0 = (A + 1.0) + (A - 1.0) * cosw + 2.0 * sqrtA * alpha
    a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cosw)
    a2 = (A + 1.0) + (A - 1.0) * cosw - 2.0 * sqrtA * alpha
    b0, b1, b2, a0, a1, a2 = [x / a0 for x in (b0, b1, b2, a0, a1, a2)]
    return b0, b1, b2, a1, a2


def rbj_high_shelf(fc, gain_db, Q, fs):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * torch.pi * fc / fs
    alpha = torch.sin(w0) / (2.0 * Q)
    cosw = torch.cos(w0)
    sqrtA = torch.sqrt(A)
    b0 = A * ((A + 1.0) + (A - 1.0) * cosw + 2.0 * sqrtA * alpha)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cosw)
    b2 = A * ((A + 1.0) + (A - 1.0) * cosw - 2.0 * sqrtA * alpha)
    a0 = (A + 1.0) - (A - 1.0) * cosw + 2.0 * sqrtA * alpha
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cosw)
    a2 = (A + 1.0) - (A - 1.0) * cosw - 2.0 * sqrtA * alpha
    b0, b1, b2, a0, a1, a2 = [x / a0 for x in (b0, b1, b2, a0, a1, a2)]
    return b0, b1, b2, a1, a2


def rbj_peaking(fc, gain_db, Q, fs):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * torch.pi * fc / fs
    alpha = torch.sin(w0) / (2.0 * Q)
    cosw = torch.cos(w0)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cosw
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cosw
    a2 = 1.0 - alpha / A
    b0, b1, b2, a0, a1, a2 = [x / a0 for x in (b0, b1, b2, a0, a1, a2)]
    return b0, b1, b2, a1, a2


def biquad_step(xn, z1, z2, b0, b1, b2, a1, a2):
    yn = b0 * xn + z1
    z1n = b1 * xn - a1 * yn + z2
    z2n = b2 * xn - a2 * yn
    return yn, z1n, z2n


class ToneStack(nn.Module):
    def __init__(self, fs=48000):
        super().__init__()
        self.fs = fs

    def forward(self, x: torch.Tensor, low_gain_db, mid_gain_db, mid_fc, mid_Q, high_gain_db):
        """
        Time-varying 3-band tone stack using RBJ biquads.

        Args (all tensors shaped (B, N) or broadcastable):
            x: input audio (B, N)
            low_gain_db: low shelf gain in dB
            mid_gain_db: peaking EQ gain in dB
            mid_fc: peaking EQ center freq in Hz
            mid_Q: peaking EQ Q factor
            high_gain_db: high shelf gain in dB
        """
        B, N = x.shape
        device = x.device
        # Broadcast params
        def ensure(t):
            if not torch.is_tensor(t):
                t = torch.tensor(t, device=device)
            while t.dim() < 2:
                t = t.unsqueeze(0)
            if t.shape[0] == 1 and B > 1:
                t = t.expand(B, -1)
            if t.shape[1] == 1 and N > 1:
                t = t.expand(-1, N)
            return t

        low_gain_db = ensure(low_gain_db)
        mid_gain_db = ensure(mid_gain_db)
        mid_fc = ensure(torch.clamp(mid_fc, 50.0, 8000.0))
        mid_Q = ensure(torch.clamp(mid_Q, 0.2, 4.0))
        high_gain_db = ensure(high_gain_db)

        y = torch.zeros_like(x)
        for b in range(B):
            z1_l = torch.tensor(0.0, device=device)
            z2_l = torch.tensor(0.0, device=device)
            z1_m = torch.tensor(0.0, device=device)
            z2_m = torch.tensor(0.0, device=device)
            z1_h = torch.tensor(0.0, device=device)
            z2_h = torch.tensor(0.0, device=device)
            for n in range(N):
                xn = x[b, n]
                # Low shelf at ~120 Hz
                b0, b1, b2, a1, a2 = rbj_low_shelf(
                    torch.tensor(120.0, device=device), low_gain_db[b, n], torch.tensor(0.707, device=device), self.fs
                )
                yl, z1_l, z2_l = biquad_step(xn, z1_l, z2_l, b0, b1, b2, a1, a2)
                # Mid peaking
                b0, b1, b2, a1, a2 = rbj_peaking(mid_fc[b, n], mid_gain_db[b, n], mid_Q[b, n], self.fs)
                ym, z1_m, z2_m = biquad_step(yl, z1_m, z2_m, b0, b1, b2, a1, a2)
                # High shelf at ~4 kHz
                b0, b1, b2, a1, a2 = rbj_high_shelf(
                    torch.tensor(4000.0, device=device), high_gain_db[b, n], torch.tensor(0.707, device=device), self.fs
                )
                yh, z1_h, z2_h = biquad_step(ym, z1_h, z2_h, b0, b1, b2, a1, a2)
                y[b, n] = yh
        return y
