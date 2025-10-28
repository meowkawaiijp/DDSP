import torch
from torch import nn
from einops import rearrange


class HarmonicSynth(nn.Module):
    def __init__(self, num_harmonics: int = 64, sample_rate: int = 48000):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.sample_rate = sample_rate

    def forward(self, f0_hz: torch.Tensor, amp_harm: torch.Tensor, phase: torch.Tensor | None = None):
        # f0_hz: (B, T)
        # amp_harm: (B, T, H) non-negative amplitudes per harmonic
        B, T = f0_hz.shape
        H = self.num_harmonics
        assert amp_harm.shape == (B, T, H)
        if phase is None:
            phase = torch.zeros(B, H, device=f0_hz.device)
        # Placeholder synthesis per-frame; OLA engine should handle sample-rate synthesis.
        block = 1
        f0 = f0_hz  # (B,T)
        omega = 2 * torch.pi * f0 / self.sample_rate  # (B,T)
        t = torch.zeros(B, T, device=f0.device)
        phases = rearrange(phase, 'b h -> b 1 h') + rearrange(omega, 'b t -> b t 1') * t.unsqueeze(-1)
        harm_idx = torch.arange(1, H + 1, device=f0.device)
        phases = phases * harm_idx  # (B,T,H)
        y = (amp_harm * torch.sin(phases)).sum(dim=-1)  # (B,T)
        return y, phase
