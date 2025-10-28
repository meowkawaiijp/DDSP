import torch
from torch import nn
from einops import rearrange


class HarmonicSynth(nn.Module):
    def __init__(self, num_harmonics: int = 64, sample_rate: int = 48000):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.sample_rate = sample_rate

    def forward(self, f0_hz: torch.Tensor, amp_harm: torch.Tensor, phase: torch.Tensor | None = None):
        """
        Sample-rate harmonic synthesis with cumulative phase.

        Args:
            f0_hz: (B, N) fundamental frequency in Hz per sample
            amp_harm: (B, N, H) non-negative amplitudes per harmonic
            phase: (B, H) optional initial phase per harmonic (radians)

        Returns:
            y: (B, N) synthesized harmonic signal
            phase_out: (B, H) final phase per harmonic, can be fed to next block
        """
        B, N = f0_hz.shape
        H = self.num_harmonics
        assert amp_harm.shape == (B, N, H)
        device = f0_hz.device
        if phase is None:
            phase = torch.zeros(B, H, device=device)

        # Instantaneous angular frequency per sample
        omega = 2 * torch.pi * f0_hz / self.sample_rate  # (B, N)
        # Expand for each harmonic
        harm_idx = torch.arange(1, H + 1, device=device).view(1, 1, H)  # (1,1,H)
        omega_h = omega.unsqueeze(-1) * harm_idx  # (B, N, H)

        # Cumulative phase for each harmonic, starting from initial phase
        phases = torch.cumsum(omega_h, dim=1) + phase.unsqueeze(1)  # (B, N, H)

        # Synthesize
        y = (amp_harm * torch.sin(phases)).sum(dim=-1)  # (B, N)

        # Final phase (last sample)
        phase_out = torch.remainder(phases[:, -1, :], 2 * torch.pi)  # (B, H)
        return y, phase_out
