import torch
from torch import nn


class MultibandNoiseSynth(nn.Module):
    def __init__(self, num_bands: int = 8, sample_rate: int = 48000):
        super().__init__()
        self.num_bands = num_bands
        self.sample_rate = sample_rate

    def forward(self, band_env: torch.Tensor, num_samples: int):
        # band_env: (B, T, BANDS)
        B, T, K = band_env.shape
        assert K == self.num_bands
        noise = torch.randn(B, num_samples, device=band_env.device)
        env = band_env.mean(dim=-1)
        env = torch.nn.functional.interpolate(
            env.unsqueeze(1), size=num_samples, mode='linear', align_corners=False
        ).squeeze(1)
        return noise * env
