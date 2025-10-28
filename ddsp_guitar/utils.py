import torch
import torch.nn.functional as F


def hz_to_rads_per_sample(f_hz: torch.Tensor, sample_rate: int) -> torch.Tensor:
    return 2 * torch.pi * f_hz / sample_rate


def safe_softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x, beta=1.0)
