import torch
from torch import nn
import torch.nn.functional as F


def moving_average_envelope(x: torch.Tensor, win_ms: float = 5.0, sample_rate: int = 48000) -> torch.Tensor:
    """
    Simple transient envelope via rectified moving average of first difference.
    Args:
        x: (B, N) audio
    Returns:
        env: (B, N) non-negative envelope emphasizing onsets
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    B, N = x.shape
    # High-pass-like diff
    d = F.pad(x[:, 1:] - x[:, :-1], (1, 0))
    d = d.abs()
    win = max(1, int(win_ms * sample_rate / 1000))
    kernel = torch.ones(1, 1, win, device=x.device) / float(win)
    env = F.conv1d(d.unsqueeze(1), kernel, padding=win // 2).squeeze(1)
    return env


class TransientLoss(nn.Module):
    def __init__(self, sample_rate: int = 48000, eps: float = 1e-6, focus_onsets: bool = True):
        super().__init__()
        self.sr = sample_rate
        self.eps = eps
        self.focus_onsets = focus_onsets

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        Ep = moving_average_envelope(y_pred, sample_rate=self.sr)
        Et = moving_average_envelope(y_true, sample_rate=self.sr)
        lp = torch.log(Ep + self.eps)
        lt = torch.log(Et + self.eps)
        diff = (lp - lt).abs()
        if self.focus_onsets:
            # Weight by target envelope normalized per sample
            w = Et / (Et.amax(dim=-1, keepdim=True) + self.eps)
            diff = diff * (0.2 + 0.8 * w)  # base weight + onset emphasis
        return diff.mean()
