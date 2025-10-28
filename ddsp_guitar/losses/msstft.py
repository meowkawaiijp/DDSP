import torch
from torch import nn


class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(2048, 1024, 512), hops=(512, 256, 128), win_lengths=(2048, 1024, 512)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hops = hops
        self.win_lengths = win_lengths

    def stft(self, y: torch.Tensor, n_fft: int, hop: int, win: int):
        window = torch.hann_window(win, device=y.device)
        return torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, window=window, center=True, return_complex=True)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # Expect (B, N) or (N,), handle batching
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)
        loss = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hops, self.win_lengths):
            Yp = self.stft(y_pred, n_fft, hop, win)
            Yt = self.stft(y_true, n_fft, hop, win)
            mag_diff = (Yp.abs() - Yt.abs()).abs().mean()
            sc_diff = (torch.log(Yp.abs().clamp_min(1e-7)) - torch.log(Yt.abs().clamp_min(1e-7))).abs().mean()
            loss = loss + mag_diff + 0.5 * sc_diff
        return loss
