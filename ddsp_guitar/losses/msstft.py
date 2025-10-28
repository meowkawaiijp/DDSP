import torch
from torch import nn


class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(2048, 1024, 512), hops=(512, 256, 128), win_lengths=(2048, 1024, 512)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hops = hops
        self.win_lengths = win_lengths

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        loss = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hops, self.win_lengths):
            Yp = torch.stft(y_pred, n_fft=n_fft, hop_length=hop, win_length=win, return_complex=True)
            Yt = torch.stft(y_true, n_fft=n_fft, hop_length=hop, win_length=win, return_complex=True)
            loss = loss + (Yp.abs() - Yt.abs()).abs().mean()
        return loss
