import torch
from torch import nn

from .modules.harmonic_synth import HarmonicSynth
from .modules.noise_synth import MultibandNoiseSynth
from .modules.waveshaper import ParamTanhWS
from .modules.tonestack import ToneStack
from .modules.transient import TransientSeparator


class Encoder(nn.Module):
    def __init__(self, in_ch=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 9, padding=4),
            nn.GELU(),
            nn.Conv1d(64, 128, 9, padding=4),
            nn.GELU(),
            nn.Conv1d(128, hidden, 9, padding=4),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class TCNController(nn.Module):
    def __init__(self, in_ch=128, hidden=256, layers=6):
        super().__init__()
        blocks = []
        for i in range(layers):
            d = 2 ** i
            blocks += [
                nn.Conv1d(in_ch, hidden, 3, padding=d, dilation=d),
                nn.GELU(),
                nn.Conv1d(hidden, in_ch, 1),
            ]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class GuitarDDSP(nn.Module):
    def __init__(self, sample_rate=48000, num_harm=64, num_noise=8):
        super().__init__()
        self.encoder = Encoder()
        self.controller = TCNController()
        self.harm = HarmonicSynth(num_harm, sample_rate)
        self.noise = MultibandNoiseSynth(num_noise, sample_rate)
        self.ws = ParamTanhWS()
        self.ts = ToneStack(sample_rate)
        self.sep = TransientSeparator()
        self.sample_rate = sample_rate

        out_dim = num_harm + num_noise + 2 + 3
        self.head = nn.Conv1d(128, out_dim, 1)

    def forward(self, x, f0_hz, loudness):
        B, N = x.shape
        h = self.encoder(x.unsqueeze(1))
        c = self.controller(h)
        p = torch.tanh(self.head(c))  # (B, out_dim, N)

        num_harm = self.harm.num_harmonics
        num_noise = self.noise.num_bands
        harm_amp = torch.nn.functional.softplus(p[:, :num_harm, :])
        noise_env = torch.nn.functional.softplus(p[:, num_harm : num_harm + num_noise, :])
        alpha = 0.5 + 4.5 * (p[:, num_harm + num_noise : num_harm + num_noise + 1, :])
        beta = torch.relu(p[:, num_harm + num_noise + 1 : num_harm + num_noise + 2, :])
        low = 12 * p[:, -3:-2, :]
        mid = 12 * p[:, -2:-1, :]
        high = 12 * p[:, -1:, :]

        harm_amp_t = harm_amp.transpose(1, 2)  # (B,N,H)
        f0 = torch.nn.functional.interpolate(
            f0_hz.unsqueeze(1), size=N, mode="linear", align_corners=False
        ).squeeze(1)
        y_h, _ = self.harm(f0, harm_amp_t)
        y_n = self.noise(noise_env.transpose(1, 2), N)
        y = y_h + y_n

        a, s, gate = self.sep(y)
        y_a = self.ws(a, alpha.squeeze(1), beta.squeeze(1))
        y_s = self.ts(
            s,
            low.squeeze(1),
            mid.squeeze(1),
            torch.tensor(800.0, device=x.device),
            torch.tensor(0.707, device=x.device),
            high.squeeze(1),
        )
        y_out = y_a + y_s
        return y_out
