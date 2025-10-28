import torch
from torch.utils.data import Dataset
import soundfile as sf
import random
from pathlib import Path

from .dsp_teacher import DSPTeacherChain


class GuitarDIDataset(Dataset):
    """
    Dataset that loads DI WAVs and synthesizes target via DSPTeacherChain on-the-fly.
    """

    def __init__(self, wav_dir: str, sample_rate: int = 48000, segment_seconds: float = 1.0):
        super().__init__()
        self.wav_paths = sorted([str(p) for p in Path(wav_dir).glob('*.wav')])
        self.sr = sample_rate
        self.seg = int(sample_rate * segment_seconds)
        self.teacher = DSPTeacherChain(sample_rate)

    def __len__(self):
        return max(1, len(self.wav_paths) * 16)

    def _load_random(self):
        if len(self.wav_paths) == 0:
            # fallback to noise
            x = torch.randn(self.seg)
            return x
        p = random.choice(self.wav_paths)
        x, sr = sf.read(p)
        x = torch.tensor(x, dtype=torch.float32)
        if x.dim() > 1:
            x = x.mean(dim=-1)
        if sr != self.sr:
            # simple resample via linear (torch only). For accurate, use torchaudio on runtime env.
            n = int(len(x) * self.sr / sr)
            x = torch.nn.functional.interpolate(x.view(1,1,-1), size=n, mode='linear', align_corners=False).view(-1)
        if len(x) < self.seg:
            pad = self.seg - len(x)
            x = torch.nn.functional.pad(x, (0, pad))
        else:
            start = random.randint(0, len(x)-self.seg)
            x = x[start:start+self.seg]
        return x

    def __getitem__(self, idx):
        x = self._load_random()
        B = 1
        xB = x.unsqueeze(0)
        params = self.teacher.random_params(B, xB.shape[-1], xB.device)
        with torch.no_grad():
            y = self.teacher(xB, **params)
        y = y.squeeze(0)
        f0 = torch.full((self.seg // 256 + 1,), 110.0)
        loud = torch.zeros_like(f0)
        return x, f0, loud, y
