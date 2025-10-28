import torch
from torch.utils.data import Dataset
import soundfile as sf
import random
from pathlib import Path

from .dsp_teacher import DSPTeacherChain


class GuitarDIDataset(Dataset):
    """
    Dataset that loads DI WAVs and synthesizes target via DSPTeacherChain on-the-fly.
    キャッシュとGPU対応により高速化。
    """

    def __init__(self, wav_dir: str, sample_rate: int = 48000, segment_seconds: float = 1.0, 
                 device: str = 'cpu', use_cache: bool = True, cache_size: int = 100):
        super().__init__()
        self.wav_paths = sorted([str(p) for p in Path(wav_dir).glob('*.wav')])
        self.sr = sample_rate
        self.seg = int(sample_rate * segment_seconds)
        self.device = torch.device(device)
        self.teacher = DSPTeacherChain(sample_rate).to(self.device)
        
        # キャッシュ設定
        self.use_cache = use_cache
        self.cache = {}
        self.cache_size = cache_size
        
        # データを事前に生成してキャッシュ（オプション）
        if self.use_cache:
            print(f"データキャッシュを生成中... (最大 {cache_size} サンプル)")
            self._populate_cache()

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

    def _generate_sample(self):
        """1つのサンプルを生成（GPU上で処理）"""
        x = self._load_random()
        B = 1
        xB = x.unsqueeze(0).to(self.device)
        params = self.teacher.random_params(B, xB.shape[-1], self.device)
        with torch.no_grad():
            y = self.teacher(xB, **params)
        y = y.squeeze(0).cpu()  # CPUに戻す
        x = x.cpu()
        f0 = torch.full((self.seg // 256 + 1,), 110.0)
        loud = torch.zeros_like(f0)
        return x, f0, loud, y
    
    def _populate_cache(self):
        """事前にキャッシュを生成"""
        for i in range(min(self.cache_size, len(self))):
            self.cache[i] = self._generate_sample()
            if (i + 1) % 10 == 0:
                print(f"  キャッシュ生成中: {i+1}/{min(self.cache_size, len(self))}")

    def __getitem__(self, idx):
        # キャッシュから取得
        if self.use_cache:
            cache_idx = idx % self.cache_size
            if cache_idx in self.cache:
                return self.cache[cache_idx]
        
        # キャッシュにない場合は生成
        return self._generate_sample()
