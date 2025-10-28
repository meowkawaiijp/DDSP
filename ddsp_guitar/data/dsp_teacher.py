import torch
from torch import nn

from ddsp_guitar.modules.waveshaper import ParamTanhWS
from ddsp_guitar.modules.tonestack import ToneStack
from ddsp_guitar.modules.transient import TransientSeparator


class DSPTeacherChain(nn.Module):
    """
    Handcrafted DSP chain used to generate target audio from DI.
    Consists of transient separation -> waveshaper on attack -> tone stack on sustain -> mix.
    Parameters are provided per-sample for time variation or as scalars broadcasted over time.
    """

    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.ws = ParamTanhWS()
        self.ts = ToneStack(sample_rate)
        self.sep = TransientSeparator(kernel_size=15)
        self.sample_rate = sample_rate

    def forward(
        self,
        x: torch.Tensor,
        *,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        low_gain_db: torch.Tensor,
        mid_gain_db: torch.Tensor,
        mid_fc: torch.Tensor,
        mid_Q: torch.Tensor,
        high_gain_db: torch.Tensor,
        alpha_a: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N)
            all params: (B, N) or broadcastable
        Returns:
            y: (B, N)
        """
        a, s, _gate = self.sep(x)
        y_a = self.ws(a, alpha, beta)
        y_s = self.ts(s, low_gain_db, mid_gain_db, mid_fc, mid_Q, high_gain_db)
        mix = alpha_a
        return mix * y_a + (1.0 - mix) * y_s

    @staticmethod
    def _ensure_shape(param: torch.Tensor | float, shape, device):
        if not torch.is_tensor(param):
            param = torch.tensor(param, device=device)
        while param.dim() < len(shape):
            param = param.unsqueeze(0)
        # broadcast to (B,N)
        if param.shape[-1] == 1:
            param = param.expand(*shape)
        return param

    def random_params(self, batch: int, num_samples: int, device) -> dict:
        """
        Sample musically reasonable parameter ranges.
        """
        N = num_samples
        shape = (batch, N)
        def uni(lo, hi):
            return torch.rand(batch, 1, device=device) * (hi - lo) + lo
        alpha = self._ensure_shape(uni(0.5, 6.0), shape, device)
        beta = self._ensure_shape(uni(0.0, 0.8), shape, device)
        low = self._ensure_shape(uni(-9.0, 9.0), shape, device)
        mid_gain = self._ensure_shape(uni(-6.0, 6.0), shape, device)
        high = self._ensure_shape(uni(-9.0, 9.0), shape, device)
        mid_fc = self._ensure_shape(uni(300.0, 2000.0), shape, device)
        mid_Q = self._ensure_shape(uni(0.4, 2.0), shape, device)
        alpha_a = self._ensure_shape(uni(0.3, 0.7), shape, device)
        return {
            'alpha': alpha,
            'beta': beta,
            'low_gain_db': low,
            'mid_gain_db': mid_gain,
            'mid_fc': mid_fc,
            'mid_Q': mid_Q,
            'high_gain_db': high,
            'alpha_a': alpha_a,
        }
