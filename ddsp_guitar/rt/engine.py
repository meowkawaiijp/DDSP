import torch
from collections import deque

from ddsp_guitar.model import GuitarDDSP


class RTBlockEngine:
    """
    Minimal real-time block engine with phase continuity.
    """

    def __init__(self, sample_rate=48000, block_size=256):
        self.model = GuitarDDSP(sample_rate=sample_rate)
        self.sr = sample_rate
        self.block = block_size
        self._phase = None

    @torch.no_grad()
    def process_block(self, x_block: torch.Tensor, f0_block: torch.Tensor, loud_block: torch.Tensor) -> torch.Tensor:
        # x_block: (1, B), f0_block: (1, B), loud_block: (1,B)
        y, self._phase = self.model.forward_with_phase(x_block, f0_block, loud_block, initial_phase=self._phase)
        return y
