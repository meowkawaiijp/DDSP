from dataclasses import dataclass


@dataclass
class AudioConfig:
    sample_rate: int = 48000
    frame_ms: float = 16.0
    block_size: int = 256

    @property
    def hop_size(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)
