from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset


FEATURE_OPEN = 0
FEATURE_HIGH = 1
FEATURE_LOW = 2
FEATURE_CLOSE = 3
FEATURE_VOLUME = 4
FEATURE_VWAP = 5
FEATURE_DIM = 6


@dataclass
class Bar:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float


class BarFeatureExtractor:
    def __init__(self, log_norm: float = 15.0, fixed_ref: float | None = None):
        self.log_norm = log_norm
        self.fixed_ref = fixed_ref

    def __call__(self, bars: list[Bar]) -> np.ndarray:
        if not bars:
            raise ValueError("bars list must not be empty")
        ref = self.fixed_ref if self.fixed_ref is not None else (bars[0].open if bars[0].open > 0 else 100.0)
        last = bars[-1]
        return np.array([
            last.open / ref,
            last.high / ref,
            last.low / ref,
            last.close / ref,
            np.log1p(last.volume) / self.log_norm,
            last.vwap / ref,
        ], dtype=np.float32)


class BarDataset(Dataset):
    def __init__(self, bars: list[Bar], seq_len: int = 1, fixed_ref: float | None = None):
        self.bars = bars
        self.seq_len = seq_len
        self.ref_price = fixed_ref if fixed_ref is not None else (bars[0].open if bars else 100.0)
        self.extractor = BarFeatureExtractor(fixed_ref=self.ref_price)
        self.features = [self.extractor([b]) for b in self.bars]
        self.feat_dim = self.features[0].shape[0] if self.features else 0

    def __len__(self) -> int:
        return max(0, len(self.bars) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        state = np.concatenate(self.features[idx:idx + self.seq_len])
        next_bar = self.bars[idx + self.seq_len]
        target = self.extractor([next_bar])
        return torch.tensor(state, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

    def get_feature_dim(self) -> int:
        return self.feat_dim * self.seq_len

    def get_target_dim(self) -> int:
        return self.feat_dim
