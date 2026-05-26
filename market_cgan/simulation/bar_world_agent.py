from __future__ import annotations

import torch
import numpy as np
from market_cgan.models.bar_generator import BarGenerator
from market_cgan.simulation.bar_exchange import BarExchange
from market_cgan.data.bar import Bar, BarFeatureExtractor


class BarWorldAgent:
    def __init__(self, generator: BarGenerator, exchange: BarExchange, noise_dim: int = 64, device: str | None = None, fixed_ref: float | None = None):
        self.generator = generator
        self.exchange = exchange
        self.noise_dim = noise_dim
        self.extractor = BarFeatureExtractor(fixed_ref=fixed_ref)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.generator.to(self.device)
        self.generator.eval()

    def _get_features(self) -> np.ndarray:
        window = self.exchange.get_window(min(3, len(self.exchange.bars)))
        if not window:
            return np.zeros(6, dtype=np.float32)
        return self.extractor(window)

    def step(self) -> Bar:
        features = self._get_features()
        feat_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        noise = torch.randn(1, self.noise_dim, device=self.device)
        with torch.no_grad():
            bar_tensor = self.generator(noise, feat_tensor).squeeze(0)
        o, h, l, c, v, vw = bar_tensor.tolist()
        ts = int(self.exchange.bars[-1].timestamp) + 1 if self.exchange.bars else 1
        new_bar = Bar(timestamp=ts, open=o, high=h, low=l, close=c, volume=v, vwap=vw)
        self.exchange.append_bar(new_bar)
        return new_bar

    def reset(self):
        self.exchange.reset()