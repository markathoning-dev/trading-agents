from __future__ import annotations

import numpy as np
from market_cgan.data.bar import Bar


class BarExchange:
    def __init__(self, window: int = 10):
        self.bars: list[Bar] = []
        self.window = window

    def append_bar(self, bar: Bar):
        self.bars.append(bar)

    def get_state(self) -> dict:
        if not self.bars:
            return {"open": None, "high": None, "low": None, "close": None, "volume": None, "vwap": None}
        last = self.bars[-1]
        return {
            "open": last.open, "high": last.high, "low": last.low,
            "close": last.close, "volume": last.volume, "vwap": last.vwap,
        }

    def get_window(self, n: int | None = None) -> list[Bar]:
        n = n or self.window
        return self.bars[-n:]

    def reset(self):
        self.bars.clear()