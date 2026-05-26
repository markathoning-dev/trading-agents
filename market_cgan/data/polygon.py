from __future__ import annotations

import os
import numpy as np
import torch

from torch.utils.data import Dataset
from typing import Optional
from market_cgan.data.bar import Bar
from market_cgan.data.lobster import LOBSnapshot
from market_cgan.data.features import MarketFeatureExtractor

BBO_LEVELS = 10


def bbo_to_multilevel(
    bid_price: float,
    bid_size: float,
    ask_price: float,
    ask_size: float,
    levels: int = BBO_LEVELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mid = (bid_price + ask_price) / 2.0
    spread = max(ask_price - bid_price, 0.001)
    step = spread * 0.5

    bid_prices = np.array([bid_price - i * step for i in range(levels)], dtype=np.float32)
    ask_prices = np.array([ask_price + i * step for i in range(levels)], dtype=np.float32)
    decay = np.array([max(0.7 ** i, 0.05) for i in range(levels)], dtype=np.float32)
    bid_volumes = np.maximum(bid_size * decay + np.random.exponential(2, levels).astype(np.float32), 1.0)
    ask_volumes = np.maximum(ask_size * decay + np.random.exponential(2, levels).astype(np.float32), 1.0)

    return bid_prices, bid_volumes, ask_prices, ask_volumes


class PolygonDataSource:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from polygon import RESTClient
            self._client = RESTClient(self.api_key)
        return self._client

    def fetch_quotes(
        self,
        ticker: str,
        date_str: str,
    ) -> list[LOBSnapshot]:
        snapshots: list[LOBSnapshot] = []
        seen_timestamps: set[int] = set()

        raw = self.client.list_quotes(ticker, timestamp=date_str)
        quotes = raw if isinstance(raw, list) else raw.get("results", []) if isinstance(raw, dict) else []

        for quote in quotes:
            if isinstance(quote, dict):
                ts = int(quote.get("t", 0) or 0)
                bid_px = float(quote.get("bp", 0) or 0)
                bid_sz = float(quote.get("bs", 0) or 0)
                ask_px = float(quote.get("ap", 0) or 0)
                ask_sz = float(quote.get("as", 0) or 0)
            else:
                ts = int(getattr(quote, "timestamp", 0) or 0)
                bid_px = float(getattr(quote, "bid_price", 0) or 0)
                bid_sz = float(getattr(quote, "bid_size", 0) or 0)
                ask_px = float(getattr(quote, "ask_price", 0) or 0)
                ask_sz = float(getattr(quote, "ask_size", 0) or 0)

            if ts in seen_timestamps or bid_px <= 0 or ask_px <= 0:
                continue
            seen_timestamps.add(ts)

            bp, bv, ap, av = bbo_to_multilevel(bid_px, bid_sz, ask_px, ask_sz)
            snapshots.append(LOBSnapshot(timestamp=ts, bid_prices=bp, bid_volumes=bv, ask_prices=ap, ask_volumes=av))

        return snapshots

    def fetch_aggregates(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        timespan: str = "minute",
    ) -> list[Bar]:
        bars: list[Bar] = []
        agg_list = self.client.get_aggs(ticker, 1, timespan, start_date, end_date)
        for agg in agg_list:
            bars.append(Bar(
                timestamp=int(getattr(agg, "timestamp", 0) or 0),
                open=float(getattr(agg, "open", 0) or 0),
                high=float(getattr(agg, "high", 0) or 0),
                low=float(getattr(agg, "low", 0) or 0),
                close=float(getattr(agg, "close", 0) or 0),
                volume=float(getattr(agg, "volume", 0) or 0),
                vwap=float(getattr(agg, "vwap", 0) or 0),
            ))
        return bars

    def fetch_trades(self, ticker: str, date_str: str) -> list[dict]:
        trades = []
        raw = self.client.list_trades(ticker, timestamp=date_str)
        items = raw if isinstance(raw, list) else raw.get("results", []) if isinstance(raw, dict) else []
        for t in items:
            trades.append({
                "price": float(t.get("p", 0) if isinstance(t, dict) else getattr(t, "price", 0) or 0),
                "size": float(t.get("s", 0) if isinstance(t, dict) else getattr(t, "size", 0) or 0),
                "timestamp": int(t.get("t", 0) if isinstance(t, dict) else getattr(t, "timestamp", 0) or 0),
            })
        return trades


class PolygonDataset(Dataset):
    def __init__(
        self,
        ticker: str,
        dates: list[str],
        api_key: str | None = None,
        seq_len: int = 1,
        feature_extractor: Optional[MarketFeatureExtractor] = None,
        max_quotes_per_date: int = 5000,
    ):
        self.source = PolygonDataSource(api_key)
        self.seq_len = seq_len
        self.extractor = feature_extractor or MarketFeatureExtractor()
        self.action_dim = 8

        all_snapshots: list[LOBSnapshot] = []
        for d in dates:
            snaps = self.source.fetch_quotes(ticker, d)
            all_snapshots.extend(snaps[:max_quotes_per_date])

        all_snapshots.sort(key=lambda s: s.timestamp)
        self.snapshots = all_snapshots
        self.features = [self.extractor(s) for s in self.snapshots]
        self.feat_dim = self.features[0].shape[0] if self.features else 0

    def __len__(self) -> int:
        return max(0, len(self.snapshots) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.features[idx]
        next_snapshot = self.snapshots[idx + 1]
        action = self._snapshot_to_action(self.snapshots[idx], next_snapshot)
        return torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)

    def _snapshot_to_action(self, curr: LOBSnapshot, next_: LOBSnapshot) -> np.ndarray:
        mid_curr = (curr.ask_prices[0] + curr.bid_prices[0]) / 2
        mid_next = (next_.ask_prices[0] + next_.bid_prices[0]) / 2

        bid_vol_change = next_.bid_volumes.sum() - curr.bid_volumes.sum()
        ask_vol_change = next_.ask_volumes.sum() - curr.ask_volumes.sum()

        if bid_vol_change > ask_vol_change and mid_next > mid_curr:
            action_type = 0
            side = 0
            price_offset = (next_.bid_prices[0] - mid_curr) / (mid_curr + 1e-8)
            quantity = abs(bid_vol_change)
        elif ask_vol_change > bid_vol_change and mid_next < mid_curr:
            action_type = 0
            side = 1
            price_offset = (next_.ask_prices[0] - mid_curr) / (mid_curr + 1e-8)
            quantity = abs(ask_vol_change)
        elif abs(mid_next - mid_curr) < 0.01:
            action_type = 2 if abs(bid_vol_change) > 0 else 3
            side = 0
            price_offset = 0.0
            quantity = max(abs(bid_vol_change), abs(ask_vol_change))
        else:
            action_type = 1
            if mid_next > mid_curr:
                side = 0
                price_offset = (next_.bid_prices[0] - mid_curr) / (mid_curr + 1e-8)
                quantity = abs(bid_vol_change)
            else:
                side = 1
                price_offset = (next_.ask_prices[0] - mid_curr) / (mid_curr + 1e-8)
                quantity = abs(ask_vol_change)

        at = np.zeros(4, dtype=np.float32)
        at[action_type % 4] = 1.0
        sd = np.zeros(2, dtype=np.float32)
        sd[side % 2] = 1.0
        return np.concatenate([at, sd, [np.clip(price_offset, -1.0, 1.0)], [np.clip(quantity / 100, 0, 1)]])

    def get_feature_dim(self) -> int:
        return self.feat_dim

    def get_action_dim(self) -> int:
        return self.action_dim


def live_to_dataset(
    ticker: str,
    duration_seconds: int = 300,
    api_key: str | None = None,
) -> list[LOBSnapshot]:
    import time
    import json
    from polygon import WebSocketClient
    from polygon.enums import StreamCluster

    api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
    snapshots: list[LOBSnapshot] = []

    def handle_msg(msg):
        nonlocal snapshots
        try:
            data = json.loads(msg) if isinstance(msg, str) else msg
            if isinstance(data, list):
                for d in data:
                    _process_quote_msg(d, snapshots)
            else:
                _process_quote_msg(data, snapshots)
        except Exception:
            pass

    def _process_quote_msg(d: dict, out: list):
        if d.get("ev") != "Q":
            return
        bid_px = float(d.get("bp", 0) or 0)
        ask_px = float(d.get("ap", 0) or 0)
        if bid_px <= 0 or ask_px <= 0:
            return
        bid_sz = float(d.get("bs", 0) or 0)
        ask_sz = float(d.get("as", 0) or 0)
        ts = int(d.get("t", 0) or 0)
        bp, bv, ap, av = bbo_to_multilevel(bid_px, bid_sz, ask_px, ask_sz)
        out.append(LOBSnapshot(timestamp=ts, bid_prices=bp, bid_volumes=bv, ask_prices=ap, ask_volumes=av))

    ws = WebSocketClient(api_key, StreamCluster.STOCKS, on_message=handle_msg)
    ws.start_stream_thread()
    ws.subscribe_quotes(ticker)
    time.sleep(duration_seconds)
    ws.close()

    return snapshots
