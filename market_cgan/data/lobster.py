import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional


LOBSTER_LEVELS = 10


@dataclass
class LOBSnapshot:
    timestamp: float
    bid_prices: np.ndarray
    bid_volumes: np.ndarray
    ask_prices: np.ndarray
    ask_volumes: np.ndarray


def parse_lobster_orderbook(path: str) -> list[LOBSnapshot]:
    df = pd.read_csv(path, header=0)
    snapshots = []
    for _, row in df.iterrows():
        bid_prices = np.array([row[f"bid_px_{i+1}"] for i in range(LOBSTER_LEVELS)], dtype=np.float32)
        bid_volumes = np.array([row[f"bid_size_{i+1}"] for i in range(LOBSTER_LEVELS)], dtype=np.float32)
        ask_prices = np.array([row[f"ask_px_{i+1}"] for i in range(LOBSTER_LEVELS)], dtype=np.float32)
        ask_volumes = np.array([row[f"ask_size_{i+1}"] for i in range(LOBSTER_LEVELS)], dtype=np.float32)
        snapshots.append(LOBSnapshot(
            timestamp=row.get("timestamp", 0.0),
            bid_prices=bid_prices,
            bid_volumes=bid_volumes,
            ask_prices=ask_prices,
            ask_volumes=ask_volumes,
        ))
    return snapshots


def generate_sample_lob_data(num_snapshots: int = 1000) -> list[LOBSnapshot]:
    rng = np.random.default_rng(seed=42)
    snapshots = []
    mid = 100.0
    for t in range(num_snapshots):
        mid += rng.normal(0, 0.1)
        spread = abs(rng.normal(0.5, 0.2)) + 0.01
        best_bid = mid - spread / 2
        best_ask = mid + spread / 2

        bid_prices = best_bid - np.arange(LOBSTER_LEVELS) * 0.1
        ask_prices = best_ask + np.arange(LOBSTER_LEVELS) * 0.1
        bid_volumes = np.abs(rng.exponential(scale=20, size=LOBSTER_LEVELS)) + 1
        ask_volumes = np.abs(rng.exponential(scale=20, size=LOBSTER_LEVELS)) + 1

        snapshots.append(LOBSnapshot(
            timestamp=float(t),
            bid_prices=bid_prices.astype(np.float32),
            bid_volumes=bid_volumes.astype(np.float32),
            ask_prices=ask_prices.astype(np.float32),
            ask_volumes=ask_volumes.astype(np.float32),
        ))
    return snapshots


class LobsterDataset(Dataset):
    def __init__(self, snapshots: list[LOBSnapshot], feature_extractor, seq_len: int = 1):
        self.snapshots = snapshots
        self.feature_extractor = feature_extractor
        self.seq_len = seq_len
        self.features = [feature_extractor(s) for s in snapshots]
        self.feat_dim = self.features[0].shape[0] if self.features else 0
        self.action_dim = 8

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
