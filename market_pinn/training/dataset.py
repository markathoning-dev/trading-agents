import torch
from torch.utils.data import Dataset
import pandas as pd


class MarketDataset(Dataset):
    def __init__(self, price_series: pd.Series):
        prices = price_series.values
        self.t = torch.linspace(0, 1, len(prices)).unsqueeze(1).float()
        self.s = torch.tensor(prices, dtype=torch.float32).unsqueeze(1)
        self.targets = self.s.clone()

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        return self.t[idx], self.s[idx], self.targets[idx]
