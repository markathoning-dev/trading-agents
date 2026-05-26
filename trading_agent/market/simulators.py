import random
import pandas as pd
from trading_agent.market.base import MarketDataSource

class RandomWalkMarket(MarketDataSource):
    def __init__(self, start_price=100, drift=0.0, volatility=1.0, seed=42):
        random.seed(seed)
        self.price = start_price
        self.drift = drift
        self.volatility = volatility

    def step(self) -> float:
        current = self.price
        self.price += self.drift + random.gauss(0, self.volatility)
        self.price = max(self.price, 1e-6)
        return current

class HistoricalMarket(MarketDataSource):
    def __init__(self, price_series: pd.Series):
        self.prices = price_series.values
        self.idx = 0

    def step(self) -> float:
        if self.idx >= len(self.prices):
            raise StopIteration("End of data")
        price = self.prices[self.idx]
        self.idx += 1
        return float(price)

    def reset(self) -> None:
        self.idx = 0
