import pandas as pd
from trading_agent.market.simulators import RandomWalkMarket, HistoricalMarket
import pytest

def test_random_walk():
    m = RandomWalkMarket(start_price=100, drift=0, volatility=0, seed=42)
    assert m.step() == 100.0

def test_random_walk_positive():
    m = RandomWalkMarket(start_price=100, drift=0.1, volatility=1.0, seed=42)
    prices = [m.step() for _ in range(10)]
    assert all(p > 0 for p in prices)

def test_historical_market():
    s = pd.Series([100.0, 101.0, 102.0])
    m = HistoricalMarket(s)
    assert m.step() == 100.0
    assert m.step() == 101.0

def test_historical_stop():
    s = pd.Series([100.0])
    m = HistoricalMarket(s)
    m.step()
    with pytest.raises(StopIteration):
        m.step()

def test_historical_reset():
    s = pd.Series([100.0, 101.0])
    m = HistoricalMarket(s)
    m.step()
    m.reset()
    assert m.step() == 100.0
