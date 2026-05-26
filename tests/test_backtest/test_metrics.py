import pandas as pd
from trading_agent.backtest.metrics import compute_sharpe, compute_max_drawdown


def test_sharpe_zero_vol():
    s = pd.Series([0.01] * 5)
    assert compute_sharpe(s) == 0.0


def test_sharpe_positive():
    s = pd.Series([0.01, 0.02, 0.015, 0.01, 0.005])
    sharpe = compute_sharpe(s)
    assert isinstance(sharpe, float)
    assert sharpe > 0


def test_max_drawdown():
    prices = pd.Series([100, 110, 90, 95, 80, 85])
    dd = compute_max_drawdown(prices)
    assert dd < 0
    assert dd > -0.3
