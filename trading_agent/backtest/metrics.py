import pandas as pd
import numpy as np


def compute_sharpe(returns: pd.Series, annual_factor: int = 252) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(annual_factor))


def compute_max_drawdown(prices: pd.Series) -> float:
    cumulative_max = prices.expanding().max()
    drawdowns = (prices - cumulative_max) / cumulative_max
    return float(drawdowns.min())
