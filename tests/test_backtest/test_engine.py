import pandas as pd
from trading_agent.models.mock import MockGateway
from trading_agent.backtest.engine import backtest_agent


def test_backtest_returns_metrics():
    prices = pd.Series([float(100 + i) for i in range(10)])
    llm = MockGateway().get_langchain_llm()
    metrics = backtest_agent(prices, llm=llm, fee_rate=0.001, risk_lambda=0.1)
    assert "final_portfolio_value" in metrics
    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert isinstance(metrics["total_return"], float)


def test_backtest_nonzero_steps():
    prices = pd.Series([float(100 + i) for i in range(10)])
    llm = MockGateway().get_langchain_llm()
    metrics = backtest_agent(prices, llm=llm, max_steps=5)
    assert metrics["num_steps"] >= 1
