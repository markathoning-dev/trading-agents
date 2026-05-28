import pandas as pd
import pytest
from trading_agent.models.mock import MockGateway
from trading_agent.backtest.engine import backtest_agent, create_initial_state, compute_strategy_metrics


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


def test_create_initial_state():
    state = create_initial_state(100.0, initial_cash=5000.0)
    assert state["cash"] == 5000.0
    assert state["shares"] == 0
    assert state["price"] == 100.0
    assert state["price_history"] == [100.0]
    assert state["portfolio_values"] == [5000.0]
    assert state["peak_value"] == 5000.0
    assert state["step"] == 0
    assert state["done"] is False


def test_create_initial_state_defaults():
    state = create_initial_state(50.0)
    assert state["cash"] == 10000.0
    assert state["price"] == 50.0


def test_compute_strategy_metrics_flat():
    pv = [10000.0] * 10
    prices = pd.Series([100.0] * 10)
    metrics = compute_strategy_metrics(pv, 10000.0, prices)
    assert metrics["final_portfolio_value"] == 10000.0
    assert metrics["total_return"] == 0.0
    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["max_drawdown"] == 0.0


def test_compute_strategy_metrics_growing():
    pv = [10000.0, 10100.0, 10200.0, 10300.0, 10400.0]
    prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
    metrics = compute_strategy_metrics(pv, 10000.0, prices)
    assert metrics["total_return"] == pytest.approx(0.04)
    assert metrics["sharpe_ratio"] > 0
    assert metrics["max_drawdown"] == 0.0


def test_compute_strategy_metrics_with_drawdown():
    pv = [10000.0, 11000.0, 9000.0, 9500.0, 8000.0]
    prices = pd.Series([100.0, 110.0, 90.0, 95.0, 80.0])
    metrics = compute_strategy_metrics(pv, 10000.0, prices)
    assert metrics["max_drawdown"] < 0
    assert metrics["max_drawdown"] > -0.3


def test_compute_strategy_metrics_uses_portfolio_not_asset():
    pv = [10000.0, 10000.0, 10000.0, 10000.0]
    prices = pd.Series([100.0, 110.0, 120.0, 130.0])
    metrics = compute_strategy_metrics(pv, 10000.0, prices)
    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["total_return"] == 0.0
