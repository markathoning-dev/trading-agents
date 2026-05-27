import pandas as pd
from trading_agent.models.mock import MockGateway
from trading_agent.backtest.engine import backtest_agent
from trading_agent.backtest.parallel import parallel_backtest, BacktestConfig


def test_parallel_backtest_returns_metrics():
    prices = pd.Series([float(100 + i) for i in range(10)])
    llm = MockGateway().get_langchain_llm()
    configs = [
        BacktestConfig(price_series=prices, llm=llm, fee_rate=0.001),
        BacktestConfig(price_series=prices, llm=llm, fee_rate=0.002),
    ]
    results = parallel_backtest(configs, max_workers=2)
    assert len(results) == 2
    for r in results:
        assert "final_portfolio_value" in r
        assert "total_return" in r
        assert isinstance(r["total_return"], float)


def test_parallel_matches_sequential():
    prices = pd.Series([float(100 + i) for i in range(10)])
    llm = MockGateway().get_langchain_llm()
    configs = [
        BacktestConfig(price_series=prices, llm=llm, max_steps=5),
        BacktestConfig(price_series=prices, llm=llm, max_steps=5),
    ]
    results = parallel_backtest(configs, max_workers=2)
    seq = backtest_agent(prices, llm=llm, max_steps=5)
    for r in results:
        assert r["num_steps"] == seq["num_steps"]
        assert abs(r["final_portfolio_value"] - seq["final_portfolio_value"]) < 0.01


def test_parallel_single_config():
    prices = pd.Series([float(100)] * 5)
    llm = MockGateway().get_langchain_llm()
    results = parallel_backtest([BacktestConfig(price_series=prices, llm=llm)], max_workers=1)
    assert len(results) == 1
    assert "total_return" in results[0]