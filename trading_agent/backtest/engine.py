import pandas as pd
from typing import Dict, Optional
from trading_agent.core.state import AgentState
from trading_agent.core.graph import build_graph
from trading_agent.core.reward import multicomponent_reward
from trading_agent.market.simulators import HistoricalMarket
from trading_agent.backtest.metrics import compute_sharpe, compute_max_drawdown
from trading_agent.core.cache import llm_cache


def backtest_agent(
    price_series: pd.Series,
    llm,
    initial_cash: float = 10000.0,
    fee_rate: float = 0.001,
    risk_lambda: float = 0.1,
    max_steps: Optional[int] = None,
    reward_fn=None,
) -> Dict:
    llm_cache.clear()
    market = HistoricalMarket(price_series.copy())
    steps = max_steps or len(price_series)
    graph = build_graph(llm, fee_rate=fee_rate, risk_lambda=risk_lambda, max_steps=1, reward_fn=reward_fn or multicomponent_reward)

    initial: AgentState = {
        "cash": initial_cash,
        "shares": 0,
        "price": float(price_series.iloc[0]),
        "price_history": [float(price_series.iloc[0])],
        "portfolio_values": [initial_cash],
        "peak_value": initial_cash,
        "action": "",
        "trade_cost": 0.0,
        "reward": 0.0,
        "total_reward": 0.0,
        "step": 0,
        "done": False,
    }

    state = initial
    for _ in range(steps):
        try:
            price = market.step()
        except StopIteration:
            state["done"] = True
            break
        state = {**state, "price": price}
        state = graph.invoke(state)

    final_price = float(price_series.iloc[-1])
    final_value = state["cash"] + state["shares"] * final_price
    total_return = (final_value / initial_cash) - 1
    price_returns = price_series.pct_change().dropna()
    sharpe = compute_sharpe(price_returns) if len(price_returns) > 1 else 0.0
    max_dd = compute_max_drawdown(price_series)

    return {
        "final_portfolio_value": final_value,
        "total_return": total_return,
        "cumulative_reward": state["total_reward"],
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "num_steps": state["step"],
        "final_cash": state["cash"],
        "final_shares": state["shares"],
    }
