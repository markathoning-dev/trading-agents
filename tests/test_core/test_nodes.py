from trading_agent.core.state import AgentState
from trading_agent.core.schemas import TradeDecision
from trading_agent.core.nodes import execute_trade, fetch_price, calculate_reward


def make_state(**overrides) -> AgentState:
    defaults: AgentState = {
        "cash": 10000.0, "shares": 0, "price": 100.0,
        "price_history": [100.0], "portfolio_values": [10000.0],
        "peak_value": 10000.0, "action": "", "trade_cost": 0.0,
        "reward": 0.0, "total_reward": 0.0, "step": 0, "done": False,
    }
    defaults.update(overrides)
    return defaults


def test_buy_trade():
    state = make_state()
    result = execute_trade(state, TradeDecision(action="BUY", quantity=10, reason="bullish"), fee_rate=0.001)
    assert result["cash"] == 8999.0  # 10000 - 1000 (cost) - 1 (fee)
    assert result["shares"] == 10
    assert result["trade_cost"] == 1.0


def test_sell_trade():
    state = make_state(cash=0.0, shares=10)
    result = execute_trade(state, TradeDecision(action="SELL", quantity=5, reason="bearish"), fee_rate=0.001)
    assert result["cash"] == 499.5
    assert result["shares"] == 5


def test_hold_trade():
    state = make_state()
    result = execute_trade(state, TradeDecision(action="HOLD", quantity=0, reason="neutral"), fee_rate=0.001)
    assert result["cash"] == 10000.0
    assert result["shares"] == 0
    assert result["trade_cost"] == 0.0


def test_insufficient_cash_buy():
    state = make_state(cash=50.0)
    result = execute_trade(state, TradeDecision(action="BUY", quantity=10, reason="bullish"), fee_rate=0.001)
    assert result["cash"] == 50.0
    assert result["shares"] == 0


def test_fetch_price():
    state = make_state()
    result = fetch_price(state, 105.0)
    assert result["price"] == 105.0
    assert result["step"] == 1
    assert len(result["price_history"]) == 2  # appended, not replaced
    assert result["portfolio_values"][-1] == 10000.0


def test_calculate_reward():
    state = make_state(price=100.0, price_history=[100.0, 110.0], portfolio_values=[10000.0, 11000.0])
    result = calculate_reward(state, risk_penalty_lambda=0.1)
    assert result["reward"] == 1000.0
    assert result["total_reward"] == 1000.0
    assert result["peak_value"] == 11000.0
