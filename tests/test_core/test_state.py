from trading_agent.core.state import AgentState

def test_agent_state_defaults():
    state: AgentState = {
        "cash": 10000.0,
        "shares": 0,
        "price": 100.0,
        "price_history": [100.0],
        "portfolio_values": [10000.0],
        "peak_value": 10000.0,
        "action": "",
        "trade_cost": 0.0,
        "reward": 0.0,
        "total_reward": 0.0,
        "step": 0,
        "done": False,
    }
    assert state["cash"] == 10000.0
    assert state["shares"] == 0
    assert state["price"] == 100.0
    assert state["peak_value"] == 10000.0
    assert state["total_reward"] == 0.0
    assert state["done"] is False
