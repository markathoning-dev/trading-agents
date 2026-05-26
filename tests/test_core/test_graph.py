from trading_agent.core.graph import build_graph
from trading_agent.core.state import AgentState
from trading_agent.models.mock import MockGateway

def test_graph_compilation():
    graph = build_graph(llm=None, fee_rate=0.001, risk_lambda=0.1, max_steps=3)
    assert graph is not None

def test_graph_execution():
    llm = MockGateway().get_langchain_llm()
    graph = build_graph(llm, fee_rate=0.001, risk_lambda=0.1, max_steps=3)
    initial: AgentState = {
        "cash": 10000.0, "shares": 0, "price": 100.0,
        "price_history": [100.0], "portfolio_values": [10000.0],
        "peak_value": 10000.0, "action": "", "trade_cost": 0.0,
        "reward": 0.0, "total_reward": 0.0, "step": 0, "done": False,
    }
    result = graph.invoke(initial)
    assert result["step"] >= 3
    assert isinstance(result["total_reward"], float)
