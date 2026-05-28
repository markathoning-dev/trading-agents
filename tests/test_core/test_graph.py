import pytest
from trading_agent.core.graph import (
    build_graph,
    build_agent_graph,
    PriceMarketAdapter,
    LOBMarketAdapter,
    BarMarketAdapter,
    MarketAdapter,
)
from trading_agent.core.state import AgentState
from trading_agent.core.schemas import TradeDecision
from trading_agent.models.mock import MockGateway


def _initial_state(price=100.0, cash=10000.0) -> AgentState:
    return {
        "cash": cash, "shares": 0, "price": price,
        "price_history": [price], "portfolio_values": [cash],
        "peak_value": cash, "action": "", "trade_cost": 0.0,
        "reward": 0.0, "total_reward": 0.0, "step": 0, "done": False,
    }


def test_graph_compilation():
    graph = build_graph(llm=None, fee_rate=0.001, risk_lambda=0.1, max_steps=3)
    assert graph is not None

def test_graph_execution():
    llm = MockGateway().get_langchain_llm()
    graph = build_graph(llm, fee_rate=0.001, risk_lambda=0.1, max_steps=3)
    result = graph.invoke(_initial_state())
    assert result["step"] >= 3
    assert isinstance(result["total_reward"], float)


def test_build_agent_graph_price_adapter():
    llm = MockGateway().get_langchain_llm()
    graph = build_agent_graph(
        llm, market=PriceMarketAdapter(), max_steps=3,
    )
    result = graph.invoke(_initial_state())
    assert result["step"] >= 3


def test_build_agent_graph_with_custom_adapter():
    class FixedPriceAdapter:
        has_market_data_node = False

        def update_state(self, state):
            return {**state, "price": 105.0}

        def format_prompt(self, state):
            return f"price={state['price']:.2f}"

        def execute_trade(self, state, decision, fee_rate):
            from trading_agent.core.nodes import execute_trade
            return execute_trade(state, decision, fee_rate)

    llm = MockGateway().get_langchain_llm()
    graph = build_agent_graph(llm, market=FixedPriceAdapter(), max_steps=2)
    result = graph.invoke(_initial_state())
    assert result["step"] >= 2


def test_market_adapter_protocol_compliance():
    adapters = [PriceMarketAdapter(), LOBMarketAdapter(), BarMarketAdapter()]
    for adapter in adapters:
        assert isinstance(adapter, MarketAdapter)


def test_price_adapter_does_not_modify_state():
    adapter = PriceMarketAdapter()
    state = _initial_state()
    result = adapter.update_state(state)
    assert result is state


def test_price_adapter_prompt():
    adapter = PriceMarketAdapter()
    state = _initial_state(price=123.45)
    prompt = adapter.format_prompt(state)
    assert "price=123.45" in prompt
    assert "cash=10000.00" in prompt


def test_lob_adapter_prompt_with_lob():
    adapter = LOBMarketAdapter()
    state = {**_initial_state(), "lob_bid": 99.5, "lob_ask": 100.5, "lob_spread": 1.0, "lob_mid": 100.0}
    prompt = adapter.format_prompt(state)
    assert "LOB bid=99.50" in prompt
    assert "LOB ask=100.50" in prompt


def test_lob_adapter_prompt_without_lob():
    adapter = LOBMarketAdapter()
    state = _initial_state()
    prompt = adapter.format_prompt(state)
    assert "price=100.00" in prompt
    assert "LOB" not in prompt


def test_bar_adapter_prompt_with_bar():
    adapter = BarMarketAdapter()
    state = {**_initial_state(), "bar_open": 100.0, "bar_high": 105.0, "bar_low": 98.0, "bar_close": 103.0, "bar_volume": 50000}
    prompt = adapter.format_prompt(state)
    assert "O=100.00" in prompt
    assert "H=105.00" in prompt
    assert "L=98.00" in prompt
    assert "C=103.00" in prompt


def test_bar_adapter_prompt_without_bar():
    adapter = BarMarketAdapter()
    state = _initial_state()
    prompt = adapter.format_prompt(state)
    assert "price=100.00" in prompt
