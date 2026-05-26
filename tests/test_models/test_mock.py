from trading_agent.models.mock import MockGateway
from trading_agent.core.schemas import TradeDecision

def test_mock_gateway_returns_hold():
    gateway = MockGateway()
    llm = gateway.get_langchain_llm()
    structured = llm.with_structured_output(TradeDecision)
    decision = structured.invoke("test")
    assert decision.action == "HOLD"
    assert decision.quantity == 0
