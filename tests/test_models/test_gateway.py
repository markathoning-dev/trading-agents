from trading_agent.models.gateway import KiloGateway

def test_gateway_initialization():
    gw = KiloGateway(model_name="openai/gpt-4o-mini", temperature=0)
    assert gw.model_name == "openai/gpt-4o-mini"

def test_gateway_returns_llm():
    gw = KiloGateway()
    llm = gw.get_langchain_llm()
    assert hasattr(llm, "invoke")
