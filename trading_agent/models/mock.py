from langchain_core.language_models.llms import LLM
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import BaseMessage
from typing import Optional, List, Any
from trading_agent.core.schemas import TradeDecision

class MockLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "mock"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        return '{"action": "HOLD", "quantity": 0, "reason": "mock"}'

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda messages: TradeDecision(action="HOLD", quantity=0, reason="mock"))

class MockGateway:
    def get_langchain_llm(self) -> LLM:
        return MockLLM()
