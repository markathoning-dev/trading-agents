import os
from langchain_litellm import ChatLiteLLM

class KiloGateway:
    def __init__(self, model_name: str = "openai/gpt-4o-mini", temperature: float = 0):
        self.model_name = model_name
        self.temperature = temperature
        raw_key = os.environ.get("KILO_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        if not raw_key:
            try:
                from trading_agent.config.settings import settings
                raw_key = settings.openai_api_key or ""
            except Exception:
                pass
        self.api_key = raw_key.strip() if raw_key else None
        self.api_base = os.environ.get("KILO_API_BASE", "").strip() or None
        if not self.api_base:
            try:
                from trading_agent.config.settings import settings
                self.api_base = getattr(settings, 'kilo_api_base', None) or None
            except Exception:
                pass

    def get_langchain_llm(self):
        kwargs = dict(model=self.model_name, temperature=self.temperature)
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        return ChatLiteLLM(**kwargs)
