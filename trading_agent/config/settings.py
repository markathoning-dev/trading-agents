from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    database_url: str = "sqlite:///data/trading.db"
    log_level: str = "INFO"
    default_model: str = "openai/gpt-4o-mini"
    fee_rate: float = 0.001
    risk_lambda: float = 0.1
    max_steps: int = 50
    openai_api_key: Optional[str] = None
    kilo_api_base: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()
