from pydantic import BaseModel, Field
from typing import Literal


class TradeDecision(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]
    quantity: int = Field(ge=0)
    reason: str


class LimitOrder(BaseModel):
    side: Literal["BUY", "SELL"]
    price: float = Field(gt=0)
    quantity: int = Field(gt=0)
    reason: str = Field(default="")


class MarketOrder(BaseModel):
    side: Literal["BUY", "SELL"]
    quantity: int = Field(gt=0)
    reason: str = Field(default="")
