from trading_agent.core.schemas import TradeDecision
from pydantic import ValidationError
import pytest


def test_valid_buy():
    d = TradeDecision(action="BUY", quantity=10, reason="bullish")
    assert d.action == "BUY"
    assert d.quantity == 10


def test_invalid_negative_quantity():
    with pytest.raises(ValidationError):
        TradeDecision(action="BUY", quantity=-1)


def test_invalid_action():
    with pytest.raises(ValidationError):
        TradeDecision(action="MOON", quantity=0)
