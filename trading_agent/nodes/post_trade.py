from __future__ import annotations

from trading_agent.core.state import AgentState
from trading_agent.nodes.registry import register_node


@register_node
class StopLossSentinel:
    name = "stop_loss_sentinel"
    position = "post_trade"

    def __init__(self, threshold: float = 0.10):
        self.threshold = threshold

    def __call__(self, state: AgentState) -> AgentState:
        portfolio_values = state.get("portfolio_values", [])
        if len(portfolio_values) < 2:
            return state

        current = portfolio_values[-1]
        peak = state.get("peak_value", current)
        if peak <= 0:
            return state

        drawdown = (peak - current) / peak
        if drawdown > self.threshold:
            cash = state["cash"] + state["shares"] * state["price"]
            return {
                **state,
                "cash": cash,
                "shares": 0,
                "action": "STOP_LOSS",
                "trade_cost": 0.0,
            }
        return state


@register_node
class PositionSizer:
    name = "position_sizer"
    position = "post_trade"

    def __init__(self, max_position_pct: float = 0.5, fee_rate: float = 0.001):
        self.max_position_pct = max_position_pct
        self.fee_rate = fee_rate

    def __call__(self, state: AgentState) -> AgentState:
        portfolio_value = state["cash"] + state["shares"] * state["price"]
        if portfolio_value <= 0:
            return state

        max_position_value = portfolio_value * self.max_position_pct
        current_position_value = state["shares"] * state["price"]

        if current_position_value > max_position_value:
            shares_to_sell = int((current_position_value - max_position_value) / state["price"])
            shares_to_sell = min(shares_to_sell, state["shares"])
            if shares_to_sell > 0:
                proceeds = shares_to_sell * state["price"]
                fee = proceeds * self.fee_rate
                return {
                    **state,
                    "cash": state["cash"] + proceeds - fee,
                    "shares": state["shares"] - shares_to_sell,
                    "action": f"POSITION_TRIM {shares_to_sell}",
                    "trade_cost": fee,
                }
        return state


@register_node
class TrendFilter:
    name = "trend_filter"
    position = "pre_trade"

    def __init__(self, lookback: int = 20, threshold: float = 5.0):
        self.lookback = lookback
        self.threshold = threshold

    def __call__(self, state: AgentState) -> AgentState:
        prices = state.get("price_history", [])

        if len(prices) < self.lookback:
            return {**state, "trend_signal": 0.0, "trend_allowed_buy": True, "trend_allowed_sell": True}

        trend = (prices[-1] / prices[-self.lookback] - 1) * 100

        return {
            **state,
            "trend_signal": trend,
            "trend_allowed_buy": trend >= -self.threshold,
            "trend_allowed_sell": trend <= self.threshold,
        }


@register_node
class PanicSell:
    name = "panic_sell"
    position = "post_trade"

    def __init__(self, lookback: int = 3, required_losses: int = 3):
        self.lookback = lookback
        self.required_losses = required_losses

    def __call__(self, state: AgentState) -> AgentState:
        portfolio_values = state.get("portfolio_values", [])
        if len(portfolio_values) < self.lookback + 1:
            return state

        recent_losses = 0
        for i in range(-self.lookback, 0):
            if portfolio_values[i] < portfolio_values[i - 1]:
                recent_losses += 1

        if recent_losses >= self.required_losses:
            cash = state["cash"] + state["shares"] * state["price"]
            return {
                **state,
                "cash": cash,
                "shares": 0,
                "action": "PANIC_SELL",
                "trade_cost": 0.0,
            }
        return state
