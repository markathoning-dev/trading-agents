from __future__ import annotations

from trading_agent.core.state import AgentState
from trading_agent.nodes.registry import register_node


@register_node
class MomentumAnalyzer:
    name = "momentum_analyzer"
    position = "pre_trade"

    def __call__(self, state: AgentState) -> AgentState:
        prices = state.get("price_history", [])
        if len(prices) < 5:
            return {**state, "momentum_signal": 0.0}
        momentum = (prices[-1] / prices[-5] - 1) * 100
        return {**state, "momentum_signal": momentum}


@register_node
class ReversionAnalyzer:
    name = "reversion_analyzer"
    position = "pre_trade"

    def __call__(self, state: AgentState) -> AgentState:
        prices = state.get("price_history", [])
        if len(prices) < 20:
            return {**state, "reversion_signal": 0.0}
        mean_20 = sum(prices[-20:]) / 20
        std_20 = (sum((p - mean_20) ** 2 for p in prices[-20:]) / 20) ** 0.5
        if std_20 == 0:
            return {**state, "reversion_signal": 0.0}
        z_score = (prices[-1] - mean_20) / std_20
        return {**state, "reversion_signal": -z_score}


@register_node
class VolatilityAnalyzer:
    name = "volatility_analyzer"
    position = "pre_trade"

    def __call__(self, state: AgentState) -> AgentState:
        prices = state.get("price_history", [])
        if len(prices) < 10:
            return {**state, "volatility_signal": 0.0}
        returns = [(prices[i] / prices[i - 1]) - 1 for i in range(-9, 0)]
        mean_return = sum(returns) / len(returns)
        volatility = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5
        return {**state, "volatility_signal": volatility * 100}


@register_node
class HoldReinforcer:
    name = "hold_reinforcer"
    position = "pre_trade"

    def __call__(self, state: AgentState) -> AgentState:
        prices = state.get("price_history", [])
        portfolio_values = state.get("portfolio_values", [])

        hold_bias = 0.0
        if len(portfolio_values) >= 2:
            current = portfolio_values[-1]
            peak = state.get("peak_value", current)
            if peak > 0:
                drawdown = (peak - current) / peak
                hold_bias = drawdown * 100

        return {**state, "hold_bias": hold_bias}
