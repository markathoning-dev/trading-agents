from __future__ import annotations

from trading_agent.core.state import AgentState
from trading_agent.nodes.registry import register_node


@register_node
class VolatilityReward:
    name = "volatility_reward"
    position = "reward"

    def __call__(self, state: AgentState) -> AgentState:
        portfolio_values = state.get("portfolio_values", [])
        if len(portfolio_values) < 2:
            return {**state, "reward": 0.0, "total_reward": state.get("total_reward", 0.0)}

        old_value = portfolio_values[-2]
        new_value = portfolio_values[-1]
        profit = new_value - old_value

        prices = state.get("price_history", [])
        if len(prices) >= 10:
            returns = [(prices[i] / prices[i - 1]) - 1 for i in range(-9, 0)]
            mean_return = sum(returns) / len(returns)
            volatility = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5
        else:
            volatility = 0.0

        volatility_bonus = 0.5 * abs(profit) * (1 + volatility * 10)
        reward = profit + volatility_bonus - state.get("trade_cost", 0.0)
        total = state.get("total_reward", 0.0) + reward

        peak = max(state.get("peak_value", 0), new_value)
        return {
            **state,
            "peak_value": peak,
            "reward": reward,
            "total_reward": total,
        }


@register_node
class DrawdownImmune:
    name = "drawdown_immune"
    position = "reward"

    def __call__(self, state: AgentState) -> AgentState:
        portfolio_values = state.get("portfolio_values", [])
        if len(portfolio_values) < 2:
            return {**state, "reward": 0.0, "total_reward": state.get("total_reward", 0.0)}

        old_value = portfolio_values[-2]
        new_value = portfolio_values[-1]
        profit = new_value - old_value

        reward = profit - state.get("trade_cost", 0.0)
        total = state.get("total_reward", 0.0) + reward

        peak = max(state.get("peak_value", 0), new_value)
        return {
            **state,
            "peak_value": peak,
            "reward": reward,
            "total_reward": total,
        }
