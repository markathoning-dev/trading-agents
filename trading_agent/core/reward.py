def multicomponent_reward(
    old_value: float, new_value: float, peak_value: float,
    trade_cost: float, risk_penalty_lambda: float = 0.1,
) -> float:
    profit = new_value - old_value
    drawdown = max(0.0, peak_value - new_value)
    risk_penalty = risk_penalty_lambda * drawdown
    return profit - risk_penalty - trade_cost


def aggressive_reward(
    old_value: float, new_value: float, peak_value: float,
    trade_cost: float, volatility_lambda: float = 0.2,
) -> float:
    profit = new_value - old_value
    volatility_bonus = volatility_lambda * abs(profit)
    return profit + volatility_bonus - trade_cost
