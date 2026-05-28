from __future__ import annotations

from trading_agent.core.state import AgentState
from trading_agent.core.schemas import TradeDecision, LimitOrder, MarketOrder
from trading_agent.core.reward import multicomponent_reward


def apply_trade(
    cash: float,
    shares: int,
    price: float,
    quantity: int,
    side: str,
    fee_rate: float = 0.001,
) -> tuple[float, int, float]:
    trade_cost = 0.0
    if side == "BUY":
        cost = quantity * price
        if cost <= cash:
            trade_cost = fee_rate * cost
            cash -= cost + trade_cost
            shares += quantity
    elif side == "SELL":
        qty = min(quantity, shares)
        proceeds = qty * price
        trade_cost = fee_rate * proceeds
        cash += proceeds - trade_cost
        shares -= qty
    return cash, shares, trade_cost


def fetch_price(state: AgentState, price: float) -> AgentState:
    pv = state["cash"] + state["shares"] * price
    return {
        **state,
        "price": price,
        "price_history": state["price_history"] + [price],
        "portfolio_values": state["portfolio_values"] + [pv],
        "step": state["step"] + 1,
    }


def execute_trade(state: AgentState, decision: TradeDecision, fee_rate: float = 0.001) -> AgentState:
    cash, shares, trade_cost = apply_trade(
        state["cash"], state["shares"], state["price"],
        decision.quantity, decision.action, fee_rate,
    )
    pv = cash + shares * state["price"]
    return {
        **state,
        "cash": cash,
        "shares": shares,
        "price": state["price"],
        "price_history": state["price_history"] + [state["price"]],
        "portfolio_values": state["portfolio_values"] + [pv],
        "action": f"{decision.action} {decision.quantity}",
        "trade_cost": trade_cost,
    }


def execute_lob_trade(
    state: AgentState,
    decision: TradeDecision | LimitOrder | MarketOrder,
    fee_rate: float = 0.001,
) -> AgentState:
    lob_bid = state.get("lob_bid", state["price"])
    lob_ask = state.get("lob_ask", state["price"])
    order_filled = False
    fill_price = state["price"]
    trade_cost = 0.0
    cash = state["cash"]
    shares = state["shares"]

    if isinstance(decision, (LimitOrder, MarketOrder)):
        if isinstance(decision, LimitOrder):
            target_price = decision.price
        else:
            target_price = lob_ask if decision.side == "BUY" else lob_bid

        if decision.side == "BUY":
            exec_price = min(target_price, lob_ask)
        else:
            exec_price = max(target_price, lob_bid)

        new_cash, new_shares, trade_cost = apply_trade(
            state["cash"], state["shares"], exec_price,
            decision.quantity, decision.side, fee_rate,
        )
        if new_cash != state["cash"] or new_shares != state["shares"]:
            cash, shares = new_cash, new_shares
            fill_price = exec_price
            order_filled = True
        action_str = f"{type(decision).__name__} {decision.side} {decision.quantity}"
    else:
        if decision.action == "BUY":
            exec_price = min(state["price"], lob_ask)
        elif decision.action == "SELL":
            exec_price = max(state["price"], lob_bid)
        else:
            exec_price = state["price"]

        new_cash, new_shares, trade_cost = apply_trade(
            state["cash"], state["shares"], exec_price,
            decision.quantity, decision.action, fee_rate,
        )
        if new_cash != state["cash"] or new_shares != state["shares"]:
            cash, shares = new_cash, new_shares
            fill_price = exec_price
            order_filled = True
        action_str = f"{decision.action} {decision.quantity}"

    pv = cash + shares * fill_price
    return {
        **state,
        "cash": cash,
        "shares": shares,
        "price": fill_price,
        "price_history": state["price_history"] + [fill_price],
        "portfolio_values": state["portfolio_values"] + [pv],
        "action": action_str,
        "trade_cost": trade_cost,
        "lob_bid": lob_bid,
        "lob_ask": lob_ask,
        "lob_spread": state.get("lob_spread", 0.0),
        "lob_mid": state.get("lob_mid", fill_price),
        "fill_price": fill_price,
        "order_filled": order_filled,
    }


def calculate_reward(
    state: AgentState,
    risk_penalty_lambda: float = 0.1,
    reward_fn=multicomponent_reward,
) -> AgentState:
    if len(state["portfolio_values"]) < 2:
        return {**state, "reward": 0.0}

    old_value = state["portfolio_values"][-2]
    new_value = state["portfolio_values"][-1]
    peak = max(state["peak_value"], new_value)
    reward = reward_fn(old_value, new_value, peak, state["trade_cost"], risk_penalty_lambda)
    total = state["total_reward"] + reward
    return {
        **state,
        "peak_value": peak,
        "reward": reward,
        "total_reward": total,
    }
