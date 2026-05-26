from trading_agent.core.state import AgentState
from trading_agent.core.schemas import TradeDecision, LimitOrder, MarketOrder
from trading_agent.core.reward import multicomponent_reward


def fetch_price(state: AgentState, price: float) -> AgentState:
    pv = state["cash"] + state["shares"] * price
    return AgentState(
        cash=state["cash"], shares=state["shares"], price=price,
        price_history=state["price_history"] + [price],
        portfolio_values=state["portfolio_values"] + [pv],
        peak_value=state["peak_value"], action=state["action"],
        trade_cost=state["trade_cost"], reward=state["reward"],
        total_reward=state["total_reward"], step=state["step"] + 1,
        done=state["done"],
    )


def execute_trade(state: AgentState, decision: TradeDecision, fee_rate: float = 0.001) -> AgentState:
    cash, shares, price = state["cash"], state["shares"], state["price"]
    trade_cost = 0.0
    if decision.action == "BUY":
        cost = decision.quantity * price
        if cost <= cash:
            trade_cost = fee_rate * cost
            cash -= cost + trade_cost
            shares += decision.quantity
    elif decision.action == "SELL":
        qty = min(decision.quantity, shares)
        proceeds = qty * price
        trade_cost = fee_rate * proceeds
        cash += proceeds - trade_cost
        shares -= qty
    action_str = f"{decision.action} {decision.quantity}"
    pv = cash + shares * price
    return AgentState(
        cash=cash, shares=shares, price=price,
        price_history=state["price_history"] + [price],
        portfolio_values=state["portfolio_values"] + [pv],
        peak_value=state["peak_value"], action=action_str, trade_cost=trade_cost,
        reward=state["reward"], total_reward=state["total_reward"],
        step=state["step"], done=state["done"],
    )


def execute_lob_trade(
    state: AgentState,
    decision: TradeDecision | LimitOrder | MarketOrder,
    fee_rate: float = 0.001,
) -> AgentState:
    cash = state["cash"]
    shares = state["shares"]
    lob_bid = state.get("lob_bid", state["price"])
    lob_ask = state.get("lob_ask", state["price"])
    price = state["price"]
    trade_cost = 0.0
    order_filled = False
    fill_price = price

    if isinstance(decision, (LimitOrder, MarketOrder)):
        if isinstance(decision, LimitOrder):
            target_price = decision.price
        else:
            target_price = lob_ask if decision.side == "BUY" else lob_bid

        if decision.side == "BUY":
            px = min(target_price, lob_ask)
            cost = decision.quantity * px
            if cost <= cash:
                trade_cost = fee_rate * cost
                cash -= cost + trade_cost
                shares += decision.quantity
                order_filled = True
                fill_price = px
        else:
            qty = min(decision.quantity, shares)
            px = max(target_price, lob_bid)
            proceeds = qty * px
            trade_cost = fee_rate * proceeds
            cash += proceeds - trade_cost
            shares -= qty
            order_filled = True
            fill_price = px
        action_str = f"{type(decision).__name__} {decision.side} {decision.quantity}"
    else:
        if decision.action == "BUY":
            px = min(state["price"], lob_ask)
            cost = decision.quantity * px
            if cost <= cash:
                trade_cost = fee_rate * cost
                cash -= cost + trade_cost
                shares += decision.quantity
                order_filled = True
                fill_price = px
        elif decision.action == "SELL":
            qty = min(decision.quantity, shares)
            px = max(state["price"], lob_bid)
            proceeds = qty * px
            trade_cost = fee_rate * proceeds
            cash += proceeds - trade_cost
            shares -= qty
            order_filled = True
            fill_price = px
        else:
            fill_price = state["price"]
        action_str = f"{decision.action} {decision.quantity}"

    pv = cash + shares * fill_price
    new_state = AgentState(
        cash=cash, shares=shares, price=fill_price,
        price_history=state["price_history"] + [fill_price],
        portfolio_values=state["portfolio_values"] + [pv],
        peak_value=state["peak_value"], action=action_str, trade_cost=trade_cost,
        reward=state["reward"], total_reward=state["total_reward"],
        step=state["step"], done=state["done"],
        lob_bid=lob_bid, lob_ask=lob_ask, lob_spread=state.get("lob_spread", 0.0),
        lob_mid=state.get("lob_mid", fill_price),
        fill_price=fill_price, order_filled=order_filled,
    )
    return new_state


def calculate_reward(
    state: AgentState,
    risk_penalty_lambda: float = 0.1,
    reward_fn=multicomponent_reward,
) -> AgentState:
    if len(state["portfolio_values"]) < 2:
        return AgentState(
            cash=state["cash"], shares=state["shares"], price=state["price"],
            price_history=state["price_history"], portfolio_values=state["portfolio_values"],
            peak_value=state["peak_value"], action=state["action"],
            trade_cost=state["trade_cost"], reward=0.0, total_reward=state["total_reward"],
            step=state["step"], done=state["done"],
        )
    old_value = state["portfolio_values"][-2]
    new_value = state["portfolio_values"][-1]
    peak = max(state["peak_value"], new_value)
    reward = reward_fn(old_value, new_value, peak, state["trade_cost"], risk_penalty_lambda)
    total = state["total_reward"] + reward
    return AgentState(
        cash=state["cash"], shares=state["shares"], price=state["price"],
        price_history=state["price_history"], portfolio_values=state["portfolio_values"],
        peak_value=peak, action=state["action"], trade_cost=state["trade_cost"],
        reward=reward, total_reward=total, step=state["step"], done=state["done"],
    )
