from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from trading_agent.core.state import AgentState
from trading_agent.core.schemas import TradeDecision
from trading_agent.core.nodes import execute_trade, calculate_reward, execute_lob_trade
from trading_agent.core.reward import multicomponent_reward, aggressive_reward

_REWARD_DESCRIPTIONS = {
    multicomponent_reward: (
        "Maximize: reward = profit - risk_penalty - transaction_cost.\n"
        "  profit = change in portfolio value\n"
        "  risk_penalty = lambda * max(0, peak - current_value)\n"
        "  transaction_cost = fee_rate * |trade_value|\n"
    ),
    aggressive_reward: (
        "Maximize: reward = profit + volatility_bonus - transaction_cost.\n"
        "  profit = change in portfolio value\n"
        "  volatility_bonus = lambda * |profit| (large swings rewarded)\n"
        "  NO drawdown penalty\n"
        "  transaction_cost = fee_rate * |trade_value|\n"
    ),
}

def build_graph(llm, fee_rate: float = 0.001, risk_lambda: float = 0.1, max_steps: int = 50, reward_fn=multicomponent_reward):
    graph = StateGraph(AgentState)
    reward_desc = _REWARD_DESCRIPTIONS.get(reward_fn, _REWARD_DESCRIPTIONS[multicomponent_reward])

    def node_decide_and_trade(state):
        system = (
            "You are a trading agent. " + reward_desc +
            "Respond with action (BUY/SELL/HOLD), quantity (int), reason (str)."
        )
        user = (
            f"cash={state['cash']:.2f}, shares={state['shares']}, price={state['price']:.2f}, "
            f"value={state['cash'] + state['shares'] * state['price']:.2f}, "
            f"peak={state['peak_value']:.2f}, step={state['step']}"
        )
        prices = state.get("price_history", [])
        if len(prices) >= 2:
            user += f", trend={((prices[-1] / prices[0]) - 1) * 100:.1f}% over {len(prices)} steps"
        if llm is None:
            decision = TradeDecision(action="HOLD", quantity=0, reason="no llm")
        else:
            try:
                structured_llm = llm.with_structured_output(TradeDecision)
                decision = structured_llm.invoke([
                    SystemMessage(content=system),
                    HumanMessage(content=user),
                ])
            except Exception:
                text_llm = llm.bind(stop=["\n"])
                resp = text_llm.invoke([
                    SystemMessage(content=system + " Reply with one line: ACTION quantity reason"),
                    HumanMessage(content=user),
                ])
                text = resp.content.strip() if hasattr(resp, "content") else str(resp)
                import re
                m = re.match(r"(BUY|SELL|HOLD)\s+(\d+)\s+(.*)", text)
                if m:
                    decision = TradeDecision(action=m.group(1), quantity=int(m.group(2)), reason=m.group(3))
                else:
                    decision = TradeDecision(action="HOLD", quantity=0, reason="parse fallback")
        result = execute_trade(AgentState(**state), decision, fee_rate)
        return {**result, "step": state["step"] + 1, "action": f"{decision.action} {decision.quantity}"}

    def node_calculate_reward(state):
        result = calculate_reward(AgentState(**state), risk_lambda, reward_fn)
        return {k: v for k, v in result.items()}

    def should_continue(state):
        return "end" if state["step"] >= max_steps or state.get("done", False) else "continue"

    graph.add_node("decide_and_trade", node_decide_and_trade)
    graph.add_node("calculate_reward", node_calculate_reward)

    graph.set_entry_point("decide_and_trade")
    graph.add_edge("decide_and_trade", "calculate_reward")
    graph.add_conditional_edges("calculate_reward", should_continue, {"continue": "decide_and_trade", "end": END})

    return graph.compile()


def build_lob_graph(llm, world_agent=None, fee_rate: float = 0.001, risk_lambda: float = 0.1, max_steps: int = 50, reward_fn=multicomponent_reward):
    graph = StateGraph(AgentState)
    reward_desc = _REWARD_DESCRIPTIONS.get(reward_fn, _REWARD_DESCRIPTIONS[multicomponent_reward])

    def node_decide_and_trade(state):
        has_lob = "lob_bid" in state and "lob_ask" in state
        system = (
            "You are a trading agent. " + reward_desc +
            "Respond with action (BUY/SELL/HOLD), quantity (int), reason (str)."
        )
        if has_lob:
            user = (
                f"cash={state['cash']:.2f}, shares={state['shares']}, "
                f"LOB bid={state.get('lob_bid',0):.2f}, LOB ask={state.get('lob_ask',0):.2f}, "
                f"spread={state.get('lob_spread',0):.4f}, mid={state.get('lob_mid', state['price']):.2f}, "
                f"filled={state.get('order_filled', False)}, fill_price={state.get('fill_price', 0):.2f}, "
                f"value={state['cash'] + state['shares'] * state['price']:.2f}, "
                f"peak={state['peak_value']:.2f}, step={state['step']}"
            )
        else:
            user = (
                f"cash={state['cash']:.2f}, shares={state['shares']}, price={state['price']:.2f}, "
                f"value={state['cash'] + state['shares'] * state['price']:.2f}, "
                f"peak={state['peak_value']:.2f}, step={state['step']}"
            )
        prices = state.get("price_history", [])
        if len(prices) >= 2:
            user += f", trend={((prices[-1] / prices[0]) - 1) * 100:.1f}% over {len(prices)} steps"
        if llm is None:
            decision = TradeDecision(action="HOLD", quantity=0, reason="no llm")
        else:
            try:
                structured_llm = llm.with_structured_output(TradeDecision)
                decision = structured_llm.invoke([
                    SystemMessage(content=system),
                    HumanMessage(content=user),
                ])
            except Exception:
                text_llm = llm.bind(stop=["\n"])
                resp = text_llm.invoke([
                    SystemMessage(content=system + " Reply with one line: ACTION quantity reason"),
                    HumanMessage(content=user),
                ])
                text = resp.content.strip() if hasattr(resp, "content") else str(resp)
                import re
                m = re.match(r"(BUY|SELL|HOLD)\s+(\d+)\s+(.*)", text)
                if m:
                    decision = TradeDecision(action=m.group(1), quantity=int(m.group(2)), reason=m.group(3))
                else:
                    decision = TradeDecision(action="HOLD", quantity=0, reason="parse fallback")
        result = execute_lob_trade(AgentState(**state), decision, fee_rate)
        return {**result, "step": state["step"] + 1, "action": f"{decision.action} {decision.quantity}"}

    def node_update_lob(state):
        if world_agent is not None:
            lob_state = world_agent.step()
            state.update(
                lob_bid=lob_state.best_bid,
                lob_ask=lob_state.best_ask,
                lob_spread=lob_state.spread,
                lob_mid=lob_state.mid_price,
            )
        return {**state}

    def node_calculate_reward(state):
        result = calculate_reward(AgentState(**state), risk_lambda, reward_fn)
        return {k: v for k, v in result.items()}

    def should_continue(state):
        return "end" if state["step"] >= max_steps or state.get("done", False) else "continue"

    graph.add_node("update_lob", node_update_lob)
    graph.add_node("decide_and_trade", node_decide_and_trade)
    graph.add_node("calculate_reward", node_calculate_reward)

    graph.set_entry_point("update_lob")
    graph.add_edge("update_lob", "decide_and_trade")
    graph.add_edge("decide_and_trade", "calculate_reward")
    graph.add_conditional_edges("calculate_reward", should_continue, {"continue": "update_lob", "end": END})

    return graph.compile()
