from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from trading_agent.core.state import AgentState
from trading_agent.core.schemas import TradeDecision
from trading_agent.core.nodes import execute_trade, calculate_reward, execute_lob_trade
from trading_agent.core.reward import multicomponent_reward, aggressive_reward
from trading_agent.core.cache import llm_cache


@runtime_checkable
class MarketAdapter(Protocol):
    @property
    def has_market_data_node(self) -> bool:
        ...

    def update_state(self, state: AgentState) -> AgentState:
        ...

    def format_prompt(self, state: AgentState) -> str:
        ...

    def execute_trade(self, state: AgentState, decision: TradeDecision, fee_rate: float) -> AgentState:
        ...


class PriceMarketAdapter:
    has_market_data_node: bool = False

    def update_state(self, state: AgentState) -> AgentState:
        return state

    def format_prompt(self, state: AgentState) -> str:
        return (
            f"cash={state['cash']:.2f}, shares={state['shares']}, price={state['price']:.2f}, "
            f"value={state['cash'] + state['shares'] * state['price']:.2f}, "
            f"peak={state['peak_value']:.2f}, step={state['step']}"
        )

    def execute_trade(self, state: AgentState, decision: TradeDecision, fee_rate: float) -> AgentState:
        return execute_trade(state, decision, fee_rate)


class LOBMarketAdapter:
    has_market_data_node: bool = True

    def __init__(self, world_agent=None):
        self.world_agent = world_agent

    def update_state(self, state: AgentState) -> AgentState:
        if self.world_agent is not None:
            lob_state = self.world_agent.step()
            state.update(
                lob_bid=lob_state.best_bid,
                lob_ask=lob_state.best_ask,
                lob_spread=lob_state.spread,
                lob_mid=lob_state.mid_price,
            )
        return {**state}

    def format_prompt(self, state: AgentState) -> str:
        if "lob_bid" in state and "lob_ask" in state:
            return (
                f"cash={state['cash']:.2f}, shares={state['shares']}, "
                f"LOB bid={state.get('lob_bid', 0):.2f}, LOB ask={state.get('lob_ask', 0):.2f}, "
                f"spread={state.get('lob_spread', 0):.4f}, mid={state.get('lob_mid', state['price']):.2f}, "
                f"filled={state.get('order_filled', False)}, fill_price={state.get('fill_price', 0):.2f}, "
                f"value={state['cash'] + state['shares'] * state['price']:.2f}, "
                f"peak={state['peak_value']:.2f}, step={state['step']}"
            )
        return (
            f"cash={state['cash']:.2f}, shares={state['shares']}, price={state['price']:.2f}, "
            f"value={state['cash'] + state['shares'] * state['price']:.2f}, "
            f"peak={state['peak_value']:.2f}, step={state['step']}"
        )

    def execute_trade(self, state: AgentState, decision: TradeDecision, fee_rate: float) -> AgentState:
        return execute_lob_trade(state, decision, fee_rate)


class BarMarketAdapter:
    has_market_data_node: bool = True

    def __init__(self, bar_world_agent=None):
        self.bar_world_agent = bar_world_agent

    def update_state(self, state: AgentState) -> AgentState:
        if self.bar_world_agent is not None:
            bar = self.bar_world_agent.step()
            state.update(
                bar_open=bar.open,
                bar_high=bar.high,
                bar_low=bar.low,
                bar_close=bar.close,
                bar_volume=bar.volume,
                bar_vwap=bar.vwap,
                price=bar.close,
            )
        return {**state}

    def format_prompt(self, state: AgentState) -> str:
        if "bar_open" in state and "bar_close" in state:
            return (
                f"cash={state['cash']:.2f}, shares={state['shares']}, "
                f"O={state.get('bar_open', 0):.2f} H={state.get('bar_high', 0):.2f} "
                f"L={state.get('bar_low', 0):.2f} C={state.get('bar_close', state['price']):.2f} "
                f"V={state.get('bar_volume', 0):.0f}, "
                f"value={state['cash'] + state['shares'] * state['price']:.2f}, "
                f"peak={state['peak_value']:.2f}, step={state['step']}"
            )
        return (
            f"cash={state['cash']:.2f}, shares={state['shares']}, price={state['price']:.2f}, "
            f"value={state['cash'] + state['shares'] * state['price']:.2f}, "
            f"peak={state['peak_value']:.2f}, step={state['step']}"
        )

    def execute_trade(self, state: AgentState, decision: TradeDecision, fee_rate: float) -> AgentState:
        return execute_trade(state, decision, fee_rate)


def _decide_with_cache(state: AgentState, llm, system: str, user: str) -> TradeDecision:
    if llm is None:
        return TradeDecision(action="HOLD", quantity=0, reason="no llm")
    cached = llm_cache.get(state)
    if cached is not None:
        return cached
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
        m = re.match(r"(BUY|SELL|HOLD)\s+(\d+)\s+(.*)", text)
        if m:
            decision = TradeDecision(action=m.group(1), quantity=int(m.group(2)), reason=m.group(3))
        else:
            decision = TradeDecision(action="HOLD", quantity=0, reason="parse fallback")
    llm_cache.put(state, decision)
    return decision


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


def build_agent_graph(
    llm,
    market: MarketAdapter,
    fee_rate: float = 0.001,
    risk_lambda: float = 0.1,
    max_steps: int = 50,
    reward_fn=multicomponent_reward,
):
    graph = StateGraph(AgentState)
    reward_desc = _REWARD_DESCRIPTIONS.get(reward_fn, _REWARD_DESCRIPTIONS[multicomponent_reward])

    def node_decide_and_trade(state):
        system = (
            "You are a trading agent. " + reward_desc +
            "Respond with action (BUY/SELL/HOLD), quantity (int), reason (str)."
        )
        user = market.format_prompt(state)
        prices = state.get("price_history", [])
        if len(prices) >= 2:
            user += f", trend={((prices[-1] / prices[0]) - 1) * 100:.1f}% over {len(prices)} steps"
        decision = _decide_with_cache(state, llm, system, user)
        result = market.execute_trade(AgentState(**state), decision, fee_rate)
        return {**result, "step": state["step"] + 1, "action": f"{decision.action} {decision.quantity}"}

    def node_update_market(state):
        return market.update_state(state)

    def node_calculate_reward(state):
        result = calculate_reward(AgentState(**state), risk_lambda, reward_fn)
        return {k: v for k, v in result.items()}

    def should_continue(state):
        return "end" if state["step"] >= max_steps or state.get("done", False) else "continue"

    if market.has_market_data_node:
        graph.add_node("update_market", node_update_market)
        graph.add_node("decide_and_trade", node_decide_and_trade)
        graph.add_node("calculate_reward", node_calculate_reward)
        graph.set_entry_point("update_market")
        graph.add_edge("update_market", "decide_and_trade")
        graph.add_edge("decide_and_trade", "calculate_reward")
        graph.add_conditional_edges("calculate_reward", should_continue, {"continue": "update_market", "end": END})
    else:
        graph.add_node("decide_and_trade", node_decide_and_trade)
        graph.add_node("calculate_reward", node_calculate_reward)
        graph.set_entry_point("decide_and_trade")
        graph.add_edge("decide_and_trade", "calculate_reward")
        graph.add_conditional_edges("calculate_reward", should_continue, {"continue": "decide_and_trade", "end": END})

    return graph.compile()


def build_graph(llm, fee_rate: float = 0.001, risk_lambda: float = 0.1, max_steps: int = 50, reward_fn=multicomponent_reward):
    return build_agent_graph(
        llm,
        market=PriceMarketAdapter(),
        fee_rate=fee_rate,
        risk_lambda=risk_lambda,
        max_steps=max_steps,
        reward_fn=reward_fn,
    )


def build_lob_graph(llm, world_agent=None, fee_rate: float = 0.001, risk_lambda: float = 0.1, max_steps: int = 50, reward_fn=multicomponent_reward):
    return build_agent_graph(
        llm,
        market=LOBMarketAdapter(world_agent),
        fee_rate=fee_rate,
        risk_lambda=risk_lambda,
        max_steps=max_steps,
        reward_fn=reward_fn,
    )


def build_bar_graph(llm, bar_world_agent=None, fee_rate: float = 0.001, risk_lambda: float = 0.1, max_steps: int = 50, reward_fn=multicomponent_reward):
    return build_agent_graph(
        llm,
        market=BarMarketAdapter(bar_world_agent),
        fee_rate=fee_rate,
        risk_lambda=risk_lambda,
        max_steps=max_steps,
        reward_fn=reward_fn,
    )
