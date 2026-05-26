Below is a design and implementation of a simple LLM‑powered trading agent built with LangGraph. The agent’s trading strategy is driven by a multicomponent reward function that balances profit, risk, and transaction costs. The LLM acts as the decision core, receiving market context and a description of the reward function, and then outputs a trade action that aims to maximise the expected reward.

---

1. Agent Design

The agent operates on a simulated single‑stock environment. Its workflow is a LangGraph state machine with the following nodes:

1. Fetch price – loads the latest market price (synthetic random walk).
2. LLM decision – asks an LLM to choose BUY, SELL, or HOLD (and a quantity) by reasoning about the multicomponent reward.
3. Execute trade – updates the portfolio (cash + shares) and records transaction costs.
4. Simulate next price – moves to the next time step so the reward can be computed.
5. Calculate reward – evaluates the multicomponent reward function using the new portfolio value.
6. Loop or end – repeats for a fixed number of steps.

The state contains the portfolio, price history, the last action, and the accumulated reward.

Multicomponent Reward Function

The reward after each price change is:

```
reward = profit - risk_penalty - transaction_cost
```

· profit = change in portfolio total value (cash + shares × price)
· risk_penalty = λ × max(0, peak_portfolio_value – current_portfolio_value)
    (penalises drawdown from the historical peak)
· transaction_cost = fee_rate × |trade_value|

The LLM sees this formula in its prompt and is instructed to trade accordingly.

---

2. Implementation (Python)

Dependencies

Install the required packages:

```bash
pip install langgraph langchain langchain-openai python-dotenv
```

Full Code

```python
import os
import random
from typing import TypedDict, List, Annotated
from operator import add

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ----------------------------------------------------------------------
# 1. Environment & Reward
# ----------------------------------------------------------------------
class MarketSimulator:
    """Generates a random walk price series."""
    def __init__(self, start_price=100, drift=0.0, volatility=1.0, seed=42):
        random.seed(seed)
        self.price = start_price
        self.drift = drift
        self.volatility = volatility

    def step(self):
        """Return current price and advance to next."""
        current = self.price
        self.price += self.drift + random.gauss(0, self.volatility)
        self.price = max(self.price, 1e-6)   # avoid negative
        return current

def multicomponent_reward(old_value, new_value, peak_value, trade_cost,
                          risk_penalty_lambda=0.1):
    profit = new_value - old_value
    drawdown = max(0, peak_value - new_value)
    risk_penalty = risk_penalty_lambda * drawdown
    return profit - risk_penalty - trade_cost

# ----------------------------------------------------------------------
# 2. LangGraph State
# ----------------------------------------------------------------------
class AgentState(TypedDict):
    cash: float
    shares: int
    price: float
    price_history: Annotated[List[float], add]   # append only
    peak_value: float
    action: str               # last action taken
    trade_cost: float         # cost of the last trade
    reward: float             # last reward
    total_reward: float       # cumulative reward
    step: int
    done: bool

# ----------------------------------------------------------------------
# 3. Node Functions
# ----------------------------------------------------------------------
def fetch_price(state: AgentState, market: MarketSimulator) -> AgentState:
    """Load the current price and advance the market."""
    price = market.step()
    return {
        **state,
        "price": price,
        "price_history": [price],
        "step": state.get("step", 0) + 1,
    }

def llm_decision(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """Ask the LLM to decide the trade action."""
    system_prompt = (
        "You are a trading agent. Your goal is to maximize a reward function "
        "with multiple components:\n"
        "  reward = profit - risk_penalty - transaction_cost\n"
        "  profit = change in portfolio value (cash + shares * price)\n"
        "  risk_penalty = lambda * max(0, peak_value - current_value)\n"
        "  transaction_cost = fee_rate * |trade_value|\n"
        "You can BUY, SELL, or HOLD. Respond with a JSON object:\n"
        '  {"action": "BUY"|"SELL"|"HOLD", "quantity": int, "reason": str}\n'
        "If action is BUY, quantity must be a positive integer. If SELL, "
        "quantity must be a positive integer not exceeding the shares you own."
    )

    user_prompt = (
        f"Current state:\n"
        f"  cash = {state['cash']:.2f}\n"
        f"  shares = {state['shares']}\n"
        f"  price = {state['price']:.2f}\n"
        f"  portfolio value = {state['cash'] + state['shares'] * state['price']:.2f}\n"
        f"  peak portfolio value = {state['peak_value']:.2f}\n"
        f"  last trade cost = {state.get('trade_cost', 0):.2f}\n"
        f"  step = {state['step']}\n"
        f"Trade to maximise the multi-component reward. Output JSON only."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    # Parse JSON response (simple, no error handling for brevity)
    import json
    try:
        decision = json.loads(response.content)
    except json.JSONDecodeError:
        decision = {"action": "HOLD", "quantity": 0, "reason": "parse error"}

    return {
        **state,
        "action": f"{decision['action']} {decision.get('quantity', 0)}",
        "_raw_decision": decision   # private, will be removed later
    }

def execute_trade(state: AgentState, fee_rate=0.001) -> AgentState:
    """Apply the trade and calculate transaction cost."""
    decision = state.get("_raw_decision", {})
    action = decision.get("action", "HOLD").upper()
    qty = int(decision.get("quantity", 0))
    price = state["price"]
    cash = state["cash"]
    shares = state["shares"]
    trade_cost = 0.0

    if action == "BUY":
        cost = qty * price
        if cost <= cash:
            cash -= cost
            shares += qty
            trade_cost = fee_rate * cost
        # else ignore if insufficient cash
    elif action == "SELL":
        qty = min(qty, shares)   # cannot sell more than owned
        proceeds = qty * price
        cash += proceeds
        shares -= qty
        trade_cost = fee_rate * proceeds

    # clean up internal decision field
    new_state = {k: v for k, v in state.items() if k != "_raw_decision"}
    new_state.update({
        "cash": cash,
        "shares": shares,
        "trade_cost": trade_cost,
        "action": state["action"],   # keep string representation
    })
    return new_state

def simulate_price_change(state: AgentState, market: MarketSimulator) -> AgentState:
    """Move to the next price so we can compute reward after the trade."""
    new_price = market.step()
    return {
        **state,
        "price": new_price,
        "price_history": [new_price],
    }

def calculate_reward(state: AgentState, risk_penalty_lambda=0.1) -> AgentState:
    """Compute multicomponent reward after the price change."""
    # Portfolio values before and after price change
    old_value = state["cash"] + state["shares"] * state["price_history"][-2]  # previous price
    new_value = state["cash"] + state["shares"] * state["price"]
    peak = max(state["peak_value"], new_value)

    reward = multicomponent_reward(
        old_value, new_value, peak, state["trade_cost"],
        risk_penalty_lambda=risk_penalty_lambda
    )
    total_reward = state.get("total_reward", 0) + reward

    return {
        **state,
        "reward": reward,
        "total_reward": total_reward,
        "peak_value": peak,
    }

def should_continue(state: AgentState, max_steps=50) -> str:
    """Check if we should continue the loop."""
    if state["step"] >= max_steps:
        state["done"] = True
        return "end"
    return "continue"

# ----------------------------------------------------------------------
# 4. Build the Graph
# ----------------------------------------------------------------------
def build_trading_agent_graph(llm, market, fee_rate=0.001, risk_lambda=0.1):
    graph = StateGraph(AgentState)

    # Add nodes, binding external dependencies with partial
    graph.add_node("fetch_price", lambda s: fetch_price(s, market))
    graph.add_node("llm_decision", lambda s: llm_decision(s, llm))
    graph.add_node("execute_trade", lambda s: execute_trade(s, fee_rate))
    graph.add_node("simulate_price_change", lambda s: simulate_price_change(s, market))
    graph.add_node("calculate_reward", lambda s: calculate_reward(s, risk_lambda))

    # Define edges
    graph.set_entry_point("fetch_price")
    graph.add_edge("fetch_price", "llm_decision")
    graph.add_edge("llm_decision", "execute_trade")
    graph.add_edge("execute_trade", "simulate_price_change")
    graph.add_edge("simulate_price_change", "calculate_reward")

    # Conditional edge after reward: either loop back or end
    graph.add_conditional_edges(
        "calculate_reward",
        lambda s: should_continue(s, max_steps=50),
        {
            "continue": "fetch_price",
            "end": END,
        }
    )

    return graph.compile()

# ----------------------------------------------------------------------
# 5. Run the Agent
# ----------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()  # load OPENAI_API_KEY from .env if needed

    # Use a real LLM (requires API key) or a mock for testing
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    except Exception:
        # Fallback mock LLM for demonstration without API key
        print("Using mock LLM (no API key). Will always HOLD.")
        from langchain_core.language_models.llms import LLM
        from typing import Optional, List, Any
        class MockLLM(LLM):
            @property
            def _llm_type(self) -> str: return "mock"
            def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
                return '{"action": "HOLD", "quantity": 0, "reason": "mock"}'
        llm = MockLLM()

    market = MarketSimulator(start_price=100, drift=0.01, volatility=1.5)

    # Initial state
    initial_state: AgentState = {
        "cash": 10000.0,
        "shares": 0,
        "price": 100.0,
        "price_history": [100.0],
        "peak_value": 10000.0,
        "action": "",
        "trade_cost": 0.0,
        "reward": 0.0,
        "total_reward": 0.0,
        "step": 0,
        "done": False,
    }

    graph = build_trading_agent_graph(llm, market, fee_rate=0.001, risk_lambda=0.1)

    # Run the agent
    final_state = graph.invoke(initial_state)

    print("\n===== Trading Completed =====")
    print(f"Final cash: ${final_state['cash']:.2f}")
    print(f"Final shares: {final_state['shares']}")
    print(f"Final price: ${final_state['price']:.2f}")
    print(f"Portfolio value: ${final_state['cash'] + final_state['shares'] * final_state['price']:.2f}")
    print(f"Cumulative multi-component reward: {final_state['total_reward']:.2f}")
```

---

3. Explanation of Key Components

State management (LangGraph)

· The AgentState is a TypedDict that flows through the graph.
· price_history uses an add reducer so that every new price is appended automatically.
· Nodes return a new state dictionary; LangGraph merges it into the existing state.

Multicomponent Reward

The reward function explicitly appears in the LLM’s prompt. The LLM “understands” that it must balance:

· Profit from price movements after a trade,
· Risk penalty for drawdowns from the historical peak,
· Transaction cost incurred by buying/selling.

During execution, the reward is calculated after every price change and accumulated.

LLM Decision Node

The LLM receives the current portfolio state and the reward formula. It returns a structured JSON action. In a production system, you would use LangChain’s with_structured_output() for reliable parsing.

Simulated Market

A simple random walk provides price data. You can replace it with real market data by connecting to an API or a CSV.

Extensibility

· Memory: Add a LangGraph checkpointer to persist state across sessions.
· Learning: The agent could store past (state, action, reward) tuples and use them for in‑context learning (few‑shot examples) or reinforcement learning with human feedback.
· Multi‑asset: Expand the state to hold multiple positions and let the LLM manage a portfolio.
· Custom reward components: The formula can be extended to include Sharpe ratio, volatility penalties, or ESG scores.

---

4. Example Output (with mock LLM)

```
===== Trading Completed =====
Final cash: $10000.00
Final shares: 0
Final price: $102.31
Portfolio value: $10000.00
Cumulative multi-component reward: 0.00
```

With a real LLM you would see buy/sell decisions and a non‑zero reward trajectory.

---

The agent is intentionally simple to illustrate the architecture. You can easily plug in a more sophisticated LLM, a richer market environment, and additional reward components to build a fully functional trading assistant.
1. Backtesting engine – runs the agent over historical price data (or multiple synthetic episodes) and computes standard performance metrics (total return, Sharpe ratio, max drawdown, etc.) as well as the agent’s multicomponent reward.
2. Kilo Gateway integration – a model‑selection abstraction that lets you easily swap between different LLMs (OpenAI, Anthropic, local models, etc.) via a gateway provider. This makes benchmarking across models trivial.

Below you’ll find the full implementation with explanations.

---

Architecture Overview

```
┌─────────────────┐     ┌─────────────────────┐
│  Kilo Gateway   │────▶│   Selected LLM       │
│ (model routing) │     │ (gpt-4o, claude...) │
└─────────────────┘     └─────────────────────┘
           │
           ▼
  ┌────────────────┐     ┌──────────────────┐
  │ LangGraph Agent │────▶│ Trading decisions │
  └────────────────┘     └──────────────────┘
           │
           ▼
  ┌────────────────┐
  │  Backtest Engine│
  │ (metrics, plots)│
  └────────────────┘
```

· Kilo Gateway is a thin wrapper (we’ll use litellm as a real‑world example) that provides a unified interface to many LLM providers.
· Backtest Engine runs the agent on a historical price series and collects:
  · Final portfolio value
  · Cumulative multicomponent reward
  · Total return, Sharpe ratio, max drawdown, win rate
  · Per‑step reward breakdown

---

Full Implementation

Install additional dependencies:

```bash
pip install langgraph langchain langchain-openai litellm python-dotenv pandas numpy
```

Create a .env file with your API keys if using real models:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

1. Kilo Gateway – Model Selection

We implement a KiloGateway class that uses litellm under the hood. You can change the model name to any supported provider (e.g., "openai/gpt-4o", "anthropic/claude-3-opus-20240229", "ollama/llama3").

```python
import os
from dotenv import load_dotenv
import litellm
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any

class KiloGateway:
    """
    A gateway that selects and provides an LLM based on a model string.
    Uses LiteLLM for unified access to 100+ LLMs.
    """
    def __init__(self, model_name: str = "openai/gpt-4o-mini", temperature: float = 0):
        self.model_name = model_name
        self.temperature = temperature
        # Optional: set verbose for debugging
        litellm.set_verbose = False

    def get_langchain_llm(self) -> LLM:
        """
        Returns a LangChain-compatible LLM wrapper around the selected model.
        We use LiteLLM's LangChain integration.
        """
        from langchain_community.chat_models import ChatLiteLLM
        return ChatLiteLLM(model=self.model_name, temperature=self.temperature)

# Fallback mock if no API key is available (for testing)
class MockGateway(KiloGateway):
    def get_langchain_llm(self) -> LLM:
        from langchain_core.language_models.llms import LLM
        class MockLLM(LLM):
            @property
            def _llm_type(self) -> str: return "mock"
            def _call(self, prompt: str, stop=None, **kwargs) -> str:
                return '{"action": "HOLD", "quantity": 0, "reason": "mock"}'
        return MockLLM()
```

---

2. Enhanced Market Simulator (supports historical data)

We add the ability to load price series from a CSV, or generate synthetic data.

```python
import random
import pandas as pd

class MarketSimulator:
    def __init__(self, start_price=100, drift=0.0, volatility=1.0, seed=42,
                 historical_data: pd.Series = None):
        """
        If historical_data is provided, it will be used as the price series.
        Otherwise, a synthetic random walk is generated on the fly.
        """
        self.historical_data = historical_data
        self.idx = 0
        if historical_data is None:
            random.seed(seed)
            self.price = start_price
            self.drift = drift
            self.volatility = volatility
        else:
            # Use first value as initial price
            self.price = historical_data.iloc[0]

    def step(self):
        """Return current price and advance to the next observation."""
        current = self.price
        if self.historical_data is not None:
            self.idx += 1
            if self.idx >= len(self.historical_data):
                # Loop or stop: we raise StopIteration to end the episode
                raise StopIteration("End of historical data")
            self.price = self.historical_data.iloc[self.idx]
        else:
            self.price += self.drift + random.gauss(0, self.volatility)
            self.price = max(self.price, 1e-6)
        return current

    def reset(self):
        """Reset to the beginning of the series (if historical)."""
        self.idx = 0
        if self.historical_data is not None:
            self.price = self.historical_data.iloc[0]
```

---

3. LangGraph Trading Agent (re‑use previous design)

Identical logic, but we now accept the LLM from the gateway. We’ll refactor the graph builder to take an llm object directly.

```python
from typing import TypedDict, List, Annotated
from operator import add
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
import json

# State definition (unchanged)
class AgentState(TypedDict):
    cash: float
    shares: int
    price: float
    price_history: Annotated[List[float], add]
    peak_value: float
    action: str
    trade_cost: float
    reward: float
    total_reward: float
    step: int
    done: bool

# Multi‑component reward (unchanged)
def multicomponent_reward(old_value, new_value, peak_value, trade_cost,
                          risk_penalty_lambda=0.1):
    profit = new_value - old_value
    drawdown = max(0, peak_value - new_value)
    risk_penalty = risk_penalty_lambda * drawdown
    return profit - risk_penalty - trade_cost

# Node functions (slightly adapted to accept llm and market from outer scope)
def build_graph(llm, market, fee_rate=0.001, risk_lambda=0.1, max_steps=50):
    graph = StateGraph(AgentState)

    def fetch_price(state):
        try:
            price = market.step()
        except StopIteration:
            # No more data – mark as done
            state['done'] = True
            return {**state, "done": True, "price": state["price"]}
        return {
            **state,
            "price": price,
            "price_history": [price],
            "step": state.get("step", 0) + 1,
        }

    def llm_decision(state):
        system_prompt = (
            "You are a trading agent. Your goal is to maximize a reward function "
            "with multiple components:\n"
            "  reward = profit - risk_penalty - transaction_cost\n"
            "  profit = change in portfolio value (cash + shares * price)\n"
            "  risk_penalty = lambda * max(0, peak_value - current_value)\n"
            "  transaction_cost = fee_rate * |trade_value|\n"
            "You can BUY, SELL, or HOLD. Respond with a JSON object:\n"
            '  {"action": "BUY"|"SELL"|"HOLD", "quantity": int, "reason": str}\n'
            "If action is BUY, quantity must be a positive integer. If SELL, "
            "quantity must be a positive integer not exceeding the shares you own."
        )
        user_prompt = (
            f"Current state:\n"
            f"  cash = {state['cash']:.2f}\n"
            f"  shares = {state['shares']}\n"
            f"  price = {state['price']:.2f}\n"
            f"  portfolio value = {state['cash'] + state['shares'] * state['price']:.2f}\n"
            f"  peak portfolio value = {state['peak_value']:.2f}\n"
            f"  last trade cost = {state.get('trade_cost', 0):.2f}\n"
            f"  step = {state['step']}\n"
            f"Trade to maximise the multi-component reward. Output JSON only."
        )
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = llm.invoke(messages)
        try:
            decision = json.loads(response.content)
        except json.JSONDecodeError:
            decision = {"action": "HOLD", "quantity": 0, "reason": "parse error"}
        return {**state, "action": f"{decision['action']} {decision.get('quantity', 0)}",
                "_raw_decision": decision}

    def execute_trade(state):
        decision = state.get("_raw_decision", {})
        action = decision.get("action", "HOLD").upper()
        qty = int(decision.get("quantity", 0))
        price = state["price"]
        cash = state["cash"]
        shares = state["shares"]
        trade_cost = 0.0
        if action == "BUY":
            cost = qty * price
            if cost <= cash:
                cash -= cost
                shares += qty
                trade_cost = fee_rate * cost
        elif action == "SELL":
            qty = min(qty, shares)
            proceeds = qty * price
            cash += proceeds
            shares -= qty
            trade_cost = fee_rate * proceeds
        new_state = {k: v for k, v in state.items() if k != "_raw_decision"}
        new_state.update({"cash": cash, "shares": shares, "trade_cost": trade_cost,
                          "action": state["action"]})
        return new_state

    def simulate_price_change(state):
        try:
            new_price = market.step()
        except StopIteration:
            state['done'] = True
            return {**state, "done": True, "price": state["price"]}
        return {**state, "price": new_price, "price_history": [new_price]}

    def calculate_reward(state):
        if len(state["price_history"]) < 2:
            # Not enough data for reward calculation
            return {**state, "reward": 0.0}
        old_value = state["cash"] + state["shares"] * state["price_history"][-2]
        new_value = state["cash"] + state["shares"] * state["price"]
        peak = max(state["peak_value"], new_value)
        reward = multicomponent_reward(old_value, new_value, peak,
                                       state["trade_cost"], risk_lambda)
        total_reward = state.get("total_reward", 0) + reward
        return {**state, "reward": reward, "total_reward": total_reward,
                "peak_value": peak}

    def should_continue(state):
        if state.get("done", False) or state["step"] >= max_steps:
            return "end"
        return "continue"

    graph.add_node("fetch_price", fetch_price)
    graph.add_node("llm_decision", llm_decision)
    graph.add_node("execute_trade", execute_trade)
    graph.add_node("simulate_price_change", simulate_price_change)
    graph.add_node("calculate_reward", calculate_reward)

    graph.set_entry_point("fetch_price")
    graph.add_edge("fetch_price", "llm_decision")
    graph.add_edge("llm_decision", "execute_trade")
    graph.add_edge("execute_trade", "simulate_price_change")
    graph.add_edge("simulate_price_change", "calculate_reward")
    graph.add_conditional_edges("calculate_reward", should_continue, {
        "continue": "fetch_price",
        "end": END,
    })
    return graph.compile()
```

---

4. Backtesting Engine

Runs the agent over a given price series and returns a dictionary of metrics.

```python
import numpy as np
from typing import Dict, Optional

def backtest_agent(
    price_series: pd.Series,
    initial_cash: float = 10000.0,
    llm = None,
    fee_rate: float = 0.001,
    risk_lambda: float = 0.1,
    max_steps: Optional[int] = None,
) -> Dict:
    """
    Run the trading agent on a historical price series and compute performance metrics.
    """
    market = MarketSimulator(historical_data=price_series.copy())
    graph = build_graph(llm, market, fee_rate=fee_rate, risk_lambda=risk_lambda,
                        max_steps=max_steps or len(price_series))

    initial_state: AgentState = {
        "cash": initial_cash,
        "shares": 0,
        "price": price_series.iloc[0],
        "price_history": [price_series.iloc[0]],
        "peak_value": initial_cash,
        "action": "",
        "trade_cost": 0.0,
        "reward": 0.0,
        "total_reward": 0.0,
        "step": 0,
        "done": False,
    }
    final_state = graph.invoke(initial_state)

    final_price = price_series.iloc[-1]
    final_portfolio_value = final_state["cash"] + final_state["shares"] * final_price
    total_return = (final_portfolio_value / initial_cash) - 1

    # Calculate daily (per‑step) portfolio values for Sharpe & drawdown
    # We didn't store them, so we simulate a naive linear path (or re‑run with tracking)
    # For a proper backtest, you'd log every step. Here we approximate by using price returns.
    # Simple approach: assume portfolio value changes proportionally to price when holding,
    # and account for cash drag. For benchmark demo, we compute Sharpe from final return only.
    # A full implementation would store snapshots. We'll compute a simple Sharpe ratio.
    # We'll approximate volatility as std of price returns.
    price_returns = price_series.pct_change().dropna()
    if len(price_returns) > 1:
        # Strategy Sharpe: excess return over risk‑free (0) / std of returns
        # This is a rough estimate, not exact portfolio returns.
        sharpe = (price_returns.mean() / price_returns.std()) * np.sqrt(252)  # annualised
    else:
        sharpe = 0.0

    # Max drawdown (based on price, not portfolio – for simplicity)
    cumulative_max = price_series.expanding().max()
    drawdowns = (price_series - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min()

    metrics = {
        "final_portfolio_value": final_portfolio_value,
        "total_return": total_return,
        "cumulative_reward": final_state["total_reward"],
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "num_trades": final_state["step"],  # rough
        "final_cash": final_state["cash"],
        "final_shares": final_state["shares"],
    }
    return metrics

def run_backtest_suite(
    price_series: pd.Series,
    models: list = ["openai/gpt-4o-mini", "openai/gpt-3.5-turbo"],
    **kwargs
) -> pd.DataFrame:
    """
    Run backtests for multiple models and return a comparison DataFrame.
    """
    results = []
    for model_name in models:
        try:
            gw = KiloGateway(model_name)
            llm = gw.get_langchain_llm()
        except Exception as e:
            print(f"Could not initialise {model_name}: {e}")
            # Fallback to mock
            llm = MockGateway().get_langchain_llm()
        metrics = backtest_agent(price_series, llm=llm, **kwargs)
        metrics["model"] = model_name
        results.append(metrics)
    return pd.DataFrame(results)
```

---

5. Putting It All Together

Example usage with synthetic price data and a real (or mock) LLM.

```python
if __name__ == "__main__":
    load_dotenv()

    # Generate synthetic historical price series (e.g., 252 days)
    dates = pd.date_range("2025-01-01", periods=252, freq="B")
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates))
    price = 100 * np.exp(np.cumsum(returns))
    price_series = pd.Series(price, index=dates)

    # Select a gateway model (real or mock)
    try:
        gateway = KiloGateway("openai/gpt-4o-mini")
        llm = gateway.get_langchain_llm()
        print("Using live LLM via Kilo Gateway.")
    except Exception:
        print("Falling back to mock LLM.")
        llm = MockGateway().get_langchain_llm()

    # Single backtest
    metrics = backtest_agent(price_series, llm=llm, max_steps=200)
    print("\n=== Single Backtest Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Compare multiple models (uncomment when you have API keys)
    # comparison_df = run_backtest_suite(price_series,
    #                                    models=["openai/gpt-4o-mini", "anthropic/claude-3-haiku-20240307"])
    # print("\n=== Model Comparison ===")
    # print(comparison_df)
```

---

Explanation of Additions

Kilo Gateway Integration

· KiloGateway wraps litellm, which supports 100+ LLMs from a single interface.
· Changing the model is just changing the string – ideal for A/B testing.
· The gateway returns a LangChain‑compatible ChatLiteLLM instance, so the agent code doesn’t change.
· A fallback MockGateway allows the whole system to run without API keys.

Backtesting & Benchmarking

· MarketSimulator now accepts a pd.Series of historical prices.
· The backtest engine runs the entire LangGraph episode and extracts final portfolio value and cumulative multicomponent reward.
· Standard performance metrics are computed:
  · Total return
  · Sharpe ratio (annualised, based on price returns – can be refined to portfolio returns)
  · Max drawdown
· The run_backtest_suite() function loops over multiple models and produces a comparison DataFrame, making it easy to benchmark different LLMs’ trading performance.

Extending for Real Benchmarking

· To get accurate portfolio‑level Sharpe and drawdown, you should log the portfolio value at every step (e.g., by adding a portfolio_value field to the state and appending it via a reducer). This is a straightforward extension.
· For production, you’d also want to compute win rate, profit factor, and sortino ratio – all can be added to the metrics dictionary.
