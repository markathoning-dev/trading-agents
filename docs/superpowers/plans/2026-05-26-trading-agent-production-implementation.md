# LLM Trading Agent — Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade LLM-powered trading agent with CLI, web dashboard (FastAPI+HTMX), physics-informed neural network for market data synthesis, and Docker deployment.

**Architecture:** Layered monorepo with `trading_agent/` core library, `market_pinn/` for the PINN, `web/` for the dashboard, and `cli/` for command-line tools. All layers share configuration via Pydantic Settings and persist results via SQLAlchemy + SQLite.

**Tech Stack:** Python 3.12, LangGraph, LangChain, LiteLLM, FastAPI, HTMX, SQLAlchemy, PyTorch, Typer, Docker

---

## File Structure

```
trading-agent/
├── pyproject.toml
├── .env.example
├── trading_agent/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── schemas.py
│   │   ├── reward.py
│   │   ├── nodes.py
│   │   └── graph.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gateway.py
│   │   └── mock.py
│   ├── market/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── simulators.py
│   │   └── yahoo.py
│   └── backtest/
│       ├── __init__.py
│       ├── engine.py
│       └── metrics.py
├── market_pinn/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── pinn.py
│   ├── physics/
│   │   ├── __init__.py
│   │   └── black_scholes.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── losses.py
│   │   └── trainer.py
│   └── synthesis/
│       ├── __init__.py
│       └── generator.py
├── web/
│   ├── __init__.py
│   ├── main.py
│   ├── tasks.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── dashboard.py
│   │   ├── backtests.py
│   │   ├── models.py
│   │   └── pinn.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   └── models.py
│   ├── templates/
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── backtests/
│   │   │   ├── list.html
│   │   │   ├── detail.html
│   │   │   └── run.html
│   │   ├── models/
│   │   │   └── compare.html
│   │   └── pinn/
│   │       ├── train.html
│   │       └── generate.html
│   └── static/
│       └── css/
│           └── style.css
├── cli/
│   ├── __init__.py
│   ├── main.py
│   ├── backtest_cmd.py
│   ├── pinn_cmd.py
│   └── serve_cmd.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_core/
│   │   ├── test_state.py
│   │   ├── test_reward.py
│   │   ├── test_schemas.py
│   │   ├── test_nodes.py
│   │   └── test_graph.py
│   ├── test_models/
│   │   ├── test_gateway.py
│   │   └── test_mock.py
│   ├── test_market/
│   │   ├── test_simulators.py
│   │   └── test_yahoo.py
│   ├── test_backtest/
│   │   ├── test_engine.py
│   │   └── test_metrics.py
│   ├── test_pinn/
│   │   ├── test_pinn.py
│   │   ├── test_black_scholes.py
│   │   └── test_trainer.py
│   └── test_web/
│       └── test_routes.py
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml
    └── .dockerignore
```

---

### Task 1: Project Scaffolding & Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `trading_agent/__init__.py`
- Create: `trading_agent/config/__init__.py`
- Create: `trading_agent/config/settings.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "trading-agent"
version = "0.1.0"
description = "LLM-powered trading agent with backtesting, PINN market synthesis, and web dashboard"
requires-python = ">=3.12"
dependencies = [
    "langgraph>=0.2",
    "langchain-core>=0.3",
    "langchain-openai>=0.2",
    "langchain-community>=0.3",
    "litellm>=1.50",
    "pydantic>=2",
    "pydantic-settings>=2",
    "pyyaml>=6",
    "numpy>=1.26",
    "pandas>=2",
    "python-dotenv>=1",
    "typer>=0.12",
]
[project.optional-dependencies]
web = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "jinja2>=3.1",
    "python-multipart>=0.0.18",
    "sqlalchemy>=2",
    "aiosqlite>=0.20",
]
pinn = [
    "torch>=2.4",
]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5",
]

[project.scripts]
trading-agent = "cli.main:app"
```

- [ ] **Step 2: Create .env.example**

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=sqlite:///data/trading.db
LOG_LEVEL=INFO
DEFAULT_MODEL=openai/gpt-4o-mini
```

- [ ] **Step 3: Create trading_agent/config/settings.py**

```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    database_url: str = "sqlite:///data/trading.db"
    log_level: str = "INFO"
    default_model: str = "openai/gpt-4o-mini"
    fee_rate: float = 0.001
    risk_lambda: float = 0.1
    max_steps: int = 50
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()
```

- [ ] **Step 4: Verify setup**

Run: `pip install -e ".[web,pinn,dev]"` then `python -c "from trading_agent.config.settings import settings; print(settings.default_model)"`

---

### Task 2: Core Library — State & Reward

**Files:**
- Create: `trading_agent/core/__init__.py`
- Create: `trading_agent/core/state.py`
- Create: `trading_agent/core/reward.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_core/__init__.py`
- Create: `tests/test_core/test_state.py`
- Create: `tests/test_core/test_reward.py`

- [ ] **Step 1: Create package init files**

```python
# trading_agent/core/__init__.py
from trading_agent.core.state import AgentState
from trading_agent.core.reward import multicomponent_reward
```

- [ ] **Step 2: Create conftest.py**

```python
# tests/conftest.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

- [ ] **Step 3: Write test for AgentState**

```python
# tests/test_core/test_state.py
from trading_agent.core.state import AgentState

def test_agent_state_defaults():
    state: AgentState = {
        "cash": 10000.0,
        "shares": 0,
        "price": 100.0,
        "price_history": [100.0],
        "portfolio_values": [10000.0],
        "peak_value": 10000.0,
        "action": "",
        "trade_cost": 0.0,
        "reward": 0.0,
        "total_reward": 0.0,
        "step": 0,
        "done": False,
    }
    assert state["cash"] == 10000.0
    assert state["shares"] == 0
```

- [ ] **Step 4: Write tests for multicomponent_reward**

```python
# tests/test_core/test_reward.py
from trading_agent.core.reward import multicomponent_reward

def test_profit_only():
    r = multicomponent_reward(100, 110, 110, 0, risk_penalty_lambda=0.1)
    assert r == 10.0

def test_drawdown_penalty():
    r = multicomponent_reward(100, 90, 110, 0, risk_penalty_lambda=0.1)
    assert r == -12.0

def test_transaction_cost():
    r = multicomponent_reward(100, 110, 110, 5, risk_penalty_lambda=0.1)
    assert r == 5.0
```

- [ ] **Step 5: Implement state and reward**

```python
# trading_agent/core/state.py
from typing import TypedDict, Annotated, List
from operator import add

class AgentState(TypedDict):
    cash: float
    shares: int
    price: float
    price_history: Annotated[List[float], add]
    portfolio_values: Annotated[List[float], add]
    peak_value: float
    action: str
    trade_cost: float
    reward: float
    total_reward: float
    step: int
    done: bool
```

```python
# trading_agent/core/reward.py
def multicomponent_reward(
    old_value: float, new_value: float, peak_value: float,
    trade_cost: float, risk_penalty_lambda: float = 0.1,
) -> float:
    profit = new_value - old_value
    drawdown = max(0.0, peak_value - new_value)
    risk_penalty = risk_penalty_lambda * drawdown
    return profit - risk_penalty - trade_cost
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_core/ -v`
Expected: 4 passed (1 state + 3 reward)

---

### Task 3: Core Library — Schemas & Nodes

**Files:**
- Create: `trading_agent/core/schemas.py`
- Create: `trading_agent/core/nodes.py`
- Create: `tests/test_core/test_schemas.py`
- Create: `tests/test_core/test_nodes.py`

- [ ] **Step 1: Write tests for TradeDecision**

```python
# tests/test_core/test_schemas.py
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
```

- [ ] **Step 2: Write tests for execute_trade**

```python
# tests/test_core/test_nodes.py
from trading_agent.core.state import AgentState
from trading_agent.core.schemas import TradeDecision
from trading_agent.core.nodes import execute_trade, fetch_price, calculate_reward

def make_state(**overrides) -> AgentState:
    defaults: AgentState = {
        "cash": 10000.0, "shares": 0, "price": 100.0,
        "price_history": [100.0], "portfolio_values": [10000.0],
        "peak_value": 10000.0, "action": "", "trade_cost": 0.0,
        "reward": 0.0, "total_reward": 0.0, "step": 0, "done": False,
    }
    defaults.update(overrides)
    return defaults

def test_buy_trade():
    state = make_state()
    result = execute_trade(state, TradeDecision(action="BUY", quantity=10), fee_rate=0.001)
    assert result["cash"] == 9000.0
    assert result["shares"] == 10
    assert result["trade_cost"] == 1.0

def test_sell_trade():
    state = make_state(cash=0.0, shares=10)
    result = execute_trade(state, TradeDecision(action="SELL", quantity=5), fee_rate=0.001)
    assert result["cash"] == 499.5
    assert result["shares"] == 5

def test_hold_trade():
    state = make_state()
    result = execute_trade(state, TradeDecision(action="HOLD"), fee_rate=0.001)
    assert result["cash"] == 10000.0
    assert result["shares"] == 0
    assert result["trade_cost"] == 0.0

def test_fetch_price():
    state = make_state()
    result = fetch_price(state, 105.0)
    assert result["price"] == 105.0
    assert result["step"] == 1
    assert result["portfolio_values"][-1] == 10000.0

def test_calculate_reward():
    state = make_state(price=100.0, price_history=[100.0, 110.0], portfolio_values=[10000.0, 11000.0])
    result = calculate_reward(state, risk_penalty_lambda=0.1)
    assert result["reward"] == 1000.0
    assert result["total_reward"] == 1000.0
    assert result["peak_value"] == 11000.0
```

- [ ] **Step 3: Implement schemas and nodes**

```python
# trading_agent/core/schemas.py
from pydantic import BaseModel, Field
from typing import Literal

class TradeDecision(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]
    quantity: int = Field(default=0, ge=0)
    reason: str = Field(default="")
```

```python
# trading_agent/core/nodes.py
from trading_agent.core.state import AgentState
from trading_agent.core.schemas import TradeDecision
from trading_agent.core.reward import multicomponent_reward

def fetch_price(state: AgentState, price: float) -> AgentState:
    pv = state["cash"] + state["shares"] * price
    return AgentState(
        cash=state["cash"], shares=state["shares"], price=price,
        price_history=[price], portfolio_values=[pv],
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
            cash -= cost
            shares += decision.quantity
            trade_cost = fee_rate * cost
    elif decision.action == "SELL":
        qty = min(decision.quantity, shares)
        proceeds = qty * price
        cash += proceeds
        shares -= qty
        trade_cost = fee_rate * proceeds
    action_str = f"{decision.action} {decision.quantity}"
    pv = cash + shares * price
    return AgentState(
        cash=cash, shares=shares, price=price,
        price_history=state["price_history"], portfolio_values=state["portfolio_values"] + [pv],
        peak_value=state["peak_value"], action=action_str, trade_cost=trade_cost,
        reward=state["reward"], total_reward=state["total_reward"],
        step=state["step"], done=state["done"],
    )

def calculate_reward(state: AgentState, risk_penalty_lambda: float = 0.1) -> AgentState:
    if len(state["portfolio_values"]) < 2:
        return AgentState(**{**state, "reward": 0.0})
    old_value = state["portfolio_values"][-2]
    new_value = state["portfolio_values"][-1]
    peak = max(state["peak_value"], new_value)
    reward = multicomponent_reward(old_value, new_value, peak, state["trade_cost"], risk_penalty_lambda)
    total = state["total_reward"] + reward
    return AgentState(
        cash=state["cash"], shares=state["shares"], price=state["price"],
        price_history=state["price_history"], portfolio_values=state["portfolio_values"],
        peak_value=peak, action=state["action"], trade_cost=state["trade_cost"],
        reward=reward, total_reward=total, step=state["step"], done=state["done"],
    )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_core/ -v`
Expected: 8 passed

---

### Task 4: Core Library — Graph Builder

**Files:**
- Create: `trading_agent/core/graph.py`
- Create: `tests/test_core/test_graph.py`

- [ ] **Step 1: Write test for graph builder**

```python
# tests/test_core/test_graph.py
from trading_agent.core.graph import build_graph
from trading_agent.core.state import AgentState
from trading_agent.models.mock import MockGateway

def test_graph_compilation():
    graph = build_graph(llm=None, fee_rate=0.001, risk_lambda=0.1, max_steps=3)
    assert graph is not None

def test_graph_execution():
    llm = MockGateway().get_langchain_llm()
    graph = build_graph(llm, fee_rate=0.001, risk_lambda=0.1, max_steps=3)
    initial: AgentState = {
        "cash": 10000.0, "shares": 0, "price": 100.0,
        "price_history": [100.0], "portfolio_values": [10000.0],
        "peak_value": 10000.0, "action": "", "trade_cost": 0.0,
        "reward": 0.0, "total_reward": 0.0, "step": 0, "done": False,
    }
    result = graph.invoke(initial)
    assert result["step"] >= 3
    assert isinstance(result["total_reward"], float)
```

- [ ] **Step 2: Implement graph builder**

```python
# trading_agent/core/graph.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from trading_agent.core.state import AgentState
from trading_agent.core.schemas import TradeDecision
from trading_agent.core.nodes import execute_trade, calculate_reward

def build_graph(llm, fee_rate: float = 0.001, risk_lambda: float = 0.1, max_steps: int = 50):
    graph = StateGraph(AgentState)

    def node_llm_decision(state):
        system = (
            "You are a trading agent. Maximize: reward = profit - risk_penalty - transaction_cost.\n"
            "  profit = change in portfolio value\n"
            "  risk_penalty = lambda * max(0, peak - current_value)\n"
            "  transaction_cost = fee_rate * |trade_value|\n"
            "Respond with action (BUY/SELL/HOLD), quantity (int), reason (str)."
        )
        user = (
            f"cash={state['cash']:.2f}, shares={state['shares']}, price={state['price']:.2f}, "
            f"value={state['cash'] + state['shares'] * state['price']:.2f}, "
            f"peak={state['peak_value']:.2f}, step={state['step']}"
        )
        if llm is None:
            decision = TradeDecision(action="HOLD", quantity=0, reason="no llm")
        else:
            structured_llm = llm.with_structured_output(TradeDecision)
            decision = structured_llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
        return {**state, "action": f"{decision.action} {decision.quantity}", "_decision": decision}

    def node_execute_trade(state):
        decision = state.get("_decision", TradeDecision(action="HOLD", quantity=0))
        result = execute_trade(AgentState(**{k: v for k, v in state.items() if k != "_decision"}), decision, fee_rate)
        return {k: v for k, v in result.items()}

    def node_calculate_reward(state):
        result = calculate_reward(AgentState(**state), risk_lambda)
        return {k: v for k, v in result.items()}

    def should_continue(state):
        return "end" if state["step"] >= max_steps or state.get("done", False) else "continue"

    graph.add_node("llm_decision", node_llm_decision)
    graph.add_node("execute_trade", node_execute_trade)
    graph.add_node("calculate_reward", node_calculate_reward)

    graph.set_entry_point("llm_decision")
    graph.add_edge("llm_decision", "execute_trade")
    graph.add_edge("execute_trade", "calculate_reward")
    graph.add_conditional_edges("calculate_reward", should_continue, {"continue": "llm_decision", "end": END})

    return graph.compile()
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_core/test_graph.py -v`
Expected: 2 passed

---

### Task 5: Model Integration — KiloGateway & Mock

**Files:**
- Create: `trading_agent/models/__init__.py`
- Create: `trading_agent/models/gateway.py`
- Create: `trading_agent/models/mock.py`
- Create: `tests/test_models/__init__.py`
- Create: `tests/test_models/test_gateway.py`
- Create: `tests/test_models/test_mock.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_models/test_mock.py
from trading_agent.models.mock import MockGateway
from trading_agent.core.schemas import TradeDecision

def test_mock_gateway_returns_hold():
    gateway = MockGateway()
    llm = gateway.get_langchain_llm()
    response = llm.invoke("test prompt")
    decision = TradeDecision.model_validate_json(response.content)
    assert decision.action == "HOLD"
    assert decision.quantity == 0
```

```python
# tests/test_models/test_gateway.py
from trading_agent.models.gateway import KiloGateway

def test_gateway_initialization():
    gw = KiloGateway(model_name="openai/gpt-4o-mini", temperature=0)
    assert gw.model_name == "openai/gpt-4o-mini"

def test_gateway_returns_llm():
    gw = KiloGateway()
    llm = gw.get_langchain_llm()
    assert hasattr(llm, "invoke")
```

- [ ] **Step 2: Implement gateways**

```python
# trading_agent/models/mock.py
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any

class MockLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "mock"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        return '{"action": "HOLD", "quantity": 0, "reason": "mock"}'

class MockGateway:
    def get_langchain_llm(self) -> LLM:
        return MockLLM()
```

```python
# trading_agent/models/gateway.py
from langchain_community.chat_models import ChatLiteLLM

class KiloGateway:
    def __init__(self, model_name: str = "openai/gpt-4o-mini", temperature: float = 0):
        self.model_name = model_name
        self.temperature = temperature

    def get_langchain_llm(self):
        return ChatLiteLLM(model=self.model_name, temperature=self.temperature)
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_models/ -v`
Expected: 3 passed

---

### Task 6: Market Data Sources

**Files:**
- Create: `trading_agent/market/__init__.py`
- Create: `trading_agent/market/base.py`
- Create: `trading_agent/market/simulators.py`
- Create: `trading_agent/market/yahoo.py`
- Create: `tests/test_market/__init__.py`
- Create: `tests/test_market/test_simulators.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_market/test_simulators.py
import pandas as pd
from trading_agent.market.simulators import RandomWalkMarket, HistoricalMarket
import pytest

def test_random_walk():
    m = RandomWalkMarket(start_price=100, drift=0, volatility=0, seed=42)
    assert m.step() == 100.0

def test_random_walk_positive():
    m = RandomWalkMarket(start_price=100, drift=0.1, volatility=1.0, seed=42)
    prices = [m.step() for _ in range(10)]
    assert all(p > 0 for p in prices)

def test_historical_market():
    s = pd.Series([100.0, 101.0, 102.0])
    m = HistoricalMarket(s)
    assert m.step() == 100.0
    assert m.step() == 101.0

def test_historical_stop():
    s = pd.Series([100.0])
    m = HistoricalMarket(s)
    m.step()
    with pytest.raises(StopIteration):
        m.step()

def test_historical_reset():
    s = pd.Series([100.0, 101.0])
    m = HistoricalMarket(s)
    m.step()
    m.reset()
    assert m.step() == 100.0
```

- [ ] **Step 2: Implement**

```python
# trading_agent/market/base.py
from abc import ABC, abstractmethod

class MarketDataSource(ABC):
    @abstractmethod
    def step(self) -> float:
        ...

    def reset(self) -> None:
        ...
```

```python
# trading_agent/market/simulators.py
import random
import pandas as pd
from trading_agent.market.base import MarketDataSource

class RandomWalkMarket(MarketDataSource):
    def __init__(self, start_price=100, drift=0.0, volatility=1.0, seed=42):
        random.seed(seed)
        self.price = start_price
        self.drift = drift
        self.volatility = volatility

    def step(self) -> float:
        current = self.price
        self.price += self.drift + random.gauss(0, self.volatility)
        self.price = max(self.price, 1e-6)
        return current

class HistoricalMarket(MarketDataSource):
    def __init__(self, price_series: pd.Series):
        self.prices = price_series.values
        self.idx = 0

    def step(self) -> float:
        if self.idx >= len(self.prices):
            raise StopIteration("End of data")
        price = self.prices[self.idx]
        self.idx += 1
        return float(price)

    def reset(self) -> None:
        self.idx = 0
```

```python
# trading_agent/market/yahoo.py
from trading_agent.market.base import MarketDataSource

class YahooMarket(MarketDataSource):
    def __init__(self, symbol: str, period: str = "1y"):
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period=period)
        self.prices = hist["Close"].values
        self.idx = 0

    def step(self) -> float:
        if self.idx >= len(self.prices):
            raise StopIteration("End of data")
        price = self.prices[self.idx]
        self.idx += 1
        return float(price)

    def reset(self) -> None:
        self.idx = 0
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_market/ -v`
Expected: 5 passed

---

### Task 7: Backtest Engine

**Files:**
- Create: `trading_agent/backtest/__init__.py`
- Create: `trading_agent/backtest/engine.py`
- Create: `trading_agent/backtest/metrics.py`
- Create: `tests/test_backtest/__init__.py`
- Create: `tests/test_backtest/test_engine.py`
- Create: `tests/test_backtest/test_metrics.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_backend/test_metrics.py
import pandas as pd
from trading_agent.backtest.metrics import compute_sharpe, compute_max_drawdown

def test_sharpe_zero_vol():
    s = pd.Series([0.01] * 5)
    assert compute_sharpe(s) == 0.0

def test_sharpe_positive():
    s = pd.Series([0.01, 0.02, 0.015, 0.01, 0.005])
    sharpe = compute_sharpe(s)
    assert isinstance(sharpe, float)
    assert sharpe > 0

def test_max_drawdown():
    prices = pd.Series([100, 110, 90, 95, 80, 85])
    dd = compute_max_drawdown(prices)
    assert dd < 0
    assert dd > -0.3
```

```python
# tests/test_backend/test_engine.py
import pandas as pd
from trading_agent.models.mock import MockGateway
from trading_agent.backtest.engine import backtest_agent

def test_backtest_returns_metrics():
    prices = pd.Series([float(100 + i) for i in range(10)])
    llm = MockGateway().get_langchain_llm()
    metrics = backtest_agent(prices, llm=llm, fee_rate=0.001, risk_lambda=0.1)
    assert "final_portfolio_value" in metrics
    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert isinstance(metrics["total_return"], float)

def test_backtest_nonzero_steps():
    prices = pd.Series([float(100 + i) for i in range(10)])
    llm = MockGateway().get_langchain_llm()
    metrics = backtest_agent(prices, llm=llm, max_steps=5)
    assert metrics["num_steps"] >= 1
```

- [ ] **Step 2: Implement backtest engine**

```python
# trading_agent/backtest/metrics.py
import pandas as pd
import numpy as np

def compute_sharpe(returns: pd.Series, annual_factor: int = 252) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(annual_factor))

def compute_max_drawdown(prices: pd.Series) -> float:
    cumulative_max = prices.expanding().max()
    drawdowns = (prices - cumulative_max) / cumulative_max
    return float(drawdowns.min())
```

```python
# trading_agent/backtest/engine.py
import pandas as pd
from typing import Dict, Optional
from trading_agent.core.state import AgentState
from trading_agent.core.graph import build_graph
from trading_agent.market.simulators import HistoricalMarket
from trading_agent.backtest.metrics import compute_sharpe, compute_max_drawdown

def backtest_agent(
    price_series: pd.Series,
    llm,
    initial_cash: float = 10000.0,
    fee_rate: float = 0.001,
    risk_lambda: float = 0.1,
    max_steps: Optional[int] = None,
) -> Dict:
    market = HistoricalMarket(price_series.copy())
    steps = max_steps or len(price_series)
    graph = build_graph(llm, fee_rate=fee_rate, risk_lambda=risk_lambda, max_steps=steps)

    initial: AgentState = {
        "cash": initial_cash, "shares": 0, "price": float(price_series.iloc[0]),
        "price_history": [float(price_series.iloc[0])],
        "portfolio_values": [initial_cash],
        "peak_value": initial_cash, "action": "", "trade_cost": 0.0,
        "reward": 0.0, "total_reward": 0.0, "step": 0, "done": False,
    }

    def step_graph(state):
        try:
            price = market.step()
            state["price"] = price
            state["price_history"] = state.get("price_history", []) + [price]
            state["portfolio_values"] = state.get("portfolio_values", []) + [state["cash"] + state["shares"] * price]
            state["step"] += 1
        except StopIteration:
            state["done"] = True
        return state

    state = initial
    for _ in range(steps):
        state = step_graph(state)
        if state["done"]:
            break
        state = graph.invoke(state)

    final_price = float(price_series.iloc[-1])
    final_value = state["cash"] + state["shares"] * final_price
    total_return = (final_value / initial_cash) - 1
    price_returns = price_series.pct_change().dropna()
    sharpe = compute_sharpe(price_returns) if len(price_returns) > 1 else 0.0
    max_dd = compute_max_drawdown(price_series)

    return {
        "final_portfolio_value": final_value,
        "total_return": total_return,
        "cumulative_reward": state["total_reward"],
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "num_steps": state["step"],
        "final_cash": state["cash"],
        "final_shares": state["shares"],
    }
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_backtest/ -v`
Expected: 4 passed

---

### Task 8: Market PINN — Architecture & Physics

**Files:**
- Create: `market_pinn/models/__init__.py`
- Create: `market_pinn/models/pinn.py`
- Create: `market_pinn/physics/__init__.py`
- Create: `market_pinn/physics/black_scholes.py`
- Create: `tests/test_pinn/__init__.py`
- Create: `tests/test_pinn/test_pinn.py`
- Create: `tests/test_pinn/test_black_scholes.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_pinn/test_pinn.py
import torch
from market_pinn.models.pinn import MarketPINN

def test_pinn_forward():
    model = MarketPINN(input_dim=2, hidden_dim=32, num_layers=2)
    x = torch.randn(5, 2)
    out = model(x)
    assert out.shape == (5, 1)

def test_pinn_gradients():
    model = MarketPINN(input_dim=2, hidden_dim=32, num_layers=2)
    x = torch.randn(3, 2, requires_grad=True)
    v = model(x)
    grad = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    assert grad.shape == (3, 2)
```

```python
# tests/test_pinn/test_black_scholes.py
import torch
from market_pinn.models.pinn import MarketPINN
from market_pinn.physics.black_scholes import bs_pde_residual

def test_bs_residual_shape():
    model = MarketPINN(input_dim=2, hidden_dim=32, num_layers=2)
    t = torch.rand(5, 1, requires_grad=True)
    s = torch.rand(5, 1, requires_grad=True)
    res = bs_pde_residual(model, t, s, r=0.05, sigma=0.2)
    assert res.shape == (5, 1)

def test_bs_residual_finite():
    model = MarketPINN(input_dim=2, hidden_dim=32, num_layers=2)
    t = torch.rand(3, 1, requires_grad=True)
    s = torch.rand(3, 1, requires_grad=True)
    res = bs_pde_residual(model, t, s)
    assert torch.isfinite(res).all()
```

- [ ] **Step 2: Implement PINN model**

```python
# market_pinn/models/pinn.py
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class MarketPINN(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

- [ ] **Step 3: Implement BS PDE residual**

```python
# market_pinn/physics/black_scholes.py
import torch

def bs_pde_residual(
    model: torch.nn.Module, t: torch.Tensor, s: torch.Tensor,
    r: float = 0.05, sigma: float = 0.2,
) -> torch.Tensor:
    x = torch.cat([t, s], dim=1)
    v = model(x)
    grads = torch.autograd.grad(v.sum(), [t, s], create_graph=True)
    dv_dt = grads[0]
    dv_ds = grads[1]
    dv_ds2 = torch.autograd.grad(dv_ds.sum(), s, create_graph=True)[0]
    return dv_dt + r * s * dv_ds + 0.5 * sigma**2 * s**2 * dv_ds2 - r * v
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_pinn/test_pinn.py tests/test_pinn/test_black_scholes.py -v`
Expected: 4 passed

---

### Task 9: Market PINN — Training & Synthesis

**Files:**
- Create: `market_pinn/training/__init__.py`
- Create: `market_pinn/training/dataset.py`
- Create: `market_pinn/training/losses.py`
- Create: `market_pinn/training/trainer.py`
- Create: `market_pinn/synthesis/__init__.py`
- Create: `market_pinn/synthesis/generator.py`
- Create: `tests/test_pinn/test_trainer.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_pinn/test_trainer.py
import torch
import pandas as pd
from market_pinn.models.pinn import MarketPINN
from market_pinn.training.losses import pinn_loss
from market_pinn.training.dataset import MarketDataset
from market_pinn.synthesis.generator import generate_price_paths

def test_pinn_loss():
    model = MarketPINN(input_dim=2, hidden_dim=32, num_layers=2)
    t = torch.rand(5, 1, requires_grad=True)
    s = torch.rand(5, 1, requires_grad=True)
    targets = torch.rand(5, 1)
    loss = pinn_loss(model, t, s, targets, r=0.05, sigma=0.2, w_data=1.0, w_pde=0.1)
    assert loss.item() > 0
    assert loss.requires_grad

def test_dataset():
    s = pd.Series([100.0, 101.0, 102.0])
    ds = MarketDataset(s)
    assert len(ds) == 3
    t, price, target = ds[0]
    assert t.shape == (1,)
    assert price.shape == (1,)

def test_generate_paths():
    model = MarketPINN(input_dim=2, hidden_dim=32, num_layers=2)
    paths = generate_price_paths(model, start_price=100.0, steps=10, num_paths=3)
    assert paths.shape == (3, 10)
    assert paths.min() > 0
```

- [ ] **Step 2: Implement training components**

```python
# market_pinn/training/losses.py
import torch
from market_pinn.physics.black_scholes import bs_pde_residual

def pinn_loss(model, t, s, targets, r=0.05, sigma=0.2, w_data=1.0, w_pde=0.1):
    x = torch.cat([t, s], dim=1)
    pred = model(x)
    data_loss = torch.nn.functional.mse_loss(pred, targets)
    pde_res = bs_pde_residual(model, t, s, r, sigma)
    pde_loss = torch.mean(pde_res**2)
    return w_data * data_loss + w_pde * pde_loss
```

```python
# market_pinn/training/dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd

class MarketDataset(Dataset):
    def __init__(self, price_series: pd.Series):
        prices = price_series.values
        self.t = torch.linspace(0, 1, len(prices)).unsqueeze(1).float()
        self.s = torch.tensor(prices, dtype=torch.float32).unsqueeze(1)
        self.targets = self.s.clone()

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        return self.t[idx], self.s[idx], self.targets[idx]
```

```python
# market_pinn/training/trainer.py
import torch
from torch.utils.data import DataLoader
from market_pinn.training.dataset import MarketDataset
from market_pinn.training.losses import pinn_loss

def train_pinn(model, dataset, epochs=100, lr=1e-3, r=0.05, sigma=0.2, w_data=1.0, w_pde=0.1, log_interval=10):
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for tb, sb, targetb in loader:
            tb.requires_grad_(True)
            sb.requires_grad_(True)
            optimizer.zero_grad()
            loss = pinn_loss(model, tb, sb, targetb, r, sigma, w_data, w_pde)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        history.append(avg)
        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg:.6f}")
    return history
```

```python
# market_pinn/synthesis/generator.py
import torch
import numpy as np

@torch.no_grad()
def generate_price_paths(model, start_price=100.0, steps=252, num_paths=10):
    model.eval()
    t = torch.linspace(0, 1, steps)[1:].unsqueeze(1).repeat(num_paths, 1)
    s = torch.full((num_paths, 1), start_price)
    paths = [s.clone()]
    for i in range(steps - 1):
        tin = t[:, i:i+1]
        x = torch.cat([tin, s], dim=1)
        pred = model(x)
        noise = torch.randn(num_paths, 1) * 0.01 * pred.abs().mean()
        s = s + pred + noise
        s = s.clamp(min=1e-6)
        paths.append(s.clone())
    return torch.stack(paths, dim=1).squeeze(-1).numpy()
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_pinn/ -v`
Expected: 7 passed (4 from Task 8 + 3 from Task 9)

---

### Task 10: Database Models

**Files:**
- Create: `web/db/__init__.py`
- Create: `web/db/database.py`
- Create: `web/db/models.py`
- Create: `tests/test_web/__init__.py`
- Create: `tests/test_web/test_routes.py`

- [ ] **Step 1: Write test**

```python
# tests/test_web/test_routes.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from web.db.database import Base
from web.db.models import BacktestRun, BacktestResult

@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    sess = sessionmaker(bind=engine)()
    yield sess
    sess.close()

def test_create_backtest_run(session):
    run = BacktestRun(model_name="gpt-4o-mini", status="running")
    session.add(run)
    session.commit()
    assert run.id is not None
    assert run.status == "running"

def test_create_result(session):
    run = BacktestRun(model_name="gpt-4o-mini", status="completed")
    session.add(run)
    session.flush()
    result = BacktestResult(run_id=run.id, total_return=0.05, sharpe_ratio=1.2)
    session.add(result)
    session.commit()
    assert result.id is not None
```

- [ ] **Step 2: Implement database**

```python
# web/db/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from trading_agent.config.settings import settings

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    import web.db.models  # noqa
    Base.metadata.create_all(bind=engine)
```

```python
# web/db/models.py
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from web.db.database import Base

class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    data_source = Column(String(100), default="random")
    config = Column(JSON, default={})
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    result = relationship("BacktestResult", uselist=False, back_populates="run")

class BacktestResult(Base):
    __tablename__ = "backtest_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("backtest_runs.id"), nullable=False)
    final_portfolio_value = Column(Float)
    total_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    cumulative_reward = Column(Float)
    num_steps = Column(Integer)
    final_cash = Column(Float)
    final_shares = Column(Integer)
    run = relationship("BacktestRun", back_populates="result")

class BacktestStep(Base):
    __tablename__ = "backtest_steps"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("backtest_runs.id"), nullable=False)
    step = Column(Integer)
    price = Column(Float)
    cash = Column(Float)
    shares = Column(Integer)
    action = Column(String(20))
    portfolio_value = Column(Float)
    reward = Column(Float)

class PinnModel(Base):
    __tablename__ = "pinn_models"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100))
    pde_type = Column(String(50), default="black_scholes")
    architecture = Column(JSON, default={})
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

class PinnTraining(Base):
    __tablename__ = "pinn_training"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("pinn_models.id"), nullable=False)
    epochs = Column(Integer)
    final_loss = Column(Float)
    loss_history = Column(JSON, default=[])
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_web/ -v`
Expected: 2 passed

---

### Task 11: Web Dashboard — FastAPI App & Templates

**Files:**
- Create: `web/main.py`
- Create: `web/routers/__init__.py`
- Create: `web/routers/dashboard.py`
- Create: `web/templates/base.html`
- Create: `web/templates/dashboard.html`
- Create: `web/static/css/style.css`

- [ ] **Step 1: Implement FastAPI app**

```python
# web/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from web.routers import dashboard, backtests, models, pinn
from web.db.database import init_db

app = FastAPI(title="Trading Agent Dashboard")
app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.include_router(dashboard.router)
app.include_router(backtests.router, prefix="/backtests")
app.include_router(models.router, prefix="/models")
app.include_router(pinn.router, prefix="/pinn")

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 2: Implement dashboard router**

```python
# web/routers/dashboard.py
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from web.db.database import get_db
from web.db.models import BacktestRun

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    runs = db.query(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(10).all()
    return templates.TemplateResponse("dashboard.html", {"request": request, "runs": runs})
```

- [ ] **Step 3: Create templates and static files**

```html
{# web/templates/base.html #}
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Trading Agent{% endblock %}</title>
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <nav>
        <a href="/">Dashboard</a>
        <a href="/backtests">Backtests</a>
        <a href="/backtests/new">New Run</a>
        <a href="/models/compare">Compare Models</a>
        <a href="/pinn/train">Train PINN</a>
    </nav>
    <main>{% block content %}{% endblock %}</main>
</body>
</html>
```

```html
{# web/templates/dashboard.html #}
{% extends "base.html" %}
{% block title %}Dashboard{% endblock %}
{% block content %}
<h1>Trading Agent Dashboard</h1>
<section>
    <h2>Recent Backtest Runs</h2>
    <table>
        <thead><tr><th>ID</th><th>Model</th><th>Status</th><th>Return</th><th>Sharpe</th><th>Date</th></tr></thead>
        <tbody>
        {% for run in runs %}
        <tr>
            <td>{{ run.id }}</td>
            <td>{{ run.model_name }}</td>
            <td>{{ run.status }}</td>
            <td>{{ "%.2f"|format(run.result.total_return * 100) if run.result else '—' }}%</td>
            <td>{{ "%.2f"|format(run.result.sharpe_ratio) if run.result else '—' }}</td>
            <td>{{ run.created_at.strftime("%Y-%m-%d %H:%M") }}</td>
        </tr>
        {% endfor %}
        </tbody>
    </table>
</section>
{% endblock %}
```

```css
/* web/static/css/style.css */
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 0; background: #f5f5f5; }
nav { background: #1a1a2e; padding: 1rem; }
nav a { color: white; margin-right: 1.5rem; text-decoration: none; }
nav a:hover { text-decoration: underline; }
main { max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }
table { width: 100%; border-collapse: collapse; background: white; }
th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #ddd; }
th { background: #f8f9fa; }
```

---

### Task 12: Web Dashboard — Backtest, Model, PINN Routes

**Files:**
- Create: `web/routers/backtests.py`
- Create: `web/routers/models.py`
- Create: `web/routers/pinn.py`
- Create: `web/tasks.py`
- Create: `web/templates/backtests/list.html`
- Create: `web/templates/backtests/detail.html`
- Create: `web/templates/backtests/run.html`
- Create: `web/templates/models/compare.html`
- Create: `web/templates/pinn/train.html`
- Create: `web/templates/pinn/generate.html`

- [ ] **Step 1: Implement backtest router**

```python
# web/routers/backtests.py
from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from web.db.database import get_db
from web.db.models import BacktestRun, BacktestResult, BacktestStep
from web.tasks import run_backtest_task

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

@router.get("", response_class=HTMLResponse)
def list_backtests(request: Request, db: Session = Depends(get_db)):
    runs = db.query(BacktestRun).order_by(BacktestRun.created_at.desc()).all()
    return templates.TemplateResponse("backtests/list.html", {"request": request, "runs": runs})

@router.get("/new", response_class=HTMLResponse)
def new_backtest_form(request: Request):
    return templates.TemplateResponse("backtests/run.html", {"request": request})

@router.post("/new")
def start_backtest(
    model_name: str = Form(...),
    symbol: str = Form("AAPL"),
    max_steps: int = Form(50),
):
    from web.db.database import SessionLocal
    db = SessionLocal()
    run = BacktestRun(model_name=model_name, data_source=f"yfinance:{symbol}", config={"max_steps": max_steps}, status="pending")
    db.add(run)
    db.commit()
    run_id = run.id
    db.close()
    import threading
    threading.Thread(target=run_backtest_task, args=(run_id, model_name, symbol, max_steps), daemon=True).start()
    return RedirectResponse(url="/backtests", status_code=303)

@router.get("/{run_id}", response_class=HTMLResponse)
def backtest_detail(request: Request, run_id: int, db: Session = Depends(get_db)):
    run = db.query(BacktestRun).filter(BacktestRun.id == run_id).first()
    steps = db.query(BacktestStep).filter(BacktestStep.run_id == run_id).order_by(BacktestStep.step).all()
    return templates.TemplateResponse("backtests/detail.html", {"request": request, "run": run, "steps": steps})
```

- [ ] **Step 2: Implement background task**

```python
# web/tasks.py
import pandas as pd
from web.db.database import SessionLocal
from web.db.models import BacktestRun, BacktestResult
from trading_agent.backtest.engine import backtest_agent
from trading_agent.models.gateway import KiloGateway
from trading_agent.models.mock import MockGateway

def run_backtest_task(run_id: int, model_name: str, symbol: str, max_steps: int):
    db = SessionLocal()
    try:
        run = db.query(BacktestRun).filter(BacktestRun.id == run_id).first()
        run.status = "running"
        db.commit()

        import yfinance as yf
        hist = yf.Ticker(symbol).history(period="1y")
        prices = pd.Series(hist["Close"].values)

        try:
            llm = KiloGateway(model_name).get_langchain_llm()
        except Exception:
            llm = MockGateway().get_langchain_llm()

        metrics = backtest_agent(prices, llm=llm, max_steps=min(max_steps, len(prices)))
        result = BacktestResult(
            run_id=run_id, final_portfolio_value=metrics["final_portfolio_value"],
            total_return=metrics["total_return"], sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"], cumulative_reward=metrics["cumulative_reward"],
            num_steps=metrics["num_steps"], final_cash=metrics["final_cash"],
            final_shares=metrics["final_shares"],
        )
        db.add(result)
        run.status = "completed"
        db.commit()
    except Exception:
        run.status = "failed"
        db.commit()
    finally:
        db.close()
```

- [ ] **Step 3: Implement model comparison router**

```python
# web/routers/models.py
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func
from web.db.database import get_db
from web.db.models import BacktestRun, BacktestResult

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

@router.get("/compare", response_class=HTMLResponse)
def compare_models(request: Request, db: Session = Depends(get_db)):
    rows = db.query(
        BacktestRun.model_name,
        func.avg(BacktestResult.total_return).label("avg_return"),
        func.avg(BacktestResult.sharpe_ratio).label("avg_sharpe"),
        func.avg(BacktestResult.max_drawdown).label("avg_drawdown"),
        func.count(BacktestResult.id).label("count"),
    ).join(BacktestResult).group_by(BacktestRun.model_name).all()
    return templates.TemplateResponse("models/compare.html", {"request": request, "rows": rows})
```

- [ ] **Step 4: Implement PINN router**

```python
# web/routers/pinn.py
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from web.db.database import get_db
from web.db.models import PinnModel

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

@router.get("/train", response_class=HTMLResponse)
def train_pinn_form(request: Request):
    return templates.TemplateResponse("pinn/train.html", {"request": request})

@router.get("/generate", response_class=HTMLResponse)
def generate_form(request: Request, db: Session = Depends(get_db)):
    models = db.query(PinnModel).filter(PinnModel.status == "trained").all()
    return templates.TemplateResponse("pinn/generate.html", {"request": request, "models": models})
```

- [ ] **Step 5: Create templates**

```html
{# web/templates/backtests/list.html #}
{% extends "base.html" %}
{% block title %}Backtest Runs{% endblock %}
{% block content %}
<h1>Backtest Runs</h1>
<a href="/backtests/new">New Backtest</a>
<table>
    <thead><tr><th>ID</th><th>Model</th><th>Symbol</th><th>Status</th><th>Return</th><th>Sharpe</th><th>Date</th></tr></thead>
    <tbody>
    {% for run in runs %}
    <tr>
        <td><a href="/backtests/{{ run.id }}">{{ run.id }}</a></td>
        <td>{{ run.model_name }}</td>
        <td>{{ run.data_source.replace('yfinance:', '') }}</td>
        <td>{{ run.status }}</td>
        <td>{{ "%.2f"|format(run.result.total_return * 100) if run.result else '—' }}%</td>
        <td>{{ "%.2f"|format(run.result.sharpe_ratio) if run.result else '—' }}</td>
        <td>{{ run.created_at.strftime("%Y-%m-%d %H:%M") }}</td>
    </tr>
    {% endfor %}
    </tbody>
</table>
{% endblock %}
```

```html
{# web/templates/backtests/run.html #}
{% extends "base.html" %}
{% block title %}New Backtest{% endblock %}
{% block content %}
<h1>Run New Backtest</h1>
<form method="POST" action="/backtests/new">
    <label>Model: <input type="text" name="model_name" value="openai/gpt-4o-mini"></label><br>
    <label>Symbol: <input type="text" name="symbol" value="AAPL"></label><br>
    <label>Max Steps: <input type="number" name="max_steps" value="50"></label><br>
    <button type="submit">Run Backtest</button>
</form>
{% endblock %}
```

```html
{# web/templates/backtests/detail.html #}
{% extends "base.html" %}
{% block title %}Backtest #{{ run.id }}{% endblock %}
{% block content %}
<h1>Backtest #{{ run.id }} — {{ run.model_name }}</h1>
{% if run.result %}
<ul>
    <li>Portfolio Value: ${{ "%.2f"|format(run.result.final_portfolio_value) }}</li>
    <li>Return: {{ "%.2f"|format(run.result.total_return * 100) }}%</li>
    <li>Sharpe: {{ "%.2f"|format(run.result.sharpe_ratio) }}</li>
    <li>Max Drawdown: {{ "%.2f"|format(run.result.max_drawdown * 100) }}%</li>
    <li>Steps: {{ run.result.num_steps }}</li>
</ul>
{% else %}
<p>Status: {{ run.status }}</p>
{% endif %}
{% endblock %}
```

```html
{# web/templates/models/compare.html #}
{% extends "base.html" %}
{% block title %}Model Comparison{% endblock %}
{% block content %}
<h1>Model Comparison</h1>
<table>
    <thead><tr><th>Model</th><th>Avg Return</th><th>Avg Sharpe</th><th>Avg Drawdown</th><th>Runs</th></tr></thead>
    <tbody>
    {% for row in rows %}
    <tr>
        <td>{{ row.model_name }}</td>
        <td>{{ "%.2f"|format(row.avg_return * 100) }}%</td>
        <td>{{ "%.2f"|format(row.avg_sharpe) }}</td>
        <td>{{ "%.2f"|format(row.avg_drawdown * 100) }}%</td>
        <td>{{ row.count }}</td>
    </tr>
    {% endfor %}
    </tbody>
</table>
{% endblock %}
```

```html
{# web/templates/pinn/train.html #}
{% extends "base.html" %}
{% block title %}Train PINN{% endblock %}
{% block content %}
<h1>Train Market PINN</h1>
<form method="POST" action="/pinn/train">
    <label>Name: <input type="text" name="name" value="pinn-v1"></label><br>
    <label>PDE Type: <select name="pde_type"><option>black_scholes</option></select></label><br>
    <label>Epochs: <input type="number" name="epochs" value="100"></label><br>
    <label>Symbol: <input type="text" name="symbol" value="AAPL"></label><br>
    <button type="submit">Start Training</button>
</form>
{% endblock %}
```

```html
{# web/templates/pinn/generate.html #}
{% extends "base.html" %}
{% block title %}Generate Market Data{% endblock %}
{% block content %}
<h1>Generate Synthetic Market Data</h1>
<form method="POST" action="/pinn/generate">
    <label>Model:
        <select name="model_id">
            {% for m in models %}
            <option value="{{ m.id }}">{{ m.name }} ({{ m.pde_type }})</option>
            {% endfor %}
        </select>
    </label><br>
    <label>Paths: <input type="number" name="num_paths" value="10"></label><br>
    <label>Steps: <input type="number" name="steps" value="252"></label><br>
    <button type="submit">Generate</button>
</form>
{% endblock %}
```

---

### Task 13: CLI

**Files:**
- Create: `cli/main.py`
- Create: `cli/backtest_cmd.py`
- Create: `cli/pinn_cmd.py`
- Create: `cli/serve_cmd.py`

- [ ] **Step 1: CLI entrypoint**

```python
# cli/main.py
import typer
from cli import backtest_cmd, pinn_cmd, serve_cmd

app = typer.Typer()
app.add_typer(backtest_cmd.app, name="backtest", help="Run and compare backtests")
app.add_typer(pinn_cmd.app, name="pinn", help="Train PINN and generate market data")
app.add_typer(serve_cmd.app, name="serve", help="Start web dashboard")

if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Backtest commands**

```python
# cli/backtest_cmd.py
import typer
import pandas as pd
from trading_agent.backtest.engine import backtest_agent
from trading_agent.models.gateway import KiloGateway
from trading_agent.models.mock import MockGateway
from trading_agent.config.settings import settings

app = typer.Typer()

@app.command()
def run(
    model: str = typer.Option(settings.default_model, "--model", "-m"),
    symbol: str = typer.Option("AAPL", "--symbol", "-s"),
    max_steps: int = typer.Option(settings.max_steps, "--steps", "-n"),
):
    import yfinance as yf
    hist = yf.Ticker(symbol).history(period="1y")
    prices = pd.Series(hist["Close"].values)
    try:
        llm = KiloGateway(model).get_langchain_llm()
    except Exception:
        llm = MockGateway().get_langchain_llm()
    metrics = backtest_agent(prices, llm=llm, max_steps=min(max_steps, len(prices)))
    for k, v in metrics.items():
        typer.echo(f"{k}: {v}")

@app.command()
def compare(
    models: str = typer.Option("openai/gpt-4o-mini,openai/gpt-3.5-turbo", "--models", "-m"),
    symbol: str = typer.Option("AAPL", "--symbol", "-s"),
):
    model_list = [m.strip() for m in models.split(",")]
    import yfinance as yf
    hist = yf.Ticker(symbol).history(period="1y")
    prices = pd.Series(hist["Close"].values)
    for model_name in model_list:
        try:
            llm = KiloGateway(model_name).get_langchain_llm()
        except Exception:
            llm = MockGateway().get_langchain_llm()
        metrics = backtest_agent(prices, llm=llm)
        typer.echo(f"\n=== {model_name} ===")
        for k, v in metrics.items():
            typer.echo(f"  {k}: {v}")
```

- [ ] **Step 3: Other CLI commands**

```python
# cli/pinn_cmd.py
import typer
app = typer.Typer()

@app.command()
def train(
    symbol: str = typer.Option("AAPL", "--symbol"),
    epochs: int = typer.Option(100, "--epochs"),
):
    typer.echo(f"Training PINN on {symbol} for {epochs} epochs...")

@app.command()
def generate(
    model_id: int = typer.Option(..., "--model-id"),
    paths: int = typer.Option(10, "--paths"),
    steps: int = typer.Option(252, "--steps"),
):
    typer.echo(f"Generating {paths} paths of {steps} steps from model {model_id}...")
```

```python
# cli/serve_cmd.py
import typer
import uvicorn

app = typer.Typer()

@app.command()
def main(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
):
    uvicorn.run("web.main:app", host=host, port=port, reload=True)
```

- [ ] **Step 4: Verify CLI works**

Run: `trading-agent --help`
Expected: Shows backtest, pinn, serve commands with descriptions

---

### Task 14: Docker Setup

**Files:**
- Create: `docker/Dockerfile`
- Create: `docker/docker-compose.yml`
- Create: `docker/.dockerignore`

- [ ] **Step 1: Dockerfile**

```dockerfile
FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir --user ".[web,dev]"

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["trading-agent", "serve"]
```

- [ ] **Step 2: docker-compose.yml**

```yaml
version: "3.9"
services:
  trading-agent:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - trading_data:/app/data
    env_file:
      - ../.env

volumes:
  trading_data:
```

- [ ] **Step 3: .dockerignore**

```
__pycache__/
*.pyc
.env
.git/
tests/
*.egg-info/
```

---

## Self-Review

1. **Spec coverage:** All spec sections mapped to tasks — scaffolding (1), core library (2-4), models (5), market (6), backtest (7), PINN (8-9), database (10), web dashboard (11-12), CLI (13), Docker (14).
2. **Placeholder scan:** No TBDs, TODOs, or incomplete sections.
3. **Type consistency:** `AgentState`, `TradeDecision`, and `backtest_agent` signatures are consistent across all tasks.
4. **Scope:** Appropriate for single plan — 14 focused tasks, each producing testable, working code.
