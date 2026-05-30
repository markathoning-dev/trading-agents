# Trading Agent — Architecture

## System Overview

`trading-agent` is an LLM-powered trading agent built on LangGraph state machines. It connects an LLM decision core to a composable strategy system (MTG-inspired "deck" of strategy cards), multiple market data sources, two market simulators (CGAN and PINN), a parallel backtesting engine, and a terminal-themed web dashboard.

```
┌─────────────────────────────────────────────────────────┐
│                    CLI (Typer)                          │
│  backtest  │  cards  │  cgan  │  pinn  │  serve         │
└────┬────┬──┴───┬─────┴───┬────┴───┬────┴───┬───────────┘
     │    │      │         │        │        │
     ▼    ▼      ▼         ▼        ▼        ▼
┌──────────────────────────────────────────────────────────────┐
│                   trading_agent/ (Core Library)               │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │  LangGraph    │   │  LLM Gateway │   │  Market Adapters │  │
│  │  State Machine│   │  (LiteLLM)   │   │  Price/LOB/Bar   │  │
│  └──────┬───────┘   └──────┬───────┘   └────────┬─────────┘  │
│         │                  │                     │            │
│  ┌──────┴──────────────────┴─────────────────────┴────────┐  │
│  │              Strategy Cards System                      │  │
│  │    Card Registry → Deck → Pre/Post/Reward Nodes        │  │
│  └────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Backtest Engine                            │  │
│  │    Sequential + Parallel (ThreadPoolExecutor)           │  │
│  │    Metrics: Sharpe, Max Drawdown, Win Rate              │  │
│  │    LLM Response Cache (thread-local LRU)                │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
         │              │                    │
         ▼              ▼                    ▼
┌──────────────┐ ┌────────────┐    ┌──────────────────┐
│  market_cgan │ │ market_pinn│    │  web/ (FastAPI)  │
│  CGAN Gen    │ │ PINN+BS PDE│    │  + React SPA     │
│  Bar + LOB   │ │ Price Paths│    │  + SQLite DB     │
└──────────────┘ └────────────┘    └──────────────────┘
```

## Module Architecture

### 1. CLI Layer (`cli/`)

Typer-based command dispatcher. Each subcommand maps to a domain module:

| Command | Entry Point | Module |
|---------|------------|--------|
| `trading-agent backtest run` | `backtest_cmd.py` | `trading_agent.backtest` |
| `trading-agent backtest compare` | `backtest_cmd.py` | `trading_agent.backtest.parallel` |
| `trading-agent cards list` | `cards_cmd.py` | `trading_agent.cards` |
| `trading-agent cgan train` | `cgan_cmd.py` | `market_cgan.training` |
| `trading-agent pinn train` | `pinn_cmd.py` | `market_pinn.training` |
| `trading-agent serve` | `serve_cmd.py` | `web/main.py` |

Architecture decision: **thin CLI layer.** Commands deserialize config, instantiate core objects, call library functions. No business logic lives in CLI files.

### 2. Core Library (`trading_agent/`)

#### 2.1 LangGraph State Machine (`core/`)

The agent runs as a compiled `StateGraph` with typed state:

**State** (`core/state.py`):
```
AgentState:
  cash, shares, price, price_history, portfolio_values
  peak_value, action, trade_cost, reward, total_reward
  step, done
  lob_bid, lob_ask, lob_spread, lob_mid (NotRequired - LOB mode)
  fill_price, order_filled (NotRequired - LOB mode)
```

**Graph flow (default, price-only mode):**
```
decide_and_trade → calculate_reward → decide_and_trade (loop)
```

**Graph flow (LOB/bar market data mode):**
```
update_market → decide_and_trade → calculate_reward → update_market (loop)
```

Termination is conditional: `step >= max_steps` or `done == True` routes to `END`.

**MarketAdapter Protocol** (`core/graph.py`):

A `@runtime_checkable` Protocol with three implementations — this is the key extensibility interface:

| Adapter | `has_market_data_node` | Trade Execution | Market Context |
|---------|----------------------|----------------|----------------|
| `PriceMarketAdapter` | `False` | Simple `execute_trade` | cash, shares, price, trend |
| `LOBMarketAdapter` | `True` | `execute_lob_trade` (bid/ask spread matching) | LOB bid, ask, spread, mid, fill status |
| `BarMarketAdapter` | `True` | Simple `execute_trade` | O, H, L, C, V, VWAP |

Custom adapters implementing the Protocol can be passed directly to `build_agent_graph`, enabling arbitrary new data sources without modifying graph construction.

**LLM Decision Flow:**

1. System prompt assembled from reward function description + deck prompt modifiers (if any)
2. Human prompt assembled via `MarketAdapter.format_prompt(state)`, with trailing trend calculation
3. `_decide_with_cache` checks thread-local LRU cache first
4. LLM called with `with_structured_output(TradeDecision)` — Pydantic schema via LCEL
5. Fallback: text parsing regex on `with_structured_output` failure
6. Result cached by market state fingerprint

#### 2.2 Strategy Cards System (`cards/` + `nodes/`)

MTG-inspired composable strategy system:

```
Card (JSON definition)
  → CardRegistry (loads all JSON on startup)
    → Deck (validated composition, mana budget)
      → NODE_REGISTRY (node implementations by name)
        → Pre/Post/Reward nodes injected into LangGraph
```

**Card structure** (JSON):
```json
{
  "id": "momentum-rider",
  "name": "Momentum Rider",
  "rarity": "rare",
  "mana": 3,
  "nodes": ["momentum_analyzer"],
  "reward": "aggressive",
  "prompt_modifier": "Prioritize momentum over fundamentals.",
  "stats": { "analysis": 8, "execution": 6, "risk": 3 }
}
```

**Graph transformation with deck:**

```
Default:    update_market → decide_and_trade → calculate_reward
Deck mode:  update_market → [pre_trade nodes] → decide_and_trade
            → [post_trade nodes] → [reward node] → loop
```

Each card injects nodes at a lifecycle position:
- **Pre-trade** — analyze market state before LLM decision (momentum, reversion, volatility)
- **Post-trade** — filter, modify, or veto LLM's decision (stop-loss, position sizing, trend filter)
- **Reward** — replace the default reward function entirely

Nodes are registered via `@register_node` decorator and collected into `NODE_REGISTRY` dict.

#### 2.3 LLM Gateway (`models/`)

**`KiloGateway`** — thin wrapper over LiteLLM's `completion()` supporting 100+ providers through unified interface. Accepts `model` string in `provider/model` format (e.g., `openai/gpt-4o`, `anthropic/claude-sonnet`).

**`MockLLM` / `MockGateway`** — returns `TradeDecision(action="HOLD", quantity=0, reason="mock")` for testing without API calls.

#### 2.4 Market Data Sources (`market/`)

| Source | Type | Implementation | Notes |
|--------|------|----------------|-------|
| Polygon.io | Real OHLCV | `PolygonDataSource` → `fetch_aggregates()` | Requires `POLYGON_API_KEY` |
| Random Walk | Synthetic | `RandomWalkMarket` | Drift + gaussian noise |
| Historical | Replay | `HistoricalMarket` | Replays price series |
| CGAN Source | Synthetic | `CGANMarketSource` | Duck-typed, bridges CGAN output |

**Design decision:** `CGANMarketSource` avoids cross-package imports — it uses duck typing rather than importing from `market_cgan`. This keeps the simulator as a standalone package that can be developed, tested, and distributed separately from the core agent.

#### 2.5 Backtest Engine (`backtest/`)

**Sequential mode:** `backtest_agent` runs a single backtest with a compiled graph, returning `BacktestResult` (total_return, sharpe, max_drawdown, win_rate, trades, portfolio_values, steps).

**Parallel mode:** `parallel_backtest` uses `ThreadPoolExecutor` to run N backtests concurrently (identical logic, different model/deck/reward configurations). Each thread maintains its own LLM response cache. Typical speedup: ~4x with 4 workers on I/O-bound LLM calls.

**LLM Response Cache:** Thread-local LRU cache keyed by market state fingerprint. On flat/oscillating markets, ~30-50% of LLM calls are eliminated via cache hits. Cache size configurable via `LLM_CACHE_SIZE`.

### 3. Market Simulators

#### 3.1 CGAN (`market_cgan/`)

Conditional GAN for synthetic market data generation, with two operational modes:

**Bar Mode (primary):**
- Input: 64-dim noise + 6-dim features (open, high, low, close, volume, vwap)
- Generator: MLP (256→256→128) with 6 output heads
- Training data: Polygon.io OHLCV aggregates or synthetic
- Simulation: `BarWorldAgent` steps through generated bars via `BarExchange`
- Physics-informed loss: optional PDE regularization on generated bars

**LOB Mode (legacy):**
- Input: 64-dim noise + 42-dim LOB snapshot features
- Generator: MLP with 4 output heads (action_type, side, price_offset, quantity)
- Simulation: `LOBExchange` matching engine with full order book
- Data sources: LOBSTER NASDAQ L3 data, Polygon.io NBBO quotes

Training loop: `train_cgan_bar` (bar mode) and `train_cgan` (LOB mode) share a common `train_gan` loop managing generator/discriminator adversarial training.

#### 3.2 PINN (`market_pinn/`)

Physics-Informed Neural Network for synthetic price path generation:

**Network architecture:**
```
Input: (t, S) → Linear → Tanh → [ResidualBlock × N] → Linear → V(t,S)
```

**Loss function:**
```
L = w_data × MSE(V(t,S), market_price) + w_pde × MSE(PDE_residual, 0)
```

**PDE constraint:** Black-Scholes equation:
```
∂V/∂t + r·S·∂V/∂S + ½·σ²·S²·∂²V/∂S² − r·V = 0
```

The PDE residual is computed via PyTorch autograd (`torch.autograd.grad`) on the network's outputs with respect to its inputs.

**Usage flow:** Train on real market data → generate synthetic price paths → feed into backtest engine as alternative data source.

### 4. Web Dashboard (`web/`)

#### Backend (FastAPI)

- **Routers:** `api.py` (generic endpoints), `backtests.py` (CRUD + compare), `models.py` (model comparison), `dashboard.py` (overview), `cgan.py`, `pinn.py`
- **Database:** SQLAlchemy + aiosqlite, file-based at `data/trading.db`
- **Background tasks:** `tasks.py` for long-running operations
- **Templates:** Jinja2 for legacy routes; primary API is JSON

**API endpoints:**
```
/api/dashboard          → recent runs summary
/api/backtests          → list all runs
/api/backtests/{id}     → run detail with step-level data
/api/backtests/new      → start new backtest
/api/backtests/compare  → parallel model comparison
/api/models/compare     → aggregated model performance
/api/cards              → list strategy cards
/api/cards/{id}         → card detail
/api/decks              → list/manage decks
/api/pinn/models        → trained PINN models
```

#### Frontend (React SPA)

**Stack:** Vite + TypeScript + Tailwind CSS v4 + React Router

**Pages:**
- `/app` — Dashboard with KPI stat cards, sparklines, recent runs table
- `/app/backtests` — Runs table (sortable, color-coded gains/losses)
- `/app/backtests/new` — New backtest form with symbol chips, deck selector
- `/app/backtests/:runId` — Run detail with portfolio chart (lightweight-charts) + trade log
- `/app/models/compare` — Model comparison bar chart
- `/app/cards` — Card collection (table/grid toggle, rarity filter)
- `/app/decks` — Deck builder with mana budget validation
- `/app/pinn/train` — PINN training form
- `/app/pinn/generate` — PINN generation form

**Shared components:** CmdBar, TerminalTable, Sparkline, StatCard, StatusBadge, StrategyCard, Btn, Modal, CmdInput, TickerBar

**Design system:** Green-on-black terminal-phosphor aesthetic. Monospace typography (JetBrains Mono via Tailwind config). Color-coded values (green = gain, red = loss). Dark slate background with subtle grid pattern.

### 5. Configuration (`config/`)

Pydantic `BaseSettings` loaded from `.env` file. Singleton via `@lru_cache` on `get_settings()`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | `sqlite:///data/trading.db` | Web DB path |
| `DEFAULT_MODEL` | `openai/gpt-4o-mini` | Default LLM |
| `FEE_RATE` | `0.001` | Trading fee per tx |
| `RISK_LAMBDA` | `0.1` | Risk penalty weight |
| `LLM_CACHE_SIZE` | `1000` | LRU cache capacity |
| `LLM_CACHE_ENABLED` | `true` | Toggle dedup |

## Data Flow

### Backtest Run Flow

```
1. CLI receives parameters (symbol, steps, deck, model, reward)
2. Settings loaded from .env
3. LLM instantiated via KiloGateway (or MockLLM in test)
4. MarketAdapter created based on mode (price/LOB/bar)
5. Graph compiled via build_agent_graph (with or without deck)
6. Initial AgentState created: cash=10000, shares=0
7. For each step:
   a. MarketAdapter.update_state() → new price/lob/bar data
   b. Pre-trade nodes analyze state (if deck)
   c. LLM decides BUY/SELL/HOLD (with cache lookup)
   d. MarketAdapter.execute_trade() → update cash/shares
   e. Post-trade nodes filter/modify trade (if deck)
   f. calculate_reward → profit + risk_penalty + trade_cost
   g. Should continue? step < max_steps → loop, else END
8. Compute strategy metrics: Sharpe, max drawdown, win rate
```

### Web Dashboard Flow

```
React SPA → fetch(/api/backtests) → FastAPI → SQLAlchemy → SQLite
                                         ↓
                                  trading_agent (backtest)
```

The web API calls into the same core library functions as the CLI, serializing results to JSON.

### Training Flow (CGAN/PINN)

```
CGAN:  Real Data (Polygon/LOBSTER) → Feature Extraction → Train GAN → Generate
PINN:  Real Prices (Polygon) → MarketDataset → Train PINN → Generate Paths
```

## Deployment Architecture

### Docker

Multi-stage build:
1. **Stage 1 (Node 20):** Build React SPA → `web/static/dist/`
2. **Stage 2 (Python 3.12):** Install dependencies + copy built frontend + start uvicorn

**docker-compose.yml** — single service, exposes port 8000.

## Extensibility Points

| Point | Interface | What You Can Add |
|-------|-----------|-----------------|
| Market data | `MarketAdapter` Protocol | New data sources (WebSocket, custom API, CSV) |
| Strategy | `@register_node` + card JSON | New trading strategies with custom logic |
| LLM provider | LiteLLM model string | Any LiteLLM-supported provider |
| Reward function | Function signature `(old, new, peak, cost, lambda) → float` | New reward shapes |
| Market simulation | Standalone package (like `market_cgan/`) | New synthetic market models |
| Backtest comparison | `parallel_backtest` workers | Any combination of model/deck/reward |
| Web UI | React page component + FastAPI router | New dashboard pages |

## Technology Stack

| Layer | Technology |
|-------|-----------|
| LLM | LiteLLM (100+ providers) |
| Agent orchestration | LangGraph (`StateGraph`, compiled) |
| LLM output parsing | Pydantic (`with_structured_output`) + regex fallback |
| Configuration | Pydantic Settings (`.env`) |
| CLI | Typer |
| Backtest parallelism | `ThreadPoolExecutor` |
| GAN framework | PyTorch |
| PINN framework | PyTorch + autograd |
| Web backend | FastAPI |
| Database | SQLAlchemy + SQLite (aiosqlite) |
| Frontend | Vite + TypeScript + React + Tailwind CSS v4 |
| Charts | lightweight-charts |
| Containerization | Docker multi-stage |
| Testing | pytest (209 tests) |