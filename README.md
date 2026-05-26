# Trading Agent

LLM-powered trading agent with LangGraph orchestration, backtesting, PINN and CGAN market simulation, and a web dashboard.

<p align="center">
  <img src="screenshots/cli.png" alt="trading-agent CLI" width="720">
</p>

```
                    ┌──────────────┐
                    │   CLI (typer) │
                    └──┬───┬───┬───┘
                       │   │   │
         ┌─────────────┘   │   └──────────────┐
         ▼                 ▼                  ▼
   ┌──────────┐    ┌──────────────┐    ┌──────────┐
   │ Backtest │    │  PINN/CGAN   │    │  Web     │
   │ Commands │    │  Commands    │    │ Dashboard│
   └────┬─────┘    └──────┬───────┘    └────┬─────┘
        │                 │                 │
        ▼                 ▼                 ▼
   ┌──────────────────────────────────────────────┐
   │              trading_agent/ (Core)           │
   │  ┌────────┐  ┌──────────┐  ┌────────────┐   │
   │  │  Core  │  │  Models  │  │   Market   │   │
   │  │ State, │  │ LLM Gate │  │   Sources  │   │
   │  │ Graph, │  │  ways    │  │ Sim, Yahoo │   │
   │  │ Nodes, │  │          │  │ LOB Source │   │
   │  │ Reward │  │          │  │            │   │
   │  └────────┘  └──────────┘  └────────────┘   │
   │  ┌────────────────────────────┐              │
   │  │     Backtest Engine        │              │
   │  │     + Metrics              │              │
   │  └────────────────────────────┘              │
   │  ┌────────────────────────────┐              │
   │  │   Config (pydantic)        │              │
   │  └────────────────────────────┘              │
   └──────────────────────────────────────────────┘
        │                 │                  │
        ▼                 ▼                  ▼
   ┌──────────┐    ┌──────────────┐    ┌──────────┐
   │market_cgan│   │ market_pinn  │    │  web/    │
   │ CGAN Gen  │   │  PINN Model  │    │ FastAPI  │
   │ LOB Sim   │   │  BS PDE      │    │ DB + UI  │
   └──────────┘    └──────────────┘    └──────────┘
```

## Features

- **LLM Agent** — LangGraph agent that decides BUY/SELL/HOLD using any LLM provider via LiteLLM
- **Multi-Provider LLM** — Kilo API, OpenAI, Anthropic, or any LiteLLM-supported provider
- **Reward Functions** — Risk-averse (`multicomponent_reward` with drawdown penalty) and risk-taker (`aggressive_reward` with volatility bonus)
- **LOB-Aware Trading** — Extended agent that interacts with a full limit order book (market orders, limit orders, fills, spreads)
- **Backtesting** — Historical (yfinance) and simulated data, with Sharpe ratio and max drawdown metrics
- **CGAN Market Simulator** — Conditional GAN that learns limit order book dynamics from real data and generates synthetic LOB sequences
- **PINN Market Simulator** — Physics-Informed Neural Network constrained by Black-Scholes PDE for synthetic price path generation
- **World Agent** — Autonomous market-making agent driven by the trained CGAN generator
- **Physics-Informed Training** — CGAN generator can be regularized with spread, action distribution, side balance, and quantity constraints
- **Web Dashboard** — FastAPI + Jinja2/HTMX UI for managing backtests and PINN training
- **Docker Deployment** — Dockerfile + docker-compose for one-command startup

## Quick Start

### Prerequisites

- Python 3.12+
- pip

### Installation

```bash
git clone <repo-url> && cd trading-agent
pip install -e ".[web,pinn,dev]"
```

Optional data sources:

```bash
pip install polygon>=1.2       # Polygon.io NBBO quotes
pip install yfinance           # already included via langchain-community
```

### Configuration

Copy and edit `.env`:

```bash
cp .env.example .env
```

Key settings:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Kilo API or OpenAI key for LLM calls |
| `KILO_API_BASE` | — | Custom API base URL for Kilo gateway |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `POLYGON_API_KEY` | — | Polygon.io API key for LOB data |
| `DATABASE_URL` | `sqlite:///data/trading.db` | Web dashboard database |
| `DEFAULT_MODEL` | `openai/gpt-4o-mini` | Default LLM for agent decisions |
| `FEE_RATE` | `0.001` | Trading fee rate per transaction |
| `RISK_LAMBDA` | `0.1` | Default risk penalty weight |
| `LOG_LEVEL` | `INFO` | Logging level |

## CLI Reference

```bash
trading-agent --help
```

### Backtest Commands

```bash
# Run a single backtest
trading-agent backtest run --symbol AAPL --steps 50

# Compare multiple models side-by-side
trading-agent backtest compare --symbol AAPL --models "gpt-4o-mini,claude-sonnet"

# Compare risk-averse vs risk-taker on the same data
trading-agent backtest compare-risk --symbol AAPL --steps 3600
```

### CGAN Commands

```bash
# Train on synthetic data
trading-agent cgan train --epochs 100 --data-source synthetic

# Train on Polygon.io NBBO data (Starter plan sufficient)
export POLYGON_API_KEY=poly_...
trading-agent cgan train --data-source polygon --polygon-ticker AAPL \
  --polygon-dates 2025-01-10,2025-01-13 --epochs 50 --physics-weight 0.01

# Generate action sequences from trained model
trading-agent cgan generate --model models/cgan/generator.pt --steps 252

# Run interactive LOB simulation
trading-agent cgan simulate --model models/cgan/generator.pt --steps 100
```

### PINN Commands

```bash
# Train PINN
trading-agent pinn train --symbol AAPL --epochs 1000

# Generate synthetic price paths
trading-agent pinn generate --model-id 1 --paths 100 --steps 252
```

### Web Dashboard

```bash
trading-agent serve
```

Then open `http://localhost:8000`.

## Architecture

### Trading Agent Core (`trading_agent/`)

The agent is built as a **LangGraph** with two nodes:

```
┌────────────────┐
│  Market        │  price → (from MarketDataSource)
└───────┬────────┘
        │
        ▼
┌──────────────────────────────────┐
│  decide_and_trade                │  LLM decides BUY/SELL/HOLD
│  - structured output via LCEL    │  Trade is executed, state updated
│  - text parsing fallback         │
└───────┬──────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  calculate_reward                │  portfolio_value change
│  - multicomponent_reward         │  risk penalties or volatility bonus
│  - aggressive_reward             │
└───────┬──────────────────────────┘
        │
        ▼
   (loop or END)
```

**Key files:**

| File | Purpose |
|---|---|
| `core/state.py` | `AgentState` TypedDict — cash, shares, price, history, portfolio values |
| `core/schemas.py` | `TradeDecision`, `LimitOrder`, `MarketOrder` — Pydantic LLM output schemas |
| `core/nodes.py` | `fetch_price`, `execute_trade`, `execute_lob_trade`, `calculate_reward` |
| `core/graph.py` | `build_graph` (standard agent), `build_lob_graph` (LOB-aware agent) |
| `core/reward.py` | `multicomponent_reward` (risk-averse), `aggressive_reward` (risk-taker) |
| `models/gateway.py` | `KiloGateway` — LiteLLM abstraction supporting 100+ providers |
| `models/mock.py` | `MockLLM` / `MockGateway` — always returns HOLD for testing |
| `market/simulators.py` | `RandomWalkMarket`, `HistoricalMarket` |
| `market/yahoo.py` | `YahooMarket` — fetches yfinance data |
| `market/lob_source.py` | `CGANMarketSource` — bridges CGAN simulation into agent |
| `backtest/engine.py` | `backtest_agent` — main backtesting loop |
| `backtest/metrics.py` | `compute_sharpe`, `compute_max_drawdown` |

#### Reward Functions

**`multicomponent_reward`** (risk-averse):

```
reward = profit − λ_drawdown × max(0, peak − new_value) − trade_cost
```

**`aggressive_reward`** (risk-taker):

```
reward = profit + λ_volatility × |profit| − trade_cost
```

### CGAN Market Simulator (`market_cgan/`)

Conditional GAN that learns to generate realistic limit order book actions.

```
┌──────────┐    noise (64d)
│ Generator│    features (42d market state)
│   MLP    │    ──────────────────────────────►
│ 256-256- │    action_type (4-class softmax)
│ 128      │    side (2-class softmax)
└──────────┘    price_offset (tanh → [-1, 1])
                 quantity (sigmoid → [0, 1])
```

**Training flow:**

```
                   fake_actions
Generator ◄──────────────┐
  │                       │
  │ fake_actions          │
  ▼                       │
Discriminator ◄───────────┤ (detached for D update)
  │                       │
  ├─ real_actions ────────┤ (from dataset)
  │                       │
  └─ loss ───────────────►┘
    BCE(real=1, fake=0)

Generator loss:
  BCE(fake_logits, 1)                     ← fool the discriminator
  + physics_weight × physics_loss         ← spread, action dist, side, quantity
  + feature_matching_weight × FM_loss     ← match real feature stats (optional)
```

**Key files:**

| File | Purpose |
|---|---|
| `models/generator.py` | `Generator` — 3-layer MLP with 4 output heads |
| `models/discriminator.py` | `Discriminator` — 4-layer MLP binary classifier |
| `data/lobster.py` | `LOBSnapshot`, `LobsterDataset`, `generate_sample_lob_data` |
| `data/features.py` | `MarketFeatureExtractor` — 42-dim feature vector from LOB |
| `data/polygon.py` | `PolygonDataSource`, `PolygonDataset` — real NBBO via Polygon.io |
| `training/losses.py` | `generator_loss`, `discriminator_loss`, `gradient_penalty` (WGAN-GP) |
| `training/physics_loss.py` | `physics_informed_loss` — spread, action dist, side balance, quantity |
| `training/trainer.py` | `train_cgan` — adversarial training loop with optional physics + GP |
| `simulation/exchange.py` | `OrderBook`, `LOBExchange` — full LOB matching engine |
| `simulation/world_agent.py` | `WorldAgent` — CGAN-driven market-making agent |

#### Physics-Informed Loss (Layer A)

Four constraint terms that regularize the CGAN generator toward realistic market behavior:

1. **Spread-respecting price** — penalizes limit order prices that cross the bid-ask spread
2. **Action type distribution** — pushes batch action mix toward `[30% market, 30% limit-buy, 30% limit-sell, 10% cancel]`
3. **Side balance** — keeps buy/sell ratio near 50/50
4. **Quantity matching** — constrains mean quantity toward a target

### PINN Market Simulator (`market_pinn/`)

Physics-Informed Neural Network for synthetic price path generation.

```
Input: (t, S) ─► Linear ─► Tanh ─► [ResidualBlock × N] ─► Linear ─► V(t,S)
                    ▲              │
                    └──────────────┘
```

**Loss:** `L = w_data × MSE(V(t,S), market_price) + w_pde × MSE(PDE_residual, 0)`

PDE constraint is the **Black-Scholes equation**:

```
∂V/∂t + r·S·∂V/∂S + ½·σ²·S²·∂²V/∂S² − r·V = 0
```

**Key files:**

| File | Purpose |
|---|---|
| `models/pinn.py` | `MarketPINN` — deep residual network |
| `physics/black_scholes.py` | `bs_pde_residual` — autograd-based PDE residual |
| `training/dataset.py` | `MarketDataset` — wraps pandas price series |
| `training/losses.py` | `pinn_loss` — weighted data + PDE loss |
| `training/trainer.py` | `train_pinn` — standard PyTorch training |
| `synthesis/generator.py` | `generate_price_paths` — synthetic path generation |

### Web Dashboard (`web/`)

FastAPI + Jinja2/HTMX + SQLAlchemy:

```
/               → dashboard.html (recent backtest runs)
/backtests      → list.html (all runs)
/backtests/new  → run.html (form to start backtest)
/backtests/{id} → detail.html (run results)
/models/compare → compare.html (model performance)
/pinn/train     → PINN training form
/pinn/generate  → PINN path generation form
```

### Data Sources

| Source | Type | Cost | Data Provided |
|---|---|---|---|
| **Polygon.io Starter** | NBBO quotes | $29/mo | Historical top-of-book, 5-yr history |
| **Polygon.io Advanced** | NBBO + WebSocket | $199/mo | + real-time streaming |
| **LOBSTER** | NASDAQ L3 | €500-€2000 | Full order book depth (real) |
| **Synthetic** | Generated | Free | Random walk LOB (for testing) |
| **yfinance** | OHLCV | Free | Daily price data |

## Testing

```bash
# All tests
pytest

# Specific areas
pytest tests/test_cgan/
pytest tests/test_pinn/
pytest tests/test_core/
pytest tests/test_backtest/

# With coverage
pytest --cov=. --cov-report=term-missing
```

Currently **105 tests** across all modules.

## Docker

```bash
docker compose up
```

Opens `http://localhost:8000`. The web dashboard persists backtest results in a Docker volume.

## Project Structure

```
trading-agent/
├── cli/                    # Typer command-line interface
│   ├── main.py             # Root dispatcher (backtest, pinn, cgan, serve)
│   ├── backtest_cmd.py     # run, compare, compare-risk
│   ├── cgan_cmd.py         # train, generate, simulate
│   ├── pinn_cmd.py         # train, generate
│   └── serve_cmd.py        # serve (uvicorn)
├── trading_agent/          # Core library
│   ├── core/               # Agent state, graph, nodes, reward, schemas
│   ├── models/             # LLM gateway (Kilo, Mock)
│   ├── market/             # Data sources (random walk, historical, yahoo, LOB)
│   ├── backtest/           # Engine + metrics
│   └── config/             # Pydantic settings
├── market_cgan/            # Conditional GAN for LOB simulation
│   ├── models/             # Generator, Discriminator
│   ├── data/               # LOBSTER parser, features, Polygon.io adapter
│   ├── training/           # Losses, trainer, physics-informed loss
│   └── simulation/         # Exchange (order book), World Agent
├── market_pinn/            # Physics-Informed NN for price synthesis
│   ├── models/             # MarketPINN
│   ├── physics/            # Black-Scholes PDE
│   ├── training/           # Dataset, loss, trainer
│   └── synthesis/          # Path generator
├── web/                    # FastAPI dashboard
│   ├── db/                 # SQLAlchemy models + database
│   ├── routers/            # dashboard, backtests, models, pinn
│   ├── templates/          # Jinja2 + HTMX
│   └── static/             # CSS
├── docker/                 # Dockerfile + compose
├── tests/                  # 105 pytest tests
│   ├── test_cgan/
│   ├── test_pinn/
│   ├── test_core/
│   ├── test_backtest/
│   ├── test_market/
│   ├── test_models/
│   └── test_web/
└── scripts/                # Standalone utility scripts
```

## Development

```bash
# Install dev dependencies
pip install -e ".[web,pinn,dev]"

# Run tests
pytest -v

# Start web dashboard
trading-agent serve
```
