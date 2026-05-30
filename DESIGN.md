# Trading Agent — Design

## Design Philosophy

The trading agent is designed around three core principles:

1. **LLM as the decision core** — Market analysis and trade decisions are delegated to an LLM, not hardcoded rules. The reward function provides the signal; the LLM discovers the strategy.
2. **Composable, not monolithic** — Strategies are assembled from cards (discrete units of behavior), not written as monolithic scripts. Cards encode both the *what* (a trade analysis node) and the *reward signal* the LLM optimizes for.
3. **Simulation realism** — Market simulators (CGAN, PINN) generate synthetic data that respects statistical or physical constraints, rather than naive random walks, to produce more meaningful backtest results.

---

## Key Design Decisions

### 1. LangGraph State Machine Over Imperative Loop

**Decision:** The agent runs as a compiled LangGraph `StateGraph`, not a Python `for` loop calling functions sequentially.

**Rationale:**
- LangGraph provides first-class support for conditional branching, which the deck system exploits to inject custom nodes at lifecycle positions
- Compiling the graph enables future optimizations (parallel node execution, checkpointing, streaming)
- The graph structure makes agent behavior inspectable and testable at the node level
- The `StateGraph` pattern separates state mutation from orchestration logic

**Trade-off:** ~2x initialization overhead vs a plain loop. Accepted because backtest runs are long-lived (50-500 steps) amortizing the cost.

**Alternatives considered:**
- **Plain loop** — Rejected because node injection and conditional branching would require ad-hoc control flow logic
- **LangChain AgentExecutor** — Rejected. Too opinionated about tool-use patterns. We needed custom state transitions and the ability to inject arbitrary pre/post-processing nodes

### 2. MarketAdapter Protocol Over Inheritance

**Decision:** Market data integration uses a structural Protocol (duck typing), not an abstract base class or configuration enum.

**Rationale:**
- A Protocol allows new adapters to be defined outside the package — downstream code can implement `MarketAdapter` without importing the core agent
- `@runtime_checkable` enables explicit type checking while preserving duck-typed flexibility
- Each adapter self-describes whether it needs a market data node (`has_market_data_node`), allowing the graph builder to conditionally include one
- The three built-in adapters (Price, LOB, Bar) demonstrate the range: from zero-overhead (price-only) to stateful simulation (LOB with world_agent)

**Trade-off:** Protocol methods use `**state` unpacking, which provides no static guarantees about which fields are read or written. Accepted because the state TypedDict is small (22 fields) and well-documented.

### 3. Dual Reward Function Design

**Decision:** Two canonical reward functions with explicit formulas exposed to the LLM in the system prompt.

**`multicomponent_reward` (risk-averse):**
```
reward = profit − λ_drawdown × drawdown − trade_cost
```
- Drawdown penalty (from peak portfolio value) discourages large losses
- Safe trades (HOLD with small price changes) yield small positive/negative rewards
- **Best for:** conservative strategies, sideways markets

**`aggressive_reward` (risk-taker):**
```
reward = profit + λ_volatility × |profit| − trade_cost
```
- Volatility bonus amplifies gains AND losses — the LLM is incentivized to make large directional trades
- No drawdown penalty — peak portfolio value is irrelevant
- **Best for:** trending markets, momentum strategies

**Rationale for exposing formulas to the LLM:**
- The LLM needs to understand *how* it's being evaluated to optimize for the right signal
- The reward formula is included in the system prompt as a concrete optimization target
- Cards can override the reward function entirely via custom reward nodes

**Trade-off:** LLMs may try to "game" the understood reward function (e.g., trading tiny quantities repeatedly to accumulate profit). The trade_cost term and the `@register_node` decorator for custom reward logic mitigate this.

### 4. Strategy Cards as Discrete Behavior Units

**Decision:** Trading strategies are assembled from cards — each card defines node implementations, a reward type, and prompt modifiers — combined into decks with mana budgets.

**Why MTG-inspired mechanics?**
- **Mana budget** (default 10) — prevents strategy bloat. A deck can't include every good card, forcing trade-off decisions
- **Rarity tiers** (Common/Epic/Legendary) — encodes complexity and power expectations. A Legendary card (5 mana) should feel powerful and unique
- **Stat bars** (analysis/execution/risk) — gives the LLM visual signal about a card's strengths during deck building
- **Flavor text + prompt modifiers** — the card's strategic intent is communicated to the LLM via injected system prompt text, not hardcoded logic

**Why this works:**
- Strategy discovery becomes a combinatorial optimization problem (pick N cards within a mana budget)
- Cards decouple *what* a strategy does (node logic) from *how* it's evaluated (reward function) from *how* it's described to the LLM (prompt modifier)
- The card system makes strategies testable independently: verify that "Stop-Loss Sentinel" actually prevents trades below a threshold, then compose it freely with other cards

**Example deck (Aggro):**
- **Volatility Vampire** (4 mana) — volatility analysis + custom volatility reward
- **Momentum Rider** (3 mana) — momentum pre-trade analysis + aggressive reward
- **Position Sizer** (1 mana) — position sizing post-trade
- **Stop-Loss Sentinel** (1 mana) — stop loss vetoe
- Total: 9/10 mana. Leaves room for one more card.

### 5. LLM Response Cache

**Decision:** Thread-local LRU cache deduplicating LLM calls for identical market state fingerprints.

**Rationale:**
- Market states are often repeated (flat prices, same portfolio composition between trades)
- LLM calls are the primary cost (both API latency and monetary cost) in a backtest
- Testing showed 30-50% cache hit rates on oscillating/flat markets
- Cache is thread-local to avoid locking in parallel backtesting

**State fingerprint definition:** `(cash, shares, price, price_history[last 5], reward_type)` — sufficient uniqueness for decision identity.

**Trade-off:** Cache can mask LLM nondeterminism — the same state always gets the same decision. Accepted because backtest reproducibility is more valuable than maximum realism, and the cache is configurable (`LLM_CACHE_ENABLED`).

### 6. CGAN — Dual Mode (Bar + LOB)

**Decision:** Two operational modes sharing a common training infrastructure.

**Bar mode (primary):**
- Generates OHLCV bars — the most common input format for trading algorithms
- 6-dimensional feature space (O, H, L, C, V, VWAP)
- Faster to train and more stable than LOB mode
- Compatible with Polygon.io data for supervised training

**LOB mode (legacy):**
- Generates limit order book actions (action_type, side, price_offset, quantity)
- 42-dimensional feature space from LOB snapshots
- Requires real LOB data (LOBSTER or Polygon NBBO)
- Enables more granular market simulation (spread dynamics, fill probability)

**Why maintain both?** The LOB sim creates more realistic micro-structure effects (slippage, partial fills) that affect strategy behavior differently than OHLCV-level simulation. Bar mode is for strategy development; LOB mode is for production validation.

**Physics-informed loss:** Optional PDE regularization on generated bars ensures the generated OHLCV data respects basic financial constraints (close ≤ high, close ≥ low, volume ≥ 0, etc.) and optionally a simplified Black-Scholes equation.

### 7. PINN — Physics-Constrained Generation

**Decision:** Generate synthetic price paths using a Physics-Informed Neural Network constrained by the Black-Scholes PDE, rather than a purely data-driven generative model.

**Why PINN over pure GAN for price paths?**
- The Black-Scholes equation encodes a known physical constraint on price evolution
- The PDE residual loss term penalizes unphysical price behavior during training
- This produces more realistic price paths than unconstrained models, especially for out-of-sample market conditions
- The weight parameter `w_pde` controls the strength of the physics constraint vs data fit — tunable per use case

**Architecture choice:** Deep residual network with skip connections, rather than a simple MLP. Residual connections prevent vanishing gradients in deeper networks, which is important when training with a PDE constraint that requires second-order derivatives via autograd.

**Trade-off:** The Black-Scholes equation assumes constant volatility and no jumps — real markets violate both. The data-fitting term (`w_data`) compensates, but the model will produce less realistic paths during extreme market events.

### 8. Parallel Backtesting Architecture

**Decision:** `ThreadPoolExecutor` with thread-local LLM caches.

**Why ThreadPoolExecutor over ProcessPoolExecutor?**
- LLM calls are I/O-bound (HTTP requests), not CPU-bound
- Thread pools have lower overhead than process pools (no serialization, shared memory)
- Each thread needs its own LLM session cache, which thread-local storage provides naturally
- Python's GIL is not a bottleneck for I/O-wait operations

**Scalability limit:** API rate limiting on the LLM provider. With 4+ workers all calling the same LLM provider concurrently, rate limit errors become likely. No built-in rate limiting — the user must configure their provider accordingly.

### 9. Web Dashboard: Dual Rendering (React SPA + Jinja2)

**Decision:** React SPA as primary frontend, Jinja2 templates retained for backward compatibility.

**Rationale:**
- The React SPA provides rich interactivity (sortable tables, interactive charts, terminal-theme design system)
- Jinja2 templates serve as a fallback / fast-render path for simple views
- The FastAPI JSON API serves both — endpoints return JSON, consumed by both React and Jinja2 if needed

**Design language:** Terminal-phosphor aesthetic (green on black, monospace, glow effects) was chosen to:
- Signal a data-dense, professional-grade tool (not a toy)
- Create visual distinction from mainstream trading dashboards
- Match the CLI-first character of the project

**Deployment:** Frontend is pre-built as static files (Vite `build`) served by FastAPI at `/app/*`. Zero infrastructure required — no separate Node server, CDN, or Nginx proxy.

### 10. Configuration Design

**Decision:** `BaseSettings` from `pydantic-settings` loaded from `.env`, cached via `@lru_cache`.

**Why not YAML or TOML?**
- `.env` files are the lowest-friction configuration format for command-line tools
- Docker and CI systems all support `.env` natively
- Pydantic provides type coercion and validation at load time
- `lru_cache` on `get_settings()` ensures O(1) access with zero overhead after first load

**Why a module-level `settings` singleton AND a cached function?** The function API (`get_settings()`) supports testing with environment variable overrides. The singleton (`settings`) provides backward compatibility for code that was written before the function API existed.

---

## Error Handling Strategy

| Failure Mode | Handling |
|-------------|----------|
| LLM structured output parse failure | Regex fallback → default HOLD |
| LLM API timeout | Exception → state continues without trade |
| Polygon.io data unavailable | Falls back to synthetic data (RandomWalkMarket) |
| Invalid deck mana budget | Validation error with specific message |
| Cache full (LRU eviction) | Oldest entry evicted, cache continues |
| Web dashboard DB lock | SQLite WAL mode via aiosqlite |

---

## Testing Strategy

209 tests across 8 test modules, organized by domain:

| Test Module | Tests | Focus |
|------------|-------|-------|
| `test_core/` | State, nodes, graph, cache, reward, schemas, cache | Core state machine logic |
| `test_backtest/` | Engine, metrics, parallel | Backtest correctness |
| `test_cards/` | Registry, deck validation, nodes | Card system integrity |
| `test_cgan/` | Bar/LOB generator + discriminator, trainer, exchange, integration | Simulator correctness |
| `test_pinn/` | Model, Black-Scholes residual, trainer | PINN physics constraint |
| `test_market/` | Simulators | Market data source correctness |
| `test_models/` | Gateway, mock | LLM abstraction |
| `test_web/` | Routes | API endpoint coverage |

Key testing patterns:
- **MockLLM** replaces real LLM calls in core tests — fast, deterministic, zero API cost
- **Integration tests** for CGAN cover the full train → generate → simulate pipeline with small toy datasets
- **Graph tests** verify conditional branching (loop vs END) at step boundaries
- **Reward tests** verify edge cases: zero change, negative profit, exactly at peak
- **Cache tests** verify LRU eviction order and thread isolation
- **LOB trade tests** verify limit order fill/no-fill logic against bid/ask prices

---

## Performance Characteristics

| Operation | Latency | Bottleneck |
|-----------|---------|-----------|
| Single backtest (50 steps) | ~30-60s | LLM API calls |
| Parallel compare (4 models, 50 steps) | ~30-90s | ThreadPoolExecutor + LLM rate limits |
| CGAN training (100 epochs) | ~5-30 min | GPU (optional) / CPU |
| PINN training (1000 epochs) | ~2-10 min | CPU / GPU |
| Web page load | <500ms | SQLite query latency |

**Cache efficiency:** 30-50% hit rate on oscillating/flat markets; <10% on strongly trending markets.

---

## Future Design Considerations

- **WebSocket streaming** — Real-time market data via Polygon.io Advanced WebSocket or custom streaming adapters
- **Multi-asset portfolio** — Extend `AgentState` with multiple positions; requires cross-asset market adapters
- **Reinforcement learning baseline** — Compare LLM-driven decisions against trained RL policies on the same backtest engine
- **Distributed backtesting** — Move from thread-level concurrency to Ray or Dask for multi-node backtesting
- **CGAN production serving** — Export trained generator as ONNX or TorchScript for low-latency inference