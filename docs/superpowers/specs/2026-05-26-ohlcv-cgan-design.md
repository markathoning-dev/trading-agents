# OHLCV Bar-to-Bar CGAN Design

## Motivation

Polygon.io Starter tier ($29/mo) provides unlimited minute-aggregate bars (OHLCV) but does NOT authorize NBBO quotes (requires Advanced at $199/mo). This spec pivots the CGAN-based market simulator from LOB data to OHLCV minute bars while preserving the adversarial training framework, World Agent architecture, and RL agent integration.

## Architecture

```
Polygon /v2/aggs/ticker/{ticker}/range/1/minute
    → BarDataset (current_bar → next_bar)
    → Generator: noise + current_bar (6-dim) → next_bar (6-dim)
    → Discriminator: (next_bar + current_bar) → real/fake
    → BarExchange: appends generated bars
    → BarWorldAgent: step() generates next bar, feeds state to RL agents
```

### Data Layer (`market_cgan/data/bar.py`)

- **`Bar` dataclass**: `timestamp, open, high, low, close, volume, vwap`
- **`BarFeatureExtractor`**: normalizes raw OHLCV into a 6-dim feature vector:
  - `[open/ref, high/ref, low/ref, close/ref, log(volume+1)/log_norm, vwap/ref]`
  - `ref` = reference price (e.g., first bar's close), `log_norm` = log(max_volume+1)
- **`BarDataset`**: consecutive bar pairs `(current_bar_features, next_bar_features)`
  - `feat_dim = 6`, `target_dim = 6`
  - Target values use the SAME normalization as features
- **`PolygonDataSource.fetch_aggregates(ticker, start_date, end_date, timespan='minute')`**:
  - Calls `client.get_aggregate_bars(ticker, timespan, start_date, end_date)`
  - Returns `list[Bar]`
  - Handles pagination via `next_url`

### Models

**`market_cgan/models/bar_generator.py`** — `BarGenerator(nn.Module)`:
- Input: `noise (noise_dim) + features (6)` → concat
- Hidden: 3-layer MLP with LayerNorm + LeakyReLU (256→256→128)
- Output: 6 continuous values via a constrained output head:
  - `open`, `close`: `ref_price * (1 + tanh(·) * max_move)`
  - `high`: `max(open, close) + softplus(·)`
  - `low`: `min(open, close) - softplus(·)`
  - `volume`: `softplus(·)`
  - `vwap`: `ref_price * (1 + tanh(·) * max_move)`
- The output head encodes OHLCV constraints directly into the architecture (no post-hoc clipping).

**`market_cgan/models/bar_discriminator.py`** — `BarDiscriminator(nn.Module)`:
- Input: `bar (6-dim) + features (6-dim)` = 12-dim
- Hidden: 3-layer MLP with LeakyReLU + Dropout (128→128→64)
- Output: single logit → sigmoid

### Training

**Trainer** (`market_cgan/training/trainer.py`):
- Modified `train_cgan_bar()` that uses `BarGenerator`, `BarDiscriminator`
- Same adversarial loop: D loss (real vs fake logits + optional GP) → G loss (fool D + optional physics)
- Dataset yields `(features, target_bars)` instead of `(features, actions)`
- Generator output is the full 6-dim bar tensor instead of action components

**Physics loss** (`market_cgan/training/bar_physics_loss.py`):
1. **HL constraint**: `relu(max(O,C) - H) + relu(L - min(O,C))` — penalizes invalid high/low
2. **Volume positivity**: `relu(-V)` — small penalty for negative volume
3. **Return distribution matching**: KL or MMD between real and fake log-return distributions
4. **Volatility clustering**: autocorrelation of absolute returns in batches

### Simulation Layer

**`market_cgan/simulation/bar_exchange.py`** — `BarExchange`:
- Holds a list of `Bar` objects (history)
- `append_bar(bar)`: adds generated bar to history
- `get_state()`: returns latest bar + recent N-bar window as a feature dict
- `reset()`: clears history

**`market_cgan/simulation/bar_world_agent.py`** — `BarWorldAgent`:
- Takes `BarGenerator`, `BarExchange`, noise_dim
- `step()`: extracts current bar features → generates next bar → appends to exchange → returns new bar
- `reset()`: clears exchange, generator stays as-is

### Integration with RL Agents

- RL agents (Risk-Averse / Risk-Taker) see `BarExchange.get_state()` as market context
- For agent `decide_and_trade`: state includes latest bar OHLCV + volume + VWAP
- BUY/SELL fills at bar's close price (if bar is complete) or VWAP (intra-bar)
- Backtest engine: steps through generated bar sequence, agent decides each bar

### CLI Changes (`cli/cgan_cmd.py`)

- `train` command: add `--bar-mode` flag; when set, uses `BarDataset`, `BarGenerator`, `BarDiscriminator` with 6-dim features and targets
- `generate` command: `--bar-mode` outputs CSV with columns: `step,open,high,low,close,volume,vwap`
- `simulate` command: `--bar-mode` uses `BarExchange` + `BarWorldAgent` instead of LOB equivalents

### Files Changed Summary

| File | Change |
|------|--------|
| `market_cgan/data/polygon.py` | Add `fetch_aggregates()` |
| `market_cgan/data/bar.py` | **New** — Bar, BarFeatureExtractor, BarDataset |
| `market_cgan/models/bar_generator.py` | **New** — BarGenerator with constrained OHLCV output |
| `market_cgan/models/bar_discriminator.py` | **New** — BarDiscriminator |
| `market_cgan/training/bar_physics_loss.py` | **New** — OHLCV constraint losses |
| `market_cgan/training/trainer.py` | Add `train_cgan_bar()` parallel path |
| `market_cgan/simulation/bar_exchange.py` | **New** — BarExchange |
| `market_cgan/simulation/bar_world_agent.py` | **New** — BarWorldAgent |
| `cli/cgan_cmd.py` | Add `--bar-mode` to train/generate/simulate |
| `scripts/train_spus_q1_2026.py` | Update to use `BarDataset` + `fetch_aggregates` |

### Out of Scope (for this design)

- WebSocket streaming (requires Advanced $199/mo tier)
- LOB data generation (requires Advanced tier; existing LOB code preserved for future use)
- Full LOB→OHLCV bridge (generated LOB → bar conversion)
