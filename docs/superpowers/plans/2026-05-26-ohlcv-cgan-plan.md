# OHLCV Bar-to-Bar CGAN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pivot CGAN pipeline from LOB data to Polygon.io minute aggregate bars (OHLCV), preserving adversarial training and World Agent architecture.

**Architecture:** New parallel modules (BarGenerator, BarDiscriminator, BarDataset, BarExchange, BarWorldAgent) sit alongside existing LOB modules. Generator produces 6-dim (O, H, L, C, V, VWAP) bars with architectural OHLCV validity constraints.

**Tech Stack:** Python 3.12, PyTorch, Polygon.io REST API, pytest

---

### Task 1: Add `fetch_aggregates()` to `PolygonDataSource`

**Files:**
- Modify: `market_cgan/data/polygon.py`
- Test: `tests/test_cgan/test_polygon.py`

- [ ] **Step 1: Write failing tests for fetch_aggregates**

Add to `tests/test_cgan/test_polygon.py`:

```python
from unittest.mock import MagicMock, patch
from market_cgan.data.polygon import PolygonDataSource


class MockAggBar:
    def __init__(self, o, h, l, c, v, vw, ts):
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v
        self.vwap = vw
        self.timestamp = ts


@patch("market_cgan.data.polygon.PolygonDataSource.client")
def test_fetch_aggregates(mock_client_prop):
    mock_client = MagicMock()
    mock_client.get_aggregate_bars.return_value = [
        MockAggBar(o=100.0, h=101.0, l=99.5, c=100.5, v=10000, vw=100.3, ts=1000001),
        MockAggBar(o=100.5, h=102.0, l=100.0, c=101.5, v=15000, vw=101.0, ts=1000002),
    ]
    source = PolygonDataSource(api_key="test_key")
    with patch.object(source, "client", mock_client):
        bars = source.fetch_aggregates("SPUS", "2026-03-26", "2026-03-26")
    assert len(bars) == 2
    assert bars[0].open == 100.0
    assert bars[0].high == 101.0
    assert bars[0].low == 99.5
    assert bars[0].close == 100.5
    assert bars[0].volume == 10000
    assert bars[1].vwap == 101.0


@patch("market_cgan.data.polygon.PolygonDataSource.client")
def test_fetch_aggregates_pagination(mock_client_prop):
    from market_cgan.data.bar import Bar
    mock_client = MagicMock()
    page1 = [MockAggBar(o=100.0, h=101.0, l=99.5, c=100.5, v=10000, vw=100.3, ts=1000001)]
    mock_client.get_aggregate_bars.return_value = page1
    mock_client.get_aggregate_bars.raw_response.headers.get.return_value = None

    source = PolygonDataSource(api_key="test_key")
    with patch.object(source, "client", mock_client):
        bars = source.fetch_aggregates("SPUS", "2026-03-26", "2026-03-26")
    assert isinstance(bars[0], Bar)


def test_fetch_aggregates_empty_list_on_error():
    source = PolygonDataSource(api_key="")
    bars = source.fetch_aggregates("SPUS", "bad-date", "bad-date")
    assert bars == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_cgan/test_polygon.py::test_fetch_aggregates tests/test_cgan/test_polygon.py::test_fetch_aggregates_pagination tests/test_cgan/test_polygon.py::test_fetch_aggregates_empty_list_on_error -v
```
Expected: `FAILED` (3 failures, `fetch_aggregates` not defined, `Bar` not importable)

- [ ] **Step 3: Implement `fetch_aggregates()`**

Import `Bar` in `market_cgan/data/polygon.py` and add method:

```python
from market_cgan.data.bar import Bar

def fetch_aggregates(
    self,
    ticker: str,
    start_date: str,
    end_date: str,
    timespan: str = "minute",
) -> list[Bar]:
    bars: list[Bar] = []
    try:
        raw = self.client.get_aggregate_bars(ticker, timespan, start_date, end_date)
        results = raw if isinstance(raw, list) else raw.results if hasattr(raw, "results") else []
        for item in results:
            bars.append(Bar(
                timestamp=int(getattr(item, "timestamp", 0) or 0),
                open=float(getattr(item, "open", 0) or 0),
                high=float(getattr(item, "high", 0) or 0),
                low=float(getattr(item, "low", 0) or 0),
                close=float(getattr(item, "close", 0) or 0),
                volume=float(getattr(item, "volume", 0) or 0),
                vwap=float(getattr(item, "vwap", 0) or 0),
            ))
    except Exception:
        pass
    return bars
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cgan/test_polygon.py::test_fetch_aggregates tests/test_cgan/test_polygon.py::test_fetch_aggregates_pagination tests/test_cgan/test_polygon.py::test_fetch_aggregates_empty_list_on_error -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add market_cgan/data/polygon.py tests/test_cgan/test_polygon.py
git commit -m "feat: add fetch_aggregates() to PolygonDataSource"
```

---

### Task 2: Create `Bar`, `BarFeatureExtractor`, `BarDataset`

**Files:**
- Create: `market_cgan/data/bar.py`
- Test: `tests/test_cgan/test_bar.py`

- [ ] **Step 1: Write failing tests**

`tests/test_cgan/test_bar.py`:

```python
import numpy as np
import torch
from market_cgan.data.bar import Bar, BarFeatureExtractor, BarDataset


def test_bar_dataclass():
    b = Bar(timestamp=1000, open=100.0, high=101.0, low=99.5, close=100.5, volume=10000, vwap=100.3)
    assert b.open == 100.0
    assert b.high == 101.0
    assert b.low == 99.5
    assert b.close == 100.5
    assert b.volume == 10000
    assert b.vwap == 100.3
    assert b.timestamp == 1000


def test_bar_feature_extractor():
    bars = [
        Bar(ts=1, o=100.0, h=101.0, l=99.5, c=100.5, v=10000, vw=100.3),
        Bar(ts=2, o=100.5, h=102.0, l=100.0, c=101.5, v=15000, vw=101.0),
    ]
    extractor = BarFeatureExtractor()
    feats = extractor(bars)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (6,)  # O, H, L, C, V, VWAP normalized
    assert feats.dtype == np.float32


def test_bar_feature_extractor_normalization():
    bars = [
        Bar(ts=1, o=100.0, h=100.0, l=100.0, c=100.0, v=1000, vw=100.0),
    ]
    extractor = BarFeatureExtractor()
    feats = extractor(bars)
    # All prices are 100, ref is 100 → all should be 1.0
    assert feats[0] == pytest.approx(1.0, abs=1e-6)
    assert feats[3] == pytest.approx(1.0, abs=1e-6)


def test_bar_dataset():
    bars = [Bar(ts=i, o=100.0 + i, h=101.0 + i, l=99.5 + i, c=100.5 + i, v=10000, vw=100.3) for i in range(10)]
    ds = BarDataset(bars, seq_len=1)
    assert len(ds) == 9
    state, target = ds[0]
    assert isinstance(state, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert state.shape == (6,)
    assert target.shape == (6,)
    assert state.dtype == torch.float32
    assert target.dtype == torch.float32


def test_bar_dataset_seq_len():
    bars = [Bar(ts=i, o=100.0 + i, h=101.0 + i, l=99.5 + i, c=100.5 + i, v=10000, vw=100.3) for i in range(10)]
    ds = BarDataset(bars, seq_len=3)
    assert len(ds) == 7
    state, target = ds[0]
    assert state.shape == (18,)  # 3 * 6


def test_bar_dataset_get_feature_dim():
    bars = [Bar(ts=i, o=100.0, h=101.0, l=99.5, c=100.5, v=10000, vw=100.3) for i in range(5)]
    ds = BarDataset(bars, seq_len=1)
    assert ds.get_feature_dim() == 6


def test_bar_dataset_empty():
    ds = BarDataset([], seq_len=1)
    assert len(ds) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cgan/test_bar.py -v
```
Expected: `FAILED` (import errors, all undefined)

- [ ] **Step 3: Implement `Bar`, `BarFeatureExtractor`, `BarDataset`**

`market_cgan/data/bar.py`:

```python
from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class Bar:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float


class BarFeatureExtractor:
    def __init__(self, log_norm: float = 15.0, max_move: float = 0.1):
        self.log_norm = log_norm
        self.max_move = max_move

    def __call__(self, bars: list[Bar]) -> np.ndarray:
        if not bars:
            return np.zeros(6, dtype=np.float32)
        ref = bars[0].open if bars[0].open > 0 else 100.0
        last = bars[-1]
        return np.array([
            last.open / ref,
            last.high / ref,
            last.low / ref,
            last.close / ref,
            np.log1p(last.volume) / self.log_norm,
            last.vwap / ref,
        ], dtype=np.float32)


class BarDataset(Dataset):
    def __init__(self, bars: list[Bar], seq_len: int = 1):
        self.bars = bars
        self.seq_len = seq_len
        self.extractor = BarFeatureExtractor()

    def __len__(self) -> int:
        return max(0, len(self.bars) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        curr = self.bars[idx:idx + self.seq_len]
        next_bar = self.bars[idx + self.seq_len]
        state = self.extractor(curr)
        target = self.extractor([next_bar])
        return torch.tensor(state, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

    def get_feature_dim(self) -> int:
        return 6

    def get_target_dim(self) -> int:
        return 6
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cgan/test_bar.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add market_cgan/data/bar.py tests/test_cgan/test_bar.py
git commit -m "feat: add Bar, BarFeatureExtractor, BarDataset"
```

---

### Task 3: Create `BarGenerator`

**Files:**
- Create: `market_cgan/models/bar_generator.py`
- Test: `tests/test_cgan/test_bar_generator.py`

- [ ] **Step 1: Write failing tests**

`tests/test_cgan/test_bar_generator.py`:

```python
import torch
from market_cgan.models.bar_generator import BarGenerator


def test_bar_generator_forward_shape():
    gen = BarGenerator(noise_dim=64, feature_dim=6, ref_price=100.0)
    noise = torch.randn(4, 64)
    features = torch.randn(4, 6)
    out = gen(noise, features)
    assert out.shape == (4, 6)


def test_bar_generator_ohlcv_constraints():
    gen = BarGenerator(noise_dim=64, feature_dim=6, ref_price=100.0, max_move=0.02)
    for _ in range(50):
        noise = torch.randn(1, 64)
        features = torch.randn(1, 6)
        out = gen(noise, features).squeeze(0)
        o, h, l, c, v, vw = out.tolist()
        assert h >= max(o, c), f"H={h} < max(O={o}, C={c})"
        assert l <= min(o, c), f"L={l} > min(O={o}, C={c})"
        assert v >= 0, f"V={v} < 0"


def test_bar_generator_volume_positive():
    gen = BarGenerator(noise_dim=64, feature_dim=6, ref_price=100.0)
    noise = torch.randn(10, 64)
    features = torch.randn(10, 6)
    out = gen(noise, features)
    assert torch.all(out[:, 4] >= 0)


def test_bar_generator_different_noise_different_output():
    gen = BarGenerator(noise_dim=64, feature_dim=6, ref_price=100.0)
    features = torch.randn(1, 6)
    n1 = torch.randn(1, 64)
    n2 = torch.randn(1, 64)
    o1 = gen(n1, features)
    o2 = gen(n2, features)
    assert not torch.allclose(o1, o2, rtol=1e-3)


def test_bar_generator_gradients():
    gen = BarGenerator(noise_dim=64, feature_dim=6, ref_price=100.0)
    noise = torch.randn(2, 64)
    features = torch.randn(2, 6)
    out = gen(noise, features)
    loss = out.mean()
    loss.backward()
    for p in gen.parameters():
        assert p.grad is not None
        break


def test_bar_generator_noise_dim_mismatch():
    gen = BarGenerator(noise_dim=64, feature_dim=6, ref_price=100.0)
    noise = torch.randn(1, 32)
    features = torch.randn(1, 6)
    try:
        gen(noise, features)
        assert False, "should have raised"
    except RuntimeError:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cgan/test_bar_generator.py -v
```
Expected: FAILED (import error)

- [ ] **Step 3: Implement BarGenerator**

`market_cgan/models/bar_generator.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BarGenerator(nn.Module):
    def __init__(
        self,
        noise_dim: int = 64,
        feature_dim: int = 6,
        ref_price: float = 100.0,
        max_move: float = 0.05,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim
        self.ref_price = ref_price
        self.max_move = max_move
        input_dim = noise_dim + feature_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
        )

        self.open_head = nn.Linear(128, 1)
        self.close_head = nn.Linear(128, 1)
        self.high_head = nn.Linear(128, 1)
        self.low_head = nn.Linear(128, 1)
        self.volume_head = nn.Linear(128, 1)
        self.vwap_head = nn.Linear(128, 1)

    def forward(self, noise: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noise, features], dim=-1)
        h = self.net(x)

        o_raw = torch.tanh(self.open_head(h))
        c_raw = torch.tanh(self.close_head(h))
        v_raw = F.softplus(self.volume_head(h))
        vw_raw = torch.tanh(self.vwap_head(h))
        h_raw = F.softplus(self.high_head(h))
        l_raw = F.softplus(self.low_head(h))

        feature_ref = features[:, 3:4] * self.ref_price  # normalized close * ref
        ref = feature_ref.clamp(min=1.0)

        o = ref * (1.0 + o_raw * self.max_move)
        c = ref * (1.0 + c_raw * self.max_move)
        o_max = torch.max(o, c)
        o_min = torch.min(o, c)
        h = o_max + h_raw * ref * self.max_move
        l = o_min - l_raw * ref * self.max_move
        v = v_raw * 10000
        vw = ref * (1.0 + vw_raw * self.max_move)

        return torch.cat([o, h, l, c, v, vw], dim=1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cgan/test_bar_generator.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add market_cgan/models/bar_generator.py tests/test_cgan/test_bar_generator.py
git commit -m "feat: add BarGenerator with OHLCV-constrained output"
```

---

### Task 4: Create `BarDiscriminator`

**Files:**
- Create: `market_cgan/models/bar_discriminator.py`
- Test: `tests/test_cgan/test_bar_discriminator.py`

- [ ] **Step 1: Write failing tests**

`tests/test_cgan/test_bar_discriminator.py`:

```python
import torch
from market_cgan.models.bar_discriminator import BarDiscriminator


def test_bar_discriminator_forward_shape():
    disc = BarDiscriminator(bar_dim=6, feature_dim=6)
    bar = torch.randn(4, 6)
    features = torch.randn(4, 6)
    out = disc(bar, features)
    assert out.shape == (4, 1)
    assert torch.all(out >= 0)
    assert torch.all(out <= 1)


def test_bar_discriminator_logits():
    disc = BarDiscriminator(bar_dim=6, feature_dim=6)
    bar = torch.randn(4, 6)
    features = torch.randn(4, 6)
    logits = disc.forward_logits(bar, features)
    assert logits.shape == (4, 1)


def test_bar_discriminator_differentiable():
    disc = BarDiscriminator(bar_dim=6, feature_dim=6)
    bar = torch.randn(2, 6, requires_grad=True)
    features = torch.randn(2, 6)
    logits = disc.forward_logits(bar, features)
    loss = logits.mean()
    loss.backward()
    assert bar.grad is not None


def test_bar_discriminator_gradients():
    disc = BarDiscriminator(bar_dim=6, feature_dim=6)
    bar = torch.randn(2, 6)
    features = torch.randn(2, 6)
    out = disc(bar, features)
    loss = out.mean()
    loss.backward()
    for p in disc.parameters():
        assert p.grad is not None
        break
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cgan/test_bar_discriminator.py -v
```
Expected: FAILED (import error)

- [ ] **Step 3: Implement BarDiscriminator**

`market_cgan/models/bar_discriminator.py`:

```python
import torch
import torch.nn as nn


class BarDiscriminator(nn.Module):
    def __init__(self, bar_dim: int = 6, feature_dim: int = 6):
        super().__init__()
        self.bar_dim = bar_dim
        self.feature_dim = feature_dim
        input_dim = bar_dim + feature_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, bar: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([bar, features], dim=-1)
        return torch.sigmoid(self.net(x))

    def forward_logits(self, bar: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([bar, features], dim=-1)
        return self.net(x)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cgan/test_bar_discriminator.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add market_cgan/models/bar_discriminator.py tests/test_cgan/test_bar_discriminator.py
git commit -m "feat: add BarDiscriminator"
```

---

### Task 5: Create `bar_physics_loss`

**Files:**
- Create: `market_cgan/training/bar_physics_loss.py`
- Test: `tests/test_cgan/test_bar_physics_loss.py`

- [ ] **Step 1: Write failing tests**

`tests/test_cgan/test_bar_physics_loss.py`:

```python
import torch
from market_cgan.training.bar_physics_loss import (
    hl_validity_loss,
    volume_positivity_loss,
    return_distribution_loss,
    volatility_clustering_loss,
    bar_physics_loss,
)


def test_hl_validity_loss_pass():
    gen_bars = torch.tensor([[100.0, 101.0, 99.0, 100.5, 10000, 100.3]], dtype=torch.float32)
    loss = hl_validity_loss(gen_bars)
    assert loss.item() == 0.0


def test_hl_validity_loss_violation_high():
    gen_bars = torch.tensor([[100.0, 99.0, 99.0, 100.5, 10000, 100.3]], dtype=torch.float32)
    loss = hl_validity_loss(gen_bars)
    assert loss.item() > 0


def test_hl_validity_loss_violation_low():
    gen_bars = torch.tensor([[100.0, 101.0, 101.0, 100.5, 10000, 100.3]], dtype=torch.float32)
    loss = hl_validity_loss(gen_bars)
    assert loss.item() > 0


def test_volume_positivity_loss_pass():
    gen_bars = torch.tensor([[100.0, 101.0, 99.0, 100.5, 10000, 100.3]], dtype=torch.float32)
    loss = volume_positivity_loss(gen_bars)
    assert loss.item() == 0.0


def test_volume_positivity_loss_violation():
    gen_bars = torch.tensor([[100.0, 101.0, 99.0, 100.5, -100, 100.3]], dtype=torch.float32)
    loss = volume_positivity_loss(gen_bars)
    assert loss.item() > 0


def test_return_distribution_loss():
    gen_bars = torch.randn(10, 6)
    real_bars = torch.randn(10, 6)
    loss = return_distribution_loss(gen_bars, real_bars)
    assert loss.item() >= 0
    assert loss.shape == ()


def test_volatility_clustering_loss():
    gen_bars = torch.randn(20, 6)
    loss = volatility_clustering_loss(gen_bars)
    assert loss.item() >= 0
    assert loss.shape == ()


def test_bar_physics_loss_returns_dict():
    gen_bars = torch.randn(10, 6)
    real_bars = torch.randn(10, 6)
    terms = bar_physics_loss(gen_bars, real_bars)
    assert isinstance(terms, dict)
    assert "bar_hl_validity" in terms
    assert "bar_volume_positivity" in terms
    assert "bar_return_dist" in terms
    assert "bar_vol_clustering" in terms
    for v in terms.values():
        assert isinstance(v, torch.Tensor)
        assert v.shape == ()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cgan/test_bar_physics_loss.py -v
```
Expected: FAILED (import error)

- [ ] **Step 3: Implement bar_physics_loss**

`market_cgan/training/bar_physics_loss.py`:

```python
import torch
import torch.nn.functional as F


def hl_validity_loss(gen_bars: torch.Tensor) -> torch.Tensor:
    o = gen_bars[:, 0]
    h = gen_bars[:, 1]
    l = gen_bars[:, 2]
    c = gen_bars[:, 3]
    o_max = torch.max(o, c)
    o_min = torch.min(o, c)
    high_violation = torch.relu(o_max - h)
    low_violation = torch.relu(l - o_min)
    return (high_violation + low_violation).mean()


def volume_positivity_loss(gen_bars: torch.Tensor) -> torch.Tensor:
    v = gen_bars[:, 4]
    return torch.relu(-v).mean()


def return_distribution_loss(
    gen_bars: torch.Tensor,
    real_bars: torch.Tensor,
) -> torch.Tensor:
    gen_returns = torch.diff(torch.log(gen_bars[:, 3].clamp(min=1e-6)))
    real_returns = torch.diff(torch.log(real_bars[:, 3].clamp(min=1e-6)))
    if gen_returns.numel() < 2 or real_returns.numel() < 2:
        return torch.tensor(0.0, device=gen_bars.device)
    gen_mean = gen_returns.mean()
    real_mean = real_returns.mean()
    gen_std = gen_returns.std() + 1e-8
    real_std = real_returns.std() + 1e-8
    gen_norm = (gen_returns - gen_mean) / gen_std
    real_norm = (real_returns - real_mean) / real_std
    n = min(gen_norm.numel(), real_norm.numel())
    return F.mse_loss(gen_norm[:n], real_norm[:n])


def volatility_clustering_loss(gen_bars: torch.Tensor) -> torch.Tensor:
    returns = torch.diff(torch.log(gen_bars[:, 3].clamp(min=1e-6)))
    if returns.numel() < 4:
        return torch.tensor(0.0, device=gen_bars.device)
    abs_ret = returns.abs()
    lag1 = abs_ret[:-1]
    lag2 = abs_ret[1:]
    if lag1.numel() < 2 or lag2.numel() < 2:
        return torch.tensor(0.0, device=gen_bars.device)
    corr = torch.corrcoef(torch.stack([lag1[:len(lag2)], lag2]))[0, 1]
    return (corr - 0.2).pow(2)


def bar_physics_loss(
    gen_bars: torch.Tensor,
    real_bars: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "bar_hl_validity": hl_validity_loss(gen_bars),
        "bar_volume_positivity": volume_positivity_loss(gen_bars),
        "bar_return_dist": return_distribution_loss(gen_bars, real_bars),
        "bar_vol_clustering": volatility_clustering_loss(gen_bars),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cgan/test_bar_physics_loss.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add market_cgan/training/bar_physics_loss.py tests/test_cgan/test_bar_physics_loss.py
git commit -m "feat: add bar_physics_loss with OHLCV constraints"
```

---

### Task 6: Add `train_cgan_bar()` to trainer

**Files:**
- Modify: `market_cgan/training/trainer.py`
- Test: `tests/test_cgan/test_trainer.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_cgan/test_trainer.py`:

```python
from market_cgan.models.bar_generator import BarGenerator
from market_cgan.models.bar_discriminator import BarDiscriminator
from market_cgan.training.trainer import train_cgan_bar


def test_train_cgan_bar_overfit():
    gen = BarGenerator(noise_dim=8, feature_dim=6, ref_price=100.0)
    disc = BarDiscriminator(bar_dim=6, feature_dim=6)
    features = torch.randn(64, 6)
    bars = torch.randn(64, 6)
    dataset = TensorDataset(features, bars)
    loader = DataLoader(dataset, batch_size=16)
    history = train_cgan_bar(gen, disc, loader, epochs=5, lr=1e-3, log_interval=100)
    assert "g_loss" in history
    assert "d_loss" in history
    assert len(history["g_loss"]) == 5
    assert len(history["d_loss"]) == 5


def test_train_cgan_bar_with_validation():
    gen = BarGenerator(noise_dim=8, feature_dim=6, ref_price=100.0)
    disc = BarDiscriminator(bar_dim=6, feature_dim=6)
    features = torch.randn(32, 6)
    bars = torch.randn(32, 6)
    dataset = TensorDataset(features, bars)
    train_loader = DataLoader(dataset, batch_size=8)
    val_loader = DataLoader(dataset, batch_size=8)
    history = train_cgan_bar(gen, disc, train_loader, val_loader, epochs=3, lr=1e-3, log_interval=100)
    assert "val_g_loss" in history
    assert len(history["val_g_loss"]) == 3


def test_train_cgan_bar_with_physics():
    gen = BarGenerator(noise_dim=8, feature_dim=6, ref_price=100.0)
    disc = BarDiscriminator(bar_dim=6, feature_dim=6)
    features = torch.randn(32, 6)
    bars = torch.randn(32, 6)
    dataset = TensorDataset(features, bars)
    loader = DataLoader(dataset, batch_size=8)
    history = train_cgan_bar(gen, disc, loader, epochs=3, lr=1e-3, physics_weight=0.5, log_interval=100)
    assert "bar_hl_validity" in history
    assert "bar_volume_positivity" in history
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cgan/test_trainer.py::test_train_cgan_bar_overfit tests/test_cgan/test_trainer.py::test_train_cgan_bar_with_validation tests/test_cgan/test_trainer.py::test_train_cgan_bar_with_physics -v
```
Expected: FAILED (train_cgan_bar not defined)

- [ ] **Step 3: Update loss functions to support label smoothing**

Modify functions in `market_cgan/training/losses.py`:

```python
def generator_loss(
    fake_logits: torch.Tensor,
    fake_features: torch.Tensor | None = None,
    real_features: torch.Tensor | None = None,
    feature_matching_weight: float = 10.0,
    smooth_real: float = 1.0,
) -> torch.Tensor:
    g_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits) * smooth_real)
    total = g_loss
    if fake_features is not None and real_features is not None:
        fm_loss = F.mse_loss(fake_features.mean(0), real_features.mean(0))
        total = total + feature_matching_weight * fm_loss
    return total


def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    smooth_real: float = 1.0,
    smooth_fake: float = 0.0,
) -> torch.Tensor:
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits) * smooth_real)
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits) + smooth_fake)
    return (real_loss + fake_loss) / 2
```

- [ ] **Step 4: Implement `train_cgan_bar()`**

Append to `market_cgan/training/trainer.py`:

```python
from market_cgan.training.bar_physics_loss import bar_physics_loss as bar_physics_loss_fn

BAR_PHYSICS_TERM_NAMES = ["bar_hl_validity", "bar_volume_positivity", "bar_return_dist", "bar_vol_clustering"]


def train_cgan_bar(
    generator: BarGenerator,
    discriminator: BarDiscriminator,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 100,
    lr: float = 2e-4,
    betas: tuple[float, float] = (0.5, 0.999),
    label_smoothing: float = 0.0,
    gp_weight: float = 0.0,
    physics_weight: float = 0.0,
    log_interval: int = 10,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    generator.to(device)
    discriminator.to(device)

    g_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    history: dict[str, list[float]] = {
        "g_loss": [], "d_loss": [], "val_g_loss": [],
        **{name: [] for name in BAR_PHYSICS_TERM_NAMES},
    }
    noise_dim = generator.noise_dim
    smooth_real = 1.0 - label_smoothing
    smooth_fake = label_smoothing
    best_val_loss = float("inf")

    for epoch in range(epochs):
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0
        epoch_physics: dict[str, float] = {name: 0.0 for name in BAR_PHYSICS_TERM_NAMES}
        batches = 0

        generator.train()
        discriminator.train()

        for features, real_bars in train_loader:
            features = features.to(device)
            real_bars = real_bars.to(device)
            batch_size = features.size(0)

            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_bars = generator(noise, features)

            real_logits = discriminator.forward_logits(real_bars, features)
            fake_logits_d = discriminator.forward_logits(fake_bars.detach(), features)

            d_loss = discriminator_loss(real_logits, fake_logits_d, smooth_real, smooth_fake)
            if gp_weight > 0:
                d_loss = d_loss + gradient_penalty(discriminator, real_bars, fake_bars.detach(), features)

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            fake_logits_g = discriminator.forward_logits(fake_bars, features)
            g_loss = generator_loss(fake_logits_g, smooth_real=smooth_real)

            if physics_weight > 0:
                physics_terms = bar_physics_loss_fn(fake_bars, real_bars)
                for name in BAR_PHYSICS_TERM_NAMES:
                    term = physics_terms[name]
                    g_loss = g_loss + physics_weight * term
                    epoch_physics[name] += term.item()

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            g_epoch_loss += g_loss.item()
            d_epoch_loss += d_loss.item()
            batches += 1

        avg_g = g_epoch_loss / max(batches, 1)
        avg_d = d_epoch_loss / max(batches, 1)
        history["g_loss"].append(avg_g)
        history["d_loss"].append(avg_d)
        for name in BAR_PHYSICS_TERM_NAMES:
            history[name].append(epoch_physics[name] / max(batches, 1))

        val_loss = 0.0
        if val_loader is not None:
            generator.eval()
            val_batches = 0
            with torch.no_grad():
                for features, real_bars in val_loader:
                    features = features.to(device)
                    real_bars = real_bars.to(device)
                    batch_size = features.size(0)
                    noise = torch.randn(batch_size, noise_dim, device=device)
                    fake_bars = generator(noise, features)
                    fake_logits = discriminator.forward_logits(fake_bars, features)
                    v_loss = F.binary_cross_entropy_with_logits(
                        fake_logits, torch.ones_like(fake_logits) * smooth_real
                    )
                    val_loss += v_loss.item()
                    val_batches += 1
            val_loss /= max(val_batches, 1)
            history["val_g_loss"].append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        if (epoch + 1) % log_interval == 0:
            msg = f"Epoch {epoch+1}/{epochs} | G: {avg_g:.4f} | D: {avg_d:.4f}"
            if physics_weight > 0:
                physics_msgs = [f"{k}={history[k][-1]:.4f}" for k in BAR_PHYSICS_TERM_NAMES]
                msg += " | " + " ".join(physics_msgs)
            if val_loader is not None:
                msg += f" | Val: {val_loss:.4f}"
            print(msg)

    return history
```

- [ ] **Step 5: Verify existing tests still pass**

```bash
pytest tests/test_cgan/test_losses.py tests/test_cgan/test_trainer.py -v
```
Expected: all PASS (backward-compatible defaults)

- [ ] **Step 6: Run new bar trainer tests to verify they pass**

```bash
pytest tests/test_cgan/test_trainer.py::test_train_cgan_bar_overfit tests/test_cgan/test_trainer.py::test_train_cgan_bar_with_validation tests/test_cgan/test_trainer.py::test_train_cgan_bar_with_physics -v
```
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add market_cgan/training/losses.py market_cgan/training/trainer.py tests/test_cgan/test_trainer.py
git commit -m "feat: add train_cgan_bar with physics-informed training for OHLCV"
```

---

### Task 7: Create `BarExchange`

**Files:**
- Create: `market_cgan/simulation/bar_exchange.py`
- Test: `tests/test_cgan/test_bar_exchange.py`

- [ ] **Step 1: Write failing tests**

`tests/test_cgan/test_bar_exchange.py`:

```python
import numpy as np
from market_cgan.simulation.bar_exchange import BarExchange
from market_cgan.data.bar import Bar


def test_bar_exchange_initial_state():
    ex = BarExchange()
    state = ex.get_state()
    assert "open" in state
    assert "high" in state
    assert "low" in state
    assert "close" in state
    assert state["close"] is None


def test_bar_exchange_append_bar():
    ex = BarExchange()
    bar = Bar(timestamp=1, open=100.0, high=101.0, low=99.5, close=100.5, volume=10000, vwap=100.3)
    ex.append_bar(bar)
    state = ex.get_state()
    assert state["close"] == 100.5
    assert state["open"] == 100.0


def test_bar_exchange_get_window():
    ex = BarExchange()
    for i in range(10):
        b = Bar(timestamp=i, open=100.0 + i, high=101.0 + i, low=99.5 + i, close=100.5 + i, volume=10000, vwap=100.3)
        ex.append_bar(b)
    window = ex.get_window(5)
    assert len(window) == 5
    assert window[0].close == 100.5 + 5
    assert window[-1].close == 100.5 + 9


def test_bar_exchange_reset():
    ex = BarExchange()
    bar = Bar(timestamp=1, open=100.0, high=101.0, low=99.5, close=100.5, volume=10000, vwap=100.3)
    ex.append_bar(bar)
    ex.reset()
    assert len(ex.bars) == 0
    assert ex.get_state()["close"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cgan/test_bar_exchange.py -v
```
Expected: FAILED (import error)

- [ ] **Step 3: Implement BarExchange**

`market_cgan/simulation/bar_exchange.py`:

```python
from __future__ import annotations

import numpy as np
from market_cgan.data.bar import Bar


class BarExchange:
    def __init__(self, window: int = 10):
        self.bars: list[Bar] = []
        self.window = window

    def append_bar(self, bar: Bar):
        self.bars.append(bar)

    def get_state(self) -> dict:
        if not self.bars:
            return {"open": None, "high": None, "low": None, "close": None, "volume": None, "vwap": None}
        last = self.bars[-1]
        return {
            "open": last.open,
            "high": last.high,
            "low": last.low,
            "close": last.close,
            "volume": last.volume,
            "vwap": last.vwap,
        }

    def get_window(self, n: int | None = None) -> list[Bar]:
        n = n or self.window
        return self.bars[-n:]

    def reset(self):
        self.bars.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cgan/test_bar_exchange.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add market_cgan/simulation/bar_exchange.py tests/test_cgan/test_bar_exchange.py
git commit -m "feat: add BarExchange for bar sequence management"
```

---

### Task 8: Create `BarWorldAgent`

**Files:**
- Create: `market_cgan/simulation/bar_world_agent.py`
- Test: `tests/test_cgan/test_bar_world_agent.py`

- [ ] **Step 1: Write failing tests**

`tests/test_cgan/test_bar_world_agent.py`:

```python
import torch
import numpy as np
from market_cgan.simulation.bar_world_agent import BarWorldAgent
from market_cgan.simulation.bar_exchange import BarExchange
from market_cgan.models.bar_generator import BarGenerator
from market_cgan.data.bar import Bar


def test_bar_world_agent_step():
    gen = BarGenerator(noise_dim=8, feature_dim=6, ref_price=100.0)
    ex = BarExchange()
    # seed with one bar so features are meaningful
    ex.append_bar(Bar(timestamp=1, open=100.0, high=101.0, low=99.5, close=100.5, volume=10000, vwap=100.3))
    agent = BarWorldAgent(generator=gen, exchange=ex, noise_dim=8)
    new_bar = agent.step()
    assert isinstance(new_bar, Bar)
    assert new_bar.open > 0
    assert new_bar.high >= max(new_bar.open, new_bar.close)
    assert new_bar.low <= min(new_bar.open, new_bar.close)
    assert new_bar.volume >= 0
    assert len(ex.bars) == 2


def test_bar_world_agent_reset():
    gen = BarGenerator(noise_dim=8, feature_dim=6, ref_price=100.0)
    ex = BarExchange()
    ex.append_bar(Bar(timestamp=1, open=100.0, high=101.0, low=99.5, close=100.5, volume=10000, vwap=100.3))
    agent = BarWorldAgent(generator=gen, exchange=ex, noise_dim=8)
    agent.reset()
    assert len(ex.bars) == 0


def test_bar_world_agent_feature_extraction():
    gen = BarGenerator(noise_dim=8, feature_dim=6, ref_price=100.0)
    ex = BarExchange()
    for i in range(5):
        ex.append_bar(Bar(timestamp=i, open=100.0 + i, high=101.0 + i, low=99.5 + i, close=100.5 + i, volume=10000, vwap=100.3))
    agent = BarWorldAgent(generator=gen, exchange=ex, noise_dim=8)
    assert agent._get_features().shape == (6,)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cgan/test_bar_world_agent.py -v
```
Expected: FAILED (import error)

- [ ] **Step 3: Implement BarWorldAgent**

`market_cgan/simulation/bar_world_agent.py`:

```python
from __future__ import annotations

import torch
import numpy as np
from market_cgan.models.bar_generator import BarGenerator
from market_cgan.simulation.bar_exchange import BarExchange
from market_cgan.data.bar import Bar, BarFeatureExtractor


class BarWorldAgent:
    def __init__(
        self,
        generator: BarGenerator,
        exchange: BarExchange,
        noise_dim: int = 64,
        device: str | None = None,
    ):
        self.generator = generator
        self.exchange = exchange
        self.noise_dim = noise_dim
        self.extractor = BarFeatureExtractor()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.generator.to(self.device)
        self.generator.eval()

    def _get_features(self) -> np.ndarray:
        window = self.exchange.get_window(min(3, len(self.exchange.bars)))
        if not window:
            return np.zeros(6, dtype=np.float32)
        return self.extractor(window)

    def step(self) -> Bar:
        features = self._get_features()
        feat_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        noise = torch.randn(1, self.noise_dim, device=self.device)

        with torch.no_grad():
            bar_tensor = self.generator(noise, feat_tensor).squeeze(0)

        o, h, l, c, v, vw = bar_tensor.tolist()
        ts = int(self.exchange.bars[-1].timestamp) + 1 if self.exchange.bars else 1
        new_bar = Bar(timestamp=ts, open=o, high=h, low=l, close=c, volume=v, vwap=vw)
        self.exchange.append_bar(new_bar)
        return new_bar

    def reset(self):
        self.exchange.reset()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cgan/test_bar_world_agent.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add market_cgan/simulation/bar_world_agent.py tests/test_cgan/test_bar_world_agent.py
git commit -m "feat: add BarWorldAgent for bar-to-bar generation loop"
```

---

### Task 9: Add `--bar-mode` to CLI

**Files:**
- Modify: `cli/cgan_cmd.py`

- [ ] **Step 1: Write failing tests**

`tests/test_cgan/test_cli.py`:

```python
import pytest
from typer.testing import CliRunner
from cli.cgan_cmd import app

runner = CliRunner()


def test_train_bar_mode_flag():
    result = runner.invoke(app, ["train", "--help"])
    assert "--bar-mode" in result.output
    assert "--data-source" in result.output


def test_generate_bar_mode_flag():
    result = runner.invoke(app, ["generate", "--help"])
    assert "--bar-mode" in result.output
    assert "--steps" in result.output


def test_simulate_bar_mode_flag():
    result = runner.invoke(app, ["simulate", "--help"])
    assert "--bar-mode" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cgan/test_cli.py -v
```
Expected: FAILED (help text doesn't include --bar-mode yet)

- [ ] **Step 3: Update CLI to support `--bar-mode`**

Modify `cli/cgan_cmd.py`:

Add new constants at top:
```python
BAR_FEATURE_DIM = 6
BAR_TARGET_DIM = 6
```

Update `train` command signature:
```python
def train(
    ...
    bar_mode: bool = typer.Option(False, "--bar-mode", help="Use OHLCV bar CGAN instead of LOB CGAN"),
    ...
):
```

Update train body - add bar-mode branching after dataset creation:
```python
    if bar_mode:
        from market_cgan.models.bar_generator import BarGenerator
        from market_cgan.models.bar_discriminator import BarDiscriminator
        from market_cgan.training.trainer import train_cgan_bar

        gen = BarGenerator(noise_dim=NOISE_DIM, feature_dim=BAR_FEATURE_DIM, ref_price=100.0)
        disc = BarDiscriminator(bar_dim=BAR_TARGET_DIM, feature_dim=BAR_FEATURE_DIM)
        typer.echo(f"BarGenerator params: {sum(p.numel() for p in gen.parameters()):,}")
        typer.echo(f"BarDiscriminator params: {sum(p.numel() for p in disc.parameters()):,}")

        history = train_cgan_bar(
            gen, disc,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            gp_weight=gp_weight,
            physics_weight=physics_weight,
            log_interval=log_interval,
            device=device,
        )
    else:
        gen = Generator(noise_dim=NOISE_DIM, feature_dim=FEATURE_DIM)
        disc = Discriminator(action_dim=ACTION_DIM, feature_dim=FEATURE_DIM)
        typer.echo(f"Generator params: {sum(p.numel() for p in gen.parameters()):,}")
        typer.echo(f"Discriminator params: {sum(p.numel() for p in disc.parameters()):,}")

        history = train_cgan(
            gen, disc,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            gp_weight=gp_weight,
            physics_weight=physics_weight,
            log_interval=log_interval,
            device=device,
        )
```

Update `generate` command:
```python
def generate(
    ...
    bar_mode: bool = typer.Option(False, "--bar-mode", help="Output OHLCV bars instead of LOB actions"),
    ...
):
```

In generate body, add bar-mode branching:
```python
    if bar_mode:
        from market_cgan.models.bar_generator import BarGenerator
        from market_cgan.simulation.bar_exchange import BarExchange
        from market_cgan.simulation.bar_world_agent import BarWorldAgent

        gen = BarGenerator(noise_dim=NOISE_DIM, feature_dim=BAR_FEATURE_DIM, ref_price=100.0)
        gen.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        gen.eval()

        ex = BarExchange()
        ex.append_bar(Bar(timestamp=1, open=100.0, high=101.0, low=99.5, close=100.5, volume=10000, vwap=100.3))
        agent = BarWorldAgent(gen, ex, noise_dim=NOISE_DIM)

        all_bars = []
        for _ in range(steps):
            bar = agent.step()
            all_bars.append(bar)

        import csv
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step", "open", "high", "low", "close", "volume", "vwap"])
            w.writeheader()
            for i, b in enumerate(all_bars):
                w.writerow({"step": i, "open": b.open, "high": b.high, "low": b.low,
                           "close": b.close, "volume": b.volume, "vwap": b.vwap})
        typer.echo(f"Generated {steps} bars -> {out}")
    else:
        ...existing generate code...
```

Update `simulate` command:
```python
def simulate(
    ...
    bar_mode: bool = typer.Option(False, "--bar-mode", help="Simulate OHLCV bars instead of LOB"),
    ...
):
```

In simulate body, add bar-mode branching:
```python
    if bar_mode:
        from market_cgan.models.bar_generator import BarGenerator
        from market_cgan.simulation.bar_exchange import BarExchange
        from market_cgan.simulation.bar_world_agent import BarWorldAgent

        gen = BarGenerator(noise_dim=NOISE_DIM, feature_dim=BAR_FEATURE_DIM, ref_price=100.0)
        gen.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        gen.to(device)
        gen.eval()

        ex = BarExchange()
        ex.append_bar(Bar(timestamp=1, open=100.0, high=101.0, low=99.5, close=100.5, volume=10000, vwap=100.3))
        agent = BarWorldAgent(gen, ex, noise_dim=NOISE_DIM)

        typer.echo(f"Running bar simulation ({steps} steps) with model {model_path}...")
        for i in range(steps):
            bar = agent.step()
            if i % 20 == 0:
                typer.echo(f"  step {i}: O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} C={bar.close:.2f} V={bar.volume:.0f}")
    else:
        ...existing simulate code...
```

Import `Bar` at top of `_build_dataset`:
```python
from market_cgan.data.bar import Bar
```

Also update `_build_dataset` for bar mode:
```python
def _build_dataset(data_source: str, polygon_ticker: str, polygon_dates: list[str],
                   seq_len: int = 1, bar_mode: bool = False):
    if bar_mode:
        extractor = MarketFeatureExtractor()
        if data_source == "synthetic":
            from market_cgan.data.bar import BarDataset
            # Generate synthetic bars
            rng = np.random.default_rng(seed=42)
            bars = []
            px = 100.0
            for i in range(2000):
                bar = Bar(timestamp=i, open=px, high=px + 0.5, low=px - 0.5,
                          close=px + rng.normal(0, 0.1), volume=abs(rng.normal(10000, 2000)),
                          vwap=px + rng.normal(0, 0.05))
                bars.append(bar)
                px = bar.close
            return BarDataset(bars, seq_len=seq_len)
        elif data_source == "polygon":
            from market_cgan.data.bar import BarDataset
            from market_cgan.data.polygon import PolygonDataSource
            api_key = os.environ.get("POLYGON_API_KEY", "")
            if not api_key:
                typer.echo("ERROR: POLYGON_API_KEY not set", err=True)
                raise typer.Exit(1)
            source = PolygonDataSource(api_key)
            all_bars = []
            for d in polygon_dates:
                bars = source.fetch_aggregates(polygon_ticker, d, d)
                all_bars.extend(bars)
            return BarDataset(all_bars, seq_len=seq_len)
    else:
        ...existing _build_dataset code...
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cgan/test_cli.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cli/cgan_cmd.py tests/test_cgan/test_cli.py
git commit -m "feat: add --bar-mode flag to CLI commands for OHLCV CGAN"
```

---

### Task 10: Update training script `train_spus_q1_2026.py`

**Files:**
- Modify: `scripts/train_spus_q1_2026.py`

- [ ] **Step 1: Update the training script**

`scripts/train_spus_q1_2026.py`:

```python
"""
Train OHLCV bar CGAN on SPUS Q1 2026 minute aggregate data from Polygon.io.

Usage:
    python scripts/train_spus_q1_2026.py [--epochs 100] [--physics-weight 0.1]
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, random_split
from market_cgan.data.bar import BarDataset
from market_cgan.data.polygon import PolygonDataSource
from market_cgan.models.bar_generator import BarGenerator
from market_cgan.models.bar_discriminator import BarDiscriminator
from market_cgan.training.trainer import train_cgan_bar

TICKER = "SPUS"
DATES = [
    "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
    "2026-01-09", "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15",
    "2026-01-16", "2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23",
    "2026-01-26", "2026-01-27", "2026-01-28", "2026-01-29", "2026-01-30",
    "2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05", "2026-02-06",
    "2026-02-09", "2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13",
    "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20", "2026-02-23",
    "2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27",
    "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06",
    "2026-03-09", "2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13",
    "2026-03-16", "2026-03-17", "2026-03-18", "2026-03-19", "2026-03-20",
    "2026-03-23", "2026-03-24", "2026-03-25", "2026-03-26", "2026-03-27",
    "2026-03-30", "2026-03-31",
]

OUT_DIR = Path("models/cgan/spus_q1_2026")
NOISE_DIM = 64
FEATURE_DIM = 6
BATCH_SIZE = 128
EPOCHS = 100
LR = 2e-4
PHYSICS_WEIGHT = 0.1
VAL_SPLIT = 0.1


def main():
    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY environment variable not set")
        sys.exit(1)

    print(f"Fetching {TICKER} minute aggregates for {len(DATES)} trading days...")
    source = PolygonDataSource(api_key)
    all_bars = []
    for d in DATES:
        bars = source.fetch_aggregates(TICKER, d, d)
        all_bars.extend(bars)
        print(f"  {d}: {len(bars)} bars")

    print(f"Total bars: {len(all_bars)}")
    if len(all_bars) < 10:
        print("ERROR: too few bars. Check API key and date range.")
        sys.exit(1)

    dataset = BarDataset(all_bars, seq_len=1)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE) if val_size > 0 else None

    gen = BarGenerator(noise_dim=NOISE_DIM, feature_dim=FEATURE_DIM, ref_price=100.0)
    disc = BarDiscriminator(bar_dim=FEATURE_DIM, feature_dim=FEATURE_DIM)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    print(f"Generator params: {sum(p.numel() for p in gen.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in disc.parameters()):,}")

    history = train_cgan_bar(
        gen, disc,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        physics_weight=PHYSICS_WEIGHT,
        log_interval=10,
        device=device,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(gen.state_dict(), OUT_DIR / "generator.pt")
    torch.save(disc.state_dict(), OUT_DIR / "discriminator.pt")
    print(f"Models saved to {OUT_DIR.resolve()}")
    print(f"Final G loss: {history['g_loss'][-1]:.4f} | D loss: {history['d_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train_spus_q1_2026.py
git commit -m "feat: update training script for OHLCV bar CGAN with SPUS Q1 2026"
```
