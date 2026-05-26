import numpy as np
import torch
import pytest
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
        Bar(timestamp=1, open=100.0, high=101.0, low=99.5, close=100.5, volume=10000, vwap=100.3),
        Bar(timestamp=2, open=100.5, high=102.0, low=100.0, close=101.5, volume=15000, vwap=101.0),
    ]
    extractor = BarFeatureExtractor()
    feats = extractor(bars)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (6,)  # O, H, L, C, V, VWAP normalized
    assert feats.dtype == np.float32


def test_bar_feature_extractor_normalization():
    bars = [
        Bar(timestamp=1, open=100.0, high=100.0, low=100.0, close=100.0, volume=1000, vwap=100.0),
    ]
    extractor = BarFeatureExtractor()
    feats = extractor(bars)
    # All prices are 100, ref is 100 -> all should be 1.0
    assert feats[0] == pytest.approx(1.0, abs=1e-6)
    assert feats[3] == pytest.approx(1.0, abs=1e-6)


def test_bar_dataset():
    bars = [Bar(timestamp=i, open=100.0 + i, high=101.0 + i, low=99.5 + i, close=100.5 + i, volume=10000, vwap=100.3) for i in range(10)]
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
    bars = [Bar(timestamp=i, open=100.0 + i, high=101.0 + i, low=99.5 + i, close=100.5 + i, volume=10000, vwap=100.3) for i in range(10)]
    ds = BarDataset(bars, seq_len=3)
    assert len(ds) == 7
    state, target = ds[0]
    assert state.shape == (18,)  # 3 * 6


def test_bar_dataset_get_feature_dim():
    bars = [Bar(timestamp=i, open=100.0, high=101.0, low=99.5, close=100.5, volume=10000, vwap=100.3) for i in range(5)]
    ds1 = BarDataset(bars, seq_len=1)
    assert ds1.get_feature_dim() == 6
    ds3 = BarDataset(bars, seq_len=3)
    assert ds3.get_feature_dim() == 18


def test_bar_feature_extractor_empty_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        BarFeatureExtractor()([])

def test_bar_dataset_empty():
    ds = BarDataset([], seq_len=1)
    assert len(ds) == 0
