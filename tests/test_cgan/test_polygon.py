from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import torch
import pytest
from market_cgan.data.polygon import (
    bbo_to_multilevel,
    PolygonDataSource,
    PolygonDataset,
)


class MockQuote:
    def __init__(self, bid_price, bid_size, ask_price, ask_size, timestamp):
        self.bid_price = bid_price
        self.bid_size = bid_size
        self.ask_price = ask_price
        self.ask_size = ask_size
        self.timestamp = timestamp


def test_bbo_to_multilevel_output_shape():
    bp, bv, ap, av = bbo_to_multilevel(99.5, 100, 100.5, 80)
    assert len(bp) == 10
    assert len(bv) == 10
    assert len(ap) == 10
    assert len(av) == 10
    assert np.all(bp > 0)
    assert np.all(ap > 0)
    assert np.all(bv >= 1)
    assert np.all(av >= 1)


def test_bbo_to_multilevel_bids_decrease():
    bp, _, ap, _ = bbo_to_multilevel(99.5, 100, 100.5, 80)
    for i in range(len(bp) - 1):
        assert bp[i] > bp[i + 1]
    for i in range(len(ap) - 1):
        assert ap[i] < ap[i + 1]


def test_bbo_to_multilevel_best_bid_ask_preserved():
    bp, _, ap, _ = bbo_to_multilevel(99.5, 100, 100.5, 80)
    assert bp[0] == pytest.approx(99.5, abs=0.01)
    assert ap[0] == pytest.approx(100.5, abs=0.01)


def test_bbo_to_multilevel_decay():
    _, bv, _, av = bbo_to_multilevel(99.5, 100, 100.5, 80)
    for i in range(min(5, len(bv) - 1)):
        assert bv[i] >= bv[i + 1] * 0.5
        assert av[i] >= av[i + 1] * 0.5


@patch("market_cgan.data.polygon.PolygonDataSource.client")
def test_fetch_quotes(mock_client_prop):
    mock_client = MagicMock()
    mock_client.list_quotes.return_value = [
        MockQuote(bid_price=99.5, bid_size=100, ask_price=100.5, ask_size=80, timestamp=1000001),
        MockQuote(bid_price=99.6, bid_size=90, ask_price=100.4, ask_size=70, timestamp=1000002),
    ]

    source = PolygonDataSource(api_key="test_key")
    with patch.object(source, "client", mock_client):
        snaps = source.fetch_quotes("AAPL", "2025-01-10")

    assert len(snaps) == 2
    assert snaps[0].bid_prices[0] == pytest.approx(99.5, abs=0.01)
    assert snaps[1].bid_prices[0] == pytest.approx(99.6, abs=0.01)
    assert snaps[0].bid_prices.shape == (10,)


@patch("market_cgan.data.polygon.PolygonDataSource.fetch_quotes")
def test_polygon_dataset_length(mock_fetch):
    mock_fetch.return_value = [
        _make_snapshot(ts=100 * i, bid=99.5 + i * 0.01, ask=100.5 + i * 0.01)
        for i in range(20)
    ]
    ds = PolygonDataset("AAPL", ["2025-01-10"], api_key="test", seq_len=1)
    assert len(ds) == 19


@patch("market_cgan.data.polygon.PolygonDataSource.fetch_quotes")
def test_polygon_dataset_getitem(mock_fetch):
    mock_fetch.return_value = [
        _make_snapshot(ts=100 * i, bid=99.5 + i * 0.01, ask=100.5 + i * 0.01)
        for i in range(5)
    ]
    ds = PolygonDataset("AAPL", ["2025-01-10"], api_key="test", seq_len=1)
    state, action = ds[0]
    assert state.shape == (42,)
    assert action.shape == (8,)
    assert state.dtype == torch.float32
    assert action.dtype == torch.float32


@patch("market_cgan.data.polygon.PolygonDataSource.fetch_quotes")
def test_polygon_dataset_feature_dim(mock_fetch):
    mock_fetch.return_value = [
        _make_snapshot(ts=i, bid=99.5, ask=100.5) for i in range(5)
    ]
    ds = PolygonDataset("AAPL", ["2025-01-10"], api_key="test")
    assert ds.get_feature_dim() == 42
    assert ds.get_action_dim() == 8


@patch("market_cgan.data.polygon.PolygonDataSource.fetch_quotes")
def test_polygon_dataset_multiple_dates(mock_fetch):
    def side_effect(ticker, date_str, **kw):
        return [
            _make_snapshot(ts=int(date_str[-2:]) * 1000 + i, bid=99.5, ask=100.5)
            for i in range(3)
        ]
    mock_fetch.side_effect = side_effect
    ds = PolygonDataset("AAPL", ["2025-01-10", "2025-01-11"], api_key="test")
    assert len(ds) == 5


def _make_snapshot(ts=0, bid=99.5, ask=100.5):
    from market_cgan.data.lobster import LOBSnapshot
    bp, bv, ap, av = bbo_to_multilevel(bid, 100, ask, 80)
    return LOBSnapshot(timestamp=float(ts), bid_prices=bp, bid_volumes=bv, ask_prices=ap, ask_volumes=av)
