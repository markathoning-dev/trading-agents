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
        {"bp": 99.5, "bs": 100, "ap": 100.5, "as": 80, "t": 1000001},
        {"bp": 99.6, "bs": 90, "ap": 100.4, "as": 70, "t": 1000002},
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
    mock_client.get_aggs.return_value = [
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
def test_fetch_aggregates_calls_get_aggs(mock_client_prop):
    from market_cgan.data.bar import Bar
    mock_client = MagicMock()
    mock_client.get_aggs.return_value = [
        MockAggBar(o=100.0, h=101.0, l=99.5, c=100.5, v=10000, vw=100.3, ts=1000001),
    ]
    source = PolygonDataSource(api_key="test_key")
    with patch.object(source, "client", mock_client):
        source.fetch_aggregates("SPUS", "2026-03-26", "2026-03-26")
    mock_client.get_aggs.assert_called_once_with("SPUS", 1, "minute", "2026-03-26", "2026-03-26")


@patch("market_cgan.data.polygon.PolygonDataSource.client")
def test_fetch_aggregates_error_propagates(mock_client_prop):
    mock_client = MagicMock()
    mock_client.get_aggs.side_effect = Exception("API error")
    source = PolygonDataSource(api_key="test_key")
    with patch.object(source, "client", mock_client):
        with pytest.raises(Exception, match="API error"):
            source.fetch_aggregates("SPUS", "bad-date", "bad-date")
