import numpy as np
from market_cgan.data.lobster import generate_sample_lob_data, LobsterDataset, LOBSnapshot
from market_cgan.data.features import MarketFeatureExtractor


def test_generate_sample_lob_data():
    snapshots = generate_sample_lob_data(50)
    assert len(snapshots) == 50
    for s in snapshots:
        assert s.bid_prices.shape == (10,)
        assert s.ask_prices.shape == (10,)
        assert s.bid_prices[0] < s.ask_prices[0]


def test_lobster_dataset():
    snapshots = generate_sample_lob_data(100)
    extractor = MarketFeatureExtractor()
    dataset = LobsterDataset(snapshots, extractor, seq_len=1)
    assert len(dataset) > 0
    state, action = dataset[0]
    assert state.shape == (42,)
    assert action.shape == (8,)


def test_lobster_dataset_actions():
    snapshots = generate_sample_lob_data(100)
    extractor = MarketFeatureExtractor()
    dataset = LobsterDataset(snapshots, extractor)
    for i in range(min(10, len(dataset))):
        _, action = dataset[i]
        action_type = action[:4]
        side = action[4:6]
        price_offset = action[6]
        quantity = action[7]
        assert abs(action_type.sum() - 1.0) < 1e-5
        assert abs(side.sum() - 1.0) < 1e-5
        assert -1.0 <= price_offset <= 1.0
        assert 0.0 <= quantity <= 1.0


def test_lob_snapshot_dataclass():
    snap = LOBSnapshot(
        timestamp=1.0,
        bid_prices=np.array([100.0, 99.9], dtype=np.float32),
        bid_volumes=np.array([10, 20], dtype=np.float32),
        ask_prices=np.array([100.1, 100.2], dtype=np.float32),
        ask_volumes=np.array([15, 25], dtype=np.float32),
    )
    assert float(snap.timestamp) == 1.0
    assert float(snap.bid_prices[0]) == 100.0
    assert abs(float(snap.ask_prices[0]) - 100.1) < 1e-5
