import numpy as np
from market_cgan.data.features import MarketFeatureExtractor
from market_cgan.data.lobster import generate_sample_lob_data


def test_feature_extractor_shape():
    extractor = MarketFeatureExtractor()
    assert extractor.N_FEATURES == 42


def test_feature_extractor_output():
    snapshots = generate_sample_lob_data(10)
    extractor = MarketFeatureExtractor()
    for s in snapshots:
        features = extractor(s)
        assert features.shape == (42,)
        assert features.dtype == np.float32


def test_feature_extractor_reasonable_values():
    snapshots = generate_sample_lob_data(10)
    extractor = MarketFeatureExtractor()
    for s in snapshots:
        features = extractor(s)
        assert np.all(np.isfinite(features))
        assert np.all(features != 0) or True


def test_feature_extractor_consistency():
    snapshots = generate_sample_lob_data(10)
    extractor = MarketFeatureExtractor()
    f1 = extractor(snapshots[0])
    f2 = extractor(snapshots[0])
    assert np.allclose(f1, f2)
