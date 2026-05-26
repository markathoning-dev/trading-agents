import torch
from torch.utils.data import DataLoader
from market_cgan.data.lobster import generate_sample_lob_data, LobsterDataset
from market_cgan.data.features import MarketFeatureExtractor
from market_cgan.models.generator import Generator
from market_cgan.models.discriminator import Discriminator
from market_cgan.training.trainer import train_cgan


def test_cgan_end_to_end():
    snapshots = generate_sample_lob_data(200)
    extractor = MarketFeatureExtractor()
    dataset = LobsterDataset(snapshots, extractor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    gen = Generator(noise_dim=16, feature_dim=42)
    disc = Discriminator(action_dim=8, feature_dim=42)

    history = train_cgan(gen, disc, loader, epochs=3, lr=1e-3, log_interval=100)

    assert len(history["g_loss"]) == 3
    assert len(history["d_loss"]) == 3

    gen.eval()
    with torch.no_grad():
        features = torch.randn(4, 42)
        noise = torch.randn(4, 16)
        out = gen.sample(noise, features)

    assert out["action_type"].shape == (4,)
    assert out["side"].shape == (4,)
    assert out["price_offset"].shape == (4,)
    assert out["quantity"].shape == (4,)


def test_cgan_generator_produces_valid_actions():
    snapshots = generate_sample_lob_data(100)
    extractor = MarketFeatureExtractor()
    dataset = LobsterDataset(snapshots, extractor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    gen = Generator(noise_dim=16, feature_dim=42)
    disc = Discriminator(action_dim=8, feature_dim=42)

    train_cgan(gen, disc, loader, epochs=2, lr=1e-3, log_interval=100)

    gen.eval()
    with torch.no_grad():
        features = torch.randn(10, 42)
        noise = torch.randn(10, 16)
        out = gen.sample(noise, features)

    for i in range(10):
        assert 0 <= out["action_type"][i].item() <= 3
        assert 0 <= out["side"][i].item() <= 1
        assert -1.0 <= out["price_offset"][i].item() <= 1.0
        assert 0.0 <= out["quantity"][i].item() <= 1.0
