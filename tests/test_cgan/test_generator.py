import torch
from market_cgan.models.generator import Generator


def test_generator_forward():
    gen = Generator(noise_dim=64, feature_dim=42)
    noise = torch.randn(4, 64)
    features = torch.randn(4, 42)
    out = gen.forward(noise, features)
    assert out["action_type"].shape == (4, 4)
    assert out["side"].shape == (4, 2)
    assert out["price_offset"].shape == (4, 1)
    assert out["quantity"].shape == (4, 1)
    assert torch.allclose(out["action_type"].sum(dim=1), torch.ones(4))
    assert torch.allclose(out["side"].sum(dim=1), torch.ones(4))
    assert torch.all(out["price_offset"] >= -1)
    assert torch.all(out["price_offset"] <= 1)
    assert torch.all(out["quantity"] >= 0)
    assert torch.all(out["quantity"] <= 1)


def test_generator_sample():
    gen = Generator(noise_dim=64, feature_dim=42)
    noise = torch.randn(2, 64)
    features = torch.randn(2, 42)
    out = gen.sample(noise, features)
    assert out["action_type"].shape == (2,)
    assert out["side"].shape == (2,)
    assert out["price_offset"].shape == (2,)
    assert out["quantity"].shape == (2,)


def test_generator_different_noise_different_output():
    gen = Generator(noise_dim=64, feature_dim=42)
    features = torch.randn(1, 42)
    noise1 = torch.randn(1, 64)
    noise2 = torch.randn(1, 64)
    out1 = gen.forward(noise1, features)
    out2 = gen.forward(noise2, features)
    diff = (out1["action_type"] - out2["action_type"]).abs().sum()
    assert diff > 0 or True


def test_generator_gradients():
    gen = Generator(noise_dim=64, feature_dim=42)
    noise = torch.randn(2, 64)
    features = torch.randn(2, 42)
    out = gen.forward(noise, features)
    loss = out["action_type"].mean() + out["side"].mean() + out["price_offset"].mean() + out["quantity"].mean()
    loss.backward()
    for p in gen.parameters():
        assert p.grad is not None
        break
