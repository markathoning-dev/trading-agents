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
