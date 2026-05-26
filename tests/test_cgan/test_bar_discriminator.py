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