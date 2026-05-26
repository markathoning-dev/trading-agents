import torch
from market_cgan.models.discriminator import Discriminator


def test_discriminator_forward():
    disc = Discriminator(action_dim=8, feature_dim=42)
    action = torch.randn(4, 8)
    features = torch.randn(4, 42)
    out = disc(action, features)
    assert out.shape == (4, 1)
    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_discriminator_logits():
    disc = Discriminator(action_dim=8, feature_dim=42)
    action = torch.randn(3, 8)
    features = torch.randn(3, 42)
    logits = disc.forward_logits(action, features)
    assert logits.shape == (3, 1)


def test_discriminator_real_vs_fake():
    disc = Discriminator(action_dim=8, feature_dim=42)
    features = torch.randn(4, 42)
    action = torch.randn(4, 8)
    with torch.no_grad():
        p_real = disc(action, features)
        p_random = disc(torch.randn(4, 8), features)
    assert p_real.shape == p_random.shape


def test_discriminator_gradients():
    disc = Discriminator(action_dim=8, feature_dim=42)
    action = torch.randn(2, 8)
    features = torch.randn(2, 42)
    loss = disc.forward_logits(action, features).mean()
    loss.backward()
    has_grad = False
    for p in disc.parameters():
        if p.grad is not None:
            has_grad = True
            break
    assert has_grad
