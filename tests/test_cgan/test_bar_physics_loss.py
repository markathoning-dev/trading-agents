import torch
from market_cgan.training.bar_physics_loss import (
    hl_validity_loss,
    volume_positivity_loss,
    return_distribution_loss,
    volatility_clustering_loss,
    bar_physics_loss,
)


def test_hl_validity_loss_pass():
    gen_bars = torch.tensor([[100.0, 101.0, 99.0, 100.5, 10000, 100.3]], dtype=torch.float32)
    loss = hl_validity_loss(gen_bars)
    assert loss.item() == 0.0


def test_hl_validity_loss_violation_high():
    gen_bars = torch.tensor([[100.0, 99.0, 99.0, 100.5, 10000, 100.3]], dtype=torch.float32)
    loss = hl_validity_loss(gen_bars)
    assert loss.item() > 0


def test_hl_validity_loss_violation_low():
    gen_bars = torch.tensor([[100.0, 101.0, 101.0, 100.5, 10000, 100.3]], dtype=torch.float32)
    loss = hl_validity_loss(gen_bars)
    assert loss.item() > 0


def test_volume_positivity_loss_pass():
    gen_bars = torch.tensor([[100.0, 101.0, 99.0, 100.5, 10000, 100.3]], dtype=torch.float32)
    loss = volume_positivity_loss(gen_bars)
    assert loss.item() == 0.0


def test_volume_positivity_loss_violation():
    gen_bars = torch.tensor([[100.0, 101.0, 99.0, 100.5, -100, 100.3]], dtype=torch.float32)
    loss = volume_positivity_loss(gen_bars)
    assert loss.item() > 0


def test_return_distribution_loss():
    gen_bars = torch.randn(10, 6)
    real_bars = torch.randn(10, 6)
    loss = return_distribution_loss(gen_bars, real_bars)
    assert loss.item() >= 0
    assert loss.shape == ()


def test_volatility_clustering_loss():
    gen_bars = torch.randn(20, 6)
    loss = volatility_clustering_loss(gen_bars)
    assert loss.item() >= 0
    assert loss.shape == ()


def test_bar_physics_loss_returns_dict():
    gen_bars = torch.randn(10, 6)
    real_bars = torch.randn(10, 6)
    terms = bar_physics_loss(gen_bars, real_bars)
    assert isinstance(terms, dict)
    assert "bar_hl_validity" in terms
    assert "bar_volume_positivity" in terms
    assert "bar_return_dist" in terms
    assert "bar_vol_clustering" in terms
    for v in terms.values():
        assert isinstance(v, torch.Tensor)
        assert v.shape == ()