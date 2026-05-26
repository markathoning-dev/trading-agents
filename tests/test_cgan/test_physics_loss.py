import torch
from market_cgan.training.physics_loss import (
    spread_respecting_price_loss,
    action_distribution_loss,
    side_balance_loss,
    quantity_matching_loss,
    physics_informed_loss,
)


def _make_fake_outputs(batch_size=4):
    return {
        "price_offset": torch.tanh(torch.randn(batch_size, 1)),
        "action_type": torch.softmax(torch.randn(batch_size, 4), dim=1),
        "side": torch.softmax(torch.randn(batch_size, 2), dim=1),
        "quantity": torch.sigmoid(torch.randn(batch_size, 1)),
    }


def _make_features(batch_size=4):
    f = torch.randn(batch_size, 42)
    f[:, 0] = torch.abs(f[:, 0]) + 0.5
    f[:, 1] = torch.abs(f[:, 1]) + 0.001
    return f


def test_spread_respecting_price_loss_passes_good_offsets():
    price_offset = torch.zeros(4, 1)
    action_type = torch.zeros(4, 4)
    action_type[:, 1] = 1.0
    features = torch.ones(4, 42)
    features[:, 0] = 1.0
    features[:, 1] = 0.02
    loss = spread_respecting_price_loss(price_offset, action_type, features)
    assert loss.item() == 0.0


def test_spread_respecting_price_loss_penalizes_wide_offsets():
    price_offset = torch.full((4, 1), 0.5)
    action_type = torch.zeros(4, 4)
    action_type[:, 1] = 1.0
    features = torch.ones(4, 42)
    features[:, 0] = 1.0
    features[:, 1] = 0.002
    loss = spread_respecting_price_loss(price_offset, action_type, features)
    assert loss.item() > 0


def test_spread_respecting_price_loss_ignores_non_limit():
    price_offset = torch.full((4, 1), 0.5)
    action_type = torch.zeros(4, 4)
    action_type[:, 0] = 1.0
    features = torch.ones(4, 42)
    features[:, 0] = 1.0
    features[:, 1] = 0.002
    loss = spread_respecting_price_loss(price_offset, action_type, features)
    assert loss.item() == 0.0


def test_action_distribution_loss_perfect_match():
    B = 64
    action_type = torch.zeros(B, 4)
    action_type[:, 0] = 0.3
    action_type[:, 1] = 0.3
    action_type[:, 2] = 0.3
    action_type[:, 3] = 0.1
    loss = action_distribution_loss(action_type)
    assert loss.item() < 1e-6


def test_action_distribution_loss_mismatch():
    B = 64
    action_type = torch.zeros(B, 4)
    action_type[:, 0] = 1.0
    loss = action_distribution_loss(action_type)
    assert loss.item() > 0


def test_side_balance_loss_perfect_match():
    B = 64
    side = torch.zeros(B, 2)
    side[:, 0] = 0.5
    side[:, 1] = 0.5
    loss = side_balance_loss(side)
    assert loss.item() < 1e-6


def test_side_balance_loss_imbalanced():
    B = 64
    side = torch.zeros(B, 2)
    side[:, 0] = 1.0
    side[:, 1] = 0.0
    loss = side_balance_loss(side)
    assert loss.item() > 0


def test_quantity_matching_loss_perfect_match():
    B = 64
    quantity = torch.full((B, 1), 0.15)
    loss = quantity_matching_loss(quantity)
    assert loss.item() < 1e-6


def test_quantity_matching_loss_mismatch():
    B = 64
    quantity = torch.full((B, 1), 0.5)
    loss = quantity_matching_loss(quantity)
    assert loss.item() > 0


def test_physics_informed_loss_returns_all_terms():
    B = 4
    fake_outputs = _make_fake_outputs(B)
    features = _make_features(B)
    terms = physics_informed_loss(fake_outputs, features)
    expected = {"physics_spread", "physics_action_dist", "physics_side_balance", "physics_quantity"}
    assert set(terms.keys()) == expected
    for v in terms.values():
        assert v.shape == ()
        assert v.item() >= 0
