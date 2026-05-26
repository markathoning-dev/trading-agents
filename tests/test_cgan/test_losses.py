import torch
from market_cgan.training.losses import generator_loss, discriminator_loss


def test_generator_loss():
    fake_logits = torch.randn(4, 1)
    loss = generator_loss(fake_logits)
    assert loss.item() > 0
    assert loss.shape == ()


def test_discriminator_loss():
    real_logits = torch.randn(4, 1)
    fake_logits = torch.randn(4, 1)
    loss = discriminator_loss(real_logits, fake_logits)
    assert loss.item() > 0
    assert loss.shape == ()


def test_discriminator_loss_real_high_fake_low():
    real_logits = torch.ones(4, 1) * 5
    fake_logits = torch.ones(4, 1) * (-5)
    loss = discriminator_loss(real_logits, fake_logits)
    assert loss.item() < 0.1


def test_discriminator_loss_real_low_fake_high():
    real_logits = torch.ones(4, 1) * (-5)
    fake_logits = torch.ones(4, 1) * 5
    loss = discriminator_loss(real_logits, fake_logits)
    assert loss.item() > 2.0


def test_generator_loss_with_feature_matching():
    fake_logits = torch.randn(4, 1)
    fake_features = torch.randn(4, 10)
    real_features = torch.randn(4, 10)
    loss = generator_loss(fake_logits, fake_features, real_features, feature_matching_weight=10.0)
    assert loss.item() > 0
