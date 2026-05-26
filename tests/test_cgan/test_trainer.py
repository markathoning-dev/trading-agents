import torch
from torch.utils.data import DataLoader, TensorDataset
from market_cgan.models.generator import Generator
from market_cgan.models.discriminator import Discriminator
from market_cgan.training.trainer import train_cgan, train_cgan_bar
from market_cgan.training.losses import gradient_penalty
from market_cgan.models.bar_generator import BarGenerator
from market_cgan.models.bar_discriminator import BarDiscriminator


def test_train_cgan_overfit():
    gen = Generator(noise_dim=8, feature_dim=10)
    disc = Discriminator(action_dim=8, feature_dim=10)
    features = torch.randn(64, 10)
    actions = torch.randn(64, 8)
    dataset = TensorDataset(features, actions)
    loader = DataLoader(dataset, batch_size=16)
    history = train_cgan(gen, disc, loader, epochs=5, lr=1e-3, log_interval=100)
    assert "g_loss" in history
    assert "d_loss" in history
    assert len(history["g_loss"]) == 5
    assert len(history["d_loss"]) == 5


def test_train_cgan_with_validation():
    gen = Generator(noise_dim=8, feature_dim=10)
    disc = Discriminator(action_dim=8, feature_dim=10)
    features = torch.randn(32, 10)
    actions = torch.randn(32, 8)
    dataset = TensorDataset(features, actions)
    train_loader = DataLoader(dataset, batch_size=8)
    val_loader = DataLoader(dataset, batch_size=8)
    history = train_cgan(gen, disc, train_loader, val_loader, epochs=3, lr=1e-3, log_interval=100)
    assert "val_g_loss" in history
    assert len(history["val_g_loss"]) == 3


def test_gradient_penalty():
    disc = Discriminator(action_dim=8, feature_dim=10)
    real = torch.randn(4, 8, requires_grad=True)
    fake = torch.randn(4, 8, requires_grad=True)
    features = torch.randn(4, 10)
    gp = gradient_penalty(disc, real, fake, features, lambda_gp=10.0)
    assert gp.item() >= 0
    assert gp.shape == ()


def test_train_cgan_bar_overfit():
    gen = BarGenerator(noise_dim=8, feature_dim=6, ref_price=100.0)
    disc = BarDiscriminator(bar_dim=6, feature_dim=6)
    features = torch.randn(64, 6)
    bars = torch.randn(64, 6)
    dataset = TensorDataset(features, bars)
    loader = DataLoader(dataset, batch_size=16)
    history = train_cgan_bar(gen, disc, loader, epochs=5, lr=1e-3, log_interval=100)
    assert "g_loss" in history
    assert "d_loss" in history
    assert len(history["g_loss"]) == 5
    assert len(history["d_loss"]) == 5


def test_train_cgan_bar_with_validation():
    gen = BarGenerator(noise_dim=8, feature_dim=6, ref_price=100.0)
    disc = BarDiscriminator(bar_dim=6, feature_dim=6)
    features = torch.randn(32, 6)
    bars = torch.randn(32, 6)
    dataset = TensorDataset(features, bars)
    train_loader = DataLoader(dataset, batch_size=8)
    val_loader = DataLoader(dataset, batch_size=8)
    history = train_cgan_bar(gen, disc, train_loader, val_loader, epochs=3, lr=1e-3, log_interval=100)
    assert "val_g_loss" in history
    assert len(history["val_g_loss"]) == 3


def test_train_cgan_bar_with_physics():
    gen = BarGenerator(noise_dim=8, feature_dim=6, ref_price=100.0)
    disc = BarDiscriminator(bar_dim=6, feature_dim=6)
    features = torch.randn(32, 6)
    bars = torch.randn(32, 6)
    dataset = TensorDataset(features, bars)
    loader = DataLoader(dataset, batch_size=8)
    history = train_cgan_bar(gen, disc, loader, epochs=3, lr=1e-3, physics_weight=0.5, log_interval=100)
    assert "bar_hl_validity" in history
    assert "bar_volume_positivity" in history
