import torch
import pandas as pd
from market_pinn.models.pinn import MarketPINN
from market_pinn.training.losses import pinn_loss
from market_pinn.training.dataset import MarketDataset
from market_pinn.synthesis.generator import generate_price_paths


def test_pinn_loss():
    model = MarketPINN(input_dim=2, hidden_dim=32, num_layers=2)
    t = torch.rand(5, 1, requires_grad=True)
    s = torch.rand(5, 1, requires_grad=True)
    targets = torch.rand(5, 1)
    loss = pinn_loss(model, t, s, targets, r=0.05, sigma=0.2, w_data=1.0, w_pde=0.1)
    assert loss.item() > 0
    assert loss.requires_grad


def test_dataset():
    s = pd.Series([100.0, 101.0, 102.0])
    ds = MarketDataset(s)
    assert len(ds) == 3
    t, price, target = ds[0]
    assert t.shape == (1,)
    assert price.shape == (1,)


def test_generate_paths():
    model = MarketPINN(input_dim=2, hidden_dim=32, num_layers=2)
    paths = generate_price_paths(model, start_price=100.0, steps=10, num_paths=3)
    assert paths.shape == (3, 10)
    assert paths.min() > 0
