import torch
from market_pinn.models.pinn import MarketPINN
from market_pinn.physics.black_scholes import bs_pde_residual


def test_bs_residual_shape():
    model = MarketPINN(input_dim=2, hidden_dim=32, num_layers=2)
    t = torch.rand(5, 1, requires_grad=True)
    s = torch.rand(5, 1, requires_grad=True)
    res = bs_pde_residual(model, t, s, r=0.05, sigma=0.2)
    assert res.shape == (5, 1)


def test_bs_residual_finite():
    model = MarketPINN(input_dim=2, hidden_dim=32, num_layers=2)
    t = torch.rand(3, 1, requires_grad=True)
    s = torch.rand(3, 1, requires_grad=True)
    res = bs_pde_residual(model, t, s)
    assert torch.isfinite(res).all()
