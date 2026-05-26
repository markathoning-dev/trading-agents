import torch
from market_pinn.models.pinn import MarketPINN


def test_pinn_forward():
    model = MarketPINN(input_dim=2, hidden_dim=32, num_layers=2)
    x = torch.randn(5, 2)
    out = model(x)
    assert out.shape == (5, 1)


def test_pinn_gradients():
    model = MarketPINN(input_dim=2, hidden_dim=32, num_layers=2)
    x = torch.randn(3, 2, requires_grad=True)
    v = model(x)
    grad = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    assert grad.shape == (3, 2)
