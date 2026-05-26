import torch
from market_pinn.physics.black_scholes import bs_pde_residual


def pinn_loss(model, t, s, targets, r=0.05, sigma=0.2, w_data=1.0, w_pde=0.1):
    x = torch.cat([t, s], dim=1)
    pred = model(x)
    data_loss = torch.nn.functional.mse_loss(pred, targets)
    pde_res = bs_pde_residual(model, t, s, r, sigma)
    pde_loss = torch.mean(pde_res**2)
    return w_data * data_loss + w_pde * pde_loss
