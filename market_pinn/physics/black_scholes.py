import torch


def bs_pde_residual(
    model: torch.nn.Module, t: torch.Tensor, s: torch.Tensor,
    r: float = 0.05, sigma: float = 0.2,
) -> torch.Tensor:
    x = torch.cat([t, s], dim=1)
    v = model(x)
    grads = torch.autograd.grad(v.sum(), [t, s], create_graph=True)
    dv_dt = grads[0]
    dv_ds = grads[1]
    dv_ds2 = torch.autograd.grad(dv_ds.sum(), s, create_graph=True)[0]
    return dv_dt + r * s * dv_ds + 0.5 * sigma**2 * s**2 * dv_ds2 - r * v
