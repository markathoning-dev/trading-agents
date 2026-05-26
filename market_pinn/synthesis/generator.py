import torch
import numpy as np


@torch.no_grad()
def generate_price_paths(model, start_price=100.0, steps=252, num_paths=10):
    model.eval()
    t_vals = torch.linspace(0, 1, steps)[1:]
    s = torch.full((num_paths, 1), start_price)
    paths = [s.clone()]
    for i in range(steps - 1):
        tin = torch.full((num_paths, 1), t_vals[i].item())
        x = torch.cat([tin, s], dim=1)
        pred = model(x)
        noise = torch.randn(num_paths, 1) * 0.01 * pred.abs().mean()
        s = s + pred + noise
        s = s.clamp(min=1e-6)
        paths.append(s.clone())
    return torch.stack(paths, dim=1).squeeze(-1).numpy()
