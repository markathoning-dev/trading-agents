import torch
import torch.nn.functional as F
from market_cgan.data.bar import FEATURE_OPEN, FEATURE_HIGH, FEATURE_LOW, FEATURE_CLOSE, FEATURE_VOLUME


def hl_validity_loss(gen_bars: torch.Tensor) -> torch.Tensor:
    o = gen_bars[:, FEATURE_OPEN]
    h = gen_bars[:, FEATURE_HIGH]
    l = gen_bars[:, FEATURE_LOW]
    c = gen_bars[:, FEATURE_CLOSE]
    o_max = torch.max(o, c)
    o_min = torch.min(o, c)
    high_violation = torch.relu(o_max - h)
    low_violation = torch.relu(l - o_min)
    return (high_violation + low_violation).mean()


def volume_positivity_loss(gen_bars: torch.Tensor) -> torch.Tensor:
    v = gen_bars[:, FEATURE_VOLUME]
    return torch.relu(-v).mean()


def return_distribution_loss(gen_bars: torch.Tensor, real_bars: torch.Tensor) -> torch.Tensor:
    gen_returns = torch.diff(torch.log(gen_bars[:, FEATURE_CLOSE].clamp(min=1e-6)))
    real_returns = torch.diff(torch.log(real_bars[:, FEATURE_CLOSE].clamp(min=1e-6)))
    if gen_returns.numel() < 2 or real_returns.numel() < 2:
        return torch.tensor(0.0, device=gen_bars.device)
    gen_mean = gen_returns.mean()
    real_mean = real_returns.mean()
    gen_std = gen_returns.std() + 1e-8
    real_std = real_returns.std() + 1e-8
    gen_norm = (gen_returns - gen_mean) / gen_std
    real_norm = (real_returns - real_mean) / real_std
    n = min(gen_norm.numel(), real_norm.numel())
    return F.mse_loss(gen_norm[:n], real_norm[:n])


def volatility_clustering_loss(gen_bars: torch.Tensor) -> torch.Tensor:
    returns = torch.diff(torch.log(gen_bars[:, FEATURE_CLOSE].clamp(min=1e-6)))
    if returns.numel() < 4:
        return torch.tensor(0.0, device=gen_bars.device)
    abs_ret = returns.abs()
    lag1 = abs_ret[:-1]
    lag2 = abs_ret[1:]
    if lag1.numel() < 2 or lag2.numel() < 2:
        return torch.tensor(0.0, device=gen_bars.device)
    corr = torch.corrcoef(torch.stack([lag1[:len(lag2)], lag2]))[0, 1]
    return (corr - 0.2).pow(2)


def bar_physics_loss(gen_bars: torch.Tensor, real_bars: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "bar_hl_validity": hl_validity_loss(gen_bars),
        "bar_volume_positivity": volume_positivity_loss(gen_bars),
        "bar_return_dist": return_distribution_loss(gen_bars, real_bars),
        "bar_vol_clustering": volatility_clustering_loss(gen_bars),
    }