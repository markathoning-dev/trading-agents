import torch
import torch.nn as nn
import torch.nn.functional as F


def generator_loss(
    fake_logits: torch.Tensor,
    fake_features: torch.Tensor | None = None,
    real_features: torch.Tensor | None = None,
    feature_matching_weight: float = 10.0,
    smooth_real: float = 1.0,
) -> torch.Tensor:
    g_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits) * smooth_real)
    total = g_loss
    if fake_features is not None and real_features is not None:
        fm_loss = F.mse_loss(fake_features.mean(0), real_features.mean(0))
        total = total + feature_matching_weight * fm_loss
    return total


def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    smooth_real: float = 1.0,
    smooth_fake: float = 0.0,
) -> torch.Tensor:
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits) * smooth_real)
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits) + smooth_fake)
    return (real_loss + fake_loss) / 2


def gradient_penalty(
    discriminator: nn.Module,
    real_actions: torch.Tensor,
    fake_actions: torch.Tensor,
    features: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    alpha = torch.rand(real_actions.size(0), 1, device=real_actions.device)
    interpolates = alpha * real_actions + (1 - alpha) * fake_actions
    interpolates.requires_grad_(True)
    features_gp = features.detach()
    d_interpolates = discriminator.forward_logits(interpolates, features_gp)
    grad = torch.autograd.grad(
        outputs=d_interpolates.sum(),
        inputs=interpolates,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_norm = grad.view(grad.size(0), -1).norm(2, dim=1)
    gp = ((grad_norm - 1) ** 2).mean()
    return lambda_gp * gp
