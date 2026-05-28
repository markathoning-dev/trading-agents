from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from market_cgan.training.losses import generator_loss, discriminator_loss, gradient_penalty
from market_cgan.training.physics_loss import physics_informed_loss
from market_cgan.training.bar_physics_loss import bar_physics_loss as bar_physics_loss_fn


PHYSICS_TERM_NAMES = ["physics_spread", "physics_action_dist", "physics_side_balance", "physics_quantity"]
BAR_PHYSICS_TERM_NAMES = ["bar_hl_validity", "bar_volume_positivity", "bar_return_dist", "bar_vol_clustering"]


def _lob_assemble_fake(generated_output: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([
        generated_output["action_type"],
        generated_output["side"],
        generated_output["price_offset"],
        generated_output["quantity"],
    ], dim=1)


def _bar_assemble_fake(generated_output: torch.Tensor) -> torch.Tensor:
    return generated_output


def _lob_physics_loss(generated_output, features, _real_data):
    return physics_informed_loss(generated_output, features)


def _bar_physics_loss(generated_output, _features, real_data):
    return bar_physics_loss_fn(generated_output, real_data)


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 100,
    lr: float = 2e-4,
    betas: tuple[float, float] = (0.5, 0.999),
    label_smoothing: float = 0.0,
    gp_weight: float = 0.0,
    physics_weight: float = 0.0,
    physics_term_names: list[str] | None = None,
    assemble_fake: Callable = _bar_assemble_fake,
    compute_physics_loss: Callable | None = None,
    log_interval: int = 10,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    generator.to(device)
    discriminator.to(device)

    g_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    term_names = physics_term_names or []
    smooth_real = 1.0 - label_smoothing
    smooth_fake = label_smoothing

    history: dict[str, list[float]] = {
        "g_loss": [], "d_loss": [], "val_g_loss": [],
        **{name: [] for name in term_names},
    }
    noise_dim = generator.noise_dim

    for epoch in range(epochs):
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0
        epoch_physics: dict[str, float] = {name: 0.0 for name in term_names}
        batches = 0

        generator.train()
        discriminator.train()

        for features, real_data in train_loader:
            features = features.to(device)
            real_data = real_data.to(device)
            batch_size = features.size(0)

            noise = torch.randn(batch_size, noise_dim, device=device)
            generated_output = generator(noise, features)
            fake_data = assemble_fake(generated_output)

            real_logits = discriminator.forward_logits(real_data, features)
            fake_logits_d = discriminator.forward_logits(fake_data.detach(), features)

            d_loss = discriminator_loss(real_logits, fake_logits_d, smooth_real, smooth_fake)
            if gp_weight > 0:
                d_loss = d_loss + gradient_penalty(discriminator, real_data, fake_data.detach(), features)

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            fake_logits_g = discriminator.forward_logits(fake_data, features)
            g_loss = generator_loss(fake_logits_g, smooth_real=smooth_real)

            if physics_weight > 0 and compute_physics_loss is not None:
                physics_terms = compute_physics_loss(generated_output, features, real_data)
                for name in term_names:
                    term = physics_terms[name]
                    g_loss = g_loss + physics_weight * term
                    epoch_physics[name] += term.item()

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            g_epoch_loss += g_loss.item()
            d_epoch_loss += d_loss.item()
            batches += 1

        avg_g = g_epoch_loss / max(batches, 1)
        avg_d = d_epoch_loss / max(batches, 1)
        history["g_loss"].append(avg_g)
        history["d_loss"].append(avg_d)
        for name in term_names:
            history[name].append(epoch_physics[name] / max(batches, 1))

        val_loss = 0.0
        if val_loader is not None:
            generator.eval()
            val_batches = 0
            with torch.no_grad():
                for features, real_data in val_loader:
                    features = features.to(device)
                    real_data = real_data.to(device)
                    batch_size = features.size(0)
                    noise = torch.randn(batch_size, noise_dim, device=device)
                    generated_output = generator(noise, features)
                    fake_data = assemble_fake(generated_output)
                    fake_logits = discriminator.forward_logits(fake_data, features)
                    v_loss = F.binary_cross_entropy_with_logits(
                        fake_logits, torch.ones_like(fake_logits) * smooth_real
                    )
                    val_loss += v_loss.item()
                    val_batches += 1
            val_loss /= max(val_batches, 1)
            history["val_g_loss"].append(val_loss)

        if (epoch + 1) % log_interval == 0:
            msg = f"Epoch {epoch+1}/{epochs} | G: {avg_g:.4f} | D: {avg_d:.4f}"
            if physics_weight > 0 and term_names:
                physics_msgs = [f"{k}={history[k][-1]:.4f}" for k in term_names]
                msg += " | " + " ".join(physics_msgs)
            if val_loader is not None:
                msg += f" | Val: {val_loss:.4f}"
            print(msg)

    return history


def train_cgan(
    generator,
    discriminator,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 100,
    lr: float = 2e-4,
    betas: tuple[float, float] = (0.5, 0.999),
    label_smoothing: float = 0.1,
    gp_weight: float = 0.0,
    physics_weight: float = 0.0,
    log_interval: int = 10,
    device: str | None = None,
):
    return train_gan(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        betas=betas,
        label_smoothing=label_smoothing,
        gp_weight=gp_weight,
        physics_weight=physics_weight,
        physics_term_names=PHYSICS_TERM_NAMES,
        assemble_fake=_lob_assemble_fake,
        compute_physics_loss=_lob_physics_loss if physics_weight > 0 else None,
        log_interval=log_interval,
        device=device,
    )


def train_cgan_bar(
    generator,
    discriminator,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 100,
    lr: float = 2e-4,
    betas: tuple[float, float] = (0.5, 0.999),
    label_smoothing: float = 0.0,
    gp_weight: float = 0.0,
    physics_weight: float = 0.0,
    log_interval: int = 10,
    device: str | None = None,
):
    return train_gan(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        betas=betas,
        label_smoothing=label_smoothing,
        gp_weight=gp_weight,
        physics_weight=physics_weight,
        physics_term_names=BAR_PHYSICS_TERM_NAMES,
        assemble_fake=_bar_assemble_fake,
        compute_physics_loss=_bar_physics_loss if physics_weight > 0 else None,
        log_interval=log_interval,
        device=device,
    )
