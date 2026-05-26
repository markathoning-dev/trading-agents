import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from market_cgan.models.generator import Generator
from market_cgan.models.discriminator import Discriminator
from market_cgan.training.losses import generator_loss, discriminator_loss, gradient_penalty
from market_cgan.training.physics_loss import physics_informed_loss
from market_cgan.training.bar_physics_loss import bar_physics_loss as bar_physics_loss_fn

PHYSICS_TERM_NAMES = ["physics_spread", "physics_action_dist", "physics_side_balance", "physics_quantity"]


def train_cgan(
    generator: Generator,
    discriminator: Discriminator,
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
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    generator.to(device)
    discriminator.to(device)

    g_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    history: dict[str, list[float]] = {
        "g_loss": [], "d_loss": [], "val_g_loss": [],
        **{name: [] for name in PHYSICS_TERM_NAMES},
    }
    noise_dim = generator.noise_dim
    best_val_loss = float("inf")

    for epoch in range(epochs):
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0
        epoch_physics: dict[str, float] = {name: 0.0 for name in PHYSICS_TERM_NAMES}
        batches = 0

        generator.train()
        discriminator.train()

        for features, real_actions in train_loader:
            features = features.to(device)
            real_actions = real_actions.to(device)
            batch_size = features.size(0)

            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_outputs = generator(noise, features)
            fake_actions = torch.cat([
                fake_outputs["action_type"],
                fake_outputs["side"],
                fake_outputs["price_offset"],
                fake_outputs["quantity"],
            ], dim=1)

            real_logits = discriminator.forward_logits(real_actions, features)
            fake_logits_d = discriminator.forward_logits(fake_actions.detach(), features)

            d_loss = discriminator_loss(real_logits, fake_logits_d)
            if gp_weight > 0:
                d_loss = d_loss + gradient_penalty(discriminator, real_actions, fake_actions.detach(), features)

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            fake_logits_g = discriminator.forward_logits(fake_actions, features)
            g_loss = generator_loss(fake_logits_g)

            if physics_weight > 0:
                physics_terms = physics_informed_loss(fake_outputs, features)
                for name in PHYSICS_TERM_NAMES:
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
        for name in PHYSICS_TERM_NAMES:
            history[name].append(epoch_physics[name] / max(batches, 1))

        val_loss = 0.0
        if val_loader is not None:
            generator.eval()
            val_batches = 0
            with torch.no_grad():
                for features, real_actions in val_loader:
                    features = features.to(device)
                    real_actions = real_actions.to(device)
                    batch_size = features.size(0)
                    noise = torch.randn(batch_size, noise_dim, device=device)
                    fake_outputs = generator(noise, features)
                    fake_actions = torch.cat([
                        fake_outputs["action_type"],
                        fake_outputs["side"],
                        fake_outputs["price_offset"],
                        fake_outputs["quantity"],
                    ], dim=1)
                    fake_logits = discriminator.forward_logits(fake_actions, features)
                    v_loss = F.binary_cross_entropy_with_logits(
                        fake_logits, torch.ones_like(fake_logits)
                    )
                    val_loss += v_loss.item()
                    val_batches += 1
            val_loss /= max(val_batches, 1)
            history["val_g_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        if (epoch + 1) % log_interval == 0:
            msg = f"Epoch {epoch+1}/{epochs} | G: {avg_g:.4f} | D: {avg_d:.4f}"
            if physics_weight > 0:
                physics_msgs = [f"{k}={history[k][-1]:.4f}" for k in PHYSICS_TERM_NAMES]
                msg += " | " + " ".join(physics_msgs)
            if val_loader is not None:
                msg += f" | Val: {val_loss:.4f}"
            print(msg)

    return history


BAR_PHYSICS_TERM_NAMES = ["bar_hl_validity", "bar_volume_positivity", "bar_return_dist", "bar_vol_clustering"]


def train_cgan_bar(
    generator,
    discriminator,
    train_loader,
    val_loader=None,
    epochs: int = 100,
    lr: float = 2e-4,
    betas: tuple[float, float] = (0.5, 0.999),
    label_smoothing: float = 0.0,
    gp_weight: float = 0.0,
    physics_weight: float = 0.0,
    log_interval: int = 10,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    generator.to(device)
    discriminator.to(device)

    g_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    history: dict[str, list[float]] = {
        "g_loss": [], "d_loss": [], "val_g_loss": [],
        **{name: [] for name in BAR_PHYSICS_TERM_NAMES},
    }
    noise_dim = generator.noise_dim
    smooth_real = 1.0 - label_smoothing
    smooth_fake = label_smoothing
    best_val_loss = float("inf")

    for epoch in range(epochs):
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0
        epoch_physics: dict[str, float] = {name: 0.0 for name in BAR_PHYSICS_TERM_NAMES}
        batches = 0

        generator.train()
        discriminator.train()

        for features, real_bars in train_loader:
            features = features.to(device)
            real_bars = real_bars.to(device)
            batch_size = features.size(0)

            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_bars = generator(noise, features)

            real_logits = discriminator.forward_logits(real_bars, features)
            fake_logits_d = discriminator.forward_logits(fake_bars.detach(), features)

            d_loss = discriminator_loss(real_logits, fake_logits_d, smooth_real, smooth_fake)
            if gp_weight > 0:
                d_loss = d_loss + gradient_penalty(discriminator, real_bars, fake_bars.detach(), features)

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            fake_logits_g = discriminator.forward_logits(fake_bars, features)
            g_loss = generator_loss(fake_logits_g, smooth_real=smooth_real)

            if physics_weight > 0:
                physics_terms = bar_physics_loss_fn(fake_bars, real_bars)
                for name in BAR_PHYSICS_TERM_NAMES:
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
        for name in BAR_PHYSICS_TERM_NAMES:
            history[name].append(epoch_physics[name] / max(batches, 1))

        val_loss = 0.0
        if val_loader is not None:
            generator.eval()
            val_batches = 0
            with torch.no_grad():
                for features, real_bars in val_loader:
                    features = features.to(device)
                    real_bars = real_bars.to(device)
                    batch_size = features.size(0)
                    noise = torch.randn(batch_size, noise_dim, device=device)
                    fake_bars = generator(noise, features)
                    fake_logits = discriminator.forward_logits(fake_bars, features)
                    v_loss = F.binary_cross_entropy_with_logits(
                        fake_logits, torch.ones_like(fake_logits) * smooth_real
                    )
                    val_loss += v_loss.item()
                    val_batches += 1
            val_loss /= max(val_batches, 1)
            history["val_g_loss"].append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        if (epoch + 1) % log_interval == 0:
            msg = f"Epoch {epoch+1}/{epochs} | G: {avg_g:.4f} | D: {avg_d:.4f}"
            if physics_weight > 0:
                physics_msgs = [f"{k}={history[k][-1]:.4f}" for k in BAR_PHYSICS_TERM_NAMES]
                msg += " | " + " ".join(physics_msgs)
            if val_loader is not None:
                msg += f" | Val: {val_loss:.4f}"
            print(msg)

    return history
