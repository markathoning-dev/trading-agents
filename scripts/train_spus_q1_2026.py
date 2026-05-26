"""
Train OHLCV bar CGAN on SPUS Q1 2026 minute aggregate data from Polygon.io.

Usage:
    python scripts/train_spus_q1_2026.py [--epochs 100] [--physics-weight 0.1]
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, random_split
from market_cgan.data.bar import BarDataset
from market_cgan.data.polygon import PolygonDataSource
from market_cgan.models.bar_generator import BarGenerator
from market_cgan.models.bar_discriminator import BarDiscriminator
from market_cgan.training.trainer import train_cgan_bar

TICKER = "SPUS"
DATES = [
    "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
    "2026-01-09", "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15",
    "2026-01-16", "2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23",
    "2026-01-26", "2026-01-27", "2026-01-28", "2026-01-29", "2026-01-30",
    "2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05", "2026-02-06",
    "2026-02-09", "2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13",
    "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20", "2026-02-23",
    "2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27",
    "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06",
    "2026-03-09", "2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13",
    "2026-03-16", "2026-03-17", "2026-03-18", "2026-03-19", "2026-03-20",
    "2026-03-23", "2026-03-24", "2026-03-25", "2026-03-26", "2026-03-27",
    "2026-03-30", "2026-03-31",
]

OUT_DIR = Path("models/cgan/spus_q1_2026")
NOISE_DIM = 64
FEATURE_DIM = 6
BATCH_SIZE = 128
EPOCHS = 100
LR = 2e-4
PHYSICS_WEIGHT = 0.1
VAL_SPLIT = 0.1


def main():
    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY environment variable not set")
        sys.exit(1)

    print(f"Fetching {TICKER} minute aggregates for {len(DATES)} trading days...")
    source = PolygonDataSource(api_key)
    all_bars = []
    for d in DATES:
        bars = source.fetch_aggregates(TICKER, d, d)
        all_bars.extend(bars)
        print(f"  {d}: {len(bars)} bars")

    print(f"Total bars: {len(all_bars)}")
    if len(all_bars) < 10:
        print("ERROR: too few bars. Check API key and date range.")
        sys.exit(1)

    dataset = BarDataset(all_bars, seq_len=1)
    ref_price = dataset.ref_price
    print(f"Reference price: {ref_price:.2f}")
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE) if val_size > 0 else None

    gen = BarGenerator(noise_dim=NOISE_DIM, feature_dim=FEATURE_DIM, ref_price=ref_price)
    disc = BarDiscriminator(bar_dim=FEATURE_DIM, feature_dim=FEATURE_DIM)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    print(f"Generator params: {sum(p.numel() for p in gen.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in disc.parameters()):,}")

    history = train_cgan_bar(
        gen, disc,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        physics_weight=PHYSICS_WEIGHT,
        log_interval=10,
        device=device,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(gen.state_dict(), OUT_DIR / "generator.pt")
    torch.save(disc.state_dict(), OUT_DIR / "discriminator.pt")
    print(f"Models saved to {OUT_DIR.resolve()}")
    print(f"Final G loss: {history['g_loss'][-1]:.4f} | D loss: {history['d_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()