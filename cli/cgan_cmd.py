import os
from pathlib import Path

import typer
import torch
from torch.utils.data import DataLoader, random_split

from market_cgan.models.generator import Generator
from market_cgan.models.discriminator import Discriminator
from market_cgan.training.trainer import train_cgan
from market_cgan.data.lobster import generate_sample_lob_data, LobsterDataset, LOBSnapshot
from market_cgan.data.features import MarketFeatureExtractor

app = typer.Typer()

MODELS_DIR = Path("models/cgan")
FEATURE_DIM = 42
ACTION_DIM = 8
NOISE_DIM = 64


def _build_dataset(data_source: str, polygon_ticker: str, polygon_dates: list[str], seq_len: int = 1):
    extractor = MarketFeatureExtractor()
    if data_source == "synthetic":
        snaps = generate_sample_lob_data(2000)
        return LobsterDataset(snaps, extractor, seq_len=seq_len)
    elif data_source == "polygon":
        from market_cgan.data.polygon import PolygonDataset
        api_key = os.environ.get("POLYGON_API_KEY", "")
        if not api_key:
            typer.echo("ERROR: POLYGON_API_KEY not set", err=True)
            raise typer.Exit(1)
        return PolygonDataset(polygon_ticker, polygon_dates, api_key=api_key, seq_len=seq_len)
    else:
        typer.echo(f"ERROR: unknown data source '{data_source}'", err=True)
        raise typer.Exit(1)


@app.command()
def train(
    epochs: int = typer.Option(100, "--epochs", "-e"),
    batch_size: int = typer.Option(128, "--batch-size", "-b"),
    lr: float = typer.Option(2e-4, "--lr"),
    gp_weight: float = typer.Option(0.0, "--gp-weight"),
    physics_weight: float = typer.Option(0.0, "--physics-weight"),
    data_source: str = typer.Option("synthetic", "--data-source", help="synthetic, lobster, or polygon"),
    polygon_ticker: str = typer.Option("AAPL", "--polygon-ticker"),
    polygon_dates: str = typer.Option("2025-01-10", "--polygon-dates", help="comma-separated dates"),
    log_interval: int = typer.Option(10, "--log-interval"),
    val_split: float = typer.Option(0.1, "--val-split", help="fraction of data for validation"),
    out_dir: Path = typer.Option(MODELS_DIR, "--out-dir", "-o"),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    typer.echo(f"Training CGAN | device={device} | data={data_source} | epochs={epochs} "
               f"batch={batch_size} lr={lr} gp={gp_weight} physics={physics_weight}")

    dataset = _build_dataset(data_source, polygon_ticker, polygon_dates.split(","))

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_size > 0 else None

    gen = Generator(noise_dim=NOISE_DIM, feature_dim=FEATURE_DIM)
    disc = Discriminator(action_dim=ACTION_DIM, feature_dim=FEATURE_DIM)

    typer.echo(f"Generator params: {sum(p.numel() for p in gen.parameters()):,}")
    typer.echo(f"Discriminator params: {sum(p.numel() for p in disc.parameters()):,}")

    history = train_cgan(
        gen, disc,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        gp_weight=gp_weight,
        physics_weight=physics_weight,
        log_interval=log_interval,
        device=device,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(gen.state_dict(), out_dir / "generator.pt")
    torch.save(disc.state_dict(), out_dir / "discriminator.pt")
    typer.echo(f"Models saved to {out_dir.resolve()}")
    typer.echo(f"Final G loss: {history['g_loss'][-1]:.4f} | D loss: {history['d_loss'][-1]:.4f}")


@app.command()
def generate(
    model_path: Path = typer.Option("models/cgan/generator.pt", "--model", "-m"),
    steps: int = typer.Option(252, "--steps", "-n"),
    seed: int = typer.Option(42, "--seed"),
    out: Path = typer.Option("generated_lob.csv", "--out", "-o"),
):
    torch.manual_seed(seed)
    gen = Generator(noise_dim=NOISE_DIM, feature_dim=FEATURE_DIM)
    gen.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    gen.eval()

    dummy_features = torch.zeros(1, FEATURE_DIM)
    all_actions = []
    features = torch.zeros(1, FEATURE_DIM)
    for _ in range(steps):
        noise = torch.randn(1, NOISE_DIM)
        with torch.no_grad():
            out_dict = gen.sample(noise, features)
        all_actions.append({
            "action_type": int(out_dict["action_type"][0].item()),
            "side": int(out_dict["side"][0].item()),
            "price_offset": float(out_dict["price_offset"][0].item()),
            "quantity": float(out_dict["quantity"][0].item()),
        })

    import csv
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "action_type", "side", "price_offset", "quantity"])
        w.writeheader()
        for i, a in enumerate(all_actions):
            w.writerow({"step": i, **a})
    typer.echo(f"Generated {steps} actions -> {out}")


@app.command()
def simulate(
    model_path: Path = typer.Option("models/cgan/generator.pt", "--model", "-m"),
    steps: int = typer.Option(100, "--steps", "-n"),
    ticker: str = typer.Option("CGAN", "--ticker"),
):
    from market_cgan.simulation.exchange import LOBExchange
    from market_cgan.data.features import MarketFeatureExtractor
    from market_cgan.data.lobster import LOBSnapshot

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = Generator(noise_dim=NOISE_DIM, feature_dim=FEATURE_DIM)
    gen.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    gen.to(device)
    gen.eval()

    extractor = MarketFeatureExtractor()
    exchange = LOBExchange(feature_extractor=extractor)

    typer.echo(f"Running LOB simulation ({steps} steps) with model {model_path}...")
    for i in range(steps):
        features_t = torch.from_numpy(exchange.get_features()).unsqueeze(0).to(device)
        noise = torch.randn(1, NOISE_DIM, device=device)
        with torch.no_grad():
            out = gen(noise, features_t)

        at = torch.multinomial(out["action_type"], 1).item()
        sd = torch.multinomial(out["side"], 1).item()
        po = out["price_offset"].item()
        qty = out["quantity"].item()

        state = exchange.get_lob_state()
        trades = exchange.process_action(at, sd, po, qty, state.mid_price)
        if i % 20 == 0:
            typer.echo(f"  step {i}: action={at} side={sd} mid={state.mid_price:.2f} trades={len(trades)}")
