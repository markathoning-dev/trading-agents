import typer
app = typer.Typer()

@app.command()
def train(
    symbol: str = typer.Option("AAPL", "--symbol"),
    epochs: int = typer.Option(100, "--epochs"),
):
    typer.echo(f"Training PINN on {symbol} for {epochs} epochs...")

@app.command()
def generate(
    model_id: int = typer.Option(..., "--model-id"),
    paths: int = typer.Option(10, "--paths"),
    steps: int = typer.Option(252, "--steps"),
):
    typer.echo(f"Generating {paths} paths of {steps} steps from model {model_id}...")
