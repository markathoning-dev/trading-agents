import typer
from cli import backtest_cmd, pinn_cmd, serve_cmd, cgan_cmd, cards_cmd

app = typer.Typer()
app.add_typer(backtest_cmd.app, name="backtest", help="Run and compare backtests")
app.add_typer(pinn_cmd.app, name="pinn", help="Train PINN and generate market data")
app.add_typer(cgan_cmd.app, name="cgan", help="Train CGAN world agent and simulate LOB markets")
app.add_typer(serve_cmd.app, name="serve", help="Start web dashboard")
app.add_typer(cards_cmd.app, name="cards", help="Manage strategy cards and decks")

if __name__ == "__main__":
    app()
