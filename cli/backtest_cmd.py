import typer
from trading_agent.backtest.engine import backtest_agent
from trading_agent.backtest.parallel import parallel_backtest, BacktestConfig
from trading_agent.core.reward import multicomponent_reward, aggressive_reward
from trading_agent.models.gateway import KiloGateway
from trading_agent.models.mock import MockGateway
from trading_agent.config.settings import settings
from trading_agent.market.polygon_market import fetch_prices

app = typer.Typer()


def _resolve_llm(model: str):
    try:
        llm = KiloGateway(model).get_langchain_llm()
        from langchain_core.messages import HumanMessage
        llm.invoke([HumanMessage(content="test")])
        return llm
    except Exception:
        return MockGateway().get_langchain_llm()


@app.command()
def run(
    model: str = typer.Option(settings.default_model, "--model", "-m"),
    symbol: str = typer.Option("AAPL", "--symbol", "-s"),
    max_steps: int = typer.Option(settings.max_steps, "--steps", "-n"),
):
    prices = fetch_prices(symbol)
    llm = _resolve_llm(model)
    metrics = backtest_agent(prices, llm=llm, max_steps=min(max_steps, len(prices)))
    for k, v in metrics.items():
        typer.echo(f"{k}: {v}")


@app.command()
def compare(
    models: str = typer.Option("openai/gpt-4o-mini,openai/gpt-3.5-turbo", "--models", "-m"),
    symbol: str = typer.Option("AAPL", "--symbol", "-s"),
    parallel_workers: int = typer.Option(4, "--workers", "-w", help="Max parallel workers"),
):
    model_list = [m.strip() for m in models.split(",")]
    prices = fetch_prices(symbol)

    configs = [
        BacktestConfig(price_series=prices, llm=_resolve_llm(m))
        for m in model_list
    ]
    results = parallel_backtest(configs, max_workers=parallel_workers)

    for model_name, metrics in zip(model_list, results):
        typer.echo(f"\n=== {model_name} ===")
        for k, v in metrics.items():
            typer.echo(f"  {k}: {v}")


@app.command()
def compare_risk(
    model: str = typer.Option("openai/gpt-4o-mini", "--model", "-m"),
    symbol: str = typer.Option("AAPL", "--symbol", "-s"),
    steps: int = typer.Option(3600, "--steps", "-n", help="Number of steps (3600 = ~1 hour)"),
    parallel_workers: int = typer.Option(2, "--workers", "-w"),
):
    prices = fetch_prices(symbol)
    llm = _resolve_llm(model)
    use_steps = min(steps, len(prices))

    configs = [
        BacktestConfig(price_series=prices, llm=llm, max_steps=use_steps, reward_fn=multicomponent_reward),
        BacktestConfig(price_series=prices, llm=llm, max_steps=use_steps, reward_fn=aggressive_reward),
    ]
    results = parallel_backtest(configs, max_workers=parallel_workers)

    labels = ["Risk-Averse (multicomponent)", "Risk-Taker (aggressive)"]
    typer.echo(f"\n{'Metric':<30} {'Risk-Averse':<18} {'Risk-Taker':<18}")
    typer.echo("-" * 66)
    for key in results[0]:
        v1 = results[0][key]
        v2 = results[1][key]
        if isinstance(v1, float):
            typer.echo(f"{key:<30} {v1:<18.4f} {v2:<18.4f}")
        else:
            typer.echo(f"{key:<30} {str(v1):<18} {str(v2):<18}")