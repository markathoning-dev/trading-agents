import typer
import pandas as pd
from trading_agent.backtest.engine import backtest_agent
from trading_agent.core.reward import multicomponent_reward, aggressive_reward
from trading_agent.models.gateway import KiloGateway
from trading_agent.models.mock import MockGateway
from trading_agent.config.settings import settings

app = typer.Typer()

@app.command()
def run(
    model: str = typer.Option(settings.default_model, "--model", "-m"),
    symbol: str = typer.Option("AAPL", "--symbol", "-s"),
    max_steps: int = typer.Option(settings.max_steps, "--steps", "-n"),
):
    import yfinance as yf
    hist = yf.Ticker(symbol).history(period="1y")
    prices = pd.Series(hist["Close"].values)
    llm = None
    try:
        llm = KiloGateway(model).get_langchain_llm()
        from langchain_core.messages import HumanMessage
        llm.invoke([HumanMessage(content="test")])
    except Exception:
        llm = MockGateway().get_langchain_llm()
    metrics = backtest_agent(prices, llm=llm, max_steps=min(max_steps, len(prices)))
    for k, v in metrics.items():
        typer.echo(f"{k}: {v}")

@app.command()
def compare(
    models: str = typer.Option("openai/gpt-4o-mini,openai/gpt-3.5-turbo", "--models", "-m"),
    symbol: str = typer.Option("AAPL", "--symbol", "-s"),
):
    model_list = [m.strip() for m in models.split(",")]
    import yfinance as yf
    hist = yf.Ticker(symbol).history(period="1y")
    prices = pd.Series(hist["Close"].values)
    for model_name in model_list:
        llm = None
        try:
            llm = KiloGateway(model_name).get_langchain_llm()
            from langchain_core.messages import HumanMessage
            llm.invoke([HumanMessage(content="test")])
        except Exception:
            llm = MockGateway().get_langchain_llm()
        metrics = backtest_agent(prices, llm=llm)
        typer.echo(f"\n=== {model_name} ===")
        for k, v in metrics.items():
            typer.echo(f"  {k}: {v}")

@app.command()
def compare_risk(
    model: str = typer.Option("openai/gpt-4o-mini", "--model", "-m"),
    symbol: str = typer.Option("AAPL", "--symbol", "-s"),
    steps: int = typer.Option(3600, "--steps", "-n", help="Number of steps (3600 = ~1 hour)"),
):
    import yfinance as yf
    hist = yf.Ticker(symbol).history(period="1y")
    prices = pd.Series(hist["Close"].values)
    llm = None
    try:
        llm = KiloGateway(model).get_langchain_llm()
        # Test that the LLM actually works (one quick call)
        from langchain_core.messages import HumanMessage
        llm.invoke([HumanMessage(content="test")])
    except Exception:
        llm = MockGateway().get_langchain_llm()

    agents = [
        ("Risk-Averse (multicomponent)", multicomponent_reward),
        ("Risk-Taker (aggressive)", aggressive_reward),
    ]

    results = {}
    for label, reward_fn in agents:
        metrics = backtest_agent(
            prices, llm=llm,
            max_steps=min(steps, len(prices)),
            reward_fn=reward_fn,
        )
        results[label] = metrics

    typer.echo(f"\n{'Metric':<30} {'Risk-Averse':<18} {'Risk-Taker':<18}")
    typer.echo("-" * 66)
    for key in results[agents[0][0]]:
        v1 = results[agents[0][0]][key]
        v2 = results[agents[1][0]][key]
        if isinstance(v1, float):
            typer.echo(f"{key:<30} {v1:<18.4f} {v2:<18.4f}")
        else:
            typer.echo(f"{key:<30} {str(v1):<18} {str(v2):<18}")
