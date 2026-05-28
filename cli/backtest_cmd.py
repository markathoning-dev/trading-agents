import typer
from typing import Optional
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


def _load_deck(deck_id: Optional[str]):
    if not deck_id:
        return None
    try:
        from web.db.database import SessionLocal
        from web.db.models import Deck as DeckDB
        from trading_agent.cards.deck import Deck

        db = SessionLocal()
        db_deck = db.query(DeckDB).filter(DeckDB.id == deck_id).first()
        db.close()

        if not db_deck:
            typer.echo(f"Deck '{deck_id}' not found.")
            raise typer.Exit(1)

        deck = Deck(
            id=db_deck.id,
            name=db_deck.name,
            card_ids=db_deck.card_ids,
            mana_budget=db_deck.mana_budget,
        )
        errors = deck.validate()
        if errors:
            typer.echo(f"Deck '{deck_id}' is invalid:")
            for err in errors:
                typer.echo(f"  - {err}")
            raise typer.Exit(1)
        return deck
    except Exception as e:
        typer.echo(f"Error loading deck: {e}")
        raise typer.Exit(1)


@app.command()
def run(
    model: str = typer.Option(settings.default_model, "--model", "-m"),
    symbol: str = typer.Option("AAPL", "--symbol", "-s"),
    max_steps: int = typer.Option(settings.max_steps, "--steps", "-n"),
    deck_id: Optional[str] = typer.Option(None, "--deck", "-d", help="Strategy deck ID"),
):
    deck = _load_deck(deck_id)
    prices = fetch_prices(symbol)
    llm = _resolve_llm(model)
    metrics = backtest_agent(prices, llm=llm, max_steps=min(max_steps, len(prices)), deck=deck)
    for k, v in metrics.items():
        typer.echo(f"{k}: {v}")


@app.command()
def compare(
    models: str = typer.Option("openai/gpt-4o-mini,openai/gpt-3.5-turbo", "--models", "-m"),
    symbol: str = typer.Option("AAPL", "--symbol", "-s"),
    parallel_workers: int = typer.Option(4, "--workers", "-w", help="Max parallel workers"),
    deck_ids: Optional[str] = typer.Option(None, "--decks", "-d", help="Comma-separated deck IDs"),
):
    model_list = [m.strip() for m in models.split(",")]
    prices = fetch_prices(symbol)

    decks = []
    if deck_ids:
        deck_id_list = [d.strip() for d in deck_ids.split(",")]
        for did in deck_id_list:
            decks.append(_load_deck(did))

    configs = []
    for m in model_list:
        if decks:
            for deck in decks:
                configs.append(BacktestConfig(price_series=prices, llm=_resolve_llm(m), deck=deck))
        else:
            configs.append(BacktestConfig(price_series=prices, llm=_resolve_llm(m)))

    results = parallel_backtest(configs, max_workers=parallel_workers)

    for i, metrics in enumerate(results):
        if decks:
            model_idx = i % len(model_list)
            deck_idx = i // len(model_list)
            label = f"{model_list[model_idx]} + {decks[deck_idx].name}"
        else:
            label = model_list[i]
        typer.echo(f"\n=== {label} ===")
        for k, v in metrics.items():
            typer.echo(f"  {k}: {v}")


@app.command()
def compare_decks(
    model: str = typer.Option("openai/gpt-4o-mini", "--model", "-m"),
    symbol: str = typer.Option("AAPL", "--symbol", "-s"),
    deck_ids: str = typer.Option(..., "--decks", "-d", help="Comma-separated deck IDs"),
    steps: int = typer.Option(100, "--steps", "-n"),
    parallel_workers: int = typer.Option(2, "--workers", "-w"),
):
    """Compare multiple decks with the same model."""
    deck_id_list = [d.strip() for d in deck_ids.split(",")]
    decks = [_load_deck(did) for did in deck_id_list]
    prices = fetch_prices(symbol)
    llm = _resolve_llm(model)
    use_steps = min(steps, len(prices))

    configs = [
        BacktestConfig(price_series=prices, llm=llm, max_steps=use_steps, deck=deck)
        for deck in decks
    ]
    results = parallel_backtest(configs, max_workers=parallel_workers)

    typer.echo(f"\n{'Deck':<30} {'Return':<12} {'Sharpe':<12} {'MaxDD':<12} {'Reward':<12}")
    typer.echo("-" * 78)
    for deck, metrics in zip(decks, results):
        typer.echo(
            f"{deck.name:<30} "
            f"{metrics.get('total_return', 0):<12.4f} "
            f"{metrics.get('sharpe_ratio', 0):<12.4f} "
            f"{metrics.get('max_drawdown', 0):<12.4f} "
            f"{metrics.get('cumulative_reward', 0):<12.4f}"
        )