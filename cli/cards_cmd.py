import typer
from typing import Optional
from trading_agent.cards.registry import list_cards, get_card
from trading_agent.cards.deck import Deck

app = typer.Typer()


@app.command("list")
def list_cards_cmd():
    """List all available strategy cards."""
    cards = list_cards()
    if not cards:
        typer.echo("No cards found.")
        return

    typer.echo(f"\n{'ID':<25} {'Name':<25} {'Rarity':<10} {'Mana':<6} {'Reward':<15}")
    typer.echo("-" * 81)
    for card in sorted(cards, key=lambda c: (c.rarity, c.mana_cost)):
        reward = card.reward_type or "—"
        typer.echo(f"{card.id:<25} {card.name:<25} {card.rarity:<10} {card.mana_cost:<6} {reward:<15}")
    typer.echo(f"\nTotal: {len(cards)} cards")


@app.command("show")
def show_card_cmd(card_id: str):
    """Show detailed information about a card."""
    card = get_card(card_id)
    if card is None:
        typer.echo(f"Card '{card_id}' not found.")
        raise typer.Exit(1)

    typer.echo(f"\n{'=' * 50}")
    typer.echo(f"  {card.name}")
    typer.echo(f"  {card.rarity.upper()} • Mana Cost: {card.mana_cost}")
    typer.echo(f"{'=' * 50}")
    typer.echo(f"\n{card.description}")
    typer.echo(f"\n\"{card.flavor_text}\"")
    typer.echo(f"\nStats:")
    typer.echo(f"  Risk Tolerance:      {'█' * int(card.stats.risk_tolerance * 10)}{'░' * (10 - int(card.stats.risk_tolerance * 10))} {card.stats.risk_tolerance:.1f}")
    typer.echo(f"  Volatility Pref:     {'█' * int(card.stats.volatility_preference * 10)}{'░' * (10 - int(card.stats.volatility_preference * 10))} {card.stats.volatility_preference:.1f}")
    typer.echo(f"  Drawdown Penalty:    {'█' * int(card.stats.drawdown_penalty * 10)}{'░' * (10 - int(card.stats.drawdown_penalty * 10))} {card.stats.drawdown_penalty:.1f}")
    typer.echo(f"  Trade Frequency:     {'█' * int(card.stats.trade_frequency * 10)}{'░' * (10 - int(card.stats.trade_frequency * 10))} {card.stats.trade_frequency:.1f}")
    typer.echo(f"\nReward Type: {card.reward_type or 'None (utility card)'}")
    typer.echo(f"Nodes: {', '.join(card.nodes) if card.nodes else 'None'}")
    if card.prompt_modifier:
        typer.echo(f"\nPrompt Modifier:\n  {card.prompt_modifier}")


@app.command("deck-list")
def list_decks_cmd():
    """List all saved decks."""
    try:
        from web.db.database import SessionLocal
        from web.db.models import Deck as DeckDB

        db = SessionLocal()
        decks = db.query(DeckDB).all()
        db.close()

        if not decks:
            typer.echo("No decks found.")
            return

        typer.echo(f"\n{'ID':<20} {'Name':<25} {'Cards':<8} {'Mana':<10}")
        typer.echo("-" * 63)
        for d in decks:
            card_count = len(d.card_ids) if d.card_ids else 0
            typer.echo(f"{d.id:<20} {d.name:<25} {card_count:<8} {d.mana_budget:<10}")
        typer.echo(f"\nTotal: {len(decks)} decks")
    except Exception as e:
        typer.echo(f"Error listing decks: {e}")


@app.command("deck-create")
def create_deck_cmd(
    deck_id: str = typer.Option(..., "--id", help="Unique deck ID"),
    name: str = typer.Option(..., "--name", "-n", help="Deck name"),
    cards: str = typer.Option(..., "--cards", "-c", help="Comma-separated card IDs"),
    mana_budget: int = typer.Option(10, "--mana", "-m", help="Mana budget"),
):
    """Create a new deck."""
    card_ids = [c.strip() for c in cards.split(",") if c.strip()]
    deck = Deck(id=deck_id, name=name, card_ids=card_ids, mana_budget=mana_budget)

    errors = deck.validate()
    if errors:
        typer.echo("Validation errors:")
        for err in errors:
            typer.echo(f"  - {err}")
        raise typer.Exit(1)

    try:
        from web.db.database import SessionLocal
        from web.db.models import Deck as DeckDB

        db = SessionLocal()
        existing = db.query(DeckDB).filter(DeckDB.id == deck_id).first()
        if existing:
            typer.echo(f"Deck '{deck_id}' already exists.")
            db.close()
            raise typer.Exit(1)

        db_deck = DeckDB(
            id=deck_id,
            name=name,
            card_ids=card_ids,
            mana_budget=mana_budget,
        )
        db.add(db_deck)
        db.commit()
        db.close()

        typer.echo(f"Deck '{name}' created successfully!")
        typer.echo(f"  Cards: {', '.join(card_ids)}")
        typer.echo(f"  Mana: {deck.get_total_mana()}/{mana_budget}")
    except Exception as e:
        typer.echo(f"Error creating deck: {e}")
        raise typer.Exit(1)


@app.command("deck-validate")
def validate_deck_cmd(deck_id: str):
    """Validate a deck."""
    try:
        from web.db.database import SessionLocal
        from web.db.models import Deck as DeckDB

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
            typer.echo(f"Deck '{deck.name}' is INVALID:")
            for err in errors:
                typer.echo(f"  ✗ {err}")
            raise typer.Exit(1)
        else:
            typer.echo(f"Deck '{deck.name}' is VALID!")
            typer.echo(f"  Mana: {deck.get_total_mana()}/{deck.mana_budget}")
            typer.echo(f"  Cards: {len(deck.card_ids)}")
            typer.echo(f"  Reward: {deck.get_reward_type()}")
    except Exception as e:
        typer.echo(f"Error validating deck: {e}")
        raise typer.Exit(1)
