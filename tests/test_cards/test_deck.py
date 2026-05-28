import pytest
from trading_agent.cards.deck import Deck
from trading_agent.cards.registry import CardRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    CardRegistry.reset()
    yield
    CardRegistry.reset()


def test_deck_validation_valid():
    deck = Deck(
        id="test-deck",
        name="Test Deck",
        card_ids=["momentum-rider", "stop-loss-sentinel"],
        mana_budget=10,
    )
    errors = deck.validate()
    assert len(errors) == 0


def test_deck_validation_over_budget():
    deck = Deck(
        id="test-deck",
        name="Test Deck",
        card_ids=["diamond-hands", "momentum-rider"],
        mana_budget=5,
    )
    errors = deck.validate()
    assert any("mana" in e.lower() for e in errors)


def test_deck_validation_no_reward():
    deck = Deck(
        id="test-deck",
        name="Test Deck",
        card_ids=["stop-loss-sentinel", "position-sizer"],
        mana_budget=10,
    )
    errors = deck.validate()
    assert any("reward" in e.lower() for e in errors)


def test_deck_validation_multiple_rewards():
    deck = Deck(
        id="test-deck",
        name="Test Deck",
        card_ids=["momentum-rider", "mean-reversion-mage"],
        mana_budget=10,
    )
    errors = deck.validate()
    assert any("reward" in e.lower() for e in errors)


def test_deck_validation_duplicate_cards():
    deck = Deck(
        id="test-deck",
        name="Test Deck",
        card_ids=["momentum-rider", "momentum-rider"],
        mana_budget=10,
    )
    errors = deck.validate()
    assert any("duplicate" in e.lower() for e in errors)


def test_deck_validation_unknown_card():
    deck = Deck(
        id="test-deck",
        name="Test Deck",
        card_ids=["nonexistent-card"],
        mana_budget=10,
    )
    errors = deck.validate()
    assert any("unknown" in e.lower() for e in errors)


def test_deck_validation_empty_name():
    deck = Deck(
        id="test-deck",
        name="",
        card_ids=["momentum-rider"],
        mana_budget=10,
    )
    errors = deck.validate()
    assert any("name" in e.lower() for e in errors)


def test_deck_get_cards():
    deck = Deck(
        id="test-deck",
        name="Test Deck",
        card_ids=["momentum-rider", "stop-loss-sentinel"],
        mana_budget=10,
    )
    cards = deck.get_cards()
    assert len(cards) == 2
    assert cards[0].id == "momentum-rider"
    assert cards[1].id == "stop-loss-sentinel"


def test_deck_get_total_mana():
    deck = Deck(
        id="test-deck",
        name="Test Deck",
        card_ids=["momentum-rider", "stop-loss-sentinel"],
        mana_budget=10,
    )
    assert deck.get_total_mana() == 4


def test_deck_get_reward_type():
    deck = Deck(
        id="test-deck",
        name="Test Deck",
        card_ids=["momentum-rider", "stop-loss-sentinel"],
        mana_budget=10,
    )
    assert deck.get_reward_type() == "aggressive"


def test_deck_get_prompt_modifiers():
    deck = Deck(
        id="test-deck",
        name="Test Deck",
        card_ids=["momentum-rider", "stop-loss-sentinel"],
        mana_budget=10,
    )
    modifiers = deck.get_prompt_modifiers()
    assert "momentum" in modifiers.lower()
    assert "stop-loss" in modifiers.lower()


def test_deck_to_dict():
    deck = Deck(
        id="test-deck",
        name="Test Deck",
        card_ids=["momentum-rider", "stop-loss-sentinel"],
        mana_budget=10,
    )
    data = deck.to_dict()
    assert data["id"] == "test-deck"
    assert data["name"] == "Test Deck"
    assert data["total_mana"] == 4
    assert data["is_valid"] is True


def test_deck_from_dict():
    data = {
        "id": "test-deck",
        "name": "Test Deck",
        "card_ids": ["momentum-rider"],
        "mana_budget": 10,
    }
    deck = Deck.from_dict(data)
    assert deck.id == "test-deck"
    assert len(deck.card_ids) == 1
