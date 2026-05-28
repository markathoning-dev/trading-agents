import pytest
from trading_agent.cards.registry import CardRegistry, get_card, list_cards, StrategyCard


@pytest.fixture(autouse=True)
def reset_registry():
    CardRegistry.reset()
    yield
    CardRegistry.reset()


def test_load_all_cards():
    cards = list_cards()
    assert len(cards) == 8


def test_get_card_by_id():
    card = get_card("momentum-rider")
    assert card is not None
    assert card.name == "Momentum Rider"
    assert card.rarity == "rare"
    assert card.mana_cost == 3


def test_get_nonexistent_card():
    card = get_card("nonexistent-card")
    assert card is None


def test_card_stats():
    card = get_card("momentum-rider")
    assert card.stats.risk_tolerance == 0.7
    assert card.stats.volatility_preference == 0.8


def test_card_to_dict():
    card = get_card("momentum-rider")
    data = card.to_dict()
    assert data["id"] == "momentum-rider"
    assert data["name"] == "Momentum Rider"
    assert "stats" in data


def test_card_from_dict():
    data = {
        "id": "test-card",
        "name": "Test Card",
        "rarity": "common",
        "mana_cost": 1,
        "description": "A test card",
        "flavor_text": "Test flavor",
        "stats": {
            "risk_tolerance": 0.5,
            "volatility_preference": 0.5,
            "drawdown_penalty": 0.5,
            "trade_frequency": 0.5,
        },
        "reward_type": None,
        "nodes": [],
        "prompt_modifier": "Test modifier",
    }
    card = StrategyCard.from_dict(data)
    assert card.id == "test-card"
    assert card.name == "Test Card"


def test_list_cards_by_rarity():
    registry = CardRegistry.get_instance()
    common_cards = registry.list_cards_by_rarity("common")
    assert len(common_cards) > 0
    assert all(c.rarity == "common" for c in common_cards)


def test_all_cards_have_required_fields():
    cards = list_cards()
    for card in cards:
        assert card.id
        assert card.name
        assert card.rarity in ["common", "rare", "epic", "legendary"]
        assert card.mana_cost >= 1
        assert card.mana_cost <= 5
        assert card.description
        assert card.flavor_text
