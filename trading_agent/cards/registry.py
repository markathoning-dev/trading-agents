from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CardStats:
    risk_tolerance: float = 0.5
    volatility_preference: float = 0.5
    drawdown_penalty: float = 0.5
    trade_frequency: float = 0.5


@dataclass
class StrategyCard:
    id: str
    name: str
    rarity: str
    mana_cost: int
    description: str
    flavor_text: str
    stats: CardStats
    reward_type: Optional[str]
    nodes: list[str]
    prompt_modifier: str

    @classmethod
    def from_dict(cls, data: dict) -> StrategyCard:
        stats = CardStats(**data.get("stats", {}))
        return cls(
            id=data["id"],
            name=data["name"],
            rarity=data["rarity"],
            mana_cost=data["mana_cost"],
            description=data.get("description", ""),
            flavor_text=data.get("flavor_text", ""),
            stats=stats,
            reward_type=data.get("reward_type"),
            nodes=data.get("nodes", []),
            prompt_modifier=data.get("prompt_modifier", ""),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "rarity": self.rarity,
            "mana_cost": self.mana_cost,
            "description": self.description,
            "flavor_text": self.flavor_text,
            "stats": {
                "risk_tolerance": self.stats.risk_tolerance,
                "volatility_preference": self.stats.volatility_preference,
                "drawdown_penalty": self.stats.drawdown_penalty,
                "trade_frequency": self.stats.trade_frequency,
            },
            "reward_type": self.reward_type,
            "nodes": self.nodes,
            "prompt_modifier": self.prompt_modifier,
        }


class CardRegistry:
    _instance: Optional[CardRegistry] = None
    _cards: dict[str, StrategyCard] = {}

    @classmethod
    def get_instance(cls) -> CardRegistry:
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_cards()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None
        cls._cards = {}

    def _load_cards(self):
        cards_dir = Path(__file__).parent
        for json_file in cards_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                card = StrategyCard.from_dict(data)
                self._cards[card.id] = card
            except Exception as e:
                logger.warning("Failed to load card from %s: %s", json_file, e)

    def get_card(self, card_id: str) -> Optional[StrategyCard]:
        return self._cards.get(card_id)

    def list_cards(self) -> list[StrategyCard]:
        return list(self._cards.values())

    def list_cards_by_rarity(self, rarity: str) -> list[StrategyCard]:
        return [c for c in self._cards.values() if c.rarity == rarity]


def get_card(card_id: str) -> Optional[StrategyCard]:
    return CardRegistry.get_instance().get_card(card_id)


def list_cards() -> list[StrategyCard]:
    return CardRegistry.get_instance().list_cards()
