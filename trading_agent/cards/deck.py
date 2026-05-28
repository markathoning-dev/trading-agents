from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from trading_agent.cards.registry import get_card, StrategyCard


@dataclass
class Deck:
    id: str
    name: str
    card_ids: list[str]
    mana_budget: int = 10

    def validate(self) -> list[str]:
        """Return list of validation errors. Empty if valid."""
        errors = []

        if not self.name or not self.name.strip():
            errors.append("Deck name is required")

        if not self.card_ids:
            errors.append("Deck must contain at least one card")
            return errors

        seen_ids = set()
        total_cost = 0
        reward_cards = 0

        for card_id in self.card_ids:
            if card_id in seen_ids:
                errors.append(f"Duplicate card: {card_id}")
            seen_ids.add(card_id)

            card = get_card(card_id)
            if card is None:
                errors.append(f"Unknown card: {card_id}")
                continue

            total_cost += card.mana_cost
            if card.reward_type:
                reward_cards += 1

        if total_cost > self.mana_budget:
            errors.append(f"Deck costs {total_cost} mana, budget is {self.mana_budget}")

        if reward_cards == 0:
            errors.append("Deck must have at least 1 card with a reward_type")
        elif reward_cards > 1:
            errors.append(f"Deck must have exactly 1 reward card, found {reward_cards}")

        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0

    def get_cards(self) -> list[StrategyCard]:
        """Return resolved card objects in deck order."""
        cards = []
        for card_id in self.card_ids:
            card = get_card(card_id)
            if card:
                cards.append(card)
        return cards

    def get_total_mana(self) -> int:
        """Return total mana cost of all cards in deck."""
        return sum(c.mana_cost for c in self.get_cards())

    def get_reward_type(self) -> Optional[str]:
        """Return the reward_type from the reward card in the deck."""
        for card in self.get_cards():
            if card.reward_type:
                return card.reward_type
        return None

    def get_prompt_modifiers(self) -> str:
        """Concatenate all prompt modifiers from cards."""
        return " ".join(c.prompt_modifier for c in self.get_cards() if c.prompt_modifier)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "card_ids": self.card_ids,
            "mana_budget": self.mana_budget,
            "total_mana": self.get_total_mana(),
            "is_valid": self.is_valid(),
            "errors": self.validate(),
            "cards": [c.to_dict() for c in self.get_cards()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Deck:
        return cls(
            id=data["id"],
            name=data["name"],
            card_ids=data["card_ids"],
            mana_budget=data.get("mana_budget", 10),
        )
