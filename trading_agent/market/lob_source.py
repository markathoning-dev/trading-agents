from __future__ import annotations

from trading_agent.market.base import MarketDataSource


class CGANMarketSource(MarketDataSource):
    def __init__(self, world_agent):
        self.world_agent = world_agent
        self._last_lob = None

    def step(self) -> float:
        lob_state = self.world_agent.step()
        self._last_lob = lob_state
        return lob_state.mid_price

    def get_lob_state(self):
        return self._last_lob

    def reset(self) -> None:
        self.world_agent.reset()
        self._last_lob = None
