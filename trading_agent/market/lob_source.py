from trading_agent.market.base import MarketDataSource
from market_cgan.simulation.world_agent import WorldAgent
from market_cgan.simulation.exchange import LOBState


class CGANMarketSource(MarketDataSource):
    def __init__(self, world_agent: WorldAgent):
        self.world_agent = world_agent
        self._last_state: LOBState | None = None

    def step(self) -> float:
        self._last_state = self.world_agent.step()
        return self._last_state.mid_price

    def get_lob_state(self) -> LOBState | None:
        return self._last_state

    def reset(self) -> None:
        self.world_agent.reset()
        self._last_state = None
