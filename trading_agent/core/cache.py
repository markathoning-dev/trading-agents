from __future__ import annotations

import threading
from collections import OrderedDict
from trading_agent.core.state import AgentState
from trading_agent.core.schemas import TradeDecision


class _ThreadLocalCache:
    def __init__(self, maxsize: int = 1000):
        self.cache: OrderedDict[str, TradeDecision] = OrderedDict()
        self.maxsize = maxsize

    def _make_key(self, state: AgentState) -> str:
        return (
            f"{state['cash']:.2f}|{state['shares']}|{state['price']:.2f}|"
            f"{state['peak_value']:.2f}|{state['step']}"
        )

    def get(self, state: AgentState) -> TradeDecision | None:
        key = self._make_key(state)
        decision = self.cache.get(key)
        if decision is not None:
            self.cache.move_to_end(key)
        return decision

    def put(self, state: AgentState, decision: TradeDecision) -> None:
        key = self._make_key(state)
        self.cache[key] = decision
        while len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        self.cache.clear()

    @property
    def size(self) -> int:
        return len(self.cache)


class LLMResponseCache:
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.enabled = True
        self._local = threading.local()

    def _get_cache(self) -> _ThreadLocalCache:
        if not hasattr(self._local, "cache"):
            self._local.cache = _ThreadLocalCache(maxsize=self.maxsize)
        return self._local.cache

    def get(self, state: AgentState) -> TradeDecision | None:
        if not self.enabled:
            return None
        return self._get_cache().get(state)

    def put(self, state: AgentState, decision: TradeDecision) -> None:
        if not self.enabled:
            return
        self._get_cache().put(state, decision)

    def clear(self) -> None:
        if hasattr(self._local, "cache"):
            self._local.cache.clear()

    @property
    def size(self) -> int:
        return self._get_cache().size if hasattr(self._local, "cache") else 0


llm_cache = LLMResponseCache()