from trading_agent.core.cache import LLMResponseCache
from trading_agent.core.schemas import TradeDecision


def make_state(**overrides) -> dict:
    state = {
        "cash": 10000.0,
        "shares": 0,
        "price": 100.0,
        "price_history": [100.0],
        "portfolio_values": [10000.0],
        "peak_value": 10000.0,
        "action": "",
        "trade_cost": 0.0,
        "reward": 0.0,
        "total_reward": 0.0,
        "step": 0,
        "done": False,
    }
    state.update(overrides)
    return state


def test_cache_hit():
    cache = LLMResponseCache()
    state = make_state()
    decision = TradeDecision(action="BUY", quantity=10, reason="test")
    cache.put(state, decision)
    cached = cache.get(state)
    assert cached is not None
    assert cached.action == "BUY"
    assert cached.quantity == 10


def test_cache_miss():
    cache = LLMResponseCache()
    state = make_state()
    assert cache.get(state) is None


def test_cache_different_state_different_key():
    cache = LLMResponseCache()
    s1 = make_state(cash=10000.0)
    s2 = make_state(cash=9900.0)
    cache.put(s1, TradeDecision(action="HOLD", quantity=0, reason="a"))
    assert cache.get(s2) is None


def test_cache_clear():
    cache = LLMResponseCache()
    state = make_state()
    cache.put(state, TradeDecision(action="BUY", quantity=10, reason="test"))
    cache.clear()
    assert cache.get(state) is None
    assert cache.size == 0


def test_cache_lru_eviction():
    cache = LLMResponseCache(maxsize=2)
    s1 = make_state(step=0)
    s2 = make_state(step=1)
    s3 = make_state(step=2)
    d1 = TradeDecision(action="HOLD", quantity=0, reason="a")
    d2 = TradeDecision(action="BUY", quantity=10, reason="b")
    d3 = TradeDecision(action="SELL", quantity=5, reason="c")
    cache.put(s1, d1)
    cache.put(s2, d2)
    cache.put(s3, d3)
    assert cache.get(s1) is None
    assert cache.get(s2) is not None
    assert cache.get(s3) is not None


def test_cache_disabled():
    cache = LLMResponseCache()
    cache.enabled = False
    state = make_state()
    cache.put(state, TradeDecision(action="BUY", quantity=10, reason="test"))
    assert cache.get(state) is None


def test_cache_thread_safety():
    import threading
    cache = LLMResponseCache(maxsize=100)
    errors = []

    def worker(i: int):
        try:
            s = make_state(step=i)
            d = TradeDecision(action="HOLD", quantity=0, reason=str(i))
            cache.put(s, d)
            cache.get(s)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(errors) == 0