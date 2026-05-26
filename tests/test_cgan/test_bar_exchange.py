from market_cgan.simulation.bar_exchange import BarExchange
from market_cgan.data.bar import Bar


def test_bar_exchange_initial_state():
    ex = BarExchange()
    state = ex.get_state()
    assert state["close"] is None


def test_bar_exchange_append_bar():
    ex = BarExchange()
    bar = Bar(timestamp=1, open=100.0, high=101.0, low=99.5, close=100.5, volume=10000, vwap=100.3)
    ex.append_bar(bar)
    assert ex.get_state()["close"] == 100.5


def test_bar_exchange_get_window():
    ex = BarExchange()
    for i in range(10):
        ex.append_bar(Bar(timestamp=i, open=100.0+i, high=101.0+i, low=99.5+i, close=100.5+i, volume=10000, vwap=100.3))
    window = ex.get_window(5)
    assert len(window) == 5
    assert window[0].close == 100.5 + 5


def test_bar_exchange_reset():
    ex = BarExchange()
    ex.append_bar(Bar(timestamp=1, open=100.0, high=101.0, low=99.5, close=100.5, volume=10000, vwap=100.3))
    ex.reset()
    assert len(ex.bars) == 0