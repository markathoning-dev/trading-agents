import torch
from market_cgan.simulation.bar_world_agent import BarWorldAgent
from market_cgan.simulation.bar_exchange import BarExchange
from market_cgan.models.bar_generator import BarGenerator
from market_cgan.data.bar import Bar


def test_bar_world_agent_step():
    gen = BarGenerator(noise_dim=8, feature_dim=6, ref_price=100.0)
    ex = BarExchange()
    ex.append_bar(Bar(timestamp=1, open=100.0, high=101.0, low=99.5, close=100.5, volume=10000, vwap=100.3))
    agent = BarWorldAgent(generator=gen, exchange=ex, noise_dim=8)
    new_bar = agent.step()
    assert isinstance(new_bar, Bar)
    assert new_bar.high >= max(new_bar.open, new_bar.close)
    assert new_bar.low <= min(new_bar.open, new_bar.close)
    assert new_bar.volume >= 0
    assert len(ex.bars) == 2


def test_bar_world_agent_reset():
    gen = BarGenerator(noise_dim=8, feature_dim=6, ref_price=100.0)
    ex = BarExchange()
    ex.append_bar(Bar(timestamp=1, open=100.0, high=101.0, low=99.5, close=100.5, volume=10000, vwap=100.3))
    agent = BarWorldAgent(generator=gen, exchange=ex, noise_dim=8)
    agent.reset()
    assert len(ex.bars) == 0