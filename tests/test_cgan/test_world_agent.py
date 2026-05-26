import torch
from market_cgan.models.generator import Generator
from market_cgan.simulation.exchange import LOBExchange
from market_cgan.simulation.world_agent import WorldAgent
from market_cgan.data.features import MarketFeatureExtractor


def test_world_agent_step():
    extractor = MarketFeatureExtractor()
    exchange = LOBExchange(feature_extractor=extractor)
    exchange.book.add_limit_order("SELL", 101.0, 100)
    exchange.book.add_limit_order("BUY", 99.0, 100)

    gen = Generator(noise_dim=64, feature_dim=42)
    agent = WorldAgent(gen, exchange, noise_dim=64)
    state = agent.step()
    assert state.mid_price > 0
    assert state.best_bid > 0 or state.best_ask < float("inf")
    assert hasattr(state, "spread")


def test_world_agent_multiple_steps():
    extractor = MarketFeatureExtractor()
    exchange = LOBExchange(feature_extractor=extractor)
    exchange.book.add_limit_order("SELL", 101.0, 100)
    exchange.book.add_limit_order("BUY", 99.0, 100)

    gen = Generator(noise_dim=64, feature_dim=42)
    agent = WorldAgent(gen, exchange, noise_dim=64)
    prices = []
    for _ in range(5):
        state = agent.step()
        prices.append(state.mid_price)
    assert len(prices) == 5
    assert all(p > 0 for p in prices)


def test_world_agent_reset():
    extractor = MarketFeatureExtractor()
    exchange = LOBExchange(feature_extractor=extractor)
    gen = Generator(noise_dim=64, feature_dim=42)
    agent = WorldAgent(gen, exchange, noise_dim=64)
    agent.reset()
    state = agent.step()
    assert state is not None
