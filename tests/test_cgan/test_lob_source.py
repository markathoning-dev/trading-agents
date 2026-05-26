import torch
from market_cgan.models.generator import Generator
from market_cgan.simulation.exchange import LOBExchange
from market_cgan.simulation.world_agent import WorldAgent
from market_cgan.data.features import MarketFeatureExtractor
from trading_agent.market.lob_source import CGANMarketSource


def test_cgan_market_source_step():
    extractor = MarketFeatureExtractor()
    exchange = LOBExchange(feature_extractor=extractor)
    exchange.book.add_limit_order("SELL", 101.0, 100)
    exchange.book.add_limit_order("BUY", 99.0, 100)

    gen = Generator(noise_dim=64, feature_dim=42)
    agent = WorldAgent(gen, exchange, noise_dim=64)
    source = CGANMarketSource(agent)
    price = source.step()
    assert price > 0


def test_cgan_market_source_get_lob_state():
    extractor = MarketFeatureExtractor()
    exchange = LOBExchange(feature_extractor=extractor)
    exchange.book.add_limit_order("SELL", 101.0, 100)
    exchange.book.add_limit_order("BUY", 99.0, 100)

    gen = Generator(noise_dim=64, feature_dim=42)
    agent = WorldAgent(gen, exchange, noise_dim=64)
    source = CGANMarketSource(agent)
    source.step()
    state = source.get_lob_state()
    assert state is not None
    assert state.mid_price > 0


def test_cgan_market_source_reset():
    extractor = MarketFeatureExtractor()
    exchange = LOBExchange(feature_extractor=extractor)
    gen = Generator(noise_dim=64, feature_dim=42)
    agent = WorldAgent(gen, exchange, noise_dim=64)
    source = CGANMarketSource(agent)
    source.step()
    source.reset()
    assert source.get_lob_state() is None
