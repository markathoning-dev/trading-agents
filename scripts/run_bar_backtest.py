"""
Backtest Risk-Averse vs Risk-Taker on CGAN-generated OHLCV bar data.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pandas as pd
from market_cgan.models.bar_generator import BarGenerator
from market_cgan.simulation.bar_exchange import BarExchange
from market_cgan.simulation.bar_world_agent import BarWorldAgent
from market_cgan.data.bar import Bar
from trading_agent.backtest.engine import backtest_agent
from trading_agent.core.reward import multicomponent_reward, aggressive_reward
from trading_agent.models.gateway import KiloGateway
from trading_agent.config.settings import settings

gen = BarGenerator(noise_dim=64, feature_dim=6, ref_price=100.0)
gen.load_state_dict(torch.load('models/cgan/test_run/generator.pt', map_location='cpu', weights_only=True))
gen.eval()

ex = BarExchange()
ex.append_bar(Bar(timestamp=1, open=57.5, high=58.0, low=57.3, close=57.7, volume=386605, vwap=57.77))
agent = BarWorldAgent(gen, ex, noise_dim=64)

prices = [57.7]
for _ in range(200):
    bar = agent.step()
    prices.append(bar.close)

print(f"Generated {len(prices)} bars. Close range: {min(prices):.2f} - {max(prices):.2f}")
price_series = pd.Series(prices)

try:
    llm = KiloGateway(settings.default_model).get_langchain_llm()
    from langchain_core.messages import HumanMessage
    llm.invoke([HumanMessage(content="test")])
    print("Kilo LLM OK")
except Exception as e:
    from trading_agent.models.mock import MockGateway
    llm = MockGateway().get_langchain_llm()
    print(f"Using Mock LLM ({e})")

agents = [
    ("Risk-Averse (multicomponent)", multicomponent_reward),
    ("Risk-Taker (aggressive)", aggressive_reward),
]

results = {}
for label, reward_fn in agents:
    print(f"Running {label}...")
    metrics = backtest_agent(price_series, llm=llm, max_steps=len(price_series), reward_fn=reward_fn)
    results[label] = metrics

print()
header = f"{'Metric':<30} {'Risk-Averse':<18} {'Risk-Taker':<18}"
print(header)
print("-" * 66)
for key in results[agents[0][0]]:
    v1 = results[agents[0][0]][key]
    v2 = results[agents[1][0]][key]
    if isinstance(v1, float):
        print(f"{key:<30} {v1:<18.4f} {v2:<18.4f}")
    else:
        print(f"{key:<30} {str(v1):<18} {str(v2):<18}")