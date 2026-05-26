import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from trading_agent.core.reward import multicomponent_reward, aggressive_reward
from trading_agent.backtest.engine import backtest_agent
from trading_agent.models.mock import MockGateway
from trading_agent.market.simulators import RandomWalkMarket

STEPS = 3600

prices = pd.Series([100.0])
market = RandomWalkMarket(start_price=100.0, drift=0.01, volatility=0.5, seed=42)
for _ in range(STEPS - 1):
    prices = pd.concat([prices, pd.Series([market.step()])], ignore_index=True)

llm = MockGateway().get_langchain_llm()

agents = [
    ("Risk-Averse (multicomponent)", multicomponent_reward),
    ("Risk-Taker (aggressive)", aggressive_reward),
]

results = {}
for label, reward_fn in agents:
    metrics = backtest_agent(prices, llm=llm, max_steps=STEPS, reward_fn=reward_fn)
    results[label] = metrics

print(f"\n{'Metric':<35} {'Risk-Averse':<20} {'Risk-Taker':<20}")
print("-" * 75)
for key in results[agents[0][0]]:
    v1 = results[agents[0][0]][key]
    v2 = results[agents[1][0]][key]
    if isinstance(v1, float):
        print(f"{key:<35} {v1:<20.4f} {v2:<20.4f}")
    else:
        print(f"{key:<35} {str(v1):<20} {str(v2):<20}")

print("\n--- Manual reward comparison on sample trades ---")
print(f"{'Scenario':<35} {'Multicomponent':<20} {'Aggressive':<20}")
print("-" * 75)
scenarios = [
    ("Profit (+10, peak=110)", 100, 110, 110, 0),
    ("Loss (-10, peak=110)", 100, 90, 110, 0),
    ("Round-trip +100 then -100", 100, 200, 200, 0),
    ("Loss with trade cost", 100, 90, 110, 2),
    ("Large profit, high vol bonus", 100, 150, 150, 0),
]
for label, old_v, new_v, peak, cost in scenarios:
    mc = multicomponent_reward(old_v, new_v, peak, cost, risk_penalty_lambda=0.1)
    ag = aggressive_reward(old_v, new_v, peak, cost, volatility_lambda=0.2)
    print(f"{label:<35} {mc:<20.4f} {ag:<20.4f}")
