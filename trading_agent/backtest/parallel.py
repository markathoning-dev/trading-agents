from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pandas as pd
from trading_agent.backtest.engine import backtest_agent
from trading_agent.core.reward import multicomponent_reward


@dataclass
class BacktestConfig:
    price_series: pd.Series
    llm: Any
    fee_rate: float = 0.001
    risk_lambda: float = 0.1
    max_steps: int | None = None
    reward_fn: Callable = multicomponent_reward
    deck: Any = None


def parallel_backtest(
    configs: list[BacktestConfig],
    max_workers: int = 4,
) -> list[dict]:
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map: dict[Any, int] = {}
        for i, cfg in enumerate(configs):
            kwargs = {}
            for field in dataclasses.fields(BacktestConfig):
                val = getattr(cfg, field.name)
                kwargs[field.name] = val
            fut = ex.submit(backtest_agent, **kwargs)
            fut_map[fut] = i

        results: list[dict | None] = [None] * len(configs)
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = {"error": str(e)}
        return results