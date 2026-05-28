from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from trading_agent.backtest.engine import backtest_agent
from trading_agent.models.gateway import KiloGateway
from trading_agent.models.mock import MockGateway

logger = logging.getLogger(__name__)


@dataclass
class BacktestJobResult:
    metrics: dict[str, Any]
    llm_used: str


def run_backtest_core(
    prices: pd.Series,
    model_name: str,
    max_steps: int,
    fee_rate: float = 0.001,
    risk_lambda: float = 0.1,
    deck=None,
) -> BacktestJobResult:
    try:
        llm = KiloGateway(model_name).get_langchain_llm()
        llm_label = model_name
    except Exception:
        llm = MockGateway().get_langchain_llm()
        llm_label = "mock"

    metrics = backtest_agent(
        prices, llm=llm,
        max_steps=min(max_steps, len(prices)),
        fee_rate=fee_rate,
        risk_lambda=risk_lambda,
        deck=deck,
    )
    return BacktestJobResult(metrics=metrics, llm_used=llm_label)


def run_backtest_task(run_id: int, model_name: str, symbol: str, max_steps: int, deck_id: str = None):
    from web.db.database import SessionLocal
    from web.db.models import BacktestRun, BacktestResult, Deck
    from trading_agent.market.polygon_market import fetch_prices
    from trading_agent.cards.deck import Deck as DeckModel

    db = SessionLocal()
    try:
        run = db.query(BacktestRun).filter(BacktestRun.id == run_id).first()
        run.status = "running"
        db.commit()

        deck = None
        if deck_id:
            deck_record = db.query(Deck).filter(Deck.id == deck_id).first()
            if deck_record:
                deck = DeckModel(
                    id=deck_record.id,
                    name=deck_record.name,
                    card_ids=deck_record.card_ids,
                    mana_budget=deck_record.mana_budget,
                )

        prices = fetch_prices(symbol)
        result = run_backtest_core(prices, model_name, max_steps, deck=deck)

        db_result = BacktestResult(
            run_id=run_id,
            final_portfolio_value=result.metrics["final_portfolio_value"],
            total_return=result.metrics["total_return"],
            sharpe_ratio=result.metrics["sharpe_ratio"],
            max_drawdown=result.metrics["max_drawdown"],
            cumulative_reward=result.metrics["cumulative_reward"],
            num_steps=result.metrics["num_steps"],
            final_cash=result.metrics["final_cash"],
            final_shares=result.metrics["final_shares"],
        )
        db.add(db_result)
        run.status = "completed"
        db.commit()
    except Exception:
        logger.exception("Backtest run %d failed", run_id)
        run.status = "failed"
        db.commit()
    finally:
        db.close()
