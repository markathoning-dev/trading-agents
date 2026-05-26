import pandas as pd
from web.db.database import SessionLocal
from web.db.models import BacktestRun, BacktestResult
from trading_agent.backtest.engine import backtest_agent
from trading_agent.models.gateway import KiloGateway
from trading_agent.models.mock import MockGateway

def run_backtest_task(run_id: int, model_name: str, symbol: str, max_steps: int):
    db = SessionLocal()
    try:
        run = db.query(BacktestRun).filter(BacktestRun.id == run_id).first()
        run.status = "running"
        db.commit()

        import yfinance as yf
        hist = yf.Ticker(symbol).history(period="1y")
        prices = pd.Series(hist["Close"].values)

        try:
            llm = KiloGateway(model_name).get_langchain_llm()
        except Exception:
            llm = MockGateway().get_langchain_llm()

        metrics = backtest_agent(prices, llm=llm, max_steps=min(max_steps, len(prices)))
        result = BacktestResult(
            run_id=run_id, final_portfolio_value=metrics["final_portfolio_value"],
            total_return=metrics["total_return"], sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"], cumulative_reward=metrics["cumulative_reward"],
            num_steps=metrics["num_steps"], final_cash=metrics["final_cash"],
            final_shares=metrics["final_shares"],
        )
        db.add(result)
        run.status = "completed"
        db.commit()
    except Exception:
        run.status = "failed"
        db.commit()
    finally:
        db.close()
