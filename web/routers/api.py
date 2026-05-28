from datetime import datetime
import threading
import pandas as pd
from fastapi import APIRouter, Depends, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from web.db.database import get_db, SessionLocal
from web.db.models import BacktestRun, BacktestResult, BacktestStep, PinnModel, Deck
from web.tasks import run_backtest_task
from trading_agent.backtest.parallel import parallel_backtest, BacktestConfig
from trading_agent.models.gateway import KiloGateway
from trading_agent.models.mock import MockGateway
from trading_agent.market.polygon_market import fetch_prices
from trading_agent.cards.registry import list_cards, get_card
from trading_agent.cards.deck import Deck as DeckModel

router = APIRouter()


def _serialize_run(run: BacktestRun) -> dict:
    return {
        "id": run.id,
        "model_name": run.model_name,
        "data_source": run.data_source,
        "config": run.config,
        "status": run.status,
        "deck_id": run.deck_id,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "result": {
            "final_portfolio_value": run.result.final_portfolio_value,
            "total_return": run.result.total_return,
            "sharpe_ratio": run.result.sharpe_ratio,
            "max_drawdown": run.result.max_drawdown,
            "cumulative_reward": run.result.cumulative_reward,
            "num_steps": run.result.num_steps,
            "final_cash": run.result.final_cash,
            "final_shares": run.result.final_shares,
        } if run.result else None,
    }


@router.get("/dashboard")
def dashboard_api(db: Session = Depends(get_db)):
    runs = db.query(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(10).all()
    return [_serialize_run(r) for r in runs]


@router.get("/backtests")
def list_backtests_api(db: Session = Depends(get_db)):
    runs = db.query(BacktestRun).order_by(BacktestRun.created_at.desc()).all()
    return [_serialize_run(r) for r in runs]


@router.get("/backtests/{run_id}")
def backtest_detail_api(run_id: int, db: Session = Depends(get_db)):
    run = db.query(BacktestRun).filter(BacktestRun.id == run_id).first()
    if not run:
        return JSONResponse({"error": "not found"}, status_code=404)
    steps = db.query(BacktestStep).filter(BacktestStep.run_id == run_id).order_by(BacktestStep.step).all()
    return {
        **{k: v for k, v in _serialize_run(run).items() if k != "result"},
        "result": _serialize_run(run)["result"],
        "steps": [
            {
                "step": s.step, "price": s.price, "cash": s.cash,
                "shares": s.shares, "action": s.action,
                "portfolio_value": s.portfolio_value, "reward": s.reward,
            }
            for s in steps
        ],
    }


@router.post("/backtests/new")
def start_backtest_api(
    model_name: str = Form(...),
    symbol: str = Form("AAPL"),
    max_steps: int = Form(50),
    deck_id: str = Form(None),
):
    db = SessionLocal()
    run = BacktestRun(
        model_name=model_name, data_source=f"polygon:{symbol}",
        config={"max_steps": max_steps}, status="pending",
        deck_id=deck_id,
    )
    db.add(run)
    db.commit()
    run_id = run.id
    db.close()
    threading.Thread(
        target=run_backtest_task, args=(run_id, model_name, symbol, max_steps, deck_id),
        daemon=True,
    ).start()
    return {"run_id": run_id, "status": "started"}


@router.post("/backtests/compare")
def compare_backtests_api(
    models: str = Form(...),
    symbol: str = Form("AAPL"),
    max_steps: int = Form(50),
    max_workers: int = Form(4),
):
    import pandas as pd
    from trading_agent.backtest.parallel import parallel_backtest, BacktestConfig
    from trading_agent.models.gateway import KiloGateway
    from trading_agent.models.mock import MockGateway
    from trading_agent.market.polygon_market import fetch_prices

    prices = fetch_prices(symbol)
    use_steps = min(max_steps, len(prices))

    model_list = [m.strip() for m in models.split(",")]

    try:
        prices = fetch_prices(symbol)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    use_steps = min(max_steps, len(prices))

    configs = []
    for m in model_list:
        try:
            llm = KiloGateway(m).get_langchain_llm()
            from langchain_core.messages import HumanMessage
            llm.invoke([HumanMessage(content="test")])
        except Exception:
            llm = MockGateway().get_langchain_llm()
        configs.append(BacktestConfig(price_series=prices, llm=llm, max_steps=use_steps))

    results = parallel_backtest(configs, max_workers=max_workers)
    return {"models": model_list, "results": results, "symbol": symbol}


@router.get("/models/compare")
def compare_models_api(db: Session = Depends(get_db)):
    rows = (
        db.query(
            BacktestRun.model_name,
            func.avg(BacktestResult.total_return).label("avg_return"),
            func.avg(BacktestResult.sharpe_ratio).label("avg_sharpe"),
            func.avg(BacktestResult.max_drawdown).label("avg_drawdown"),
            func.count(BacktestResult.id).label("count"),
        )
        .join(BacktestResult)
        .group_by(BacktestRun.model_name)
        .all()
    )
    return [
        {
            "model_name": r.model_name,
            "avg_return": r.avg_return,
            "avg_sharpe": r.avg_sharpe,
            "avg_drawdown": r.avg_drawdown,
            "count": r.count,
        }
        for r in rows
    ]


@router.get("/pinn/models")
def list_pinn_models_api(db: Session = Depends(get_db)):
    models = db.query(PinnModel).filter(PinnModel.status == "trained").all()
    return [
        {
            "id": m.id, "name": m.name, "pde_type": m.pde_type,
            "architecture": m.architecture, "status": m.status,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in models
    ]


@router.get("/pinn/train/form")
def pinn_train_form_api():
    return {
        "fields": [
            {"name": "name", "type": "text", "default": "pinn-v1"},
            {"name": "pde_type", "type": "select", "options": ["black_scholes"], "default": "black_scholes"},
            {"name": "epochs", "type": "number", "default": 100},
            {"name": "symbol", "type": "text", "default": "AAPL"},
        ]
    }


@router.get("/pinn/generate/form")
def pinn_generate_form_api(db: Session = Depends(get_db)):
    models = db.query(PinnModel).filter(PinnModel.status == "trained").all()
    return {
        "models": [
            {"id": m.id, "name": m.name, "pde_type": m.pde_type} for m in models
        ],
        "fields": [
            {"name": "num_paths", "type": "number", "default": 10},
            {"name": "steps", "type": "number", "default": 252},
        ],
    }


@router.get("/cards")
def list_cards_api():
    cards = list_cards()
    return [c.to_dict() for c in cards]


@router.get("/cards/{card_id}")
def get_card_api(card_id: str):
    card = get_card(card_id)
    if card is None:
        return JSONResponse({"error": "Card not found"}, status_code=404)
    return card.to_dict()


@router.get("/decks")
def list_decks_api(db: Session = Depends(get_db)):
    decks = db.query(Deck).order_by(Deck.created_at.desc()).all()
    return [
        {
            "id": d.id,
            "name": d.name,
            "card_ids": d.card_ids,
            "mana_budget": d.mana_budget,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in decks
    ]


@router.get("/decks/{deck_id}")
def get_deck_api(deck_id: str, db: Session = Depends(get_db)):
    deck = db.query(Deck).filter(Deck.id == deck_id).first()
    if not deck:
        return JSONResponse({"error": "Deck not found"}, status_code=404)
    deck_model = DeckModel(
        id=deck.id,
        name=deck.name,
        card_ids=deck.card_ids,
        mana_budget=deck.mana_budget,
    )
    return deck_model.to_dict()


@router.post("/decks")
def create_deck_api(
    deck_id: str = Form(...),
    name: str = Form(...),
    card_ids: str = Form(...),
    mana_budget: int = Form(10),
    db: Session = Depends(get_db),
):
    card_id_list = [c.strip() for c in card_ids.split(",") if c.strip()]
    deck_model = DeckModel(
        id=deck_id,
        name=name,
        card_ids=card_id_list,
        mana_budget=mana_budget,
    )
    errors = deck_model.validate()
    if errors:
        return JSONResponse({"error": "Validation failed", "errors": errors}, status_code=400)

    existing = db.query(Deck).filter(Deck.id == deck_id).first()
    if existing:
        return JSONResponse({"error": "Deck ID already exists"}, status_code=409)

    deck = Deck(
        id=deck_id,
        name=name,
        card_ids=card_id_list,
        mana_budget=mana_budget,
    )
    db.add(deck)
    db.commit()
    return deck_model.to_dict()


@router.delete("/decks/{deck_id}")
def delete_deck_api(deck_id: str, db: Session = Depends(get_db)):
    deck = db.query(Deck).filter(Deck.id == deck_id).first()
    if not deck:
        return JSONResponse({"error": "Deck not found"}, status_code=404)
    db.delete(deck)
    db.commit()
    return {"message": "Deck deleted"}


@router.post("/backtests/new")
def start_backtest_api(
    model_name: str = Form(...),
    symbol: str = Form("AAPL"),
    max_steps: int = Form(50),
    deck_id: str = Form(None),
):
    db = SessionLocal()
    run = BacktestRun(
        model_name=model_name, data_source=f"polygon:{symbol}",
        config={"max_steps": max_steps}, status="pending",
        deck_id=deck_id,
    )
    db.add(run)
    db.commit()
    run_id = run.id
    db.close()
    threading.Thread(
        target=run_backtest_task, args=(run_id, model_name, symbol, max_steps, deck_id),
        daemon=True,
    ).start()
    return {"run_id": run_id, "status": "started"}