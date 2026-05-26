from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from web.db.database import get_db, SessionLocal
from web.db.models import BacktestRun, BacktestResult, BacktestStep
from web.tasks import run_backtest_task

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

@router.get("", response_class=HTMLResponse)
def list_backtests(request: Request, db: Session = Depends(get_db)):
    runs = db.query(BacktestRun).order_by(BacktestRun.created_at.desc()).all()
    return templates.TemplateResponse("backtests/list.html", {"request": request, "runs": runs})

@router.get("/new", response_class=HTMLResponse)
def new_backtest_form(request: Request):
    return templates.TemplateResponse("backtests/run.html", {"request": request})

@router.post("/new")
def start_backtest(
    model_name: str = Form(...),
    symbol: str = Form("AAPL"),
    max_steps: int = Form(50),
):
    db = SessionLocal()
    run = BacktestRun(model_name=model_name, data_source=f"yfinance:{symbol}", config={"max_steps": max_steps}, status="pending")
    db.add(run)
    db.commit()
    run_id = run.id
    db.close()
    import threading
    threading.Thread(target=run_backtest_task, args=(run_id, model_name, symbol, max_steps), daemon=True).start()
    return RedirectResponse(url="/backtests", status_code=303)

@router.get("/{run_id}", response_class=HTMLResponse)
def backtest_detail(request: Request, run_id: int, db: Session = Depends(get_db)):
    run = db.query(BacktestRun).filter(BacktestRun.id == run_id).first()
    steps = db.query(BacktestStep).filter(BacktestStep.run_id == run_id).order_by(BacktestStep.step).all()
    return templates.TemplateResponse("backtests/detail.html", {"request": request, "run": run, "steps": steps})
