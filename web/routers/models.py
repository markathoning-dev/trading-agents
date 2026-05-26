from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func
from web.db.database import get_db
from web.db.models import BacktestRun, BacktestResult

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

@router.get("/compare", response_class=HTMLResponse)
def compare_models(request: Request, db: Session = Depends(get_db)):
    rows = db.query(
        BacktestRun.model_name,
        func.avg(BacktestResult.total_return).label("avg_return"),
        func.avg(BacktestResult.sharpe_ratio).label("avg_sharpe"),
        func.avg(BacktestResult.max_drawdown).label("avg_drawdown"),
        func.count(BacktestResult.id).label("count"),
    ).join(BacktestResult).group_by(BacktestRun.model_name).all()
    return templates.TemplateResponse("models/compare.html", {"request": request, "rows": rows})
