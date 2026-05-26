from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from web.db.database import get_db
from web.db.models import PinnModel

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

@router.get("/train", response_class=HTMLResponse)
def train_pinn_form(request: Request):
    return templates.TemplateResponse("pinn/train.html", {"request": request})

@router.get("/generate", response_class=HTMLResponse)
def generate_form(request: Request, db: Session = Depends(get_db)):
    models = db.query(PinnModel).filter(PinnModel.status == "trained").all()
    return templates.TemplateResponse("pinn/generate.html", {"request": request, "models": models})
