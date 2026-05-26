from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from web.routers import dashboard, backtests, models, pinn, cgan
from web.db.database import init_db

app = FastAPI(title="Trading Agent Dashboard")
app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.include_router(dashboard.router)
app.include_router(backtests.router, prefix="/backtests")
app.include_router(models.router, prefix="/models")
app.include_router(pinn.router, prefix="/pinn")
app.include_router(cgan.router)

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/health")
def health():
    return {"status": "ok"}
