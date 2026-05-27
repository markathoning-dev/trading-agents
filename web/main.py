from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from web.routers import dashboard, backtests, models, pinn, cgan, api
from web.db.database import init_db

app = FastAPI(title="Trading Agent Dashboard")
app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.include_router(dashboard.router)
app.include_router(backtests.router, prefix="/backtests")
app.include_router(models.router, prefix="/models")
app.include_router(pinn.router, prefix="/pinn")
app.include_router(cgan.router)
app.include_router(api.router, prefix="/api")

DIST_DIR = Path("web/static/dist")


@app.get("/app/{path:path}")
def spa_fallback(path: str):
    index = DIST_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"error": "frontend not built yet, run: cd web/frontend && npm run build"}, 501


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}