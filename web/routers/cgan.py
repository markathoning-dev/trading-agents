from __future__ import annotations

import json
from pathlib import Path
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import torch
import pandas as pd
from market_cgan.models.bar_generator import BarGenerator
from market_cgan.simulation.bar_exchange import BarExchange
from market_cgan.simulation.bar_world_agent import BarWorldAgent
from market_cgan.data.bar import Bar
from trading_agent.backtest.engine import backtest_agent
from trading_agent.core.reward import multicomponent_reward, aggressive_reward
from trading_agent.core.graph import build_bar_graph
from trading_agent.core.state import AgentState
from trading_agent.models.gateway import KiloGateway
from trading_agent.config.settings import settings

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")
MODELS_DIR = Path("models/cgan")


def _discover_models() -> list[dict]:
    models = []
    if not MODELS_DIR.exists():
        return models
    for d in sorted(MODELS_DIR.iterdir()):
        info_path = d / "model_info.json"
        gen_path = d / "generator.pt"
        if info_path.exists() and gen_path.exists():
            info = json.loads(info_path.read_text())
            models.append({
                "name": d.name,
                "path": str(gen_path),
                "ref_price": info.get("ref_price", 100.0),
            })
    return models


def _load_generator(model_path: str, ref_price: float) -> BarGenerator:
    gen = BarGenerator(noise_dim=64, feature_dim=6, ref_price=ref_price)
    gen.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    gen.eval()
    return gen


def _generate_bars(gen: BarGenerator, ref_price: float, n_bars: int = 100, seed_bars: list[Bar] | None = None):
    ex = BarExchange()
    if seed_bars:
        for b in seed_bars:
            ex.append_bar(b)
    else:
        ex.append_bar(Bar(1, ref_price, ref_price*1.005, ref_price*0.995, ref_price, 10000, ref_price))
    agent = BarWorldAgent(gen, ex, noise_dim=64, fixed_ref=ref_price)
    for _ in range(n_bars):
        agent.step()
    return ex.bars


@router.get("/cgan", response_class=HTMLResponse)
def cgan_dashboard(request: Request):
    models = _discover_models()
    return templates.TemplateResponse("cgan/generator.html", {"request": request, "models": models})


@router.post("/cgan/generate", response_class=JSONResponse)
def cgan_generate(
    model_path: str = Form(...),
    ref_price: float = Form(...),
    n_bars: int = Form(100),
):
    gen = _load_generator(model_path, ref_price)
    bars = _generate_bars(gen, ref_price, n_bars)
    chart_data = []
    for b in bars:
        chart_data.append({
            "t": b.timestamp,
            "o": round(b.open, 2),
            "h": round(b.high, 2),
            "l": round(b.low, 2),
            "c": round(b.close, 2),
            "v": round(b.volume),
            "vw": round(b.vwap, 2),
        })
    return {"bars": chart_data, "count": len(chart_data)}


@router.post("/cgan/compare", response_class=JSONResponse)
def cgan_compare(
    model_path: str = Form(...),
    ref_price: float = Form(...),
    n_bars: int = Form(50),
    n_trade_steps: int = Form(20),
):
    gen = _load_generator(model_path, ref_price)
    bars = _generate_bars(gen, ref_price, n_bars)
    price_series = pd.Series([b.close for b in bars])

    try:
        llm = KiloGateway(settings.default_model).get_langchain_llm()
        from langchain_core.messages import HumanMessage
        llm.invoke([HumanMessage(content="ping")])
    except Exception:
        from trading_agent.models.mock import MockGateway
        llm = MockGateway().get_langchain_llm()

    agents = [
        ("Risk-Averse", multicomponent_reward),
        ("Risk-Taker", aggressive_reward),
    ]
    from trading_agent.backtest.parallel import parallel_backtest, BacktestConfig
    configs = [
        BacktestConfig(price_series=price_series, llm=llm, max_steps=n_trade_steps, reward_fn=reward_fn)
        for _, reward_fn in agents
    ]
    parallel_results = parallel_backtest(configs, max_workers=2)
    results = {}
    for (label, _), metrics in zip(agents, parallel_results):
        results[label] = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}

    bar_data = [{"t": b.timestamp, "o": round(b.open, 2), "h": round(b.high, 2),
                  "l": round(b.low, 2), "c": round(b.close, 2), "v": round(b.volume)} for b in bars]

    return {"bars": bar_data, "results": results}