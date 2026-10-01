"""FastAPI backend for the predictor dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

TEMPLATES = Path(__file__).parent / "templates"


def create_app(state: dict) -> FastAPI:
    app = FastAPI(title="XVANTAGE BTC 15m Predictor", docs_url=None, redoc_url=None)
    index_html = (TEMPLATES / "index.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(index_html)

    @app.get("/api/state")
    async def api_state():
        return JSONResponse({
            "cycle": state.get("cycle", 0),
            "last_update": state.get("last_update"),
            "error": state.get("error"),
            "prediction": state.get("prediction"),
            "health": state.get("health"),
            "backtest": state.get("backtest"),
            "db_stats": state.get("db_stats"),
        })

    @app.get("/api/health")
    async def api_health():
        return JSONResponse(state.get("health") or {})

    return app
