"""FastAPI backend for the predictor dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

TEMPLATES = Path(__file__).parent / "templates"


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion so JSONResponse never 500s on dataclasses/etc."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _jsonable(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return _jsonable(vars(obj))
    return str(obj)


def create_app(state: dict) -> FastAPI:
    app = FastAPI(title="XVANTAGE BTC 15m Predictor", docs_url=None, redoc_url=None)
    index_html = (TEMPLATES / "index.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(index_html)

    @app.get("/api/state")
    async def api_state():
        try:
            payload = {
                "cycle": state.get("cycle", 0),
                "last_update": state.get("last_update"),
                "error": state.get("error"),
                "prediction": state.get("prediction"),
                "health": state.get("health"),
                "backtest": state.get("backtest"),
                "db_stats": state.get("db_stats"),
            }
            return JSONResponse(_jsonable(payload))
        except Exception as e:
            return JSONResponse(
                {"error": f"state serialization failed: {e}", "cycle": state.get("cycle", 0)},
                status_code=500,
            )

    @app.get("/api/health")
    async def api_health():
        try:
            return JSONResponse(_jsonable(state.get("health") or {}))
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    return app
