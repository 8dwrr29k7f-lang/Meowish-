#!/usr/bin/env python3
"""
XVANTAGE BTC 15-Minute Predictor
--------------------------------
Multi-model ensemble with live data, regime detection, self-critique,
prediction logging, walk-forward backtest, and clean real-time UI.

Education / research only. Not financial advice.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

# Ensure package root on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.predictor import Predictor, Prediction
from learning.database import PredictionDB
from learning.monitor import check_health

POLL = float(os.getenv("POLL_SECONDS", "2.0"))
SERIES = os.getenv("SERIES", "KXBTC15M")

STATE: dict[str, Any] = {
    "prediction": None,
    "health": None,
    "backtest": None,
    "db_stats": {},
    "cycle": 0,
    "error": None,
    "last_update": None,
    "kalshi": None,
}


def resolve_pending(db: PredictionDB, predictor: Predictor) -> int:
    """Resolve predictions whose 15m window has elapsed using current price."""
    pending = db.pending_resolutions()
    if not pending:
        return 0
    snap = predictor.feed.last_snapshot
    if not snap or snap.price <= 0:
        return 0
    n = 0
    for row in pending:
        entry_px = row["price"]
        if not entry_px or entry_px <= 0:
            continue
        ret = (snap.price / entry_px - 1.0) * 100
        actual = "UP" if ret > 0 else "DOWN"
        db.resolve(row["id"], actual, ret)
        n += 1
    return n


def cycle_once(predictor: Predictor, db: PredictionDB, log_every: bool = True) -> Prediction:
    pred = predictor.predict()
    STATE["prediction"] = pred.to_dict()
    STATE["last_update"] = datetime.now(timezone.utc).isoformat()
    STATE["error"] = None if pred.usable else (pred.data_errors or ["data not usable"])

    if log_every and pred.usable and pred.direction != "NEUTRAL":
        # Log directional signals at most once per ~45s
        last_log = STATE.get("_last_log_ts") or 0
        if time.time() - last_log >= 45:
            db.log_prediction(
                direction=pred.direction,
                p_up=pred.p_up,
                p_down=pred.p_down,
                confidence=pred.confidence,
                price=pred.price,
                regime=pred.regime,
                models=pred.model_scores,
                notes=pred.critique_notes,
            )
            STATE["_last_log_ts"] = time.time()

    resolve_pending(db, predictor)
    STATE["db_stats"] = db.stats()
    STATE["health"] = check_health(db)
    STATE["cycle"] = STATE.get("cycle", 0) + 1
    return pred


def main_cli() -> None:
    predictor = Predictor()
    db = PredictionDB()
    print("XVANTAGE BTC 15m predictor (CLI)")
    print(f"  poll={POLL}s series={SERIES}")
    while True:
        try:
            pred = cycle_once(predictor, db)
            ts = datetime.now().strftime("%H:%M:%S")
            if pred.usable:
                print(
                    f"[{ts}] {pred.direction:7s}  "
                    f"p_up={pred.p_up:.1%} p_down={pred.p_down:.1%}  "
                    f"conf={pred.confidence:.2f}  regime={pred.regime}  "
                    f"px={pred.price:.2f}"
                )
            else:
                print(f"[{ts}] DATA NOT USABLE: {pred.data_errors}")
        except Exception as e:
            STATE["error"] = str(e)
            print(f"cycle error: {e}")
        time.sleep(POLL)


def main_web(host: str = "0.0.0.0", port: int = 8080) -> None:
    from webapp import create_app

    predictor = Predictor()
    db = PredictionDB()

    def poller() -> None:
        # one-shot backtest on start
        try:
            from backtest.engine import run_walk_forward
            bt = run_walk_forward(bars_15m=400)
            STATE["backtest"] = {
                "n": bt.n,
                "accuracy": bt.accuracy,
                "f1": bt.f1,
                "brier": bt.brier,
                "baseline_prev_acc": bt.baseline_prev_acc,
                "neutral_rate": bt.neutral_rate,
                "by_regime": bt.by_regime,
                "notes": bt.notes,
            }
        except Exception as e:
            STATE["backtest"] = {"error": str(e)}

        while True:
            try:
                cycle_once(predictor, db)
            except Exception as e:
                STATE["error"] = str(e)
            time.sleep(POLL)

    t = threading.Thread(target=poller, daemon=True)
    t.start()

    app = create_app(STATE)
    import uvicorn
    print(f"\n  XVANTAGE Predictor → http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")))
    parser.add_argument("--backtest-only", action="store_true")
    args = parser.parse_args()

    if args.backtest_only:
        from backtest.engine import run_walk_forward
        bt = run_walk_forward(bars_15m=500)
        print("WALK-FORWARD BACKTEST")
        print(f"  n={bt.n}  accuracy={bt.accuracy:.2%}  f1={bt.f1:.3f}  brier={bt.brier:.4f}")
        print(f"  baseline prev-candle={bt.baseline_prev_acc:.2%}  random=50%")
        print(f"  neutral_rate={bt.neutral_rate:.1%}")
        print(f"  by_regime={bt.by_regime}")
        for n in bt.notes:
            print(f"  note: {n}")
        sys.exit(0)

    if args.web:
        main_web(args.host, args.port)
    else:
        main_cli()
