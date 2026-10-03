"""SQLite prediction log + outcome resolution."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class PredictionDB:
    def __init__(self, path: str = "data/predictions.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.path), timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _init(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    resolve_after REAL NOT NULL,
                    direction TEXT,
                    p_up REAL,
                    p_down REAL,
                    confidence REAL,
                    price REAL,
                    regime TEXT,
                    agreement REAL,
                    model_version TEXT,
                    features_json TEXT,
                    signals_json TEXT,
                    actual_direction TEXT,
                    actual_return_pct REAL,
                    resolved INTEGER DEFAULT 0,
                    correct INTEGER,
                    created_at REAL
                )
                """
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(ts)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_pred_resolved ON predictions(resolved)"
            )

    def log_prediction(
        self,
        direction: str,
        p_up: float,
        p_down: float,
        confidence: float,
        price: float,
        regime: str = "",
        agreement: float = 0.0,
        model_version: str = "v3",
        features: Optional[dict] = None,
        signals: Optional[list] = None,
        models: Optional[list] = None,
        notes: Optional[list] = None,
        horizon_sec: float = 900.0,
        **kwargs: Any,
    ) -> int:
        now = time.time()
        sig = signals or notes or models or []
        with self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO predictions (
                    ts, resolve_after, direction, p_up, p_down, confidence,
                    price, regime, agreement, model_version, features_json,
                    signals_json, resolved, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?)
                """,
                (
                    now,
                    now + horizon_sec,
                    direction,
                    p_up,
                    p_down,
                    confidence,
                    price,
                    regime,
                    agreement,
                    model_version,
                    json.dumps(features or {}),
                    json.dumps(sig),
                    now,
                ),
            )
            return int(cur.lastrowid)

    def pending_resolutions(self) -> List[dict]:
        now = time.time()
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM predictions WHERE resolved=0 AND resolve_after<=? ORDER BY id",
                (now,),
            ).fetchall()
            return [dict(r) for r in rows]

    def resolve(self, pred_id: int, actual: str, ret_pct: float) -> None:
        with self._conn() as c:
            row = c.execute(
                "SELECT direction FROM predictions WHERE id=?", (pred_id,)
            ).fetchone()
            if not row:
                return
            correct = 1 if row["direction"] == actual else 0
            if row["direction"] == "NEUTRAL":
                correct = None
            c.execute(
                """
                UPDATE predictions SET resolved=1, actual_direction=?,
                actual_return_pct=?, correct=? WHERE id=?
                """,
                (actual, ret_pct, correct, pred_id),
            )

    def stats(self) -> Dict[str, Any]:
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            resolved = c.execute(
                "SELECT COUNT(*) FROM predictions WHERE resolved=1"
            ).fetchone()[0]
            directional = c.execute(
                "SELECT COUNT(*) FROM predictions WHERE resolved=1 AND direction!='NEUTRAL'"
            ).fetchone()[0]
            wins = c.execute(
                "SELECT COUNT(*) FROM predictions WHERE correct=1"
            ).fetchone()[0]
        acc = (wins / directional) if directional else None
        return {
            "total": total,
            "resolved": resolved,
            "directional": directional,
            "wins": wins,
            "accuracy": acc,
        }

    def recent_accuracy(self, n: int = 50) -> Dict[str, Any]:
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT correct, p_up, direction, actual_direction FROM predictions
                WHERE resolved=1 AND direction!='NEUTRAL'
                ORDER BY id DESC LIMIT ?
                """,
                (n,),
            ).fetchall()
        if not rows:
            return {"n": 0, "accuracy": None, "brier": None}
        correct = [r["correct"] for r in rows if r["correct"] is not None]
        acc = sum(correct) / len(correct) if correct else None
        brier_vals = []
        for r in rows:
            if r["actual_direction"] in ("UP", "DOWN"):
                y = 1.0 if r["actual_direction"] == "UP" else 0.0
                brier_vals.append((r["p_up"] - y) ** 2)
        brier = sum(brier_vals) / len(brier_vals) if brier_vals else None
        return {"n": len(correct), "accuracy": acc, "brier": brier}
