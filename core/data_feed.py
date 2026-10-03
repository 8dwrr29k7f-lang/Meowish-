"""
Multi-source live market data for BTC 15m prediction.
Binance klines / depth / trades + Coinbase cross-check.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

BINANCE = "https://api.binance.com"
COINBASE = "https://api.exchange.coinbase.com"


@dataclass
class Candle:
    open_time: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class MarketSnapshot:
    price: float = 0.0
    price_ts: float = 0.0
    price_age: float = 9999.0
    bid: float = 0.0
    ask: float = 0.0
    imbalance: float = 0.0
    trade_imbalance: float = 0.0
    funding_rate: Optional[float] = None
    book_stale: bool = True
    price_divergence_bps: float = 0.0
    klines: Dict[str, List[Candle]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def is_usable(self) -> bool:
        return self.price > 0 and self.price_age < 30 and not (
            len(self.errors) > 3 and self.price_age > 15
        )


class DataFeed:
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self._client = httpx.Client(timeout=8.0)
        self.last_snapshot: Optional[MarketSnapshot] = None
        self._cache: Dict[str, Any] = {}
        self._cache_ts: Dict[str, float] = {}

    def _cached(self, key: str, ttl: float, fn):
        now = time.time()
        if key in self._cache and now - self._cache_ts.get(key, 0) < ttl:
            return self._cache[key]
        try:
            val = fn()
            self._cache[key] = val
            self._cache_ts[key] = now
            return val
        except Exception as e:
            if key in self._cache:
                return self._cache[key]
            raise

    def _binance_klines(self, interval: str, limit: int = 100) -> List[Candle]:
        r = self._client.get(
            f"{BINANCE}/api/v3/klines",
            params={"symbol": self.symbol, "interval": interval, "limit": limit},
        )
        r.raise_for_status()
        out = []
        for row in r.json():
            out.append(
                Candle(
                    open_time=row[0] / 1000.0,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        return out

    def _binance_ticker(self) -> dict:
        r = self._client.get(
            f"{BINANCE}/api/v3/ticker/bookTicker",
            params={"symbol": self.symbol},
        )
        r.raise_for_status()
        return r.json()

    def _binance_depth(self) -> dict:
        r = self._client.get(
            f"{BINANCE}/api/v3/depth",
            params={"symbol": self.symbol, "limit": 20},
        )
        r.raise_for_status()
        return r.json()

    def _binance_trades(self) -> list:
        r = self._client.get(
            f"{BINANCE}/api/v3/trades",
            params={"symbol": self.symbol, "limit": 50},
        )
        r.raise_for_status()
        return r.json()

    def _coinbase_price(self) -> Optional[float]:
        try:
            r = self._client.get(f"{COINBASE}/products/BTC-USD/ticker")
            r.raise_for_status()
            return float(r.json().get("price") or 0)
        except Exception:
            return None

    def snapshot(self) -> MarketSnapshot:
        snap = MarketSnapshot()
        errors = []

        def task(name, fn):
            try:
                return name, fn(), None
            except Exception as e:
                return name, None, str(e)

        jobs = [
            ("ticker", lambda: self._cached("ticker", 1.5, self._binance_ticker)),
            ("depth", lambda: self._cached("depth", 2.0, self._binance_depth)),
            ("trades", lambda: self._cached("trades", 2.0, self._binance_trades)),
            ("k1m", lambda: self._cached("k1m", 5.0, lambda: self._binance_klines("1m", 120))),
            ("k5m", lambda: self._cached("k5m", 15.0, lambda: self._binance_klines("5m", 60))),
            ("k15m", lambda: self._cached("k15m", 30.0, lambda: self._binance_klines("15m", 60))),
            ("k30m", lambda: self._cached("k30m", 60.0, lambda: self._binance_klines("30m", 40))),
            ("k1h", lambda: self._cached("k1h", 120.0, lambda: self._binance_klines("1h", 40))),
            ("cb", lambda: self._cached("cb", 5.0, self._coinbase_price)),
        ]

        results = {}
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = [ex.submit(task, n, fn) for n, fn in jobs]
            for fut in as_completed(futs):
                name, val, err = fut.result()
                if err:
                    errors.append(f"{name}: {err}")
                else:
                    results[name] = val

        ticker = results.get("ticker") or {}
        if ticker:
            snap.bid = float(ticker.get("bidPrice") or 0)
            snap.ask = float(ticker.get("askPrice") or 0)
            if snap.bid and snap.ask:
                snap.price = (snap.bid + snap.ask) / 2
                snap.price_ts = time.time()
                snap.price_age = 0.0

        depth = results.get("depth") or {}
        if depth:
            bids = depth.get("bids") or []
            asks = depth.get("asks") or []
            bid_vol = sum(float(b[1]) for b in bids[:10]) if bids else 0
            ask_vol = sum(float(a[1]) for a in asks[:10]) if asks else 0
            total = bid_vol + ask_vol
            snap.imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0
            snap.book_stale = False

        trades = results.get("trades") or []
        if trades:
            buy = sum(float(t["qty"]) for t in trades if not t.get("isBuyerMaker"))
            sell = sum(float(t["qty"]) for t in trades if t.get("isBuyerMaker"))
            tot = buy + sell
            snap.trade_imbalance = (buy - sell) / tot if tot > 0 else 0.0

        for key, iv in [("k1m", "1m"), ("k5m", "5m"), ("k15m", "15m"), ("k30m", "30m"), ("k1h", "1h")]:
            kl = results.get(key)
            if kl:
                snap.klines[iv] = kl
                if snap.price <= 0 and kl:
                    snap.price = kl[-1].close
                    snap.price_ts = kl[-1].open_time + 60
                    snap.price_age = max(0.0, time.time() - snap.price_ts)

        cb = results.get("cb")
        if cb and snap.price > 0:
            snap.price_divergence_bps = abs(cb - snap.price) / snap.price * 10000

        snap.errors = errors
        if snap.price_ts:
            snap.price_age = max(0.0, time.time() - snap.price_ts)
        self.last_snapshot = snap
        return snap
