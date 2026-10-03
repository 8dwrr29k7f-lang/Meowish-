"""
Multi-source live market data for BTC 15m prediction.
Coinbase primary (works from Railway US) + Kraken + optional Binance.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

COINBASE = "https://api.exchange.coinbase.com"
KRAKEN = "https://api.kraken.com"
BINANCE = "https://api.binance.com"  # often 451 from US cloud regions
BINANCE_US = "https://api.binance.us"


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
            len(self.errors) > 5 and self.price_age > 15
        )


class DataFeed:
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self._client = httpx.Client(timeout=10.0, follow_redirects=True)
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

    # ---- Coinbase (primary — works from US cloud) ----

    def _coinbase_ticker(self) -> dict:
        r = self._client.get(f"{COINBASE}/products/BTC-USD/ticker")
        r.raise_for_status()
        return r.json()

    def _coinbase_book(self) -> dict:
        r = self._client.get(
            f"{COINBASE}/products/BTC-USD/book", params={"level": 2}
        )
        r.raise_for_status()
        return r.json()

    def _coinbase_candles(self, granularity: int, limit: int = 100) -> List[Candle]:
        """granularity in seconds: 60, 300, 900, 3600, ..."""
        r = self._client.get(
            f"{COINBASE}/products/BTC-USD/candles",
            params={"granularity": granularity},
        )
        r.raise_for_status()
        rows = r.json()
        # Coinbase returns [time, low, high, open, close, volume] newest first
        rows = sorted(rows, key=lambda x: x[0])[-limit:]
        out = []
        for row in rows:
            out.append(
                Candle(
                    open_time=float(row[0]),
                    open=float(row[3]),
                    high=float(row[2]),
                    low=float(row[1]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        return out

    # ---- Kraken ----

    def _kraken_ticker(self) -> dict:
        r = self._client.get(
            f"{KRAKEN}/0/public/Ticker", params={"pair": "XBTUSD"}
        )
        r.raise_for_status()
        data = r.json()
        if data.get("error"):
            raise RuntimeError(str(data["error"]))
        return data["result"].get("XXBTZUSD") or data["result"].get("XBTUSD") or {}

    def _kraken_ohlc(self, interval: int, limit: int = 60) -> List[Candle]:
        """interval in minutes: 1, 5, 15, 60, ..."""
        r = self._client.get(
            f"{KRAKEN}/0/public/OHLC",
            params={"pair": "XBTUSD", "interval": interval},
        )
        r.raise_for_status()
        data = r.json()
        if data.get("error"):
            raise RuntimeError(str(data["error"]))
        result = data["result"]
        key = next(k for k in result if k != "last")
        rows = result[key][-limit:]
        out = []
        for row in rows:
            out.append(
                Candle(
                    open_time=float(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[6]),
                )
            )
        return out

    # ---- Binance (optional; often 451 from US) ----

    def _binance_klines(self, base: str, interval: str, limit: int = 100) -> List[Candle]:
        r = self._client.get(
            f"{base}/api/v3/klines",
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

    def _binance_ticker(self, base: str) -> dict:
        r = self._client.get(
            f"{base}/api/v3/ticker/bookTicker",
            params={"symbol": self.symbol},
        )
        r.raise_for_status()
        return r.json()

    def snapshot(self) -> MarketSnapshot:
        snap = MarketSnapshot()
        errors: List[str] = []

        def task(name, fn):
            try:
                return name, fn(), None
            except Exception as e:
                return name, None, str(e)[:180]

        jobs = [
            ("cb_ticker", lambda: self._cached("cb_ticker", 1.5, self._coinbase_ticker)),
            ("cb_book", lambda: self._cached("cb_book", 2.5, self._coinbase_book)),
            ("cb_1m", lambda: self._cached("cb_1m", 8.0, lambda: self._coinbase_candles(60, 120))),
            ("cb_5m", lambda: self._cached("cb_5m", 20.0, lambda: self._coinbase_candles(300, 60))),
            ("cb_15m", lambda: self._cached("cb_15m", 40.0, lambda: self._coinbase_candles(900, 60))),
            ("cb_1h", lambda: self._cached("cb_1h", 120.0, lambda: self._coinbase_candles(3600, 40))),
            ("kr_ticker", lambda: self._cached("kr_ticker", 2.0, self._kraken_ticker)),
            ("kr_1m", lambda: self._cached("kr_1m", 15.0, lambda: self._kraken_ohlc(1, 60))),
            ("kr_15m", lambda: self._cached("kr_15m", 45.0, lambda: self._kraken_ohlc(15, 40))),
            # Binance as soft fallback (may 451)
            ("bn_ticker", lambda: self._cached("bn_ticker", 5.0, lambda: self._binance_ticker(BINANCE))),
        ]

        results: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(task, n, fn) for n, fn in jobs]
            for fut in as_completed(futs):
                name, val, err = fut.result()
                if err:
                    # Downgrade Binance geo-blocks to quiet notes
                    if "451" in err or "binance" in name:
                        errors.append(f"{name}: geo-blocked/unavailable")
                    else:
                        errors.append(f"{name}: {err}")
                else:
                    results[name] = val

        # ---- Price + book from Coinbase ----
        cb = results.get("cb_ticker") or {}
        if cb:
            try:
                snap.price = float(cb.get("price") or 0)
                snap.bid = float(cb.get("bid") or 0)
                snap.ask = float(cb.get("ask") or 0)
                if snap.price > 0:
                    snap.price_ts = time.time()
                    snap.price_age = 0.0
            except (TypeError, ValueError):
                pass

        book = results.get("cb_book") or {}
        if book:
            bids = book.get("bids") or []
            asks = book.get("asks") or []
            bid_vol = sum(float(b[1]) for b in bids[:10]) if bids else 0.0
            ask_vol = sum(float(a[1]) for a in asks[:10]) if asks else 0.0
            total = bid_vol + ask_vol
            snap.imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0
            snap.book_stale = False
            if snap.price <= 0 and bids and asks:
                snap.bid = float(bids[0][0])
                snap.ask = float(asks[0][0])
                snap.price = (snap.bid + snap.ask) / 2
                snap.price_ts = time.time()
                snap.price_age = 0.0

        # Kraken cross-check / fallback price
        kr = results.get("kr_ticker") or {}
        if kr:
            try:
                # a = ask [price, whole lot, lot], b = bid, c = last trade
                last = float((kr.get("c") or [0])[0] or 0)
                bid = float((kr.get("b") or [0])[0] or 0)
                ask = float((kr.get("a") or [0])[0] or 0)
                if snap.price > 0 and last > 0:
                    snap.price_divergence_bps = abs(last - snap.price) / snap.price * 10000
                elif last > 0 and snap.price <= 0:
                    snap.price = last
                    snap.bid = bid or last
                    snap.ask = ask or last
                    snap.price_ts = time.time()
                    snap.price_age = 0.0
            except (TypeError, ValueError, IndexError):
                pass

        # Binance soft fallback price
        bn = results.get("bn_ticker") or {}
        if bn and snap.price <= 0:
            try:
                bid = float(bn.get("bidPrice") or 0)
                ask = float(bn.get("askPrice") or 0)
                if bid and ask:
                    snap.price = (bid + ask) / 2
                    snap.bid = bid
                    snap.ask = ask
                    snap.price_ts = time.time()
                    snap.price_age = 0.0
            except (TypeError, ValueError):
                pass

        # ---- Klines: prefer Coinbase, fill gaps with Kraken ----
        mapping = [
            ("1m", "cb_1m", "kr_1m"),
            ("5m", "cb_5m", None),
            ("15m", "cb_15m", "kr_15m"),
            ("1h", "cb_1h", None),
        ]
        for iv, primary, secondary in mapping:
            kl = results.get(primary) or (results.get(secondary) if secondary else None)
            if kl:
                snap.klines[iv] = kl
                if snap.price <= 0 and kl:
                    snap.price = kl[-1].close
                    snap.price_ts = kl[-1].open_time + (60 if iv == "1m" else 300)
                    snap.price_age = max(0.0, time.time() - snap.price_ts)

        # Approximate trade imbalance from recent 1m volume direction if no trades feed
        # (Coinbase doesn't give easy public trade aggression without auth)
        c1 = snap.klines.get("1m") or []
        if len(c1) >= 3 and snap.trade_imbalance == 0.0:
            ups = sum(1 for c in c1[-5:] if c.close >= c.open)
            downs = len(c1[-5:]) - ups
            tot = ups + downs
            if tot:
                snap.trade_imbalance = (ups - downs) / tot * 0.3  # muted proxy

        snap.errors = errors
        if snap.price_ts:
            snap.price_age = max(0.0, time.time() - snap.price_ts)
        self.last_snapshot = snap
        return snap
