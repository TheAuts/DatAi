"""MarketData.app options ingest for the DatAi MVP.

Uses GET /v1/options/chain/{underlyingSymbol}/. Returns a DataFrame compatible
with quant_engine.calculate_greeks (S, K, T, sigma, option_type).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv

from quant_engine import _time_to_expiry_years

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_PATH = PROJECT_ROOT / "logs" / "ingestion.log"
API_BASE = "https://api.marketdata.app/v1"
CHAIN_PATH = "/options/chain/{symbol}/"
EXPIRATIONS_PATH = "/options/expirations/{symbol}/"
QUOTES_PATH = "/stocks/quotes/{symbol}/"
REQUEST_TIMEOUT_SEC = 30
EASTERN = ZoneInfo("America/New_York")

load_dotenv(PROJECT_ROOT / ".env")

CHAIN_COLUMNS = (
    "ticker",
    "expiration",
    "option_type",
    "strike",
    "lastPrice",
    "impliedVolatility",
    "bid",
    "ask",
    "underlyingPrice",
    "volume",
    "openInterest",
    "S",
    "K",
    "T",
    "sigma",
)


def _setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("datiai.ingestion")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


log = _setup_logger()


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    ticker: str
    message: str
    expiry: str | None = None
    expirations: tuple[str, ...] = ()


def _normalize_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper()


def _expiry_iso(expiry: Any) -> str | None:
    if expiry is None or (isinstance(expiry, float) and pd.isna(expiry)):
        return None
    if isinstance(expiry, datetime):
        return expiry.date().isoformat()
    if isinstance(expiry, date):
        return expiry.isoformat()
    text = str(expiry).strip()[:10]
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def get_api_key() -> str | None:
    key = os.getenv("MARKETDATA_API_KEY", "").strip()
    return key or None


def _empty_chain() -> pd.DataFrame:
    return pd.DataFrame(columns=list(CHAIN_COLUMNS))


def _headers() -> dict[str, str]:
    key = get_api_key()
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _marketdata_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    url = f"{API_BASE}{path}"
    query = dict(params or {})
    try:
        response = requests.get(url, headers=_headers(), params=query, timeout=REQUEST_TIMEOUT_SEC)
    except requests.RequestException as exc:
        log.error("MarketData request failed url=%s error=%s", path, exc)
        return None
    if response.status_code >= 400:
        log.error("MarketData HTTP %s url=%s body=%s", response.status_code, path, response.text[:300])
        return None
    try:
        payload = response.json()
    except ValueError:
        log.error("MarketData non-JSON url=%s body=%s", path, response.text[:300])
        return None
    if not isinstance(payload, dict):
        return None
    status = str(payload.get("s") or "").lower()
    if status and status not in {"ok", "no_data"}:
        log.error("MarketData error url=%s errmsg=%s", path, payload.get("errmsg") or payload)
        return None
    return payload


def _unix_to_iso(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str) and len(value) >= 10 and value[4] == "-":
        return _expiry_iso(value)
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return _expiry_iso(value)
    if ts > 1e12:
        ts /= 1000.0
    try:
        return datetime.fromtimestamp(ts, tz=EASTERN).date().isoformat()
    except (OSError, OverflowError, ValueError):
        return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()


def _series(payload: dict[str, Any], *names: str) -> list[Any]:
    for name in names:
        value = payload.get(name)
        if isinstance(value, list):
            return value
    return []


def _coerce_iv(value: Any) -> float | None:
    """Return a positive finite IV or None when the API sends null/invalid."""
    if value is None or value == "" or str(value).strip().lower() in {"null", "nan", "none"}:
        return None
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return None
    number = float(parsed)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _payload_to_frame(payload: dict[str, Any], ticker: str) -> pd.DataFrame:
    strikes = _series(payload, "strike")
    if not strikes:
        return _empty_chain()
    n = len(strikes)
    expirations = _series(payload, "expiration")
    sides = _series(payload, "side")
    bids = _series(payload, "bid")
    asks = _series(payload, "ask")
    lasts = _series(payload, "last")
    mids = _series(payload, "mid")
    ivs = _series(payload, "iv", "impliedVolatility", "implied_volatility")
    spots = _series(payload, "underlyingPrice")
    volumes = _series(payload, "volume")
    open_interest = _series(payload, "openInterest")

    def at(values: list[Any], index: int) -> Any:
        return values[index] if index < len(values) else None

    rows: list[dict[str, Any]] = []
    for i in range(n):
        strike = pd.to_numeric(at(strikes, i), errors="coerce")
        if pd.isna(strike) or float(strike) <= 0:
            continue
        expiration = _unix_to_iso(at(expirations, i))
        if not expiration:
            continue
        last = pd.to_numeric(at(lasts, i), errors="coerce")
        mid = pd.to_numeric(at(mids, i), errors="coerce")
        last_price = last if pd.notna(last) else mid
        iv = _coerce_iv(at(ivs, i))
        spot = pd.to_numeric(at(spots, i), errors="coerce")
        side = str(at(sides, i) or "call").strip().lower()
        rows.append(
            {
                "ticker": ticker,
                "expiration": expiration,
                "option_type": "put" if side.startswith("p") else "call",
                "strike": float(strike),
                "lastPrice": None if pd.isna(last_price) else float(last_price),
                "impliedVolatility": iv,
                "bid": None if pd.isna(pd.to_numeric(at(bids, i), errors="coerce")) else float(at(bids, i)),
                "ask": None if pd.isna(pd.to_numeric(at(asks, i), errors="coerce")) else float(at(asks, i)),
                "underlyingPrice": None if pd.isna(spot) else float(spot),
                "volume": pd.to_numeric(at(volumes, i), errors="coerce"),
                "openInterest": pd.to_numeric(at(open_interest, i), errors="coerce"),
                "S": None if pd.isna(spot) else float(spot),
                "K": float(strike),
                "T": _time_to_expiry_years(expiration),
                "sigma": iv,
            }
        )
    if not rows:
        return _empty_chain()
    return pd.DataFrame(rows).loc[:, list(CHAIN_COLUMNS)]


def _with_date_query(path: str, date: str | None) -> str:
    """Append ``date=YYYY-MM-DD`` to ``path`` (``?`` or ``&`` as appropriate) for historical EOD data."""
    text = str(date or "").strip()
    if not text:
        return path
    separator = "&" if "?" in path else "?"
    return f"{path}{separator}date={text}"


def fetch_option_chain(ticker: str, expiry: Any = None, date: str | None = None) -> pd.DataFrame:
    """Fetch the MarketData.app options chain for ``ticker``.

    Args:
        ticker: Underlying symbol.
        expiry: Optional ``YYYY-MM-DD``. If omitted, the API returns the next monthly expiry.
        date: Optional ``YYYY-MM-DD``. When given, requests the historical end-of-day
            chain as of that date (``?date=`` query parameter) instead of live data.

    Returns:
        DataFrame with strike, expiration, bid, ask, underlyingPrice, plus S/K/T/sigma.
        Empty DataFrame if the ticker has no chain or the request fails.
    """
    symbol = _normalize_ticker(ticker)
    if not symbol:
        return _empty_chain()
    if not get_api_key():
        log.error("MARKETDATA_API_KEY is missing")
        return _empty_chain()

    params: dict[str, Any] = {}
    expiry_iso = _expiry_iso(expiry)
    if expiry_iso:
        params["expiration"] = expiry_iso

    path = _with_date_query(CHAIN_PATH.format(symbol=symbol), _expiry_iso(date) if date else None)
    payload = _marketdata_get(path, params)
    if not payload or str(payload.get("s") or "").lower() == "no_data":
        return _empty_chain()
    try:
        return _payload_to_frame(payload, symbol)
    except Exception as exc:
        log.error("Failed to parse option chain ticker=%s error=%s", symbol, exc)
        return _empty_chain()


def fetch_options_data(ticker: str, expiry: Any = None) -> pd.DataFrame:
    """Alias used by the dashboard/CLI."""
    return fetch_option_chain(ticker, expiry)


def fetch_historical_options(ticker: str, expiry: Any = None, **_: Any) -> pd.DataFrame:
    return fetch_option_chain(ticker, expiry)


def get_available_expirations(ticker: str) -> list[str]:
    """Return listed expirations, or [] if none / on error."""
    symbol = _normalize_ticker(ticker)
    if not symbol:
        return []
    payload = _marketdata_get(EXPIRATIONS_PATH.format(symbol=symbol))
    if payload:
        raw = _series(payload, "expirations", "expiration")
        out: list[str] = []
        for item in raw:
            iso = _unix_to_iso(item) or _expiry_iso(item)
            if iso:
                out.append(iso)
        if out:
            return sorted(dict.fromkeys(out))
    frame = fetch_option_chain(symbol)
    if frame.empty or "expiration" not in frame.columns:
        return []
    return sorted(dict.fromkeys(str(value) for value in frame["expiration"].dropna()))


def get_current_price(ticker: str) -> float | None:
    """Return underlying last price from quotes, else from the option chain."""
    symbol = _normalize_ticker(ticker)
    if not symbol:
        return None
    payload = _marketdata_get(QUOTES_PATH.format(symbol=symbol))
    if payload:
        last = _series(payload, "last", "mid")
        if last:
            price = pd.to_numeric(last[0], errors="coerce")
            if pd.notna(price) and float(price) > 0:
                return float(price)
    frame = fetch_option_chain(symbol)
    if frame.empty or "underlyingPrice" not in frame.columns:
        return None
    spots = pd.to_numeric(frame["underlyingPrice"], errors="coerce").dropna()
    spots = spots.loc[spots > 0]
    if spots.empty:
        return None
    return float(spots.iloc[0])


def preflight_check(ticker: str, expiry: Any = None) -> PreflightResult:
    """Verify API key, ticker, and optional expiration before a bulk fetch."""
    symbol = _normalize_ticker(ticker)
    if not symbol:
        return PreflightResult(False, ticker, "Ticker is empty.")
    if not get_api_key():
        return PreflightResult(False, symbol, "MARKETDATA_API_KEY is missing. Set it in .env.")

    expirations = get_available_expirations(symbol)
    if not expirations:
        frame = fetch_option_chain(symbol, expiry)
        if frame.empty:
            return PreflightResult(False, symbol, f"No option chain for {symbol}.")
        expirations = sorted(dict.fromkeys(str(value) for value in frame["expiration"].dropna()))

    expiry_iso = _expiry_iso(expiry)
    if expiry_iso is None:
        return PreflightResult(
            True,
            symbol,
            f"Ticker {symbol} has {len(expirations)} listed expirations.",
            expirations=tuple(expirations),
        )
    if expiry_iso not in expirations:
        return PreflightResult(
            False,
            symbol,
            f"No option chain for {symbol} on {expiry_iso}.",
            expiry=expiry_iso,
            expirations=tuple(expirations),
        )
    return PreflightResult(
        True,
        symbol,
        f"Ticker {symbol} has an option chain for {expiry_iso}.",
        expiry=expiry_iso,
        expirations=tuple(expirations),
    )


def _nearest_expiration(requested: str, listed: list[str]) -> str | None:
    if not listed:
        return None
    if requested in listed:
        return requested
    target = date.fromisoformat(requested)
    return min(listed, key=lambda iso: abs((date.fromisoformat(iso) - target).days))


def get_contract_iv(
    ticker: str,
    expiry: Any,
    strike: float,
    option_type: str = "call",
) -> float | None:
    """Return impliedVolatility for one contract, or None if missing/null."""
    try:
        symbol = _normalize_ticker(ticker)
        expiry_iso = _expiry_iso(expiry)
        k = float(strike)
        side = "put" if str(option_type).strip().lower().startswith("p") else "call"
        if not symbol or not expiry_iso or k <= 0:
            return None
        frame = fetch_option_chain(symbol, expiry_iso)
        if frame.empty:
            return None
        strikes = pd.to_numeric(frame["strike"], errors="coerce")
        types = frame["option_type"].astype(str).str.lower()
        matched = frame.loc[(types == side) & ((strikes - k).abs() < 1e-6)]
        if matched.empty and strikes.notna().any():
            nearest = (strikes - k).abs()
            matched = frame.loc[types == side]
            if not matched.empty:
                nearest_side = (pd.to_numeric(matched["strike"], errors="coerce") - k).abs()
                matched = matched.loc[[nearest_side.idxmin()]]
            else:
                matched = frame.loc[[nearest.idxmin()]]
        if matched.empty:
            return None
        return _coerce_iv(matched.iloc[0].get("impliedVolatility"))
    except Exception as exc:
        log.error("get_contract_iv failed ticker=%s error=%s", ticker, exc)
        return None


def get_available_strikes(ticker: str, expiry: Any) -> list[float]:
    """Return unique strikes for ``ticker`` on ``expiry``, or [] if none."""
    try:
        symbol = _normalize_ticker(ticker)
        expiry_iso = _expiry_iso(expiry)
        if not symbol or not expiry_iso:
            return []
        listed = get_available_expirations(symbol)
        resolved = _nearest_expiration(expiry_iso, listed) if listed else expiry_iso
        if resolved is None:
            return []
        frame = fetch_option_chain(symbol, resolved)
        if frame.empty or "strike" not in frame.columns:
            return []
        strikes = pd.to_numeric(frame["strike"], errors="coerce").dropna()
        strikes = strikes.loc[strikes > 0].unique()
        return sorted(float(value) for value in strikes)
    except Exception as exc:
        log.error("get_available_strikes failed ticker=%s expiry=%s error=%s", ticker, expiry, exc)
        return []


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DatAi MarketData.app options ingest.")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--expiration", default=None, help="YYYY-MM-DD; default next monthly.")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _build_cli().parse_args()
    symbol = _normalize_ticker(args.ticker)
    if args.preflight_only:
        result = preflight_check(symbol, args.expiration)
        print(json.dumps(asdict(result), indent=2))
        return 0 if result.ok else 1
    frame = fetch_option_chain(symbol, args.expiration)
    print(frame.head().to_json(orient="records", indent=2))
    print(f"rows={len(frame)}")
    return 0 if not frame.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
