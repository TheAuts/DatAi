"""MarketData.app options ingest for the DatAi MVP.

Uses GET /v1/options/chain/{underlyingSymbol}/. Returns a DataFrame compatible
with quant_engine.calculate_greeks (S, K, T, sigma, option_type).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
import math
from datetime import date, datetime, timedelta, timezone
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
API_RETRY_ATTEMPTS = 3
API_RETRY_BACKOFF_SEC = 0.5
HEALTH_CHECK_SYMBOL = "SPY"
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

# Last API call outcome for dashboard / callers (status_code may be int or None).
_LAST_API_ERROR: dict[str, Any] = {
    "ok": True,
    "status_code": None,
    "message": "",
    "token_missing": False,
    "attempts": 0,
}


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


def clear_last_api_error() -> None:
    """Reset the module-level last-error record (used by tests and callers)."""
    _LAST_API_ERROR.update(
        {"ok": True, "status_code": None, "message": "", "token_missing": False, "attempts": 0}
    )


def get_last_api_error() -> dict[str, Any]:
    """Return a shallow copy of the last API outcome for UI / diagnostics."""
    return dict(_LAST_API_ERROR)


def _record_api_error(
    *,
    status_code: Any = None,
    message: str = "",
    token_missing: bool = False,
    attempts: int = 0,
    ok: bool = False,
) -> None:
    _LAST_API_ERROR.update(
        {
            "ok": bool(ok),
            "status_code": status_code,
            "message": str(message or ""),
            "token_missing": bool(token_missing),
            "attempts": int(attempts),
        }
    )


def _empty_chain() -> pd.DataFrame:
    return pd.DataFrame(columns=list(CHAIN_COLUMNS))


def is_mock_frame(frame: Any) -> bool:
    """True when ``frame`` is a mock fallback produced after an API failure."""
    if not isinstance(frame, pd.DataFrame):
        return False
    try:
        return bool(frame.attrs.get("is_mock"))
    except Exception:
        return False


def mock_option_chain(
    ticker: str = "SPY",
    *,
    error_code: Any = None,
    token_missing: bool = False,
    message: str = "",
) -> pd.DataFrame:
    """Minimal valid chain DataFrame so the dashboard does not crash on API failure.

    Columns match ``CHAIN_COLUMNS`` (what the dashboard / engine expect). Marked
    with ``attrs['is_mock']=True`` so callers must not treat it as a live fetch.
    """
    symbol = _normalize_ticker(ticker) or "SPY"
    expiry = (date.today() + timedelta(days=30)).isoformat()
    spot = 100.0
    iv = 0.20
    frame = pd.DataFrame(
        [
            {
                "ticker": symbol,
                "expiration": expiry,
                "option_type": "call",
                "strike": spot,
                "lastPrice": 1.0,
                "impliedVolatility": iv,
                "bid": 0.9,
                "ask": 1.1,
                "underlyingPrice": spot,
                "volume": 0,
                "openInterest": 0,
                "S": spot,
                "K": spot,
                "T": _time_to_expiry_years(expiry),
                "sigma": iv,
            }
        ]
    ).loc[:, list(CHAIN_COLUMNS)]
    frame.attrs["is_mock"] = True
    frame.attrs["token_missing"] = bool(token_missing)
    frame.attrs["api_error"] = error_code
    frame.attrs["api_message"] = str(message or "")
    return frame


def _headers() -> dict[str, str]:
    key = get_api_key()
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _marketdata_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """GET MarketData with simple retry (3 attempts) for transient network blips."""
    url = f"{API_BASE}{path}"
    query = dict(params or {})
    last_status: Any = None
    last_message = ""
    attempts = 0
    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        attempts = attempt
        try:
            response = requests.get(url, headers=_headers(), params=query, timeout=REQUEST_TIMEOUT_SEC)
        except requests.RequestException as exc:
            last_status = None
            last_message = str(exc)
            log.error(
                "MarketData request failed url=%s attempt=%s/%s status=%s error=%s",
                path,
                attempt,
                API_RETRY_ATTEMPTS,
                last_status,
                exc,
            )
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_BACKOFF_SEC * attempt)
                continue
            _record_api_error(status_code=None, message=last_message, attempts=attempts)
            return None
        last_status = int(response.status_code)
        if response.status_code >= 400:
            last_message = f"HTTP {response.status_code}: {response.text[:300]}"
            log.error(
                "MarketData HTTP %s url=%s attempt=%s/%s body=%s",
                response.status_code,
                path,
                attempt,
                API_RETRY_ATTEMPTS,
                response.text[:300],
            )
            # Retry transient server errors; permanent client errors stop early.
            if response.status_code >= 500 and attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_BACKOFF_SEC * attempt)
                continue
            _record_api_error(status_code=last_status, message=last_message, attempts=attempts)
            return None
        try:
            payload = response.json()
        except ValueError:
            last_message = f"non-JSON body: {response.text[:300]}"
            log.error("MarketData non-JSON url=%s body=%s", path, response.text[:300])
            _record_api_error(status_code=last_status, message=last_message, attempts=attempts)
            return None
        if not isinstance(payload, dict):
            last_message = "payload is not a JSON object"
            _record_api_error(status_code=last_status, message=last_message, attempts=attempts)
            return None
        status = str(payload.get("s") or "").lower()
        if status and status not in {"ok", "no_data"}:
            last_message = str(payload.get("errmsg") or payload)
            log.error("MarketData error url=%s errmsg=%s", path, last_message)
            _record_api_error(status_code=last_status, message=last_message, attempts=attempts)
            return None
        _record_api_error(status_code=last_status, message="ok", attempts=attempts, ok=True)
        return payload
    _record_api_error(status_code=last_status, message=last_message or "request failed", attempts=attempts)
    return None


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
        On request failure or missing token, returns a mock DataFrame (``attrs['is_mock']``)
        so the dashboard does not crash; check ``get_last_api_error()`` / attrs for status.
    """
    symbol = _normalize_ticker(ticker)
    if not symbol:
        return _empty_chain()
    if not get_api_key():
        log.error("MARKETDATA_API_KEY is missing")
        _record_api_error(
            status_code=None,
            message="API Token Missing",
            token_missing=True,
            attempts=0,
        )
        return mock_option_chain(symbol, token_missing=True, message="API Token Missing")

    params: dict[str, Any] = {}
    expiry_iso = _expiry_iso(expiry)
    if expiry_iso:
        params["expiration"] = expiry_iso

    path = _with_date_query(CHAIN_PATH.format(symbol=symbol), _expiry_iso(date) if date else None)
    try:
        payload = _marketdata_get(path, params)
    except Exception as exc:
        log.error("MarketData chain fetch raised ticker=%s error=%s", symbol, exc)
        _record_api_error(status_code=None, message=str(exc), attempts=API_RETRY_ATTEMPTS)
        return mock_option_chain(symbol, error_code="Exception", message=str(exc))
    if not payload:
        err = get_last_api_error()
        code = err.get("status_code") if err.get("status_code") is not None else "RequestFailed"
        return mock_option_chain(
            symbol,
            error_code=code,
            message=str(err.get("message") or "request failed"),
        )
    if str(payload.get("s") or "").lower() == "no_data":
        return _empty_chain()
    try:
        return _payload_to_frame(payload, symbol)
    except Exception as exc:
        log.error("Failed to parse option chain ticker=%s error=%s", symbol, exc)
        _record_api_error(status_code=None, message=str(exc), attempts=1)
        return mock_option_chain(symbol, error_code="ParseError", message=str(exc))


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
    try:
        payload = _marketdata_get(EXPIRATIONS_PATH.format(symbol=symbol))
    except Exception as exc:
        log.error("get_available_expirations failed ticker=%s error=%s", symbol, exc)
        return []
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
    if is_mock_frame(frame) or frame.empty or "expiration" not in frame.columns:
        return []
    return sorted(dict.fromkeys(str(value) for value in frame["expiration"].dropna()))


def get_current_price(ticker: str) -> float | None:
    """Return underlying last price from quotes, else from the option chain."""
    symbol = _normalize_ticker(ticker)
    if not symbol:
        return None
    try:
        payload = _marketdata_get(QUOTES_PATH.format(symbol=symbol))
    except Exception as exc:
        log.error("get_current_price failed ticker=%s error=%s", symbol, exc)
        payload = None
    if payload:
        last = _series(payload, "last", "mid")
        if last:
            price = pd.to_numeric(last[0], errors="coerce")
            if pd.notna(price) and float(price) > 0:
                return float(price)
    frame = fetch_option_chain(symbol)
    if is_mock_frame(frame) or frame.empty or "underlyingPrice" not in frame.columns:
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
        if frame.empty or is_mock_frame(frame):
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
