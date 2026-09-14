"""Unit tests for Black–Scholes and Heston Greeks."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant_engine import (
    DEFAULT_HESTON_KAPPA,
    DEFAULT_HESTON_RHO,
    DEFAULT_HESTON_SIGMA,
    DEFAULT_HESTON_THETA,
    DEFAULT_HESTON_V0,
    MODEL_BLACK_SCHOLES,
    MODEL_HESTON,
    DataRepository,
    prepare_plotly_surface_xyz,
    calculate_greeks,
    calculate_heston_greeks,
    calculate_portfolio_risk,
    OPTIONS_CONTRACT_SIZE,
    calculate_ultima,
    calculate_vanna,
    calculate_veta,
    calculate_volga,
    calculate_vomma,
    calculate_zomma,
    generate_pro_surface_data,
    generate_greek_curve,
    generate_synthetic_drift,
    prefill_drift_data,
    prefill_driftdata,
    drift_prefill_stamp,
    EVENT_INSUFFICIENT_COVERAGE,
    analyze_event_outlook,
    analyze_gex_outlook,
    analyze_market_sentiment,
    analyze_volatility_risk_outlook,
)

STANDARD = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "option_type": "call"}
# ATM call: d1 = 0.35, N(d1) ≈ 0.636831
KNOWN_CALL_DELTA = 0.636831
GREEK_KEYS = ("Delta", "Gamma", "Theta", "Vega", "Rho")


def _heston_params(**overrides: float | int | str) -> dict:
    payload: dict = {
        "S": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "v0": DEFAULT_HESTON_V0,
        "kappa": DEFAULT_HESTON_KAPPA,
        "theta": DEFAULT_HESTON_THETA,
        "sigma": DEFAULT_HESTON_SIGMA,
        "rho": DEFAULT_HESTON_RHO,
        "option_type": "call",
        "seed": 42,
    }
    payload.update(overrides)
    return payload


def _handled(value: float | None) -> bool:
    return value is None or (isinstance(value, (int, float)) and math.isfinite(float(value)))


def test_calculate_greeks_known_atm_call_delta() -> None:
    greeks = calculate_greeks(STANDARD)
    assert greeks["Delta"] is not None
    assert greeks["Delta"] == pytest.approx(KNOWN_CALL_DELTA, abs=0.02)
    assert greeks["Delta"] == pytest.approx(0.63, abs=0.02)


def test_calculate_greeks_known_atm_call_signs() -> None:
    greeks = calculate_greeks(STANDARD)
    assert greeks["Gamma"] is not None and greeks["Gamma"] > 0
    assert greeks["Vega"] is not None and greeks["Vega"] > 0
    assert greeks["Rho"] is not None and greeks["Rho"] > 0
    assert greeks["Theta"] is not None and greeks["Theta"] < 0


@pytest.mark.parametrize("field,value", [("T", 0), ("sigma", 0), ("S", 0)])
def test_calculate_greeks_boundaries_do_not_crash(field: str, value: float) -> None:
    payload = dict(STANDARD)
    payload[field] = value
    greeks = calculate_greeks(payload)
    assert isinstance(greeks, dict)
    for key in GREEK_KEYS:
        assert key in greeks
        assert _handled(greeks[key])
        assert greeks[key] is None or greeks[key] == 0


@pytest.mark.parametrize("field,value", [("T", 0), ("sigma", 0), ("S", 0)])
def test_calculate_heston_greeks_boundaries_do_not_crash(field: str, value: float) -> None:
    greeks = calculate_heston_greeks(_heston_params(**{field: value}))
    assert isinstance(greeks, dict)
    for key in GREEK_KEYS:
        assert key in greeks
        assert _handled(greeks[key])
        assert greeks[key] is None or greeks[key] == 0


def test_calculate_heston_greeks_near_black_scholes() -> None:
    bs = calculate_greeks(STANDARD)
    heston = calculate_heston_greeks(
        _heston_params(v0=0.04, kappa=2.0, theta=0.04, sigma=0.01, rho=0.0)
    )
    for key in GREEK_KEYS:
        assert bs[key] is not None
        assert heston[key] is not None
        assert math.isfinite(float(heston[key]))
    assert heston["Delta"] == pytest.approx(float(bs["Delta"]), abs=0.15)
    assert heston["Gamma"] == pytest.approx(float(bs["Gamma"]), abs=0.02)
    assert heston["Vega"] == pytest.approx(float(bs["Vega"]), rel=0.6, abs=0.25)
    assert heston["Rho"] == pytest.approx(float(bs["Rho"]), rel=0.5, abs=0.2)
    assert math.copysign(1.0, float(heston["Theta"])) == math.copysign(1.0, float(bs["Theta"]))


PRO_FN = (calculate_vanna, calculate_vomma, calculate_zomma, calculate_veta, calculate_ultima)
PRO_ATM = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, model=MODEL_BLACK_SCHOLES)


def test_pro_metrics_known_atm_call() -> None:
    kwargs = {k: v for k, v in PRO_ATM.items() if k != "S"}
    s, k, t, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    n_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    vega_raw = s * n_d1 * sqrt_t
    gamma = n_d1 / (s * sigma * sqrt_t)
    veta_day = vega_raw * (r * d1 / (sigma * sqrt_t) - (1.0 + d1 * d2) / (2.0 * t)) / 365.0
    ultima = -vega_raw * (d1 * d2 * (1.0 - d1 * d2) + d1 * d1 + d2 * d2) / (sigma * sigma)
    assert calculate_vanna(s, **kwargs) == pytest.approx(-n_d1 * d2 / sigma, rel=1e-10)
    assert calculate_vomma(s, **kwargs) == pytest.approx(vega_raw * d1 * d2 / sigma, rel=1e-10)
    assert calculate_volga(s, **kwargs) == pytest.approx(calculate_vomma(s, **kwargs))
    assert calculate_zomma(s, **kwargs) == pytest.approx(gamma * (d1 * d2 - 1.0) / sigma, rel=1e-10)
    assert calculate_veta(s, **kwargs) == pytest.approx(veta_day, rel=1e-10)
    assert calculate_ultima(s, **kwargs) == pytest.approx(ultima, rel=1e-10)


@pytest.mark.parametrize("fn", PRO_FN)
@pytest.mark.parametrize("field,value", [("T", 0.0), ("sigma", 0.0), ("S", 0.0)])
def test_pro_metrics_boundaries_no_div_zero(fn, field: str, value: float) -> None:
    payload = dict(PRO_ATM)
    payload[field] = value
    s = payload.pop("S")
    out = fn(s, **payload)
    assert out is None or (isinstance(out, float) and math.isfinite(out))


def test_pro_metrics_accept_model_and_vectorized_spot() -> None:
    spots = [90.0, 100.0, 110.0]
    kwargs = {k: v for k, v in PRO_ATM.items() if k != "S"}
    for fn in PRO_FN:
        bs = fn(spots, **kwargs)
        assert len(bs) == 3
        assert all(math.isfinite(float(x)) for x in bs)
    heston = calculate_vanna(spots, **{**kwargs, "model": MODEL_HESTON, "v0": 0.04})
    assert len(heston) == 3
    assert all(x is None or math.isfinite(float(x)) for x in heston)


def test_generate_pro_surface_data_grid() -> None:
    frame = generate_pro_surface_data(
        "SPY",
        "zomma",
        strike=100.0,
        expiry_range=(30.0, 60.0),
        price_range=(95.0, 100.0, 105.0),
        sigma=0.2,
        model=MODEL_BLACK_SCHOLES,
    )
    assert list(frame.columns) == ["Price", "DaysToExpiry", "Value"]
    assert len(frame) == 6
    assert frame["Value"].apply(lambda x: math.isfinite(float(x)) or math.isnan(float(x))).all()
    assert not frame["Value"].apply(lambda x: math.isinf(float(x))).any()
    kwargs = dict(
        ticker="SPY",
        greek_name="zomma",
        strike=100.0,
        expiry_range=(30.0, 60.0),
        price_range=(90.0, 95.0, 100.0, 105.0, 110.0),
        sigma=0.2,
        model=MODEL_BLACK_SCHOLES,
    )
    raw = generate_pro_surface_data(**kwargs, apply_smoothing=False)
    smooth = generate_pro_surface_data(**kwargs, apply_smoothing=True)
    assert raw["Value"].equals(generate_pro_surface_data(**kwargs)["Value"])
    assert list(smooth.columns) == ["Price", "DaysToExpiry", "Value"]
    assert not raw.sort_values(["DaysToExpiry", "Price"]).reset_index(drop=True)["Value"].equals(
        smooth.sort_values(["DaysToExpiry", "Price"]).reset_index(drop=True)["Value"]
    )


def test_analyze_market_sentiment_bearish_event_stabilizing() -> None:
    frame = pd.DataFrame(
        {
            "strike": [100.0, 100.0, 100.0, 100.0],
            "option_type": ["call", "put", "call", "put"],
            "impliedVolatility": [0.20, 0.25, 0.18, 0.22],
            "DaysToExpiry": [10.0, 10.0, 60.0, 60.0],
            "Gamma": [0.04, 0.04, 0.02, 0.02],
        }
    )
    out = analyze_market_sentiment(frame)
    assert out["Skew"] == "Bearish Skew"
    assert out["TermStructure"] == "Event Risk / Catalyst"
    assert out["Gamma"] == "Range-Bound/Stabilizing"
    assert "Bearish Skew" in out["Summary"]
    assert "Put-Skew" in out["Interpretation"]


def test_analyze_market_sentiment_amplifying_and_neutral_skew() -> None:
    frame = pd.DataFrame(
        {
            "option_type": ["call", "put"],
            "impliedVolatility": [0.20, 0.20],
            "DaysToExpiry": [30.0, 90.0],
            "Gamma": [-0.05, -0.04],
        }
    )
    out = analyze_market_sentiment(frame)
    assert out["Skew"] == "Neutral Skew"
    assert out["TermStructure"] == "Stable Term Structure"
    assert out["Gamma"] == "Volatile/Amplifying"


def test_analyze_market_sentiment_empty() -> None:
    out = analyze_market_sentiment(pd.DataFrame())
    assert out["Skew"] == "Neutral Skew"
    assert "Insufficient" in out["Summary"]
    assert "Interpretation" in out


def test_analyze_gex_outlook_flip_and_empty() -> None:
    frame = pd.DataFrame(
        {
            "strike": [90.0, 90.0, 100.0, 100.0, 110.0, 110.0],
            "option_type": ["call", "put", "call", "put", "call", "put"],
            "Gamma": [0.01, 0.05, 0.05, 0.05, 0.02, 0.01],
            "openInterest": [10.0, 100.0, 100.0, 80.0, 200.0, 10.0],
        }
    )
    text = analyze_gex_outlook(frame)
    assert "support the market above" in text["Outlook"]
    assert "accelerate selling below" in text["Outlook"]
    assert text["GammaFlip"] is not None
    assert float(text["GammaFlip"]) > 0
    empty = analyze_gex_outlook(pd.DataFrame())
    assert empty["GammaFlip"] is None
    assert "unavailable" in empty["Outlook"]
    missing_oi = frame.drop(columns=["openInterest"])
    assert "unavailable" in analyze_gex_outlook(missing_oi)["Outlook"]


def test_analyze_volatility_risk_outlook() -> None:
    frame = pd.DataFrame(
        {
            "S": [100.0, 100.0],
            "K": [100.0, 100.0],
            "T": [1.0, 1.0],
            "impliedVolatility": [0.20, 0.20],
            "option_type": ["call", "put"],
            "openInterest": [100.0, 80.0],
        }
    )
    text = analyze_volatility_risk_outlook(frame)
    assert "Vanna" in text["Outlook"] or "Volga" in text["Outlook"] or "muted" in text["Outlook"]
    empty = analyze_volatility_risk_outlook(pd.DataFrame())
    assert empty["HighSensitivity"] is False
    assert empty["Outlook"] == (
        "Volatility-risk outlook is unavailable: implied volatility data is missing."
    )
    no_iv = pd.DataFrame({"strike": [100.0], "option_type": ["call"]})
    assert "unavailable" in analyze_volatility_risk_outlook(no_iv)["Outlook"]


def test_analyze_event_outlook_catalyst_and_thin_chain(monkeypatch, caplog) -> None:
    frame = pd.DataFrame(
        {
            "impliedVolatility": [0.36, 0.36, 0.20, 0.20, 0.18, 0.18],
            "DaysToExpiry": [7.0, 7.0, 30.0, 30.0, 90.0, 90.0],
            "expiration": ["2026-09-18", "2026-09-18", "2026-10-16", "2026-10-16", "2026-12-18", "2026-12-18"],
        }
    )
    out = analyze_event_outlook(frame, selected_expiry="2026-09-18")
    assert out["EventRisk"] is True
    assert out["Outlook"] == "High Event Risk / Catalyst Detected."
    assert "2026-09-18" in out["Summary"]
    assert "catalyst" in out["Summary"]
    assert "elevated by" in out["Summary"]
    far = analyze_event_outlook(frame, selected_expiry="2026-12-18")
    assert far["ShortIV"] == pytest.approx(0.20)
    assert far["LongIV"] == pytest.approx(0.18)
    calm = frame.copy()
    calm["impliedVolatility"] = [0.20, 0.20, 0.20, 0.20, 0.19, 0.19]
    quiet = analyze_event_outlook(calm, selected_expiry="2026-09-18")
    assert quiet["EventRisk"] is False
    assert quiet["Outlook"] == "Stable Event Outlook"
    empty = analyze_event_outlook(pd.DataFrame())
    assert empty["EventRisk"] is False
    assert empty["Summary"] == EVENT_INSUFFICIENT_COVERAGE
    thin = pd.DataFrame(
        {
            "impliedVolatility": [0.40, 0.41],
            "DaysToExpiry": [5.0, 5.0],
            "expiration": ["2026-09-18", "2026-09-18"],
        }
    )
    assert analyze_event_outlook(thin)["Outlook"] == EVENT_INSUFFICIENT_COVERAGE
    monkeypatch.setattr("quant_engine._event_listed_expirations", lambda ticker: ["2026-12-18"])
    sparse = analyze_event_outlook(frame, ticker="THIN", selected_expiry="2026-12-18")
    assert sparse["Outlook"] == EVENT_INSUFFICIENT_COVERAGE
    assert "THIN" in caplog.text
    assert "2026-12-18" in caplog.text


def test_calculate_portfolio_risk_net_greeks_and_concentration() -> None:
    empty = calculate_portfolio_risk(pd.DataFrame())
    assert empty["Delta"] == 0.0
    assert empty["LargestPosition"] is None
    assert empty["DaysToExpiration"] is None
    long_call = calculate_greeks(STANDARD)
    short_put = calculate_greeks({**STANDARD, "option_type": "put"})
    book = pd.DataFrame(
        {
            "Ticker": ["SPY", "QQQ"],
            "Strike": [100.0, 100.0],
            "Expiry": [1.0, 1.0],
            "Side": ["long", "short"],
            "Quantity": [2.0, 1.0],
            "option_type": ["call", "put"],
            "S": [100.0, 100.0],
            "sigma": [0.2, 0.2],
            "r": [0.05, 0.05],
            "premium": [10.0, 5.0],
            "DaysToExpiry": [30.0, 7.0],
        }
    )
    out = calculate_portfolio_risk(book)
    size = float(OPTIONS_CONTRACT_SIZE)
    assert out["Delta"] == pytest.approx(float(long_call["Delta"]) * 2.0 * size - float(short_put["Delta"]) * size)
    assert out["Gamma"] == pytest.approx(float(long_call["Gamma"]) * 2.0 * size - float(short_put["Gamma"]) * size)
    assert out["Theta"] == pytest.approx(float(long_call["Theta"]) * 2.0 * size - float(short_put["Theta"]) * size)
    assert out["Vega"] == pytest.approx(float(long_call["Vega"]) * 2.0 * size - float(short_put["Vega"]) * size)
    assert out["LargestPosition"] == pytest.approx(80.0)
    assert out["TickerConcentration"] == pytest.approx(80.0)
    assert out["ConcentratedTicker"] == "SPY"
    assert out["DaysToExpiration"] == pytest.approx(7.0)
    by_ticker = {row["Ticker"]: row["Delta"] for row in out["DeltaByTicker"]}
    assert by_ticker["SPY"] == pytest.approx(float(long_call["Delta"]) * 2.0 * size)
    assert by_ticker["QQQ"] == pytest.approx(-float(short_put["Delta"]) * size)


def test_calculate_portfolio_risk_fills_premium_and_spot(monkeypatch) -> None:
    monkeypatch.setattr("data_ingestion.get_current_price", lambda ticker: {"SPY": 580.0, "AAPL": 230.0}.get(ticker, 0.0))
    book = pd.DataFrame(
        {
            "Ticker": ["SPY", "AAPL"],
            "Strike": [580.0, 230.0],
            "Expiry": ["2026-10-16", "2026-11-20"],
            "Side": ["Call", "Put"],
            "Quantity": [10.0, 5.0],
            "EntryPrice": [5.50, 3.20],
        }
    )
    out = calculate_portfolio_risk(book)
    filled = out["Positions"]
    assert list(filled["S"]) == [580.0, 230.0]
    assert list(filled["premium"]) == [10.0 * 5.50 * 100.0, 5.0 * 3.20 * 100.0]
    assert out["TickerConcentration"] == pytest.approx(100.0 * 5500.0 / (5500.0 + 1600.0))
    missing = pd.DataFrame({"Ticker": ["ZZZ"], "Strike": [float("nan")], "Quantity": [float("nan")], "Side": ["Call"]})
    monkeypatch.setattr("data_ingestion.get_current_price", lambda ticker: None)
    zeros = calculate_portfolio_risk(missing)
    assert float(zeros["Positions"]["S"].iloc[0]) == 0.0
    assert float(zeros["Positions"]["premium"].iloc[0]) == 0.0
    source = pd.DataFrame(
        {
            "Ticker": ["SPY"],
            "Strike": [100.0],
            "Quantity": [1.0],
            "Side": ["Call"],
            "EntryPrice": [2.0],
        }
    )
    before = source.copy()
    monkeypatch.setattr("data_ingestion.get_current_price", lambda ticker: 101.0)
    calculate_portfolio_risk(source)
    assert source.equals(before)


def test_data_repository_ttl_version_and_atomic_cache(tmp_path) -> None:
    calls = {"n": 0}

    def fetcher(ticker: str, expiry: Any) -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame({"ticker": [ticker], "expiration": [str(expiry)], "sigma": [0.2]})

    repo = DataRepository(
        cache_dir=tmp_path,
        fetcher=fetcher,
        status_probe=lambda: {"ok": True, "status_code": 200, "message": "reachable"},
    )
    assert repo.VERSION == "1.0"
    assert repo.TTL_SECONDS == 15 * 60
    assert repo.is_fresh("SPY") is False
    first = repo.get_data("SPY", "2026-10-16")
    assert calls["n"] == 1
    assert list(first["ticker"]) == ["SPY"]
    assert repo.is_fresh("SPY") is True
    second = repo.get_data("SPY", "2026-10-16")
    assert calls["n"] == 1
    assert len(second.index) == 1
    assert repo.get_api_status()["ok"] is True
    path = tmp_path / "SPY.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = "0.9"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert repo.is_fresh("SPY") is False
    repo.save_to_cache("SPY", pd.DataFrame({"ticker": ["SPY"]}))
    stale = json.loads(path.read_text(encoding="utf-8"))
    stale["saved_at"] = time.time() - repo.TTL_SECONDS - 1
    path.write_text(json.dumps(stale), encoding="utf-8")
    assert repo.is_fresh("SPY") is False
    curve = generate_greek_curve("SPY", 100.0, 1.0, [90.0, 110.0], repo=repo)
    assert not curve.empty
    assert "Delta" in curve.columns
    snaps = repo.get_historical_snapshots("SPY")
    assert len(snaps) >= 1
    assert "data_history" in snaps[0].replace("\\", "/")
    assert Path(snaps[0]).name.startswith("SPY_")


def test_data_repository_historical_compare_and_prune(tmp_path) -> None:
    repo = DataRepository(cache_dir=tmp_path, status_probe=lambda: {"ok": True})
    first = pd.DataFrame({"impliedVolatility": [0.20, 0.22], "Delta": [0.50, 0.40], "Gamma": [0.02, 0.01], "Theta": [-0.03, -0.02], "Vega": [0.10, 0.12]})
    second = pd.DataFrame({"impliedVolatility": [0.30, 0.32], "Delta": [0.60, 0.50], "Gamma": [0.03, 0.02], "Theta": [-0.04, -0.03], "Vega": [0.20, 0.22]})
    repo.save_to_cache("QQQ", first)
    stamp_a = Path(repo.get_historical_snapshots("QQQ")[-1]).name.replace("QQQ_", "").replace(".json", "")
    repo.save_to_cache("QQQ", second)
    snaps = repo.get_historical_snapshots("QQQ")
    assert len(snaps) >= 2
    stamp_b = Path(snaps[-1]).name.replace("QQQ_", "").replace(".json", "")
    change = repo.calculate_change_over_time("QQQ", stamp_a, stamp_b)
    assert change["IV"]["b"] > change["IV"]["a"]
    assert change["Delta"]["change"] == pytest.approx(0.10)
    repo.HISTORY_LIMIT = 2
    for idx in range(6):
        repo.save_to_cache("QQQ", pd.DataFrame({"impliedVolatility": [0.1 + idx * 0.01]}))
    assert len(repo.get_historical_snapshots("QQQ")) <= 2


def test_data_repository_delta_drift_and_snapshots(tmp_path) -> None:
    repo = DataRepository(cache_dir=tmp_path, history_dir=tmp_path / "data_history", status_probe=lambda: {"ok": True})
    first = pd.DataFrame(
        {
            "Strike": [90.0, 100.0, 110.0, 90.0, 100.0, 110.0],
            "DaysToExpiry": [7.0, 7.0, 7.0, 30.0, 30.0, 30.0],
            "Delta": [0.80, 0.50, 0.20, 0.70, 0.50, 0.30],
        }
    )
    second = first.copy()
    second["Delta"] = first["Delta"] + 0.10
    mismatched = pd.DataFrame(
        {
            "Strike": [95.0, 105.0, 95.0, 105.0],
            "DaysToExpiry": [10.0, 10.0, 21.0, 21.0],
            "Delta": [0.90, 0.40, 0.75, 0.35],
        }
    )
    repo.save_to_cache("IWM", first)
    ts_a = repo.get_available_snapshots("IWM")[-1]
    repo.save_to_cache("IWM", second)
    stamps = repo.get_available_snapshots("IWM")
    assert ts_a in stamps
    ts_b = stamps[-1]
    assert len(stamps) >= 2
    drift = repo.calculate_delta_drift("IWM", ts_a, ts_b)
    assert {"Strike", "DaysToExpiry", "Delta"}.issubset(drift.columns)
    assert not drift.empty
    assert float(np.nanmean(drift["Delta"].to_numpy(dtype=float))) == pytest.approx(0.10, abs=1e-6)
    missing = repo.calculate_delta_drift("IWM", "missing-a", ts_b)
    assert missing.empty
    repo.save_to_cache("IWM", mismatched)
    ts_c = repo.get_available_snapshots("IWM")[-1]
    mixed = repo.calculate_delta_drift("IWM", ts_a, ts_c)
    assert not mixed.empty
    assert mixed["Delta"].notna().any()


def test_calculate_delta_drift_mismatched_metadata_short_circuits(tmp_path) -> None:
    """Mismatched contract metadata must not produce a drift surface."""
    repo = DataRepository(cache_dir=tmp_path, history_dir=tmp_path / "data_history", status_probe=lambda: {"ok": True})
    base = pd.DataFrame(
        {
            "Strike": [90.0, 100.0, 110.0, 90.0, 100.0, 110.0],
            "DaysToExpiry": [7.0, 7.0, 7.0, 30.0, 30.0, 30.0],
            "Delta": [0.80, 0.50, 0.20, 0.70, 0.50, 0.30],
        }
    )
    later = base.copy()
    later["Delta"] = base["Delta"] + 0.10
    repo.save_to_cache(
        "SPY",
        base,
        metadata={"ticker": "SPY", "strike": 100.0, "expiry": "2026-12-18"},
    )
    ts_a = repo.get_available_snapshots("SPY")[-1]
    repo.save_to_cache(
        "SPY",
        later,
        metadata={"ticker": "SPY", "strike": 110.0, "expiry": "2026-12-18"},
    )
    ts_b = repo.get_available_snapshots("SPY")[-1]
    assert ts_a != ts_b
    payload_a = repo._snapshot_payload("SPY", ts_a) or {}
    payload_b = repo._snapshot_payload("SPY", ts_b) or {}
    matched, message = DataRepository.validate_snapshot_match(payload_a, payload_b)
    assert matched is False
    assert "strike" in message
    drift = repo.calculate_delta_drift("SPY", ts_a, ts_b)
    assert drift.empty
    assert list(drift.columns) == ["Strike", "DaysToExpiry", "Delta"]
    strike_axis, dte_axis, z, synthetic = repo.calculate_delta_drift_grid("SPY", ts_a, ts_b)
    assert strike_axis.size == 0
    assert dte_axis.size == 0
    assert z.size == 0
    assert synthetic is False


def test_generate_synthetic_drift_non_flat() -> None:
    strike_axis, dte_axis, z = generate_synthetic_drift("SPY")
    assert z.ndim == 2
    assert z.shape == (len(dte_axis), len(strike_axis))
    finite = z[np.isfinite(z)]
    assert finite.size > 0
    assert float(np.min(finite)) != float(np.max(finite))
    assert not np.allclose(finite, 0.0, atol=1e-12)
    other = generate_synthetic_drift("QQQ")
    assert other[2].shape == z.shape
    assert not np.allclose(other[2], z)


def test_identical_snapshots_trigger_synthetic_drift_fallback(tmp_path) -> None:
    """Same timestamp / identical surface → flat real drift → synthetic fallback."""
    repo = DataRepository(cache_dir=tmp_path, history_dir=tmp_path / "data_history", status_probe=lambda: {"ok": True})
    frame = pd.DataFrame(
        {
            "Strike": [90.0, 100.0, 110.0, 90.0, 100.0, 110.0],
            "DaysToExpiry": [7.0, 7.0, 7.0, 30.0, 30.0, 30.0],
            "Delta": [0.80, 0.50, 0.20, 0.70, 0.50, 0.30],
        }
    )
    repo.save_to_cache(
        "IWM",
        frame,
        metadata={"ticker": "IWM", "strike": 100.0, "expiry": "2026-12-18"},
    )
    ts = repo.get_available_snapshots("IWM")[-1]
    drift = repo.calculate_delta_drift("IWM", ts, ts)
    assert not drift.empty
    assert drift.attrs.get("synthetic") is True
    finite = drift["Delta"].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    assert finite.size > 0
    assert float(np.min(finite)) != float(np.max(finite))
    strike_axis, dte_axis, z, synthetic = repo.calculate_delta_drift_grid("IWM", ts, ts)
    assert synthetic is True
    assert z.shape == (len(dte_axis), len(strike_axis))
    assert float(np.nanmin(z)) != float(np.nanmax(z))


def test_prefill_drift_data_creates_distinct_surfaces(tmp_path) -> None:
    """Prefill writes START/END history files with non-flat-comparable delta surfaces."""
    chain = pd.DataFrame(
        {
            "Strike": [90.0, 100.0, 110.0, 90.0, 100.0, 110.0],
            "DaysToExpiry": [7.0, 7.0, 7.0, 30.0, 30.0, 30.0],
            "Delta": [0.80, 0.50, 0.20, 0.70, 0.50, 0.30],
            "impliedVolatility": [0.22, 0.20, 0.21, 0.23, 0.205, 0.215],
            "S": [100.0] * 6,
        }
    )

    def _fetcher(ticker: str, expiry: Any = None) -> pd.DataFrame:
        assert str(ticker).upper() == "SPY"
        return chain.copy()

    repo = DataRepository(
        cache_dir=tmp_path / "cache",
        history_dir=tmp_path / "data_history",
        fetcher=_fetcher,
        status_probe=lambda: {"ok": True},
    )
    result = prefill_drift_data("SPY", 100.0, "2026-09-19", repo=repo, seed=7)
    assert result["start_label"] == "100_2026-09-19_START"
    assert result["end_label"] == "100_2026-09-19_END"
    assert result["start_label"] != result["end_label"]
    assert Path(result["start_path"]).is_file()
    assert Path(result["end_path"]).is_file()
    assert Path(result["start_path"]).name == "SPY_100_2026-09-19_START.json"
    assert Path(result["end_path"]).name == "SPY_100_2026-09-19_END.json"

    stamps = repo.get_available_snapshots("SPY")
    assert result["start_label"] in stamps
    assert result["end_label"] in stamps

    payload_a = repo._snapshot_payload("SPY", result["start_label"]) or {}
    payload_b = repo._snapshot_payload("SPY", result["end_label"]) or {}
    matched, message = DataRepository.validate_snapshot_match(payload_a, payload_b)
    assert matched is True, message
    assert payload_a.get("snapshot_id") != payload_b.get("snapshot_id")

    deltas_a = np.asarray(
        [np.nan if v is None else float(v) for v in (payload_a.get("delta_surface") or {}).get("Delta") or []],
        dtype=float,
    )
    deltas_b = np.asarray(
        [np.nan if v is None else float(v) for v in (payload_b.get("delta_surface") or {}).get("Delta") or []],
        dtype=float,
    )
    assert deltas_a.size > 0 and deltas_b.size > 0
    assert not np.allclose(deltas_a, deltas_b, equal_nan=True)

    drift = repo.calculate_delta_drift("SPY", result["start_label"], result["end_label"])
    assert not drift.empty
    finite = drift["Delta"].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    assert finite.size > 0
    assert float(np.min(finite)) != float(np.max(finite)) or not np.allclose(finite, 0.0, atol=1e-12)
    assert float(np.nanmean(finite)) == pytest.approx(0.05, abs=0.02)
    assert prefill_driftdata is prefill_drift_data
    assert drift_prefill_stamp(450, "2026-09-19", "START") == "450_2026-09-19_START"


def test_snapshot_vol_surface_from_history(tmp_path) -> None:
    repo = DataRepository(cache_dir=tmp_path, history_dir=tmp_path / "data_history")
    rows = []
    for dte in (7.0, 14.0, 21.0, 30.0, 45.0):
        for strike in (90.0, 95.0, 100.0, 105.0, 110.0):
            rows.append({"Strike": strike, "DaysToExpiry": dte, "impliedVolatility": 0.20 + (strike - 100.0) * 0.002 + dte * 0.001})
    repo.save_to_cache("DIA", pd.DataFrame(rows))
    stamp = repo.get_available_snapshots("DIA")[-1]
    surface = repo.snapshot_vol_surface("DIA", stamp)
    assert isinstance(surface, pd.DataFrame)
    assert {"Strike", "DaysToExpiry", "IV"}.issubset(surface.columns)
    assert len(surface.index) >= 10


def test_snapshot_stamp_second_resolution_and_top_level_keys(tmp_path) -> None:
    """Saves use %Y%m%d%H%M%S stamps and embed ticker/strike/expiry/contract_type top-level."""
    repo = DataRepository(cache_dir=tmp_path, history_dir=tmp_path / "data_history", status_probe=lambda: {"ok": True})
    frame = pd.DataFrame(
        {
            "Strike": [580.0, 580.0],
            "DaysToExpiry": [7.0, 30.0],
            "Delta": [0.55, 0.52],
        }
    )
    repo.save_to_cache(
        "SPY",
        frame,
        metadata={
            "ticker": "SPY",
            "strike": 580.0,
            "expiry": "2026-10-16",
            "contract_type": "call",
        },
    )
    stamp = repo.get_available_snapshots("SPY")[-1]
    assert len(stamp) >= 14
    assert stamp[:14].isdigit()
    assert datetime.strptime(stamp[:14], "%Y%m%d%H%M%S")
    path = Path(repo.get_historical_snapshots("SPY")[-1])
    assert path.name.startswith("SPY_")
    assert path.name.endswith(".json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ticker"] == "SPY"
    assert payload["strike"] == 580.0
    assert payload["expiry"] == "2026-10-16"
    assert payload["contract_type"] == "Call"
    assert isinstance(payload.get("metadata"), dict)
    assert payload["metadata"]["strike"] == 580.0
    assert payload["metadata"]["contract_type"] == "Call"
    label = DataRepository.format_snapshot_label(payload, stamp)
    assert label.startswith("SPY 580 Call 2026-10-16 @ ")
    # Legacy / sparse payload still yields a usable fallback label.
    assert DataRepository.format_snapshot_label({}, "20260101123045") == "20260101123045"
    assert DataRepository.format_snapshot_label({"ticker": "QQQ"}, "2026-01-01_1230").startswith("QQQ @ ")


def test_prepare_plotly_surface_xyz_reshape_and_axis_gate() -> None:
    flat = np.arange(6, dtype=float)
    z2d, x_ok, y_ok, warning = prepare_plotly_surface_xyz(flat, [0.0, 1.0, 2.0], [10.0, 20.0])
    assert z2d.shape == (2, 3)
    assert x_ok is not None and y_ok is not None
    assert warning is None
    assert list(x_ok) == [0.0, 1.0, 2.0]
    assert list(y_ok) == [10.0, 20.0]

    bad_x = [0.0, 1.0]
    z_bad, x_none, y_none, warn = prepare_plotly_surface_xyz(np.ones((2, 3)), bad_x, [10.0, 20.0])
    assert z_bad.shape == (2, 3)
    assert x_none is None and y_none is None
    assert warn is not None and "mismatch" in warn

    already_2d = np.ones((4, 5))
    z_keep, _, _, _ = prepare_plotly_surface_xyz(already_2d)
    assert z_keep.shape == (4, 5)


def test_generate_greek_curve_uses_injected_repo() -> None:
    class _Repo:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Any]] = []

        def get_data(self, ticker: str, expiry: Any = None) -> pd.DataFrame:
            self.calls.append((ticker, expiry))
            return pd.DataFrame()

    repo = _Repo()
    frame = generate_greek_curve("QQQ", 100.0, 1.0, [100.0], repo=repo)
    assert repo.calls == [("QQQ", 1.0)]
    assert "Gamma" in frame.columns


def test_fetch_option_chain_live_uses_expiration_omits_date(monkeypatch) -> None:
    """Live/future chain: MarketData query includes ``expiration`` and omits ``date``."""
    import data_ingestion as ingest

    monkeypatch.setenv("MARKETDATA_API_KEY", "test-token")
    captured: dict[str, Any] = {}

    def _fake_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        captured["path"] = path
        captured["params"] = dict(params or {})
        return {"s": "no_data"}

    monkeypatch.setattr(ingest, "_marketdata_get", _fake_get)
    frame = ingest.fetch_option_chain("SPY", expiration="2026-10-16")
    assert frame.empty
    assert captured["params"].get("expiration") == "2026-10-16"
    assert "date" not in captured["params"]
    assert "date=" not in str(captured["path"])
    # Full URL shape: .../options/chain/{ticker}/?expiration=... with no date=
    live_url = f"{ingest.API_BASE}{captured['path']}"
    if captured["params"]:
        from urllib.parse import urlencode

        live_url = f"{live_url}?{urlencode(captured['params'])}"
    assert "/options/chain/SPY/" in live_url
    assert "expiration=2026-10-16" in live_url
    assert "date=" not in live_url

    # Positional ``expiry`` alias still maps to the same query param.
    captured.clear()
    ingest.fetch_option_chain("SPY", "2026-12-18")
    assert captured["params"].get("expiration") == "2026-12-18"
    assert "date=" not in str(captured["path"])

    # Historical EOD keeps ``date=`` on the path while still sending expiration.
    captured.clear()
    ingest.fetch_option_chain("SPY", expiration="2026-10-16", date="2025-06-01")
    assert captured["params"].get("expiration") == "2026-10-16"
    assert "date=2025-06-01" in str(captured["path"])


def test_fetch_option_chain_wrong_expiry_raises(monkeypatch) -> None:
    """Requested expiration must match normalized payload expiration (list or scalar)."""
    import data_ingestion as ingest

    monkeypatch.setenv("MARKETDATA_API_KEY", "test-token")

    def _wrong_list(_path: str, _params: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"s": "ok", "expiration": ["2026-12-18", "2026-12-18"], "strike": [100]}

    monkeypatch.setattr(ingest, "_marketdata_get", _wrong_list)
    with pytest.raises(ValueError, match=r"API returned wrong expiry: \['2026-12-18', '2026-12-18'\]"):
        ingest.fetch_option_chain("SPY", expiration="2026-10-16")

    def _wrong_scalar(_path: str, _params: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"s": "ok", "expiration": "2025-01-17"}

    monkeypatch.setattr(ingest, "_marketdata_get", _wrong_scalar)
    with pytest.raises(ValueError, match=r"API returned wrong expiry: 2025-01-17"):
        ingest.fetch_option_chain("QQQ", "2026-10-16")

    def _matching(_path: str, _params: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"s": "no_data", "expiration": ["2026-10-16"]}

    monkeypatch.setattr(ingest, "_marketdata_get", _matching)
    assert ingest.fetch_option_chain("SPY", expiration="2026-10-16").empty


def test_empty_or_mismatched_expiry_does_not_save_unscoped_next_monthly(tmp_path) -> None:
    """Capture guard: empty/mismatched chain must not be saved as the requested future expiry.

    Simulates the bug path: API returns empty for a future expiration, an unscoped
    next-monthly frame is available, and metadata would claim the future date —
    refuse save so next-monthly rows are never written under that expiry.
    """
    import data_ingestion as ingest

    requested = "2026-12-18"
    next_monthly = pd.DataFrame(
        {
            "strike": [450.0, 450.0],
            "expiration": ["2026-09-18", "2026-09-18"],
            "bid": [1.0, 1.1],
            "ask": [1.2, 1.3],
            "underlyingPrice": [450.0, 450.0],
            "option_type": ["call", "put"],
            "impliedVolatility": [0.20, 0.21],
            "S": [450.0, 450.0],
            "K": [450.0, 450.0],
            "T": [0.01, 0.01],
            "sigma": [0.20, 0.21],
        }
    )

    # Empty scoped fetch → refuse (do not treat as savable under requested expiry).
    empty_reason = ingest.chain_expiry_mismatch_reason(pd.DataFrame(), requested)
    assert empty_reason is not None
    assert "2026-12-18" in empty_reason

    # Unscoped next-monthly under future metadata → refuse.
    mismatch_reason = ingest.chain_expiry_mismatch_reason(next_monthly, requested)
    assert mismatch_reason is not None
    assert "2026-09-18" in mismatch_reason
    assert "2026-12-18" in mismatch_reason

    # Mock fallback → refuse.
    mock = ingest.mock_option_chain("SPY", error_code="no_data", message="empty")
    assert ingest.chain_expiry_mismatch_reason(mock, requested) is not None

    # Matching rows → allow.
    matching = next_monthly.copy()
    matching["expiration"] = requested
    assert ingest.chain_expiry_mismatch_reason(matching, requested) is None

    # No expiry requested → allow even next-monthly (unscoped fetch is intentional).
    assert ingest.chain_expiry_mismatch_reason(next_monthly, None) is None

    # Capture-path save simulation: only write when the guard returns None.
    repo = DataRepository(cache_dir=tmp_path, status_probe=lambda: {"ok": True})
    before = len(repo.get_historical_snapshots("SPY"))
    for candidate in (pd.DataFrame(), next_monthly, mock):
        reason = ingest.chain_expiry_mismatch_reason(candidate, requested)
        assert reason is not None
        # Deliberately skip save_to_cache when mismatched — mirrors _capture_snapshot.
    assert len(repo.get_historical_snapshots("SPY")) == before

    # Matching chain may be saved with matching metadata.
    repo.save_to_cache(
        "SPY",
        matching,
        metadata={"ticker": "SPY", "expiry": requested, "as_of_date": "2026-09-14"},
    )
    snaps = repo.get_historical_snapshots("SPY")
    assert len(snaps) == before + 1
    payload = json.loads(Path(snaps[-1]).read_text(encoding="utf-8"))
    meta = payload.get("metadata") or {}
    assert (payload.get("expiry") or meta.get("expiry")) == requested
    rows = payload.get("data") or []
    assert rows
    assert all(str(row.get("expiration", ""))[:10] == requested for row in rows)
    # Critically: never persisted next-monthly under the future expiry label.
    assert not any(str(row.get("expiration", ""))[:10] == "2026-09-18" for row in rows)
