"""Unit tests for Black–Scholes and Heston Greeks."""

from __future__ import annotations

import json
import math
import time
from typing import Any

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
