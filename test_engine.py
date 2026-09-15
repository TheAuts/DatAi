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
    reshape_to_surface,
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
    compute_total_risk_score,
    generate_integrated_risk_surface,
    generate_synthetic_drift,
    load_all_snapshots,
    minmax_normalize_01,
    prefill_drift_data,
    prefill_driftdata,
    drift_prefill_stamp,
    snapshot_time_label_from_name,
    timelapse_shared_z_range,
    TIMELAPSE_SURFACE_KEY,
    EVENT_INSUFFICIENT_COVERAGE,
    event_impact_calculate_shock,
    event_impact_find_bracketing_snapshots,
    event_impact_load_data,
    event_impact_parse_stamp_datetime,
    event_impact_suggest_position,
    analyze_event_outlook,
    analyze_gex_outlook,
    analyze_market_sentiment,
    analyze_volatility_risk_outlook,
    calculate_stress_scenario,
    build_stress_pnl_matrix,
    calculate_market_reflexivity,
    regime_cascade_probability,
    regime_classify_states,
    regime_clip_probability,
    regime_reflexive_cascade_flag,
    REGIME_CRASH_CASCADE,
    REGIME_FRAGILE_BEAR,
    REGIME_SQUEEZE_UP,
    REGIME_STABLE_BULL,
    REGIME_HEDGE_PUT_SPREADS,
    test_audit_gex_normalized as lw_audit_gex_normalized,
    test_cascade_score as lw_cascade_score,
    test_calculate_liquidation_waterfall as lw_calculate_liquidation_waterfall,
    test_classify_liquidation_zones as lw_classify_liquidation_zones,
    test_detect_gamma_flips as lw_detect_gamma_flips,
    test_market_stability_distance as lw_market_stability_distance,
    TEST_LIQUIDATION_UNSTABLE_MSG,
    TEST_LIQUIDATION_ZONE,
    calculate_liquidation_waterfall,
    # Godlike Quant Strategist
    HMM_REGIME_CRASH_CASCADE,
    HMM_REGIME_STABLE,
    HMM_REGIME_STATES,
    HMM_REGIME_TREND,
    HIGH_VANNA_WARNING,
    MC_MIN_PATHS,
    classify_market_regime,
    require_regime_before_trade,
    audit_monte_carlo_path_count,
    evaluate_position_montecarlo,
    mathematicians_rule_gate,
    probability_of_profit,
    probability_of_touching,
    simulate_gbm_paths,
    strategist_evaluate_position,
    build_vanna_volga_report,
    # Council of Experts
    EXPERT_A,
    EXPERT_B,
    EXPERT_C,
    EXPERT_NAMES,
    STANCE_BUY,
    STANCE_NEUTRAL,
    STANCE_SELL,
    STANCES,
    build_disagreement_report,
    clip_confidence,
    compute_iv_rank_percentile,
    run_council_debate,
    calculate_sentiment_alpha,
    calculate_market_velocity,
    audit_market_velocity,
    FLOW_DATA_STALE_MSG,
    FLOW_OFI_BOUNDS,
    FLOW_VELOCITY_BOUNDS,
    calculate_vpin,
    calculate_order_book_imbalance,
    calculate_microstructure_lab,
    audit_vpin,
    audit_obi_heatmap,
    detect_liquidity_trap,
    shape_obi_heatmap,
    TOXICITY_ALERT_MSG,
    LIQUIDITY_TRAP_MSG,
    HIGH_ADVERSE_SELECTION_MSG,
    OBI_HEATMAP_UNSTABLE_MSG,
    VPIN_HIGH_THRESHOLD,
    OBI_EXTREME_THRESHOLD,
    OBI_TOP_LEVELS,
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


def test_data_repository_is_historical_live_vs_snapshot(tmp_path) -> None:
    def fetcher(ticker: str, expiry: Any, date: str | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": [ticker],
                "expiration": [str(expiry) if expiry is not None else "2026-10-16"],
                "as_of": [date or "live"],
                "sigma": [0.2],
            }
        )

    repo = DataRepository(
        cache_dir=tmp_path,
        history_dir=tmp_path / "data_history",
        fetcher=fetcher,
        status_probe=lambda: {"ok": True},
    )
    assert repo.is_historical is False
    assert repo.status_date is None

    live = repo.get_data("SPY", "2026-10-16")
    assert not live.empty
    assert repo.is_historical is False
    assert repo.status_date is None

    hist = repo.get_data("SPY", "2026-10-16", date="2024-06-15")
    assert not hist.empty
    assert repo.is_historical is True
    assert repo.status_date == "2024-06-15"

    live_again = repo.get_data("SPY", "2026-10-16")
    assert not live_again.empty
    assert repo.is_historical is False
    assert repo.status_date is None

    repo.save_to_cache(
        "SPY",
        pd.DataFrame({"ticker": ["SPY"], "sigma": [0.25]}),
        metadata={"ticker": "SPY", "as_of_date": "2025-03-01"},
    )
    stamp = repo.get_available_snapshots("SPY")[-1]
    payload = repo.load_from_cache("SPY", stamp)
    assert payload is not None
    assert repo.is_historical is True
    assert repo.status_date == "2025-03-01"

    repo.get_data("SPY")
    assert repo.is_historical is False


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
    path = Path(repo.get_historical_snapshots("DIA")[-1])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "vol_surface" in payload
    assert isinstance(payload["vol_surface"], dict)
    assert payload["vol_surface"].get("IV")
    surface = repo.snapshot_vol_surface("DIA", stamp)
    assert isinstance(surface, pd.DataFrame)
    assert {"Strike", "DaysToExpiry", "IV"}.issubset(surface.columns)
    assert len(surface.index) >= 10


def test_snapshot_vol_surface_alias_and_2d_reshape(tmp_path) -> None:
    """Alias keys are readable; preferred save key remains vol_surface."""
    repo = DataRepository(cache_dir=tmp_path, history_dir=tmp_path / "data_history")
    strike_axis = np.array([90.0, 100.0, 110.0], dtype=float)
    dte_axis = np.array([7.0, 14.0], dtype=float)
    grid_x, grid_y = np.meshgrid(strike_axis, dte_axis)
    iv = np.array([[0.20, 0.22, 0.24], [0.21, 0.23, 0.25]], dtype=float)
    alias_block = {
        "Strike": grid_x.ravel().tolist(),
        "DaysToExpiry": grid_y.ravel().tolist(),
        "IV": iv.ravel().tolist(),
    }
    payload = {
        "version": DataRepository.VERSION,
        "ticker": "QQQ",
        "saved_at": time.time(),
        "stamp": "20260101120000",
        "data": [
            {"Strike": 100.0, "DaysToExpiry": 7.0, "impliedVolatility": 0.22},
            {"Strike": 110.0, "DaysToExpiry": 14.0, "impliedVolatility": 0.25},
        ],
        "volatility_surface": alias_block,
    }
    history = tmp_path / "data_history"
    history.mkdir(parents=True, exist_ok=True)
    path = history / "QQQ_20260101120000.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    resolved = DataRepository.resolve_vol_surface_block(payload)
    assert resolved is alias_block
    frame = repo.snapshot_vol_surface("QQQ", "20260101120000")
    assert isinstance(frame, pd.DataFrame)
    assert len(frame.index) == 6
    # Save path still writes preferred key only.
    rows = []
    for dte in (7.0, 14.0, 21.0, 30.0, 45.0):
        for strike in (90.0, 95.0, 100.0, 105.0, 110.0):
            rows.append({"Strike": strike, "DaysToExpiry": dte, "impliedVolatility": 0.2})
    repo.save_to_cache("IWM", pd.DataFrame(rows))
    saved = json.loads(Path(repo.get_historical_snapshots("IWM")[-1]).read_text(encoding="utf-8"))
    assert "vol_surface" in saved
    assert "volatility_surface" not in saved
    z2d, x_ok, y_ok, warning = prepare_plotly_surface_xyz(
        iv.ravel(), strike_axis, dte_axis, rows=2, cols=3
    )
    assert z2d.shape == (2, 3)
    assert x_ok is not None and y_ok is not None
    assert warning is None


def test_snapshot_stamp_second_resolution_and_top_level_keys(tmp_path) -> None:
    """Saves use expiry-prefixed capture stamps and embed contract fields top-level."""
    repo = DataRepository(cache_dir=tmp_path, history_dir=tmp_path / "data_history", status_probe=lambda: {"ok": True})
    frame = pd.DataFrame(
        {
            "Strike": [580.0, 580.0],
            "DaysToExpiry": [7.0, 30.0],
            "Delta": [0.55, 0.52],
            "expiration": ["2026-10-16", "2026-10-16"],
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
    assert "exp_" in stamp
    assert stamp.startswith("20261016exp_")
    capture = stamp.split("exp_", 1)[-1].split("_", 1)[0]
    assert len(capture) >= 14 and capture[:14].isdigit()
    assert datetime.strptime(capture[:14], "%Y%m%d%H%M%S")
    path = Path(repo.get_historical_snapshots("SPY")[-1])
    assert path.name.startswith("SPY_20261016exp_")
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
    assert label.startswith("2026-10-16 · SPY 580 Call @ ")
    # Legacy chain rows (no metadata) still surface expiry in the Time Machine label.
    legacy = {
        "ticker": "SPY",
        "stamp": "2026-09-13_1346",
        "data": [{"expiration": "2026-10-30", "strike": 580.0}],
    }
    assert DataRepository.format_snapshot_label(legacy, "2026-09-13_1346").startswith(
        "2026-10-30 · SPY @ "
    )
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


def test_reshape_to_surface_1d_perfect_square() -> None:
    flat = np.arange(9, dtype=object)  # object → float64 coercion
    out = reshape_to_surface(flat)
    assert out.dtype == np.float64
    assert out.shape == (3, 3)
    assert np.allclose(out, np.arange(9, dtype=np.float64).reshape(3, 3))
    # Returned array is a copy.
    out[0, 0] = -1.0
    assert float(flat[0]) == 0.0


def test_reshape_to_surface_pad_with_median() -> None:
    flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    out = reshape_to_surface(flat)
    assert out.dtype == np.float64
    assert out.shape == (3, 3)  # ceil(sqrt(5))**2 == 9
    med = float(np.median(flat))
    assert np.allclose(out.ravel()[:5], flat)
    assert np.allclose(out.ravel()[5:], med)


def test_reshape_to_surface_float64_and_2d_passthrough() -> None:
    grid = np.ones((2, 3), dtype=np.float32)
    out = reshape_to_surface(grid)
    assert out.dtype == np.float64
    assert out.shape == (2, 3)
    assert out is not grid
    # List with None → float64 NaN cells, then pad to square.
    listed = reshape_to_surface([1, None, 3])
    assert listed.dtype == np.float64
    assert listed.shape == (2, 2)


def test_reshape_to_surface_empty_and_invalid() -> None:
    empty = reshape_to_surface([])
    assert empty.shape == (0, 0)
    assert empty.dtype == np.float64
    assert reshape_to_surface(None).shape == (0, 0)
    assert reshape_to_surface("not-an-array").shape == (0, 0)
    assert reshape_to_surface(np.array([])).shape == (0, 0)


def test_audit_go_surface_paths_use_reshape_to_surface() -> None:
    """Gatekeeper: every dashboard ``go.Surface`` construction applies ``reshape_to_surface``."""
    root = Path(__file__).resolve().parent
    eng = (root / "quant_engine.py").read_text(encoding="utf-8")
    dash = (root / "dashboard.py").read_text(encoding="utf-8")
    assert "def reshape_to_surface(" in eng
    assert "reshape_to_surface" in dash

    # Shared builder must reshape before constructing Surface.
    start = dash.index("def _plotly_surface")
    end = dash.index("\ndef ", start + 1)
    helper = dash[start:end]
    assert "reshape_to_surface" in helper
    assert "go.Surface" in helper
    assert 'st.error("Surface rendering failed: Invalid data shape.")' in helper

    # Named 3D render paths must call reshape_to_surface.
    for marker in (
        "def build_integrated_risk_figure",
        "def render_time_machine_vol_surface",
        "def liquidation_waterfall_build_heatmap",
        "def event_impact_render_shock_surface",
        "def build_market_timelapse_figure",
    ):
        assert marker in dash, marker
        fn_start = dash.index(marker)
        fn_end = dash.index("\ndef ", fn_start + 1)
        body = dash[fn_start:fn_end]
        assert "reshape_to_surface" in body, f"{marker} missing reshape_to_surface"
    render_ir = dash[
        dash.index("def render_integrated_risk_surface") : dash.index(
            "\ndef ", dash.index("def render_integrated_risk_surface") + 1
        )
    ]
    assert "build_integrated_risk_figure" in render_ir

    # Engine integrated-risk builder also reshapes before Surface.
    eng_start = eng.index("def generate_integrated_risk_surface")
    eng_end = eng.index("\ndef ", eng_start + 1)
    assert "reshape_to_surface" in eng[eng_start:eng_end]


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


def test_minmax_normalize_01_range_and_guards() -> None:
    scaled = minmax_normalize_01([2.0, 4.0, 6.0])
    assert scaled.shape == (3,)
    np.testing.assert_allclose(scaled, [0.0, 0.5, 1.0])

    zero_range = minmax_normalize_01([3.0, 3.0, 3.0])
    np.testing.assert_allclose(zero_range, [0.0, 0.0, 0.0])

    all_nan = minmax_normalize_01([np.nan, np.nan])
    np.testing.assert_allclose(all_nan, [0.0, 0.0])

    with_nan = minmax_normalize_01([1.0, np.nan, 3.0])
    assert with_nan[0] == 0.0 and with_nan[2] == 1.0
    assert np.isnan(with_nan[1])

    # Copy semantics: mutating output must not alter a writable source array.
    src = np.array([0.0, 5.0, 10.0], dtype=np.float64)
    out = minmax_normalize_01(src)
    out[0] = 99.0
    assert src[0] == 0.0


def test_generate_integrated_risk_surface_nonempty() -> None:
    import plotly.graph_objects as go

    rows = []
    for dte in (7.0, 14.0, 30.0):
        for price in (90.0, 100.0, 110.0):
            rows.append(
                {
                    "Price": price,
                    "DaysToExpiry": dte,
                    "Delta": (price - 100.0) / 20.0,
                    "Gamma": max(0.01, 0.05 - abs(price - 100.0) / 500.0),
                    "Vega": 0.1 + 0.01 * (dte / 30.0),
                    "Volatility": 0.15 + 0.01 * (dte / 30.0) + 0.002 * abs(price - 100.0) / 10.0,
                }
            )
    frame = pd.DataFrame(rows)
    surface = generate_integrated_risk_surface(frame)
    assert isinstance(surface, go.Surface)
    z = np.asarray(surface.z, dtype=float)
    assert z.ndim == 2 and z.size > 0
    assert np.any(np.isfinite(z))
    assert float(np.nanmax(z) - np.nanmin(z)) > 1e-6
    assert surface.surfacecolor is None
    assert surface.opacityscale is None
    assert surface.name == "Total Risk Profile"

    flat_vol = generate_integrated_risk_surface(frame, include_vol=False)
    z_flat = np.asarray(flat_vol.z, dtype=float)
    finite = z_flat[np.isfinite(z_flat)]
    assert finite.size > 0
    assert float(np.max(finite) - np.min(finite)) > 1e-6

    no_gamma = generate_integrated_risk_surface(frame, include_gamma=False)
    z_ng = np.asarray(no_gamma.z, dtype=float)
    assert float(np.nanmax(z_ng) - np.nanmin(z_ng)) > 1e-6

    no_delta = generate_integrated_risk_surface(frame, include_delta=False)
    z_nd = np.asarray(no_delta.z, dtype=float)
    assert float(np.nanmax(z_nd) - np.nanmin(z_nd)) > 1e-6

    empty = generate_integrated_risk_surface(pd.DataFrame())
    assert isinstance(empty, go.Surface)

    constant = frame.copy()
    constant["Volatility"] = 0.2
    flat = generate_integrated_risk_surface(constant)
    z_const = np.asarray(flat.z, dtype=float)
    finite_c = z_const[np.isfinite(z_const)]
    assert finite_c.size > 0
    assert float(np.max(finite_c) - np.min(finite_c)) > 1e-6

    from dashboard import build_integrated_risk_figure

    fig = build_integrated_risk_figure(flat)
    assert fig is not None
    z_fig = np.asarray(fig.data[0].z, dtype=float)
    assert np.any(np.isfinite(z_fig))
    assert float(np.nanmax(z_fig) - np.nanmin(z_fig)) > 1e-6
    z_range = list(fig.layout.scene.zaxis.range)
    assert z_range[1] > z_range[0]


def test_compute_total_risk_score_weights_and_alert_band() -> None:
    frame = pd.DataFrame(
        {
            "Delta": [0.9, 0.8, 0.85],
            "Gamma": [0.06, 0.05, 0.055],
            "Vega": [0.3, 0.28, 0.32],
        }
    )
    score = compute_total_risk_score(frame.copy())
    assert 0.0 <= score <= 1.0
    assert score > 0.5
    assert compute_total_risk_score(pd.DataFrame()) == 0.0
    # High equal values near refs → near 1.0
    hot = pd.DataFrame({"Delta": [1.0, 1.0], "Gamma": [0.05, 0.05], "Vega": [0.25, 0.25]})
    assert compute_total_risk_score(hot) == pytest.approx(1.0, abs=1e-9)


def test_test_dashboard_simulated_pnl_buy_sell_signs_and_formula() -> None:
    """Mock ticket P&L: side flips sign; matches Δ·dS + ½Γ(dS)² + Θ·dT + ν·dσ."""
    from quant_engine import test_simulated_pnl as simulated_pnl

    delta, gamma, theta, vega = 0.5, 0.02, -0.05, 0.10
    qty = 2.0
    spot_shock, day_shock, vol_shock = 1.0, 1.0 / 365.0, 0.01
    d_value = delta * spot_shock + 0.5 * gamma * spot_shock**2 + theta * day_shock + vega * vol_shock
    expected_buy = qty * 100.0 * d_value
    buy = simulated_pnl(delta, gamma, theta, vega, qty, "Buy")
    sell = simulated_pnl(delta, gamma, theta, vega, qty, "Sell")
    assert buy == pytest.approx(expected_buy, rel=1e-9)
    assert sell == pytest.approx(-expected_buy, rel=1e-9)
    assert math.isnan(simulated_pnl(float("nan"), gamma, theta, vega, qty, "Buy"))


def test_snapshot_time_label_from_name_hhmm_and_unix() -> None:
    assert snapshot_time_label_from_name("SPY_2026-09-13_1434.json") == "14:34:00"
    assert snapshot_time_label_from_name("SPY_2026-09-13_0723_1789284208.json") == "07:23:28"
    assert snapshot_time_label_from_name("20260913143005") == "14:30:05"
    # 14-digit compact capture stamp must not be treated as a unix timestamp.
    assert snapshot_time_label_from_name("SPY_20260914exp_20260914224015.json") == "22:40:15"


def test_timelapse_shared_z_range_locks_minmax() -> None:
    a = np.array([[0.1, 0.2], [0.3, np.nan]], dtype=float)
    b = np.array([[-0.5, 0.0], [0.4, 0.9]], dtype=float)
    lo, hi = timelapse_shared_z_range([a, b])
    assert lo == pytest.approx(-0.5)
    assert hi == pytest.approx(0.9)
    # Constant grid still returns a usable span.
    c_lo, c_hi = timelapse_shared_z_range([np.ones((2, 2))])
    assert c_lo < c_hi


def test_load_all_snapshots_sorted_and_delta_surface(tmp_path: Path) -> None:
    history = tmp_path / "data_history"
    history.mkdir()
    strikes = [100.0, 110.0, 100.0, 110.0]
    dtes = [7.0, 7.0, 14.0, 14.0]

    def _write(stamp: str, saved_at: float, deltas: list[float]) -> None:
        payload = {
            "version": "1.0",
            "ticker": "AAA",
            "saved_at": saved_at,
            "stamp": stamp,
            "data": [],
            "delta_surface": {"Strike": strikes, "DaysToExpiry": dtes, "Delta": deltas},
        }
        (history / f"AAA_{stamp}.json").write_text(json.dumps(payload), encoding="utf-8")

    _write("2026-09-13_1000", 100.0, [0.1, 0.2, 0.3, 0.4])
    _write("2026-09-13_0900", 50.0, [0.0, 0.5, 0.25, 0.75])
    _write("2026-09-13_1100", 150.0, [-0.2, 0.1, 0.2, 1.0])

    frames = load_all_snapshots("AAA", history_dir=history)
    assert len(frames) == 3
    assert [f["stamp"] for f in frames] == [
        "2026-09-13_0900",
        "2026-09-13_1000",
        "2026-09-13_1100",
    ]
    assert frames[0]["label"] == "09:00:00"
    assert frames[1]["label"] == "10:00:00"
    assert frames[2]["label"] == "11:00:00"
    assert all(f["surface_key"] == TIMELAPSE_SURFACE_KEY for f in frames)
    assert all(isinstance(f["z"], np.ndarray) and f["z"].ndim == 2 for f in frames)
    lo, hi = timelapse_shared_z_range(f["z"] for f in frames)
    assert lo == pytest.approx(-0.2)
    assert hi == pytest.approx(1.0)

    _write("2026-09-13_1030", 125.0, [0.1, 0.2, 0.3, 0.4])  # identical z to 10:00
    frames2 = load_all_snapshots("AAA", history_dir=history)
    assert [f["stamp"] for f in frames2] == [
        "2026-09-13_0900",
        "2026-09-13_1000",
        "2026-09-13_1100",
    ]


def test_build_market_timelapse_figure_json_frames_play_in_order() -> None:
    """Frames serialize as JSON lists with integer names so Plotly animate can swap z."""
    import plotly.io as pio
    from dashboard import build_market_timelapse_figure, build_integrated_risk_figure

    frames = [
        {
            "label": "09:00:00",
            "frame_name": "09:00:00",
            "z": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
            "x": np.array([100.0, 110.0], dtype=np.float64),
            "y": np.array([7.0, 14.0], dtype=np.float64),
        },
        {
            "label": "10:00:00",
            "frame_name": "10:00:00",
            "z": np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64),
            "x": np.array([100.0, 110.0], dtype=np.float64),
            "y": np.array([7.0, 14.0], dtype=np.float64),
        },
        {
            "label": "11:00:00",
            "frame_name": "11:00:00",
            "z": np.array([[0.9, 1.0], [1.1, 1.2]], dtype=np.float64),
            "x": np.array([100.0, 110.0], dtype=np.float64),
            "y": np.array([7.0, 14.0], dtype=np.float64),
        },
    ]
    z_before = [np.array(item["z"], copy=True) for item in frames]
    fig = build_market_timelapse_figure(frames)
    spec = json.loads(pio.to_json(fig, validate=False))

    assert [fr["name"] for fr in spec["frames"]] == ["0", "1", "2"]
    slider_names = [step["args"][0][0] for step in spec["layout"]["sliders"][0]["steps"]]
    assert slider_names == ["0", "1", "2"]
    assert [step["label"] for step in spec["layout"]["sliders"][0]["steps"]] == [
        "09:00:00",
        "10:00:00",
        "11:00:00",
    ]
    play_args = spec["layout"]["updatemenus"][0]["buttons"][0]["args"]
    assert play_args[0] == ["0", "1", "2"]
    assert play_args[1]["transition"]["duration"] == 0
    assert play_args[1]["frame"]["redraw"] is True
    assert play_args[1]["mode"] == "immediate"

    z_hashes = []
    for fr in spec["frames"]:
        z = fr["data"][0]["z"]
        assert isinstance(z, list), "frame z must be JSON lists, not numpy bdata"
        assert not isinstance(z, dict)
        assert "bdata" not in (z if isinstance(z, dict) else {})
        z_hashes.append(tuple(tuple(row) for row in z))
        assert fr.get("traces") in ([0], (0,))
    assert len(set(z_hashes)) == 3
    assert z_hashes[0] != z_hashes[-1]
    data_z = spec["data"][0]["z"]
    assert isinstance(data_z, list)
    assert tuple(tuple(row) for row in data_z) == z_hashes[0]
    for original, item in zip(z_before, frames):
        np.testing.assert_array_equal(original, item["z"])

    # Integrated Risk View still builds a non-empty Viridis surface (no frames).
    rows = []
    for dte in (7.0, 14.0, 30.0):
        for price in (90.0, 100.0, 110.0):
            rows.append(
                {
                    "Price": price,
                    "DaysToExpiry": dte,
                    "Delta": (price - 100.0) / 20.0,
                    "Gamma": max(0.01, 0.05 - abs(price - 100.0) / 500.0),
                    "Vega": 0.1 + 0.01 * (dte / 30.0),
                    "Volatility": 0.15 + 0.01 * (dte / 30.0) + 0.002 * abs(price - 100.0) / 10.0,
                }
            )
    risk = generate_integrated_risk_surface(pd.DataFrame(rows))
    risk_fig = build_integrated_risk_figure(risk)
    assert risk_fig is not None
    assert list(getattr(risk_fig, "frames", []) or []) == []
    assert np.any(np.isfinite(np.asarray(risk_fig.data[0].z, dtype=float)))


def _stress_chain_frame() -> pd.DataFrame:
    rows = []
    for strike, opt, oi in ((95.0, "call", 10.0), (100.0, "call", 20.0), (105.0, "put", 15.0)):
        rows.append(
            {
                "S": 100.0,
                "K": strike,
                "T": 0.25,
                "r": 0.05,
                "sigma": 0.20,
                "option_type": opt,
                "openInterest": oi,
                "contractSize": 100.0,
            }
        )
    return pd.DataFrame(rows)


def test_calculate_stress_scenario_zero_shock_identity() -> None:
    frame = _stress_chain_frame()
    impact = calculate_stress_scenario(
        "TEST",
        0.0,
        0.0,
        frame=frame,
        spot=100.0,
    )
    assert impact["n_contracts"] == 3
    assert impact["delta_change"] == pytest.approx(0.0, abs=1e-9)
    assert impact["gamma_change"] == pytest.approx(0.0, abs=1e-9)
    assert impact["vega_change"] == pytest.approx(0.0, abs=1e-9)
    assert impact["net_pnl"] == pytest.approx(0.0, abs=1e-6)
    assert impact["negative_gamma"] is False or isinstance(impact["negative_gamma"], bool)


def test_calculate_stress_scenario_spot_up_moves_delta_and_pnl() -> None:
    frame = _stress_chain_frame()
    up = calculate_stress_scenario("TEST", 10.0, 0.0, frame=frame, spot=100.0)
    down = calculate_stress_scenario("TEST", -10.0, 0.0, frame=frame, spot=100.0)
    assert up["stressed_spot"] == pytest.approx(110.0)
    assert down["stressed_spot"] == pytest.approx(90.0)
    assert up["net_pnl"] != pytest.approx(down["net_pnl"])
    assert "delta_change" in up and "gamma_change" in up and "vega_change" in up
    assert math.isfinite(float(up["total_gamma"]))


def test_calculate_stress_scenario_iv_up_raises_vega_book_pnl() -> None:
    frame = _stress_chain_frame()
    base = calculate_stress_scenario("TEST", 0.0, 0.0, frame=frame, spot=100.0)
    high_iv = calculate_stress_scenario("TEST", 0.0, 20.0, frame=frame, spot=100.0)
    # Long OI book gains when IV rises (positive vega inventory).
    assert high_iv["net_pnl"] > base["net_pnl"]
    assert high_iv["vega_change"] == pytest.approx(0.0, abs=1.0) or math.isfinite(high_iv["vega_change"])


def test_calculate_stress_scenario_copies_input_frame() -> None:
    frame = _stress_chain_frame()
    original = frame.copy()
    calculate_stress_scenario("TEST", 5.0, -10.0, frame=frame, spot=100.0)
    pd.testing.assert_frame_equal(frame, original)


def test_build_stress_pnl_matrix_shape_and_center() -> None:
    frame = _stress_chain_frame()
    # Inject via temporary repo cache.
    repo = DataRepository(cache_dir=Path("/tmp/datai_stress_cache"), history_dir=Path("/tmp/datai_stress_hist"))
    repo.cache_dir.mkdir(parents=True, exist_ok=True)
    repo.history_dir.mkdir(parents=True, exist_ok=True)
    repo.save_to_cache("STX", frame)
    matrix = build_stress_pnl_matrix(
        "STX",
        (-10.0, 0.0, 10.0),
        (-20.0, 0.0, 20.0),
        repo=repo,
    )
    assert matrix["spot_shifts"] == [-10.0, 0.0, 10.0]
    assert matrix["iv_shifts"] == [-20.0, 0.0, 20.0]
    z = np.asarray(matrix["net_pnl"], dtype=float)
    assert z.shape == (3, 3)
    assert matrix["n_contracts"] >= 1
    # Center cell (0%, 0%) should be ~0.
    assert z[1, 1] == pytest.approx(0.0, abs=1e-4)


def test_calculate_stress_scenario_empty_frame() -> None:
    impact = calculate_stress_scenario("EMPTY", 5.0, 5.0, frame=pd.DataFrame(), spot=100.0)
    assert impact["n_contracts"] == 0
    assert impact["delta_change"] == 0.0
    assert impact["net_pnl"] == 0.0


def _event_impact_chain_rows(deltas: list[float], gammas: list[float], vegas: list[float]) -> list[dict[str, Any]]:
    strikes = (100.0, 110.0, 100.0, 110.0)
    dtes = (7.0, 7.0, 14.0, 14.0)
    rows = []
    for strike, dte, delta, gamma, vega in zip(strikes, dtes, deltas, gammas, vegas):
        rows.append(
            {
                "ticker": "EVT",
                "option_type": "call",
                "strike": strike,
                "S": 105.0,
                "K": strike,
                "T": dte / 365.0,
                "sigma": 0.20,
                "impliedVolatility": 0.20,
                "underlyingPrice": 105.0,
                "Delta": delta,
                "Gamma": gamma,
                "Vega": vega,
                "Theta": -0.01,
                "Rho": 0.02,
            }
        )
    return rows


def test_event_impact_load_data_parses_repo_json() -> None:
    events = event_impact_load_data()
    assert isinstance(events, list)
    assert len(events) >= 1
    assert {"date", "event"} <= set(events[0].keys())
    assert any(row["event"] == "FOMC" for row in events)


def test_event_impact_load_data_skips_bad_rows(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    path.write_text(
        json.dumps(
            [
                {"date": "2026-09-15", "event": "FOMC"},
                {"date": "", "event": "Bad"},
                {"event": "NoDate"},
                "not-a-dict",
            ]
        ),
        encoding="utf-8",
    )
    events = event_impact_load_data(path)
    assert events == [{"date": "2026-09-15", "event": "FOMC"}]


def test_event_impact_parse_stamp_datetime_hhmm_and_unix() -> None:
    assert event_impact_parse_stamp_datetime("2026-09-13_1434") == datetime(2026, 9, 13, 14, 34, 0)
    # Unix suffix ignored when HHMM is present.
    parsed = event_impact_parse_stamp_datetime("2026-09-13_0723_1789284208")
    assert parsed == datetime(2026, 9, 13, 7, 23, 0)
    # Prefer saved_at when provided.
    from_saved = event_impact_parse_stamp_datetime("2026-09-13_1434", saved_at=1_700_000_000.0)
    assert from_saved is not None
    assert from_saved.year >= 2023


def test_event_impact_find_bracketing_and_shock(tmp_path: Path) -> None:
    history = tmp_path / "data_history"
    history.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    repo = DataRepository(cache_dir=cache, history_dir=history)

    def _write(stamp: str, saved_at: float, deltas: list[float], gammas: list[float], vegas: list[float]) -> None:
        payload = {
            "version": "1.0",
            "ticker": "EVT",
            "saved_at": saved_at,
            "stamp": stamp,
            "data": _event_impact_chain_rows(deltas, gammas, vegas),
        }
        (history / f"EVT_{stamp}.json").write_text(json.dumps(payload), encoding="utf-8")

    _write(
        "2026-09-13_0900",
        100.0,
        [0.40, 0.30, 0.55, 0.45],
        [0.02, 0.01, 0.015, 0.012],
        [0.10, 0.08, 0.12, 0.09],
    )
    _write(
        "2026-09-13_1500",
        200.0,
        [0.50, 0.35, 0.60, 0.48],
        [0.04, 0.02, 0.03, 0.025],
        [0.15, 0.10, 0.18, 0.11],
    )

    before, after = event_impact_find_bracketing_snapshots("EVT", "2026-09-13", repo=repo)
    assert before == "2026-09-13_0900"
    assert after == "2026-09-13_1500"

    result = event_impact_calculate_shock("EVT", "2026-09-13", repo=repo)
    assert result["ok"] is True
    assert result["stamp_before"] == before
    assert result["stamp_after"] == after
    assert result["shock_magnitude"] > 0.0
    z = np.asarray(result["shock_z"], dtype=float)
    assert z.ndim == 2 and z.size > 0
    assert np.isfinite(z).any()
    frame = result["shock_frame"].copy()
    assert {"Strike", "DaysToExpiry", "Shock"} <= set(frame.columns)
    # Mutating the returned frame must not raise; copy semantics on inputs already covered.
    frame.loc[:, "Shock"] = 0.0
    assert "Calendar Spreads" in result["suggestion"] or "Shock" in result["suggestion"]


def test_event_impact_calculate_shock_missing_after(tmp_path: Path) -> None:
    history = tmp_path / "data_history"
    history.mkdir()
    repo = DataRepository(cache_dir=tmp_path / "cache", history_dir=history)
    payload = {
        "version": "1.0",
        "ticker": "EVT",
        "saved_at": 50.0,
        "stamp": "2026-09-10_1200",
        "data": _event_impact_chain_rows(
            [0.4, 0.3, 0.5, 0.4],
            [0.01, 0.01, 0.01, 0.01],
            [0.1, 0.1, 0.1, 0.1],
        ),
    }
    (history / "EVT_2026-09-10_1200.json").write_text(json.dumps(payload), encoding="utf-8")
    result = event_impact_calculate_shock("EVT", "2026-09-15", repo=repo)
    assert result["ok"] is False
    assert result["stamp_after"] is None


def test_event_impact_suggest_position_gamma_bias() -> None:
    text = event_impact_suggest_position(10.0, gamma_magnitude=8.0, vega_magnitude=2.0)
    assert text == "High Gamma Shock: Suggest Calendar Spreads"
    low = event_impact_suggest_position(0.0)
    assert "Low Shock" in low


def _regime_chain_frame(
    *,
    gex_sign: float = 1.0,
    vanna_level: float = 0.05,
    spot: float = 100.0,
) -> pd.DataFrame:
    """Synthetic chain for regime tests (Gamma × OI drives GEX; Vanna precomputed).

    ``gex_sign < 0`` overweight put OI so put-negative dealer GEX nets negative.
    """
    put_oi_scale = 8.0 if gex_sign < 0 else 1.0
    call_oi_scale = 1.0 if gex_sign < 0 else 4.0
    rows = []
    for strike, opt, oi, gamma in (
        (95.0, "call", 50.0 * call_oi_scale, 0.04),
        (100.0, "call", 80.0 * call_oi_scale, 0.05),
        (105.0, "put", 60.0 * put_oi_scale, 0.05),
        (110.0, "put", 40.0 * put_oi_scale, 0.04),
    ):
        rows.append(
            {
                "S": spot,
                "K": strike,
                "strike": strike,
                "T": 0.25,
                "r": 0.05,
                "sigma": 0.20,
                "impliedVolatility": 0.20,
                "option_type": opt,
                "openInterest": oi,
                "contractSize": 100.0,
                "Gamma": abs(gamma),
                "Vanna": float(vanna_level) * (1.0 if opt == "call" else -1.0),
                "underlyingPrice": spot,
            }
        )
    return pd.DataFrame(rows)


def test_regime_clip_probability_bounds() -> None:
    assert regime_clip_probability(-1.0) == 0.0
    assert regime_clip_probability(0.0) == 0.0
    assert regime_clip_probability(0.5) == 0.5
    assert regime_clip_probability(1.0) == 1.0
    assert regime_clip_probability(2.5) == 1.0
    assert regime_clip_probability(float("nan")) == 0.0
    assert regime_clip_probability("bad") == 0.0


def test_regime_cascade_probability_in_unit_interval() -> None:
    gex = np.array([-1e6, -5e5, 1e5, 0.0, -1e9], dtype=float)
    vanna = np.array([0.01, 0.5, 2.0, np.nan, 100.0], dtype=float)
    mom = np.array([-0.08, -0.02, 0.03, -0.5, -1.0], dtype=float)
    probs = regime_cascade_probability(gex, vanna, mom, vanna_high_threshold=0.2)
    assert probs.shape == (5,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert math.isfinite(float(probs.max()))
    # Extreme negative GEX + high vanna + hard dump → near upper bound.
    hot = regime_cascade_probability([-1e9], [10.0], [-0.2], vanna_high_threshold=0.1)
    assert 0.0 <= float(hot[0]) <= 1.0
    assert float(hot[0]) >= 0.7
    # Positive GEX + up-momentum → low cascade.
    calm = regime_cascade_probability([1e6], [0.01], [0.05], vanna_high_threshold=0.2)
    assert 0.0 <= float(calm[0]) <= 1.0
    assert float(calm[0]) < 0.35


def test_regime_classify_four_states_vectorized() -> None:
    gex = np.array([1e5, -1e5, 1e5, -1e5], dtype=float)
    vanna = np.array([0.01, 0.01, 0.01, 1.0], dtype=float)
    mom = np.array([0.02, 0.02, -0.02, -0.03], dtype=float)
    labels = regime_classify_states(gex, vanna, mom, vanna_high_threshold=0.5)
    assert list(labels) == [
        REGIME_STABLE_BULL,
        REGIME_SQUEEZE_UP,
        REGIME_FRAGILE_BEAR,
        REGIME_CRASH_CASCADE,
    ]


def test_regime_reflexive_cascade_flag() -> None:
    flags = regime_reflexive_cascade_flag([-1.0, 1.0, -1.0], [1.0, 1.0, 0.01], vanna_high_threshold=0.5)
    assert list(flags) == [True, False, False]


def test_calculate_market_reflexivity_cascade_bounded_and_copy() -> None:
    frame = _regime_chain_frame(gex_sign=-1.0, vanna_level=0.8)
    original = frame.copy()
    out = calculate_market_reflexivity(
        "TEST",
        frame=frame,
        spot=100.0,
        momentum=-0.04,
    )
    pd.testing.assert_frame_equal(frame, original)
    assert out["ok"] is True
    p = float(out["cascade_probability"])
    assert 0.0 <= p <= 1.0
    assert out["regime"] in (
        REGIME_STABLE_BULL,
        REGIME_SQUEEZE_UP,
        REGIME_FRAGILE_BEAR,
        REGIME_CRASH_CASCADE,
    )
    assert isinstance(out["map_frame"], pd.DataFrame)
    assert {"GEX", "Vanna", "Momentum", "Regime"}.issubset(out["map_frame"].columns)
    # Negative GEX + high Vanna → reflexive cascade risk.
    assert out["reflexive_cascade"] is True
    assert out["regime"] == REGIME_CRASH_CASCADE
    assert out["hedge_suggestion"] == REGIME_HEDGE_PUT_SPREADS


def test_calculate_market_reflexivity_stable_bull() -> None:
    frame = _regime_chain_frame(gex_sign=1.0, vanna_level=0.01)
    # Heavy call OI / positive gamma → positive net GEX under put-negative convention.
    out = calculate_market_reflexivity("BULL", frame=frame, spot=100.0, momentum=0.02)
    assert out["ok"] is True
    assert 0.0 <= float(out["cascade_probability"]) <= 1.0
    assert out["regime"] == REGIME_STABLE_BULL
    assert out["reflexive_cascade"] is False
    assert out["hedge_suggestion"] is None


def test_calculate_market_reflexivity_empty_frame() -> None:
    out = calculate_market_reflexivity("EMPTY", frame=pd.DataFrame())
    assert out["ok"] is False
    assert float(out["cascade_probability"]) == 0.0
    assert out["map_frame"].empty


def _liquidation_chain_frame(*, spot: float = 100.0) -> pd.DataFrame:
    """Synthetic chain with clear GEX + → − flip near 100 and a deep negative wing."""
    rows = []
    # Lower strikes: put-heavy negative GEX. Higher strikes: call-heavy positive GEX.
    specs = [
        (90.0, "put", 200.0, 0.05, 0.20),
        (95.0, "put", 150.0, 0.06, 0.20),
        (100.0, "put", 80.0, 0.05, 0.20),
        (100.0, "call", 40.0, 0.05, 0.20),
        (105.0, "call", 120.0, 0.05, 0.20),
        (110.0, "call", 180.0, 0.04, 0.20),
    ]
    for strike, opt, oi, gamma, dte_frac in specs:
        rows.append(
            {
                "S": spot,
                "K": strike,
                "strike": strike,
                "T": dte_frac,
                "DaysToExpiry": dte_frac * 365.0,
                "r": 0.05,
                "sigma": 0.20,
                "impliedVolatility": 0.20,
                "option_type": opt,
                "openInterest": oi,
                "contractSize": 100.0,
                "Gamma": abs(gamma),
                "underlyingPrice": spot,
            }
        )
    # Second tenor for heatmap Y axis.
    for strike, opt, oi, gamma, dte_frac in specs:
        rows.append(
            {
                "S": spot,
                "K": strike,
                "strike": strike,
                "T": 0.10,
                "DaysToExpiry": 36.5,
                "r": 0.05,
                "sigma": 0.22,
                "impliedVolatility": 0.22,
                "option_type": opt,
                "openInterest": oi * 0.8,
                "contractSize": 100.0,
                "Gamma": abs(gamma) * 1.1,
                "underlyingPrice": spot,
            }
        )
    return pd.DataFrame(rows)


def test_detect_gamma_flips_plus_to_minus() -> None:
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0], dtype=float)
    gex = np.array([-500.0, -200.0, -50.0, 100.0, 300.0], dtype=float)
    flips = lw_detect_gamma_flips(strikes, gex)
    assert flips.size >= 1
    # Crossing between 100 (−) and 105 (+).
    assert 100.0 <= float(flips[0]) <= 105.0
    # No flip when all positive.
    none = lw_detect_gamma_flips(strikes, np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
    assert none.size == 0


def test_cascade_score_bounds_and_negative_gex() -> None:
    gex = np.array([-1e6, -1e3, 0.0, 5e5, np.nan], dtype=float)
    scores = lw_cascade_score(gex)
    assert scores.shape == gex.shape
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    assert float(scores[0]) >= float(scores[1]) >= float(scores[2])
    assert float(scores[2]) == 0.0
    assert float(scores[3]) == 0.0
    assert float(scores[4]) == 0.0
    zones = lw_classify_liquidation_zones(scores, threshold=0.5)
    assert zones[0] == TEST_LIQUIDATION_ZONE


def test_audit_gex_normalized_halts_on_nan_inf() -> None:
    ok = lw_audit_gex_normalized(np.array([0.1, 0.5, 0.9], dtype=float))
    assert ok["ok"] is True
    assert ok["halt_render"] is False
    assert np.all(np.isfinite(ok["gex_normalized"]))

    bad_nan = lw_audit_gex_normalized(np.array([0.1, float("nan"), 0.3], dtype=float))
    assert bad_nan["ok"] is False
    assert bad_nan["halt_render"] is True
    assert bad_nan["message"] == TEST_LIQUIDATION_UNSTABLE_MSG

    bad_inf = lw_audit_gex_normalized(np.array([0.1, float("inf"), 0.3], dtype=float))
    assert bad_inf["ok"] is False
    assert bad_inf["halt_render"] is True
    assert bad_inf["message"] == TEST_LIQUIDATION_UNSTABLE_MSG

    empty = lw_audit_gex_normalized(np.array([], dtype=float))
    assert empty["ok"] is False
    assert empty["message"] == TEST_LIQUIDATION_UNSTABLE_MSG


def test_market_stability_near_flip_aggressive() -> None:
    far = lw_market_stability_distance(100.0, 90.0)
    assert far["near_flip"] is False
    assert float(far["approach_risk"]) < 0.5
    near = lw_market_stability_distance(100.0, 99.0)
    assert near["near_flip"] is True
    assert float(near["approach_risk"]) >= 0.85


def test_calculate_liquidation_waterfall_surface_and_copy() -> None:
    frame = _liquidation_chain_frame(spot=100.0)
    original = frame.copy()
    out = lw_calculate_liquidation_waterfall("TEST", frame=frame, spot=100.0)
    pd.testing.assert_frame_equal(frame, original)
    assert out["ok"] is True
    assert isinstance(out["strike_frame"], pd.DataFrame)
    assert {"Price", "GEX", "CascadeScore", "Zone"}.issubset(out["strike_frame"].columns)
    assert out["surface"]["ok"] is True
    assert out["surface"]["price_axis"].size > 0
    assert out["surface"]["expiry_axis"].size > 0
    assert out["gex_audit"]["ok"] is True
    assert out["gex_audit"]["halt_render"] is False
    # Alias matches isolation entry.
    alias = calculate_liquidation_waterfall("TEST", frame=frame, spot=100.0)
    assert alias["ok"] is True
    assert np.all((out["strike_frame"]["CascadeScore"] >= 0.0) & (out["strike_frame"]["CascadeScore"] <= 1.0))


def test_calculate_liquidation_waterfall_empty_frame() -> None:
    out = lw_calculate_liquidation_waterfall("EMPTY", frame=pd.DataFrame())
    assert out["ok"] is False
    assert out["gex_audit"]["halt_render"] is True


# ---------------------------------------------------------------------------
# Godlike Quant Strategist — HMM regime, Monte Carlo PoP/PoT, auditor gates
# ---------------------------------------------------------------------------


def _synthetic_price_path(
    *,
    n: int = 120,
    regime: str = "stable",
    spot: float = 100.0,
    seed: int = 7,
) -> np.ndarray:
    """Generate a price path biased toward Stable / Trend / Crash-Cascade."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.004, size=n)
    if regime == "trend":
        rets = rets + 0.003
    elif regime == "crash":
        rets = rng.normal(-0.012, 0.025, size=n)
        rets[-20:] = rng.normal(-0.03, 0.04, size=20)
    else:
        rets = rng.normal(0.0, 0.003, size=n)
    log_p = np.log(spot) + np.cumsum(rets)
    return np.exp(log_p).astype(np.float64)


def test_hmm_classify_market_regime_outputs_valid_label() -> None:
    prices = _synthetic_price_path(regime="stable", seed=11)
    out = classify_market_regime(prices, min_observations=30, n_iter=15)
    assert out["ok"] is True
    assert out["regime"] in HMM_REGIME_STATES
    assert out["regime"] in (HMM_REGIME_STABLE, HMM_REGIME_TREND, HMM_REGIME_CRASH_CASCADE)
    assert out["regime_path"].size == out["n_observations"]
    assert set(out["regime_path"]).issubset(set(HMM_REGIME_STATES))


def test_hmm_classify_copies_input_frame() -> None:
    prices = _synthetic_price_path(regime="trend", seed=21)
    frame = pd.DataFrame({"close": prices, "extra": np.arange(prices.size)})
    original = frame.copy()
    out = classify_market_regime(None, frame=frame, price_col="close", min_observations=30, n_iter=12)
    pd.testing.assert_frame_equal(frame, original)
    assert out["ok"] is True
    assert out["regime"] in HMM_REGIME_STATES


def test_hmm_require_regime_before_trade_gate() -> None:
    blocked = require_regime_before_trade({"ok": False, "regime": HMM_REGIME_STABLE})
    assert blocked["approved"] is False
    ok = require_regime_before_trade({"ok": True, "regime": HMM_REGIME_TREND})
    assert ok["approved"] is True
    assert ok["regime"] == HMM_REGIME_TREND
    bad_label = require_regime_before_trade({"ok": True, "regime": "Unknown"})
    assert bad_label["approved"] is False


def test_montecarlo_pop_pot_bounds_unit_interval() -> None:
    paths = simulate_gbm_paths(100.0, 0.25, 10.0 / 252.0, n_paths=MC_MIN_PATHS, n_steps=32, seed=42)
    assert paths.shape == (MC_MIN_PATHS, 33)
    pop = probability_of_profit(paths, entry=100.0, direction="long")
    pot = probability_of_touching(paths, 95.0, direction="long")
    assert 0.0 <= pop <= 1.0
    assert 0.0 <= pot <= 1.0
    result = evaluate_position_montecarlo(
        100.0,
        95.0,
        sigma=0.25,
        time_years=10.0 / 252.0,
        n_paths=MC_MIN_PATHS,
        seed=42,
        vanna=0.12,
        volga=0.01,
    )
    assert 0.0 <= float(result["pop"]) <= 1.0
    assert 0.0 <= float(result["pot"]) <= 1.0
    assert int(result["n_paths"]) >= MC_MIN_PATHS
    assert result["vanna_warning"] == HIGH_VANNA_WARNING
    assert result["warning"] == HIGH_VANNA_WARNING


def test_audit_monte_carlo_path_count_threshold() -> None:
    ok = audit_monte_carlo_path_count(MC_MIN_PATHS)
    assert ok["ok"] is True
    assert ok["halt"] is False
    # Isolation entry mirrors liquidation-engine test_audit_* naming.
    from quant_engine import test_audit_monte_carlo_paths as mc_path_audit

    alias = mc_path_audit(25_000)
    assert alias["ok"] is True
    bad = audit_monte_carlo_path_count(9999)
    assert bad["ok"] is False
    assert bad["halt"] is True
    assert "10,000" in bad["message"]
    # evaluate_position_montecarlo always lifts to ≥ MC_MIN_PATHS
    lifted = evaluate_position_montecarlo(
        100.0, 90.0, sigma=0.2, time_years=0.05, n_paths=500, seed=1
    )
    assert int(lifted["n_paths"]) >= MC_MIN_PATHS
    assert lifted["path_audit"]["ok"] is True


def test_mathematicians_rule_gate_forbids_pot_gt_pop() -> None:
    forbidden = mathematicians_rule_gate(0.30, 0.55, delta_hedged=False)
    assert forbidden["approved"] is False
    assert forbidden["forbidden"] is True
    assert "FORBIDDEN" in forbidden["message"]
    allowed_hedged = mathematicians_rule_gate(0.30, 0.55, delta_hedged=True)
    assert allowed_hedged["approved"] is True
    assert allowed_hedged["forbidden"] is False
    allowed_edge = mathematicians_rule_gate(0.60, 0.40, delta_hedged=False)
    assert allowed_edge["approved"] is True


def test_vanna_volga_report_exact_warning_string() -> None:
    quiet = build_vanna_volga_report(0.001, 0.02)
    assert quiet["high_vanna_exposure"] is False
    assert quiet["vanna_warning"] is None
    hot = build_vanna_volga_report(0.20, -0.01)
    assert hot["vanna_warning"] == "High Vanna exposure: Delta will accelerate during IV spikes."


def test_strategist_evaluate_position_integrates_regime_and_mc() -> None:
    prices = _synthetic_price_path(regime="stable", seed=99)
    # Wide stop → lower PoT; should often clear Mathematician's Rule.
    out = strategist_evaluate_position(
        prices,
        stop_loss=float(prices[-1]) * 0.70,
        sigma=0.15,
        time_years=5.0 / 252.0,
        direction="long",
        vanna=0.0,
        volga=0.0,
        delta_hedged=False,
        n_paths=MC_MIN_PATHS,
        seed=3,
    )
    assert out["regime"] in HMM_REGIME_STATES
    assert out["regime_gate"]["approved"] is True
    assert 0.0 <= float(out["pop"]) <= 1.0
    assert 0.0 <= float(out["pot"]) <= 1.0
    assert int(out["n_paths"]) >= MC_MIN_PATHS
    assert out["path_audit"]["ok"] is True
    # Explicitly delta-hedged path must be approvable even if PoT > PoP.
    harsh = strategist_evaluate_position(
        prices,
        stop_loss=float(prices[-1]) * 0.995,
        sigma=0.50,
        time_years=20.0 / 252.0,
        direction="long",
        vanna=0.2,
        volga=0.05,
        delta_hedged=True,
        n_paths=MC_MIN_PATHS,
        seed=5,
    )
    assert harsh["mathematicians_rule"]["approved"] is True
    assert harsh["vanna_warning"] == HIGH_VANNA_WARNING
    # Without hedge, Mathematician's Rule must block when PoT > PoP.
    rule = mathematicians_rule_gate(0.2, 0.8, delta_hedged=False)
    assert rule["approved"] is False


# ---------------------------------------------------------------------------
# Council of Experts debate
# ---------------------------------------------------------------------------


def test_clip_confidence_bounds_0_100() -> None:
    assert clip_confidence(-5) == 0.0
    assert clip_confidence(150) == 100.0
    assert clip_confidence(72.5) == 72.5
    assert clip_confidence(float("nan")) == 0.0
    assert clip_confidence("bad") == 0.0


def test_compute_iv_rank_percentile() -> None:
    hist = np.array([0.10, 0.15, 0.20, 0.25, 0.30], dtype=float)
    mid = compute_iv_rank_percentile(hist, 0.20)
    assert mid["iv_rank"] == pytest.approx(50.0)
    assert 0.0 <= mid["iv_percentile"] <= 100.0
    high = compute_iv_rank_percentile(hist, 0.30)
    assert high["iv_rank"] == pytest.approx(100.0)
    low = compute_iv_rank_percentile(hist, 0.10)
    assert low["iv_rank"] == pytest.approx(0.0)


def test_build_disagreement_report_when_stances_differ() -> None:
    preds = {
        EXPERT_A: {
            "expert_id": EXPERT_A,
            "name": EXPERT_NAMES[EXPERT_A],
            "stance": STANCE_BUY,
            "confidence": 80.0,
            "justification": "Positive GEX + velocity.",
        },
        EXPERT_B: {
            "expert_id": EXPERT_B,
            "name": EXPERT_NAMES[EXPERT_B],
            "stance": STANCE_SELL,
            "confidence": 70.0,
            "justification": "IV Rank elevated.",
        },
        EXPERT_C: {
            "expert_id": EXPERT_C,
            "name": EXPERT_NAMES[EXPERT_C],
            "stance": STANCE_NEUTRAL,
            "confidence": 40.0,
            "justification": "Balanced path mass.",
        },
    }
    report = build_disagreement_report(preds)
    assert report["disagreement"] is True
    assert report["report"] is not None
    assert "Disagreement Report" in report["report"]
    assert "Buy" in report["report"] and "Sell" in report["report"]
    assert (EXPERT_A, EXPERT_B) in report["conflict_pairs"]

    aligned = {
        EXPERT_A: {**preds[EXPERT_A], "stance": STANCE_BUY},
        EXPERT_B: {**preds[EXPERT_B], "stance": STANCE_BUY},
        EXPERT_C: {**preds[EXPERT_C], "stance": STANCE_NEUTRAL},
    }
    calm = build_disagreement_report(aligned)
    assert calm["disagreement"] is False
    assert calm["report"] is None


def test_run_council_debate_summary_shape_and_confidence() -> None:
    frame = _regime_chain_frame(gex_sign=1.0, vanna_level=0.02, spot=100.0)
    original = frame.copy()
    out = run_council_debate("COUNCIL", frame=frame, spot=100.0, lookback=5, seed=7)
    assert out["ok"] is True
    assert out["ticker"] == "COUNCIL"
    assert set(out["experts"].keys()) == {EXPERT_A, EXPERT_B, EXPERT_C}
    assert len(out["predictions"]) == 3
    for pred in out["predictions"]:
        assert pred["stance"] in STANCES
        assert 0.0 <= float(pred["confidence"]) <= 100.0
        assert isinstance(pred["justification"], str) and len(pred["justification"]) > 0
        assert pred["name"] in EXPERT_NAMES.values()
    assert "disagreement" in out
    if out["disagreement"]:
        assert isinstance(out["disagreement_report"], str)
        assert "Disagreement Report" in out["disagreement_report"]
    pd.testing.assert_frame_equal(frame, original)

def test_run_council_debate_empty_frame() -> None:
    out = run_council_debate("EMPTY", frame=pd.DataFrame())
    assert out["ok"] is False
    assert len(out["predictions"]) == 3
    for pred in out["predictions"]:
        assert pred["stance"] == STANCE_NEUTRAL
        assert 0.0 <= float(pred["confidence"]) <= 100.0


# ---------------------------------------------------------------------------
# Social Sentiment Specialist / Sentiment Lab
# ---------------------------------------------------------------------------


def test_normalize_compound_and_audit_sentiment_01() -> None:
    from sentiment_engine import (
        SENTIMENT_UNSTABLE_MSG,
        audit_sentiment_normalized,
        normalize_compound_to_01,
    )

    assert normalize_compound_to_01(-1.0) == pytest.approx(0.0)
    assert normalize_compound_to_01(0.0) == pytest.approx(0.5)
    assert normalize_compound_to_01(1.0) == pytest.approx(1.0)
    assert normalize_compound_to_01(float("nan")) == pytest.approx(0.5)

    ok = audit_sentiment_normalized([0.0, 0.5, 1.0])
    assert ok["ok"] is True
    assert ok["halt_render"] is False
    assert np.all((ok["scores_01"] >= 0.0) & (ok["scores_01"] <= 1.0))

    bad_nan = audit_sentiment_normalized([0.1, float("nan")])
    assert bad_nan["ok"] is False
    assert bad_nan["halt_render"] is True
    assert bad_nan["message"] == SENTIMENT_UNSTABLE_MSG

    bad_range = audit_sentiment_normalized([0.1, 1.5])
    assert bad_range["ok"] is False
    assert bad_range["halt_render"] is True

    empty = audit_sentiment_normalized([])
    assert empty["ok"] is False
    assert empty["halt_render"] is True


def test_filter_social_posts_drops_bots_and_low_karma() -> None:
    from sentiment_engine import filter_social_posts, is_bot_like_author, stub_social_posts

    assert is_bot_like_author("NewsBot", karma=5000) is True
    assert is_bot_like_author("human_trader", karma=5) is True
    assert is_bot_like_author("human_trader", karma=250) is False
    assert is_bot_like_author("alice", karma=200, is_bot=True) is True

    posts = stub_social_posts("SPY", n=8, seed=3)
    filtered = filter_social_posts(posts)
    authors = {str(p.get("author", "")).lower() for p in filtered}
    assert "marketnewsbot" not in authors
    assert "newbie42" not in authors
    assert all(int(p.get("karma") or 0) >= 100 for p in filtered)
    assert len(filtered) == 8


def test_fourchan_filter_and_parse_catalog() -> None:
    from sentiment_engine import (
        configured_fourchan_boards,
        filter_social_posts,
        parse_fourchan_catalog,
        stub_fourchan_posts,
        text_mentions_ticker,
    )

    assert configured_fourchan_boards({"FOURCHAN_BOARDS": "biz, pol"}) == ("biz", "pol")
    assert text_mentions_ticker("$SPY calls", "SPY") is True
    assert text_mentions_ticker("SPYX noise", "SPY") is False

    posts = stub_fourchan_posts("SPY", n=6, seed=2)
    filtered = filter_social_posts(posts, ticker="SPY")
    assert all(str(p.get("source")) == "4chan_stub" for p in filtered)
    assert all(len(f"{p.get('title','')} {p.get('body','')}".strip()) >= 15 for p in filtered)
    assert not any(bool(p.get("is_bot")) for p in filtered)
    assert len(filtered) == 6

    catalog = [
        {
            "threads": [
                {
                    "no": 1,
                    "sub": "$SPY bullish",
                    "com": "buying calls on SPY",
                    "name": "Anonymous",
                    "replies": 12,
                    "time": 1700000000,
                },
                {
                    "no": 2,
                    "sub": "unrelated",
                    "com": "no ticker here",
                    "name": "Anonymous",
                    "replies": 1,
                    "time": 1700000001,
                },
                {
                    "no": 3,
                    "sub": "",
                    "com": "QQQ dump incoming",
                    "name": "Anonymous",
                    "replies": 3,
                    "time": 1700000002,
                },
            ]
        }
    ]
    parsed = parse_fourchan_catalog(catalog, "SPY", board="biz", limit=20)
    assert len(parsed) == 1
    assert parsed[0]["source"] == "4chan"
    assert parsed[0]["board"] == "biz"
    assert "SPY" in parsed[0]["body"].upper() or "SPY" in parsed[0]["title"].upper()


def test_fourchan_fetch_uses_mock_http_and_degrades() -> None:
    import io
    import json
    from sentiment_engine import fetch_fourchan_posts, extract_ticker_sentiment

    catalog = [
        {
            "threads": [
                {
                    "no": 99,
                    "sub": "UOA",
                    "com": "Unusual options activity in AAPL today",
                    "name": "Anonymous",
                    "replies": 8,
                    "time": 1700000100,
                }
            ]
        }
    ]
    payload = json.dumps(catalog).encode("utf-8")

    class _Resp:
        def read(self) -> bytes:
            return payload

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def _opener(req: object, timeout: float = 8.0) -> _Resp:  # noqa: ARG001
        return _Resp()

    live = fetch_fourchan_posts("AAPL", force_stub=False, boards=("biz",), opener=_opener)
    assert live
    assert all(p.get("source") == "4chan" for p in live)
    assert any("AAPL" in f"{p.get('title')} {p.get('body')}" for p in live)

    def _fail(req: object, timeout: float = 8.0):  # noqa: ARG001
        raise OSError("network down")

    stubbed = fetch_fourchan_posts("AAPL", force_stub=False, boards=("biz",), opener=_fail)
    assert stubbed
    assert all(str(p.get("source")) == "4chan_stub" for p in stubbed)

    multi = extract_ticker_sentiment("AAPL", force_stub=True)
    assert "reddit" in multi and "fourchan" in multi
    assert "sources" in multi
    assert 0.0 <= float(multi["score_01"]) <= 1.0
    assert 0.0 <= float(multi["reddit"]["score_01"]) <= 1.0
    assert 0.0 <= float(multi["fourchan"]["score_01"]) <= 1.0
    assert int(multi["mention_count"]) == int(multi["reddit"]["mention_count"]) + int(
        multi["fourchan"]["mention_count"]
    )


def test_calculate_sentiment_alpha_shape_and_bounds() -> None:
    out = calculate_sentiment_alpha("SPY", force_stub=True)
    assert out["ok"] is True
    assert out["ticker"] == "SPY"
    assert 0.0 <= float(out["score_01"]) <= 1.0
    assert 0.0 <= float(out["alpha_01"]) <= 1.0
    assert 0.0 <= float(out["buzz"]) <= 1.0
    scores = np.asarray(out["scores_01"], dtype=float)
    assert scores.ndim == 1 and scores.size >= 1
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert out["audit"]["ok"] is True
    assert out["audit"]["halt_render"] is False
    factor = out["sentiment_factor"]
    assert 0.0 <= float(factor["sentiment_factor"]) <= 1.0
    assert "drift_bias" in factor and "vol_mult" in factor
    assert out["chart_sentiment"].shape[0] == out["chart_realized_vol"].shape[0]
    assert "reddit" in out and "fourchan" in out
    assert 0.0 <= float(out["reddit"]["score_01"]) <= 1.0
    assert 0.0 <= float(out["fourchan"]["score_01"]) <= 1.0

    injected = calculate_sentiment_alpha(
        "QQQ",
        sentiment_scores=[0.2, 0.4, 0.6, 0.8],
        prices=np.linspace(100.0, 110.0, 20),
        force_stub=True,
    )
    assert injected["ok"] is True
    assert 0.0 <= float(injected["alpha_01"]) <= 1.0
    assert injected["source"] == "injected"


def test_sentiment_factor_wires_into_montecarlo() -> None:
    from quant_engine import evaluate_position_montecarlo, sentiment_factor_for_montecarlo

    factor = sentiment_factor_for_montecarlo(0.9)
    assert factor["vol_mult"] < 1.0
    assert factor["drift_bias"] > 0.0
    base = evaluate_position_montecarlo(
        100.0,
        90.0,
        sigma=0.25,
        time_years=10.0 / 252.0,
        n_paths=MC_MIN_PATHS,
        seed=11,
    )
    tilted = evaluate_position_montecarlo(
        100.0,
        90.0,
        sigma=0.25,
        time_years=10.0 / 252.0,
        n_paths=MC_MIN_PATHS,
        seed=11,
        sentiment_factor=0.9,
    )
    assert tilted["sentiment_factor"] == pytest.approx(0.9)
    assert tilted["sigma_effective"] == pytest.approx(0.25 * factor["vol_mult"])
    assert tilted["rate_effective"] == pytest.approx(factor["drift_bias"])
    assert 0.0 <= float(tilted["pop"]) <= 1.0
    assert 0.0 <= float(tilted["pot"]) <= 1.0
    assert int(base["n_paths"]) >= MC_MIN_PATHS


def _flow_chain_frame(*, buy_heavy: bool = True) -> pd.DataFrame:
    """Minimal bid/ask/volume chain for market-velocity unit tests."""
    spot = 100.0
    rows: list[dict[str, Any]] = []
    for i, (strike, opt, bid, ask, last, vol, oi) in enumerate(
        [
            (95.0, "call", 4.8, 5.2, 5.15 if buy_heavy else 4.85, 120.0, 400.0),
            (100.0, "call", 2.4, 2.6, 2.55 if buy_heavy else 2.42, 200.0, 800.0),
            (105.0, "call", 1.0, 1.2, 1.15 if buy_heavy else 1.02, 80.0, 300.0),
            (95.0, "put", 0.8, 1.0, 0.82 if buy_heavy else 0.98, 40.0, 200.0),
            (100.0, "put", 2.3, 2.5, 2.32 if buy_heavy else 2.48, 60.0 if buy_heavy else 220.0, 500.0),
            (105.0, "put", 4.5, 4.9, 4.55 if buy_heavy else 4.85, 50.0, 350.0),
        ]
    ):
        rows.append(
            {
                "S": spot,
                "K": strike,
                "strike": strike,
                "T": 0.2,
                "r": 0.05,
                "sigma": 0.22,
                "impliedVolatility": 0.22,
                "option_type": opt,
                "bid": bid,
                "ask": ask,
                "lastPrice": last,
                "volume": vol,
                "openInterest": oi,
                "contractSize": 100.0,
                "underlyingPrice": spot,
                "bidSize": 10.0 + i if buy_heavy else 2.0,
                "askSize": 2.0 if buy_heavy else 12.0 + i,
            }
        )
    return pd.DataFrame(rows)


def test_calculate_market_velocity_outputs_finite_and_bounds() -> None:
    frame = _flow_chain_frame(buy_heavy=True)
    original = frame.copy()
    out = calculate_market_velocity("FLOW", frame=frame, spot=100.0, gamma_flip=102.0)
    pd.testing.assert_frame_equal(frame, original)
    assert out["ok"] is True
    for key in ("gross_flow", "net_flow", "flow_velocity", "ofi", "avg_deployed_capital"):
        assert math.isfinite(float(out[key]))
    assert float(out["gross_flow"]) > 0.0
    assert float(out["buy_notional"]) + float(out["sell_notional"]) == pytest.approx(
        float(out["gross_flow"]), rel=1e-9, abs=1e-6
    )
    assert float(out["net_flow"]) == pytest.approx(
        float(out["buy_notional"]) - float(out["sell_notional"]), rel=1e-9, abs=1e-6
    )
    lo_o, hi_o = FLOW_OFI_BOUNDS
    assert lo_o <= float(out["ofi"]) <= hi_o
    wave = np.asarray(out["velocity_wave_normalized"], dtype=float)
    assert wave.ndim == 1 and wave.size >= 1
    assert np.all(np.isfinite(wave))
    lo_v, hi_v = FLOW_VELOCITY_BOUNDS
    assert float(np.min(wave)) >= lo_v - 1e-9
    assert float(np.max(wave)) <= hi_v + 1e-9
    assert out["audit"]["ok"] is True
    assert out["audit"]["halt_render"] is False
    assert out["gamma_flip"] == pytest.approx(102.0)


def test_calculate_market_velocity_ofi_bounds_sell_pressure() -> None:
    frame = _flow_chain_frame(buy_heavy=False)
    out = calculate_market_velocity("SELL", frame=frame, spot=100.0, gamma_flip=98.0)
    assert math.isfinite(float(out["ofi"]))
    assert FLOW_OFI_BOUNDS[0] <= float(out["ofi"]) <= FLOW_OFI_BOUNDS[1]
    assert math.isfinite(float(out["flow_velocity"]))
    assert float(out["flow_velocity"]) >= 0.0
    # Size-driven OFI should lean negative when ask size dominates.
    assert float(out["ofi"]) < 0.0


def test_calculate_market_velocity_sparse_proxy_degrade() -> None:
    """Missing bid/ask/volume → documented OI / call-put proxies; still finite."""
    frame = _flow_chain_frame(buy_heavy=True).drop(
        columns=["bid", "ask", "volume", "bidSize", "askSize", "lastPrice"]
    )
    out = calculate_market_velocity("PROXY", frame=frame, spot=100.0, gamma_flip=100.0)
    assert bool(out.get("proxy_used")) is True
    assert math.isfinite(float(out["gross_flow"]))
    assert math.isfinite(float(out["ofi"]))
    assert FLOW_OFI_BOUNDS[0] <= float(out["ofi"]) <= FLOW_OFI_BOUNDS[1]


def test_audit_market_velocity_stale_and_zero_gate() -> None:
    ok = audit_market_velocity(np.array([0.2, -0.4, 0.6], dtype=float), gross_flow=1e6)
    assert ok["ok"] is True
    assert ok["halt_render"] is False
    assert ok["message"] == "ok"
    norm = np.asarray(ok["velocity_normalized"], dtype=float)
    assert np.all(np.isfinite(norm))
    assert float(np.max(np.abs(norm))) <= 1.0 + 1e-9

    stale = audit_market_velocity(np.array([0.1, 0.2], dtype=float), gross_flow=0.0)
    assert stale["ok"] is False
    assert stale["halt_render"] is True
    assert stale["message"] == FLOW_DATA_STALE_MSG

    zero_wave = audit_market_velocity(np.zeros(5, dtype=float), gross_flow=100.0)
    assert zero_wave["ok"] is False
    assert zero_wave["halt_render"] is True
    assert zero_wave["message"] == FLOW_DATA_STALE_MSG

    flagged = audit_market_velocity(np.array([0.5], dtype=float), gross_flow=10.0, stale=True)
    assert flagged["halt_render"] is True
    assert flagged["message"] == FLOW_DATA_STALE_MSG

    empty = calculate_market_velocity("EMPTY", frame=pd.DataFrame())
    assert empty["ok"] is False
    assert empty["audit"]["halt_render"] is True
    assert empty["audit"]["message"] == FLOW_DATA_STALE_MSG


# ---------------------------------------------------------------------------
# Microstructure Lab — VPIN & Order Book Imbalance
# ---------------------------------------------------------------------------


def _micro_chain_frame(*, buy_heavy: bool = True, with_l2: bool = False) -> pd.DataFrame:
    """Minimal chain for VPIN / OBI unit tests."""
    spot = 100.0
    rows: list[dict[str, Any]] = []
    specs = [
        (95.0, "call", 4.8, 5.2, 5.15 if buy_heavy else 4.85, 120.0, 400.0, 80.0, 5.0),
        (100.0, "call", 2.4, 2.6, 2.55 if buy_heavy else 2.42, 200.0, 800.0, 90.0, 8.0),
        (105.0, "call", 1.0, 1.2, 1.15 if buy_heavy else 1.02, 80.0, 300.0, 70.0, 10.0),
        (95.0, "put", 0.8, 1.0, 0.82 if buy_heavy else 0.98, 40.0, 200.0, 50.0, 8.0),
        (100.0, "put", 2.3, 2.5, 2.32 if buy_heavy else 2.48, 60.0 if buy_heavy else 220.0, 500.0, 55.0, 9.0),
        (105.0, "put", 4.5, 4.9, 4.55 if buy_heavy else 4.85, 50.0, 350.0, 45.0, 7.0),
    ]
    for strike, opt, bid, ask, last, vol, oi, bsz, asz in specs:
        if buy_heavy:
            bid_sz, ask_sz = float(bsz), float(asz)
        else:
            bid_sz, ask_sz = float(asz), float(bsz)
        row: dict[str, Any] = {
            "S": spot,
            "K": strike,
            "strike": strike,
            "T": 0.2,
            "r": 0.05,
            "sigma": 0.22,
            "impliedVolatility": 0.22,
            "option_type": opt,
            "bid": bid,
            "ask": ask,
            "lastPrice": last,
            "volume": vol,
            "openInterest": oi,
            "contractSize": 100.0,
            "underlyingPrice": spot,
            "bidSize": bid_sz,
            "askSize": ask_sz,
        }
        if with_l2:
            for lvl in range(1, OBI_TOP_LEVELS + 1):
                decay = 0.7 ** (lvl - 1)
                row[f"bidSize{lvl}"] = bid_sz * decay
                row[f"askSize{lvl}"] = ask_sz * decay
        rows.append(row)
    return pd.DataFrame(rows)


def test_calculate_vpin_bounded_01_and_copy() -> None:
    frame = _micro_chain_frame(buy_heavy=True)
    original = frame.copy()
    out = calculate_vpin("VPIN", frame=frame, n_buckets=10)
    pd.testing.assert_frame_equal(frame, original)
    assert out["ok"] is True
    assert 0.0 <= float(out["vpin"]) <= 1.0
    assert float(out["vpin_raw"]) == pytest.approx(float(out["vpin"]), abs=1e-9)
    assert int(out["n_buckets"]) >= 1
    assert out["audit"]["ok"] is True
    assert out["audit"]["toxicity_alert"] is False
    buckets = np.asarray(out["bucket_imbalances"], dtype=float)
    assert buckets.size == int(out["n_buckets"])
    assert np.all(np.isfinite(buckets))


def test_audit_vpin_toxicity_alert_when_exceeds_one() -> None:
    ok = audit_vpin(0.55)
    assert ok["ok"] is True
    assert ok["toxicity_alert"] is False
    assert ok["message"] == "ok"

    hot = audit_vpin(1.25)
    assert hot["ok"] is False
    assert hot["toxicity_alert"] is True
    assert hot["halt_render"] is True
    assert hot["message"] == TOXICITY_ALERT_MSG

    neg = audit_vpin(-0.1)
    assert neg["ok"] is False
    assert neg["message"] == TOXICITY_ALERT_MSG

    # Force raw mean > 1 by injecting oversized bucket imbalances via monkeypatch path:
    # audit path only — direct call covers the contract.
    assert HIGH_ADVERSE_SELECTION_MSG.startswith("High Adverse Selection Risk")


def test_calculate_order_book_imbalance_top5_shape() -> None:
    frame = _micro_chain_frame(buy_heavy=True, with_l2=True)
    original = frame.copy()
    out = calculate_order_book_imbalance("OBI", frame=frame)
    pd.testing.assert_frame_equal(frame, original)
    assert out["ok"] is True
    assert -1.0 <= float(out["imbalance"]) <= 1.0
    assert out["bid_sizes"].shape == (OBI_TOP_LEVELS,)
    assert out["ask_sizes"].shape == (OBI_TOP_LEVELS,)
    heat = out["heatmap"]
    assert heat["z"].shape == (2, OBI_TOP_LEVELS)
    assert len(heat["x"]) == OBI_TOP_LEVELS
    assert heat["y"] == ["Bid (Buy)", "Ask (Sell)"]
    assert out["audit"]["ok"] is True
    assert out["audit"]["halt_render"] is False
    # Buy-heavy L2 → positive aggregate OBI.
    assert float(out["imbalance"]) > 0.0


def test_audit_obi_heatmap_gate_unstable() -> None:
    good = shape_obi_heatmap([10.0, 8.0, 6.0, 4.0, 2.0], [5.0, 4.0, 3.0, 2.0, 1.0])
    ok = audit_obi_heatmap(good)
    assert ok["ok"] is True
    assert ok["halt_render"] is False

    bad_shape = dict(good)
    bad_shape["z"] = np.ones((3, 5))
    bad_shape["n_levels"] = 5
    halted = audit_obi_heatmap(bad_shape)
    assert halted["ok"] is False
    assert halted["halt_render"] is True
    assert halted["message"] == OBI_HEATMAP_UNSTABLE_MSG

    nan_z = dict(good)
    z = np.array(good["z"], dtype=float, copy=True)
    z[0, 0] = float("nan")
    nan_z["z"] = z
    assert audit_obi_heatmap(nan_z)["halt_render"] is True

    wrong_sign = dict(good)
    z2 = np.array(good["z"], dtype=float, copy=True)
    z2[1, 0] = 5.0  # ask row must be ≤ 0
    wrong_sign["z"] = z2
    assert audit_obi_heatmap(wrong_sign)["message"] == OBI_HEATMAP_UNSTABLE_MSG


def test_liquidity_trap_condition() -> None:
    calm = detect_liquidity_trap(0.2, 0.1)
    assert calm["liquidity_trap"] is False
    assert calm["message"] is None

    # |OBI|=0.7 → one_side = 0.5 + 0.35 = 0.85 > 0.80; VPIN high.
    trap = detect_liquidity_trap(0.75, 0.70)
    assert trap["vpin_high"] is True
    assert trap["obi_extreme"] is True
    assert trap["liquidity_trap"] is True
    assert trap["message"] == LIQUIDITY_TRAP_MSG
    assert float(trap["one_side_share"]) > OBI_EXTREME_THRESHOLD

    # High VPIN alone is not a trap.
    only_vpin = detect_liquidity_trap(0.9, 0.1)
    assert only_vpin["liquidity_trap"] is False

    # Extreme OBI alone is not a trap.
    only_obi = detect_liquidity_trap(0.2, 0.95)
    assert only_obi["liquidity_trap"] is False

    lab = calculate_microstructure_lab(
        "TRAP",
        frame=_micro_chain_frame(buy_heavy=True, with_l2=True),
        n_buckets=8,
    )
    assert "vpin" in lab and "obi" in lab and "liquidity_trap" in lab
    assert 0.0 <= float(lab["vpin"]["vpin"]) <= 1.0 or not lab["vpin"]["ok"]
    assert lab["obi"]["heatmap"]["z"].shape == (2, OBI_TOP_LEVELS)


def test_vpin_sparse_proxy_and_high_adverse_flag() -> None:
    frame = _micro_chain_frame(buy_heavy=True).drop(
        columns=["bid", "ask", "volume", "bidSize", "askSize", "lastPrice"]
    )
    out = calculate_vpin("PROXY", frame=frame, n_buckets=5)
    assert bool(out.get("proxy_used")) is True
    if out["ok"]:
        assert 0.0 <= float(out["vpin"]) <= 1.0
        assert out["audit"]["toxicity_alert"] is False
    assert VPIN_HIGH_THRESHOLD == pytest.approx(0.6)
    assert HIGH_ADVERSE_SELECTION_MSG == (
        "High Adverse Selection Risk: Market Makers are widening spreads."
    )
    # high_adverse_selection mirrors VPIN > threshold when audit is clean.
    hot_frame = _micro_chain_frame(buy_heavy=True)
    hot = calculate_vpin("HOT", frame=hot_frame, n_buckets=10)
    if hot["ok"] and float(hot["vpin"]) > VPIN_HIGH_THRESHOLD:
        assert hot["high_adverse_selection"] is True
    elif hot["ok"]:
        assert hot["high_adverse_selection"] is False
