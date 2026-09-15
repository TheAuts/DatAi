"""DatAi Streamlit dashboard: Price vs Delta / Gamma / Theta / Rho."""

from __future__ import annotations

import hashlib
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_ingestion import (
    chain_expiry_mismatch_reason,
    get_available_expirations,
)
from chart_views import render_charts_section
from quant_engine import (
    DAYS_PER_YEAR,
    DEFAULT_HESTON_KAPPA,
    DEFAULT_HESTON_RHO,
    DEFAULT_HESTON_SIGMA,
    DEFAULT_HESTON_THETA,
    DEFAULT_HESTON_V0,
    DEFAULT_RATE,
    DEFAULT_SIGMA,
    CONCENTRATION_WARN_PCT,
    EVENT_INSUFFICIENT_COVERAGE,
    GAMMA_FLIP_NEAR_PCT,
    MODEL_BLACK_SCHOLES,
    MODEL_HESTON,
    DataRepository,
    prepare_plotly_surface_xyz,
    reshape_to_surface,
    analyze_event_outlook,
    analyze_gex_outlook,
    analyze_market_sentiment,
    analyze_volatility_risk_outlook,
    calculate_charm,
    calculate_color,
    calculate_gamma_theta_ratio,
    calculate_greeks,
    calculate_portfolio_risk,
    calculate_stress_scenario,
    build_stress_pnl_matrix,
    calculate_market_reflexivity,
    REGIME_CRASH_CASCADE,
    REGIME_FRAGILE_BEAR,
    REGIME_HEDGE_PUT_SPREADS,
    test_calculate_liquidation_waterfall,
    test_audit_gex_normalized,
    test_simulate_liquidation_greeks,
    TEST_LIQUIDATION_UNSTABLE_MSG,
    TEST_LIQUIDATION_ZONE,
    TEST_STABILITY_NEAR_PCT,
    calculate_speed,
    calculate_ultima,
    calculate_vanna,
    calculate_veta,
    calculate_volga,
    calculate_vomma,
    calculate_zomma,
    generate_3d_gamma_surface,
    generate_3d_greek_surface,
    generate_delta_term_structure,
    generate_greek_curve,
    RISK_SCORE_ALERT_THRESHOLD,
    RISK_WEIGHT_DELTA,
    RISK_WEIGHT_GAMMA,
    RISK_WEIGHT_VEGA,
    compute_total_risk_score,
    generate_integrated_risk_surface,
    generate_pro_surface_data,
    generate_vol_surface_data,
    load_all_snapshots,
    prefill_drift_data,
    test_simulated_pnl,
    timelapse_shared_z_range,
    TIMELAPSE_SURFACE_KEY,
    event_impact_calculate_shock,
    event_impact_load_data,
    run_council_debate,
    STANCE_BUY,
    STANCE_SELL,
    STANCE_NEUTRAL,
    calculate_sentiment_alpha,
    audit_sentiment_normalized,
    SENTIMENT_UNSTABLE_MSG,
    calculate_market_velocity,
    audit_market_velocity,
    FLOW_DATA_STALE_MSG,
    calculate_vpin,
    calculate_order_book_imbalance,
    calculate_microstructure_lab,
    audit_vpin,
    audit_obi_heatmap,
    detect_liquidity_trap,
    TOXICITY_ALERT_MSG,
    LIQUIDITY_TRAP_MSG,
    HIGH_ADVERSE_SELECTION_MSG,
    OBI_HEATMAP_UNSTABLE_MSG,
    VPIN_HIGH_THRESHOLD,
    OBI_EXTREME_THRESHOLD,
    diagnose_surface_anomaly,
    build_anomaly_ghost_column_meshes,
    ANOMALY_LABEL_GAMMA_WALL,
    ANOMALY_LABEL_REGIME_MISMATCH,
)

DATA_REPO = DataRepository()

PAGE_TITLE = "DatAi"
DEFAULT_TICKER = "SPY"
GREEK_COLUMNS = ("Delta", "Gamma", "Theta", "Vega", "Rho")
CHART_ROWS = (("Delta", "Gamma"), ("Theta", "Vega"), ("Rho",))
PORTFOLIO_COLUMNS = (
    "Ticker",
    "Strike",
    "Expiry",
    "Side",
    "Quantity",
    "option_type",
    "S",
    "sigma",
    "premium",
    "EntryPrice",
)


def _empty_portfolio_positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ticker": pd.Series([""], dtype="string"),
            "Strike": pd.Series([np.nan], dtype="float64"),
            "Expiry": pd.Series([""], dtype="string"),
            "Side": pd.Series(["long"], dtype="string"),
            "Quantity": pd.Series([1.0], dtype="float64"),
            "option_type": pd.Series(["call"], dtype="string"),
            "S": pd.Series([np.nan], dtype="float64"),
            "sigma": pd.Series([float(DEFAULT_SIGMA)], dtype="float64"),
            "premium": pd.Series([np.nan], dtype="float64"),
            "EntryPrice": pd.Series([np.nan], dtype="float64"),
        }
    )


def _normalize_portfolio_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return _empty_portfolio_positions()
    out = frame.copy()
    rename = {}
    lookup = {str(col).strip().lower(): col for col in out.columns}
    aliases = {
        "ticker": "Ticker",
        "symbol": "Ticker",
        "underlying": "Ticker",
        "strike": "Strike",
        "k": "Strike",
        "expiry": "Expiry",
        "expiration": "Expiry",
        "side": "Side",
        "quantity": "Quantity",
        "qty": "Quantity",
        "contracts": "Quantity",
        "option_type": "option_type",
        "type": "option_type",
        "s": "S",
        "spot": "S",
        "underlyingprice": "S",
        "sigma": "sigma",
        "iv": "sigma",
        "impliedvolatility": "sigma",
        "premium": "premium",
        "lastprice": "premium",
        "mark": "premium",
        "entryprice": "EntryPrice",
        "entry_price": "EntryPrice",
        "entry": "EntryPrice",
    }
    for key, dest in aliases.items():
        if dest not in out.columns and key in lookup:
            rename[lookup[key]] = dest
    if rename:
        out = out.rename(columns=rename)
    if "Side" in out.columns:
        side_text = out["Side"].astype(str).str.strip().str.lower()
        is_cp = side_text.str.startswith("p") | side_text.str.startswith("c")
        if bool(is_cp.any()):
            if "option_type" not in out.columns:
                out["option_type"] = "call"
            out.loc[is_cp, "option_type"] = np.where(side_text.loc[is_cp].str.startswith("p"), "put", "call")
    for column in PORTFOLIO_COLUMNS:
        if column not in out.columns:
            template = _empty_portfolio_positions()[column]
            out[column] = template.iloc[0]
    return out.loc[:, list(PORTFOLIO_COLUMNS)]


CSV_COLUMNS = ("Price", "Delta", "Gamma", "Theta", "Vega", "IV")
GREEK_COLORS = {
    "Delta": "#58a6ff",
    "Gamma": "#3fb950",
    "Theta": "#d29922",
    "Vega": "#f778ba",
    "Rho": "#bc8cff",
    "IV": "#8b949e",
    "Vanna": "#79c0ff",
    "Volga": "#ffa657",
    "Vomma": "#ffa657",
    "Zomma": "#db6d28",
    "Veta": "#39d353",
    "Ultima": "#e3b341",
    "Charm": "#d2a8ff",
    "Speed": "#f0883e",
    "Color": "#a371f7",
}
RATIO_THRESHOLD = 0.5
PRICE_SPAN = 0.30
CURVE_POINTS = 201
PRO_METRICS = ("Vanna", "Vomma", "Zomma", "Veta", "Ultima")
PRO_METRIC_HELP = (
    "Vanna: sensitivity of delta to volatility. "
    "Vomma: sensitivity of vega to volatility. "
    "Zomma: sensitivity of gamma to volatility. "
    "Veta: sensitivity of vega to time. "
    "Ultima: sensitivity of vomma to volatility."
)
DARK_BG = "#0e1117"
PANEL_BG = "#161b22"
CARD_BG = "#1c2330"
GRID = "#30363d"
TEXT = "#e6edf3"
STATUS_LIVE = "#3fb950"
STATUS_HISTORICAL = "#d29922"


def _data_status_parts(repo: DataRepository | None = None) -> tuple[bool, str | None]:
    """Return ``(is_historical, YYYY-MM-DD|None)`` for the global status badge."""
    as_of = _historical_date_iso()
    if as_of:
        return True, as_of
    source = repo if repo is not None else _get_repo()
    if getattr(source, "is_historical", False):
        return True, getattr(source, "status_date", None)
    return False, None


def render_global_data_status(repo: DataRepository | None = None) -> None:
    """Persistent LIVE / HISTORICAL badge (sidebar + main header)."""
    historical, as_of = _data_status_parts(repo)
    if historical:
        label = f"HISTORICAL: {as_of}" if as_of else "HISTORICAL"
        css_class = "historical"
        color = STATUS_HISTORICAL
    else:
        label = "LIVE"
        css_class = "live"
        color = STATUS_LIVE
    st.markdown(
        f"""
        <style>
        .data-status-badge {{
            display: inline-block;
            font-weight: 700;
            letter-spacing: 0.06em;
            font-variant-numeric: tabular-nums;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            border: 1px solid {color};
            background: rgba(22, 27, 34, 0.85);
        }}
        .data-status-badge.live {{
            color: {STATUS_LIVE};
            text-shadow: 0 0 8px {STATUS_LIVE}, 0 0 18px rgba(63, 185, 80, 0.55);
            box-shadow: 0 0 12px rgba(63, 185, 80, 0.35);
        }}
        .data-status-badge.historical {{
            color: {STATUS_HISTORICAL};
            text-shadow: 0 0 8px {STATUS_HISTORICAL}, 0 0 18px rgba(210, 153, 34, 0.55);
            box-shadow: 0 0 12px rgba(210, 153, 34, 0.35);
        }}
        .data-status-wrap {{
            display: flex;
            justify-content: flex-end;
            margin: 0.15rem 0 0.6rem 0;
        }}
        </style>
        <div class="data-status-wrap">
            <span class="data-status-badge {css_class}">{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "ticker": DEFAULT_TICKER,
        "expiry": date.today() + timedelta(days=30),
        "current_price": 0.0,
        "option_type": "call",
        "last_latency_ms": None,
        "last_ok": False,
        "recent_tickers": [DEFAULT_TICKER],
        "greeks_frame": None,
        "manual_strike": False,
        "manual_strike_text": "",
        "price_ticker": "",
        "recent_pick": DEFAULT_TICKER,
        "expiry_iso": None,
        "listed_strike": None,
        "pricing_model": "Black-Scholes",
        "heston_v0": DEFAULT_HESTON_V0,
        "heston_kappa": DEFAULT_HESTON_KAPPA,
        "heston_theta": DEFAULT_HESTON_THETA,
        "heston_sigma": DEFAULT_HESTON_SIGMA,
        "heston_rho": DEFAULT_HESTON_RHO,
        "show_second_order": False,
        "show_gamma_flip_line": True,
        "advanced_view_3d": False,
        "advanced_surface_greek": "Charm",
        "main_view": "Greeks",
        "app_section": "Greeks",
        "chart_interval": "D",
        "chart_lookback": 180,
        "chart_source": "TradingView",
        "sim_fingerprint": None,
        "iv_fingerprint": None,
        "persisted_iv": None,
        "gamma_surface_frame": None,
        "gamma_surface_fp": None,
        "greek_surface_frame": None,
        "greek_surface_fp": None,
        "term_structure_frame": None,
        "term_structure_fp": None,
        "vanna_volga_frame": None,
        "vanna_volga_fp": None,
        "vol_surface_frame": None,
        "vol_surface_fp": None,
        "run_heston_gamma_surface": False,
        "run_heston_term": False,
        "run_heston_pro_surface": False,
        "pro_metric_choice": "Vanna",
        "pro_surface_smoothing": False,
        "portfolio_csv_sig": None,
        "snapshot_saved": False,
        "snapshot_skipped": False,
        "snapshot_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not str(st.session_state.ticker).strip():
        st.session_state.ticker = DEFAULT_TICKER
    if "portfolio_positions" not in st.session_state or not isinstance(st.session_state.portfolio_positions, pd.DataFrame):
        st.session_state.portfolio_positions = _empty_portfolio_positions()
    if "data_repo" not in st.session_state:
        st.session_state.data_repo = DATA_REPO


def _get_repo() -> DataRepository:
    repo = st.session_state.get("data_repo")
    if repo is None or type(repo) is not DataRepository or not hasattr(repo, "get_available_snapshots"):
        st.session_state.data_repo = DATA_REPO
        return DATA_REPO
    return repo


def _api_status_ok(status: Any) -> bool:
    if status is False or status is None:
        return False
    if isinstance(status, dict):
        return bool(status.get("ok"))
    return bool(status)


def _surface_ingest_alerts(repo: DataRepository | None = None, frame: Any = None) -> None:
    """Show ``API Token Missing`` / ``API Error: [code]`` from repo or mock frame attrs."""
    status: dict[str, Any] = {}
    if repo is not None:
        status = dict(getattr(repo, "last_ingest_status", None) or {})
        ui_msg = None
        try:
            ui_msg = repo.ui_error_message()
        except Exception:
            ui_msg = None
        if ui_msg:
            if "Token Missing" in ui_msg:
                st.error(ui_msg)
            else:
                st.warning(ui_msg)
            return
    if isinstance(frame, pd.DataFrame):
        try:
            if frame.attrs.get("token_missing"):
                st.error("API Token Missing")
                return
            if frame.attrs.get("is_mock"):
                code = frame.attrs.get("api_error")
                if code is None or code == "":
                    code = "RequestFailed"
                st.warning(f"API Error: {code}")
                return
        except Exception:
            pass
    if status.get("token_missing"):
        st.error("API Token Missing")
    elif status and not status.get("ok"):
        code = status.get("status_code")
        if code is None or code == "":
            code = "RequestFailed"
        st.warning(f"API Error: {code}")


def _apply_theme() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {DARK_BG}; color: {TEXT}; }}
        [data-testid="stSidebar"] {{ background-color: {PANEL_BG}; }}
        [data-testid="stMetric"] {{
            background-color: {CARD_BG};
            border: 1px solid {GRID};
            border-radius: 12px;
            padding: 0.85rem 1rem;
        }}
        [data-testid="stMetricValue"] {{
            font-variant-numeric: tabular-nums;
            font-size: 1.55rem;
        }}
        [data-testid="stMetricLabel"] {{
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        div[data-testid="stVerticalBlock"] > div {{ gap: 0.6rem; }}
        div[data-testid="stSegmentedControl"] {{
            margin: 0.15rem 0 0.85rem 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_section_switcher() -> str:
    """Top-of-page Greeks | Charts section control."""
    selected = st.segmented_control(
        "Section",
        options=("Greeks", "Charts"),
        key="app_section",
        help=(
            "Greeks: Integrated Risk View, Market Timelapse, and all option-risk tabs. "
            "Charts: underlying candlestick and 3D volume view."
        ),
    )
    current = str(selected or st.session_state.get("app_section") or "Greeks")
    if current not in {"Greeks", "Charts"}:
        return "Greeks"
    return current


def _heston_from_state() -> tuple[float, float, float, float, float]:
    return (
        float(st.session_state.get("heston_v0", DEFAULT_HESTON_V0)),
        float(st.session_state.get("heston_kappa", DEFAULT_HESTON_KAPPA)),
        float(st.session_state.get("heston_theta", DEFAULT_HESTON_THETA)),
        float(st.session_state.get("heston_sigma", DEFAULT_HESTON_SIGMA)),
        float(st.session_state.get("heston_rho", DEFAULT_HESTON_RHO)),
    )


def _model_from_state() -> str:
    return MODEL_HESTON if str(st.session_state.get("pricing_model")) == "Heston" else MODEL_BLACK_SCHOLES


def _sim_fingerprint(
    ticker: str,
    strike: float,
    expiry: date,
    option_type: str,
    current_price: float,
    sigma: float,
    model: str,
    v0: float,
    kappa: float,
    theta: float,
    heston_sigma: float,
    rho: float,
) -> tuple[Any, ...]:
    heston = (
        round(float(v0), 8),
        round(float(kappa), 8),
        round(float(theta), 8),
        round(float(heston_sigma), 8),
        round(float(rho), 8),
    )
    if str(model) != MODEL_HESTON:
        heston = (0.0, 0.0, 0.0, 0.0, 0.0)
    return (
        ticker,
        round(float(strike), 8),
        expiry.isoformat() if isinstance(expiry, date) else str(expiry),
        str(option_type),
        round(float(current_price), 6),
        round(float(sigma), 8),
        str(model),
        *heston,
    )


def _clear_persisted_results() -> None:
    st.session_state.greeks_frame = None
    st.session_state.sim_fingerprint = None
    st.session_state.gamma_surface_frame = None
    st.session_state.gamma_surface_fp = None
    st.session_state.greek_surface_frame = None
    st.session_state.greek_surface_fp = None
    st.session_state.term_structure_frame = None
    st.session_state.term_structure_fp = None
    st.session_state.vanna_volga_frame = None
    st.session_state.vanna_volga_fp = None
    st.session_state.vol_surface_frame = None
    st.session_state.vol_surface_fp = None


def _price_range(spot: float, span: float = PRICE_SPAN, n: int = CURVE_POINTS) -> tuple[float, ...]:
    lo = max(spot * (1.0 - span), 1e-6)
    hi = spot * (1.0 + span)
    if hi <= lo:
        hi = lo * 1.01
    return tuple(float(x) for x in np.linspace(lo, hi, max(int(n), 200)))


def _historical_date_iso() -> str | None:
    """Selected 'Fetch Historical Data' date as YYYY-MM-DD, or None for live data."""
    picked = st.session_state.get("historical_date")
    if isinstance(picked, (list, tuple)):
        picked = picked[0] if picked else None
    if isinstance(picked, date):
        return picked.isoformat()
    return None


def _expiry_to_iso(expiry: Any) -> str | None:
    """Normalize sidebar expiry (date / ISO string) to ``YYYY-MM-DD`` for API ``expiration``."""
    if expiry is None:
        return None
    if isinstance(expiry, date):
        return expiry.isoformat()
    text = str(expiry).strip()[:10]
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def _repo_chain(ticker: str, expiry: Any = None) -> pd.DataFrame:
    repo = _get_repo()
    symbol = str(ticker or "").strip().upper()
    as_of = _historical_date_iso()
    expiration = _expiry_to_iso(expiry)
    try:
        health = repo.verify_connection("SPY")
        if isinstance(health, dict) and health.get("token_missing"):
            from data_ingestion import mock_option_chain

            frame = mock_option_chain(symbol, token_missing=True, message="API Token Missing")
            st.session_state["ingest_alert"] = "API Token Missing"
            return frame
    except Exception:
        pass
    try:
        if as_of:
            # Historical EOD: keep ``date`` path; still forward selected expiration when set.
            frame = repo.get_data(symbol, expiration, date=as_of)
        else:
            # Live/future: pass expiration through so MarketData gets ``?expiration=``.
            frame = repo.get_data(symbol, expiration)
            # Empty scoped fetch: do NOT fall back to unscoped next-monthly.
            from data_ingestion import is_mock_frame

            if (
                expiration
                and isinstance(frame, pd.DataFrame)
                and frame.empty
                and not is_mock_frame(frame)
            ):
                msg = (
                    f"No contracts found for expiration {expiration}. "
                    "Not falling back to an unscoped (next monthly) chain."
                )
                st.warning(msg)
                st.session_state["ingest_alert"] = msg
    except Exception as exc:
        from data_ingestion import mock_option_chain

        frame = mock_option_chain(symbol, error_code="Exception", message=str(exc))
    msg = None
    try:
        msg = repo.ui_error_message()
    except Exception:
        msg = None
    if msg:
        st.session_state["ingest_alert"] = msg
    elif isinstance(frame, pd.DataFrame) and frame.attrs.get("is_mock"):
        code = frame.attrs.get("api_error") or "RequestFailed"
        st.session_state["ingest_alert"] = (
            "API Token Missing" if frame.attrs.get("token_missing") else f"API Error: {code}"
        )
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _reset_time_machine_keys(ticker: str) -> None:
    symbol = str(ticker or "").strip().upper()
    for key in (f"tm_start_{symbol}", f"tm_end_{symbol}", f"tm_scrub_{symbol}"):
        if key in st.session_state:
            del st.session_state[key]


def _fresh_chain_for_snapshot(symbol: str, expiry: Any = None) -> pd.DataFrame:
    """Bypass every cache layer (st.cache_data and the repo TTL cache) so a snapshot
    captures live data.

    When a specific expiration was requested and the live fetch is empty/mock, do
    **not** fall back to an unscoped chain (MarketData next monthly) — that would
    save the wrong expiry under the UI's future metadata.
    """
    from data_ingestion import is_mock_frame

    repo = _get_repo()
    live_spot: float | None = None
    try:
        from data_ingestion import get_current_price as _uncached_price

        live_spot = _uncached_price(symbol)
    except Exception:
        live_spot = None
    for cached_fn in (cached_analyst_chain, cached_available_expirations, cached_vol_surface):
        clear = getattr(cached_fn, "clear", None)
        if callable(clear):
            try:
                clear()
            except Exception:
                pass
    frame = pd.DataFrame()
    as_of = _historical_date_iso()
    expiration = _expiry_to_iso(expiry)
    try:
        if as_of:
            frame = repo._fetch(symbol, expiration, date=as_of)
        else:
            frame = repo._fetch(symbol, expiration)
    except Exception:
        frame = pd.DataFrame()
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame()
    # Scoped expiry + empty/mock → refuse unscoped fallback (capture will not save).
    if expiration and (frame.empty or is_mock_frame(frame)):
        reason = chain_expiry_mismatch_reason(frame, expiration)
        if reason:
            st.session_state["ingest_alert"] = reason
        return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        # No expiry requested: cached/live unscoped chain is acceptable.
        frame = _repo_chain(symbol, expiry)
    # Never stamp today's live spot onto a historical EOD chain.
    if as_of is None and live_spot is not None and isinstance(frame, pd.DataFrame) and not frame.empty:
        frame = frame.copy()
        frame["underlyingPrice"] = float(live_spot)
        frame["S"] = float(live_spot)
    st.session_state.snapshot_live_spot = live_spot
    return frame


def _ui_contract_from_widgets(
    fallback_ticker: str,
    fallback_expiry: Any,
    fallback_strike: float | None,
    fallback_option_type: str | None = None,
) -> tuple[str, Any, float | None, str | None]:
    """Read ticker / strike / expiry / type from current widget session values (this render)."""
    symbol = str(st.session_state.get("ticker") or fallback_ticker or "").strip().upper() or DEFAULT_TICKER
    expiry: Any = st.session_state.get("expiry_iso")
    if expiry in (None, ""):
        expiry = st.session_state.get("expiry")
    if expiry in (None, ""):
        expiry = fallback_expiry
    strike: float | None
    if bool(st.session_state.get("manual_strike")):
        strike = _parse_manual_strike(str(st.session_state.get("manual_strike_text") or ""))
        if strike is None:
            try:
                strike = float(fallback_strike) if fallback_strike is not None else None
            except (TypeError, ValueError):
                strike = None
    else:
        raw_strike = st.session_state.get("listed_strike")
        if raw_strike is None or raw_strike == "" or raw_strike == "—":
            try:
                strike = float(fallback_strike) if fallback_strike is not None else None
            except (TypeError, ValueError):
                strike = None
        else:
            try:
                strike = float(raw_strike)
            except (TypeError, ValueError):
                strike = None
    option_type = st.session_state.get("option_type")
    if option_type in (None, ""):
        option_type = fallback_option_type
    return symbol, expiry, strike, str(option_type) if option_type not in (None, "") else None


def _capture_snapshot(
    ticker: str,
    expiry: Any = None,
    strike: float | None = None,
    contract_type: str | None = None,
) -> None:
    symbol = str(ticker or "").strip().upper() or DEFAULT_TICKER
    repo = _get_repo()
    requested = _expiry_to_iso(expiry)
    data = _fresh_chain_for_snapshot(symbol, expiry)
    # Refuse to persist empty/mock/wrong-expiry chains under the UI's requested expiry.
    mismatch = chain_expiry_mismatch_reason(data, requested)
    if mismatch:
        st.session_state.snapshot_error = mismatch
        st.session_state["ingest_alert"] = mismatch
        return
    as_of = _historical_date_iso() or date.today().isoformat()
    metadata = {
        "ticker": symbol,
        "strike": strike,
        "expiry": requested
        if requested
        else (expiry.isoformat() if isinstance(expiry, date) else (str(expiry)[:10] if expiry else None)),
        "contract_type": contract_type,
        "as_of_date": as_of,
    }
    before = len(repo.get_historical_snapshots(symbol))
    repo.save_to_cache(symbol, data, metadata=metadata)
    after = len(repo.get_historical_snapshots(symbol))
    _reset_time_machine_keys(symbol)
    try:
        cached_delta_drift.clear()
        cached_delta_drift_grid.clear()
        cached_snapshot_vol_surface.clear()
    except Exception:
        pass
    if after > before:
        st.session_state.snapshot_saved = True
    else:
        st.session_state.snapshot_skipped = True


def cached_current_price(ticker: str) -> float | None:
    try:
        frame = _repo_chain(ticker)
        for column in ("underlyingPrice", "S", "Price"):
            if column not in frame.columns:
                continue
            spots = pd.to_numeric(frame[column], errors="coerce")
            spots = spots.loc[spots > 0]
            if not spots.empty:
                return float(spots.iloc[0])
        return None
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner="Loading expirations…")
def cached_available_expirations(ticker: str) -> list[str]:
    try:
        return list(get_available_expirations(ticker) or [])
    except Exception:
        return []


def cached_available_strikes(ticker: str, expiry: str) -> list[float]:
    try:
        frame = _repo_chain(ticker, expiry)
        for column in ("strike", "K", "Strike"):
            if column not in frame.columns:
                continue
            strikes = pd.to_numeric(frame[column], errors="coerce").dropna()
            strikes = strikes.loc[strikes > 0]
            if not strikes.empty:
                return sorted({float(value) for value in strikes.tolist()})
        return []
    except Exception:
        return []


def cached_contract_iv(ticker: str, expiry: str, strike: float, option_type: str) -> float | None:
    try:
        frame = _repo_chain(ticker, expiry)
        if frame.empty:
            return None
        k_col = next((c for c in ("strike", "K", "Strike") if c in frame.columns), None)
        iv_col = next((c for c in ("impliedVolatility", "IV", "sigma", "iv") if c in frame.columns), None)
        if k_col is None or iv_col is None:
            return None
        work = frame.copy()
        work["_k"] = pd.to_numeric(work[k_col], errors="coerce")
        work["_iv"] = pd.to_numeric(work[iv_col], errors="coerce")
        work = work.dropna(subset=["_k", "_iv"])
        work = work.loc[work["_iv"] > 0]
        if work.empty:
            return None
        want = str(option_type).strip().lower()
        type_col = next((c for c in ("option_type", "type") if c in work.columns), None)
        if type_col is not None:
            text = work[type_col].astype(str).str.strip().str.lower()
            typed = work.loc[text.str.startswith(want[:1])]
            if not typed.empty:
                work = typed
        idx = (work["_k"] - float(strike)).abs().idxmin()
        return float(work.loc[idx, "_iv"])
    except Exception:
        return None


def _nearest_strike(strikes: list[float], spot: float) -> float:
    return min(strikes, key=lambda strike: abs(strike - spot))


def _parse_manual_strike(raw: str) -> float | None:
    text = raw.strip().replace(",", "")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0 else None


@st.cache_data(show_spinner="Computing Greeks…")
def cached_greek_curve(
    ticker: str,
    strike: float,
    expiry: str,
    prices: tuple[float, ...],
    option_type: str,
    sigma: float,
    model: str,
    v0: float,
    kappa: float,
    theta: float,
    heston_sigma: float,
    rho: float,
) -> pd.DataFrame:
    """Cached Price vs Greeks. Ticker/strike persist independently of model."""
    return generate_greek_curve(
        ticker,
        strike,
        expiry,
        prices,
        r=DEFAULT_RATE,
        sigma=sigma,
        option_type=option_type,
        model=model,
        v0=v0,
        kappa=kappa,
        theta=theta,
        heston_sigma=heston_sigma,
        rho=rho,
        repo=DATA_REPO,
    )


@st.cache_data(show_spinner=False)
def cached_vanna_volga(
    prices: tuple[float, ...],
    strike: float,
    expiry: str,
    sigma: float,
    option_type: str,
    v0: float,
    kappa: float,
    theta: float,
    heston_sigma: float,
    rho: float,
) -> pd.DataFrame:
    spots = np.asarray(prices, dtype="float64")
    time_years = max((date.fromisoformat(expiry) - date.today()).days, 0) / DAYS_PER_YEAR
    kwargs = {
        "q": 0.0,
        "option_type": option_type,
        "v0": v0,
        "kappa": kappa,
        "theta": theta,
        "heston_sigma": heston_sigma,
        "rho": rho,
    }
    return pd.DataFrame(
        {
            "Price": spots,
            "Vanna_BS": calculate_vanna(spots, strike, time_years, DEFAULT_RATE, sigma, MODEL_BLACK_SCHOLES, **kwargs),
            "Vanna_Heston": calculate_vanna(spots, strike, time_years, DEFAULT_RATE, sigma, MODEL_HESTON, **kwargs),
            "Volga_BS": calculate_volga(spots, strike, time_years, DEFAULT_RATE, sigma, MODEL_BLACK_SCHOLES, **kwargs),
            "Volga_Heston": calculate_volga(spots, strike, time_years, DEFAULT_RATE, sigma, MODEL_HESTON, **kwargs),
        }
    )


def _remember_ticker(ticker: str) -> None:
    symbol = ticker.strip().upper()
    if not symbol:
        return
    recent: list[str] = list(st.session_state.recent_tickers)
    if symbol in recent:
        recent.remove(symbol)
    recent.insert(0, symbol)
    st.session_state.recent_tickers = recent[:12]
    st.session_state.recent_pick = symbol


def _on_ticker_commit() -> None:
    symbol = str(st.session_state.get("ticker", "")).strip().upper()
    st.session_state.ticker = symbol
    _remember_ticker(symbol)


def _on_recent_ticker() -> None:
    picked = str(st.session_state.get("recent_pick") or "").strip().upper()
    if not picked:
        return
    st.session_state.ticker = picked
    _remember_ticker(picked)


def _sync_current_price(ticker: str) -> None:
    symbol = ticker.strip().upper()
    if not symbol or st.session_state.get("price_ticker") == symbol:
        return
    try:
        price = cached_current_price(symbol)
    except Exception:
        price = None
    if price is not None and price > 0:
        st.session_state.current_price = float(price)
    st.session_state.price_ticker = symbol


def _format_iv(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np_isfinite(number) or number <= 0:
        return "N/A"
    return f"{number:.2%}"


def np_isfinite(number: float) -> bool:
    return number == number and number not in (float("inf"), float("-inf"))


def _format_greek(name: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np_isfinite(number):
        return "N/A"
    if name == "Gamma":
        return f"{number:.6f}"
    return f"{number:.4f}"


def _spot_greeks(frame: pd.DataFrame, spot: float, use_heston: bool = False) -> dict[str, Any]:
    names = GREEK_COLUMNS
    if frame.empty or "Price" not in frame.columns:
        return {name: None for name in names}
    prices = pd.to_numeric(frame["Price"], errors="coerce")
    valid = prices.notna()
    if not valid.any():
        return {name: None for name in names}
    idx = (prices[valid] - spot).abs().idxmin()
    row = frame.loc[idx]
    out: dict[str, Any] = {}
    for name in names:
        column = f"{name}_Heston" if use_heston and f"{name}_Heston" in frame.columns else name
        out[name] = row[column] if column in frame.columns else None
    return out


def render_greek_chart(
    frame: pd.DataFrame,
    y_column: str,
    container: Any,
    revision: str = "greeks",
    gamma_flip: float | None = None,
) -> None:
    """Plot one Greek vs Price from a caller-supplied DataFrame."""
    if y_column not in frame.columns or "Price" not in frame.columns:
        container.warning(f"Column '{y_column}' is missing from the Greeks frame.")
        return

    plot_df = frame[["Price", y_column]].copy()
    plot_df[y_column] = pd.to_numeric(plot_df[y_column], errors="coerce")
    plot_df["Price"] = pd.to_numeric(plot_df["Price"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Price", y_column])
    heston_col = f"{y_column}_Heston"
    has_heston = heston_col in frame.columns
    if plot_df.empty and not has_heston:
        container.info(f"No valid {y_column} points to plot.")
        return

    yaxis_title = "Theta (per day)" if y_column == "Theta" else y_column
    color = GREEK_COLORS.get(y_column, "#58a6ff")
    traces = []
    if not plot_df.empty:
        traces.append(
            go.Scatter(
                x=plot_df["Price"],
                y=plot_df[y_column],
                mode="lines",
                name="Black-Scholes",
                line={"color": color, "width": 2, "dash": "solid"},
                hovertemplate="Price=%{x:.4f}<br>BS " + y_column + "=%{y:.6f}<extra></extra>",
            )
        )
    if has_heston:
        heston_df = frame[["Price", heston_col]].copy()
        heston_df[heston_col] = pd.to_numeric(heston_df[heston_col], errors="coerce")
        heston_df["Price"] = pd.to_numeric(heston_df["Price"], errors="coerce")
        heston_df = heston_df.dropna(subset=["Price", heston_col])
        if not heston_df.empty:
            traces.append(
                go.Scatter(
                    x=heston_df["Price"],
                    y=heston_df[heston_col],
                    mode="lines",
                    name="Heston",
                    line={"color": color, "width": 2, "dash": "dot"},
                    hovertemplate="Price=%{x:.4f}<br>Heston " + y_column + "=%{y:.6f}<extra></extra>",
                )
            )
    layout_extra: dict[str, Any] = {}
    iv_series = None
    if "IV" in frame.columns:
        iv_plot = frame[["Price", "IV"]].copy()
        iv_plot["IV"] = pd.to_numeric(iv_plot["IV"], errors="coerce")
        iv_plot["Price"] = pd.to_numeric(iv_plot["Price"], errors="coerce")
        iv_plot = iv_plot.dropna(subset=["Price", "IV"])
        if not iv_plot.empty:
            iv_series = iv_plot
            traces.append(
                go.Scatter(
                    x=iv_plot["Price"],
                    y=iv_plot["IV"],
                    mode="lines",
                    name="IV",
                    yaxis="y2",
                    line={"color": GREEK_COLORS["IV"], "width": 1.5, "dash": "dash"},
                    hovertemplate="Price=%{x:.4f}<br>IV=%{y:.2%}<extra></extra>",
                )
            )
            layout_extra["yaxis2"] = {
                "title": "IV",
                "overlaying": "y",
                "side": "right",
                "gridcolor": GRID,
                "zeroline": False,
                "tickformat": ".0%",
            }
    if not traces:
        container.info(f"No valid {y_column} points to plot.")
        return
    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font={"color": TEXT},
        margin={"l": 48, "r": 56 if iv_series is not None else 16, "t": 40, "b": 48},
        height=340,
        title={"text": y_column, "x": 0.0, "xanchor": "left"},
        xaxis={
            "title": "Underlying price",
            "gridcolor": GRID,
            "zeroline": False,
        },
        yaxis={
            "title": yaxis_title,
            "gridcolor": GRID,
            "zeroline": False,
        },
        hovermode="x unified",
        showlegend=True,
        legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
        uirevision=revision,
        **layout_extra,
    )
    if (
        bool(st.session_state.get("show_gamma_flip_line", True))
        and gamma_flip is not None
        and np.isfinite(float(gamma_flip))
        and float(gamma_flip) > 0
    ):
        fig.add_vline(
            x=float(gamma_flip),
            line_width=1.5,
            line_dash="dash",
            line_color="#e3b341",
            annotation_text="Gamma Flip",
            annotation_position="top",
        )
    container.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": True},
        key=f"greek-chart-{y_column}-{revision}",
    )


def _listed_strikes(ticker: str, expiry: date) -> list[float]:
    try:
        listed = list(cached_available_strikes(ticker, expiry.isoformat()))
    except Exception:
        listed = []
    return sorted({float(strike) for strike in listed if float(strike) > 0})


def _active_expiry(ticker: str) -> date:
    listed: list[str] = []
    try:
        listed = list(cached_available_expirations(ticker))
    except Exception:
        listed = []
    if listed:
        current = st.session_state.get("expiry_iso")
        if current not in listed:
            target = date.today() + timedelta(days=30)
            st.session_state.expiry_iso = min(
                listed,
                key=lambda iso: abs((date.fromisoformat(iso) - target).days),
            )
        chosen = st.selectbox("Expiry", options=listed, key="expiry_iso")
        return date.fromisoformat(str(chosen))
    picked = st.date_input("Expiry", key="expiry")
    if isinstance(picked, date):
        return picked
    return date.today() + timedelta(days=30)


def _active_strike(ticker: str, expiry: date, current_price: float) -> float:
    listed = _listed_strikes(ticker, expiry)
    manual = st.checkbox("Manual Strike", key="manual_strike")
    if manual:
        raw = st.text_input(
            "Manual Strike",
            key="manual_strike_text",
            placeholder="Enter strike",
        )
        parsed = _parse_manual_strike(raw)
        if parsed is not None:
            return parsed
        if listed:
            return listed[0]
        return 0.0

    if not listed:
        st.warning("No listed strikes returned for this ticker and expiry.")
        st.selectbox("Select Strike", options=["—"], disabled=True, key="listed_strike_empty")
        return 0.0

    previous = st.session_state.get("listed_strike")
    if previous not in listed:
        anchor = current_price if current_price > 0 else listed[len(listed) // 2]
        st.session_state.listed_strike = _nearest_strike(listed, anchor)
    return float(
        st.selectbox(
            "Select Strike",
            options=listed,
            key="listed_strike",
            format_func=lambda value: f"{value:g}",
        )
    )


def _sidebar_inputs() -> tuple[str, float, date, float, str, bool, Any]:
    with st.sidebar:
        render_global_data_status(_get_repo())
        st.header("Contract")
        ticker = st.text_input(
            "Ticker",
            key="ticker",
            placeholder="Underlying symbol",
            help="Any listed underlying. Defaults to SPY.",
            on_change=_on_ticker_commit,
        ).strip()

        ticker_norm = ticker.strip().upper() or DEFAULT_TICKER
        recent = [item for item in st.session_state.recent_tickers if str(item).strip()]
        if ticker_norm not in recent:
            _remember_ticker(ticker_norm)
            recent = [item for item in st.session_state.recent_tickers if str(item).strip()]
        if st.session_state.get("recent_pick") not in recent:
            st.session_state.recent_pick = ticker_norm if ticker_norm in recent else recent[0]
        st.selectbox(
            "Recent Tickers",
            options=recent,
            key="recent_pick",
            on_change=_on_recent_ticker,
        )
        ticker = str(st.session_state.ticker).strip()
        ticker_norm = ticker.strip().upper() or DEFAULT_TICKER
        _sync_current_price(ticker_norm)

        expiry = _active_expiry(ticker_norm)
        current_price = st.number_input(
            "Current Price",
            min_value=0.0,
            step=0.01,
            format="%.4f",
            key="current_price",
        )
        option_type = st.selectbox(
            "Type",
            options=("call", "put"),
            key="option_type",
        )
        strike = _active_strike(ticker_norm, expiry, float(current_price))

        st.radio(
            "Model",
            options=("Black-Scholes", "Heston"),
            key="pricing_model",
            horizontal=True,
        )
        st.caption("Model and Heston parameters stay in session until you change them.")
        st.number_input("kappa", min_value=0.01, max_value=20.0, step=0.05, format="%.4f", key="heston_kappa")
        st.number_input("theta", min_value=0.0001, max_value=1.0, step=0.005, format="%.4f", key="heston_theta")
        st.number_input("sigma", min_value=0.01, max_value=2.0, step=0.01, format="%.4f", key="heston_sigma")
        st.number_input("rho", min_value=-0.999, max_value=0.999, step=0.01, format="%.3f", key="heston_rho")
        st.number_input("v0", min_value=0.0001, max_value=1.0, step=0.005, format="%.4f", key="heston_v0")

        st.toggle("Second-Order Greeks", key="show_second_order")
        st.sidebar.checkbox("Show Gamma Flip Line", value=True, key="show_gamma_flip_line")

        refresh = st.button("Refresh", use_container_width=True, type="primary")
        if refresh:
            _clear_persisted_results()
            cached_greek_curve.clear()
            cached_available_expirations.clear()
            cached_gamma_surface.clear()
            cached_greek_surface.clear()
            cached_delta_term_structure.clear()
            cached_vanna_volga.clear()
            cached_vol_surface.clear()
            cached_pro_surface.clear()
            cached_pro_curve.clear()
            cached_market_sentiment.clear()
            cached_gex_outlook.clear()
            cached_vol_risk_outlook.clear()
            cached_event_outlook.clear()
            st.session_state.price_ticker = ""
            st.session_state.iv_fingerprint = None
            st.session_state.persisted_iv = None
            st.rerun()

        historical = st.date_input(
            "Fetch Historical Data",
            value=None,
            max_value=date.today(),
            key="historical_date",
            help="Pick a past date to load that day's end-of-day chain instead of live data. Clear to return to live.",
        )
        if isinstance(historical, date):
            st.caption(f"Historical mode: chain as of {historical.isoformat()}")

        if st.button("Capture Snapshot", width="stretch"):
            ui_ticker, ui_expiry, ui_strike, ui_type = _ui_contract_from_widgets(
                ticker_norm, expiry, strike, str(option_type)
            )
            _capture_snapshot(ui_ticker, ui_expiry, ui_strike, contract_type=ui_type)
            st.rerun()
        if st.session_state.pop("snapshot_saved", False):
            st.success("Snapshot saved!")
        if st.session_state.pop("snapshot_skipped", False):
            st.warning("Snapshot identical to previous—not saving.")
        snap_err = st.session_state.pop("snapshot_error", None)
        if snap_err:
            st.error(str(snap_err))

        download_slot = st.empty()

        latency = st.session_state.last_latency_ms
        st.metric(
            "Calc latency",
            "—" if latency is None else f"{latency:.1f} ms",
            help="Wall time of the last Greek curve call (cache hits are typically sub-ms).",
        )
        st.caption(
            f"Model: {st.session_state.get('pricing_model', 'Black-Scholes')} · MarketData · r={DEFAULT_RATE:.1%} · q=0 · 365-day"
        )
    return ticker, strike, expiry, float(current_price), str(option_type), refresh, download_slot


def greeks_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize Price, Delta, Gamma, Theta from a generate_greek_curve DataFrame."""
    present = [col for col in list(CSV_COLUMNS) + [f"{n}_Heston" for n in GREEK_COLUMNS] if col in frame.columns]
    return frame.loc[:, present].to_csv(index=False).encode("utf-8")


def render_sidebar_download(
    slot: Any,
    frame: pd.DataFrame,
    ticker: str,
    strike: float,
    expiry: date,
) -> None:
    stamp = expiry.isoformat() if isinstance(expiry, date) else str(expiry)
    file_name = f"{ticker}_{strike:g}_{stamp}_greeks.csv"
    slot.download_button(
        "Download CSV",
        data=greeks_csv_bytes(frame),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
        key="download_greeks_csv",
    )


def _contract_iv(ticker: str, expiry: date, strike: float, option_type: str) -> float | None:
    key = (ticker, expiry.isoformat(), round(float(strike), 8), str(option_type))
    if st.session_state.get("iv_fingerprint") == key:
        return st.session_state.get("persisted_iv")
    try:
        iv = cached_contract_iv(ticker, expiry.isoformat(), float(strike), option_type)
    except Exception:
        iv = None
    st.session_state.iv_fingerprint = key
    st.session_state.persisted_iv = iv
    return iv


def _load_curve(
    ticker: str,
    strike: float,
    expiry: date,
    current_price: float,
    option_type: str,
    sigma: float,
    model: str,
    v0: float,
    kappa: float,
    theta: float,
    heston_sigma: float,
    rho: float,
) -> pd.DataFrame | None:
    fingerprint = _sim_fingerprint(
        ticker, strike, expiry, option_type, current_price, sigma, model, v0, kappa, theta, heston_sigma, rho
    )
    existing = st.session_state.get("greeks_frame")
    heston_ok = True
    if str(model) == MODEL_HESTON:
        heston_ok = (
            isinstance(existing, pd.DataFrame)
            and "Delta_Heston" in existing.columns
            and pd.to_numeric(existing["Delta_Heston"], errors="coerce").notna().any()
        )
    if (
        fingerprint == st.session_state.get("sim_fingerprint")
        and isinstance(existing, pd.DataFrame)
        and not existing.empty
        and heston_ok
    ):
        st.session_state.last_ok = True
        return existing

    prices = _price_range(current_price)
    expiry_iso = expiry.isoformat()
    started = time.perf_counter()
    try:
        frame = cached_greek_curve(
            ticker,
            strike,
            expiry_iso,
            prices,
            option_type,
            sigma,
            model,
            v0,
            kappa,
            theta,
            heston_sigma,
            rho,
        )
    except Exception:
        st.session_state.last_latency_ms = (time.perf_counter() - started) * 1000.0
        st.session_state.last_ok = False
        st.session_state.greeks_frame = None
        st.session_state.sim_fingerprint = None
        st.error(
            "Could not generate Greek charts. Check ticker, strike, expiry, and current price, then try Refresh."
        )
        return None
    st.session_state.last_latency_ms = (time.perf_counter() - started) * 1000.0
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        st.session_state.last_ok = False
        st.session_state.greeks_frame = None
        st.session_state.sim_fingerprint = None
        st.warning("The engine returned an empty Greeks table for these inputs.")
        return None
    st.session_state.last_ok = True
    st.session_state.greeks_frame = frame
    st.session_state.sim_fingerprint = fingerprint
    return frame


def render_vanna_volga_chart(
    frame: pd.DataFrame,
    strike: float,
    expiry: date,
    sigma: float,
    option_type: str,
    heston_v0: float,
    heston_kappa: float,
    heston_theta: float,
    heston_sigma: float,
    heston_rho: float,
    revision: str = "vanna-volga",
) -> None:
    if "Price" not in frame.columns:
        st.warning("Price column missing.")
        return
    price_tuple = tuple(
        float(p) for p in pd.to_numeric(frame["Price"], errors="coerce").to_numpy(dtype="float64") if np.isfinite(p)
    )
    fingerprint = (
        price_tuple,
        round(float(strike), 8),
        expiry.isoformat(),
        round(float(sigma), 8),
        str(option_type),
        round(float(heston_v0), 8),
        round(float(heston_kappa), 8),
        round(float(heston_theta), 8),
        round(float(heston_sigma), 8),
        round(float(heston_rho), 8),
    )
    vv = st.session_state.get("vanna_volga_frame")
    if fingerprint != st.session_state.get("vanna_volga_fp") or not isinstance(vv, pd.DataFrame) or vv.empty:
        vv = cached_vanna_volga(
            price_tuple,
            float(strike),
            expiry.isoformat(),
            float(sigma),
            option_type,
            float(heston_v0),
            float(heston_kappa),
            float(heston_theta),
            float(heston_sigma),
            float(heston_rho),
        )
        st.session_state.vanna_volga_frame = vv
        st.session_state.vanna_volga_fp = fingerprint
    prices = pd.to_numeric(vv["Price"], errors="coerce")
    series_map = {
        "Vanna (Black-Scholes)": ("Vanna_BS", GREEK_COLORS["Vanna"], "solid"),
        "Vanna (Heston)": ("Vanna_Heston", GREEK_COLORS["Vanna"], "dot"),
        "Volga (Black-Scholes)": ("Volga_BS", GREEK_COLORS["Volga"], "solid"),
        "Volga (Heston)": ("Volga_Heston", GREEK_COLORS["Volga"], "dot"),
    }
    traces = []
    for name, (column, color, dash) in series_map.items():
        series = pd.to_numeric(vv[column], errors="coerce")
        valid = prices.notna() & series.notna()
        if not valid.any():
            continue
        traces.append(
            go.Scatter(
                x=prices.loc[valid],
                y=series.loc[valid],
                mode="lines",
                name=name,
                line={"color": color, "width": 2, "dash": dash},
                hovertemplate="Price=%{x:.4f}<br>" + name + "=%{y:.6f}<extra></extra>",
            )
        )
    if not traces:
        st.info("No valid Vanna/Volga points to plot.")
        return
    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font={"color": TEXT},
        height=380,
        title={"text": "Vanna and Volga vs Price", "x": 0.0, "xanchor": "left"},
        xaxis={"title": "Price", "gridcolor": GRID, "zeroline": False},
        yaxis={"title": "Second-order Greek", "gridcolor": GRID, "zeroline": False},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
        margin={"l": 48, "r": 16, "t": 48, "b": 48},
        uirevision=revision,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True}, key="vanna-volga-chart")


def render_gamma_theta_ratio_chart(frame: pd.DataFrame) -> None:
    if "Price" not in frame.columns:
        st.warning("Price column missing.")
        return
    plot_df = frame.copy()
    plot_df["Price"] = pd.to_numeric(plot_df["Price"], errors="coerce")
    ratio = pd.Series(
        calculate_gamma_theta_ratio(
            pd.to_numeric(plot_df.get("Gamma"), errors="coerce"),
            pd.to_numeric(plot_df.get("Theta"), errors="coerce"),
        ),
        index=plot_df.index,
    )
    valid = plot_df["Price"].notna() & ratio.notna()
    if not valid.any():
        st.info("No valid Gamma/Theta ratio points to plot.")
        return
    x = plot_df.loc[valid, "Price"].to_numpy(dtype=float)
    y = ratio.loc[valid].to_numpy(dtype=float)
    exceeds = bool((y > RATIO_THRESHOLD).any())
    ratio_color = "#f85149" if exceeds else "#58a6ff"
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="Gamma/Theta Ratio",
                line={"color": ratio_color, "width": 2.5},
                hovertemplate="Price=%{x:.4f}<br>Gamma/Theta Ratio=%{y:.4f}<extra></extra>",
            )
        ]
    )
    fig.add_hline(
        y=RATIO_THRESHOLD,
        line_dash="dash",
        line_color="#f85149" if exceeds else "#8b949e",
        line_width=2,
        annotation_text="0.5",
        annotation_position="top left",
        annotation_font_color="#f85149" if exceeds else "#8b949e",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font={"color": TEXT},
        height=400,
        title={"text": "Gamma/Theta Ratio vs Price", "x": 0.0, "xanchor": "left"},
        xaxis={"title": "Price", "gridcolor": GRID, "zeroline": False},
        yaxis={"title": "Gamma/Theta Ratio", "gridcolor": GRID, "zeroline": True},
        hovermode="x unified",
        showlegend=True,
        legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
        margin={"l": 48, "r": 16, "t": 48, "b": 48},
        uirevision="gamma-theta-ratio",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True}, key="gamma-theta-ratio-chart")


def _price_greek_chart(
    frame: pd.DataFrame,
    y_column: str,
    title: str,
    y_title: str,
) -> None:
    if "Price" not in frame.columns:
        st.warning("Price column missing.")
        return
    prices = pd.to_numeric(frame["Price"], errors="coerce")
    traces = []
    if y_column in frame.columns:
        series = pd.to_numeric(frame[y_column], errors="coerce")
        valid = prices.notna() & series.notna()
        if valid.any():
            traces.append(
                go.Scatter(
                    x=prices.loc[valid],
                    y=series.loc[valid],
                    mode="lines",
                    name=y_column,
                    line={"color": GREEK_COLORS.get(y_column, "#58a6ff"), "width": 2.5},
                    hovertemplate="Price=%{x:.4f}<br>" + y_column + "=%{y:.6f}<extra></extra>",
                )
            )
    heston_col = f"{y_column}_Heston"
    if heston_col in frame.columns:
        series = pd.to_numeric(frame[heston_col], errors="coerce")
        valid = prices.notna() & series.notna()
        if valid.any():
            traces.append(
                go.Scatter(
                    x=prices.loc[valid],
                    y=series.loc[valid],
                    mode="lines",
                    name=f"{y_column} (Heston)",
                    line={"color": GREEK_COLORS.get(y_column, "#58a6ff"), "width": 2, "dash": "dot"},
                    hovertemplate="Price=%{x:.4f}<br>Heston " + y_column + "=%{y:.6f}<extra></extra>",
                )
            )
    if not traces:
        st.info(f"No valid {y_column} points to plot.")
        return
    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font={"color": TEXT},
        height=380,
        title={"text": title, "x": 0.0, "xanchor": "left"},
        xaxis={"title": "Price", "gridcolor": GRID, "zeroline": False},
        yaxis={"title": y_title, "gridcolor": GRID, "zeroline": True},
        hovermode="x unified",
        showlegend=True,
        legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
        margin={"l": 48, "r": 16, "t": 48, "b": 48},
        uirevision=y_column,
    )
    model_tag = "heston" if f"{y_column}_Heston" in frame.columns else "bs"
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": True},
        key=f"price-greek-{y_column}-{model_tag}",
    )


def render_speed_chart(frame: pd.DataFrame) -> None:
    _price_greek_chart(frame, "Speed", "Risk Acceleration (Speed vs Price)", "Speed (∂Γ/∂S)")


def render_color_chart(frame: pd.DataFrame) -> None:
    _price_greek_chart(frame, "Color", "Gamma Decay (Color vs Price)", "Color (∂Γ/∂t per day)")


@st.cache_data(show_spinner="Building gamma surface…")
def cached_gamma_surface(
    ticker: str,
    strike: float,
    expiry_range: tuple[float, ...],
    prices: tuple[float, ...],
    sigma: float,
    option_type: str,
    model: str,
    v0: float,
    kappa: float,
    theta: float,
    heston_sigma: float,
    rho: float,
) -> pd.DataFrame:
    return generate_3d_gamma_surface(
        ticker,
        strike,
        expiry_range,
        price_range=prices if prices else None,
        r=DEFAULT_RATE,
        sigma=sigma,
        option_type=option_type,
        model=model,
        v0=v0,
        kappa=kappa,
        theta=theta,
        heston_sigma=heston_sigma,
        rho=rho,
    )


@st.cache_data(show_spinner="Building advanced Greek surface…")
def cached_greek_surface(
    ticker: str,
    strike: float,
    expiry_range: tuple[float, ...],
    prices: tuple[float, ...],
    sigma: float,
    option_type: str,
    model: str,
    v0: float,
    kappa: float,
    theta: float,
    heston_sigma: float,
    rho: float,
    greek: str,
) -> pd.DataFrame:
    return generate_3d_greek_surface(
        ticker,
        strike,
        expiry_range,
        greek=greek,
        price_range=prices if prices else None,
        r=DEFAULT_RATE,
        sigma=sigma,
        option_type=option_type,
        model=model,
        v0=v0,
        kappa=kappa,
        theta=theta,
        heston_sigma=heston_sigma,
        rho=rho,
    )


@st.cache_data(show_spinner="Building pro metric surface…")
def cached_pro_surface(
    ticker: str,
    greek_name: str,
    strike: float,
    expiry_range: tuple[float, ...],
    prices: tuple[float, ...],
    sigma: float,
    option_type: str,
    model: str,
    v0: float,
    kappa: float,
    theta: float,
    heston_sigma: float,
    rho: float,
    apply_smoothing: bool = False,
) -> pd.DataFrame:
    return generate_pro_surface_data(
        ticker,
        greek_name,
        strike=strike,
        expiry_range=expiry_range,
        price_range=prices if prices else None,
        r=DEFAULT_RATE,
        sigma=sigma,
        option_type=option_type,
        model=model,
        v0=v0,
        kappa=kappa,
        theta=theta,
        heston_sigma=heston_sigma,
        rho=rho,
        apply_smoothing=bool(apply_smoothing),
    )


@st.cache_data(show_spinner=False)
def cached_pro_curve(
    prices: tuple[float, ...],
    greek_name: str,
    strike: float,
    expiry: str,
    sigma: float,
    option_type: str,
    model: str,
    v0: float,
    kappa: float,
    theta: float,
    heston_sigma: float,
    rho: float,
) -> pd.DataFrame:
    spots = np.asarray(prices, dtype="float64")
    time_years = max((date.fromisoformat(expiry) - date.today()).days, 1) / DAYS_PER_YEAR
    kwargs = {
        "q": 0.0,
        "option_type": option_type,
        "v0": v0,
        "kappa": kappa,
        "theta": theta,
        "heston_sigma": heston_sigma,
        "rho": rho,
    }
    calculators = {
        "Vanna": calculate_vanna,
        "Vomma": calculate_vomma,
        "Zomma": calculate_zomma,
        "Veta": calculate_veta,
        "Ultima": calculate_ultima,
    }
    fn = calculators.get(str(greek_name), calculate_vanna)
    values = fn(spots, strike, time_years, DEFAULT_RATE, sigma, model, **kwargs)
    return pd.DataFrame({"Price": spots, "Value": np.asarray(values, dtype="float64")})


@st.cache_data(show_spinner="Building delta term structure…")
def cached_delta_term_structure(
    ticker: str,
    strike: float,
    spot: float,
    expiry_range: tuple[float, ...],
    sigma: float,
    option_type: str,
    model: str,
    v0: float,
    kappa: float,
    theta: float,
    heston_sigma: float,
    rho: float,
) -> pd.DataFrame:
    return generate_delta_term_structure(
        ticker,
        strike,
        spot,
        expiry_range,
        r=DEFAULT_RATE,
        sigma=sigma,
        option_type=option_type,
        model=model,
        v0=v0,
        kappa=kappa,
        theta=theta,
        heston_sigma=heston_sigma,
        rho=rho,
    )


@st.cache_data(ttl=300, show_spinner="Building implied volatility surface…")
def cached_vol_surface(ticker: str) -> pd.DataFrame | str:
    return generate_vol_surface_data(ticker)


@st.cache_data(ttl=120, show_spinner="Calculating delta drift…")
def cached_delta_drift(ticker: str, ts_a: str, ts_b: str) -> pd.DataFrame:
    return DATA_REPO.calculate_delta_drift(ticker, ts_a, ts_b)


@st.cache_data(ttl=120, show_spinner="Calculating delta drift grid…")
def cached_delta_drift_grid(
    ticker: str, ts_a: str, ts_b: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    return DATA_REPO.calculate_delta_drift_grid(ticker, ts_a, ts_b)


@st.cache_data(ttl=120, show_spinner="Loading snapshot volatility surface…")
def cached_snapshot_vol_surface(ticker: str, timestamp: str) -> pd.DataFrame | str:
    return DATA_REPO.snapshot_vol_surface(ticker, timestamp)


def cached_analyst_chain(ticker: str) -> pd.DataFrame:
    try:
        expiries = list(get_available_expirations(ticker) or [])
    except Exception:
        expiries = []
    expiry = expiries[0] if expiries else None
    try:
        return _repo_chain(ticker, expiry)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner="Scoring market sentiment…")
def cached_market_sentiment(ticker: str) -> dict[str, str]:
    return analyze_market_sentiment(cached_analyst_chain(ticker))


@st.cache_data(ttl=120, show_spinner="Scoring dealer GEX…")
def cached_gex_outlook(ticker: str) -> dict[str, Any]:
    return analyze_gex_outlook(cached_analyst_chain(ticker))


@st.cache_data(ttl=120, show_spinner="Scoring volatility risk…")
def cached_vol_risk_outlook(ticker: str) -> dict[str, Any]:
    return analyze_volatility_risk_outlook(cached_analyst_chain(ticker))


@st.cache_data(ttl=120, show_spinner="Scoring event outlook…")
def cached_event_outlook(ticker: str, selected_expiry: str = "") -> dict[str, Any]:
    return analyze_event_outlook(
        cached_analyst_chain(ticker),
        ticker=ticker,
        selected_expiry=selected_expiry or None,
    )


@st.cache_data(ttl=120, show_spinner="Scoring portfolio risk…")
def cached_portfolio_risk(positions: pd.DataFrame) -> dict[str, Any]:
    return calculate_portfolio_risk(positions)


def _format_iv_metric(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return f"{number * 100.0:.1f}%"


def render_delta_term_structure(frame: pd.DataFrame) -> None:
    required = {"DaysToExpiry", "Delta"}
    if frame.empty or not required.issubset(frame.columns):
        st.info("No delta term-structure data to plot.")
        return
    plot_df = frame.copy()
    plot_df["DaysToExpiry"] = pd.to_numeric(plot_df["DaysToExpiry"], errors="coerce")
    plot_df["Delta"] = pd.to_numeric(plot_df["Delta"], errors="coerce")
    valid = plot_df["DaysToExpiry"].notna() & plot_df["Delta"].notna()
    if not valid.any():
        st.info("No valid Delta vs DTE points to plot.")
        return
    traces = [
        go.Scatter(
            x=plot_df.loc[valid, "DaysToExpiry"],
            y=plot_df.loc[valid, "Delta"],
            mode="lines",
            name="Delta",
            line={"color": GREEK_COLORS["Delta"], "width": 2.5},
            hovertemplate="DTE=%{x:.1f}<br>Delta=%{y:.6f}<extra></extra>",
        )
    ]
    layout_extra: dict[str, Any] = {}
    if "Charm" in plot_df.columns:
        charm = pd.to_numeric(plot_df["Charm"], errors="coerce")
        charm_ok = valid & charm.notna()
        if charm_ok.any():
            traces.append(
                go.Scatter(
                    x=plot_df.loc[charm_ok, "DaysToExpiry"],
                    y=charm.loc[charm_ok],
                    mode="lines",
                    name="Charm (Δ per day)",
                    yaxis="y2",
                    line={"color": GREEK_COLORS["Charm"], "width": 2, "dash": "dot"},
                    hovertemplate="DTE=%{x:.1f}<br>Charm=%{y:.6f}<extra></extra>",
                )
            )
            layout_extra["yaxis2"] = {
                "title": "Charm (per day)",
                "overlaying": "y",
                "side": "right",
                "gridcolor": GRID,
                "zeroline": False,
            }
    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font={"color": TEXT},
        height=420,
        title={"text": "Delta vs Days to Expiry", "x": 0.0, "xanchor": "left"},
        xaxis={"title": "Days to Expiry", "gridcolor": GRID, "zeroline": False, "autorange": "reversed"},
        yaxis={"title": "Delta", "gridcolor": GRID, "zeroline": True},
        hovermode="x unified",
        showlegend=True,
        legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
        margin={"l": 48, "r": 64, "t": 48, "b": 48},
        uirevision="delta-term-structure",
        **layout_extra,
    )
    st.caption("As DTE falls (right to left), Delta leaks toward 0 or ±1. Charm is that daily leak.")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True}, key="delta-term-structure-chart")


def build_integrated_risk_figure(surface: go.Surface) -> go.Figure | None:
    """Build the Total Risk Profile figure, or ``None`` when z is empty."""
    raw = getattr(surface, "z", None)
    z_data = reshape_to_surface(raw if raw is not None else [])
    if z_data.ndim != 2 or z_data.size == 0 or not np.any(np.isfinite(z_data)):
        return None
    x_raw = getattr(surface, "x", None)
    y_raw = getattr(surface, "y", None)
    z2d, x_ok, y_ok, warning = prepare_plotly_surface_xyz(z_data, x_raw, y_raw)
    if z2d.ndim != 2 or z2d.size == 0 or not np.any(np.isfinite(z2d)):
        return None
    if warning:
        st.warning(warning)
    finite = z2d[np.isfinite(z2d)]
    z_lo = float(np.min(finite))
    z_hi = float(np.max(finite))
    if z_hi - z_lo < 1e-6:
        z_lo, z_hi = z_lo - 0.5, z_hi + 0.5
    else:
        pad = 0.08 * (z_hi - z_lo)
        z_lo, z_hi = z_lo - pad, z_hi + pad
    fig = go.Figure(
        data=[
            _plotly_surface(
                z2d,
                x_ok,
                y_ok,
                colorscale="Viridis",
                colorbar={"title": "Total Risk"},
                opacity=1.0,
                hovertemplate="Price=%{x:.2f}<br>DTE=%{y:.1f}<br>Risk=%{z:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT},
        height=560,
        title={"text": "Total Risk Profile", "x": 0.0, "xanchor": "left"},
        scene={
            "xaxis": {"title": "Price"},
            "yaxis": {"title": "Days to Expiry"},
            "zaxis": {"title": "Risk (norm)", "range": [z_lo, z_hi]},
            "bgcolor": PANEL_BG,
            "dragmode": "orbit",
            "camera": {"eye": {"x": 1.6, "y": 1.6, "z": 1.15}},
        },
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        uirevision="integrated-risk-surface-v2",
    )
    return fig


def _figure_z_span(fig: go.Figure, fallback: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
    """Visible z-span for Ghost Columns; prefers the scene axis range."""
    try:
        zrange = fig.layout.scene.zaxis.range
        if zrange is not None and len(zrange) >= 2:
            lo, hi = float(zrange[0]), float(zrange[1])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                return lo, hi
    except Exception:
        pass
    return fallback


def overlay_anomaly_ghost_columns(fig: go.Figure, diagnosis: Mapping[str, Any] | None) -> go.Figure:
    """Append semi-transparent Ghost Column Mesh3d traces; existing traces unchanged."""
    if fig is None or not diagnosis:
        return fig
    anomalies = diagnosis.get("anomalies") or []
    if not anomalies:
        return fig
    try:
        dx = float(diagnosis.get("column_dx") or 1.0)
        dy = float(diagnosis.get("column_dy") or 1.0)
        z_lo, z_hi = _figure_z_span(fig, (float(diagnosis.get("z_lo") or 0.0), float(diagnosis.get("z_hi") or 1.0)))
        meshes = build_anomaly_ghost_column_meshes(anomalies, dx=dx, dy=dy, z_lo=z_lo, z_hi=z_hi)
    except Exception:
        return fig
    for mesh in meshes:
        label = mesh.get("label")
        if label == ANOMALY_LABEL_GAMMA_WALL:
            color = "#7ec8e3"
        elif label == ANOMALY_LABEL_REGIME_MISMATCH:
            color = "#e8a87c"
        else:
            color = "#9aa4b2"
        fig.add_trace(
            go.Mesh3d(
                x=mesh["x"],
                y=mesh["y"],
                z=mesh["z"],
                i=mesh["i"],
                j=mesh["j"],
                k=mesh["k"],
                opacity=0.28,
                color=color,
                name=str(label or "Anomaly"),
                hovertext=str(mesh.get("message") or ""),
                hoverinfo="text",
                showlegend=False,
                flatshading=True,
            )
        )
    return fig


def render_surface_anomaly_messages(diagnosis: Mapping[str, Any] | None) -> None:
    """Print the exact anomaly lines (and labels) for the diagnostic overlay."""
    if not diagnosis:
        return
    for item in diagnosis.get("anomalies") or []:
        if not isinstance(item, Mapping):
            continue
        message = item.get("message")
        if message:
            st.text(str(message))
        label = item.get("label")
        if label:
            st.text(str(label))


@st.cache_data(show_spinner=False)
def cached_diagnose_surface_anomaly(
    ticker: str,
    strike_range: tuple[float, ...],
    expiry_range: tuple[float, ...],
) -> dict[str, Any]:
    return diagnose_surface_anomaly(str(ticker), strike_range, expiry_range)


def render_integrated_risk_surface(
    surface: go.Surface,
    *,
    diagnosis: Mapping[str, Any] | None = None,
) -> None:
    """Plot a single Total Risk Profile ``go.Surface`` (Integrated Risk View)."""
    try:
        fig = build_integrated_risk_figure(surface)
        if fig is None:
            st.info("No integrated risk surface data to plot.")
            return
        overlay_anomaly_ghost_columns(fig, diagnosis)
        st.caption(
            "Z and color = mean of enabled layers (Volatility, Gamma/GEX, |Delta|), "
            "Viridis. Constant IV falls back to Gamma so the mesh has relief."
        )
        st.plotly_chart(
            fig,
            width="stretch",
            config={"displayModeBar": True, "scrollZoom": True},
            key="integrated-risk-surface-v2",
        )
        render_surface_anomaly_messages(diagnosis)
    except Exception:
        st.error("Surface rendering failed: Invalid data shape.")


def _build_integrated_risk_frame(
    ticker: str,
    strike: float,
    expiry_range: tuple[float, ...],
    prices: tuple[float, ...],
    sigma: float,
    option_type: str,
    model: str,
    v0: float,
    kappa: float,
    theta: float,
    heston_sigma: float,
    rho: float,
) -> pd.DataFrame:
    """Merge Delta / Gamma / Vega grids and broadcast Volatility for the risk surface."""
    empty_cols = ["Price", "DaysToExpiry", "Delta", "Gamma", "Vega", "Volatility"]

    def _greek_grid(greek: str) -> pd.DataFrame:
        return generate_3d_greek_surface(
            ticker,
            strike,
            expiry_range,
            greek=greek,
            price_range=prices if prices else None,
            r=DEFAULT_RATE,
            sigma=sigma,
            option_type=option_type,
            model=model,
            v0=v0,
            kappa=kappa,
            theta=theta,
            heston_sigma=heston_sigma,
            rho=rho,
        ).copy()

    delta = _greek_grid("Delta")
    gamma = _greek_grid("Gamma")
    vega = _greek_grid("Vega")
    if delta.empty or gamma.empty or vega.empty:
        return pd.DataFrame(columns=empty_cols)
    delta = delta.rename(columns={"Value": "Delta"})
    gamma = gamma.rename(columns={"Value": "Gamma"})
    vega = vega.rename(columns={"Value": "Vega"})
    merged = (
        delta.merge(gamma, on=["Price", "DaysToExpiry"], how="inner")
        .merge(vega, on=["Price", "DaysToExpiry"], how="inner")
        .copy()
    )
    # Prefer per-node IV from the live vol surface when strike/DTE align; else sigma.
    vol = np.full(len(merged.index), float(sigma), dtype=np.float64)
    try:
        vol_frame = generate_vol_surface_data(ticker)
        if isinstance(vol_frame, pd.DataFrame) and not vol_frame.empty:
            vf = vol_frame.copy()
            x_name = "Price" if "Price" in vf.columns else ("Strike" if "Strike" in vf.columns else None)
            if x_name and {"DaysToExpiry", "IV"}.issubset(vf.columns):
                vf[x_name] = pd.to_numeric(vf[x_name], errors="coerce")
                vf["DaysToExpiry"] = pd.to_numeric(vf["DaysToExpiry"], errors="coerce")
                vf["IV"] = pd.to_numeric(vf["IV"], errors="coerce")
                vf = vf.replace([np.inf, -np.inf], np.nan).dropna()
                if not vf.empty:
                    pts = vf.loc[:, [x_name, "DaysToExpiry", "IV"]].to_numpy(dtype=np.float64)
                    for i, (px, dte) in enumerate(
                        zip(
                            merged["Price"].to_numpy(dtype=np.float64),
                            merged["DaysToExpiry"].to_numpy(dtype=np.float64),
                        )
                    ):
                        dist = (pts[:, 0] - px) ** 2 + (pts[:, 1] - dte) ** 2
                        j = int(np.argmin(dist))
                        iv = float(pts[j, 2])
                        if np.isfinite(iv) and iv > 0:
                            vol[i] = iv
    except Exception:
        pass
    merged["Volatility"] = np.array(vol, dtype=np.float64, copy=True)
    return merged.copy()


@st.cache_data(show_spinner="Building integrated risk surface…")
def cached_integrated_risk_frame(
    ticker: str,
    strike: float,
    expiry_range: tuple[float, ...],
    prices: tuple[float, ...],
    sigma: float,
    option_type: str,
    model: str,
    v0: float,
    kappa: float,
    theta: float,
    heston_sigma: float,
    rho: float,
) -> pd.DataFrame:
    return _build_integrated_risk_frame(
        ticker,
        strike,
        expiry_range,
        prices,
        sigma,
        option_type,
        model,
        v0,
        kappa,
        theta,
        heston_sigma,
        rho,
    )


def render_gamma_surface(frame: pd.DataFrame) -> None:
    required = {"Price", "DaysToExpiry", "Gamma"}
    if frame.empty or not required.issubset(frame.columns):
        st.info("No gamma surface data to plot.")
        return
    pivot = frame.pivot_table(index="DaysToExpiry", columns="Price", values="Gamma", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    fig = go.Figure(
        data=[
            _plotly_surface(
                pivot.to_numpy(dtype=float),
                pivot.columns.to_numpy(dtype=float),
                pivot.index.to_numpy(dtype=float),
                colorscale="Viridis",
                colorbar={"title": "Gamma"},
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT},
        height=560,
        title={"text": "Gamma surface", "x": 0.0, "xanchor": "left"},
        scene={
            "xaxis_title": "Price",
            "yaxis_title": "Days to Expiry",
            "zaxis_title": "Gamma",
            "bgcolor": PANEL_BG,
        },
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        uirevision="gamma-surface",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": True, "scrollZoom": True},
        key="gamma-surface-chart",
    )


def render_pro_surface(frame: pd.DataFrame, greek_name: str, *, apply_smoothing: bool = False) -> None:
    required = {"Price", "DaysToExpiry", "Value"}
    if frame.empty or not required.issubset(frame.columns):
        st.info(f"No {greek_name} surface data to plot.")
        return
    plot = frame.loc[:, ["Price", "DaysToExpiry", "Value"]].copy()
    plot["Price"] = pd.to_numeric(plot["Price"], errors="coerce")
    plot["DaysToExpiry"] = pd.to_numeric(plot["DaysToExpiry"], errors="coerce")
    plot["Value"] = pd.to_numeric(plot["Value"], errors="coerce")
    plot = plot.replace([np.inf, -np.inf], np.nan).dropna(subset=["Price", "DaysToExpiry"])
    if plot.empty or not plot["Value"].notna().any():
        st.info(f"No {greek_name} surface data to plot.")
        return
    pivot = plot.pivot_table(index="DaysToExpiry", columns="Price", values="Value", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    z = np.array(pivot.to_numpy(dtype=float), copy=True)
    z[~np.isfinite(z)] = np.nan
    fig = go.Figure(
        data=[
            _plotly_surface(
                z,
                pivot.columns.to_numpy(dtype=float),
                pivot.index.to_numpy(dtype=float),
                colorscale="Viridis",
                connectgaps=False,
                colorbar={"title": greek_name},
                hovertemplate="Price=%{x:.2f}<br>DTE=%{y:.1f}<br>" + greek_name + "=%{z:.6f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT},
        height=560,
        title={"text": f"{greek_name} surface", "x": 0.0, "xanchor": "left"},
        scene={
            "xaxis_title": "Price",
            "yaxis_title": "Days to Expiry",
            "zaxis_title": greek_name,
            "bgcolor": PANEL_BG,
            "dragmode": "orbit",
        },
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        uirevision=f"pro-surface-{greek_name}-{int(bool(apply_smoothing))}",
    )
    st.plotly_chart(
        fig,
        width="stretch",
        height=560,
        theme=None,
        config={"displayModeBar": True, "scrollZoom": True},
        key=f"pro-surface-chart-{greek_name}-{int(bool(apply_smoothing))}",
    )


def render_pro_line(frame: pd.DataFrame, greek_name: str) -> None:
    if frame.empty or "Price" not in frame.columns or "Value" not in frame.columns:
        st.info(f"No {greek_name} line data to plot.")
        return
    prices = pd.to_numeric(frame["Price"], errors="coerce")
    values = pd.to_numeric(frame["Value"], errors="coerce")
    valid = prices.notna() & values.notna()
    if not valid.any():
        st.info(f"No valid {greek_name} points to plot.")
        return
    fig = go.Figure(
        data=[
            go.Scatter(
                x=prices.loc[valid],
                y=values.loc[valid],
                mode="lines",
                name=greek_name,
                line={"color": GREEK_COLORS.get(greek_name, "#58a6ff"), "width": 2.5},
                hovertemplate="Price=%{x:.4f}<br>" + greek_name + "=%{y:.6f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font={"color": TEXT},
        height=340,
        title={"text": f"{greek_name} vs Price", "x": 0.0, "xanchor": "left"},
        xaxis={"title": "Price", "gridcolor": GRID, "zeroline": False},
        yaxis={"title": greek_name, "gridcolor": GRID, "zeroline": True},
        hovermode="x unified",
        showlegend=False,
        margin={"l": 48, "r": 16, "t": 48, "b": 48},
        uirevision=f"pro-line-{greek_name}",
    )
    st.plotly_chart(
        fig,
        width="stretch",
        height=340,
        theme=None,
        config={"displayModeBar": True},
        key=f"pro-line-chart-{greek_name}",
    )


def add_pro_metrics_tab(
    tab: Any,
    ticker: str,
    strike: float,
    expiry: date,
    frame: pd.DataFrame,
    sigma: float,
    option_type: str,
    model: str,
    v0: float,
    kappa: float,
    theta: float,
    heston_sigma: float,
    rho: float,
) -> None:
    with tab:
        greek_name = st.segmented_control(
            "Pro metric",
            options=PRO_METRICS,
            key="pro_metric_choice",
            required=True,
            help=PRO_METRIC_HELP,
        )
        if greek_name not in PRO_METRICS:
            greek_name = "Vanna"
        apply_smoothing = st.checkbox(
            "Enable Surface Smoothing",
            value=False,
            key="pro_surface_smoothing",
        )
        st.caption(PRO_METRIC_HELP)
        dte = max((expiry - date.today()).days, 1)
        expiry_range = tuple(sorted({float(d) for d in range(7, 91, 7)} | {float(dte)}))
        if frame is not None and not frame.empty and "Price" in frame.columns:
            prices_all = tuple(
                float(p) for p in pd.to_numeric(frame["Price"], errors="coerce").dropna().tolist()
            )
        else:
            prices_all = ()
        if len(prices_all) < 2:
            prices_all = _price_range(float(strike))
        prices_surface = prices_all[::4] if len(prices_all) > 12 else prices_all
        surface_model = model
        if model == MODEL_HESTON:
            st.caption("Heston 3D is opt-in so the 2D curve can render first.")
            if st.button("Build Heston pro surface", key="build_heston_pro_surface"):
                st.session_state["run_heston_pro_surface"] = True
            if not st.session_state.get("run_heston_pro_surface"):
                surface_model = MODEL_BLACK_SCHOLES
        try:
            line_frame = cached_pro_curve(
                prices_all,
                str(greek_name),
                float(strike),
                expiry.isoformat(),
                float(sigma),
                option_type,
                model,
                v0,
                kappa,
                theta,
                heston_sigma,
                rho,
            )
            render_pro_line(line_frame, str(greek_name))
        except Exception as error:
            st.exception(error)
        try:
            pro_frame = cached_pro_surface(
                ticker,
                str(greek_name),
                float(strike),
                expiry_range,
                prices_surface,
                float(sigma),
                option_type,
                surface_model,
                v0,
                kappa,
                theta,
                heston_sigma,
                rho,
                bool(apply_smoothing),
            )
            render_pro_surface(pro_frame, str(greek_name), apply_smoothing=bool(apply_smoothing))
        except Exception as error:
            st.exception(error)


def _sentiment_metric_style(flag: str) -> tuple[str, str]:
    if flag in {"Bearish Skew", "Event Risk / Catalyst", "Volatile/Amplifying"}:
        return "Bearish", "red"
    if flag in {"Range-Bound/Stabilizing", "Stable Term Structure"}:
        return "Bullish", "green"
    return "Neutral", "off"


def add_market_analyst_tab(tab: Any, ticker: str, current_price: float = 0.0, expiry: Any = None) -> None:
    with tab:
        try:
            report = cached_market_sentiment(str(ticker).strip().upper())
        except Exception as error:
            st.exception(error)
            return
        with st.container(border=True):
            st.subheader("Market Outlook")
            st.write(str(report.get("Summary", "")))
        with st.container(horizontal=True):
            for label, key in (("Skew", "Skew"), ("Term structure", "TermStructure"), ("Gamma", "Gamma")):
                flag = str(report.get(key, "—"))
                delta, color = _sentiment_metric_style(flag)
                st.metric(label, flag, delta=delta, delta_color=color, border=True)
        st.text_area(
            "Market Interpretation",
            value=str(report.get("Interpretation", "")),
            height=140,
            disabled=True,
            key="market_interpretation",
        )
        try:
            gex = cached_gex_outlook(str(ticker).strip().upper())
        except Exception as error:
            st.exception(error)
            return
        outlook = str(gex.get("Outlook", ""))
        flip = gex.get("GammaFlip")
        with st.expander("Dealer Positioning (GEX)", expanded=True):
            st.write(outlook)
            if flip is not None and np.isfinite(float(flip)):
                st.metric("Gamma Flip", f"{float(flip):.2f}", border=True)
                if current_price > 0 and abs(float(flip) - current_price) / current_price <= GAMMA_FLIP_NEAR_PCT:
                    st.warning("Market is at a Gamma Flip point—expect high volatility.")
        try:
            vol_risk = cached_vol_risk_outlook(str(ticker).strip().upper())
        except Exception as error:
            st.exception(error)
            return
        with st.expander("Volatility Risk Outlook", expanded=True):
            st.write(str(vol_risk.get("Outlook", "")))
            if bool(vol_risk.get("HighSensitivity")):
                st.warning(
                    "Market is sensitive to volatility changes; expect potential adjustments in Delta and Vega."
                )
        try:
            event = cached_event_outlook(
                str(ticker).strip().upper(),
                str(expiry) if expiry is not None else "",
            )
        except Exception as error:
            st.exception(error)
            return
        short_iv = event.get("ShortIV")
        long_iv = event.get("LongIV")
        spread = None
        if short_iv is not None and long_iv is not None:
            try:
                short_n = float(short_iv)
                long_n = float(long_iv)
            except (TypeError, ValueError):
                short_n = float("nan")
                long_n = float("nan")
            if np.isfinite(short_n) and np.isfinite(long_n):
                spread = short_n - long_n
        with st.expander("Event Outlook (Term Structure)", expanded=True):
            coverage_ok = str(event.get("Outlook", "")) != EVENT_INSUFFICIENT_COVERAGE
            if coverage_ok:
                with st.container(horizontal=True):
                    st.metric("Short-Term IV", _format_iv_metric(short_iv), border=True)
                    st.metric("Long-Term IV", _format_iv_metric(long_iv), border=True)
                    st.metric("IV Spread", _format_iv_metric(spread), border=True)
            st.write(str(event.get("Summary", event.get("Outlook", ""))))
            if bool(event.get("EventRisk")):
                st.warning(
                    "Volatility is elevated for near-term expiries—expect high sensitivity to upcoming news."
                )


def _format_greek_metric(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return f"{number:,.2f}"


def add_portfolio_risk_tab(tab: Any) -> None:
    with tab:
        st.subheader("Portfolio & Risk")
        uploaded = st.file_uploader("Positions CSV", type="csv", key="portfolio_csv")
        if uploaded is not None:
            signature = (str(uploaded.name), int(getattr(uploaded, "size", 0) or 0))
            if st.session_state.get("portfolio_csv_sig") != signature:
                try:
                    parsed = pd.read_csv(uploaded)
                except Exception as error:
                    st.exception(error)
                    parsed = None
                if parsed is not None:
                    st.session_state.portfolio_positions = _normalize_portfolio_frame(parsed)
                    st.session_state.portfolio_csv_sig = signature
                    st.session_state.pop("portfolio_editor", None)
        positions = _normalize_portfolio_frame(st.session_state.portfolio_positions)
        edited = st.data_editor(
            positions,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key="portfolio_editor",
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                "Expiry": st.column_config.TextColumn("Expiry"),
                "Side": st.column_config.SelectboxColumn("Side", options=["long", "short", "Call", "Put"]),
                "Quantity": st.column_config.NumberColumn("Quantity", format="%.2f"),
                "option_type": st.column_config.SelectboxColumn("Type", options=["call", "put"]),
                "S": st.column_config.NumberColumn("Spot", format="%.2f"),
                "sigma": st.column_config.NumberColumn("IV", format="%.4f"),
                "premium": st.column_config.NumberColumn("Premium", format="%.2f"),
                "EntryPrice": st.column_config.NumberColumn("Entry price", format="%.2f"),
            },
        )
        st.session_state.portfolio_positions = edited
        book = edited.copy()
        if "Ticker" in book.columns:
            book = book[book["Ticker"].astype(str).str.strip().ne("")]
        if "Strike" in book.columns:
            strikes = pd.to_numeric(book["Strike"], errors="coerce")
            book = book[strikes.notna() & (strikes > 0)]
        if book.empty:
            st.info("Add positions in the table or upload a CSV to score portfolio risk.")
            return
        try:
            risk = cached_portfolio_risk(book)
        except Exception as error:
            st.exception(error)
            return
        with st.container(horizontal=True):
            st.metric("Net Delta", _format_greek_metric(risk.get("Delta")), border=True)
            st.metric("Net Gamma", _format_greek_metric(risk.get("Gamma")), border=True)
            st.metric("Net Theta", _format_greek_metric(risk.get("Theta")), border=True)
            st.metric("Net Vega", _format_greek_metric(risk.get("Vega")), border=True)
        filled = risk.get("Positions")
        if isinstance(filled, pd.DataFrame) and not filled.empty:
            show_cols = [c for c in ("Ticker", "Strike", "Expiry", "Side", "Quantity", "EntryPrice", "S", "premium") if c in filled.columns]
            with st.container(border=True):
                st.markdown("**Computed spots and premiums**")
                st.dataframe(filled.loc[:, show_cols] if show_cols else filled, hide_index=True, width="stretch")
        concentration = risk.get("TickerConcentration")
        if concentration is None:
            concentration = risk.get("LargestPosition")
        try:
            conc_pct = float(concentration) if concentration is not None else float("nan")
        except (TypeError, ValueError):
            conc_pct = float("nan")
        if np.isfinite(conc_pct) and conc_pct > CONCENTRATION_WARN_PCT:
            name = str(risk.get("ConcentratedTicker") or "one ticker")
            st.warning(
                f"Concentration risk: {name} is {conc_pct:.0f}% of your portfolio "
                f"(above {CONCENTRATION_WARN_PCT:.0f}%)."
            )
        delta_rows = risk.get("DeltaByTicker") or []
        delta_frame = pd.DataFrame(delta_rows)
        if delta_frame.empty or "Ticker" not in delta_frame.columns or "Delta" not in delta_frame.columns:
            st.info("No ticker-level delta exposure to chart.")
            return
        delta_frame["Delta"] = pd.to_numeric(delta_frame["Delta"], errors="coerce")
        delta_frame = delta_frame.dropna(subset=["Ticker", "Delta"])
        with st.container(border=True):
            st.markdown("**Delta exposure by ticker**")
            st.bar_chart(delta_frame, x="Ticker", y="Delta", color="#58a6ff", width="stretch")


def render_vol_surface(frame: pd.DataFrame | str) -> None:
    try:
        if isinstance(frame, str):
            st.info(frame)
            return
        x_col = "Price" if "Price" in frame.columns else "Strike"
        required = {x_col, "DaysToExpiry", "IV"}
        if frame.empty or not required.issubset(frame.columns):
            st.info("No implied volatility surface data to plot.")
            return
        plot_df = frame.loc[:, [x_col, "DaysToExpiry", "IV"]].copy()
        plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
        plot_df["DaysToExpiry"] = pd.to_numeric(plot_df["DaysToExpiry"], errors="coerce")
        plot_df["IV"] = pd.to_numeric(plot_df["IV"], errors="coerce")
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
        if plot_df.empty:
            st.info("No implied volatility surface data to plot.")
            return

        scatter = px.scatter_3d(
            plot_df,
            x=x_col,
            y="DaysToExpiry",
            z="IV",
            color="IV",
            color_continuous_scale="Viridis",
            title="Implied Volatility Surface",
        )
        scatter.update_traces(marker={"size": 3})
        scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor=DARK_BG,
            font={"color": TEXT},
            height=560,
            scene={
                "xaxis_title": "Price",
                "yaxis_title": "Days to Expiry",
                "zaxis_title": "IV",
                "bgcolor": PANEL_BG,
            },
            margin={"l": 8, "r": 8, "t": 48, "b": 8},
            uirevision="vol-surface-scatter",
        )
        st.plotly_chart(
            scatter,
            width="stretch",
            config={"displayModeBar": True, "scrollZoom": True},
            key="vol-surface-scatter",
        )

        try:
            contour = px.density_contour(
                plot_df,
                x=x_col,
                y="DaysToExpiry",
                title="Implied Volatility Surface",
            )
            contour.update_traces(
                contours_coloring="fill",
                colorscale="Viridis",
                colorbar={"title": "IV"},
            )
            contour.update_layout(
                template="plotly_dark",
                paper_bgcolor=DARK_BG,
                font={"color": TEXT},
                height=420,
                xaxis_title="Price",
                yaxis_title="Days to Expiry",
                margin={"l": 48, "r": 16, "t": 48, "b": 48},
                uirevision="vol-surface-contour",
            )
            st.plotly_chart(
                contour,
                width="stretch",
                config={"displayModeBar": True, "scrollZoom": True},
                key="vol-surface-contour",
            )
        except Exception as contour_error:
            st.exception(contour_error)
    except Exception as error:
        st.exception(error)


def _surface_payload_is_empty(surface: Any, value_key: str) -> bool:
    """True when a stored snapshot surface has no finite, non-zero values."""
    if not isinstance(surface, dict):
        return True
    values = surface.get(value_key) or []
    if not isinstance(values, list) or not values:
        return True
    arr = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    finite = arr[np.isfinite(arr)]
    return finite.size == 0 or bool(np.all(finite == 0.0))


def _plotly_surface(
    z_data: Any,
    x: Any = None,
    y: Any = None,
    *,
    rows: int | None = None,
    cols: int | None = None,
    **surface_kwargs: Any,
) -> go.Surface:
    """Build ``go.Surface`` with 2D ``z`` via ``reshape_to_surface`` + axis gate."""
    try:
        z_data = reshape_to_surface(z_data)
        if z_data.ndim != 2 or z_data.size == 0:
            raise ValueError("invalid surface shape")
        z2d, x_ok, y_ok, warning = prepare_plotly_surface_xyz(z_data, x, y, rows=rows, cols=cols)
        if z2d.ndim != 2 or z2d.size == 0:
            raise ValueError("invalid surface shape")
        if warning:
            st.warning(warning)
        payload: dict[str, Any] = {"z": z2d, **surface_kwargs}
        if x_ok is not None and y_ok is not None:
            payload["x"] = x_ok
            payload["y"] = y_ok
        return go.Surface(**payload)
    except Exception:
        st.error("Surface rendering failed: Invalid data shape.")
        return go.Surface(z=np.array([[np.nan]], dtype=np.float64))


def _snapshot_ui_metadata(payload: Mapping[str, Any] | None, stamp: str) -> dict[str, Any]:
    """Ticker / Strike / Expiry / Type / Timestamp for the Snapshot Details expander."""
    payload = payload if isinstance(payload, dict) else {}
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return {
        "Ticker": payload.get("ticker") or meta.get("ticker"),
        "Strike": payload.get("strike") if payload.get("strike") is not None else meta.get("strike"),
        "Expiry": payload.get("expiry") or meta.get("expiry"),
        "Type": payload.get("contract_type") or meta.get("contract_type"),
        "Timestamp": payload.get("stamp") or stamp,
        "Label": DataRepository.format_snapshot_label(payload, stamp),
    }


def _snapshot_option_labels(repo: DataRepository, ticker: str, snapshots: list[str]) -> dict[str, str]:
    """Map stamp → full metadata label for Time Machine dropdowns."""
    labels: dict[str, str] = {}
    for stamp in snapshots:
        payload = repo._snapshot_payload(ticker, stamp) or {}
        labels[str(stamp)] = DataRepository.format_snapshot_label(payload, str(stamp))
    return labels


def _write_snapshot_delta_ranges(payload_a: dict, payload_b: dict, label_a: str, label_b: str) -> None:
    """Show min/max of both compared delta surfaces so a flat drift can be diagnosed."""

    def _range(payload: dict) -> tuple[float | None, float | None, int]:
        values = (payload.get("delta_surface") or {}).get("Delta") or []
        arr = np.asarray([np.nan if v is None else v for v in values], dtype=float) if values else np.empty(0)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None, None, 0
        return float(np.min(finite)), float(np.max(finite)), int(finite.size)

    a_min, a_max, a_n = _range(payload_a)
    b_min, b_max, b_n = _range(payload_b)
    st.warning("Delta drift is flat; comparing the two snapshot delta surfaces:")
    st.write(
        {
            "Start": {
                "stamp": payload_a.get("stamp", label_a),
                "snapshot_id": payload_a.get("snapshot_id"),
                "spot": payload_a.get("spot"),
                "delta_min": a_min,
                "delta_max": a_max,
                "points": a_n,
            },
            "End": {
                "stamp": payload_b.get("stamp", label_b),
                "snapshot_id": payload_b.get("snapshot_id"),
                "spot": payload_b.get("spot"),
                "delta_min": b_min,
                "delta_max": b_max,
                "points": b_n,
            },
        }
    )
    if a_min == b_min and a_max == b_max and a_n == b_n:
        st.write("Both snapshots have identical delta surface ranges — the underlying data did not change.")


def render_delta_drift_surface(
    frame: pd.DataFrame,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None = None,
) -> None:
    x_axis: np.ndarray
    y_axis: np.ndarray
    Z: np.ndarray
    synthetic = bool(getattr(frame, "attrs", {}).get("synthetic")) if frame is not None else False
    if grid is not None and len(grid) >= 3 and np.asarray(grid[2]).size > 0:
        x_axis = np.asarray(grid[0], dtype=float)
        y_axis = np.asarray(grid[1], dtype=float)
        # Grid contract: Z = surface_b - surface_a (later - earlier), or synthetic.
        Z = np.array(grid[2], dtype=float, copy=True)
        if len(grid) >= 4:
            synthetic = bool(grid[3]) or synthetic
    else:
        required = {"Strike", "DaysToExpiry", "Delta"}
        if frame is None or frame.empty or not required.issubset(frame.columns):
            st.info("No delta-drift surface for the selected snapshots.")
            return
        plot = frame.loc[:, ["Strike", "DaysToExpiry", "Delta"]].copy()
        plot["Strike"] = pd.to_numeric(plot["Strike"], errors="coerce")
        plot["DaysToExpiry"] = pd.to_numeric(plot["DaysToExpiry"], errors="coerce")
        plot["Delta"] = pd.to_numeric(plot["Delta"], errors="coerce")
        plot = plot.replace([np.inf, -np.inf], np.nan).dropna(subset=["Strike", "DaysToExpiry"])
        if plot.empty or not plot["Delta"].notna().any():
            st.info("No delta-drift surface for the selected snapshots.")
            return
        pivot = plot.pivot_table(index="DaysToExpiry", columns="Strike", values="Delta", aggfunc="mean")
        pivot = pivot.sort_index().sort_index(axis=1)
        x_axis = pivot.columns.to_numpy(dtype=float)
        y_axis = pivot.index.to_numpy(dtype=float)
        Z = np.array(pivot.to_numpy(dtype=float), copy=True)
    if synthetic:
        st.info("No real drift detected. Showing synthetic test surface.")
    Z = Z.astype(float)
    Z[~np.isfinite(Z)] = np.nan
    try:
        z_data = reshape_to_surface(Z)
        if z_data.ndim != 2 or z_data.size == 0:
            raise ValueError("invalid surface shape")
        z2d, _, _, _ = prepare_plotly_surface_xyz(z_data, x_axis, y_axis)
        if z2d.ndim != 2 or not np.isfinite(z2d).any():
            st.write(f"Z-matrix shape: {z2d.shape}")
            st.error("Delta drift Z-matrix is not a 2D grid or contains no finite values; refusing to render a flat plane.")
            return
    except Exception:
        st.error("Surface rendering failed: Invalid data shape.")
        return
    z_min, z_max = float(np.nanmin(z2d)), float(np.nanmax(z2d))
    st.write(f"Z-matrix shape: {z2d.shape}, Min: {z_min}, Max: {z_max}")
    st.write({"drift_min": z_min, "drift_max": z_max, "synthetic": synthetic})
    finite = z2d[np.isfinite(z2d)]
    if not synthetic and (z_min == z_max or (finite.size > 0 and bool(np.allclose(finite, 0.0, atol=1e-12)))):
        st.write({"drift_min": z_min, "drift_max": z_max, "flat_plane": True})
    fig = go.Figure(
        data=[
            _plotly_surface(
                z2d,
                x_axis,
                y_axis,
                colorscale="RdBu",
                reversescale=True,
                cmid=0,
                colorbar={"title": "Delta drift" + (" (synthetic)" if synthetic else "")},
                hovertemplate="Strike=%{x:.2f}<br>DTE=%{y:.1f}<br>Drift=%{z:.6f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT},
        height=560,
        title={"text": "Delta drift", "x": 0.0, "xanchor": "left"},
        scene={
            "xaxis_title": "Strike",
            "yaxis_title": "Days to Expiry",
            "zaxis_title": "Delta drift",
            "bgcolor": PANEL_BG,
            "dragmode": "orbit",
        },
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        uirevision="time-machine-delta-drift",
    )
    st.plotly_chart(
        fig,
        width="stretch",
        height=560,
        theme=None,
        config={"displayModeBar": True, "scrollZoom": True},
        key="time-machine-delta-drift",
    )
    st.caption("Blue: increasing Delta (buying pressure). Red: decreasing Delta (selling pressure).")


def render_time_machine_vol_surface(frame: pd.DataFrame | str, timestamp: str) -> None:
    if isinstance(frame, str):
        st.warning("No data available for the Volatility Surface.")
        st.info(frame)
        return
    required = {"Strike", "DaysToExpiry", "IV"}
    if frame is None or frame.empty or not required.issubset(frame.columns):
        st.warning("No data available for the Volatility Surface.")
        return
    plot = frame.loc[:, ["Strike", "DaysToExpiry", "IV"]].copy()
    plot["Strike"] = pd.to_numeric(plot["Strike"], errors="coerce")
    plot["DaysToExpiry"] = pd.to_numeric(plot["DaysToExpiry"], errors="coerce")
    plot["IV"] = pd.to_numeric(plot["IV"], errors="coerce")
    plot = plot.replace([np.inf, -np.inf], np.nan).dropna(subset=["Strike", "DaysToExpiry"])
    if plot.empty or not plot["IV"].notna().any():
        st.warning("No data available for the Volatility Surface.")
        return
    pivot = plot.pivot_table(index="DaysToExpiry", columns="Strike", values="IV", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    x_axis = pivot.columns.to_numpy(dtype=float)
    y_axis = pivot.index.to_numpy(dtype=float)
    # Axes → meshgrid so Z rows/cols match Plotly Surface (len(y), len(x)).
    _mesh_x, _mesh_y = np.meshgrid(x_axis, y_axis)
    z = np.array(pivot.to_numpy(dtype=float), copy=True)
    z[~np.isfinite(z)] = np.nan
    try:
        z_data = reshape_to_surface(z)
        if z_data.ndim != 2 or z_data.size == 0:
            raise ValueError("invalid surface shape")
        z2d, x_ok, y_ok, axis_warning = prepare_plotly_surface_xyz(
            z_data, x_axis, y_axis, rows=_mesh_y.shape[0], cols=_mesh_x.shape[1]
        )
        if z2d.ndim != 2 or z2d.size == 0:
            raise ValueError("invalid surface shape")
        if not np.isfinite(z2d).any():
            st.warning("No data available for the Volatility Surface.")
            return
        if axis_warning:
            st.warning(axis_warning)
        st.write(f"IV Z-matrix shape: {z2d.shape}, Min: {np.nanmin(z2d)}, Max: {np.nanmax(z2d)}")
        surface_kwargs: dict[str, Any] = {
            "colorscale": "Viridis",
            "colorbar": {"title": "IV"},
            "hovertemplate": "Strike=%{x:.2f}<br>DTE=%{y:.1f}<br>IV=%{z:.4f}<extra></extra>",
        }
        # Axes gated inside _plotly_surface (reshape already applied).
        surface_trace = _plotly_surface(z2d, x_ok if x_ok is not None else x_axis, y_ok if y_ok is not None else y_axis, **surface_kwargs)
    except Exception:
        st.error("Surface rendering failed: Invalid data shape.")
        return
    fig = go.Figure(data=[surface_trace])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT},
        height=560,
        title={"text": f"Volatility surface · {timestamp}", "x": 0.0, "xanchor": "left"},
        scene={
            "xaxis_title": "Strike",
            "yaxis_title": "Days to Expiry",
            "zaxis_title": "IV",
            "bgcolor": PANEL_BG,
            "dragmode": "orbit",
        },
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        uirevision="time-machine-vol-surface",
    )
    st.plotly_chart(
        fig,
        width="stretch",
        height=560,
        theme=None,
        config={"displayModeBar": True, "scrollZoom": True},
        key="time-machine-vol-surface",
    )


def add_time_machine_tab(
    tab: Any,
    ticker: str,
    strike: float | None = None,
    expiry: Any = None,
) -> None:
    with tab:
        _surface_ingest_alerts(_get_repo())
        alert = st.session_state.get("ingest_alert")
        if alert and isinstance(alert, str):
            if "Token Missing" in alert:
                st.error(alert)
            elif alert.startswith("API Error:"):
                st.warning(alert)
        repo = _get_repo()
        if st.button("Pre-fill Delta Drift", key=f"tm_prefill_{ticker}"):
            try:
                active_strike = strike if strike is not None else st.session_state.get("listed_strike")
                active_expiry = expiry if expiry is not None else st.session_state.get("expiry_iso")
                result = prefill_drift_data(
                    str(ticker),
                    active_strike,
                    active_expiry,
                    repo=repo,
                )
                st.session_state[f"tm_start_{ticker}"] = result["start_label"]
                st.session_state[f"tm_end_{ticker}"] = result["end_label"]
                for cache in (cached_delta_drift, cached_delta_drift_grid, cached_snapshot_vol_surface):
                    clear = getattr(cache, "clear", None)
                    if callable(clear):
                        clear()
                st.rerun()
            except Exception as error:
                st.error(f"Pre-fill Delta Drift failed: {error}")
        # Data sync: stamps come from timestamped files under data_history/ via DataRepository.
        snapshots = repo.get_available_snapshots(ticker)
        if not snapshots:
            st.info("No snapshots in data_history/ for this ticker.")
            return
        stamp_labels = _snapshot_option_labels(repo, ticker, snapshots)

        def _label_for(stamp: Any) -> str:
            key = str(stamp)
            return stamp_labels.get(key) or DataRepository.format_snapshot_label({}, key)

        st.subheader("Delta drift")
        start_col, end_col = st.columns([1, 1])
        start_key = f"tm_start_{ticker}"
        end_key = f"tm_end_{ticker}"
        if start_key in st.session_state and st.session_state[start_key] not in snapshots:
            del st.session_state[start_key]
        if end_key in st.session_state and st.session_state[end_key] not in snapshots:
            del st.session_state[end_key]
        preferred_start = st.session_state.get(start_key)
        preferred_end = st.session_state.get(end_key)
        start_index = snapshots.index(preferred_start) if preferred_start in snapshots else 0
        end_index = (
            snapshots.index(preferred_end) if preferred_end in snapshots else max(0, len(snapshots) - 1)
        )
        with start_col:
            start = st.selectbox(
                "Start Snapshot",
                snapshots,
                index=start_index,
                key=start_key,
                format_func=_label_for,
            )
        with end_col:
            end = st.selectbox(
                "End Snapshot",
                snapshots,
                index=end_index,
                key=end_key,
                format_func=_label_for,
            )
        payload_a = repo._snapshot_payload(ticker, start) or {}
        payload_b = repo._snapshot_payload(ticker, end) or {}
        with st.expander("Snapshot Details", expanded=False):
            st.write(
                {
                    "Start": _snapshot_ui_metadata(payload_a, str(start)),
                    "End": _snapshot_ui_metadata(payload_b, str(end)),
                }
            )
        st.write(
            {
                "Start Snapshot keys": list(payload_a.keys()),
                "End Snapshot keys": list(payload_b.keys()),
            }
        )
        if "delta_surface" not in payload_a or "delta_surface" not in payload_b:
            st.error("Snapshot does not contain delta data!")
        elif _surface_payload_is_empty(payload_a.get("delta_surface"), "Delta") or _surface_payload_is_empty(
            payload_b.get("delta_surface"), "Delta"
        ):
            st.error("Snapshot data is corrupt/empty.")
        elif not DataRepository.validate_snapshot_match(payload_a, payload_b)[0]:
            st.warning(
                "Invalid Comparison: You must select snapshots of the same contract to calculate Delta Drift."
            )
        else:
            if str(start) == str(end):
                st.info("Identical snapshots selected — real drift is flat; synthetic fallback may apply.")
            else:
                st.success("Valid Contract Pair: Calculating Drift...")
            try:
                grid = cached_delta_drift_grid(str(ticker), str(start), str(end))
                drift = cached_delta_drift(str(ticker), str(start), str(end))
                synthetic = bool(getattr(drift, "attrs", {}).get("synthetic")) or (
                    len(grid) >= 4 and bool(grid[3])
                )
                z_grid = np.asarray(grid[2], dtype=float) if grid is not None and len(grid) >= 3 else np.empty(0)
                finite = z_grid[np.isfinite(z_grid)] if z_grid.size else z_grid
                if not synthetic and (
                    finite.size == 0
                    or float(np.min(finite)) == float(np.max(finite))
                    or bool(np.all(np.isclose(finite, 0.0)))
                ):
                    st.write(
                        {
                            "drift_min": float(np.min(finite)) if finite.size else None,
                            "drift_max": float(np.max(finite)) if finite.size else None,
                        }
                    )
                    _write_snapshot_delta_ranges(payload_a, payload_b, str(start), str(end))
                render_delta_drift_surface(drift, grid=grid)
            except Exception as error:
                st.error("Could not render the Delta Drift surface.")
                st.exception(error)
        st.subheader("Surface evolution")
        # Map each stamp to its on-disk snapshot file so the loader receives the
        # exact filename rather than a stamp/index that could match several files.
        stamp_to_file: dict[str, str] = {}
        for path in repo.get_historical_snapshots(ticker):
            name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            prefix = f"{repo._safe_ticker(ticker)}_"
            if name.startswith(prefix) and name.endswith(".json"):
                stamp_to_file.setdefault(name[len(prefix) : -len(".json")], path)
        stamp = st.select_slider(
            "Snapshot",
            options=snapshots,
            value=snapshots[-1],
            key=f"tm_scrub_{ticker}",
            format_func=_label_for,
        )
        snapshot_file = stamp_to_file.get(str(stamp), str(stamp))
        scrub_payload = repo._snapshot_payload(ticker, snapshot_file) or repo._snapshot_payload(ticker, stamp) or {}
        snapshot = scrub_payload if isinstance(scrub_payload, dict) else {}
        st.write(f"Volatility Surface Keys: {list(snapshot.keys())}")
        with st.expander("Selected Snapshot Details", expanded=False):
            st.write(_snapshot_ui_metadata(scrub_payload, str(stamp)))
        st.caption(f"Snapshot file: {snapshot_file}")
        try:
            vol_block = DataRepository.resolve_vol_surface_block(snapshot)
            if vol_block is not None and not DataRepository._surface_is_empty(vol_block, "IV"):
                surface = DataRepository.vol_surface_block_to_frame(vol_block)
            else:
                surface = cached_snapshot_vol_surface(str(ticker), snapshot_file)
            if isinstance(surface, pd.DataFrame) and "IV" in surface.columns and not surface.empty:
                iv_vals = pd.to_numeric(surface["IV"], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(iv_vals).any():
                    st.write(f"IV points: {iv_vals.size}, Min: {np.nanmin(iv_vals)}, Max: {np.nanmax(iv_vals)}")
                else:
                    st.write(f"IV points: {iv_vals.size}, no finite values")
                    st.warning("No data available for the Volatility Surface.")
            elif not isinstance(surface, pd.DataFrame) or (isinstance(surface, pd.DataFrame) and surface.empty):
                st.warning("No data available for the Volatility Surface.")
            render_time_machine_vol_surface(surface, str(stamp))
        except Exception as error:
            st.error("Could not render the snapshot volatility surface.")
            st.exception(error)


def render_advanced_greek_surface(frame: pd.DataFrame, greek: str) -> None:
    required = {"Price", "DaysToExpiry", "Value"}
    if frame.empty or not required.issubset(frame.columns):
        st.info(f"No {greek} surface data to plot.")
        return
    pivot = frame.pivot_table(index="DaysToExpiry", columns="Price", values="Value", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    fig = go.Figure(
        data=[
            _plotly_surface(
                pivot.to_numpy(dtype=float),
                pivot.columns.to_numpy(dtype=float),
                pivot.index.to_numpy(dtype=float),
                colorscale="Viridis",
                colorbar={"title": greek},
                hovertemplate="Price=%{x:.2f}<br>DTE=%{y:.1f}<br>" + greek + "=%{z:.6f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT},
        height=560,
        title={"text": f"{greek} surface", "x": 0.0, "xanchor": "left"},
        scene={
            "xaxis_title": "Price",
            "yaxis_title": "Days to Expiry",
            "zaxis_title": greek,
            "bgcolor": PANEL_BG,
        },
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        uirevision=f"advanced-{greek}",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": True, "scrollZoom": True},
        key=f"advanced-surface-{greek}",
    )


def render_metric_header(
    frame: pd.DataFrame,
    spot: float,
    ticker: str,
    strike: float,
    expiry: date,
    current_iv: float | None,
    use_heston: bool,
) -> None:
    iv_row = st.columns(1)
    iv_row[0].metric("Current IV", _format_iv(current_iv))
    values = _spot_greeks(frame, spot, use_heston=use_heston)
    cards = st.columns(len(GREEK_COLUMNS), gap="medium")
    for name, col in zip(GREEK_COLUMNS, cards):
        col.metric(name, _format_greek(name, values[name]))
    context = st.columns(4)
    context[0].metric("Ticker", ticker)
    context[1].metric("Strike", f"{strike:g}")
    context[2].metric("Expiry", expiry.isoformat())
    context[3].metric("Spot", f"{spot:g}")


def _load_gamma_surface(
    ticker: str,
    strike: float,
    expiry_range: tuple[float, ...],
    prices: tuple[float, ...],
    sigma: float,
    option_type: str,
    model: str,
    v0: float,
    kappa: float,
    theta: float,
    heston_sigma: float,
    rho: float,
) -> pd.DataFrame:
    fingerprint = (
        ticker,
        round(float(strike), 8),
        expiry_range,
        prices,
        round(float(sigma), 8),
        option_type,
        model,
        round(float(v0), 8),
        round(float(kappa), 8),
        round(float(theta), 8),
        round(float(heston_sigma), 8),
        round(float(rho), 8),
    )
    stored = st.session_state.get("gamma_surface_frame")
    if fingerprint == st.session_state.get("gamma_surface_fp") and isinstance(stored, pd.DataFrame) and not stored.empty:
        return stored
    surface = cached_gamma_surface(
        ticker, float(strike), expiry_range, prices, float(sigma), option_type, model, v0, kappa, theta, heston_sigma, rho
    )
    st.session_state.gamma_surface_frame = surface
    st.session_state.gamma_surface_fp = fingerprint
    return surface


def _load_greek_surface(
    ticker: str,
    strike: float,
    expiry_range: tuple[float, ...],
    prices: tuple[float, ...],
    sigma: float,
    option_type: str,
    model: str,
    v0: float,
    kappa: float,
    theta: float,
    heston_sigma: float,
    rho: float,
    greek: str,
) -> pd.DataFrame:
    fingerprint = (
        ticker,
        round(float(strike), 8),
        expiry_range,
        prices,
        round(float(sigma), 8),
        option_type,
        model,
        round(float(v0), 8),
        round(float(kappa), 8),
        round(float(theta), 8),
        round(float(heston_sigma), 8),
        round(float(rho), 8),
        str(greek),
    )
    stored = st.session_state.get("greek_surface_frame")
    if fingerprint == st.session_state.get("greek_surface_fp") and isinstance(stored, pd.DataFrame) and not stored.empty:
        return stored
    surface = cached_greek_surface(
        ticker,
        float(strike),
        expiry_range,
        prices,
        float(sigma),
        option_type,
        model,
        v0,
        kappa,
        theta,
        heston_sigma,
        rho,
        str(greek),
    )
    st.session_state.greek_surface_frame = surface
    st.session_state.greek_surface_fp = fingerprint
    return surface


def _load_term_structure(
    ticker: str,
    strike: float,
    spot: float,
    expiry_range: tuple[float, ...],
    sigma: float,
    option_type: str,
    model: str,
    v0: float,
    kappa: float,
    theta: float,
    heston_sigma: float,
    rho: float,
) -> pd.DataFrame:
    fingerprint = (
        ticker,
        round(float(strike), 8),
        round(float(spot), 6),
        expiry_range,
        round(float(sigma), 8),
        option_type,
        model,
        round(float(v0), 8),
        round(float(kappa), 8),
        round(float(theta), 8),
        round(float(heston_sigma), 8),
        round(float(rho), 8),
    )
    stored = st.session_state.get("term_structure_frame")
    if fingerprint == st.session_state.get("term_structure_fp") and isinstance(stored, pd.DataFrame) and not stored.empty:
        return stored
    term = cached_delta_term_structure(
        ticker, float(strike), float(spot), expiry_range, float(sigma), option_type, model, v0, kappa, theta, heston_sigma, rho
    )
    st.session_state.term_structure_frame = term
    st.session_state.term_structure_fp = fingerprint
    return term


def _load_vol_surface(ticker: str) -> pd.DataFrame | str:
    fingerprint = (str(ticker).strip().upper(),)
    stored = st.session_state.get("vol_surface_frame")
    if fingerprint == st.session_state.get("vol_surface_fp") and (
        (isinstance(stored, pd.DataFrame) and not stored.empty) or stored == "Insufficient Data"
    ):
        return stored
    surface = cached_vol_surface(ticker)
    st.session_state.vol_surface_frame = surface
    st.session_state.vol_surface_fp = fingerprint
    return surface


def test_fetch_data(ticker: str) -> dict[str, Any]:
    """Fetch chain + market-flow payloads for the Test Dashboard (isolated)."""
    symbol = str(ticker or "").strip().upper() or DEFAULT_TICKER
    chain = pd.DataFrame()
    sentiment: dict[str, Any] = {}
    gex: dict[str, Any] = {}
    try:
        chain = cached_analyst_chain(symbol)
    except Exception:
        chain = pd.DataFrame()
    try:
        sentiment = cached_market_sentiment(symbol)
    except Exception:
        sentiment = {}
    try:
        gex = cached_gex_outlook(symbol)
    except Exception:
        gex = {}
    return {"ticker": symbol, "chain": chain, "sentiment": sentiment, "gex": gex}


def test_fetch_portfolio_risk() -> dict[str, Any]:
    """Score session portfolio via existing risk helper (Test Dashboard only)."""
    positions = _normalize_portfolio_frame(st.session_state.get("portfolio_positions"))
    book = positions.copy()
    if "Ticker" in book.columns:
        book = book[book["Ticker"].astype(str).str.strip().ne("")]
    if "Strike" in book.columns:
        strikes = pd.to_numeric(book["Strike"], errors="coerce")
        book = book[strikes.notna() & (strikes > 0)]
    if book.empty:
        return {}
    try:
        return cached_portfolio_risk(book)
    except Exception:
        return {}


def test_estimate_ticket_greeks(
    spot: float,
    strike: float,
    sigma: float,
    option_type: str,
    *,
    time_years: float = 30.0 / DAYS_PER_YEAR,
) -> dict[str, float | None]:
    """BS Greeks for the mock trade ticket using ``calculate_greeks``."""
    try:
        return calculate_greeks(
            {
                "S": float(spot),
                "K": float(strike),
                "T": float(time_years),
                "r": float(DEFAULT_RATE),
                "sigma": float(sigma),
                "option_type": str(option_type or "call"),
            }
        )
    except Exception:
        return {"Delta": None, "Gamma": None, "Theta": None, "Vega": None, "Rho": None}


def test_render_options_chain(ticker: str) -> None:
    """Top-left: Options Chain & Market Flow."""
    st.subheader("Options Chain & Market Flow")
    payload = test_fetch_data(ticker)
    chain = payload.get("chain")
    sentiment = payload.get("sentiment") or {}
    gex = payload.get("gex") or {}
    with st.container(horizontal=True):
        st.metric("Skew", str(sentiment.get("Skew", "—")), border=True)
        st.metric("Term", str(sentiment.get("TermStructure", "—")), border=True)
        flip = gex.get("GammaFlip")
        flip_label = f"{float(flip):.2f}" if flip is not None and np.isfinite(float(flip)) else "—"
        st.metric("Gamma Flip", flip_label, border=True)
    if isinstance(chain, pd.DataFrame) and not chain.empty:
        show = [c for c in ("Strike", "expiry", "Expiration", "type", "option_type", "bid", "ask", "lastPrice", "impliedVolatility", "volume", "openInterest") if c in chain.columns]
        st.dataframe(chain.loc[:, show].head(40) if show else chain.head(40), hide_index=True, width="stretch")
    else:
        st.info("No live chain data available for this ticker.")
    outlook = str(gex.get("Outlook") or sentiment.get("Summary") or "")
    if outlook:
        st.caption(outlook)


def test_render_portfolio_risk_pane() -> None:
    """Top-right: Portfolio Risk (Net Greeks, Concentration)."""
    st.subheader("Portfolio Risk")
    risk = test_fetch_portfolio_risk()
    if not risk:
        st.info("Add positions in Portfolio & Risk to populate net Greeks.")
        return
    with st.container(horizontal=True):
        st.metric("Net Delta", _format_greek_metric(risk.get("Delta")), border=True)
        st.metric("Net Gamma", _format_greek_metric(risk.get("Gamma")), border=True)
        st.metric("Net Theta", _format_greek_metric(risk.get("Theta")), border=True)
        st.metric("Net Vega", _format_greek_metric(risk.get("Vega")), border=True)
    concentration = risk.get("TickerConcentration")
    if concentration is None:
        concentration = risk.get("LargestPosition")
    try:
        conc_pct = float(concentration) if concentration is not None else float("nan")
    except (TypeError, ValueError):
        conc_pct = float("nan")
    name = str(risk.get("ConcentratedTicker") or "—")
    if np.isfinite(conc_pct):
        st.metric("Concentration", f"{name}: {conc_pct:.0f}%", border=True)
        if conc_pct > CONCENTRATION_WARN_PCT:
            st.warning(
                f"Concentration risk: {name} is {conc_pct:.0f}% of portfolio "
                f"(above {CONCENTRATION_WARN_PCT:.0f}%)."
            )


def test_render_trade_ticket(ticker: str, spot: float, sigma: float) -> None:
    """Bottom-left: mock Trade Ticket with Simulated P&L from Greeks."""
    st.subheader("Trade Ticket")
    t_col, k_col = st.columns(2)
    with t_col:
        ticket_ticker = st.text_input("Ticker", value=str(ticker).upper(), key="test_ticket_ticker")
    with k_col:
        ticket_strike = st.number_input(
            "Strike",
            min_value=0.01,
            value=float(spot) if spot and spot > 0 else 100.0,
            step=1.0,
            key="test_ticket_strike",
        )
    s_col, q_col = st.columns(2)
    with s_col:
        ticket_side = st.selectbox("Side", options=("Buy", "Sell"), key="test_ticket_side")
    with q_col:
        ticket_qty = st.number_input("Quantity", min_value=1.0, value=1.0, step=1.0, key="test_ticket_qty")
    opt_type = st.selectbox("Option type", options=("call", "put"), key="test_ticket_type")
    use_spot = float(spot) if spot and spot > 0 else float(ticket_strike)
    use_sigma = float(sigma) if sigma and sigma > 0 else float(DEFAULT_SIGMA)
    greeks = test_estimate_ticket_greeks(use_spot, float(ticket_strike), use_sigma, str(opt_type))
    with st.container(horizontal=True):
        st.metric("Δ", _format_greek_metric(greeks.get("Delta")), border=True)
        st.metric("Γ", _format_greek_metric(greeks.get("Gamma")), border=True)
        st.metric("Θ", _format_greek_metric(greeks.get("Theta")), border=True)
        st.metric("ν", _format_greek_metric(greeks.get("Vega")), border=True)
    pnl = test_simulated_pnl(
        float(greeks.get("Delta") or 0.0),
        float(greeks.get("Gamma") or 0.0),
        float(greeks.get("Theta") or 0.0),
        float(greeks.get("Vega") or 0.0),
        float(ticket_qty),
        str(ticket_side),
    )
    st.metric("Simulated P&L", _format_greek_metric(pnl), border=True)
    st.caption(
        f"Mock only · {ticket_ticker} {ticket_side} {int(ticket_qty)}× "
        f"{opt_type} @ {ticket_strike:.2f} · see test_simulated_pnl docstring."
    )
    st.button("Submit (simulated)", key="test_ticket_submit", disabled=True)


def test_render_alerts_compliance() -> None:
    """Bottom-right: Alerts & Compliance risk-threshold monitors."""
    st.subheader("Alerts & Compliance")
    risk = test_fetch_portfolio_risk()
    alerts: list[tuple[str, str]] = []
    if not risk:
        st.info("No portfolio loaded — compliance monitors idle.")
        return
    for greek, limit in (("Delta", 500.0), ("Gamma", 50.0), ("Vega", 200.0)):
        try:
            value = float(risk.get(greek) or 0.0)
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value) and abs(value) > limit:
            alerts.append(("error", f"{greek} |net|={abs(value):.1f} exceeds limit {limit:.0f}."))
        elif np.isfinite(value) and abs(value) > limit * 0.8:
            alerts.append(("warning", f"{greek} |net|={abs(value):.1f} near limit {limit:.0f}."))
    concentration = risk.get("TickerConcentration")
    if concentration is None:
        concentration = risk.get("LargestPosition")
    try:
        conc_pct = float(concentration) if concentration is not None else float("nan")
    except (TypeError, ValueError):
        conc_pct = float("nan")
    if np.isfinite(conc_pct) and conc_pct > CONCENTRATION_WARN_PCT:
        name = str(risk.get("ConcentratedTicker") or "ticker")
        alerts.append(
            ("warning", f"Concentration: {name} at {conc_pct:.0f}% > {CONCENTRATION_WARN_PCT:.0f}%.")
        )
    dte = risk.get("DaysToExpiration")
    try:
        dte_n = float(dte) if dte is not None else float("nan")
    except (TypeError, ValueError):
        dte_n = float("nan")
    if np.isfinite(dte_n) and dte_n <= 7:
        alerts.append(("warning", f"Nearest expiry in {dte_n:.0f} DTE — liquidity / pin risk."))
    if not alerts:
        st.success("All monitored risk thresholds within policy.")
        return
    for level, message in alerts:
        if level == "error":
            st.error(message)
        else:
            st.warning(message)


def test_render_layout(ticker: str, spot: float = 0.0, sigma: float = DEFAULT_SIGMA) -> None:
    """Four-pane 2×2 Test Dashboard layout."""
    top_left, top_right = st.columns(2, gap="medium")
    with top_left:
        with st.container(border=True):
            test_render_options_chain(ticker)
    with top_right:
        with st.container(border=True):
            test_render_portfolio_risk_pane()
    bottom_left, bottom_right = st.columns(2, gap="medium")
    with bottom_left:
        with st.container(border=True):
            test_render_trade_ticket(ticker, float(spot or 0.0), float(sigma or DEFAULT_SIGMA))
    with bottom_right:
        with st.container(border=True):
            test_render_alerts_compliance()


def test_add_dashboard_tab(
    tab: Any,
    ticker: str,
    spot: float = 0.0,
    sigma: float = DEFAULT_SIGMA,
) -> None:
    """Entry point for the isolated Test Dashboard tab."""
    with tab:
        st.caption("Sandbox institutional layout — isolated from production tabs.")
        test_render_layout(ticker, spot=spot, sigma=sigma)


@st.cache_data(ttl=120, show_spinner="Running stress scenario…")
def cached_stress_scenario(ticker: str, spot_shift: float, iv_shift: float) -> dict[str, Any]:
    return calculate_stress_scenario(str(ticker), float(spot_shift), float(iv_shift), repo=DATA_REPO)


@st.cache_data(ttl=120, show_spinner="Building stress P&L matrix…")
def cached_stress_pnl_matrix(
    ticker: str,
    spot_shifts: tuple[float, ...],
    iv_shifts: tuple[float, ...],
) -> dict[str, Any]:
    return build_stress_pnl_matrix(
        str(ticker),
        spot_shifts,
        iv_shifts,
        repo=DATA_REPO,
    )


def render_stress_pnl_heatmap(matrix: Mapping[str, Any]) -> None:
    """2D Plotly heatmap of portfolio Net P&L across spot × IV scenarios."""
    spots = list(matrix.get("spot_shifts") or [])
    ivs = list(matrix.get("iv_shifts") or [])
    z = matrix.get("net_pnl")
    if not spots or not ivs or z is None:
        st.info("No stress P&L matrix to plot.")
        return
    z_arr = np.asarray(z, dtype=np.float64)
    if z_arr.size == 0 or not np.any(np.isfinite(z_arr)):
        st.info("No finite Net P&L values for this stress matrix.")
        return
    x_labels = [f"IV {v:+.0f}%" for v in ivs]
    y_labels = [f"Spot {v:+.0f}%" for v in spots]
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z_arr,
                x=x_labels,
                y=y_labels,
                colorscale="RdYlGn",
                zmid=0.0,
                colorbar={"title": "Net P&L"},
                hovertemplate="Spot=%{y}<br>%{x}<br>Net P&L=%{z:,.2f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font={"color": TEXT},
        height=420,
        title={"text": "Scenario Net P&L Heatmap", "x": 0.0, "xanchor": "left"},
        xaxis={"title": "IV shock", "side": "top"},
        yaxis={"title": "Spot shock", "autorange": "reversed"},
        margin={"l": 72, "r": 24, "t": 56, "b": 40},
        uirevision="risk-terminal-pnl-heatmap",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": True},
        key="risk-terminal-pnl-heatmap",
    )


@st.cache_data(ttl=120, show_spinner="Convening Council of Experts…")
def cached_council_debate(ticker: str) -> dict[str, Any]:
    """Cached Council debate summary for Risk Terminal."""
    return run_council_debate(str(ticker), repo=DATA_REPO)


def _council_stance_color(stance: str) -> str:
    label = str(stance or "").strip()
    if label == STANCE_BUY:
        return "#3fb950"
    if label == STANCE_SELL:
        return "#f85149"
    return TEXT


def render_council_expert_card(prediction: Mapping[str, Any]) -> None:
    """Single expert column: name, stance, confidence, justification."""
    name = str(prediction.get("name") or prediction.get("expert_id") or "Expert")
    stance = str(prediction.get("stance") or STANCE_NEUTRAL)
    confidence = float(prediction.get("confidence") or 0.0)
    justification = str(prediction.get("justification") or "")
    st.markdown(f"**Expert {prediction.get('expert_id', '')} — {name}**")
    st.markdown(
        f"<div style='font-size:1.35rem;font-weight:600;color:{_council_stance_color(stance)}'>"
        f"{stance}</div>",
        unsafe_allow_html=True,
    )
    st.metric("Confidence", f"{confidence:.0f} / 100", border=True)
    st.caption(justification)


def render_council_view(summary: Mapping[str, Any]) -> None:
    """Three expert predictions side-by-side plus optional Disagreement Report."""
    st.subheader("Council View")
    st.caption(
        "Council of Experts — Momentum (GEX/velocity/reflexivity), "
        "Mean Reversion (IV Rank/percentile), Volatility Arb (Vanna/Volga/MC)."
    )
    preds = list(summary.get("predictions") or [])
    if not preds:
        experts = summary.get("experts") or {}
        preds = [experts.get("A"), experts.get("B"), experts.get("C")]
        preds = [p for p in preds if isinstance(p, Mapping)]
    if not preds:
        st.info("Council debate unavailable for this ticker.")
        return
    cols = st.columns(min(3, len(preds)))
    for col, pred in zip(cols, preds):
        with col:
            render_council_expert_card(pred if isinstance(pred, Mapping) else {})
    if bool(summary.get("disagreement")) and summary.get("disagreement_report"):
        with st.expander("Disagreement Report", expanded=True):
            st.warning(str(summary.get("disagreement_report")))


def add_risk_terminal_tab(tab: Any, ticker: str) -> None:
    """Institutional Risk Terminal: stress matrix, P&L heatmap, dealer gamma overlay."""
    with tab:
        st.caption(
            "What-If Scenario Engine — re-prices the DataRepository chain under spot / IV shocks."
        )
        st.subheader("Stress Matrix")
        st.caption("Rows = Spot (−10%…+10%). Columns = IV (−20%…+20%).")
        spot_cols = st.columns(3)
        spot_defaults = (-10.0, 0.0, 10.0)
        spot_shifts: list[float] = []
        for i, (col, default) in enumerate(zip(spot_cols, spot_defaults)):
            with col:
                spot_shifts.append(
                    float(
                        st.slider(
                            f"Spot row {i + 1}",
                            min_value=-10.0,
                            max_value=10.0,
                            value=float(default),
                            step=1.0,
                            key=f"risk_term_spot_{i}",
                        )
                    )
                )
        iv_cols = st.columns(3)
        iv_defaults = (-20.0, 0.0, 20.0)
        iv_shifts: list[float] = []
        for i, (col, default) in enumerate(zip(iv_cols, iv_defaults)):
            with col:
                iv_shifts.append(
                    float(
                        st.slider(
                            f"IV col {i + 1}",
                            min_value=-20.0,
                            max_value=20.0,
                            value=float(default),
                            step=1.0,
                            key=f"risk_term_iv_{i}",
                        )
                    )
                )

        focus_cols = st.columns(2)
        with focus_cols[0]:
            focus_spot = float(
                st.slider(
                    "Focus spot shock %",
                    min_value=-10.0,
                    max_value=10.0,
                    value=float(spot_shifts[0]),
                    step=1.0,
                    key="risk_term_focus_spot",
                )
            )
        with focus_cols[1]:
            focus_iv = float(
                st.slider(
                    "Focus IV shock %",
                    min_value=-20.0,
                    max_value=20.0,
                    value=float(iv_shifts[0]),
                    step=1.0,
                    key="risk_term_focus_iv",
                )
            )

        try:
            matrix = cached_stress_pnl_matrix(
                str(ticker),
                tuple(float(v) for v in spot_shifts),
                tuple(float(v) for v in iv_shifts),
            )
            render_stress_pnl_heatmap(matrix)
        except Exception:
            st.error("Could not render the stress P&L heatmap.")
            matrix = {}

        try:
            impact = cached_stress_scenario(str(ticker), focus_spot, focus_iv)
        except Exception:
            impact = {}
            st.error("Could not run the focus stress scenario.")

        st.subheader("Risk Impact")
        if impact:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Δ Delta", _format_greek_metric(impact.get("delta_change")), border=True)
            m2.metric("Δ Gamma", _format_greek_metric(impact.get("gamma_change")), border=True)
            m3.metric("Δ Vega", _format_greek_metric(impact.get("vega_change")), border=True)
            m4.metric("Net P&L", _format_greek_metric(impact.get("net_pnl")), border=True)
            st.caption(
                f"Focus {focus_spot:+.0f}% spot / {focus_iv:+.0f}% IV · "
                f"contracts={impact.get('n_contracts', 0)} · "
                f"spot {impact.get('spot')} → {impact.get('stressed_spot')}"
            )
        else:
            st.info("No DataRepository snapshot available for stress analysis.")

        st.subheader("Dealer Flow Overlay")
        total_gamma = impact.get("total_gamma") if impact else None
        try:
            gamma_val = float(total_gamma) if total_gamma is not None else float("nan")
        except (TypeError, ValueError):
            gamma_val = float("nan")
        st.metric(
            "Dealer Hedging Pressure",
            _format_greek_metric(gamma_val) if np.isfinite(gamma_val) else "—",
            border=True,
        )
        st.caption("Total dealer-signed Gamma (GEX) at the stressed spot.")
        if impact and bool(impact.get("negative_gamma")):
            st.error(
                "CRITICAL WARNING: stress pushes into the Negative Gamma zone — "
                "dealer hedging may amplify moves."
            )

        try:
            council = cached_council_debate(str(ticker))
            render_council_view(council)
        except Exception:
            st.error("Could not render Council View.")

        try:
            render_market_flow_dynamics(str(ticker))
        except Exception:
            st.error("Could not render Market Flow Dynamics.")


@st.cache_data(ttl=120, show_spinner="Computing market velocity…")
def cached_market_velocity(ticker: str) -> dict[str, Any]:
    """Cached Market Velocity / OFI payload for Risk Terminal."""
    return calculate_market_velocity(str(ticker or "").strip().upper(), repo=DATA_REPO)


def market_flow_build_velocity_wave(payload: Mapping[str, Any]) -> go.Figure:
    """Line-area Velocity Wave: blue (selling) → green (buying)."""
    wave = np.asarray(payload.get("velocity_wave_normalized"), dtype=float)
    labels = list(payload.get("wave_labels") or [])
    if wave.size == 0:
        return go.Figure()
    if len(labels) < int(wave.size):
        labels = [str(i) for i in range(int(wave.size))]
    else:
        labels = labels[: int(wave.size)]
    x = list(range(int(wave.size)))
    # Dual fill: positive (buy/green) and negative (sell/blue).
    pos = np.where(wave >= 0, wave, 0.0)
    neg = np.where(wave < 0, wave, 0.0)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pos,
            mode="lines",
            name="Buying velocity",
            line={"color": "#2ecc71", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(46, 204, 113, 0.35)",
            hovertemplate="t=%{customdata}<br>vel=%{y:.3f}<extra>Buy</extra>",
            customdata=labels,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=neg,
            mode="lines",
            name="Selling velocity",
            line={"color": "#3498db", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(52, 152, 219, 0.35)",
            hovertemplate="t=%{customdata}<br>vel=%{y:.3f}<extra>Sell</extra>",
            customdata=labels,
        )
    )
    fig.add_hline(y=0.0, line={"color": TEXT, "width": 1, "dash": "dot"})
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font={"color": TEXT},
        height=320,
        title={"text": "Velocity Wave", "x": 0.0, "xanchor": "left"},
        xaxis_title="Flow path",
        yaxis_title="Normalized velocity (−1 sell → +1 buy)",
        yaxis={"range": [-1.05, 1.05]},
        margin={"l": 48, "r": 24, "t": 48, "b": 48},
        legend={"orientation": "h", "y": 1.08},
        uirevision="market-flow-velocity-wave",
    )
    return fig


def market_flow_build_net_gauge(payload: Mapping[str, Any]) -> go.Figure:
    """Gauge of current Net Market Flow pressure (normalized via OFI proxy scale)."""
    net = float(payload.get("net_flow") or 0.0)
    gross = float(payload.get("gross_flow") or 0.0)
    if gross > 0 and np.isfinite(net) and np.isfinite(gross):
        pressure = float(np.clip(net / gross, -1.0, 1.0))
    else:
        ofi = float(payload.get("ofi") or 0.0)
        pressure = float(np.clip(ofi if np.isfinite(ofi) else 0.0, -1.0, 1.0))
    # Map −1…+1 → 0…100 for the indicator dial.
    value = (pressure + 1.0) * 50.0
    bar = "#2ecc71" if pressure >= 0.15 else ("#3498db" if pressure <= -0.15 else "#f39c12")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "", "valueformat": ".1f"},
            title={"text": f"Net Market Flow · raw={net:.4g}"},
            gauge={
                "axis": {"range": [0, 100], "tickvals": [0, 50, 100], "ticktext": ["Sell", "Flat", "Buy"]},
                "bar": {"color": bar},
                "bgcolor": PANEL_BG,
                "borderwidth": 1,
                "bordercolor": TEXT,
                "steps": [
                    {"range": [0, 35], "color": "#1e2a3a"},
                    {"range": [35, 65], "color": "#2a2a1e"},
                    {"range": [65, 100], "color": "#1e3a2f"},
                ],
                "threshold": {
                    "line": {"color": TEXT, "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT},
        height=280,
        margin={"l": 24, "r": 24, "t": 64, "b": 24},
        uirevision="market-flow-net-gauge",
    )
    return fig


def market_flow_build_correlation_overlay(payload: Mapping[str, Any]) -> go.Figure:
    """Velocity Wave + Gamma Flip reference — toward vs away from flip."""
    wave = np.asarray(payload.get("velocity_wave_normalized"), dtype=float)
    if wave.size == 0:
        return go.Figure()
    x = list(range(int(wave.size)))
    flip = payload.get("gamma_flip")
    spot = payload.get("spot")
    dist = payload.get("flip_distance_pct")
    toward = bool(payload.get("toward_gamma_flip"))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=wave,
            mode="lines",
            name="Velocity Wave",
            line={"color": "#9b59b6", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(155, 89, 182, 0.20)",
        )
    )
    # Horizontal gamma-flip distance marker scaled into velocity space for overlay.
    if dist is not None and np.isfinite(float(dist)):
        flip_level = float(np.clip(float(dist) * 5.0, -1.0, 1.0))
        fig.add_hline(
            y=flip_level,
            line={"color": "#e74c3c", "width": 2, "dash": "dash"},
            annotation_text=(
                f"Gamma Flip Δ={float(dist) * 100:.2f}% · "
                f"{'toward' if toward else 'away'}"
            ),
            annotation_position="top left",
            annotation_font={"color": "#e74c3c"},
        )
    flip_txt = f"{float(flip):.2f}" if flip is not None and np.isfinite(float(flip)) else "—"
    spot_txt = f"{float(spot):.2f}" if spot is not None and np.isfinite(float(spot)) else "—"
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font={"color": TEXT},
        height=300,
        title={
            "text": f"Correlation Overlay — spot {spot_txt} · flip {flip_txt}",
            "x": 0.0,
            "xanchor": "left",
        },
        xaxis_title="Flow path",
        yaxis_title="Velocity / Flip distance (scaled)",
        yaxis={"range": [-1.1, 1.1]},
        margin={"l": 48, "r": 24, "t": 48, "b": 48},
        uirevision="market-flow-correlation-overlay",
    )
    return fig


def render_market_flow_dynamics(ticker: str) -> None:
    """Risk Terminal section: Wave, Net Flow gauge, Gamma Flip correlation overlay."""
    st.subheader("Market Flow Dynamics")
    st.caption(
        "Gross/Net Flow, Flow Velocity, and Order Flow Imbalance "
        "(`calculate_market_velocity`). Sparse bid/ask/volume → chain proxies."
    )
    try:
        payload = cached_market_velocity(str(ticker))
    except Exception:
        st.error("Could not calculate market velocity.")
        return
    if not payload:
        st.info(FLOW_DATA_STALE_MSG)
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gross Flow", f"{float(payload.get('gross_flow') or 0.0):.4g}", border=True)
    m2.metric("Net Flow", f"{float(payload.get('net_flow') or 0.0):.4g}", border=True)
    m3.metric("Flow Velocity", f"{float(payload.get('flow_velocity') or 0.0):.4g}", border=True)
    m4.metric("OFI", f"{float(payload.get('ofi') or 0.0):+.3f}", border=True)
    if bool(payload.get("proxy_used")):
        st.caption("Proxy mode: synthetic volume and/or call/put signed pressure.")

    audit = payload.get("audit") or audit_market_velocity(
        payload.get("velocity_wave"),
        gross_flow=float(payload.get("gross_flow") or 0.0),
    )
    # Auditor gate — halt Wave render when flow stale / zero / unnormalized.
    left, right = st.columns([1.35, 1.0], gap="large")
    with left:
        st.markdown("**Velocity Wave**")
        if not audit.get("ok") or bool(audit.get("halt_render")):
            st.error(str(audit.get("message") or FLOW_DATA_STALE_MSG))
        else:
            try:
                st.plotly_chart(
                    market_flow_build_velocity_wave(payload),
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key="market-flow-velocity-wave",
                )
            except Exception:
                st.error("Could not render the Velocity Wave.")
    with right:
        st.markdown("**Flow Gauge**")
        try:
            st.plotly_chart(
                market_flow_build_net_gauge(payload),
                use_container_width=True,
                config={"displayModeBar": False},
                key="market-flow-net-gauge",
            )
        except Exception:
            st.error("Could not render the Net Market Flow gauge.")

    st.markdown("**Correlation Overlay**")
    st.caption("Velocity Wave vs Gamma Flip distance (Liquidation Waterfall helpers).")
    if bool(payload.get("toward_gamma_flip")):
        st.warning("Flow velocity is moving **toward** the Gamma Flip.")
    elif payload.get("gamma_flip") is not None:
        st.info("Flow velocity is moving **away** from the Gamma Flip.")
    try:
        st.plotly_chart(
            market_flow_build_correlation_overlay(payload),
            use_container_width=True,
            config={"displayModeBar": False},
            key="market-flow-correlation-overlay",
        )
    except Exception:
        st.error("Could not render the Correlation Overlay.")


@st.cache_data(show_spinner=False)
def event_impact_cached_events() -> list[dict[str, str]]:
    """Cached event calendar from ``event_impact_data.json``."""
    return event_impact_load_data()


@st.cache_data(show_spinner=False)
def event_impact_cached_shock(ticker: str, event_date: str) -> dict[str, Any]:
    """Cached shock surface + magnitude for ``ticker`` around ``event_date``."""
    return event_impact_calculate_shock(str(ticker or "").strip().upper(), str(event_date))


def event_impact_render_shock_summary(result: Mapping[str, Any]) -> None:
    """Right-pane Shock Summary card (magnitude + suggested position)."""
    st.subheader("Shock Summary")
    if not result or not result.get("ok"):
        st.info(str((result or {}).get("message") or "Select an event with bracketing snapshots."))
        return
    magnitude = float(result.get("shock_magnitude") or 0.0)
    suggestion = str(result.get("suggestion") or "")
    st.metric("Shock Magnitude", f"{magnitude:.4f}", border=True)
    st.caption(
        f"Σ|ΔΓ| + Σ|Δν| · before=`{result.get('stamp_before')}` · after=`{result.get('stamp_after')}`"
    )
    st.success(suggestion)
    g_mag = float(result.get("gamma_magnitude") or 0.0)
    v_mag = float(result.get("vega_magnitude") or 0.0)
    st.write({"gamma_magnitude": g_mag, "vega_magnitude": v_mag})


def event_impact_render_shock_surface(result: Mapping[str, Any]) -> None:
    """Plotly ``go.Surface`` for Shock Surface with RdBu colorscale."""
    if not result or not result.get("ok"):
        return
    try:
        z_data = reshape_to_surface(result.get("shock_z"))
        if z_data.ndim != 2 or z_data.size == 0:
            raise ValueError("invalid surface shape")
        x = np.asarray(result.get("strike_axis"), dtype=float)
        y = np.asarray(result.get("dte_axis"), dtype=float)
        if z_data.size == 0 or x.size == 0 or y.size == 0:
            st.warning("Shock surface is empty.")
            return
        finite = z_data[np.isfinite(z_data)]
        if finite.size == 0:
            st.warning("Shock surface has no finite values.")
            return
        zmax = float(np.max(np.abs(finite)))
        if zmax <= 0:
            zmax = 1e-6
        surface_trace = _plotly_surface(
            z_data,
            x,
            y,
            colorscale="RdBu",
            cmin=-zmax,
            cmax=zmax,
            colorbar={"title": "ΔΔ"},
            hovertemplate="Strike=%{x:.2f}<br>DTE=%{y:.1f}<br>Shock=%{z:.6f}<extra></extra>",
        )
    except Exception:
        st.error("Surface rendering failed: Invalid data shape.")
        return
    fig = go.Figure(data=[surface_trace])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT},
        height=560,
        title={"text": "Shock Surface (Δ after − Δ before)", "x": 0.0, "xanchor": "left"},
        scene={
            "xaxis_title": "Strike",
            "yaxis_title": "Days to Expiry",
            "zaxis_title": "Delta Shock",
            "bgcolor": PANEL_BG,
            "dragmode": "orbit",
        },
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        uirevision="event-impact-shock",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


@st.cache_data(show_spinner=False)
def regime_lab_cached_reflexivity(ticker: str) -> dict[str, Any]:
    """Cached Market Regime & Reflexivity payload for ``ticker``."""
    return calculate_market_reflexivity(str(ticker or "").strip().upper())


def regime_lab_build_map_figure(map_frame: pd.DataFrame, regime: str) -> go.Figure:
    """3D Regime Map scatter: X=GEX, Y=Vanna, Z=Market Momentum."""
    frame = map_frame.copy() if isinstance(map_frame, pd.DataFrame) else pd.DataFrame()
    if frame.empty or not {"GEX", "Vanna", "Momentum"}.issubset(frame.columns):
        return go.Figure()
    color = frame["Regime"] if "Regime" in frame.columns else regime
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pd.to_numeric(frame["GEX"], errors="coerce"),
                y=pd.to_numeric(frame["Vanna"], errors="coerce"),
                z=pd.to_numeric(frame["Momentum"], errors="coerce"),
                mode="markers",
                marker={
                    "size": 4,
                    "opacity": 0.85,
                    "color": pd.Categorical(color).codes if hasattr(color, "__len__") else 0,
                    "colorscale": "Viridis",
                    "colorbar": {"title": "Regime"},
                },
                text=color.astype(str) if hasattr(color, "astype") else [str(regime)] * len(frame),
                hovertemplate=(
                    "GEX=%{x:.4g}<br>Vanna=%{y:.4g}<br>Momentum=%{z:.4g}"
                    "<br>%{text}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT},
        height=560,
        title={"text": f"Regime Map — {regime}", "x": 0.0, "xanchor": "left"},
        scene={
            "xaxis_title": "GEX",
            "yaxis_title": "Vanna",
            "zaxis_title": "Market Momentum",
            "bgcolor": PANEL_BG,
            "dragmode": "orbit",
        },
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        uirevision="regime-lab-map",
    )
    return fig


def regime_lab_build_cascade_gauge(
    probability: float,
    regime: str,
    *,
    hedge_suggestion: str | None = None,
) -> go.Figure:
    """Cascade Probability gauge; red under Fragile-Bear / Crash-Cascade."""
    p = float(np.clip(float(probability), 0.0, 1.0))
    fragile = str(regime) in (REGIME_FRAGILE_BEAR, REGIME_CRASH_CASCADE)
    bar_color = "#e74c3c" if fragile else "#2ecc71"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p * 100.0,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Cascade Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "bgcolor": PANEL_BG,
                "borderwidth": 1,
                "bordercolor": TEXT,
                "steps": [
                    {"range": [0, 40], "color": "#1e3a2f"},
                    {"range": [40, 70], "color": "#3a341e"},
                    {"range": [70, 100], "color": "#3a1e1e"},
                ],
                "threshold": {
                    "line": {"color": "#e74c3c", "width": 3},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT},
        height=280,
        margin={"l": 24, "r": 24, "t": 48, "b": 24},
        uirevision="regime-lab-gauge",
    )
    if fragile and hedge_suggestion:
        fig.add_annotation(
            text=str(hedge_suggestion),
            x=0.5,
            y=-0.12,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": "#e74c3c", "size": 13},
        )
    return fig


@st.cache_data(show_spinner=False)
def liquidation_waterfall_cached(ticker: str) -> dict[str, Any]:
    """Cached Liquidation Waterfall payload for ``ticker``."""
    return test_calculate_liquidation_waterfall(str(ticker or "").strip().upper())


def liquidation_waterfall_build_heatmap(surface: Mapping[str, Any]) -> go.Figure:
    """3D cascade surface: X=Price, Y=Time/Expiry, Z=cascade score (Liquidation Zones)."""
    price = np.asarray(surface.get("price_axis"), dtype=float)
    expiry = np.asarray(surface.get("expiry_axis"), dtype=float)
    try:
        z_data = reshape_to_surface(surface.get("cascade_z"))
        if z_data.ndim != 2 or z_data.size == 0:
            raise ValueError("invalid surface shape")
        if price.size == 0 or expiry.size == 0:
            return go.Figure()
        colorscale = [
            [0.0, "#0d1b2a"],
            [0.35, "#1b4332"],
            [0.55, "#ca6702"],
            [0.75, "#e63946"],
            [1.0, "#ff1744"],
        ]
        surface_trace = _plotly_surface(
            z_data,
            price,
            expiry,
            colorscale=colorscale,
            cmin=0.0,
            cmax=1.0,
            colorbar={"title": "Cascade"},
            hovertemplate=(
                "Price=%{x:.2f}<br>DTE=%{y:.1f}<br>Cascade=%{z:.3f}<extra></extra>"
            ),
        )
    except Exception:
        st.error("Surface rendering failed: Invalid data shape.")
        return go.Figure()
    fig = go.Figure(data=[surface_trace])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT},
        height=520,
        title={"text": "Liquidation Waterfall — GEX Cascade Surface", "x": 0.0, "xanchor": "left"},
        scene={
            "xaxis_title": "Price",
            "yaxis_title": "Time / Expiry (DTE)",
            "zaxis_title": "Cascade",
            "bgcolor": PANEL_BG,
            "dragmode": "orbit",
        },
        margin={"l": 48, "r": 24, "t": 48, "b": 48},
        uirevision="liquidation-waterfall-heatmap",
    )
    return fig


def liquidation_waterfall_build_stability_gauge(stability: Mapping[str, Any], next_flip: float | None) -> go.Figure:
    """Market Stability gauge — aggressive red as spot approaches the next Gamma Flip."""
    approach = float(stability.get("approach_risk") or 0.0)
    approach = float(np.clip(approach, 0.0, 1.0))
    near = bool(stability.get("near_flip"))
    bar_color = "#ff1744" if near or approach >= 0.7 else ("#f39c12" if approach >= 0.4 else "#2ecc71")
    dist_pct = stability.get("distance_pct")
    try:
        dist_label = f"{float(dist_pct) * 100:.2f}%" if dist_pct is not None and np.isfinite(float(dist_pct)) else "—"
    except (TypeError, ValueError):
        dist_label = "—"
    flip_txt = f"{float(next_flip):.2f}" if next_flip is not None and np.isfinite(float(next_flip)) else "—"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=approach * 100.0,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": f"Flip Approach Risk · next={flip_txt} · Δ={dist_label}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "bgcolor": PANEL_BG,
                "borderwidth": 1,
                "bordercolor": TEXT,
                "steps": [
                    {"range": [0, 40], "color": "#1e3a2f"},
                    {"range": [40, 70], "color": "#3a341e"},
                    {"range": [70, 100], "color": "#3a1e1e"},
                ],
                "threshold": {
                    "line": {"color": "#ff1744", "width": 3},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT},
        height=280,
        margin={"l": 24, "r": 24, "t": 64, "b": 24},
        uirevision="liquidation-waterfall-gauge",
    )
    return fig


def add_liquidation_waterfall_tab(tab: Any, ticker: str) -> None:
    """Liquidation Waterfall: GEX cascade heatmap, stability gauge, Simulate Liquidation."""
    with tab:
        st.caption(
            "Liquidation Waterfall — GEX surface, Gamma Flip cascade, and Liquidation Zones "
            "(`test_calculate_liquidation_waterfall`)."
        )
        try:
            result = liquidation_waterfall_cached(str(ticker))
        except Exception:
            st.error("Could not calculate the liquidation waterfall.")
            return
        if not result or not result.get("ok"):
            st.info(str((result or {}).get("message") or "No chain data for liquidation waterfall."))
            return

        spot = result.get("spot")
        next_flip = result.get("next_gamma_flip")
        stability = result.get("stability") or {}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Spot", f"{float(spot):.2f}" if spot is not None else "—", border=True)
        m2.metric(
            "Next Gamma Flip",
            f"{float(next_flip):.2f}" if next_flip is not None else "—",
            border=True,
        )
        m3.metric("Net GEX", f"{float(result.get('net_gex') or 0.0):.4g}", border=True)
        m4.metric("Liquidation Zones", int(result.get("liquidation_zone_count") or 0), border=True)

        if bool(stability.get("near_flip")):
            st.error(
                "Spot is approaching the next Gamma Flip — expect accelerated dealer hedging "
                f"(within {float(TEST_STABILITY_NEAR_PCT) * 100:.0f}% of flip)."
            )

        left, right = st.columns([1.35, 1.0], gap="large")
        with left:
            st.subheader("GEX Cascade Heatmap")
            surface = result.get("surface") or {}
            audit = result.get("gex_audit") or test_audit_gex_normalized(surface.get("gex_z"))
            if not audit.get("ok") or bool(audit.get("halt_render")):
                st.error(str(audit.get("message") or TEST_LIQUIDATION_UNSTABLE_MSG))
            elif not surface.get("ok"):
                st.info("GEX surface is empty for this chain.")
            else:
                try:
                    fig_hm = liquidation_waterfall_build_heatmap(surface)
                    st.plotly_chart(
                        fig_hm,
                        use_container_width=True,
                        config={"displayModeBar": True},
                        key="liquidation-waterfall-heatmap",
                    )
                    st.caption(
                        f"Bright red = {TEST_LIQUIDATION_ZONE} (high cascade score from negative GEX)."
                    )
                except Exception:
                    st.error("Could not render the Liquidation Waterfall heatmap.")
        with right:
            st.subheader("Market Stability")
            try:
                fig_gauge = liquidation_waterfall_build_stability_gauge(stability, next_flip)
                st.plotly_chart(
                    fig_gauge,
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key="liquidation-waterfall-gauge",
                )
            except Exception:
                st.error("Could not render the Market Stability gauge.")
            dist_pct = stability.get("distance_pct")
            if dist_pct is not None and np.isfinite(float(dist_pct)):
                st.caption(f"Distance to next Gamma Flip = {float(dist_pct) * 100:.2f}% of spot.")

        st.subheader("Positioner")
        st.caption("Simulate how chain Greeks react if price hits the next Gamma Flip zone.")
        if st.button("Simulate Liquidation", key="liquidation_simulate_btn", type="primary"):
            if next_flip is None:
                st.warning("No Gamma Flip detected — cannot simulate liquidation.")
            else:
                try:
                    sim = test_simulate_liquidation_greeks(
                        str(ticker),
                        float(next_flip),
                        spot=float(spot) if spot is not None else None,
                    )
                except Exception:
                    sim = {"ok": False, "message": "Simulation failed."}
                if not sim.get("ok"):
                    st.error(str(sim.get("message") or "Simulation unavailable."))
                else:
                    impact = sim.get("impact") or {}
                    st.success(
                        f"Simulated spot → flip {float(sim.get('flip_price') or next_flip):.2f} "
                        f"({float(sim.get('spot_shift') or 0.0):+.2f}% shock)."
                    )
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Δ Delta", f"{float(impact.get('delta_change') or 0.0):+.4g}", border=True)
                    s2.metric("Δ Gamma", f"{float(impact.get('gamma_change') or 0.0):+.4g}", border=True)
                    s3.metric("Δ Vega", f"{float(impact.get('vega_change') or 0.0):+.4g}", border=True)
                    s4.metric("Net P&L", f"{float(impact.get('net_pnl') or 0.0):+.4g}", border=True)
                    if bool(impact.get("negative_gamma")):
                        st.error("Stressed book enters Negative Gamma — liquidation cascade risk elevated.")

        flips = result.get("gamma_flips")
        if flips is not None and len(np.asarray(flips)) > 0:
            with st.expander("Detected Gamma Flip levels"):
                st.write([float(x) for x in np.asarray(flips, dtype=float).tolist()])


def add_regime_lab_tab(tab: Any, ticker: str) -> None:
    """Regime Lab: 3D Regime Map + Cascade Probability gauge (additive only)."""
    with tab:
        st.caption(
            "Market Regime & Reflexivity — correlates dealer GEX, Vanna, and price momentum "
            "(`calculate_market_reflexivity`)."
        )
        try:
            result = regime_lab_cached_reflexivity(str(ticker))
        except Exception:
            st.error("Could not calculate market reflexivity.")
            return
        if not result or not result.get("ok"):
            st.info(str((result or {}).get("message") or "No chain data for regime analysis."))
            return

        regime = str(result.get("regime") or "Stable-Bull")
        cascade_p = float(result.get("cascade_probability") or 0.0)
        hedge = result.get("hedge_suggestion")
        reflexive = bool(result.get("reflexive_cascade"))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Regime", regime, border=True)
        m2.metric("Net GEX", f"{float(result.get('gex') or 0.0):.4g}", border=True)
        m3.metric("|Vanna|", f"{float(result.get('vanna') or 0.0):.4g}", border=True)
        m4.metric("Momentum", f"{float(result.get('momentum') or 0.0):+.2%}", border=True)

        if reflexive:
            st.error("Reflexive Cascade risk: negative GEX with elevated Vanna.")
        if regime in (REGIME_FRAGILE_BEAR, REGIME_CRASH_CASCADE):
            st.warning(f"Hedge suggestion: {hedge or REGIME_HEDGE_PUT_SPREADS}")

        left, right = st.columns([1.35, 1.0], gap="large")
        with left:
            st.subheader("Regime Map")
            map_frame = result.get("map_frame")
            try:
                fig_map = regime_lab_build_map_figure(
                    map_frame if isinstance(map_frame, pd.DataFrame) else pd.DataFrame(),
                    regime,
                )
                st.plotly_chart(
                    fig_map,
                    use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True},
                    key="regime-lab-map",
                )
            except Exception:
                st.error("Could not render the Regime Map.")
        with right:
            st.subheader("Cascade Predictor")
            try:
                fig_gauge = regime_lab_build_cascade_gauge(
                    cascade_p,
                    regime,
                    hedge_suggestion=str(hedge) if hedge else None,
                )
                st.plotly_chart(
                    fig_gauge,
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key="regime-lab-gauge",
                )
            except Exception:
                st.error("Could not render the Cascade Probability gauge.")
            st.caption(f"Cascade Probability = {cascade_p:.3f} (bounded [0, 1]).")


@st.cache_data(ttl=120, show_spinner="Scanning social sentiment…")
def sentiment_lab_cached_alpha(ticker: str) -> dict[str, Any]:
    """Cached Sentiment Lab payload (`calculate_sentiment_alpha`)."""
    return calculate_sentiment_alpha(str(ticker or "").strip().upper(), force_stub=True)


def sentiment_lab_build_gauge(score_01: float) -> go.Figure:
    """Sentiment Gauge: Bearish (0) → Bullish (1)."""
    s = float(np.clip(float(score_01), 0.0, 1.0))
    if s < 0.4:
        bar = "#e74c3c"
    elif s > 0.6:
        bar = "#2ecc71"
    else:
        bar = "#f1c40f"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=s,
            number={"valueformat": ".3f"},
            title={"text": "Sentiment (Bearish → Bullish)"},
            gauge={
                "axis": {"range": [0, 1], "tickvals": [0, 0.5, 1], "ticktext": ["Bearish", "Neutral", "Bullish"]},
                "bar": {"color": bar},
                "bgcolor": PANEL_BG,
                "borderwidth": 1,
                "bordercolor": TEXT,
                "steps": [
                    {"range": [0.0, 0.4], "color": "#3a1e1e"},
                    {"range": [0.4, 0.6], "color": "#3a341e"},
                    {"range": [0.6, 1.0], "color": "#1e3a2f"},
                ],
                "threshold": {
                    "line": {"color": TEXT, "width": 2},
                    "thickness": 0.75,
                    "value": 0.5,
                },
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT},
        height=280,
        margin={"l": 24, "r": 24, "t": 48, "b": 24},
        uirevision="sentiment-lab-gauge",
    )
    return fig


def sentiment_lab_build_buzz_meter(buzz: float, mention_count: int) -> go.Figure:
    """Buzz Meter: mention-volume intensity in [0, 1]."""
    b = float(np.clip(float(buzz), 0.0, 1.0))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=b * 100.0,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": f"Buzz Meter ({int(mention_count)} mentions)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#3498db"},
                "bgcolor": PANEL_BG,
                "borderwidth": 1,
                "bordercolor": TEXT,
                "steps": [
                    {"range": [0, 33], "color": "#1e2a3a"},
                    {"range": [33, 66], "color": "#1e334a"},
                    {"range": [66, 100], "color": "#1e4060"},
                ],
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT},
        height=280,
        margin={"l": 24, "r": 24, "t": 48, "b": 24},
        uirevision="sentiment-lab-buzz",
    )
    return fig


def sentiment_lab_build_correlation_chart(
    sentiment: Any,
    realized_vol: Any,
) -> go.Figure:
    """Scatter: Sentiment Score vs Realized Volatility."""
    s = np.asarray(sentiment, dtype=float).ravel()
    v = np.asarray(realized_vol, dtype=float).ravel()
    n = min(s.size, v.size)
    s, v = s[:n], v[:n]
    mask = np.isfinite(s) & np.isfinite(v)
    fig = go.Figure()
    if int(mask.sum()) == 0:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=DARK_BG,
            plot_bgcolor=PANEL_BG,
            font={"color": TEXT},
            height=360,
            title="Sentiment Score vs Realized Volatility",
            margin={"l": 48, "r": 16, "t": 48, "b": 48},
        )
        return fig
    fig.add_trace(
        go.Scatter(
            x=s[mask],
            y=v[mask],
            mode="markers",
            marker={"size": 9, "color": "#58a6ff", "opacity": 0.85},
            name="Observations",
            hovertemplate="Sentiment=%{x:.3f}<br>RV=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font={"color": TEXT},
        height=360,
        title={"text": "Sentiment Score vs Realized Volatility", "x": 0.0, "xanchor": "left"},
        xaxis_title="Sentiment Score [0, 1]",
        yaxis_title="Realized Volatility",
        margin={"l": 48, "r": 16, "t": 48, "b": 48},
        uirevision="sentiment-lab-corr",
    )
    return fig


def add_sentiment_lab_tab(tab: Any, ticker: str) -> None:
    """Sentiment Lab: gauge, buzz meter, sentiment–vol correlation (additive only)."""
    with tab:
        st.caption(
            "Social Sentiment Lab — Reddit + 4chan scan + VADER scoring "
            "(`calculate_sentiment_alpha` / Sentiment Factor for Monte Carlo). "
            "Offline stubs when Reddit credentials or 4chan HTTP are unavailable."
        )
        try:
            result = sentiment_lab_cached_alpha(str(ticker))
        except Exception:
            st.error("Could not calculate social sentiment.")
            return
        if not result or not result.get("ok"):
            st.info(str((result or {}).get("message") or "No sentiment data."))
            return

        score_01 = float(result.get("score_01") or 0.5)
        buzz = float(result.get("buzz") or 0.0)
        alpha_01 = float(result.get("alpha_01") or 0.5)
        mentions = int(result.get("mention_count") or 0)
        audit = result.get("audit") or audit_sentiment_normalized(result.get("scores_01"))
        reddit = result.get("reddit") or {}
        fourchan = result.get("fourchan") or {}

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Combined Sentiment", f"{score_01:.3f}", border=True)
        m2.metric("Combined Buzz", f"{buzz:.3f}", border=True)
        m3.metric("Alpha [0,1]", f"{alpha_01:.3f}", border=True)
        m4.metric("Mentions", mentions, border=True)

        s1, s2 = st.columns(2, gap="large")
        with s1:
            st.subheader("Reddit")
            r_score = float(reddit.get("score_01") or 0.5)
            r_buzz = float(reddit.get("buzz") or 0.0)
            r_n = int(reddit.get("mention_count") or 0)
            r1, r2, r3 = st.columns(3)
            r1.metric("Sentiment", f"{r_score:.3f}", border=True)
            r2.metric("Buzz", f"{r_buzz:.3f}", border=True)
            r3.metric("Mentions", r_n, border=True)
            st.caption(f"source = {reddit.get('source', 'empty')}")
            if bool(reddit.get("uoa_flag")):
                st.warning("Reddit: UOA / unusual flow mentions detected.")
        with s2:
            st.subheader("4chan")
            f_score = float(fourchan.get("score_01") or 0.5)
            f_buzz = float(fourchan.get("buzz") or 0.0)
            f_n = int(fourchan.get("mention_count") or 0)
            f1, f2, f3 = st.columns(3)
            f1.metric("Sentiment", f"{f_score:.3f}", border=True)
            f2.metric("Buzz", f"{f_buzz:.3f}", border=True)
            f3.metric("Mentions", f_n, border=True)
            boards = (result.get("sources") or {}).get("fourchan", {}).get("boards") or ["biz", "pol"]
            st.caption(f"source = {fourchan.get('source', 'empty')} · boards = {', '.join(map(str, boards))}")
            if bool(fourchan.get("uoa_flag")):
                st.warning("4chan: UOA / unusual flow mentions detected.")

        if bool(result.get("uoa_flag")):
            st.warning("Novelty: Unusual Options Activity mentions detected across social feeds.")
        factor = result.get("sentiment_factor") or {}
        if isinstance(factor, dict) and factor:
            st.caption(
                f"MC Sentiment Factor = {float(factor.get('sentiment_factor', score_01)):.3f} · "
                f"drift_bias = {float(factor.get('drift_bias', 0.0)):+.4f} · "
                f"vol_mult = {float(factor.get('vol_mult', 1.0)):.3f} · "
                f"combined source = {result.get('source', 'stub')}"
            )

        left, mid, right = st.columns([1.0, 1.0, 1.2], gap="large")
        with left:
            st.subheader("Sentiment Gauge")
            if not audit.get("ok") or bool(audit.get("halt_render")):
                st.error(str(audit.get("message") or SENTIMENT_UNSTABLE_MSG))
            else:
                try:
                    fig_g = sentiment_lab_build_gauge(score_01)
                    st.plotly_chart(
                        fig_g,
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key="sentiment-lab-gauge",
                    )
                    st.caption("Combined (mention-weighted Reddit + 4chan). Gauge gated on [0, 1] audit.")
                except Exception:
                    st.error("Could not render the Sentiment Gauge.")
        with mid:
            st.subheader("Buzz Meter")
            try:
                fig_b = sentiment_lab_build_buzz_meter(buzz, mentions)
                st.plotly_chart(
                    fig_b,
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key="sentiment-lab-buzz",
                )
                st.caption(
                    f"Reddit buzz={r_buzz:.3f} ({r_n}) · 4chan buzz={f_buzz:.3f} ({f_n})"
                )
            except Exception:
                st.error("Could not render the Buzz Meter.")
        with right:
            st.subheader("Correlation")
            try:
                fig_c = sentiment_lab_build_correlation_chart(
                    result.get("chart_sentiment"),
                    result.get("chart_realized_vol"),
                )
                st.plotly_chart(
                    fig_c,
                    use_container_width=True,
                    config={"displayModeBar": True},
                    key="sentiment-lab-corr",
                )
                st.caption(
                    f"corr(sentiment, return) = {float(result.get('corr_price') or 0.0):+.3f} · "
                    f"corr(sentiment, RV) = {float(result.get('corr_vol') or 0.0):+.3f}"
                )
            except Exception:
                st.error("Could not render the Sentiment vs Realized Volatility chart.")


@st.cache_data(ttl=120, show_spinner="Computing microstructure…")
def microstructure_lab_cached(ticker: str) -> dict[str, Any]:
    """Cached Microstructure Lab payload (VPIN + OBI)."""
    return calculate_microstructure_lab(str(ticker or "").strip().upper(), repo=DATA_REPO)


def microstructure_lab_build_vpin_gauge(vpin: float) -> go.Figure:
    """VPIN Gauge — Toxicity Risk in [0, 1]."""
    s = float(np.clip(float(vpin), 0.0, 1.0))
    if s > VPIN_HIGH_THRESHOLD:
        bar = "#e74c3c"
    elif s > 0.4:
        bar = "#f1c40f"
    else:
        bar = "#2ecc71"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=s,
            number={"valueformat": ".3f"},
            title={"text": "VPIN — Toxicity Risk"},
            gauge={
                "axis": {
                    "range": [0, 1],
                    "tickvals": [0, 0.3, VPIN_HIGH_THRESHOLD, 1],
                    "ticktext": ["Calm", "Elevated", "Toxic", "Max"],
                },
                "bar": {"color": bar},
                "bgcolor": PANEL_BG,
                "borderwidth": 1,
                "bordercolor": TEXT,
                "steps": [
                    {"range": [0.0, 0.4], "color": "#1e3a2f"},
                    {"range": [0.4, VPIN_HIGH_THRESHOLD], "color": "#3a341e"},
                    {"range": [VPIN_HIGH_THRESHOLD, 1.0], "color": "#3a1e1e"},
                ],
                "threshold": {
                    "line": {"color": "#e74c3c", "width": 2},
                    "thickness": 0.75,
                    "value": VPIN_HIGH_THRESHOLD,
                },
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT},
        height=300,
        margin={"l": 24, "r": 24, "t": 48, "b": 24},
        uirevision="microstructure-vpin-gauge",
    )
    return fig


def microstructure_lab_build_obi_heatmap(heatmap: Mapping[str, Any]) -> go.Figure:
    """Order Book Imbalance Heatmap — top 5 levels; Blue=Buy, Red=Sell."""
    z = np.asarray(heatmap.get("z"), dtype=float)
    x = list(heatmap.get("x") or [f"L{i + 1}" for i in range(5)])
    y = list(heatmap.get("y") or ["Bid (Buy)", "Ask (Sell)"])
    peak = float(np.nanmax(np.abs(z))) if z.size and np.any(np.isfinite(z)) else 1.0
    if not np.isfinite(peak) or peak <= 0:
        peak = 1.0
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale="RdBu",
                zmid=0.0,
                zmin=-peak,
                zmax=peak,
                colorbar={"title": "Size (+Bid / −Ask)"},
                hovertemplate="Level=%{x}<br>Side=%{y}<br>Value=%{z:.2f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font={"color": TEXT},
        height=320,
        title={"text": "Order Book Imbalance — Top 5 Levels", "x": 0.0, "xanchor": "left"},
        xaxis_title="Book Level",
        yaxis_title="Side",
        margin={"l": 80, "r": 24, "t": 48, "b": 48},
        uirevision="microstructure-obi-heatmap",
    )
    return fig


def add_microstructure_lab_tab(tab: Any, ticker: str) -> None:
    """Microstructure Lab: VPIN gauge, OBI heatmap, Flow Toxicity Alert (additive only)."""
    with tab:
        st.caption(
            "Microstructure Lab — VPIN (equal-volume buckets) + Order Book Imbalance "
            "(`calculate_vpin` / `calculate_order_book_imbalance`). "
            "Sparse L2/tick → chain volume and size-decay proxies."
        )
        try:
            result = microstructure_lab_cached(str(ticker))
        except Exception:
            st.error("Could not calculate microstructure metrics.")
            return
        if not result:
            st.info("No microstructure data.")
            return

        vpin_block = result.get("vpin") or {}
        obi_block = result.get("obi") or {}
        trap = result.get("liquidity_trap") or {}
        vpin_audit = vpin_block.get("audit") or audit_vpin(vpin_block.get("vpin"))
        obi_audit = obi_block.get("audit") or audit_obi_heatmap(obi_block.get("heatmap"))

        vpin = vpin_block.get("vpin")
        try:
            vpin_f = float(vpin) if vpin is not None else float("nan")
        except (TypeError, ValueError):
            vpin_f = float("nan")
        imb = float(obi_block.get("imbalance") or 0.0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "VPIN",
            f"{vpin_f:.3f}" if np.isfinite(vpin_f) else "—",
            border=True,
        )
        m2.metric("OBI (agg)", f"{imb:+.3f}", border=True)
        m3.metric(
            "One-side share",
            f"{float(trap.get('one_side_share') or 0.0):.1%}",
            border=True,
        )
        m4.metric(
            "Buckets",
            int(vpin_block.get("n_buckets") or 0),
            border=True,
        )

        # Flow Toxicity Alert — Liquidity Trap when OBI extreme + VPIN high.
        if bool(trap.get("liquidity_trap")) or (
            detect_liquidity_trap(vpin_f, imb).get("liquidity_trap")
        ):
            st.error(LIQUIDITY_TRAP_MSG)
        if bool(vpin_audit.get("toxicity_alert")) or bool(result.get("toxicity_alert")):
            st.error(TOXICITY_ALERT_MSG)
        if bool(result.get("high_adverse_selection")) or (
            np.isfinite(vpin_f) and vpin_f > VPIN_HIGH_THRESHOLD
        ):
            st.warning(HIGH_ADVERSE_SELECTION_MSG)

        if bool(vpin_block.get("proxy_used")) or bool(obi_block.get("proxy_used")):
            st.caption(
                "Proxy path active: L2/tick sparse — using chain volume, "
                "Lee–Ready / call-put classification, and size-decay book levels."
            )

        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("VPIN Gauge — Toxicity Risk")
            if bool(vpin_audit.get("halt_render")) or not vpin_audit.get("ok"):
                st.error(str(vpin_audit.get("message") or TOXICITY_ALERT_MSG))
            elif not np.isfinite(vpin_f):
                st.info("VPIN unavailable.")
            else:
                try:
                    fig_g = microstructure_lab_build_vpin_gauge(vpin_f)
                    st.plotly_chart(
                        fig_g,
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key="microstructure-vpin-gauge",
                    )
                    st.caption(
                        f"Equal-volume buckets = {int(vpin_block.get('n_buckets') or 0)} · "
                        f"V = {float(vpin_block.get('bucket_volume') or 0.0):.1f} · "
                        f"alert > {VPIN_HIGH_THRESHOLD:.1f}"
                    )
                except Exception:
                    st.error("Could not render the VPIN Gauge.")
        with right:
            st.subheader("Order Book Imbalance Heatmap")
            if bool(obi_audit.get("halt_render")) or not obi_audit.get("ok"):
                st.error(str(obi_audit.get("message") or OBI_HEATMAP_UNSTABLE_MSG))
            else:
                try:
                    heat = obi_block.get("heatmap") or {}
                    fig_h = microstructure_lab_build_obi_heatmap(heat)
                    st.plotly_chart(
                        fig_h,
                        use_container_width=True,
                        config={"displayModeBar": True},
                        key="microstructure-obi-heatmap",
                    )
                    st.caption(
                        f"Top {int(heat.get('n_levels') or 5)} levels · "
                        f"Σbid={float(obi_block.get('bid_total') or 0.0):.0f} · "
                        f"Σask={float(obi_block.get('ask_total') or 0.0):.0f} · "
                        f"extreme one-side > {OBI_EXTREME_THRESHOLD:.0%}"
                    )
                except Exception:
                    st.error("Could not render the Order Book Imbalance Heatmap.")


def add_event_impact_lab_tab(tab: Any, ticker: str) -> None:
    """Event Impact Lab: event dropdown, shock summary, RdBu shock surface."""
    with tab:
        st.caption(
            "Event-driven Greek shock — Delta surface after − before, "
            "magnitude from Σ|ΔΓ| + Σ|Δν| (`event_impact_*` helpers)."
        )
        left, right = st.columns([1, 1], gap="large")
        events = event_impact_cached_events()
        if not events:
            with left:
                st.warning("No events found in `event_impact_data.json`.")
            return
        labels = [f"{row['date']} — {row['event']}" for row in events]
        with left:
            choice = st.selectbox(
                "Event",
                options=labels,
                index=0,
                key="event_impact_event_select",
            )
            selected = events[labels.index(choice)] if choice in labels else events[0]
            st.write({"date": selected["date"], "event": selected["event"]})
        try:
            result = event_impact_cached_shock(str(ticker), str(selected["date"]))
        except Exception:
            result = {"ok": False, "message": "Could not calculate event shock."}
        with right:
            event_impact_render_shock_summary(result)
        event_impact_render_shock_surface(result)


def _timelapse_history_token(ticker: str) -> str:
    """Fingerprint ``data_history/{TICKER}_*.json`` so cache invalidates on new snapshots."""
    root = Path(__file__).resolve().parent / "data_history"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(ticker or "").strip().upper()) or "_"
    parts: list[str] = []
    try:
        paths = sorted(p for p in root.glob(f"{safe}_*.json") if p.is_file())
    except OSError:
        paths = []
    for path in paths:
        try:
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
        except OSError:
            parts.append(path.name)
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


@st.cache_data(show_spinner=False)
def _cached_timelapse_frames(ticker: str, history_token: str) -> list[dict[str, Any]]:
    """Cached Market Timelapse frames (delta_surface grids + labels)."""
    _ = history_token
    return load_all_snapshots(str(ticker or "").strip().upper())


def _timelapse_surface_as_lists(
    z_data: Any,
    x: Any,
    y: Any,
    *,
    z_min: float,
    z_max: float,
) -> go.Surface:
    """Build a Surface whose x/y/z are JSON lists (not numpy ``bdata``).

    Plotly.js animate interpolates binary ``bdata`` frames incorrectly, so every
    timelapse frame must serialize as nested lists. Construct a fresh Surface;
    ``update()`` on a numpy-backed trace keeps the binary encoding.
    """
    built = _plotly_surface(
        z_data,
        x,
        y,
        cmin=z_min,
        cmax=z_max,
        colorscale="Viridis",
        colorbar={"title": "Delta"},
        hovertemplate="Strike=%{x:.2f}<br>DTE=%{y:.1f}<br>Delta=%{z:.4f}<extra></extra>",
    )
    z_list = np.array(built.z, dtype=np.float64, copy=True).tolist()
    payload: dict[str, Any] = {
        "z": z_list,
        "cmin": z_min,
        "cmax": z_max,
        "colorscale": "Viridis",
        "colorbar": {"title": "Delta"},
        "hovertemplate": "Strike=%{x:.2f}<br>DTE=%{y:.1f}<br>Delta=%{z:.4f}<extra></extra>",
    }
    if built.x is not None and built.y is not None:
        payload["x"] = np.array(built.x, dtype=np.float64, copy=True).reshape(-1).tolist()
        payload["y"] = np.array(built.y, dtype=np.float64, copy=True).reshape(-1).tolist()
    return go.Surface(**payload)


def build_market_timelapse_figure(frames: list[dict[str, Any]]) -> go.Figure:
    """Plotly Surface with ``frames``, Play/Pause, and timestamp-labeled slider.

    Z color/axis locked via shared min/max across all frames (no flicker).
    Frame names are integer indices (colon-free) so Plotly animate/slider match.
    """
    if not frames:
        return go.Figure()
    z_min, z_max = timelapse_shared_z_range(item["z"] for item in frames)

    def _axis_span(key: str) -> tuple[float, float]:
        lo, hi = float("inf"), float("-inf")
        for item in frames:
            arr = np.asarray(item.get(key), dtype=np.float64).reshape(-1)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            lo = min(lo, float(np.min(finite)))
            hi = max(hi, float(np.max(finite)))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            pad = 1.0 if not np.isfinite(lo) else (1.0 if lo == 0.0 else abs(lo) * 0.05)
            lo = 0.0 if not np.isfinite(lo) else lo - pad
            hi = 1.0 if not np.isfinite(hi) else hi + pad
        return lo, hi

    x_lo, x_hi = _axis_span("x")
    y_lo, y_hi = _axis_span("y")

    def _surface(item: dict[str, Any]) -> go.Surface:
        try:
            z_data = reshape_to_surface(item["z"])
            if z_data.ndim != 2 or z_data.size == 0:
                raise ValueError("invalid surface shape")
            return _timelapse_surface_as_lists(z_data, item["x"], item["y"], z_min=z_min, z_max=z_max)
        except Exception:
            st.error("Surface rendering failed: Invalid data shape.")
            return go.Surface(z=[[None]])

    frame_names = [str(i) for i in range(len(frames))]
    fig = go.Figure(data=[_surface(frames[0])])
    fig.frames = [
        go.Frame(
            data=[_surface(item)],
            name=name,
            traces=[0],
            layout={"sliders": [{"active": idx}]},
        )
        for idx, (name, item) in enumerate(zip(frame_names, frames))
    ]
    animate_step = {
        "frame": {"duration": 0, "redraw": True},
        "mode": "immediate",
        "transition": {"duration": 0},
    }
    slider_steps = [
        {
            "args": [[name], animate_step],
            "label": str(item["label"]),
            "method": "animate",
        }
        for name, item in zip(frame_names, frames)
    ]
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT},
        height=600,
        title={"text": "Market Timelapse (Delta surface)", "x": 0.0, "xanchor": "left"},
        scene={
            "xaxis_title": "Strike",
            "yaxis_title": "Days to Expiry",
            "zaxis_title": "Delta",
            "xaxis": {"range": [x_lo, x_hi]},
            "yaxis": {"range": [y_lo, y_hi]},
            "zaxis": {"range": [z_min, z_max]},
            "bgcolor": PANEL_BG,
            "dragmode": "orbit",
        },
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        uirevision="market-timelapse-camera",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "showactive": False,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.12,
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            frame_names,
                            {
                                "frame": {"duration": 400, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "Snapshot ",
                    "visible": True,
                    "xanchor": "right",
                },
                "pad": {"b": 10, "t": 36},
                "len": 0.9,
                "x": 0.05,
                "y": 0,
                "steps": slider_steps,
            }
        ],
    )
    return fig


def add_market_timelapse_tab(tab: Any, ticker: str) -> None:
    """Market Timelapse tab: animate historical delta_surface grids."""
    with tab:
        st.caption(
            f"Historical morph of `{TIMELAPSE_SURFACE_KEY}` from `data_history/` "
            "(shared Strike × DTE mesh so delta morphs in place; color/Z locked)."
        )
        symbol = str(ticker or "").strip().upper()
        history_token = _timelapse_history_token(symbol)
        try:
            frames = _cached_timelapse_frames(symbol, history_token)
        except Exception:
            st.error("Could not load Market Timelapse snapshots.")
            return
        if not frames:
            st.info("No historical delta surfaces found for this ticker in data_history/.")
            return
        z_min, z_max = timelapse_shared_z_range(item["z"] for item in frames)
        x0 = np.asarray(frames[0]["x"], dtype=np.float64).reshape(-1)
        y0 = np.asarray(frames[0]["y"], dtype=np.float64).reshape(-1)
        st.write(
            {
                "frames": len(frames),
                "surface_key": TIMELAPSE_SURFACE_KEY,
                "z_min": z_min,
                "z_max": z_max,
                "shared_strike": [float(np.nanmin(x0)), float(np.nanmax(x0))],
                "shared_dte": [float(np.nanmin(y0)), float(np.nanmax(y0))],
                "labels": [item["label"] for item in frames],
            }
        )
        try:
            fig = build_market_timelapse_figure(frames)
            st.plotly_chart(
                fig,
                width="stretch",
                height=600,
                theme=None,
                config={"displayModeBar": True, "scrollZoom": True},
                key=f"market-timelapse-{history_token[:16]}",
            )
        except Exception:
            st.error("Could not render the Market Timelapse animation.")


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()
    _apply_theme()
    repo = _get_repo()
    # Global header: LIVE / HISTORICAL badge (visible on every tab).
    _hdr_left, _hdr_right = st.columns([5, 2])
    with _hdr_right:
        render_global_data_status(repo)
    shown_alerts: set[str] = set()
    try:
        health = repo.verify_connection("SPY")
    except Exception as exc:
        health = {"ok": False, "status_code": None, "message": str(exc), "token_missing": False}
    if isinstance(health, dict) and health.get("token_missing"):
        st.error("API Token Missing")
        shown_alerts.add("API Token Missing")
    elif not _api_status_ok(health):
        code = None
        if isinstance(health, dict):
            code = health.get("status_code")
        if code is None or code == "":
            code = "RequestFailed"
        msg = f"API Error: {code}"
        st.warning(msg)
        shown_alerts.add(msg)
        st.error("API Service Unavailable")

    ticker, strike, expiry, current_price, option_type, _refresh, download_slot = _sidebar_inputs()
    ticker = ticker.strip().upper() or DEFAULT_TICKER
    try:
        ui_msg = repo.ui_error_message()
    except Exception:
        ui_msg = None
    if ui_msg and ui_msg not in shown_alerts:
        if "Token Missing" in ui_msg:
            st.error(ui_msg)
        else:
            st.warning(ui_msg)
        shown_alerts.add(ui_msg)
    alert = st.session_state.pop("ingest_alert", None)
    if isinstance(alert, str) and alert and alert not in shown_alerts:
        if "Token Missing" in alert:
            st.error(alert)
        else:
            st.warning(alert)

    section = _render_section_switcher()
    if section == "Charts":
        render_charts_section(ticker)
        return

    st.title("Contract Greeks")
    st.caption("Price-domain Delta, Gamma, Theta, Vega, and Rho from `generate_greek_curve`.")

    if current_price <= 0:
        st.info("Waiting for a live Current Price for this ticker.")
        return
    if strike <= 0:
        st.info("Select or enter a strike to draw the curves.")
        return

    current_iv = _contract_iv(ticker, expiry, float(strike), option_type)
    sigma = current_iv if current_iv is not None and current_iv > 0 else DEFAULT_SIGMA
    heston_selected = str(st.session_state.get("pricing_model")) == "Heston"
    model = _model_from_state()
    v0, kappa, theta, heston_sigma, rho = _heston_from_state()
    revision = str(
        _sim_fingerprint(
            ticker, float(strike), expiry, option_type, current_price, sigma, model, v0, kappa, theta, heston_sigma, rho
        )
    )
    frame = _load_curve(
        ticker,
        float(strike),
        expiry,
        current_price,
        option_type,
        sigma,
        model,
        v0,
        kappa,
        theta,
        heston_sigma,
        rho,
    )

    if frame is None:
        return

    if heston_selected and (
        "Delta_Heston" not in frame.columns
        or not pd.to_numeric(frame.get("Delta_Heston"), errors="coerce").notna().any()
    ):
        st.warning("Heston traces are missing for this run. Click Refresh, or wait for the Greeks compute to finish.")

    if current_iv is None:
        frame = frame.copy()
        frame["IV"] = pd.NA
    else:
        frame = frame.copy()
        frame["IV"] = current_iv

    render_sidebar_download(download_slot, frame, ticker, strike, expiry)

    header = st.container()
    with header:
        render_metric_header(frame, current_price, ticker, strike, expiry, current_iv, heston_selected)

    gamma_flip: float | None = None
    try:
        gex_report = cached_gex_outlook(ticker)
        raw_flip = gex_report.get("GammaFlip")
        if raw_flip is not None and np.isfinite(float(raw_flip)) and float(raw_flip) > 0:
            gamma_flip = float(raw_flip)
            if current_price > 0 and abs(gamma_flip - current_price) / current_price <= GAMMA_FLIP_NEAR_PCT:
                st.warning("Market is at a Gamma Flip point—expect high volatility.")
    except Exception:
        gamma_flip = None

    if st.session_state.get("show_second_order"):
        try:
            render_vanna_volga_chart(
                frame,
                float(strike),
                expiry,
                float(sigma),
                option_type,
                v0,
                kappa,
                theta,
                heston_sigma,
                rho,
                revision=revision,
            )
        except Exception:
            st.error("Could not render the Vanna/Volga chart.")

    # Unified surface viewport: former "3D Surface" / "Volatility Surface" tabs
    # consolidate into Integrated Risk View. Time Machine + Portfolio unchanged.
    tab_labels = [
        "Greeks",
        "Advanced Metrics",
        "Time-Sensitivity",
        "Time Machine",
        "Pro Metrics",
        "Market Analyst",
        "Portfolio & Risk",
        "Integrated Risk View",
        "Test Dashboard",
        "Market Timelapse",
        "Risk Terminal",
        "Event Impact Lab",
        "Regime Lab",
        "Liquidation Waterfall",
        "Sentiment Lab",
        "Microstructure Lab",
    ]
    (
        greeks_tab,
        advanced_tab,
        time_tab,
        machine_tab,
        pro_tab,
        analyst_tab,
        portfolio_tab,
        integrated_tab,
        test_dashboard_tab,
        timelapse_tab,
        risk_terminal_tab,
        event_impact_tab,
        regime_lab_tab,
        liquidation_waterfall_tab,
        sentiment_lab_tab,
        microstructure_lab_tab,
    ) = st.tabs(
        tab_labels,
        on_change="rerun",
        key="main_view_tabs",
        default="Greeks",
    )
    with greeks_tab:
        for pair in CHART_ROWS:
            cols = st.columns(len(pair), gap="medium")
            for column_name, col in zip(pair, cols):
                try:
                    render_greek_chart(frame, column_name, col, revision=revision, gamma_flip=gamma_flip)
                except Exception:
                    col.error(f"Could not render the {column_name} chart.")
        with st.expander("Greeks table"):
            st.dataframe(frame, use_container_width=True, hide_index=True)

    if advanced_tab.open:
        with advanced_tab:
            view_3d = st.checkbox("View 3D Surface", key="advanced_view_3d")
            if view_3d:
                greek = st.radio(
                    "Surface Greek",
                    options=("Charm", "Speed", "Color"),
                    horizontal=True,
                    key="advanced_surface_greek",
                )
                dte = max((expiry - date.today()).days, 1)
                expiry_range = tuple(sorted({float(d) for d in range(7, 91, 7)} | {float(dte)}))
                prices = tuple(
                    float(p)
                    for p in pd.to_numeric(frame["Price"], errors="coerce").dropna().tolist()[::4]
                )
                try:
                    surface = _load_greek_surface(
                        ticker,
                        float(strike),
                        expiry_range,
                        prices,
                        float(sigma),
                        option_type,
                        model,
                        v0,
                        kappa,
                        theta,
                        heston_sigma,
                        rho,
                        str(greek),
                    )
                    render_advanced_greek_surface(surface, str(greek))
                except Exception:
                    st.error(f"Could not render the {greek} 3D surface.")
            else:
                try:
                    render_gamma_theta_ratio_chart(frame)
                except Exception:
                    st.error("Could not render the Gamma/Theta Ratio chart.")
                try:
                    render_speed_chart(frame)
                except Exception:
                    st.error("Could not render the Risk Acceleration chart.")
                try:
                    render_color_chart(frame)
                except Exception:
                    st.error("Could not render the Gamma Decay chart.")

    if time_tab.open:
        with time_tab:
            dte = max((expiry - date.today()).days, 1)
            near = list(range(1, min(dte, 14) + 1))
            far = list(range(21, dte + 1, 7))
            expiry_range = tuple(sorted({float(x) for x in near + far + [dte] if x > 0}))
            term_model = model
            if heston_selected:
                st.caption("Heston term structure is opt-in so the 2D Heston curves on Greeks can render first.")
                if st.button("Build Heston term structure", key="build_heston_term"):
                    st.session_state["run_heston_term"] = True
                if not st.session_state.get("run_heston_term"):
                    term_model = MODEL_BLACK_SCHOLES
            try:
                term = _load_term_structure(
                    ticker,
                    float(strike),
                    float(current_price),
                    expiry_range,
                    float(sigma),
                    option_type,
                    term_model,
                    v0,
                    kappa,
                    theta,
                    heston_sigma,
                    rho,
                )
                render_delta_term_structure(term)
            except Exception:
                st.error("Could not render the Time-Sensitivity chart.")

    if machine_tab.open:
        add_time_machine_tab(machine_tab, ticker, strike=float(strike), expiry=expiry)

    if pro_tab.open:
        add_pro_metrics_tab(
            pro_tab,
            ticker,
            float(strike),
            expiry,
            frame,
            float(sigma),
            option_type,
            model,
            v0,
            kappa,
            theta,
            heston_sigma,
            rho,
        )

    if analyst_tab.open:
        add_market_analyst_tab(analyst_tab, ticker, float(current_price), expiry)

    if portfolio_tab.open:
        add_portfolio_risk_tab(portfolio_tab)

    if integrated_tab.open:
        with integrated_tab:
            st.caption(
                "Unified risk surface (replaces standalone 3D Gamma / Vol Surface tabs). "
                "Z = mean of enabled Volatility / Gamma / |Delta| layers."
            )
            layer_cols = st.columns(3)
            with layer_cols[0]:
                toggle_gamma = st.toggle("Toggle Gamma Layer", value=True, key="risk_toggle_gamma")
            with layer_cols[1]:
                toggle_delta = st.toggle("Toggle Delta Layer", value=True, key="risk_toggle_delta")
            with layer_cols[2]:
                toggle_vol = st.toggle("Toggle Volatility Layer", value=True, key="risk_toggle_vol")
            dte = max((expiry - date.today()).days, 1)
            expiry_range = tuple(sorted({float(d) for d in range(7, 91, 7)} | {float(dte)}))
            prices = tuple(
                float(p)
                for p in pd.to_numeric(frame["Price"], errors="coerce").dropna().tolist()[::4]
            )
            risk_model = MODEL_BLACK_SCHOLES if heston_selected else model
            if heston_selected:
                st.caption("Integrated Risk View uses Black–Scholes grids so Heston 2D curves stay snappy.")
            try:
                risk_frame = cached_integrated_risk_frame(
                    ticker,
                    float(strike),
                    expiry_range,
                    prices,
                    float(sigma),
                    option_type,
                    risk_model,
                    v0,
                    kappa,
                    theta,
                    heston_sigma,
                    rho,
                )
                surface = generate_integrated_risk_surface(
                    risk_frame.copy(),
                    include_gamma=bool(toggle_gamma),
                    include_delta=bool(toggle_delta),
                    include_vol=bool(toggle_vol),
                )
                try:
                    diagnosis = cached_diagnose_surface_anomaly(
                        str(ticker),
                        tuple(float(p) for p in prices) if prices else tuple(),
                        tuple(float(d) for d in expiry_range) if expiry_range else tuple(),
                    )
                except Exception:
                    diagnosis = None
                render_integrated_risk_surface(surface, diagnosis=diagnosis)
                # Risk Dashboard — weights documented in quant_engine (equal thirds by default).
                total_score = compute_total_risk_score(
                    risk_frame.copy(),
                    weight_delta=RISK_WEIGHT_DELTA,
                    weight_gamma=RISK_WEIGHT_GAMMA,
                    weight_vega=RISK_WEIGHT_VEGA,
                )
                st.metric("Total Risk Score", f"{total_score:.3f}", border=True)
                st.caption(
                    f"Score = {RISK_WEIGHT_DELTA:.2f}·|Δ| + {RISK_WEIGHT_GAMMA:.2f}·|Γ| + "
                    f"{RISK_WEIGHT_VEGA:.2f}·|Vega| (equal weights; mean |greek| / ref → [0,1]). "
                    f"Alert threshold = {RISK_SCORE_ALERT_THRESHOLD:.2f}."
                )
                if total_score > RISK_SCORE_ALERT_THRESHOLD:
                    st.error(
                        f"Risk Alert: Total Risk Score {total_score:.3f} exceeds "
                        f"threshold {RISK_SCORE_ALERT_THRESHOLD:.2f}."
                    )
                elif total_score > RISK_SCORE_ALERT_THRESHOLD * 0.8:
                    st.warning(
                        f"Elevated risk: Total Risk Score {total_score:.3f} is near "
                        f"threshold {RISK_SCORE_ALERT_THRESHOLD:.2f}."
                    )
            except Exception:
                st.error("Could not render the Integrated Risk View surface.")

    if test_dashboard_tab.open:
        test_add_dashboard_tab(
            test_dashboard_tab,
            ticker,
            spot=float(current_price or 0.0),
            sigma=float(sigma) if sigma is not None else float(DEFAULT_SIGMA),
        )

    if timelapse_tab.open:
        add_market_timelapse_tab(timelapse_tab, ticker)

    if risk_terminal_tab.open:
        add_risk_terminal_tab(risk_terminal_tab, ticker)

    if event_impact_tab.open:
        add_event_impact_lab_tab(event_impact_tab, ticker)

    if regime_lab_tab.open:
        add_regime_lab_tab(regime_lab_tab, ticker)

    if liquidation_waterfall_tab.open:
        add_liquidation_waterfall_tab(liquidation_waterfall_tab, ticker)

    if sentiment_lab_tab.open:
        add_sentiment_lab_tab(sentiment_lab_tab, ticker)

    if microstructure_lab_tab.open:
        add_microstructure_lab_tab(microstructure_lab_tab, ticker)


if __name__ == "__main__":
    main()
