"""DatAi Streamlit dashboard: Price vs Delta / Gamma / Theta / Rho."""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_ingestion import (
    get_available_expirations,
)
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
    analyze_event_outlook,
    analyze_gex_outlook,
    analyze_market_sentiment,
    analyze_volatility_risk_outlook,
    calculate_charm,
    calculate_color,
    calculate_gamma_theta_ratio,
    calculate_portfolio_risk,
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
    generate_pro_surface_data,
    generate_vol_surface_data,
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
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def _repo_chain(ticker: str, expiry: Any = None) -> pd.DataFrame:
    repo = _get_repo()
    symbol = str(ticker or "").strip().upper()
    if not repo.is_fresh(symbol):
        return repo.get_data(symbol, expiry)
    return repo.get_data(symbol, expiry)


def _reset_time_machine_keys(ticker: str) -> None:
    symbol = str(ticker or "").strip().upper()
    for key in (f"tm_start_{symbol}", f"tm_end_{symbol}", f"tm_scrub_{symbol}"):
        if key in st.session_state:
            del st.session_state[key]


def _capture_snapshot(ticker: str, expiry: Any = None) -> None:
    symbol = str(ticker or "").strip().upper() or DEFAULT_TICKER
    data = _repo_chain(symbol, expiry)
    _get_repo().save_to_cache(symbol, data)
    _reset_time_machine_keys(symbol)
    try:
        cached_delta_drift.clear()
        cached_snapshot_vol_surface.clear()
    except Exception:
        pass
    st.session_state.snapshot_saved = True


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

        if st.button("Capture Snapshot", width="stretch"):
            _capture_snapshot(ticker_norm, expiry)
            st.success("Snapshot saved!")
            st.rerun()
        if st.session_state.pop("snapshot_saved", False):
            st.success("Snapshot saved!")

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


def render_gamma_surface(frame: pd.DataFrame) -> None:
    required = {"Price", "DaysToExpiry", "Gamma"}
    if frame.empty or not required.issubset(frame.columns):
        st.info("No gamma surface data to plot.")
        return
    pivot = frame.pivot_table(index="DaysToExpiry", columns="Price", values="Gamma", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    fig = go.Figure(
        data=[
            go.Surface(
                x=pivot.columns.to_numpy(dtype=float),
                y=pivot.index.to_numpy(dtype=float),
                z=pivot.to_numpy(dtype=float),
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
            go.Surface(
                x=pivot.columns.to_numpy(dtype=float),
                y=pivot.index.to_numpy(dtype=float),
                z=z,
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


def render_delta_drift_surface(frame: pd.DataFrame) -> None:
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
    z = np.array(pivot.to_numpy(dtype=float), copy=True)
    z[~np.isfinite(z)] = np.nan
    fig = go.Figure(
        data=[
            go.Surface(
                x=pivot.columns.to_numpy(dtype=float),
                y=pivot.index.to_numpy(dtype=float),
                z=z,
                colorscale="RdBu",
                reversescale=True,
                cmid=0,
                colorbar={"title": "Delta drift"},
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
        st.info(frame)
        return
    required = {"Strike", "DaysToExpiry", "IV"}
    if frame is None or frame.empty or not required.issubset(frame.columns):
        st.info("Insufficient Data")
        return
    plot = frame.loc[:, ["Strike", "DaysToExpiry", "IV"]].copy()
    plot["Strike"] = pd.to_numeric(plot["Strike"], errors="coerce")
    plot["DaysToExpiry"] = pd.to_numeric(plot["DaysToExpiry"], errors="coerce")
    plot["IV"] = pd.to_numeric(plot["IV"], errors="coerce")
    plot = plot.replace([np.inf, -np.inf], np.nan).dropna(subset=["Strike", "DaysToExpiry"])
    if plot.empty or not plot["IV"].notna().any():
        st.info("Insufficient Data")
        return
    pivot = plot.pivot_table(index="DaysToExpiry", columns="Strike", values="IV", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    z = np.array(pivot.to_numpy(dtype=float), copy=True)
    z[~np.isfinite(z)] = np.nan
    fig = go.Figure(
        data=[
            go.Surface(
                x=pivot.columns.to_numpy(dtype=float),
                y=pivot.index.to_numpy(dtype=float),
                z=z,
                colorscale="Viridis",
                colorbar={"title": "IV"},
                hovertemplate="Strike=%{x:.2f}<br>DTE=%{y:.1f}<br>IV=%{z:.4f}<extra></extra>",
            )
        ]
    )
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


def add_time_machine_tab(tab: Any, ticker: str) -> None:
    with tab:
        snapshots = _get_repo().get_available_snapshots(ticker)
        if not snapshots:
            st.info("No snapshots in data_history/ for this ticker.")
            return
        st.subheader("Delta drift")
        start_col, end_col = st.columns(2)
        with start_col:
            start = st.selectbox("Start Snapshot", snapshots, index=0, key=f"tm_start_{ticker}")
        with end_col:
            end = st.selectbox("End Snapshot", snapshots, index=len(snapshots) - 1, key=f"tm_end_{ticker}")
        payload_a = _get_repo()._snapshot_payload(ticker, start) or {}
        payload_b = _get_repo()._snapshot_payload(ticker, end) or {}
        st.write(
            {
                "Start Snapshot keys": list(payload_a.keys()),
                "End Snapshot keys": list(payload_b.keys()),
            }
        )
        if "delta_surface" not in payload_a or "delta_surface" not in payload_b:
            st.error("Snapshot does not contain delta data!")
        else:
            try:
                drift = cached_delta_drift(str(ticker), str(start), str(end))
                render_delta_drift_surface(drift)
            except Exception:
                st.error("Could not render the Delta Drift surface.")
        st.subheader("Surface evolution")
        stamp = st.select_slider("Snapshot", options=snapshots, value=snapshots[-1], key=f"tm_scrub_{ticker}")
        try:
            surface = cached_snapshot_vol_surface(str(ticker), str(stamp))
            render_time_machine_vol_surface(surface, str(stamp))
        except Exception:
            st.error("Could not render the snapshot volatility surface.")


def render_advanced_greek_surface(frame: pd.DataFrame, greek: str) -> None:
    required = {"Price", "DaysToExpiry", "Value"}
    if frame.empty or not required.issubset(frame.columns):
        st.info(f"No {greek} surface data to plot.")
        return
    pivot = frame.pivot_table(index="DaysToExpiry", columns="Price", values="Value", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    fig = go.Figure(
        data=[
            go.Surface(
                x=pivot.columns.to_numpy(dtype=float),
                y=pivot.index.to_numpy(dtype=float),
                z=pivot.to_numpy(dtype=float),
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


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()
    _apply_theme()
    if not _api_status_ok(_get_repo().get_api_status()):
        st.error("API Service Unavailable")

    ticker, strike, expiry, current_price, option_type, _refresh, download_slot = _sidebar_inputs()
    ticker = ticker.strip().upper() or DEFAULT_TICKER

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

    tab_labels = [
        "Greeks",
        "Advanced Metrics",
        "3D Surface",
        "Time-Sensitivity",
        "Time Machine",
        "Volatility Surface",
        "Pro Metrics",
        "Market Analyst",
        "Portfolio & Risk",
    ]
    (
        greeks_tab,
        advanced_tab,
        surface_tab,
        time_tab,
        machine_tab,
        vol_tab,
        pro_tab,
        analyst_tab,
        portfolio_tab,
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

    if surface_tab.open:
        with surface_tab:
            dte = max((expiry - date.today()).days, 1)
            expiry_range = tuple(sorted({float(d) for d in range(7, 91, 7)} | {float(dte)}))
            prices = tuple(
                float(p)
                for p in pd.to_numeric(frame["Price"], errors="coerce").dropna().tolist()
            )
            surface_model = model
            if heston_selected:
                st.caption("Heston 3D is opt-in so the 2D Heston curves on Greeks can render first.")
                if st.button("Build Heston gamma surface", key="build_heston_gamma_surface"):
                    st.session_state["run_heston_gamma_surface"] = True
                if not st.session_state.get("run_heston_gamma_surface"):
                    surface_model = MODEL_BLACK_SCHOLES
            try:
                surface = _load_gamma_surface(
                    ticker,
                    float(strike),
                    expiry_range,
                    prices,
                    float(sigma),
                    option_type,
                    surface_model,
                    v0,
                    kappa,
                    theta,
                    heston_sigma,
                    rho,
                )
                render_gamma_surface(surface)
            except Exception:
                st.error("Could not render the 3D Gamma surface.")

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
        add_time_machine_tab(machine_tab, ticker)

    if vol_tab.open:
        with vol_tab:
            try:
                vol_surface = _load_vol_surface(ticker)
                render_vol_surface(vol_surface)
            except Exception as error:
                st.exception(error)

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


if __name__ == "__main__":
    main()
