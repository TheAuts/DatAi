"""Underlying price charts for the DatAi Streamlit dashboard.

Data-agnostic: every figure builder takes a Date/OHLCV DataFrame. Heavy fetches
are cached with ``st.cache_data``. Dark-theme Plotly plus an optional TradingView
widget. Missing OHLCV degrades to an info message — no synthetic prices.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from data_ingestion import fetch_ohlcv, normalize_ohlcv_frame

DARK_BG = "#0e1117"
PANEL_BG = "#161b22"
GRID = "#30363d"
TEXT = "#e6edf3"
UP_COLOR = "#26a69a"
DOWN_COLOR = "#ef5350"
WICK_COLOR = "#8b949e"

INTERVAL_LABELS = {
    "D": "Daily",
    "W": "Weekly",
    "60": "1 hour",
    "15": "15 minutes",
}
TV_INTERVAL = {"D": "D", "W": "W", "60": "60", "15": "15"}


def _empty_ohlcv_message(ticker: str) -> None:
    symbol = str(ticker or "").strip().upper() or "this ticker"
    st.info(
        f"No underlying price history for {symbol}. "
        "Charts use MarketData candles when the API returns OHLCV; "
        "they stay empty when that history is missing."
    )


@st.cache_data(ttl=300, show_spinner="Loading underlying price history…")
def cached_ohlcv(ticker: str, resolution: str, countback: int) -> pd.DataFrame:
    """Cached MarketData OHLCV for the Charts section."""
    frame = fetch_ohlcv(ticker, resolution=resolution, countback=int(countback))
    return normalize_ohlcv_frame(frame)


def tradingview_widget_html(ticker: str, interval: str = "D", height: int = 620) -> str:
    """Return embed HTML for a dark TradingView advanced chart widget."""
    symbol = str(ticker or "SPY").strip().upper() or "SPY"
    tv_interval = TV_INTERVAL.get(str(interval).strip().upper(), "D")
    safe_symbol = "".join(ch for ch in symbol if ch.isalnum() or ch in {".", "-", "_", ":"})
    if not safe_symbol:
        safe_symbol = "SPY"
    # Classic widget: OHLC candles + volume study, dark background matching DatAi.
    return f"""
<div class="tradingview-widget-container" style="height:{int(height)}px;width:100%;">
  <div id="datiai_tv_chart" style="height:100%;width:100%;"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget({{
    "autosize": true,
    "symbol": "{safe_symbol}",
    "interval": "{tv_interval}",
    "timezone": "America/New_York",
    "theme": "dark",
    "style": "1",
    "locale": "en",
    "backgroundColor": "{DARK_BG}",
    "gridColor": "rgba(48, 54, 61, 0.55)",
    "hide_legend": false,
    "allow_symbol_change": false,
    "save_image": false,
    "withdateranges": true,
    "hide_side_toolbar": false,
    "studies": ["STD;Volume"],
    "container_id": "datiai_tv_chart"
  }});
  </script>
</div>
"""


def build_candlestick_figure(frame: pd.DataFrame, ticker: str = "") -> go.Figure | None:
    """TradingView-like OHLC candles with a volume pane. DataFrame in, Figure out."""
    data = normalize_ohlcv_frame(frame)
    if data.empty:
        return None
    dates = pd.to_datetime(data["Date"], utc=True, errors="coerce")
    opens = data["Open"].to_numpy(dtype="float64")
    highs = data["High"].to_numpy(dtype="float64")
    lows = data["Low"].to_numpy(dtype="float64")
    closes = data["Close"].to_numpy(dtype="float64")
    volumes = data["Volume"].to_numpy(dtype="float64")
    up = closes >= opens
    vol_colors = np.where(up, UP_COLOR, DOWN_COLOR)
    symbol = str(ticker or "").strip().upper()
    title = f"{symbol} candlestick" if symbol else "Underlying candlestick"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.74, 0.26],
    )
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="OHLC",
            increasing={"line": {"color": UP_COLOR}, "fillcolor": UP_COLOR},
            decreasing={"line": {"color": DOWN_COLOR}, "fillcolor": DOWN_COLOR},
            whiskerwidth=0.6,
            hoverinfo="x+y",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=dates,
            y=volumes,
            name="Volume",
            marker_color=vol_colors.tolist(),
            marker_line_width=0,
            hovertemplate="Volume=%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font={"color": TEXT},
        height=620,
        title={"text": title, "x": 0.0, "xanchor": "left"},
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        showlegend=False,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        uirevision="ohlc-candles",
    )
    fig.update_xaxes(gridcolor=GRID, showgrid=True, zeroline=False, row=1, col=1)
    fig.update_xaxes(gridcolor=GRID, showgrid=True, zeroline=False, row=2, col=1, title_text="Date")
    fig.update_yaxes(
        title_text="Price",
        gridcolor=GRID,
        showgrid=True,
        zeroline=False,
        side="right",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Volume",
        gridcolor=GRID,
        showgrid=True,
        zeroline=False,
        side="right",
        row=2,
        col=1,
    )
    return fig


def _box_mesh(boxes: np.ndarray, color: str, name: str) -> go.Mesh3d:
    """Build one Mesh3d from rows of ``(x0,x1,y0,y1,z0,z1)``."""
    if boxes.size == 0:
        return go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], name=name)
    n = int(boxes.shape[0])
    x0, x1, y0, y1, z0, z1 = (boxes[:, i] for i in range(6))
    corners_x = np.stack([x0, x1, x1, x0, x0, x1, x1, x0], axis=1)
    corners_y = np.stack([y0, y0, y1, y1, y0, y0, y1, y1], axis=1)
    corners_z = np.stack([z0, z0, z0, z0, z1, z1, z1, z1], axis=1)
    xs = corners_x.reshape(-1)
    ys = corners_y.reshape(-1)
    zs = corners_z.reshape(-1)
    face = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 2, 6],
            [3, 6, 7],
            [1, 2, 6],
            [1, 6, 5],
            [0, 3, 7],
            [0, 7, 4],
        ],
        dtype=int,
    )
    offsets = (np.arange(n, dtype=int) * 8)[:, None, None]
    tri = face[None, :, :] + offsets
    i = tri[:, :, 0].reshape(-1)
    j = tri[:, :, 1].reshape(-1)
    k = tri[:, :, 2].reshape(-1)
    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=0.92,
        flatshading=True,
        name=name,
        hoverinfo="skip",
        lighting={"ambient": 0.55, "diffuse": 0.7, "specular": 0.15},
    )


def build_volume_3d_candlestick_figure(frame: pd.DataFrame, ticker: str = "") -> go.Figure | None:
    """3D candlesticks: x = time, y = price (OHLC body/wick), z = volume."""
    data = normalize_ohlcv_frame(frame)
    if data.empty:
        return None
    n = int(len(data))
    xs = np.arange(n, dtype=float)
    opens = data["Open"].to_numpy(dtype="float64")
    highs = data["High"].to_numpy(dtype="float64")
    lows = data["Low"].to_numpy(dtype="float64")
    closes = data["Close"].to_numpy(dtype="float64")
    volumes = data["Volume"].to_numpy(dtype="float64")
    volumes = np.where(np.isfinite(volumes) & (volumes > 0.0), volumes, 0.0)
    up = closes >= opens
    body_lo = np.minimum(opens, closes)
    body_hi = np.maximum(opens, closes)
    flat = (body_hi - body_lo) < 1e-12
    pad = np.maximum(np.abs(body_lo) * 1e-4, 1e-6)
    body_hi = np.where(flat, body_hi + pad, body_hi)
    z_top = volumes.copy()
    # Keep z = volume; a unit stub so zero-volume bars remain pickable.
    z_top = np.where(z_top > 0.0, z_top, 1.0)
    half = 0.32
    x0 = xs - half
    x1 = xs + half
    z0 = np.zeros(n, dtype=float)

    up_boxes = np.column_stack([x0[up], x1[up], body_lo[up], body_hi[up], z0[up], z_top[up]])
    down_boxes = np.column_stack(
        [x0[~up], x1[~up], body_lo[~up], body_hi[~up], z0[~up], z_top[~up]]
    )

    traces: list[Any] = [
        _box_mesh(up_boxes, UP_COLOR, "Up body"),
        _box_mesh(down_boxes, DOWN_COLOR, "Down body"),
    ]

    # Wicks at the mid-volume plane of each candle (NaN breaks the line).
    z_mid = 0.5 * z_top
    nan_col = np.full(n, np.nan)
    traces.append(
        go.Scatter3d(
            x=np.column_stack([xs, xs, nan_col]).reshape(-1),
            y=np.column_stack([lows, highs, nan_col]).reshape(-1),
            z=np.column_stack([z_mid, z_mid, nan_col]).reshape(-1),
            mode="lines",
            line={"color": WICK_COLOR, "width": 3},
            name="Wick",
            hoverinfo="skip",
        )
    )

    dates = pd.to_datetime(data["Date"], utc=True, errors="coerce")
    labels = dates.dt.strftime("%Y-%m-%d %H:%M").fillna("").astype(str).tolist()
    hover = go.Scatter3d(
        x=xs,
        y=0.5 * (opens + closes),
        z=volumes,
        mode="markers",
        marker={"size": 3, "opacity": 0.01, "color": np.where(up, UP_COLOR, DOWN_COLOR).tolist()},
        name="OHLCV",
        customdata=np.column_stack([opens, highs, lows, closes, volumes]),
        hovertemplate=(
            "%{text}<br>"
            "Open=%{customdata[0]:.4f}<br>"
            "High=%{customdata[1]:.4f}<br>"
            "Low=%{customdata[2]:.4f}<br>"
            "Close=%{customdata[3]:.4f}<br>"
            "Volume=%{customdata[4]:,.0f}"
            "<extra></extra>"
        ),
        text=labels,
    )
    traces.append(hover)

    symbol = str(ticker or "").strip().upper()
    title = f"{symbol} 3D candlestick (z = volume)" if symbol else "3D candlestick (z = volume)"
    tick_n = min(8, n)
    tick_idx = np.unique(np.linspace(0, n - 1, tick_n, dtype=int)) if n else np.array([], dtype=int)
    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT},
        height=680,
        title={"text": title, "x": 0.0, "xanchor": "left"},
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        showlegend=False,
        uirevision="ohlc-3d-volume",
        scene={
            "xaxis": {
                "title": "Time",
                "backgroundcolor": PANEL_BG,
                "gridcolor": GRID,
                "tickmode": "array",
                "tickvals": tick_idx.tolist(),
                "ticktext": [labels[int(i)] for i in tick_idx],
            },
            "yaxis": {
                "title": "Price",
                "backgroundcolor": PANEL_BG,
                "gridcolor": GRID,
            },
            "zaxis": {
                "title": "Volume",
                "backgroundcolor": PANEL_BG,
                "gridcolor": GRID,
            },
            "bgcolor": PANEL_BG,
            "camera": {"eye": {"x": 1.7, "y": 1.55, "z": 0.85}},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.4, "y": 0.85, "z": 0.7},
        },
    )
    return fig


def _render_missing_or_figure(frame: pd.DataFrame, ticker: str, kind: str) -> None:
    data = normalize_ohlcv_frame(frame)
    if data.empty:
        _empty_ohlcv_message(ticker)
        return
    fig = (
        build_candlestick_figure(data, ticker)
        if kind == "2d"
        else build_volume_3d_candlestick_figure(data, ticker)
    )
    if fig is None:
        _empty_ohlcv_message(ticker)
        return
    key = "charts-ohlc-2d" if kind == "2d" else "charts-ohlc-3d"
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": True, "scrollZoom": True},
        key=key,
    )


def render_charts_section(ticker: str) -> None:
    """Charts section: TradingView-like candles and a 3D volume candlestick."""
    symbol = str(ticker or "").strip().upper() or "SPY"
    st.title("Underlying Charts")
    st.caption(
        "Price history for the selected underlying. "
        "Candlestick uses a TradingView widget when the embed loads; "
        "Market-data candles and the 3D view use DatAi OHLCV."
    )
    controls = st.columns((2, 2, 3), gap="medium")
    with controls[0]:
        interval = st.selectbox(
            "Candle interval",
            options=list(INTERVAL_LABELS.keys()),
            format_func=lambda value: INTERVAL_LABELS.get(str(value), str(value)),
            key="chart_interval",
            help="Bar size for both the candlestick and the 3D volume view.",
        )
    with controls[1]:
        lookback = st.selectbox(
            "Lookback bars",
            options=(60, 120, 180, 252),
            key="chart_lookback",
            help="How many MarketData candles to request for this underlying.",
        )
    with controls[2]:
        st.caption("Data source for the 2D pane")
        source = st.radio(
            "Candlestick source",
            options=("TradingView", "Market data"),
            horizontal=True,
            key="chart_source",
            help=(
                "TradingView: embedded live chart (dark theme, volume). "
                "Market data: DatAi OHLCV candles from MarketData."
            ),
            label_visibility="collapsed",
        )

    frame = cached_ohlcv(symbol, str(interval), int(lookback))
    candle_tab, volume3d_tab = st.tabs(
        ("Candlestick", "3D Volume"),
        on_change="rerun",
        key="charts_view_tabs",
        default="Candlestick",
    )
    with candle_tab:
        if str(source) == "TradingView":
            st.caption(
                f"TradingView candlestick for {symbol}. "
                "Switch to Market data to plot DatAi OHLCV (OHLC + volume pane)."
            )
            components.html(
                tradingview_widget_html(symbol, str(interval), height=620),
                height=640,
                scrolling=False,
            )
        else:
            st.caption(
                f"DatAi MarketData OHLCV for {symbol}: "
                "OHLC candles with a volume pane, dark theme."
            )
            _render_missing_or_figure(frame, symbol, "2d")

    if volume3d_tab.open:
        with volume3d_tab:
            st.caption(
                "Each candle is a 3D bar: time on X, price (open/close body, high/low wick) on Y, "
                "volume on Z. Drag to rotate."
            )
            _render_missing_or_figure(frame, symbol, "3d")
            if not normalize_ohlcv_frame(frame).empty:
                with st.expander("OHLCV table"):
                    st.dataframe(
                        normalize_ohlcv_frame(frame),
                        use_container_width=True,
                        hide_index=True,
                    )
