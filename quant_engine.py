"""Black–Scholes–Merton Greeks for the DatAi MVP."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

T_MIN = 1e-4
SIGMA_MIN = 1e-8
DAYS_PER_YEAR = 365.0
DEFAULT_RATE = 0.045
DEFAULT_SIGMA = 0.20
DEFAULT_HESTON_V0 = 0.04
DEFAULT_HESTON_KAPPA = 2.0
DEFAULT_HESTON_THETA = 0.04
DEFAULT_HESTON_SIGMA = 0.40
DEFAULT_HESTON_RHO = -0.70
HESTON_PATHS = 4096
HESTON_STEPS = 600
CURVE_RESOLUTION = 201
SMOOTH_WINDOW = 5
VOL_SURFACE_IV_MIN = 0.01
VOL_SURFACE_IV_MAX = 3.00
VOL_SURFACE_STRIKE_POINTS = 80
VOL_SURFACE_DTE_POINTS = 60
VOL_SURFACE_GAUSS_SIGMA = 1.0
# Preferred snapshot key first; aliases are read-only (saves always use vol_surface).
VOL_SURFACE_PAYLOAD_KEYS: tuple[str, ...] = ("vol_surface", "volatility_surface", "iv_surface")
SKEW_BEARISH_THRESHOLD = 0.10
TERM_SHORT_DTE = 21.0
TERM_LONG_DTE = 45.0
GAMMA_SIGNIFICANCE = 0.50
OPTIONS_CONTRACT_SIZE = 100
GAMMA_FLIP_NEAR_PCT = 0.01
EVENT_SHORT_DTE = 15.0
EVENT_MEDIUM_DTE = 60.0
EVENT_LIQUID_DTE = 30.0
EVENT_IV_RATIO = 1.2
EVENT_INSUFFICIENT_COVERAGE = "Insufficient expiration coverage for term structure analysis"
CONCENTRATION_WARN_PCT = 30.0
_LOG = logging.getLogger("datiai.quant")
MODEL_BLACK_SCHOLES = "black-scholes"
MODEL_HESTON = "heston"


def prepare_plotly_surface_xyz(
    z_data: Any,
    x: Any = None,
    y: Any = None,
    *,
    rows: int | None = None,
    cols: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, str | None]:
    """Coerce ``z`` to 2D and gate Plotly Surface ``x``/``y`` axes.

    Returns ``(z2d, x_or_none, y_or_none, warning_or_none)``. Axes are only
    returned when ``len(x) == z.shape[1]`` and ``len(y) == z.shape[0]``; otherwise
    they are omitted so Plotly can fall back to default indices.
    """
    x_arr = None if x is None else np.asarray(x, dtype=float).reshape(-1)
    y_arr = None if y is None else np.asarray(y, dtype=float).reshape(-1)
    inferred_rows = int(rows) if rows is not None else (int(len(y_arr)) if y_arr is not None else None)
    inferred_cols = int(cols) if cols is not None else (int(len(x_arr)) if x_arr is not None else None)
    raw = np.asarray(
        [np.nan if v is None else v for v in z_data] if isinstance(z_data, (list, tuple)) else z_data,
        dtype=np.float64,
    )
    if raw.ndim == 2:
        z2d = raw
    elif inferred_rows is not None and inferred_cols is not None and raw.size == inferred_rows * inferred_cols:
        z2d = raw.reshape(inferred_rows, inferred_cols)
    elif raw.ndim == 1 and inferred_cols is not None and inferred_cols > 0 and raw.size % inferred_cols == 0:
        z2d = raw.reshape(-1, inferred_cols)
    elif raw.ndim == 1 and inferred_rows is not None and inferred_rows > 0 and raw.size % inferred_rows == 0:
        z2d = raw.reshape(inferred_rows, -1)
    elif raw.size == 0:
        z2d = np.empty((0, 0), dtype=np.float64)
    else:
        side = int(np.sqrt(raw.size))
        if side > 0 and side * side == raw.size:
            z2d = raw.reshape(side, side)
        else:
            z2d = raw.reshape(1, -1) if raw.size else np.empty((0, 0), dtype=np.float64)
    if z2d.ndim != 2:
        z2d = z2d.reshape(z2d.shape[0], -1) if z2d.size else np.empty((0, 0), dtype=np.float64)
    warning: str | None = None
    if x_arr is not None and y_arr is not None and len(x_arr) == z2d.shape[1] and len(y_arr) == z2d.shape[0]:
        return z2d.astype(float, copy=False), x_arr, y_arr, None
    if x_arr is not None or y_arr is not None:
        warning = (
            f"Surface axis length mismatch "
            f"(x={None if x_arr is None else len(x_arr)}, "
            f"y={None if y_arr is None else len(y_arr)}, "
            f"z={z2d.shape}); omitting x/y and using default indices."
        )
    return z2d.astype(float, copy=False), None, None, warning


def minmax_normalize_01(values: Any) -> np.ndarray:
    """Min-max normalize ``values`` to ``[0, 1]`` independently.

    Always copies. All-NaN or non-finite-only input → zeros. Zero range
    (constant finite values) → zeros on finite cells (NaN preserved as NaN).
    """
    arr = np.array(values, dtype=np.float64, copy=True)
    out = np.zeros(arr.shape, dtype=np.float64)
    if arr.size == 0:
        return out
    mask = np.isfinite(arr)
    if not np.any(mask):
        return out
    finite = arr[mask]
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    span = hi - lo
    if not np.isfinite(span) or span <= 0.0:
        out = np.array(arr, dtype=np.float64, copy=True)
        out[mask] = 0.0
        out[~mask] = np.nan
        return out
    out = np.array(arr, dtype=np.float64, copy=True)
    out[mask] = (finite - lo) / span
    out[~mask] = np.nan
    return out


_VIRIDIS_RGB: tuple[tuple[int, int, int], ...] = (
    (68, 1, 84),
    (72, 40, 120),
    (62, 74, 137),
    (49, 104, 142),
    (38, 130, 142),
    (31, 158, 137),
    (53, 183, 121),
    (110, 206, 88),
    (181, 222, 43),
    (253, 231, 37),
)


def _viridis_rgba(t: float, alpha: float) -> str:
    """Sample Viridis at ``t`` ∈ [0, 1] with opacity ``alpha`` ∈ [0, 1]."""
    if not np.isfinite(t):
        return "rgba(0,0,0,0)"
    t_clip = float(np.clip(t, 0.0, 1.0))
    a = float(np.clip(alpha if np.isfinite(alpha) else 0.0, 0.0, 1.0))
    # Floor so the mesh stays faintly visible at low Delta (ghost effect).
    a = 0.06 + 0.94 * a
    n = len(_VIRIDIS_RGB) - 1
    x = t_clip * n
    i = int(np.floor(x))
    if i >= n:
        r, g, b = _VIRIDIS_RGB[-1]
    else:
        f = x - i
        c0, c1 = _VIRIDIS_RGB[i], _VIRIDIS_RGB[i + 1]
        r = int(round(c0[0] + f * (c1[0] - c0[0])))
        g = int(round(c0[1] + f * (c1[1] - c0[1])))
        b = int(round(c0[2] + f * (c1[2] - c0[2])))
    return f"rgba({r},{g},{b},{a:.4f})"


def _gex_delta_rgba_colorscale(n_gamma: int = 48, n_delta: int = 24) -> list[list[Any]]:
    """Discrete colorscale: GEX→RGB (Viridis), Delta→alpha."""
    n_gamma = max(int(n_gamma), 2)
    n_delta = max(int(n_delta), 2)
    n = n_gamma * n_delta
    scale: list[list[Any]] = []
    denom = float(n - 1) if n > 1 else 1.0
    for gi in range(n_gamma):
        g = gi / float(n_gamma - 1)
        for di in range(n_delta):
            d = di / float(n_delta - 1)
            idx = gi * n_delta + di
            scale.append([idx / denom, _viridis_rgba(g, d)])
    return scale


def _encode_gex_delta_color(
    gamma_norm: np.ndarray,
    delta_norm: np.ndarray,
    *,
    n_gamma: int = 48,
    n_delta: int = 24,
) -> np.ndarray:
    """Pack normalized GEX + Delta into a spatially smooth colorscale coordinate."""
    n_gamma = max(int(n_gamma), 2)
    n_delta = max(int(n_delta), 2)
    g = np.array(gamma_norm, dtype=np.float64, copy=True)
    d = np.array(delta_norm, dtype=np.float64, copy=True)
    g_ok = np.isfinite(g)
    d_ok = np.isfinite(d)
    g = np.clip(g, 0.0, 1.0)
    d = np.clip(d, 0.0, 1.0)
    g[~g_ok] = 0.0
    d[~d_ok] = 0.0
    g_bin = np.floor(g * (n_gamma - 1) + 1e-12).astype(np.int64)
    d_bin = np.floor(d * (n_delta - 1) + 1e-12).astype(np.int64)
    g_bin = np.clip(g_bin, 0, n_gamma - 1)
    d_bin = np.clip(d_bin, 0, n_delta - 1)
    encoded = (g_bin * n_delta + d_bin).astype(np.float64)
    denom = float(n_gamma * n_delta - 1) or 1.0
    return encoded / denom


def _pivot_risk_field(
    frame: pd.DataFrame,
    value_col: str,
    x_col: str,
    y_col: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot a long frame to ``(z2d, x_axis, y_axis)``; always copies arrays."""
    plot = frame.copy()
    plot[x_col] = pd.to_numeric(plot[x_col], errors="coerce")
    plot[value_col] = pd.to_numeric(plot[value_col], errors="coerce")
    if y_col is None or y_col not in plot.columns:
        xs = np.array(plot[x_col].to_numpy(dtype=np.float64), copy=True)
        zs = np.array(plot[value_col].to_numpy(dtype=np.float64), copy=True)
        order = np.argsort(xs)
        x_axis = xs[order]
        z_row = zs[order]
        return np.array(z_row.reshape(1, -1), copy=True), x_axis, np.array([0.0], dtype=np.float64)
    plot[y_col] = pd.to_numeric(plot[y_col], errors="coerce")
    plot = plot.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
    if plot.empty:
        return (
            np.zeros((0, 0), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    pivot = plot.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    z2d = np.array(pivot.to_numpy(dtype=np.float64), copy=True)
    x_axis = np.array(pivot.columns.to_numpy(dtype=np.float64), copy=True)
    y_axis = np.array(pivot.index.to_numpy(dtype=np.float64), copy=True)
    return z2d, x_axis, y_axis


# Equal-weight Total Risk Score defaults (tunable). Each greek’s grid mean of
# |value| is mapped to [0, 1] via clip(mean / REF, 0, 1), then weighted.
# Score ∈ [0, 1]; alert when score exceeds RISK_SCORE_ALERT_THRESHOLD.
RISK_WEIGHT_DELTA: float = 1.0 / 3.0
RISK_WEIGHT_GAMMA: float = 1.0 / 3.0
RISK_WEIGHT_VEGA: float = 1.0 / 3.0
RISK_REF_DELTA: float = 1.0  # |Δ| already on ~[0, 1]
RISK_REF_GAMMA: float = 0.05  # typical ATM equity gamma per $1 move
RISK_REF_VEGA: float = 0.25  # typical vega-per-vol-point magnitude on the grid
RISK_SCORE_ALERT_THRESHOLD: float = 0.75


def compute_total_risk_score(
    df: pd.DataFrame,
    *,
    weight_delta: float = RISK_WEIGHT_DELTA,
    weight_gamma: float = RISK_WEIGHT_GAMMA,
    weight_vega: float = RISK_WEIGHT_VEGA,
    ref_delta: float = RISK_REF_DELTA,
    ref_gamma: float = RISK_REF_GAMMA,
    ref_vega: float = RISK_REF_VEGA,
) -> float:
    """Weighted sum of mean |Delta|, |Gamma|, |Vega| after ref-scale mapping to [0, 1].

    Weights default to equal thirds (``RISK_WEIGHT_*``). Each component is
    ``clip(mean(|greek|) / RISK_REF_*, 0, 1)``. Missing columns are skipped and
    remaining weights are renormalized. Always ``.copy()``s. Returns ``0.0`` on
    empty/invalid input.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return 0.0
    frame = df.copy()
    col_map = {str(c).strip().lower(): c for c in frame.columns}

    def _col(*names: str) -> str | None:
        for name in names:
            key = str(name).strip().lower()
            if key in col_map:
                return str(col_map[key])
        return None

    parts: list[tuple[str | None, float, float]] = [
        (_col("Delta", "delta"), float(weight_delta), float(ref_delta)),
        (_col("Gamma", "gamma", "GEX", "gex"), float(weight_gamma), float(ref_gamma)),
        (_col("Vega", "vega"), float(weight_vega), float(ref_vega)),
    ]
    score = 0.0
    weight_sum = 0.0
    for col, weight, ref in parts:
        if col is None or not np.isfinite(weight) or weight == 0.0:
            continue
        vals = np.abs(pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64))
        finite = vals[np.isfinite(vals)]
        mean_abs = float(np.mean(finite)) if finite.size else 0.0
        scale = ref if np.isfinite(ref) and ref > 0.0 else 1.0
        component = float(np.clip(mean_abs / scale, 0.0, 1.0))
        score += abs(weight) * component
        weight_sum += abs(weight)
    if weight_sum <= 0.0:
        return 0.0
    # Redistribute when a greek column is missing so the result stays in [0, 1].
    return float(np.clip(score / weight_sum, 0.0, 1.0))


def generate_integrated_risk_surface(
    df: pd.DataFrame,
    *,
    include_gamma: bool = True,
    include_delta: bool = True,
    include_vol: bool = True,
) -> Any:
    """Build a single Plotly ``go.Surface`` for the Total Risk Profile.

    Layering (Price/Strike × DaysToExpiry grid from ``df``):
    - ``z``: Volatility (IV) min-max normalized to ``[0, 1]``
    - ``surfacecolor``: GEX / Gamma normalized, packed with Delta for opacity
    - Alpha: normalized ``|Delta|`` drives per-vertex transparency (ghost effect)
      via an rgba colorscale (Plotly Surface has no independent opacity channel)

    Overlay flags (additive; default all on):
    - ``include_gamma=False`` → flat/neutral surfacecolor (mid GEX)
    - ``include_delta=False`` → full opacity (Delta layer off)
    - ``include_vol=False`` → Z flattened (hide vol contribution)

    Always ``.copy()``s the input frame and copies arrays before mutation.
    Returns an empty ``go.Surface`` when required columns or finite data are missing.
    """
    import plotly.graph_objects as go

    empty = go.Surface(
        z=np.array([[np.nan]], dtype=np.float64),
        name="Total Risk Profile",
        showscale=False,
    )
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return empty
    frame = df.copy()
    col_map = {str(c).strip().lower(): c for c in frame.columns}
    def _col(*names: str) -> str | None:
        for name in names:
            key = str(name).strip().lower()
            if key in col_map:
                return str(col_map[key])
        return None

    delta_c = _col("Delta", "delta")
    gamma_c = _col("Gamma", "gamma", "GEX", "gex")
    vol_c = _col("Volatility", "IV", "impliedVolatility", "sigma", "iv", "Vol")
    x_c = _col("Price", "Strike", "K", "strike")
    y_c = _col("DaysToExpiry", "DTE", "dte", "days_to_expiry")
    if delta_c is None or gamma_c is None or vol_c is None or x_c is None:
        return empty

    z_vol, x_axis, y_axis = _pivot_risk_field(frame, vol_c, x_c, y_c)
    z_gamma, _, _ = _pivot_risk_field(frame, gamma_c, x_c, y_c)
    z_delta, _, _ = _pivot_risk_field(frame, delta_c, x_c, y_c)
    if z_vol.size == 0 or z_gamma.shape != z_vol.shape or z_delta.shape != z_vol.shape:
        # Align via a joint pivot on a trimmed frame when shapes diverge.
        keep = [x_c, delta_c, gamma_c, vol_c] + ([y_c] if y_c else [])
        slim = frame.loc[:, [c for c in keep if c is not None]].copy()
        z_vol, x_axis, y_axis = _pivot_risk_field(slim, vol_c, x_c, y_c)
        z_gamma, _, _ = _pivot_risk_field(slim, gamma_c, x_c, y_c)
        z_delta, _, _ = _pivot_risk_field(slim, delta_c, x_c, y_c)
    if z_vol.size == 0 or z_vol.shape != z_gamma.shape or z_vol.shape != z_delta.shape:
        return empty

    vol_n = minmax_normalize_01(z_vol)
    gamma_n = minmax_normalize_01(z_gamma)
    # |Delta| so puts and calls both drive opacity toward high-risk extremes.
    delta_abs = np.abs(np.array(z_delta, dtype=np.float64, copy=True))
    delta_n = minmax_normalize_01(delta_abs)
    if not include_vol:
        # Hide vol contribution: flat Z plane at mid height.
        vol_finite = np.isfinite(vol_n)
        vol_n = np.full_like(vol_n, 0.5, dtype=np.float64)
        vol_n[~vol_finite] = np.nan
    if not include_gamma:
        # Neutral / flat GEX color (mid viridis).
        gamma_n = np.full_like(gamma_n, 0.5, dtype=np.float64)
    if not include_delta:
        # Full opacity when Delta layer is off.
        delta_n = np.ones_like(delta_n, dtype=np.float64)
    if not np.any(np.isfinite(vol_n)):
        return empty

    n_gamma, n_delta = 48, 24
    encoded = _encode_gex_delta_color(gamma_n, delta_n, n_gamma=n_gamma, n_delta=n_delta)
    colorscale = _gex_delta_rgba_colorscale(n_gamma=n_gamma, n_delta=n_delta)
    # Reinforce ghosting: opacityscale tracks the packed color (GEX×Δ).
    # When Delta is off, keep opacity fully opaque.
    if include_delta:
        opacityscale: list[list[Any]] = [[0.0, 0.08], [0.35, 0.35], [0.7, 0.7], [1.0, 1.0]]
    else:
        opacityscale = [[0.0, 1.0], [1.0, 1.0]]

    z2d, x_ok, y_ok, _warning = prepare_plotly_surface_xyz(vol_n, x_axis, y_axis)
    color_title = "GEX (Δ-opacity)"
    if not include_gamma and not include_delta:
        color_title = "Neutral"
    elif not include_gamma:
        color_title = "Δ-opacity"
    elif not include_delta:
        color_title = "GEX"
    payload: dict[str, Any] = {
        "z": np.array(z2d, dtype=np.float64, copy=True),
        "surfacecolor": np.array(encoded, dtype=np.float64, copy=True),
        "cmin": 0.0,
        "cmax": 1.0,
        "colorscale": colorscale,
        "opacityscale": opacityscale,
        "showscale": True,
        "colorbar": {"title": color_title},
        "name": "Total Risk Profile",
        "hovertemplate": (
            "X=%{x:.2f}<br>Y=%{y:.2f}<br>"
            "Vol (norm)=%{z:.3f}<br>"
            "GEX×Δ=%{surfacecolor:.3f}<extra>Total Risk Profile</extra>"
        ),
    }
    if x_ok is not None and y_ok is not None:
        payload["x"] = np.array(x_ok, dtype=np.float64, copy=True)
        payload["y"] = np.array(y_ok, dtype=np.float64, copy=True)
    return go.Surface(**payload)


_EMPTY_GREEKS: dict[str, float | None] = {
    "Delta": None,
    "Gamma": None,
    "Theta": None,
    "Vega": None,
    "Rho": None,
    "Vanna": None,
    "Volga": None,
    "Charm": None,
    "Speed": None,
    "Color": None,
}


def _model_is_heston(model: str | None) -> bool:
    return str(model or "").strip().lower() in {MODEL_HESTON, "heston"}


def _time_to_expiry_years(expiry: Any) -> float:
    """Convert an expiry date or numeric tenor to years (365-day calendar)."""
    if expiry is None:
        return float("nan")
    if isinstance(expiry, (int, float, np.floating, np.integer)):
        return float(expiry)
    if isinstance(expiry, datetime):
        expiry_date = expiry.date()
    elif isinstance(expiry, date):
        expiry_date = expiry
    else:
        try:
            expiry_date = date.fromisoformat(str(expiry)[:10])
        except ValueError:
            return float("nan")
    today = datetime.now(timezone.utc).date()
    return (expiry_date - today).days / DAYS_PER_YEAR


def _dense_grid(values: np.ndarray, n: int = CURVE_RESOLUTION) -> np.ndarray:
    """Return a linspace covering the finite span of ``values`` with >= 200 points."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.array([], dtype=np.float64)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    count = max(int(n), 200)
    if hi <= lo:
        return np.full(count, lo, dtype=np.float64)
    return np.linspace(lo, hi, count, dtype=np.float64)


def _smooth_curve_frame(frame: pd.DataFrame, x_column: str) -> pd.DataFrame:
    """Light centered rolling mean so dashboard 2D traces stay smooth."""
    if frame.empty or x_column not in frame.columns:
        return frame
    out = frame.sort_values(x_column).reset_index(drop=True)
    window = SMOOTH_WINDOW if SMOOTH_WINDOW % 2 == 1 else SMOOTH_WINDOW + 1
    skip = {x_column, "IV"}
    for column in out.columns:
        if column in skip:
            continue
        series = pd.to_numeric(out[column], errors="coerce")
        if series.notna().sum() < 4:
            continue
        out[column] = series.interpolate(method="linear", limit_direction="both").rolling(
            window, center=True, min_periods=1
        ).mean()
    return out


def _greeks_vector(
    spots: np.ndarray,
    strike: float,
    time_years: float,
    rate: float,
    sigma: float,
    dividend: float,
    is_put: bool,
) -> dict[str, np.ndarray]:
    """Vectorized Black–Scholes Greeks over an array of spots."""
    spots = np.asarray(spots, dtype=np.float64)
    n = spots.shape[0]
    nan = np.full(n, np.nan, dtype=np.float64)
    out = {
        "Delta": nan.copy(),
        "Gamma": nan.copy(),
        "Theta": nan.copy(),
        "Vega": nan.copy(),
        "Rho": nan.copy(),
        "Vanna": nan.copy(),
        "Volga": nan.copy(),
        "Charm": nan.copy(),
        "Speed": nan.copy(),
        "Color": nan.copy(),
    }

    scalars_ok = (
        np.isfinite(strike)
        and strike > 0
        and np.isfinite(time_years)
        and time_years >= T_MIN
        and np.isfinite(rate)
        and np.isfinite(sigma)
        and sigma >= SIGMA_MIN
        and np.isfinite(dividend)
    )
    if n == 0 or not scalars_ok:
        return out

    valid = np.isfinite(spots) & (spots > 0)
    if not np.any(valid):
        return out

    s = spots[valid]
    sqrt_t = math_sqrt = np.sqrt(time_years)
    d1 = (np.log(s / strike) + (rate - dividend + 0.5 * sigma * sigma) * time_years) / (sigma * math_sqrt)
    d2 = d1 - sigma * math_sqrt
    n_d1 = norm.pdf(d1)
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    disc_q = np.exp(-dividend * time_years)
    disc_r = np.exp(-rate * time_years)

    gamma = disc_q * n_d1 / (s * sigma * math_sqrt)
    # Vega: ∂V/∂σ via scipy.stats.norm.pdf(d1), displayed per 1 vol point.
    vega = s * disc_q * n_d1 * math_sqrt / 100.0

    if is_put:
        delta = disc_q * (nd1 - 1.0)
        theta_annual = (
            -(s * disc_q * n_d1 * sigma) / (2.0 * math_sqrt)
            + rate * strike * disc_r * norm.cdf(-d2)
            - dividend * s * disc_q * norm.cdf(-d1)
        )
        rho = -strike * time_years * disc_r * norm.cdf(-d2) / 100.0
    else:
        delta = disc_q * nd1
        theta_annual = (
            -(s * disc_q * n_d1 * sigma) / (2.0 * math_sqrt)
            - rate * strike * disc_r * nd2
            + dividend * s * disc_q * nd1
        )
        rho = strike * time_years * disc_r * nd2 / 100.0

    out["Delta"][valid] = np.asarray(delta, dtype=np.float64)
    out["Gamma"][valid] = np.asarray(gamma, dtype=np.float64)
    out["Theta"][valid] = np.asarray(theta_annual / DAYS_PER_YEAR, dtype=np.float64)
    out["Vega"][valid] = np.asarray(vega, dtype=np.float64)
    out["Rho"][valid] = np.asarray(rho, dtype=np.float64)
    vanna = -disc_q * n_d1 * d2 / sigma
    volga = s * disc_q * n_d1 * math_sqrt * d1 * d2 / sigma
    charm_core = n_d1 * (2.0 * (rate - dividend) * time_years - d2 * sigma * math_sqrt) / (
        2.0 * time_years * sigma * math_sqrt
    )
    if is_put:
        charm_annual = -disc_q * (charm_core + dividend * (1.0 - nd1))
    else:
        charm_annual = -disc_q * (charm_core - dividend * nd1)
    out["Vanna"][valid] = np.asarray(vanna, dtype=np.float64)
    out["Volga"][valid] = np.asarray(volga, dtype=np.float64)
    out["Charm"][valid] = np.asarray(charm_annual / DAYS_PER_YEAR, dtype=np.float64)
    speed = -gamma / s * (d1 / (sigma * math_sqrt) + 1.0)
    color_annual = (
        -disc_q
        * n_d1
        / (2.0 * s * time_years * sigma * math_sqrt)
        * (
            2.0 * dividend * time_years
            + 1.0
            + d1 * (2.0 * (rate - dividend) * time_years - d2 * sigma * math_sqrt) / (sigma * math_sqrt)
        )
    )
    out["Speed"][valid] = np.asarray(speed, dtype=np.float64)
    out["Color"][valid] = np.asarray(color_annual / DAYS_PER_YEAR, dtype=np.float64)
    return out


def calculate_greeks(option_data: Mapping[str, Any]) -> dict[str, float | None]:
    """Compute Black–Scholes Greeks from spot, strike, tenor, rate, and vol.

    Args:
        option_data: Mapping with:
            S: underlying price
            K: strike
            T: time to expiry in years
            r: continuous risk-free rate
            sigma: implied volatility (annualized)
            q: continuous dividend yield (optional, default 0)
            option_type: ``call`` or ``put`` (optional, default ``call``)

    Returns:
        Dict with Delta, Gamma, Theta, Vega, Rho.
        Theta is per calendar day (annual theta / 365).
        Vega is per 1 volatility point (raw vega / 100).
        Rho is per 1% rate change (raw rho / 100).
        Returns all-None Greeks when T <= 0, sigma <= 0, or inputs are invalid.
    """
    empty: dict[str, float | None] = {
        "Delta": None,
        "Gamma": None,
        "Theta": None,
        "Vega": None,
        "Rho": None,
    }
    try:
        spot = float(option_data["S"])
        strike = float(option_data["K"])
        time_years = float(option_data["T"])
        rate = float(option_data["r"])
        raw_sigma = option_data.get("sigma", option_data.get("impliedVolatility"))
        if raw_sigma is None or str(raw_sigma).strip().lower() in {"", "null", "nan", "none"}:
            return empty
        sigma = float(raw_sigma)
        dividend = float(option_data.get("q", 0.0) or 0.0)
    except (KeyError, TypeError, ValueError):
        return empty

    option_type = str(option_data.get("option_type", "call")).strip().lower()
    greeks = _greeks_vector(
        np.array([spot], dtype=np.float64),
        strike,
        time_years,
        rate,
        sigma,
        dividend,
        option_type.startswith("p"),
    )
    result: dict[str, float | None] = {}
    for name, values in greeks.items():
        value = float(values[0])
        result[name] = None if not np.isfinite(value) else value
    return result


def _heston_params_valid(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    dividend: float,
    v0: float,
    kappa: float,
    theta: float,
    eta: float,
    rho: float,
) -> bool:
    """Reject non-finite or economically invalid Heston inputs (including v0 <= 0)."""
    values = (spot, strike, time_years, rate, dividend, v0, kappa, theta, eta, rho)
    if not all(np.isfinite(v) for v in values):
        return False
    if spot <= 0 or strike <= 0 or time_years < T_MIN:
        return False
    if v0 <= 0 or kappa <= 0 or theta <= 0 or eta <= 0:
        return False
    if not (-0.999999 < rho < 0.999999):
        return False
    return True


def _heston_unit_shocks(
    n_steps: int,
    n_half: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Antithetic Gaussian shocks for correlated Brownian drivers."""
    z_var = rng.standard_normal((n_steps, n_half))
    z_perp = rng.standard_normal((n_steps, n_half))
    z_var = np.concatenate([z_var, -z_var], axis=1)
    z_perp = np.concatenate([z_perp, -z_perp], axis=1)
    return z_var, z_perp


def _heston_multipliers(
    time_years: float,
    rate: float,
    dividend: float,
    v0: float,
    kappa: float,
    theta: float,
    eta: float,
    rho: float,
    z_var: np.ndarray,
    z_perp: np.ndarray,
) -> np.ndarray:
    """Euler–Maruyama with full truncation; returns S_T / S_0 for each path.

    Variance is truncated at 0 before the square root (Lord–Koekkoek–van Dijk),
    and the spot is evolved in log space so S stays positive.
    """
    n_steps, n_paths = z_var.shape
    dt = time_years / n_steps
    sqrt_dt = np.sqrt(dt)
    corr_perp = np.sqrt(max(1.0 - rho * rho, 0.0))
    v = np.full(n_paths, v0, dtype=np.float64)
    log_m = np.zeros(n_paths, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for step in range(n_steps):
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            z2 = z_var[step]
            z1 = rho * z2 + corr_perp * z_perp[step]
            log_m += (rate - dividend - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * z1
            v = v + kappa * (theta - v_pos) * dt + eta * sqrt_v * sqrt_dt * z2
        log_m = np.clip(log_m, -40.0, 40.0)
        multipliers = np.exp(log_m)
    multipliers = np.where(np.isfinite(multipliers) & (multipliers > 0), multipliers, np.nan)
    return multipliers


def _heston_mc_prices(
    spots: np.ndarray,
    strike: float,
    time_years: float,
    rate: float,
    is_put: bool,
    multipliers: np.ndarray,
) -> np.ndarray:
    """Discounted MC prices for many spots from shared S_T/S_0 multipliers."""
    disc = np.exp(-rate * time_years)
    st = spots[:, None] * multipliers[None, :]
    if is_put:
        payoff = np.maximum(strike - st, 0.0)
    else:
        payoff = np.maximum(st - strike, 0.0)
    with np.errstate(invalid="ignore"):
        prices = disc * np.nanmean(payoff, axis=1)
    return prices


def _heston_greeks_vector(
    spots: np.ndarray,
    strike: float,
    time_years: float,
    rate: float,
    dividend: float,
    v0: float,
    kappa: float,
    theta: float,
    eta: float,
    rho: float,
    is_put: bool,
    *,
    n_paths: int = HESTON_PATHS,
    n_steps: int = HESTON_STEPS,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Heston Greeks from one Euler–Maruyama shock set (common random numbers)."""
    spots = np.asarray(spots, dtype=np.float64)
    n = spots.shape[0]
    nan = np.full(n, np.nan, dtype=np.float64)
    out = {
        "Delta": nan.copy(),
        "Gamma": nan.copy(),
        "Theta": nan.copy(),
        "Vega": nan.copy(),
        "Rho": nan.copy(),
        "Vanna": nan.copy(),
        "Volga": nan.copy(),
        "Charm": nan.copy(),
        "Speed": nan.copy(),
        "Color": nan.copy(),
    }
    valid = np.isfinite(spots) & (spots > 0)
    if n == 0 or not np.any(valid):
        return out
    s = spots[valid]
    sample = float(s[0]) if s.size else 0.0
    if not _heston_params_valid(sample, strike, time_years, rate, dividend, v0, kappa, theta, eta, rho):
        return out

    n_half = max(int(n_paths) // 2, 1)
    n_steps = max(int(n_steps), HESTON_STEPS)
    rng = np.random.default_rng(int(seed) if seed is not None else 42)
    z_var, z_perp = _heston_unit_shocks(n_steps, n_half, rng)
    m = _heston_multipliers(time_years, rate, dividend, v0, kappa, theta, eta, rho, z_var, z_perp)
    if not np.any(np.isfinite(m)):
        return out

    disc = np.exp(-rate * time_years)
    st = s[:, None] * m[None, :]
    if is_put:
        itm = st < strike
        path_delta = disc * np.nanmean((-m)[None, :] * itm, axis=1)
    else:
        itm = st > strike
        path_delta = disc * np.nanmean(m[None, :] * itm, axis=1)

    bump = np.maximum(s * 0.01, 0.5)
    mid = _heston_mc_prices(s, strike, time_years, rate, is_put, m)
    up = _heston_mc_prices(s + bump, strike, time_years, rate, is_put, m)
    down = _heston_mc_prices(np.maximum(s - bump, 1e-8), strike, time_years, rate, is_put, m)
    gamma = (up - 2.0 * mid + down) / (bump * bump)

    dv = max(v0 * 0.02, 1e-4)
    v0_up = v0 + dv
    sigma0 = np.sqrt(v0)
    sigma_up = np.sqrt(v0_up)
    d_sigma = sigma_up - sigma0
    m_v = _heston_multipliers(time_years, rate, dividend, v0_up, kappa, theta, eta, rho, z_var, z_perp)
    d_price_dv0 = (_heston_mc_prices(s, strike, time_years, rate, is_put, m_v) - mid) / dv
    vega = d_price_dv0 * (2.0 * sigma0) / 100.0

    st_v = s[:, None] * m_v[None, :]
    if is_put:
        path_delta_up = disc * np.nanmean((-m_v)[None, :] * (st_v < strike), axis=1)
    else:
        path_delta_up = disc * np.nanmean(m_v[None, :] * (st_v > strike), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vanna = np.where(d_sigma > 1e-12, (path_delta_up - path_delta) / d_sigma, np.nan)
        m_v2 = _heston_multipliers(time_years, rate, dividend, v0_up + dv, kappa, theta, eta, rho, z_var, z_perp)
        mid_up = _heston_mc_prices(s, strike, time_years, rate, is_put, m_v)
        d_price_dv0_up = (_heston_mc_prices(s, strike, time_years, rate, is_put, m_v2) - mid_up) / dv
        vega_raw = d_price_dv0 * (2.0 * sigma0)
        vega_raw_up = d_price_dv0_up * (2.0 * sigma_up)
        volga = np.where(d_sigma > 1e-12, (vega_raw_up - vega_raw) / d_sigma, np.nan)

    dt_day = 1.0 / DAYS_PER_YEAR
    t_down = time_years - dt_day
    if t_down >= T_MIN:
        m_t = _heston_multipliers(t_down, rate, dividend, v0, kappa, theta, eta, rho, z_var, z_perp)
        theta_day = _heston_mc_prices(s, strike, t_down, rate, is_put, m_t) - mid
        disc_t = np.exp(-rate * t_down)
        st_t = s[:, None] * m_t[None, :]
        if is_put:
            delta_down = disc_t * np.nanmean((-m_t)[None, :] * (st_t < strike), axis=1)
        else:
            delta_down = disc_t * np.nanmean(m_t[None, :] * (st_t > strike), axis=1)
        charm_day = delta_down - path_delta
        up_t = _heston_mc_prices(s + bump, strike, t_down, rate, is_put, m_t)
        down_t = _heston_mc_prices(np.maximum(s - bump, 1e-8), strike, t_down, rate, is_put, m_t)
        mid_t = _heston_mc_prices(s, strike, t_down, rate, is_put, m_t)
        gamma_t = (up_t - 2.0 * mid_t + down_t) / (bump * bump)
        color_day = gamma_t - gamma
    else:
        theta_day = np.full(s.shape, np.nan, dtype=np.float64)
        charm_day = np.full(s.shape, np.nan, dtype=np.float64)
        color_day = np.full(s.shape, np.nan, dtype=np.float64)

    up2 = _heston_mc_prices(s + 2.0 * bump, strike, time_years, rate, is_put, m)
    down2 = _heston_mc_prices(np.maximum(s - 2.0 * bump, 1e-8), strike, time_years, rate, is_put, m)
    gamma_up = (up2 - 2.0 * up + mid) / (bump * bump)
    gamma_down = (mid - 2.0 * down + down2) / (bump * bump)
    speed = (gamma_up - gamma_down) / (2.0 * bump)

    dr = 1e-4
    m_r = _heston_multipliers(time_years, rate + dr, dividend, v0, kappa, theta, eta, rho, z_var, z_perp)
    rho_greek = (_heston_mc_prices(s, strike, time_years, rate + dr, is_put, m_r) - mid) / dr / 100.0

    out["Delta"][valid] = path_delta
    out["Gamma"][valid] = gamma
    out["Theta"][valid] = theta_day
    out["Vega"][valid] = vega
    out["Rho"][valid] = rho_greek
    out["Vanna"][valid] = vanna
    out["Volga"][valid] = volga
    out["Charm"][valid] = charm_day
    out["Speed"][valid] = speed
    out["Color"][valid] = color_day
    return out


def calculate_heston_greeks(option_data: Mapping[str, Any]) -> dict[str, float | None]:
    """Compute Heston Greeks from Euler–Maruyama Monte Carlo.

    Signature and return contract match ``calculate_greeks``.

    Args:
        option_data: Mapping with:
            S: underlying price
            K: strike
            T: time to expiry in years
            r: continuous risk-free rate
            v0: initial variance (must be > 0)
            kappa: mean-reversion speed (must be > 0)
            theta: long-run variance (must be > 0)
            sigma: vol-of-vol (must be > 0)
            rho: correlation of Brownian motions, in (-1, 1)
            q: continuous dividend yield (optional, default 0)
            option_type: ``call`` or ``put`` (optional, default ``call``)

    Returns:
        Dict with Delta, Gamma, Theta, Vega, Rho.
        Theta is per calendar day. Vega is per 1 vol point in sqrt(v0).
        Rho is per 1% rate change.
        Returns all-None Greeks when inputs are missing or invalid
        (including v0 <= 0, kappa <= 0, theta <= 0, sigma <= 0, |rho| >= 1).
    """
    empty = dict(_EMPTY_GREEKS)
    try:
        spot = float(option_data["S"])
        strike = float(option_data["K"])
        time_years = float(option_data["T"])
        rate = float(option_data["r"])
        v0 = float(option_data["v0"])
        kappa = float(option_data["kappa"])
        theta = float(option_data["theta"])
        eta = float(option_data["sigma"])
        rho = float(option_data["rho"])
        dividend = float(option_data.get("q", 0.0) or 0.0)
    except (KeyError, TypeError, ValueError):
        return empty

    if not _heston_params_valid(spot, strike, time_years, rate, dividend, v0, kappa, theta, eta, rho):
        return empty

    option_type = str(option_data.get("option_type", "call")).strip().lower()
    seed = int(option_data.get("seed", 42))
    n_steps = max(int(option_data.get("n_steps", HESTON_STEPS)), HESTON_STEPS)
    n_paths = max(int(option_data.get("n_paths", HESTON_PATHS)), HESTON_PATHS)
    greeks = _heston_greeks_vector(
        np.array([spot], dtype=np.float64),
        strike,
        time_years,
        rate,
        dividend,
        v0,
        kappa,
        theta,
        eta,
        rho,
        option_type.startswith("p"),
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )
    result: dict[str, float | None] = {}
    for name, values in greeks.items():
        value = float(values[0])
        result[name] = None if not np.isfinite(value) else value
    return result


def generate_synthetic_drift(ticker: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a non-flat synthetic delta-drift surface for Time Machine fallback.

    Uses ``np.meshgrid`` over Strike × DTE with a mix of ``np.sin`` / ``np.cos``
    ripples and Gaussian "market move" bumps so ``Z`` has visible depth
    (``min != max``, not ~0). Return contract matches
    :meth:`DataRepository.calculate_delta_drift_grid`:
    ``(strike_axis, dte_axis, Z)`` with ``Z.shape == (len(dte_axis), len(strike_axis))``.

    The surface is deterministic for a given ``ticker`` (phase offset from the name).
    """
    label = str(ticker or "SYN").strip().upper() or "SYN"
    phase = (sum(ord(ch) for ch in label) % 97) / 97.0
    nx = int(max(16, min(40, VOL_SURFACE_STRIKE_POINTS // 2)))
    ny = int(max(12, min(30, VOL_SURFACE_DTE_POINTS // 2)))
    strike_axis = np.linspace(80.0, 120.0, nx, dtype=np.float64)
    dte_axis = np.linspace(7.0, 60.0, ny, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(strike_axis, dte_axis)
    xn = (grid_x - float(strike_axis.min())) / (float(strike_axis.max() - strike_axis.min()) + 1e-12)
    yn = (grid_y - float(dte_axis.min())) / (float(dte_axis.max() - dte_axis.min()) + 1e-12)
    # Sin/cos ripples give a rolling market texture; Gaussians add a localized move.
    ripples = (
        0.12 * np.sin(2.0 * np.pi * (xn + phase)) * np.cos(2.0 * np.pi * (yn - 0.5 * phase))
        + 0.06 * np.cos(3.0 * np.pi * xn + phase)
        + 0.04 * np.sin(np.pi * yn + 2.0 * phase)
    )
    bump_a = 0.22 * np.exp(
        -(((xn - (0.35 + 0.1 * phase)) ** 2) / (2.0 * 0.08**2) + ((yn - 0.45) ** 2) / (2.0 * 0.10**2))
    )
    bump_b = -0.16 * np.exp(
        -(((xn - (0.70 - 0.1 * phase)) ** 2) / (2.0 * 0.09**2) + ((yn - 0.65) ** 2) / (2.0 * 0.12**2))
    )
    drift = (ripples + bump_a + bump_b).astype(float)
    return strike_axis.astype(float), dte_axis.astype(float), drift


DRIFT_PREFILL_SHIFT = 0.05
DRIFT_PREFILL_NOISE_SIGMA = 0.01


def _drift_prefill_expiry_label(expiry: Any) -> str:
    """Normalize expiry to ``YYYY-MM-DD`` (filesystem-safe)."""
    if isinstance(expiry, datetime):
        return expiry.date().isoformat()
    if isinstance(expiry, date):
        return expiry.isoformat()
    text = str(expiry or "").strip()
    if not text:
        return "unknown"
    return text[:10]


def _drift_prefill_strike_label(strike: Any) -> str:
    """Filesystem-safe strike token (no brackets/spaces)."""
    try:
        value = float(strike)
        if np.isfinite(value):
            return f"{value:g}"
    except (TypeError, ValueError):
        pass
    raw = str(strike or "na").strip() or "na"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)


def drift_prefill_stamp(strike: Any, expiry: Any, role: str) -> str:
    """Stamp segment for Time Machine listing (after ``{TICKER}_``).

    On-disk pattern (brackets avoided for portability)::

        {TICKER}_{STRIKE}_{EXPIRY}_{START|END}.json

    Example: ``SPY_450_2026-09-19_START.json`` / ``SPY_450_2026-09-19_END.json``.
    """
    tag = str(role or "").strip().upper()
    if tag not in {"START", "END"}:
        tag = "START"
    return f"{_drift_prefill_strike_label(strike)}_{_drift_prefill_expiry_label(expiry)}_{tag}"


def _apply_market_drift_delta(
    values: Any,
    *,
    shift: float = DRIFT_PREFILL_SHIFT,
    noise_sigma: float = DRIFT_PREFILL_NOISE_SIGMA,
    rng: np.random.Generator | None = None,
) -> list[float | None]:
    """Shift Delta by ``shift`` and add i.i.d. Gaussian noise (Market Drift)."""
    generator = rng if rng is not None else np.random.default_rng()
    if not isinstance(values, (list, tuple, np.ndarray)):
        return []
    arr = np.asarray(
        [np.nan if v is None else float(v) for v in values],
        dtype=np.float64,
    )
    noise = generator.normal(0.0, float(noise_sigma), size=arr.shape)
    drifted = arr + float(shift) + noise
    return [None if not np.isfinite(v) else float(v) for v in drifted]


def _apply_market_drift_to_payload(
    payload: dict[str, Any],
    *,
    shift: float = DRIFT_PREFILL_SHIFT,
    noise_sigma: float = DRIFT_PREFILL_NOISE_SIGMA,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Return a deep-ish copy of ``payload`` with Market Drift on Delta surface + raw rows."""
    generator = rng if rng is not None else np.random.default_rng()
    out = dict(payload)
    surface = payload.get("delta_surface")
    if isinstance(surface, dict):
        drifted_surface = dict(surface)
        drifted_surface["Delta"] = _apply_market_drift_delta(
            surface.get("Delta"),
            shift=shift,
            noise_sigma=noise_sigma,
            rng=generator,
        )
        out["delta_surface"] = drifted_surface
    records = payload.get("data")
    if isinstance(records, list) and records:
        new_rows: list[Any] = []
        for row in records:
            if not isinstance(row, dict):
                new_rows.append(row)
                continue
            cloned = dict(row)
            for key in ("Delta", "delta"):
                if key not in cloned or cloned[key] is None:
                    continue
                try:
                    base = float(cloned[key])
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(base):
                    continue
                cloned[key] = float(base + shift + float(generator.normal(0.0, float(noise_sigma))))
            new_rows.append(cloned)
        out["data"] = new_rows
    return out


def prefill_drift_data(
    ticker: str,
    strike: Any = None,
    expiry: Any = None,
    *,
    repo: DataRepository | None = None,
    seed: int | None = None,
    shift: float = DRIFT_PREFILL_SHIFT,
    noise_sigma: float = DRIFT_PREFILL_NOISE_SIGMA,
) -> dict[str, Any]:
    """Fetch a live chain and write START/END snapshots with Market Drift on Delta.

    Steps:
      1. Fetch the current option chain via :class:`DataRepository` (``_fetch`` /
         injected fetcher).
      2. Persist the real surface as ``{TICKER}_{STRIKE}_{EXPIRY}_START.json``.
      3. Apply Market Drift: Delta += ``shift`` (+ small Gaussian noise).
      4. Persist the drifted surface as ``{TICKER}_{STRIKE}_{EXPIRY}_END.json``.

    Both files land under ``data_history/`` (via ``repo.history_dir``) with
    ``metadata`` (ticker, strike, expiry) so
    :meth:`DataRepository.validate_snapshot_match` still passes. Filenames use
    underscores instead of brackets for filesystem safety; the START/END role
    remains explicit.

    Returns paths and Time Machine dropdown labels (stamps) for auto-select.
    """
    repository = repo if repo is not None else DataRepository()
    symbol = str(ticker or "").strip().upper() or "SPY"
    safe = repository._safe_ticker(symbol)
    start_stamp = drift_prefill_stamp(strike, expiry, "START")
    end_stamp = drift_prefill_stamp(strike, expiry, "END")
    metadata = repository._normalize_metadata(
        symbol,
        {"ticker": symbol, "strike": strike, "expiry": expiry},
    )

    try:
        frame = repository._fetch(symbol, expiry)
    except Exception:
        frame = repository.get_data(symbol, expiry)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"No option chain data available to prefill drift for {symbol}.")

    start_payload = repository._payload_from_data(symbol, frame)
    start_payload["stamp"] = start_stamp
    start_payload["metadata"] = dict(metadata)
    for key in DataRepository.CONTRACT_TOP_LEVEL_KEYS:
        value = metadata.get(key)
        if value is not None:
            start_payload[key] = value
    start_payload["snapshot_id"] = uuid.uuid4().hex
    start_payload["saved_at"] = time.time()

    rng = np.random.default_rng(int(seed) if seed is not None else None)
    end_payload = _apply_market_drift_to_payload(
        start_payload,
        shift=shift,
        noise_sigma=noise_sigma,
        rng=rng,
    )
    end_payload["stamp"] = end_stamp
    end_payload["metadata"] = dict(metadata)
    for key in DataRepository.CONTRACT_TOP_LEVEL_KEYS:
        value = metadata.get(key)
        if value is not None:
            end_payload[key] = value
    end_payload["snapshot_id"] = uuid.uuid4().hex
    end_payload["saved_at"] = time.time() + 1e-3

    start_path = repository._atomic_json_write(
        Path(repository.history_dir) / f"{safe}_{start_stamp}.json",
        start_payload,
    )
    end_path = repository._atomic_json_write(
        Path(repository.history_dir) / f"{safe}_{end_stamp}.json",
        end_payload,
    )

    return {
        "ticker": symbol,
        "strike": metadata.get("strike"),
        "expiry": metadata.get("expiry"),
        "start_path": str(start_path),
        "end_path": str(end_path),
        "start_label": start_stamp,
        "end_label": end_stamp,
        "start_stamp": start_stamp,
        "end_stamp": end_stamp,
        "filename_pattern": "{TICKER}_{STRIKE}_{EXPIRY}_{START|END}.json",
    }


# Alias for the alternate spelling used in the request.
prefill_driftdata = prefill_drift_data


class DataRepository:
    """TTL-backed JSON cache for option-chain frames, with atomic writes."""

    VERSION = "1.0"
    TTL_SECONDS = 15 * 60
    HISTORY_LIMIT = 50

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        fetcher: Callable[[str, Any], pd.DataFrame] | None = None,
        status_probe: Callable[[], dict[str, Any]] | None = None,
        history_dir: str | Path | None = None,
    ) -> None:
        root = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parent / "cache" / "marketdata"
        self.cache_dir = root
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        hist = Path(history_dir) if history_dir is not None else (
            Path(__file__).resolve().parent / "data_history" if cache_dir is None else root / "data_history"
        )
        self.history_dir = hist
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._fetcher = fetcher
        self._status_probe = status_probe
        # Last ingest outcome for dashboard (token / API Error: [code]).
        self.last_ingest_status: dict[str, Any] = {
            "ok": True,
            "status_code": None,
            "message": "",
            "token_missing": False,
            "is_mock": False,
        }

    def _safe_ticker(self, ticker: str) -> str:
        symbol = str(ticker or "").strip().upper() or "_"
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in symbol)

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{self._safe_ticker(ticker)}.json"

    def _history_stamp(self, when: datetime | None = None) -> str:
        """Second-resolution stamp for unique snapshot filenames (``%Y%m%d%H%M%S``).

        This is **capture time**, not option expiry. When expiry is known,
        ``_stamp_with_expiry`` prefixes ``YYYYmmDDexp_`` so filenames/labels
        do not look like the wrong contract.
        """
        moment = when or datetime.now()
        return moment.strftime("%Y%m%d%H%M%S")

    @staticmethod
    def _expiry_compact(expiry: Any) -> str | None:
        """Normalize expiry to ``YYYYmmDD`` for filename/stamp tokens."""
        if expiry is None or expiry == "":
            return None
        if isinstance(expiry, datetime):
            return expiry.date().strftime("%Y%m%d")
        if isinstance(expiry, date):
            return expiry.strftime("%Y%m%d")
        text = str(expiry).strip()
        if len(text) >= 10 and text[4] == "-" and text[7] == "-":
            digits = text[:10].replace("-", "")
            return digits if len(digits) == 8 and digits.isdigit() else None
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits[:8] if len(digits) >= 8 else None

    @classmethod
    def _stamp_with_expiry(cls, capture_stamp: str, expiry: Any) -> str:
        """``{YYYYmmDDexp}_{captureStamp}`` when expiry known; else capture stamp alone."""
        base = str(capture_stamp or "").strip()
        token = cls._expiry_compact(expiry)
        if not token or not base:
            return base
        if "exp_" in base:
            return base
        return f"{token}exp_{base}"

    @staticmethod
    def _expiry_from_records(records: Any) -> str | None:
        """Single shared ISO expiry from option rows, if unambiguous."""
        if not isinstance(records, list):
            return None
        found: set[str] = set()
        for row in records:
            if not isinstance(row, dict):
                continue
            for key in ("expiration", "expiry", "Expiry", "Expiration"):
                value = row.get(key)
                if value is None or value == "":
                    continue
                text = str(value).strip()
                if len(text) >= 10 and text[4] == "-" and text[7] == "-":
                    found.add(text[:10])
                elif len(text) >= 8 and text[:8].isdigit():
                    found.add(f"{text[:4]}-{text[4:6]}-{text[6:8]}")
                break
        if len(found) == 1:
            return next(iter(found))
        return None

    def _history_path(self, ticker: str, stamp: str | None = None) -> Path:
        label = stamp or self._history_stamp()
        filename = f"{self._safe_ticker(ticker)}_{label}.json"
        joined = os.path.join("data_history", filename)
        path = Path(self.history_dir) / os.path.basename(joined)
        if path.exists():
            filename = f"{self._safe_ticker(ticker)}_{label}_{int(time.time())}.json"
            joined = os.path.join("data_history", filename)
            path = Path(self.history_dir) / os.path.basename(joined)
        return path

    CONTRACT_TOP_LEVEL_KEYS: tuple[str, ...] = ("ticker", "strike", "expiry", "contract_type")

    def _embed_contract_fields(self, payload: dict[str, Any], metadata: Mapping[str, Any] | None) -> None:
        """Write contract fields to nested ``metadata`` and matching top-level keys."""
        if not metadata:
            return
        normalized = self._normalize_metadata(str(payload.get("ticker") or ""), metadata)
        payload["metadata"] = normalized
        for key in self.CONTRACT_TOP_LEVEL_KEYS:
            value = normalized.get(key)
            if value is not None:
                payload[key] = value

    def save_to_cache(self, ticker: str, data: Any, metadata: Mapping[str, Any] | None = None) -> Path:
        os.makedirs("data_history", exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)
        payload = self._payload_from_data(ticker, data)
        self._embed_contract_fields(payload, metadata)
        # Prefix capture stamp with option expiry so on-disk names show the contract.
        expiry = payload.get("expiry")
        if expiry in (None, "") and isinstance(payload.get("metadata"), dict):
            expiry = payload["metadata"].get("expiry")
        if expiry in (None, ""):
            expiry = self._expiry_from_records(payload.get("data"))
        payload["stamp"] = self._stamp_with_expiry(str(payload.get("stamp") or self._history_stamp()), expiry)
        # Mandatory: every computed surface must exist (actually computed from the
        # raw chain, not defaulted) before anything is written to disk.
        frame = self._records_to_frame(payload.get("data"))
        for key, value_key in self.SURFACE_KEYS.items():
            if self._surface_is_empty(payload.get(key), value_key):
                payload[key] = self._compute_surfaces(frame)[key]
        for key, value_key in self.SURFACE_KEYS.items():
            payload.setdefault(key, {"Strike": [], "DaysToExpiry": [], value_key: []})
        payload.setdefault("snapshot_id", uuid.uuid4().hex)
        payload.setdefault("spot", self._spot_from_frame(frame))
        previous_path = self._latest_snapshot_path(ticker)
        if previous_path is not None:
            previous = self._read_json(previous_path) or {}
            if self._delta_surfaces_identical(previous.get("delta_surface"), payload.get("delta_surface")):
                print("Snapshot identical to previous—not saving.")
                _LOG.info("Snapshot for %s identical to %s; skipped.", ticker, previous_path.name)
                return previous_path
        live = self._atomic_json_write(self._cache_path(ticker), payload)
        history = self._atomic_json_write(self._history_path(ticker, str(payload["stamp"])), payload)
        self._prune_history(ticker)
        saved_keys = list(payload.keys())
        _LOG.info("Snapshot saved to data_history/ keys=%s", saved_keys)
        print("Snapshot saved to data_history/")
        print(f"Successfully saved {saved_keys} to snapshot.")
        return live if live else history

    METADATA_KEYS: tuple[str, ...] = ("ticker", "strike", "expiry", "contract_type", "as_of_date")

    @staticmethod
    def _normalize_contract_type(value: Any) -> str | None:
        if value is None or value == "":
            return None
        text = str(value).strip().lower()
        if text.startswith("c"):
            return "Call"
        if text.startswith("p"):
            return "Put"
        return None

    def _normalize_metadata(self, ticker: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {key: None for key in self.METADATA_KEYS}
        out["ticker"] = str(metadata.get("ticker") or ticker or "").strip().upper() or None
        strike = metadata.get("strike")
        try:
            out["strike"] = float(strike) if strike is not None and np.isfinite(float(strike)) else None
        except (TypeError, ValueError):
            out["strike"] = None
        for key in ("expiry", "as_of_date"):
            value = metadata.get(key)
            if value is None or value == "":
                continue
            out[key] = value.isoformat() if isinstance(value, (date, datetime)) else str(value).strip()[:10]
        out["contract_type"] = self._normalize_contract_type(
            metadata.get("contract_type") if metadata.get("contract_type") not in (None, "") else metadata.get("option_type")
        )
        return out

    @classmethod
    def format_snapshot_label(cls, payload: Any, stamp: str | None = None) -> str:
        """Human label with expiry first: ``2026-10-30 · SPY 580 Call @ 14:30:05``.

        Falls back to stamp alone when no contract fields exist. Expiry is taken from
        top-level / metadata, then from unambiguous ``data[].expiration`` rows (legacy
        snapshots), then from an ``YYYYmmDDexp_`` stamp prefix.
        """
        if not isinstance(payload, dict):
            return str(stamp or "")
        meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        ticker = payload.get("ticker") or meta.get("ticker")
        strike = payload.get("strike") if payload.get("strike") is not None else meta.get("strike")
        expiry = payload.get("expiry") or meta.get("expiry")
        if expiry in (None, ""):
            expiry = cls._expiry_from_records(payload.get("data"))
        ctype = payload.get("contract_type") or meta.get("contract_type")
        ctype = cls._normalize_contract_type(ctype) or ctype
        stamp_text = str(stamp or payload.get("stamp") or "").strip()
        if expiry in (None, "") and "exp_" in stamp_text:
            token = stamp_text.split("exp_", 1)[0]
            if len(token) == 8 and token.isdigit():
                expiry = f"{token[:4]}-{token[4:6]}-{token[6:8]}"
        time_part = cls._format_stamp_clock(stamp_text)
        parts: list[str] = []
        if expiry:
            parts.append(str(expiry).strip()[:10])
        if ticker:
            parts.append(str(ticker).strip().upper())
        if strike is not None and strike != "":
            try:
                parts.append(f"{float(strike):g}")
            except (TypeError, ValueError):
                parts.append(str(strike))
        if ctype:
            parts.append(str(ctype))
        if not parts:
            return stamp_text or "?"
        # Expiry-first when present: "2026-10-30 · SPY 580 Call"
        if expiry and len(parts) > 1:
            label = f"{parts[0]} · {' '.join(parts[1:])}"
        else:
            label = " ".join(parts)
        if time_part:
            label = f"{label} @ {time_part}"
        return label

    @staticmethod
    def _format_stamp_clock(stamp: str) -> str:
        """Extract ``HH:MM:SS`` (or ``HH:MM``) from new or legacy stamps."""
        text = str(stamp or "").strip()
        if not text:
            return ""
        # Expiry-prefixed: YYYYmmDDexp_YYYYMMDDHHMMSS (optional _<epoch> suffix)
        if "exp_" in text:
            text = text.split("exp_", 1)[-1]
        # New high-res: YYYYMMDDHHMMSS (optionally with _<epoch> collision suffix)
        head = text.split("_", 1)[0]
        if len(head) >= 14 and head[:14].isdigit():
            hh, mm, ss = head[8:10], head[10:12], head[12:14]
            return f"{hh}:{mm}:{ss}"
        # Legacy minute stamp: YYYY-MM-DD_HHMM
        try:
            moment = datetime.strptime(text[:15], "%Y-%m-%d_%H%M")
            return moment.strftime("%H:%M:%S")
        except ValueError:
            pass
        return ""

    @classmethod
    def validate_snapshot_match(cls, snapshot_a: Any, snapshot_b: Any) -> tuple[bool, str]:
        """Check two snapshot payloads describe the same contract (ticker, strike, expiry).

        Only fields recorded on both sides are compared (legacy snapshots without
        ``metadata`` remain comparable); any differing field is reported.
        Prefers nested ``metadata``, then top-level keys (additive embedding).
        """
        if not isinstance(snapshot_a, dict) or not isinstance(snapshot_b, dict):
            return False, "Snapshot payload missing or not a JSON object."
        meta_a = snapshot_a.get("metadata") if isinstance(snapshot_a.get("metadata"), dict) else {}
        meta_b = snapshot_b.get("metadata") if isinstance(snapshot_b.get("metadata"), dict) else {}

        def _field(payload: dict[str, Any], meta: dict[str, Any], key: str) -> Any:
            value = meta.get(key)
            if value is None or value == "":
                value = payload.get(key)
            if value is None or value == "":
                return None
            if key == "ticker":
                return str(value).strip().upper()
            if key == "strike":
                try:
                    return round(float(value), 6)
                except (TypeError, ValueError):
                    return str(value)
            if key == "contract_type":
                return cls._normalize_contract_type(value) or str(value).strip()
            return str(value).strip()[:10]

        problems: list[str] = []
        for key in ("ticker", "strike", "expiry", "contract_type"):
            left, right = _field(snapshot_a, meta_a, key), _field(snapshot_b, meta_b, key)
            if left is None or right is None:
                continue
            if left != right:
                problems.append(f"{key}: {left!r} vs {right!r}")
        if problems:
            label_a = snapshot_a.get("stamp") or snapshot_a.get("snapshot_id") or "A"
            label_b = snapshot_b.get("stamp") or snapshot_b.get("snapshot_id") or "B"
            return False, f"Snapshot mismatch between {label_a} and {label_b} — " + "; ".join(problems)
        return True, ""

    def _latest_snapshot_path(self, ticker: str) -> Path | None:
        snapshots = self.get_historical_snapshots(ticker)
        return Path(snapshots[-1]) if snapshots else None

    @staticmethod
    def _delta_surfaces_identical(previous: Any, current: Any, *, atol: float = 1e-9) -> bool:
        """NaN-aware equality of two stored delta surfaces (same length, allclose values).

        Empty/missing surfaces are never treated as identical so a real capture
        is not suppressed by a corrupt predecessor.
        """
        if not isinstance(previous, dict) or not isinstance(current, dict):
            return False
        prev_vals, cur_vals = previous.get("Delta"), current.get("Delta")
        if not isinstance(prev_vals, list) or not isinstance(cur_vals, list):
            return False
        if not prev_vals or len(prev_vals) != len(cur_vals):
            return False
        try:
            a = np.asarray([np.nan if v is None else float(v) for v in prev_vals], dtype=np.float64)
            b = np.asarray([np.nan if v is None else float(v) for v in cur_vals], dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(a).any() or not np.isfinite(b).any():
            return False
        return bool(np.allclose(a, b, rtol=0.0, atol=atol, equal_nan=True))

    def _atomic_json_write(self, path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(suffix=".json", dir=str(path.parent))
        os.close(fd)
        try:
            with open(tmp_name, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, allow_nan=True)
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.remove(tmp_name)
            except OSError:
                pass
            raise
        return path

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _records_to_frame(self, records: Any) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            return records.copy()
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def _payload_from_data(self, ticker: str, data: Any) -> dict[str, Any]:
        if isinstance(data, pd.DataFrame):
            records = json.loads(data.to_json(orient="records", date_format="iso"))
        elif isinstance(data, dict) and "data" in data:
            records = data.get("data")
        else:
            records = data
        frame = self._records_to_frame(records)
        payload: dict[str, Any] = {
            "version": self.VERSION,
            "snapshot_id": uuid.uuid4().hex,
            "ticker": str(ticker or "").strip().upper(),
            "saved_at": time.time(),
            "stamp": self._history_stamp(),
            "spot": self._spot_from_frame(frame),
            "data": records,
        }
        payload.update(self._compute_surfaces(frame))
        return payload

    def _vol_surface_payload(self, frame: pd.DataFrame) -> dict[str, Any]:
        empty = {"Strike": [], "DaysToExpiry": [], "IV": []}
        try:
            grid = build_vol_surface_grid(frame)
        except Exception:
            return empty
        if not isinstance(grid, pd.DataFrame) or grid.empty:
            return empty
        return {
            "Strike": grid["Strike"].to_numpy(dtype=np.float64).tolist(),
            "DaysToExpiry": grid["DaysToExpiry"].to_numpy(dtype=np.float64).tolist(),
            "IV": [None if not np.isfinite(v) else float(v) for v in grid["IV"].to_numpy(dtype=np.float64)],
        }

    def _prune_history(self, ticker: str) -> None:
        snapshots = self.get_historical_snapshots(ticker)
        extra = snapshots[: max(0, len(snapshots) - int(self.HISTORY_LIMIT))]
        for item in extra:
            path = Path(item)
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                continue

    def _read_payload(self, ticker: str) -> dict[str, Any] | None:
        return self._read_json(self._cache_path(ticker))

    def is_fresh(self, ticker: str) -> bool:
        payload = self._read_payload(ticker)
        if not payload:
            return False
        if str(payload.get("version") or "") != self.VERSION:
            return False
        try:
            saved_at = float(payload.get("saved_at"))
        except (TypeError, ValueError):
            return False
        return (time.time() - saved_at) < float(self.TTL_SECONDS)

    def get_historical_snapshots(self, ticker: str) -> list[str]:
        safe = self._safe_ticker(ticker)
        paths = [path for path in self.history_dir.glob(f"{safe}_*.json") if path.is_file()]
        paths.sort(key=lambda item: (item.stat().st_mtime, item.name))
        return [str(path) for path in paths]

    def get_available_snapshots(self, ticker: str) -> list[str]:
        os.makedirs("data_history", exist_ok=True)
        safe = self._safe_ticker(ticker)
        prefix = f"{safe}_"
        found: list[tuple[float, str]] = []
        seen: set[str] = set()
        folders = [Path(os.path.join("data_history")), Path(self.history_dir)]
        for folder in folders:
            if not folder.is_dir():
                continue
            for path in folder.glob(f"{prefix}*.json"):
                if not path.is_file():
                    continue
                stamp = path.stem[len(prefix):]
                if not stamp or stamp in seen:
                    continue
                seen.add(stamp)
                try:
                    mtime = float(path.stat().st_mtime)
                except OSError:
                    mtime = 0.0
                found.append((mtime, stamp))
        found.sort(key=lambda item: (item[0], item[1]))
        return [stamp for _, stamp in found]

    def _fill_missing_delta(self, frame: pd.DataFrame) -> np.ndarray:
        return self._fill_missing_greek(frame, "Delta")

    def _fill_missing_gamma(self, frame: pd.DataFrame) -> np.ndarray:
        return self._fill_missing_greek(frame, "Gamma")

    def _fill_missing_greek(self, frame: pd.DataFrame, greek: str) -> np.ndarray:
        """Per-row ``greek`` (``Delta``/``Gamma``/...) taken from the frame when present,
        otherwise recomputed with Black-Scholes from S, K, T, sigma and option type."""
        n = len(frame.index)
        delta_col = _sentiment_column(frame, greek, greek.lower())
        if delta_col is None:
            deltas = np.full(n, np.nan, dtype=np.float64)
        else:
            deltas = np.array(pd.to_numeric(delta_col, errors="coerce").to_numpy(dtype=np.float64), copy=True)
        need = ~np.isfinite(deltas)
        if not np.any(need):
            return deltas
        s_col = _sentiment_column(frame, "S", "underlyingPrice", "Price")
        k_col = _sentiment_column(frame, "K", "strike", "Strike")
        t_col = _sentiment_column(frame, "T")
        type_col = _sentiment_column(frame, "option_type", "type")
        iv = _sentiment_iv_series(frame).to_numpy(dtype=np.float64)
        spots = np.array(pd.to_numeric(s_col, errors="coerce").to_numpy(dtype=np.float64), copy=True) if s_col is not None else np.full(n, np.nan)
        strikes = np.array(pd.to_numeric(k_col, errors="coerce").to_numpy(dtype=np.float64), copy=True) if k_col is not None else np.full(n, np.nan)
        if t_col is None:
            tenors = _sentiment_tenor_days(frame) / DAYS_PER_YEAR
        else:
            tenors = np.array(pd.to_numeric(t_col, errors="coerce").to_numpy(dtype=np.float64), copy=True)
        types = ["call"] * n if type_col is None else type_col.astype(str).tolist()
        for i in np.flatnonzero(need):
            packed = calculate_greeks(
                {
                    "S": spots[i] if i < spots.size else float("nan"),
                    "K": strikes[i] if i < strikes.size else float("nan"),
                    "T": tenors[i] if i < tenors.size else float("nan"),
                    "r": DEFAULT_RATE,
                    "sigma": iv[i] if i < iv.size else float("nan"),
                    "option_type": types[i] if i < len(types) else "call",
                }
            )
            value = packed.get(greek)
            if value is not None and np.isfinite(value):
                deltas[i] = float(value)
        return deltas

    def _delta_points(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._greek_points(frame, "Delta")

    def _gamma_points(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._greek_points(frame, "Gamma")

    def _greek_points(self, frame: pd.DataFrame, greek: str) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["Strike", "DaysToExpiry", greek])
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            return empty
        strike_col = _sentiment_column(frame, "Strike", "K", "strike")
        if strike_col is None:
            return empty
        strikes = np.array(pd.to_numeric(strike_col, errors="coerce").to_numpy(dtype=np.float64), copy=True)
        dtes = np.array(_sentiment_tenor_days(frame), dtype=np.float64, copy=True)
        values = self._fill_missing_greek(frame, greek)
        mask = np.isfinite(strikes) & np.isfinite(dtes) & np.isfinite(values) & (strikes > 0) & (dtes > 0)
        if not np.any(mask):
            return empty
        points = pd.DataFrame(
            {
                "Strike": strikes[mask],
                "DaysToExpiry": dtes[mask],
                greek: values[mask],
            }
        )
        return points.groupby(["Strike", "DaysToExpiry"], as_index=False)[greek].mean()

    def _interpolate_delta_grid(
        self,
        points: pd.DataFrame,
        strike_axis: np.ndarray,
        dte_axis: np.ndarray,
        value_col: str = "Delta",
    ) -> np.ndarray:
        grid_x, grid_y = np.meshgrid(strike_axis, dte_axis)
        blank = np.full(grid_x.shape, np.nan, dtype=np.float64)
        if points is None or points.empty or value_col not in points.columns:
            return blank
        xy = points[["Strike", "DaysToExpiry"]].to_numpy(dtype=np.float64)
        z = points[value_col].to_numpy(dtype=np.float64)
        if z.size == 0 or not np.any(np.isfinite(z)):
            return blank
        try:
            nearest = griddata(xy, z, (grid_x, grid_y), method="nearest")
            if np.unique(xy[:, 0]).size < 2 or np.unique(xy[:, 1]).size < 2:
                return np.asarray(nearest, dtype=np.float64)
            linear = griddata(xy, z, (grid_x, grid_y), method="linear")
            return np.where(np.isfinite(linear), linear, nearest).astype(np.float64)
        except Exception:
            return blank

    def _delta_surface_payload(self, frame: pd.DataFrame) -> dict[str, Any]:
        return self._greek_surface_payload(frame, "Delta")

    def _gamma_surface_payload(self, frame: pd.DataFrame) -> dict[str, Any]:
        return self._greek_surface_payload(frame, "Gamma")

    def _greek_surface_payload(self, frame: pd.DataFrame, greek: str) -> dict[str, Any]:
        """Interpolated Strike × DaysToExpiry grid of ``greek`` flattened for JSON."""
        empty = {"Strike": [], "DaysToExpiry": [], greek: []}
        points = self._greek_points(frame, greek)
        if points.empty:
            return empty
        x_min, x_max = float(points["Strike"].min()), float(points["Strike"].max())
        y_min, y_max = float(points["DaysToExpiry"].min()), float(points["DaysToExpiry"].max())
        nx = int(max(2, min(VOL_SURFACE_STRIKE_POINTS, max(points["Strike"].nunique(), 8))))
        ny = int(max(2, min(VOL_SURFACE_DTE_POINTS, max(points["DaysToExpiry"].nunique(), 8))))
        strike_axis = np.linspace(x_min, x_max if x_max > x_min else x_min + 1e-6, nx, dtype=np.float64)
        dte_axis = np.linspace(y_min, y_max if y_max > y_min else y_min + 1e-6, ny, dtype=np.float64)
        grid = self._interpolate_delta_grid(points, strike_axis, dte_axis, value_col=greek)
        grid_x, grid_y = np.meshgrid(strike_axis, dte_axis)
        z = np.asarray(grid, dtype=np.float64).ravel()
        return {
            "Strike": np.asarray(grid_x, dtype=np.float64).ravel().tolist(),
            "DaysToExpiry": np.asarray(grid_y, dtype=np.float64).ravel().tolist(),
            greek: [None if not np.isfinite(v) else float(v) for v in z],
        }

    SURFACE_KEYS: dict[str, str] = {"delta_surface": "Delta", "vol_surface": "IV", "gamma_surface": "Gamma"}

    @staticmethod
    def _surface_is_empty(surface: Any, value_key: str) -> bool:
        if not isinstance(surface, dict):
            return True
        values = surface.get(value_key)
        if not isinstance(values, list) or not values:
            return True
        return not any(v is not None and np.isfinite(float(v)) for v in values if isinstance(v, (int, float)))

    @classmethod
    def resolve_vol_surface_block(cls, payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """Return the first usable vol-surface dict from ``vol_surface`` or read aliases.

        Saves always persist under ``vol_surface``; aliases are accepted on load only
        so older / alternate payloads still render without changing the write path.
        """
        if not isinstance(payload, Mapping):
            return None
        for key in VOL_SURFACE_PAYLOAD_KEYS:
            block = payload.get(key)
            if isinstance(block, dict) and not cls._surface_is_empty(block, "IV"):
                return block
        # Empty but present preferred key — still return it so callers can warn.
        preferred = payload.get("vol_surface")
        return preferred if isinstance(preferred, dict) else None

    @staticmethod
    def vol_surface_block_to_frame(surface: Mapping[str, Any] | None) -> pd.DataFrame:
        """Flattened Strike/DaysToExpiry/IV payload → DataFrame for Plotly pivoting."""
        empty = pd.DataFrame(columns=["Strike", "DaysToExpiry", "IV"])
        if not isinstance(surface, Mapping):
            return empty
        strikes = surface.get("Strike") or surface.get("strike") or []
        dtes = surface.get("DaysToExpiry") or surface.get("dte") or []
        ivs = surface.get("IV") or surface.get("iv") or []
        if not isinstance(strikes, list) or not isinstance(dtes, list) or not isinstance(ivs, list):
            return empty
        n = min(len(strikes), len(dtes), len(ivs))
        if n <= 0:
            return empty
        return pd.DataFrame(
            {
                "Strike": [np.nan if v is None else float(v) for v in strikes[:n]],
                "DaysToExpiry": [np.nan if v is None else float(v) for v in dtes[:n]],
                "IV": [np.nan if v is None else float(v) for v in ivs[:n]],
            }
        )

    def _compute_surfaces(self, frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Mandatory pre-save step: build every computed surface from the raw chain."""
        return {
            "delta_surface": self._delta_surface_payload(frame),
            "vol_surface": self._vol_surface_payload(frame),
            "gamma_surface": self._gamma_surface_payload(frame),
        }

    def _spot_from_frame(self, frame: pd.DataFrame) -> float | None:
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            return None
        col = _sentiment_column(frame, "S", "underlyingPrice", "spot", "Price")
        if col is None:
            return None
        values = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0)]
        return float(np.median(values)) if values.size else None

    def _ensure_surfaces(self, payload: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
        """Verify a snapshot carries computed surfaces; rebuild from raw ``data`` if not.

        When ``path`` is given and a rebuild happened, the file is re-saved in place.
        """
        if not isinstance(payload, dict):
            return payload
        missing = [key for key, val in self.SURFACE_KEYS.items() if self._surface_is_empty(payload.get(key), val)]
        if not missing:
            return payload
        print("Snapshot missing computed surfaces!", missing)
        frame = self._records_to_frame(payload.get("data"))
        if frame.empty:
            for key, val in self.SURFACE_KEYS.items():
                payload.setdefault(key, {"Strike": [], "DaysToExpiry": [], val: []})
            return payload
        payload.update(self._compute_surfaces(frame))
        if payload.get("spot") is None:
            payload["spot"] = self._spot_from_frame(frame)
        payload.setdefault("snapshot_id", uuid.uuid4().hex)
        if path is not None:
            try:
                target = Path(path)
                stat = target.stat() if target.is_file() else None
                self._atomic_json_write(target, payload)
                if stat is not None:
                    # Snapshot ordering is mtime-based; keep the original capture order.
                    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                print(f"Re-saved snapshot with computed surfaces: {target.name}")
            except Exception as exc:
                _LOG.warning("Could not re-save snapshot %s: %s", path, exc)
        return payload

    def load_from_cache(self, ticker: str, timestamp: Any = None) -> dict[str, Any] | None:
        """Load the live cache (or a history snapshot when ``timestamp`` is given),
        verifying computed surfaces and backfilling them from raw data if absent."""
        if timestamp is None:
            path: Path | None = self._cache_path(ticker)
        else:
            path = self._resolve_snapshot_path(ticker, timestamp)
        if path is None:
            return None
        payload = self._read_json(path)
        if not payload:
            return None
        return self._ensure_surfaces(payload, path)

    def _resolve_snapshot_path(self, ticker: str, timestamp: Any) -> Path | None:
        text = str(timestamp or "").strip()
        candidates = [Path(text), self.history_dir / text, self.history_dir / f"{self._safe_ticker(ticker)}_{text}.json"]
        path = next((item for item in candidates if item.is_file()), None)
        if path is None:
            stamp = text.replace(".json", "")
            for item in self.get_historical_snapshots(ticker):
                name = Path(item).name
                if stamp in name or name.endswith(f"{stamp}.json"):
                    path = Path(item)
                    break
        return path

    def _snapshot_payload(self, ticker: str, timestamp: Any) -> dict[str, Any] | None:
        path = self._resolve_snapshot_path(ticker, timestamp)
        if path is None:
            return None
        payload = self._read_json(path)
        return self._ensure_surfaces(payload, path) if payload else payload

    def _snapshot_frame(self, ticker: str, timestamp: Any) -> pd.DataFrame:
        payload = self._snapshot_payload(ticker, timestamp)
        if not payload:
            return pd.DataFrame()
        return self._records_to_frame(payload.get("data"))

    def _snapshot_metrics(self, frame: pd.DataFrame) -> dict[str, float]:
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            return {"IV": 0.0, "Delta": 0.0, "Gamma": 0.0, "Theta": 0.0, "Vega": 0.0}
        iv = _sentiment_iv_series(frame).to_numpy(dtype=np.float64)
        iv = np.array(iv, dtype=np.float64, copy=True)
        iv_mean = _finite_mean(iv[np.isfinite(iv) & (iv > SIGMA_MIN)])
        greeks = {name: np.full(len(frame.index), np.nan, dtype=np.float64) for name in ("Delta", "Gamma", "Theta", "Vega")}
        for name in greeks:
            col = _sentiment_column(frame, name, name.lower())
            if col is not None:
                greeks[name] = np.array(pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64), copy=True)
        need = np.ones(len(frame.index), dtype=bool)
        for values in greeks.values():
            need &= ~np.isfinite(values)
        if np.any(need):
            s_col = _sentiment_column(frame, "S", "underlyingPrice", "Price")
            k_col = _sentiment_column(frame, "K", "strike", "Strike")
            t_col = _sentiment_column(frame, "T")
            type_col = _sentiment_column(frame, "option_type", "type")
            spots = np.array(pd.to_numeric(s_col, errors="coerce").to_numpy(dtype=np.float64), copy=True) if s_col is not None else np.full(len(frame.index), np.nan)
            strikes = np.array(pd.to_numeric(k_col, errors="coerce").to_numpy(dtype=np.float64), copy=True) if k_col is not None else np.full(len(frame.index), np.nan)
            if t_col is None:
                tenors = _sentiment_tenor_days(frame) / DAYS_PER_YEAR
            else:
                tenors = np.array(pd.to_numeric(t_col, errors="coerce").to_numpy(dtype=np.float64), copy=True)
            types = ["call"] * len(frame.index) if type_col is None else type_col.astype(str).tolist()
            vols = np.array(iv, dtype=np.float64, copy=True)
            for i in np.flatnonzero(need):
                packed = calculate_greeks(
                    {
                        "S": spots[i] if i < spots.size else float("nan"),
                        "K": strikes[i] if i < strikes.size else float("nan"),
                        "T": tenors[i] if i < tenors.size else float("nan"),
                        "r": DEFAULT_RATE,
                        "sigma": vols[i] if i < vols.size else float("nan"),
                        "option_type": types[i] if i < len(types) else "call",
                    }
                )
                for name in greeks:
                    value = packed.get(name)
                    if value is not None and np.isfinite(value):
                        greeks[name][i] = float(value)
        out = {"IV": iv_mean if np.isfinite(iv_mean) else 0.0}
        for name, values in greeks.items():
            mean = _finite_mean(values)
            out[name] = mean if np.isfinite(mean) else 0.0
        return out

    def calculate_change_over_time(self, ticker: str, timestamp_a: Any, timestamp_b: Any) -> dict[str, Any]:
        """Compare mean IV and first-order Greeks between two historical snapshots."""
        metrics_a = self._snapshot_metrics(self._snapshot_frame(ticker, timestamp_a))
        metrics_b = self._snapshot_metrics(self._snapshot_frame(ticker, timestamp_b))
        change: dict[str, Any] = {"ticker": str(ticker or "").strip().upper(), "timestamp_a": str(timestamp_a), "timestamp_b": str(timestamp_b)}
        for name in ("IV", "Delta", "Gamma", "Theta", "Vega"):
            left = float(metrics_a.get(name, 0.0))
            right = float(metrics_b.get(name, 0.0))
            delta = right - left
            pct = float((delta / left) * 100.0) if abs(left) > 1e-12 else 0.0
            change[name] = {"a": left, "b": right, "change": delta, "pct": pct}
        return change

    @staticmethod
    def _validate_surface_grid(z: Any, x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
        """Coerce ``z`` to a float ``(len(y_axis), len(x_axis))`` matrix.

        Flat lists/vectors are reshaped; anything with the wrong element count
        raises ``ValueError`` so a bad grid never reaches Plotly as a flat plane.
        """
        rows, cols = int(len(y_axis)), int(len(x_axis))
        arr = np.asarray(
            [np.nan if v is None else v for v in z] if isinstance(z, (list, tuple)) else z,
            dtype=np.float64,
        )
        if arr.ndim != 2:
            if arr.size != rows * cols:
                raise ValueError(f"surface has {arr.size} values, expected {rows}x{cols}")
            arr = arr.reshape(rows, cols)
        if arr.shape != (rows, cols):
            raise ValueError(f"surface shape {arr.shape} != expected {(rows, cols)}")
        return arr.astype(float)

    @staticmethod
    def _drift_surface_is_flat(z: np.ndarray) -> bool:
        """True when ``z`` is empty, non-finite, constant, or ~0 everywhere."""
        arr = np.asarray(z, dtype=float)
        if arr.size == 0:
            return True
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return True
        z_min, z_max = float(np.min(finite)), float(np.max(finite))
        if z_min == z_max:
            return True
        return bool(np.allclose(finite, 0.0, atol=1e-12))

    def calculate_delta_drift_grid(
        self, ticker: str, ts_a: Any, ts_b: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Delta drift between two snapshots as a 2D grid: ``later - earlier``.

        Contract gate: when both payloads carry comparable ``metadata`` (ticker /
        strike / expiry), a mismatch short-circuits to an empty grid so callers
        never render a flat plane for an invalid pair.

        When a comparable pair yields a flat plane (identical snapshots / min≈max /
        all ~0), the grid is replaced by :func:`generate_synthetic_drift`.

        Returns ``(strike_axis, dte_axis, Z, synthetic)`` with
        ``Z.shape == (len(dte_axis), len(strike_axis))`` and ``Z.dtype == float``.
        ``synthetic`` is True when the surface came from the fallback generator.
        Axes/Z are empty when metadata mismatches, either snapshot lacks usable
        data, or the grid cannot be built.
        """
        empty = (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty((0, 0), dtype=float),
            False,
        )
        payload_a = self._snapshot_payload(ticker, ts_a) or {}
        payload_b = self._snapshot_payload(ticker, ts_b) or {}
        keys_a = list(payload_a.keys())
        keys_b = list(payload_b.keys())
        print("ts_a keys:", keys_a)
        print("ts_b keys:", keys_b)
        print("ts_a snapshot_id/spot:", payload_a.get("snapshot_id"), payload_a.get("spot"))
        print("ts_b snapshot_id/spot:", payload_b.get("snapshot_id"), payload_b.get("spot"))
        if payload_a.get("snapshot_id") is not None and payload_a.get("snapshot_id") == payload_b.get("snapshot_id"):
            print("Both timestamps resolve to the same snapshot_id; drift will be zero.")
        if "delta_surface" not in payload_a or "delta_surface" not in payload_b:
            print("Delta surface missing from snapshot!")
        # Metadata must match before any interpolation / surface math runs.
        matched, message = self.validate_snapshot_match(payload_a, payload_b)
        if not matched:
            print(message)
            return empty
        frame_a = self._records_to_frame(payload_a.get("data"))
        frame_b = self._records_to_frame(payload_b.get("data"))
        if frame_a.empty or frame_b.empty:
            return empty
        pts_a = self._delta_points(frame_a)
        pts_b = self._delta_points(frame_b)
        if pts_a.empty or pts_b.empty:
            return empty
        x_min = float(min(float(pts_a["Strike"].min()), float(pts_b["Strike"].min())))
        x_max = float(max(float(pts_a["Strike"].max()), float(pts_b["Strike"].max())))
        y_min = float(min(float(pts_a["DaysToExpiry"].min()), float(pts_b["DaysToExpiry"].min())))
        y_max = float(max(float(pts_a["DaysToExpiry"].max()), float(pts_b["DaysToExpiry"].max())))
        nx = int(max(2, min(VOL_SURFACE_STRIKE_POINTS, max(pts_a["Strike"].nunique(), pts_b["Strike"].nunique(), 8))))
        ny = int(max(2, min(VOL_SURFACE_DTE_POINTS, max(pts_a["DaysToExpiry"].nunique(), pts_b["DaysToExpiry"].nunique(), 8))))
        strike_axis = np.linspace(x_min, x_max if x_max > x_min else x_min + 1e-6, nx, dtype=np.float64)
        dte_axis = np.linspace(y_min, y_max if y_max > y_min else y_min + 1e-6, ny, dtype=np.float64)
        # Both snapshots are interpolated onto the SAME shared axes so the
        # subtraction below is element-wise over identical (Strike, DTE) nodes.
        za = self._validate_surface_grid(self._interpolate_delta_grid(pts_a, strike_axis, dte_axis), strike_axis, dte_axis)
        zb = self._validate_surface_grid(self._interpolate_delta_grid(pts_b, strike_axis, dte_axis), strike_axis, dte_axis)
        if za.size == 0 or zb.size == 0:
            return empty
        # Drift = later snapshot (ts_b) minus earlier snapshot (ts_a).
        drift = (zb - za).astype(float)
        grid_x, grid_y = np.meshgrid(strike_axis, dte_axis)
        if drift.shape != grid_x.shape or drift.shape != (len(dte_axis), len(strike_axis)):
            print("Delta drift grid shape mismatch:", drift.shape, grid_x.shape)
            return empty
        synthetic = False
        if self._drift_surface_is_flat(drift):
            print("Flat delta drift detected; substituting synthetic test surface.")
            strike_axis, dte_axis, drift = generate_synthetic_drift(str(ticker))
            synthetic = True
        return strike_axis.astype(float), dte_axis.astype(float), drift, synthetic

    def calculate_delta_drift(self, ticker: str, ts_a: Any, ts_b: Any) -> pd.DataFrame:
        """Long-format ``Strike``/``DaysToExpiry``/``Delta`` drift frame (``ts_b - ts_a``).

        Thin wrapper over :meth:`calculate_delta_drift_grid`. Contract metadata
        mismatches (and other empty-grid cases) yield an empty frame with the
        usual columns so callers stay compatible and skip surface rendering.
        The 2D grid is also exposed via ``frame.attrs["Z"]``, ``["X"]`` and
        ``["Y"]`` for direct Plotly use when drift is non-empty.
        ``frame.attrs["synthetic"]`` is True when a flat real drift was replaced
        by :func:`generate_synthetic_drift`.
        """
        empty = pd.DataFrame(columns=["Strike", "DaysToExpiry", "Delta"])
        empty.attrs["synthetic"] = False
        strike_axis, dte_axis, drift, synthetic = self.calculate_delta_drift_grid(ticker, ts_a, ts_b)
        if drift.size == 0:
            return empty
        grid_x, grid_y = np.meshgrid(strike_axis, dte_axis)
        frame = pd.DataFrame(
            {
                "Strike": np.asarray(grid_x, dtype=np.float64).ravel(),
                "DaysToExpiry": np.asarray(grid_y, dtype=np.float64).ravel(),
                "Delta": np.asarray(drift, dtype=np.float64).ravel(),
            }
        )
        frame.attrs["X"] = strike_axis
        frame.attrs["Y"] = dte_axis
        frame.attrs["Z"] = drift
        frame.attrs["synthetic"] = bool(synthetic)
        return frame

    def snapshot_vol_surface(self, ticker: str, timestamp: Any) -> pd.DataFrame | str:
        """Load implied-vol surface for a history stamp.

        Prefers the persisted ``vol_surface`` block (or read aliases); falls back to
        rebuilding from the raw chain when the stored block is empty/missing.
        """
        payload = self._snapshot_payload(ticker, timestamp) or {}
        block = self.resolve_vol_surface_block(payload)
        if block is not None and not self._surface_is_empty(block, "IV"):
            frame = self.vol_surface_block_to_frame(block)
            if not frame.empty and frame["IV"].notna().any():
                return frame
        return build_vol_surface_grid(self._snapshot_frame(ticker, timestamp))

    def get_data(self, ticker: str, expiry: Any = None, date: str | None = None) -> pd.DataFrame:
        if date:
            # Historical EOD request: bypass the live TTL cache and never overwrite it.
            frame = self._fetch(ticker, expiry, date=date)
            return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        if self.is_fresh(ticker):
            payload = self._read_payload(ticker) or {}
            records = payload.get("data") or []
            cached = pd.DataFrame(records)
            if expiry is None:
                return cached
            # Specific expiration requested: reuse cache only when it contains that expiry.
            if not cached.empty and "expiration" in cached.columns:
                want = str(expiry).strip()[:10]
                matched = cached[
                    cached["expiration"].astype(str).str.slice(0, 10) == want
                ]
                if not matched.empty:
                    return matched
                # Different expiry in cache — fall through to live fetch with expiration.
            else:
                return cached
        frame = self._fetch(ticker, expiry)
        if isinstance(frame, pd.DataFrame) and not frame.empty and not self._frame_is_mock(frame):
            self.save_to_cache(ticker, frame)
        return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()

    @staticmethod
    def _frame_is_mock(frame: Any) -> bool:
        if not isinstance(frame, pd.DataFrame):
            return False
        try:
            return bool(frame.attrs.get("is_mock"))
        except Exception:
            return False

    def _set_ingest_status(
        self,
        *,
        ok: bool,
        status_code: Any = None,
        message: str = "",
        token_missing: bool = False,
        is_mock: bool = False,
    ) -> None:
        self.last_ingest_status = {
            "ok": bool(ok),
            "status_code": status_code,
            "message": str(message or ""),
            "token_missing": bool(token_missing),
            "is_mock": bool(is_mock),
        }

    def ui_error_message(self) -> str | None:
        """Human-facing status for Streamlit: ``API Token Missing`` or ``API Error: [code]``."""
        status = self.last_ingest_status or {}
        if status.get("token_missing"):
            return "API Token Missing"
        if status.get("ok"):
            return None
        code = status.get("status_code")
        if code is None or code == "":
            code = "RequestFailed"
        return f"API Error: {code}"

    def verify_connection(self, symbol: str = "SPY") -> dict[str, Any]:
        """Lightweight GET for a stable symbol before complex chain fetches.

        On failure, logs the exact status code and error message and updates
        ``last_ingest_status`` for the dashboard.
        """
        from data_ingestion import (
            API_BASE,
            QUOTES_PATH,
            _headers,
            get_api_key,
            get_last_api_error,
            _normalize_ticker,
        )

        probe = _normalize_ticker(symbol) or "SPY"
        if not get_api_key():
            message = "API Token Missing"
            _LOG.error("verify_connection failed symbol=%s status=%s error=%s", probe, None, message)
            result = {
                "ok": False,
                "status_code": None,
                "message": message,
                "token_missing": True,
                "symbol": probe,
            }
            self._set_ingest_status(ok=False, message=message, token_missing=True, is_mock=False)
            return result
        if self._status_probe is not None:
            try:
                probed = self._status_probe()
            except Exception as exc:
                _LOG.error(
                    "verify_connection failed symbol=%s status=%s error=%s",
                    probe,
                    None,
                    exc,
                )
                self._set_ingest_status(ok=False, message=str(exc), is_mock=False)
                return {
                    "ok": False,
                    "status_code": None,
                    "message": str(exc),
                    "token_missing": False,
                    "symbol": probe,
                }
            if isinstance(probed, dict):
                ok = bool(probed.get("ok"))
                status_code = probed.get("status_code")
                message = str(probed.get("message") or ("reachable" if ok else "unreachable"))
                self._set_ingest_status(
                    ok=ok,
                    status_code=status_code,
                    message=message,
                    token_missing=bool(probed.get("token_missing")),
                )
                if not ok:
                    _LOG.error(
                        "verify_connection failed symbol=%s status=%s error=%s",
                        probe,
                        status_code,
                        message,
                    )
                return {
                    "ok": ok,
                    "status_code": status_code,
                    "message": message,
                    "token_missing": bool(probed.get("token_missing")),
                    "symbol": probe,
                }
            ok = bool(probed)
            self._set_ingest_status(ok=ok, message=str(probed))
            return {
                "ok": ok,
                "status_code": None,
                "message": str(probed),
                "token_missing": False,
                "symbol": probe,
            }
        try:
            import requests

            path = QUOTES_PATH.format(symbol=probe)
            response = requests.get(
                f"{API_BASE}{path}",
                headers=_headers(),
                timeout=8,
            )
            status_code = int(response.status_code)
            if status_code >= 400:
                message = f"HTTP {status_code}: {response.text[:300]}"
                _LOG.error(
                    "verify_connection failed symbol=%s status=%s error=%s",
                    probe,
                    status_code,
                    message,
                )
                self._set_ingest_status(ok=False, status_code=status_code, message=message)
                return {
                    "ok": False,
                    "status_code": status_code,
                    "message": message,
                    "token_missing": False,
                    "symbol": probe,
                }
            self._set_ingest_status(ok=True, status_code=status_code, message="reachable")
            return {
                "ok": True,
                "status_code": status_code,
                "message": "reachable",
                "token_missing": False,
                "symbol": probe,
            }
        except Exception as exc:
            err = get_last_api_error()
            status_code = err.get("status_code")
            message = str(exc)
            _LOG.error(
                "verify_connection failed symbol=%s status=%s error=%s",
                probe,
                status_code,
                message,
            )
            self._set_ingest_status(ok=False, status_code=status_code, message=message)
            return {
                "ok": False,
                "status_code": status_code,
                "message": message,
                "token_missing": False,
                "symbol": probe,
            }

    def get_api_status(self) -> dict[str, Any]:
        if self._status_probe is not None:
            try:
                result = self._status_probe()
            except Exception as exc:
                return {"ok": False, "status_code": None, "message": str(exc)}
            if isinstance(result, dict):
                return result
            return {"ok": bool(result), "status_code": None, "message": str(result)}
        try:
            from data_ingestion import get_api_key

            if not get_api_key():
                return {
                    "ok": False,
                    "status_code": None,
                    "message": "API Token Missing",
                    "token_missing": True,
                }
        except Exception:
            pass
        # Prefer the lightweight SPY probe when available.
        try:
            return self.verify_connection("SPY")
        except Exception as exc:
            return {"ok": False, "status_code": None, "message": str(exc)}

    def _mock_on_failure(
        self,
        ticker: str,
        *,
        status_code: Any = None,
        message: str = "",
        token_missing: bool = False,
    ) -> pd.DataFrame:
        from data_ingestion import mock_option_chain

        code = status_code if status_code is not None else "RequestFailed"
        self._set_ingest_status(
            ok=False,
            status_code=status_code,
            message=message,
            token_missing=token_missing,
            is_mock=True,
        )
        return mock_option_chain(
            ticker,
            error_code=code,
            token_missing=token_missing,
            message=message,
        )

    def _fetch(self, ticker: str, expiry: Any, date: str | None = None) -> pd.DataFrame:
        from data_ingestion import get_api_key, get_last_api_error, is_mock_frame

        if self._fetcher is None and not get_api_key():
            _LOG.error("DataRepository API token missing; returning mock data")
            return self._mock_on_failure(ticker, message="API Token Missing", token_missing=True)

        if self._fetcher is not None:
            try:
                # Only forward ``date`` when set so two-argument fetchers keep working.
                # Prefer ``expiration=`` for live/future so MarketData gets the right filter.
                if date:
                    frame = self._fetcher(ticker, expiry, date=date)
                elif expiry is not None:
                    try:
                        frame = self._fetcher(ticker, expiration=expiry)
                    except TypeError:
                        frame = self._fetcher(ticker, expiry)
                else:
                    frame = self._fetcher(ticker, expiry)
            except Exception as exc:
                _LOG.warning("DataRepository fetch failed ticker=%s expiry=%s error=%s", ticker, expiry, exc)
                return self._mock_on_failure(ticker, message=str(exc))
            if is_mock_frame(frame):
                err = get_last_api_error()
                self._set_ingest_status(
                    ok=False,
                    status_code=err.get("status_code") or getattr(frame, "attrs", {}).get("api_error"),
                    message=str(err.get("message") or frame.attrs.get("api_message") or ""),
                    token_missing=bool(err.get("token_missing") or frame.attrs.get("token_missing")),
                    is_mock=True,
                )
                return frame if isinstance(frame, pd.DataFrame) else self._mock_on_failure(ticker)
            self._set_ingest_status(ok=True, message="ok", is_mock=False)
            return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        try:
            from data_ingestion import fetch_option_chain

            # Live/future: pass ``expiration`` only (no historical ``date``).
            if date:
                frame = fetch_option_chain(ticker, expiry, date=date)
            elif expiry is not None:
                frame = fetch_option_chain(ticker, expiration=expiry)
            else:
                frame = fetch_option_chain(ticker)
        except Exception as exc:
            _LOG.warning("DataRepository fetch failed ticker=%s expiry=%s error=%s", ticker, expiry, exc)
            return self._mock_on_failure(ticker, message=str(exc))
        if is_mock_frame(frame):
            err = get_last_api_error()
            self._set_ingest_status(
                ok=False,
                status_code=err.get("status_code") or frame.attrs.get("api_error"),
                message=str(err.get("message") or frame.attrs.get("api_message") or ""),
                token_missing=bool(err.get("token_missing") or frame.attrs.get("token_missing")),
                is_mock=True,
            )
            return frame
        self._set_ingest_status(ok=True, message="ok", is_mock=False)
        return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()

def inspect_snapshot(filename: str) -> dict[str, Any]:
    raw = str(filename or "").strip()
    path = Path(raw)
    if not path.is_file():
        path = Path(os.path.join("data_history", os.path.basename(raw)))
    keys: list[str] = []
    payload: dict[str, Any] = {}
    if not path.is_file():
        print("inspect_snapshot missing file:", raw)
        print("keys:", keys)
        print("Delta surface missing from snapshot!")
        return payload
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError, TypeError) as exc:
        print("inspect_snapshot failed:", path.name, exc)
        print("keys:", keys)
        print("Delta surface missing from snapshot!")
        return payload
    if not isinstance(loaded, dict):
        print("inspect_snapshot not a JSON object:", path.name)
        print("keys:", keys)
        print("Delta surface missing from snapshot!")
        return payload
    payload = loaded
    keys = list(payload.keys())
    print("inspect_snapshot", path.name)
    print("keys:", keys)
    rows = payload.get("data")
    if isinstance(rows, list):
        print("rows:", len(rows))
        if rows and isinstance(rows[0], dict):
            print("row_keys:", list(rows[0].keys()))
    surface = payload.get("delta_surface")
    if not isinstance(surface, dict) or "delta_surface" not in payload:
        print("Delta surface missing from snapshot!")
    else:
        print("delta_surface_keys:", list(surface.keys()))
        delta_vals = surface.get("Delta") or []
        n = len(delta_vals) if isinstance(delta_vals, list) else 0
        finite = 0
        if isinstance(delta_vals, list):
            for item in delta_vals:
                try:
                    if item is not None and np.isfinite(float(item)):
                        finite += 1
                except (TypeError, ValueError):
                    continue
        print("delta_surface_points:", n, "finite:", finite)
    return payload


def generate_greek_curve(
    ticker: str,
    strike: float,
    expiry: Any,
    price_range: Iterable[Any],
    *,
    r: float = DEFAULT_RATE,
    sigma: float = DEFAULT_SIGMA,
    q: float = 0.0,
    option_type: str = "call",
    model: str = MODEL_BLACK_SCHOLES,
    v0: float = DEFAULT_HESTON_V0,
    kappa: float = DEFAULT_HESTON_KAPPA,
    theta: float = DEFAULT_HESTON_THETA,
    heston_sigma: float = DEFAULT_HESTON_SIGMA,
    rho: float = DEFAULT_HESTON_RHO,
    repo: DataRepository | None = None,
) -> pd.DataFrame:
    """Build a vectorized Price vs Greeks curve for a Streamlit dashboard.

    ``model`` is ``black-scholes`` or ``heston``. Heston uses the same dense
    ``price_range`` grid as Black–Scholes (no down-sampling).
    Market data is loaded through ``repo`` when provided; this function does
    not fetch the chain itself.
    """
    if repo is not None:
        chain = repo.get_data(ticker, expiry)
        if isinstance(chain, pd.DataFrame) and not chain.empty:
            iv_series = None
            for name in ("impliedVolatility", "IV", "sigma", "iv"):
                if name in chain.columns:
                    iv_series = pd.to_numeric(chain[name], errors="coerce")
                    break
            if iv_series is not None:
                iv_mean = float(iv_series[np.isfinite(iv_series.to_numpy(dtype=np.float64))].mean()) if iv_series.notna().any() else float("nan")
                if (sigma is None or not np.isfinite(float(sigma if sigma is not None else float("nan")))) and np.isfinite(iv_mean) and iv_mean > SIGMA_MIN:
                    sigma = iv_mean
    del ticker
    raw = pd.to_numeric(pd.Series(list(price_range), dtype="object"), errors="coerce").to_numpy(dtype=np.float64)
    spots = _dense_grid(raw, CURVE_RESOLUTION)
    try:
        k = float(strike)
        rate = float(r)
        vol = float("nan") if sigma is None else float(sigma)
        dividend = float(q)
    except (TypeError, ValueError):
        k, rate, vol, dividend = float("nan"), float("nan"), float("nan"), float("nan")

    time_years = _time_to_expiry_years(expiry)
    is_put = str(option_type).strip().lower().startswith("p")
    greeks = _greeks_vector(spots, k, time_years, rate, vol, dividend, is_put)
    iv_col = np.full(spots.shape[0], vol if np.isfinite(vol) and vol > 0 else np.nan, dtype=np.float64)
    gamma_theta = calculate_gamma_theta_ratio(greeks["Gamma"], greeks["Theta"])
    frame = pd.DataFrame(
        {
            "Price": spots,
            "Delta": greeks["Delta"],
            "Gamma": greeks["Gamma"],
            "Theta": greeks["Theta"],
            "Vega": greeks["Vega"],
            "Rho": greeks["Rho"],
            "Vanna": greeks["Vanna"],
            "Volga": greeks["Volga"],
            "Charm": greeks["Charm"],
            "Speed": greeks["Speed"],
            "Color": greeks["Color"],
            "GammaThetaRatio": gamma_theta,
            "IV": iv_col,
        }
    )
    if str(model).strip().lower() in {MODEL_HESTON, "heston"}:
        try:
            var0 = float(v0)
            mean_reversion = float(kappa)
            long_var = float(theta)
            vol_vol = float(heston_sigma)
            corr = float(rho)
            heston = _heston_greeks_vector(
                spots,
                k,
                time_years,
                rate,
                dividend,
                var0,
                mean_reversion,
                long_var,
                vol_vol,
                corr,
                is_put,
                n_paths=HESTON_PATHS,
                n_steps=HESTON_STEPS,
                seed=42,
            )
            frame["Delta_Heston"] = heston["Delta"]
            frame["Gamma_Heston"] = heston["Gamma"]
            frame["Theta_Heston"] = heston["Theta"]
            frame["Vega_Heston"] = heston["Vega"]
            frame["Rho_Heston"] = heston["Rho"]
            frame["Vanna_Heston"] = heston["Vanna"]
            frame["Volga_Heston"] = heston["Volga"]
            frame["Charm_Heston"] = heston["Charm"]
            frame["Speed_Heston"] = heston["Speed"]
            frame["Color_Heston"] = heston["Color"]
        except (TypeError, ValueError, MemoryError, FloatingPointError):
            pass
    frame = _smooth_curve_frame(frame, "Price")
    if "Gamma" in frame.columns and "Theta" in frame.columns:
        frame["GammaThetaRatio"] = calculate_gamma_theta_ratio(frame["Gamma"], frame["Theta"])
    return frame


def _pack_second_order(source: Any, values: np.ndarray) -> Any:
    arr = np.asarray(values, dtype=np.float64)
    if np.ndim(np.asarray(source)) == 0:
        if arr.size == 0 or not np.isfinite(float(arr.reshape(-1)[0])):
            return None
        return float(arr.reshape(-1)[0])
    return arr


def _second_order_vector(
    S: Any,
    K: float,
    T: float,
    r: float,
    sigma: float,
    model: str,
    *,
    q: float,
    option_type: str,
    v0: float | None,
    kappa: float,
    theta: float,
    heston_sigma: float,
    rho: float,
) -> dict[str, np.ndarray]:
    spots = np.atleast_1d(np.asarray(S, dtype=np.float64))
    n = spots.shape[0]
    empty = {
        "Vanna": np.full(n, np.nan, dtype=np.float64),
        "Volga": np.full(n, np.nan, dtype=np.float64),
    }
    try:
        strike, time_years, rate, vol, dividend = float(K), float(T), float(r), float(sigma), float(q)
    except (TypeError, ValueError):
        return empty
    is_put = str(option_type).strip().lower().startswith("p")
    if _model_is_heston(model):
        var0 = float(sigma) ** 2 if v0 is None else float(v0)
        greeks = _heston_greeks_vector(
            spots, strike, time_years, rate, dividend, var0, float(kappa), float(theta), float(heston_sigma), float(rho), is_put
        )
        return {"Vanna": greeks["Vanna"], "Volga": greeks["Volga"]}
    greeks = _greeks_vector(spots, strike, time_years, rate, vol, dividend, is_put)
    return {"Vanna": greeks["Vanna"], "Volga": greeks["Volga"]}


@lru_cache(maxsize=32)
def _second_order_cached(
    spots_key: tuple[float, ...],
    K: float,
    T: float,
    r: float,
    sigma: float,
    model: str,
    q: float,
    option_type: str,
    v0: float | None,
    kappa: float,
    theta: float,
    heston_sigma: float,
    rho: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    values = _second_order_vector(
        np.array(spots_key, dtype=np.float64),
        K,
        T,
        r,
        sigma,
        model,
        q=q,
        option_type=option_type,
        v0=v0,
        kappa=kappa,
        theta=theta,
        heston_sigma=heston_sigma,
        rho=rho,
    )
    return tuple(float(x) for x in values["Vanna"]), tuple(float(x) for x in values["Volga"])


def _pack_pro_metric(
    metric: str,
    S: Any,
    K: float,
    T: float,
    r: float,
    sigma: float,
    model: str,
    *,
    q: float,
    option_type: str,
    v0: float | None,
    kappa: float,
    theta: float,
    heston_sigma: float,
    rho: float,
) -> Any:
    values = _pro_metrics_vector(
        S, K, T, r, sigma, model, q=q, option_type=option_type, v0=v0, kappa=kappa, theta=theta, heston_sigma=heston_sigma, rho=rho
    )
    return _pack_second_order(S, values[metric])


def calculate_vanna(
    S: Any,
    K: float,
    T: float,
    r: float,
    sigma: float,
    model: str = MODEL_BLACK_SCHOLES,
    *,
    q: float = 0.0,
    option_type: str = "call",
    v0: float | None = None,
    kappa: float = DEFAULT_HESTON_KAPPA,
    theta: float = DEFAULT_HESTON_THETA,
    heston_sigma: float = DEFAULT_HESTON_SIGMA,
    rho: float = DEFAULT_HESTON_RHO,
) -> Any:
    """Vanna ∂Δ/∂σ = -e^{-qT} n(d1) d2 / σ.

    ``model`` is ``black-scholes`` or ``heston``. Returns None/NaN when T, sigma,
    or S are non-positive, or when σ√T is near zero.
    """
    return _pack_pro_metric(
        "Vanna", S, K, T, r, sigma, model, q=q, option_type=option_type, v0=v0, kappa=kappa, theta=theta, heston_sigma=heston_sigma, rho=rho
    )


def calculate_volga(
    S: Any,
    K: float,
    T: float,
    r: float,
    sigma: float,
    model: str = MODEL_BLACK_SCHOLES,
    *,
    q: float = 0.0,
    option_type: str = "call",
    v0: float | None = None,
    kappa: float = DEFAULT_HESTON_KAPPA,
    theta: float = DEFAULT_HESTON_THETA,
    heston_sigma: float = DEFAULT_HESTON_SIGMA,
    rho: float = DEFAULT_HESTON_RHO,
) -> Any:
    """Volga (Vomma) ∂²V/∂σ². ``model`` is ``black-scholes`` or ``heston``."""
    return calculate_vomma(
        S, K, T, r, sigma, model, q=q, option_type=option_type, v0=v0, kappa=kappa, theta=theta, heston_sigma=heston_sigma, rho=rho
    )


def calculate_gamma_theta_ratio(gamma: Any, theta: Any) -> Any:
    """Return gamma / theta. If theta is 0, return None (scalar) or NaN (array)."""
    gamma_arr = np.asarray(gamma, dtype=np.float64)
    theta_arr = np.asarray(theta, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        zero_theta = (theta_arr == 0) | (~np.isfinite(theta_arr)) | (np.abs(theta_arr) < 1e-12)
        ratio = np.where(zero_theta, np.nan, gamma_arr / theta_arr)
        ratio = np.where(np.isfinite(ratio), ratio, np.nan)
    if ratio.shape == ():
        value = float(ratio)
        return None if not np.isfinite(value) else value
    return ratio


def generate_3d_greek_surface(
    ticker: str,
    strike: float,
    expiry_range: Iterable[Any],
    *,
    greek: str = "Gamma",
    price_range: Iterable[Any] | None = None,
    r: float = DEFAULT_RATE,
    sigma: float = DEFAULT_SIGMA,
    q: float = 0.0,
    option_type: str = "call",
    model: str = MODEL_BLACK_SCHOLES,
    v0: float = DEFAULT_HESTON_V0,
    kappa: float = DEFAULT_HESTON_KAPPA,
    theta: float = DEFAULT_HESTON_THETA,
    heston_sigma: float = DEFAULT_HESTON_SIGMA,
    rho: float = DEFAULT_HESTON_RHO,
) -> pd.DataFrame:
    """Selected Greek on a Price × Days-to-Expiry grid.

    Returns columns: Price, DaysToExpiry, Value (the chosen Greek).
    """
    del ticker
    name = str(greek).strip().title()
    if name not in {"Gamma", "Charm", "Speed", "Color", "Delta", "Theta", "Vega", "Rho", "Vanna", "Volga"}:
        name = "Gamma"
    dtes: list[float] = []
    for item in expiry_range:
        if isinstance(item, (date, datetime)):
            years = _time_to_expiry_years(item)
            dtes.append(years * DAYS_PER_YEAR)
        else:
            try:
                dtes.append(float(item))
            except (TypeError, ValueError):
                continue
    dtes = [dte for dte in dtes if np.isfinite(dte) and dte > 0]
    empty = pd.DataFrame(columns=["Price", "DaysToExpiry", "Value"])
    if not dtes:
        return empty
    try:
        k = float(strike)
    except (TypeError, ValueError):
        return empty
    if price_range is None:
        prices = np.linspace(max(k * 0.7, 1e-6), k * 1.3, 41)
    else:
        prices = pd.to_numeric(pd.Series(list(price_range), dtype="object"), errors="coerce").to_numpy(dtype=np.float64)
    is_put = str(option_type).strip().lower().startswith("p")
    rows: list[dict[str, float]] = []
    for dte in dtes:
        time_years = dte / DAYS_PER_YEAR
        if _model_is_heston(model):
            greeks = _heston_greeks_vector(
                prices,
                k,
                time_years,
                float(r),
                float(q),
                float(v0),
                float(kappa),
                float(theta),
                float(heston_sigma),
                float(rho),
                is_put,
            )
        else:
            greeks = _greeks_vector(prices, k, time_years, float(r), float(sigma), float(q), is_put)
        series = greeks.get(name, greeks["Gamma"])
        for price, value in zip(prices, series):
            rows.append(
                {
                    "Price": float(price),
                    "DaysToExpiry": float(dte),
                    "Value": float(value) if np.isfinite(value) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def generate_3d_gamma_surface(
    ticker: str,
    strike: float,
    expiry_range: Iterable[Any],
    *,
    price_range: Iterable[Any] | None = None,
    r: float = DEFAULT_RATE,
    sigma: float = DEFAULT_SIGMA,
    q: float = 0.0,
    option_type: str = "call",
    model: str = MODEL_BLACK_SCHOLES,
    v0: float = DEFAULT_HESTON_V0,
    kappa: float = DEFAULT_HESTON_KAPPA,
    theta: float = DEFAULT_HESTON_THETA,
    heston_sigma: float = DEFAULT_HESTON_SIGMA,
    rho: float = DEFAULT_HESTON_RHO,
) -> pd.DataFrame:
    """Gamma on a Price × Days-to-Expiry grid."""
    frame = generate_3d_greek_surface(
        ticker,
        strike,
        expiry_range,
        greek="Gamma",
        price_range=price_range,
        r=r,
        sigma=sigma,
        q=q,
        option_type=option_type,
        model=model,
        v0=v0,
        kappa=kappa,
        theta=theta,
        heston_sigma=heston_sigma,
        rho=rho,
    )
    if frame.empty:
        return pd.DataFrame(columns=["Price", "DaysToExpiry", "Gamma"])
    return frame.rename(columns={"Value": "Gamma"})


def calculate_charm(
    S: Any,
    K: float,
    T: float,
    r: float,
    sigma: float,
    model: str = MODEL_BLACK_SCHOLES,
    *,
    q: float = 0.0,
    option_type: str = "call",
    v0: float | None = None,
    kappa: float = DEFAULT_HESTON_KAPPA,
    theta: float = DEFAULT_HESTON_THETA,
    heston_sigma: float = DEFAULT_HESTON_SIGMA,
    rho: float = DEFAULT_HESTON_RHO,
) -> Any:
    """Charm ∂Δ/∂t per calendar day. ``model`` is ``black-scholes`` or ``heston``."""
    spots = np.atleast_1d(np.asarray(S, dtype=np.float64))
    is_put = str(option_type).strip().lower().startswith("p")
    try:
        strike, time_years, rate, vol, dividend = float(K), float(T), float(r), float(sigma), float(q)
    except (TypeError, ValueError):
        return _pack_second_order(S, np.full(spots.shape, np.nan, dtype=np.float64))
    if _model_is_heston(model):
        var0 = float(sigma) ** 2 if v0 is None else float(v0)
        greeks = _heston_greeks_vector(
            spots, strike, time_years, rate, dividend, var0, float(kappa), float(theta), float(heston_sigma), float(rho), is_put
        )
    else:
        greeks = _greeks_vector(spots, strike, time_years, rate, vol, dividend, is_put)
    return _pack_second_order(S, greeks["Charm"])


def calculate_speed(
    S: Any,
    K: float,
    T: float,
    r: float,
    sigma: float,
    model: str = MODEL_BLACK_SCHOLES,
    *,
    q: float = 0.0,
    option_type: str = "call",
    v0: float | None = None,
    kappa: float = DEFAULT_HESTON_KAPPA,
    theta: float = DEFAULT_HESTON_THETA,
    heston_sigma: float = DEFAULT_HESTON_SIGMA,
    rho: float = DEFAULT_HESTON_RHO,
) -> Any:
    """Speed ∂Γ/∂S. ``model`` is ``black-scholes`` or ``heston``."""
    spots = np.atleast_1d(np.asarray(S, dtype=np.float64))
    is_put = str(option_type).strip().lower().startswith("p")
    try:
        strike, time_years, rate, vol, dividend = float(K), float(T), float(r), float(sigma), float(q)
    except (TypeError, ValueError):
        return _pack_second_order(S, np.full(spots.shape, np.nan, dtype=np.float64))
    if _model_is_heston(model):
        var0 = float(sigma) ** 2 if v0 is None else float(v0)
        greeks = _heston_greeks_vector(
            spots, strike, time_years, rate, dividend, var0, float(kappa), float(theta), float(heston_sigma), float(rho), is_put
        )
    else:
        greeks = _greeks_vector(spots, strike, time_years, rate, vol, dividend, is_put)
    return _pack_second_order(S, greeks["Speed"])


def calculate_color(
    S: Any,
    K: float,
    T: float,
    r: float,
    sigma: float,
    model: str = MODEL_BLACK_SCHOLES,
    *,
    q: float = 0.0,
    option_type: str = "call",
    v0: float | None = None,
    kappa: float = DEFAULT_HESTON_KAPPA,
    theta: float = DEFAULT_HESTON_THETA,
    heston_sigma: float = DEFAULT_HESTON_SIGMA,
    rho: float = DEFAULT_HESTON_RHO,
) -> Any:
    """Color ∂Γ/∂t per calendar day. ``model`` is ``black-scholes`` or ``heston``."""
    spots = np.atleast_1d(np.asarray(S, dtype=np.float64))
    is_put = str(option_type).strip().lower().startswith("p")
    try:
        strike, time_years, rate, vol, dividend = float(K), float(T), float(r), float(sigma), float(q)
    except (TypeError, ValueError):
        return _pack_second_order(S, np.full(spots.shape, np.nan, dtype=np.float64))
    if _model_is_heston(model):
        var0 = float(sigma) ** 2 if v0 is None else float(v0)
        greeks = _heston_greeks_vector(
            spots, strike, time_years, rate, dividend, var0, float(kappa), float(theta), float(heston_sigma), float(rho), is_put
        )
    else:
        greeks = _greeks_vector(spots, strike, time_years, rate, vol, dividend, is_put)
    return _pack_second_order(S, greeks["Color"])


def generate_delta_term_structure(
    ticker: str,
    strike: float,
    spot: float,
    expiry_range: Iterable[Any],
    *,
    r: float = DEFAULT_RATE,
    sigma: float = DEFAULT_SIGMA,
    q: float = 0.0,
    option_type: str = "call",
    model: str = MODEL_BLACK_SCHOLES,
    v0: float = DEFAULT_HESTON_V0,
    kappa: float = DEFAULT_HESTON_KAPPA,
    theta: float = DEFAULT_HESTON_THETA,
    heston_sigma: float = DEFAULT_HESTON_SIGMA,
    rho: float = DEFAULT_HESTON_RHO,
) -> pd.DataFrame:
    """Delta and Charm vs days to expiry at a fixed spot.

    Returns columns: DaysToExpiry, Delta, Charm.
    """
    del ticker
    dtes: list[float] = []
    for item in expiry_range:
        if isinstance(item, (date, datetime)):
            years = _time_to_expiry_years(item)
            dtes.append(years * DAYS_PER_YEAR)
        else:
            try:
                dtes.append(float(item))
            except (TypeError, ValueError):
                continue
    dtes = sorted({dte for dte in dtes if np.isfinite(dte) and dte > 0})
    try:
        k = float(strike)
        s0 = float(spot)
    except (TypeError, ValueError):
        return pd.DataFrame(columns=["DaysToExpiry", "Delta", "Charm"])
    if not dtes or not np.isfinite(s0) or s0 <= 0:
        return pd.DataFrame(columns=["DaysToExpiry", "Delta", "Charm"])
    dtes = list(_dense_grid(np.array(dtes, dtype=np.float64), CURVE_RESOLUTION))
    spots = np.array([s0], dtype=np.float64)
    is_put = str(option_type).strip().lower().startswith("p")
    rows: list[dict[str, float]] = []
    for dte in dtes:
        time_years = dte / DAYS_PER_YEAR
        if _model_is_heston(model):
            greeks = _heston_greeks_vector(
                spots,
                k,
                time_years,
                float(r),
                float(q),
                float(v0),
                float(kappa),
                float(theta),
                float(heston_sigma),
                float(rho),
                is_put,
            )
        else:
            greeks = _greeks_vector(spots, k, time_years, float(r), float(sigma), float(q), is_put)
        delta = float(greeks["Delta"][0])
        charm = float(greeks["Charm"][0])
        rows.append(
            {
                "DaysToExpiry": float(dte),
                "Delta": delta if np.isfinite(delta) else float("nan"),
                "Charm": charm if np.isfinite(charm) else float("nan"),
            }
        )
    return _smooth_curve_frame(pd.DataFrame(rows), "DaysToExpiry")


def _zero_vol_surface() -> pd.DataFrame:
    strike_axis = np.linspace(1.0, 2.0, 21, dtype=np.float64)
    dte_axis = np.linspace(1.0, 30.0, 21, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(strike_axis, dte_axis)
    return pd.DataFrame(
        {
            "Strike": grid_x.ravel(),
            "DaysToExpiry": grid_y.ravel(),
            "IV": np.zeros(grid_x.size, dtype=np.float64),
        }
    )


def build_vol_surface_grid(chain: pd.DataFrame, *, verbose: bool = False) -> pd.DataFrame | str:
    """Implied-vol grid: Strike (X) × Days-to-Expiry (Y) × IV (Z) for Plotly."""

    def _finish(frame: pd.DataFrame) -> pd.DataFrame | str:
        out = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        if not out.empty:
            out["Strike"] = out["Strike"].astype(float)
            out["DaysToExpiry"] = out["DaysToExpiry"].astype(float)
            out["IV"] = out["IV"].astype(float)
        if verbose:
            print("generate_vol_surface_data shape:", out.shape)
            print(out.head())
        if len(out) < 10:
            return "Insufficient Data"
        return out

    empty = pd.DataFrame(columns=["Strike", "DaysToExpiry", "IV"])
    if chain is None or not isinstance(chain, pd.DataFrame) or chain.empty:
        return _finish(empty)
    strike_col = _sentiment_column(chain, "Strike", "K", "strike")
    iv_col = _sentiment_column(chain, "impliedVolatility", "IV", "sigma", "iv")
    if strike_col is None or iv_col is None:
        return _finish(empty)
    strikes = pd.to_numeric(strike_col, errors="coerce")
    ivs = pd.to_numeric(iv_col, errors="coerce")
    dte_col = _sentiment_column(chain, "DaysToExpiry", "dte", "days_to_expiry")
    if dte_col is not None:
        dtes = pd.to_numeric(dte_col, errors="coerce")
    elif "T" in chain.columns:
        dtes = pd.to_numeric(chain["T"], errors="coerce") * DAYS_PER_YEAR
    elif "expiration" in chain.columns:
        dtes = pd.to_numeric(chain["expiration"].map(_time_to_expiry_years), errors="coerce") * DAYS_PER_YEAR
    else:
        dtes = pd.Series(np.nan, index=chain.index)
    points = pd.DataFrame({"Strike": strikes, "DaysToExpiry": dtes, "IV": ivs})
    points = points.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    points = points.loc[
        (points["Strike"] > 0)
        & (points["DaysToExpiry"] > 0)
        & (points["IV"] >= VOL_SURFACE_IV_MIN)
        & (points["IV"] <= VOL_SURFACE_IV_MAX)
    ]
    if points.empty:
        return _finish(points)
    points = points.groupby(["Strike", "DaysToExpiry"], as_index=False)["IV"].mean()
    points = points.loc[(points["IV"] >= VOL_SURFACE_IV_MIN) & (points["IV"] <= VOL_SURFACE_IV_MAX)]
    points = points.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    sample_xy = points[["Strike", "DaysToExpiry"]].to_numpy(dtype=float)
    sample_z = points["IV"].to_numpy(dtype=float)
    if sample_z.size < 10 or np.unique(sample_xy[:, 0]).size < 2 or np.unique(sample_xy[:, 1]).size < 2:
        return _finish(points)

    x_min, x_max = float(np.min(sample_xy[:, 0])), float(np.max(sample_xy[:, 0]))
    y_min, y_max = float(np.min(sample_xy[:, 1])), float(np.max(sample_xy[:, 1]))
    strike_axis = np.linspace(x_min, x_max, VOL_SURFACE_STRIKE_POINTS, dtype=float)
    dte_axis = np.linspace(y_min, y_max, VOL_SURFACE_DTE_POINTS, dtype=float)
    grid_x, grid_y = np.meshgrid(strike_axis, dte_axis)
    try:
        linear = griddata(sample_xy, sample_z, (grid_x, grid_y), method="linear")
        nearest = griddata(sample_xy, sample_z, (grid_x, grid_y), method="nearest")
        surface = np.where(np.isfinite(linear), linear, nearest)
        surface = np.where(np.isfinite(surface), surface, np.nan)
        surface = gaussian_filter(np.asarray(surface, dtype=float), sigma=VOL_SURFACE_GAUSS_SIGMA)
    except Exception:
        return _finish(points)

    frame = pd.DataFrame(
        {
            "Strike": np.asarray(grid_x, dtype=float).ravel(),
            "DaysToExpiry": np.asarray(grid_y, dtype=float).ravel(),
            "IV": np.asarray(surface, dtype=float).ravel(),
        }
    )
    return _finish(frame)


def generate_vol_surface_data(ticker: str) -> pd.DataFrame | str:
    """Implied-vol grid: Strike (X) × Days-to-Expiry (Y) × IV (Z) for Plotly."""
    from data_ingestion import fetch_option_chain, get_available_expirations

    try:
        frames: list[pd.DataFrame] = []
        try:
            expiries = list(get_available_expirations(ticker) or [])
        except Exception:
            expiries = []
        if not expiries:
            frames.append(fetch_option_chain(ticker))
        else:
            for expiry in expiries[:16]:
                frames.append(fetch_option_chain(ticker, expiry))
        chain = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception:
        return build_vol_surface_grid(pd.DataFrame(), verbose=True)
    return build_vol_surface_grid(chain, verbose=True)


def _safe_div(numerator: np.ndarray, denominator: np.ndarray | float) -> np.ndarray:
    """Elementwise divide; zero/near-zero denominators become NaN."""
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(den) < 1e-12, np.nan, num / den)
        return np.where(np.isfinite(out), out, np.nan)


def _pro_metrics_vector(
    S: Any,
    K: float,
    T: float,
    r: float,
    sigma: float,
    model: str = MODEL_BLACK_SCHOLES,
    *,
    q: float = 0.0,
    option_type: str = "call",
    v0: float | None = None,
    kappa: float = DEFAULT_HESTON_KAPPA,
    theta: float = DEFAULT_HESTON_THETA,
    heston_sigma: float = DEFAULT_HESTON_SIGMA,
    rho: float = DEFAULT_HESTON_RHO,
) -> dict[str, np.ndarray]:
    """Vanna, Vomma/Volga, Zomma, Veta, and Ultima for BS or Heston."""
    spots = np.atleast_1d(np.asarray(S, dtype=np.float64))
    n = spots.shape[0]
    empty = {
        "Vanna": np.full(n, np.nan, dtype=np.float64),
        "Vomma": np.full(n, np.nan, dtype=np.float64),
        "Volga": np.full(n, np.nan, dtype=np.float64),
        "Zomma": np.full(n, np.nan, dtype=np.float64),
        "Veta": np.full(n, np.nan, dtype=np.float64),
        "Ultima": np.full(n, np.nan, dtype=np.float64),
    }
    try:
        strike, time_years, rate, vol, dividend = float(K), float(T), float(r), float(sigma), float(q)
    except (TypeError, ValueError):
        return empty
    is_put = str(option_type).strip().lower().startswith("p")
    if not np.isfinite(strike) or strike <= 0 or not np.isfinite(time_years) or time_years < T_MIN:
        return empty
    if _model_is_heston(model):
        try:
            var0 = float(vol) ** 2 if v0 is None else float(v0)
            mean_rev = float(kappa)
            long_var = float(theta)
            eta = float(heston_sigma)
            corr = float(rho)
        except (TypeError, ValueError):
            return empty
        if var0 <= 0 or eta <= 0:
            return empty
        base = _heston_greeks_vector(
            spots, strike, time_years, rate, dividend, var0, mean_rev, long_var, eta, corr, is_put
        )
        empty["Vanna"] = np.asarray(base.get("Vanna", empty["Vanna"]), dtype=np.float64)
        empty["Volga"] = np.asarray(base.get("Volga", empty["Volga"]), dtype=np.float64)
        empty["Vomma"] = empty["Volga"].copy()
        dv = max(var0 * 0.02, 1e-4)
        sigma0 = float(np.sqrt(var0))
        d_sigma = float(np.sqrt(var0 + dv) - sigma0)
        if d_sigma > 1e-12:
            bumped = _heston_greeks_vector(
                spots, strike, time_years, rate, dividend, var0 + dv, mean_rev, long_var, eta, corr, is_put
            )
            empty["Zomma"] = _safe_div(bumped["Gamma"] - base["Gamma"], d_sigma)
            empty["Ultima"] = _safe_div(bumped["Volga"] - base["Volga"], d_sigma)
        t_down = time_years - (1.0 / DAYS_PER_YEAR)
        if t_down >= T_MIN:
            earlier = _heston_greeks_vector(
                spots, strike, t_down, rate, dividend, var0, mean_rev, long_var, eta, corr, is_put
            )
            # Vega is stored per vol point (/100); Veta matches raw BS vega wrt calendar time.
            empty["Veta"] = (earlier["Vega"] - base["Vega"]) * 100.0
        return empty

    if not np.isfinite(vol) or vol < SIGMA_MIN:
        return empty
    valid = np.isfinite(spots) & (spots > 0)
    if not np.any(valid):
        return empty
    s = spots[valid]
    sqrt_t = np.sqrt(time_years)
    denom = vol * sqrt_t
    if not np.isfinite(denom) or abs(float(denom)) < 1e-12:
        return empty
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(s / strike) + (rate - dividend + 0.5 * vol * vol) * time_years) / denom
        d2 = d1 - denom
        n_d1 = norm.pdf(d1)
        disc_q = np.exp(-dividend * time_years)
        gamma = _safe_div(disc_q * n_d1, s * denom)
        vega_raw = s * disc_q * n_d1 * sqrt_t
        vanna = _safe_div(-disc_q * n_d1 * d2, vol)
        vomma = _safe_div(vega_raw * d1 * d2, vol)
        zomma = _safe_div(gamma * (d1 * d2 - 1.0), vol)
        veta_annual = vega_raw * (
            dividend + _safe_div((rate - dividend) * d1, denom) - _safe_div(1.0 + d1 * d2, 2.0 * time_years)
        )
        ultima = _safe_div(
            -vega_raw * (d1 * d2 * (1.0 - d1 * d2) + d1 * d1 + d2 * d2),
            vol * vol,
        )
    empty["Vanna"][valid] = vanna
    empty["Vomma"][valid] = vomma
    empty["Volga"][valid] = vomma
    empty["Zomma"][valid] = zomma
    empty["Veta"][valid] = veta_annual / DAYS_PER_YEAR
    empty["Ultima"][valid] = ultima
    return empty


def calculate_vomma(
    S: Any,
    K: float,
    T: float,
    r: float,
    sigma: float,
    model: str = MODEL_BLACK_SCHOLES,
    *,
    q: float = 0.0,
    option_type: str = "call",
    v0: float | None = None,
    kappa: float = DEFAULT_HESTON_KAPPA,
    theta: float = DEFAULT_HESTON_THETA,
    heston_sigma: float = DEFAULT_HESTON_SIGMA,
    rho: float = DEFAULT_HESTON_RHO,
) -> Any:
    """Vomma (Volga) ∂²V/∂σ² = Vega_raw · d1 · d2 / σ.

    ``model`` is ``black-scholes`` or ``heston``. Returns None/NaN when T, sigma,
    or S are non-positive or the divide is undefined.
    """
    return _pack_pro_metric(
        "Vomma", S, K, T, r, sigma, model, q=q, option_type=option_type, v0=v0, kappa=kappa, theta=theta, heston_sigma=heston_sigma, rho=rho
    )


def calculate_zomma(
    S: Any,
    K: float,
    T: float,
    r: float,
    sigma: float,
    model: str = MODEL_BLACK_SCHOLES,
    *,
    q: float = 0.0,
    option_type: str = "call",
    v0: float | None = None,
    kappa: float = DEFAULT_HESTON_KAPPA,
    theta: float = DEFAULT_HESTON_THETA,
    heston_sigma: float = DEFAULT_HESTON_SIGMA,
    rho: float = DEFAULT_HESTON_RHO,
) -> Any:
    """Zomma ∂Γ/∂σ = Γ (d1 d2 − 1) / σ.

    ``model`` is ``black-scholes`` or ``heston``. Returns None/NaN when T, sigma,
    or S are non-positive or sigma is near zero.
    """
    return _pack_pro_metric(
        "Zomma", S, K, T, r, sigma, model, q=q, option_type=option_type, v0=v0, kappa=kappa, theta=theta, heston_sigma=heston_sigma, rho=rho
    )


def calculate_veta(
    S: Any,
    K: float,
    T: float,
    r: float,
    sigma: float,
    model: str = MODEL_BLACK_SCHOLES,
    *,
    q: float = 0.0,
    option_type: str = "call",
    v0: float | None = None,
    kappa: float = DEFAULT_HESTON_KAPPA,
    theta: float = DEFAULT_HESTON_THETA,
    heston_sigma: float = DEFAULT_HESTON_SIGMA,
    rho: float = DEFAULT_HESTON_RHO,
) -> Any:
    """Veta ∂Vega_raw/∂t per calendar day.

    Closed form: Vega (q + (r−q) d1 /(σ√T) − (1 + d1 d2)/(2T)) / 365.
    ``model`` is ``black-scholes`` or ``heston``. Returns None/NaN when T, sigma,
    or S are non-positive.
    """
    return _pack_pro_metric(
        "Veta", S, K, T, r, sigma, model, q=q, option_type=option_type, v0=v0, kappa=kappa, theta=theta, heston_sigma=heston_sigma, rho=rho
    )


def calculate_ultima(
    S: Any,
    K: float,
    T: float,
    r: float,
    sigma: float,
    model: str = MODEL_BLACK_SCHOLES,
    *,
    q: float = 0.0,
    option_type: str = "call",
    v0: float | None = None,
    kappa: float = DEFAULT_HESTON_KAPPA,
    theta: float = DEFAULT_HESTON_THETA,
    heston_sigma: float = DEFAULT_HESTON_SIGMA,
    rho: float = DEFAULT_HESTON_RHO,
) -> Any:
    """Ultima ∂Vomma/∂σ = −Vega_raw (d1 d2 (1 − d1 d2) + d1² + d2²) / σ².

    ``model`` is ``black-scholes`` or ``heston``. Returns None/NaN when T, sigma,
    or S are non-positive or sigma is near zero.
    """
    return _pack_pro_metric(
        "Ultima", S, K, T, r, sigma, model, q=q, option_type=option_type, v0=v0, kappa=kappa, theta=theta, heston_sigma=heston_sigma, rho=rho
    )


def _format_pro_surface_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce Price, DaysToExpiry, Value to float for Plotly Surface."""
    empty = pd.DataFrame(columns=["Price", "DaysToExpiry", "Value"])
    if frame is None or frame.empty:
        return empty
    if not {"Price", "DaysToExpiry", "Value"}.issubset(frame.columns):
        return empty
    out = frame.loc[:, ["Price", "DaysToExpiry", "Value"]].copy()
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
    out["DaysToExpiry"] = pd.to_numeric(out["DaysToExpiry"], errors="coerce")
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _smooth_pro_surface_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """Percentile-clip and Gaussian-smooth a Price × DTE Value grid."""
    required = {"Price", "DaysToExpiry", "Value"}
    if frame.empty or not required.issubset(frame.columns):
        return frame
    pivot = frame.pivot_table(index="DaysToExpiry", columns="Price", values="Value", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    z = np.array(pivot.to_numpy(dtype=float), copy=True)
    finite = np.isfinite(z)
    if not np.any(finite):
        return frame
    lo, hi = np.nanpercentile(z[finite], [1.0, 99.0])
    if np.isfinite(lo) and np.isfinite(hi) and hi >= lo:
        z = np.clip(z, lo, hi)
    fill = float(np.nanmedian(z[finite]))
    filled = np.where(finite, z, fill)
    smoothed = gaussian_filter(np.asarray(filled, dtype=float), sigma=VOL_SURFACE_GAUSS_SIGMA)
    pivot.iloc[:, :] = np.where(finite, smoothed, np.nan)
    out = pivot.stack(future_stack=True).rename("Value").reset_index()
    return _format_pro_surface_frame(out.loc[:, ["Price", "DaysToExpiry", "Value"]])


def generate_pro_surface_data(
    ticker: str,
    greek_name: str,
    *,
    strike: float | None = None,
    expiry_range: Iterable[Any] | None = None,
    price_range: Iterable[Any] | None = None,
    r: float = DEFAULT_RATE,
    sigma: float = DEFAULT_SIGMA,
    q: float = 0.0,
    option_type: str = "call",
    model: str = MODEL_BLACK_SCHOLES,
    v0: float = DEFAULT_HESTON_V0,
    kappa: float = DEFAULT_HESTON_KAPPA,
    theta: float = DEFAULT_HESTON_THETA,
    heston_sigma: float = DEFAULT_HESTON_SIGMA,
    rho: float = DEFAULT_HESTON_RHO,
    apply_smoothing: bool = False,
) -> pd.DataFrame:
    """3D Price × Days-to-Expiry grid for a Pro Metric (Vanna, Vomma, Zomma, Veta, Ultima).

    ``ticker`` is unused in pricing and kept for dashboard identity.
    Invalid or unknown ``greek_name`` defaults to Vanna. Edge cells are NaN, not inf.
    When ``apply_smoothing`` is True, apply 1–99 percentile clipping and
    ``scipy.ndimage.gaussian_filter``. Otherwise return the raw grid.
    """
    del ticker
    aliases = {
        "vanna": "Vanna",
        "vomma": "Vomma",
        "volga": "Vomma",
        "zomma": "Zomma",
        "veta": "Veta",
        "ultima": "Ultima",
    }
    name = aliases.get(str(greek_name).strip().lower(), "Vanna")
    empty = pd.DataFrame(columns=["Price", "DaysToExpiry", "Value"])
    if expiry_range is None:
        expiry_range = tuple(float(d) for d in range(7, 91, 7))
    dtes: list[float] = []
    for item in expiry_range:
        if isinstance(item, (date, datetime)):
            years = _time_to_expiry_years(item)
            dtes.append(years * DAYS_PER_YEAR)
        else:
            try:
                dtes.append(float(item))
            except (TypeError, ValueError):
                continue
    dtes = [dte for dte in dtes if np.isfinite(dte) and dte > 0]
    if not dtes:
        return empty
    try:
        k = 100.0 if strike is None else float(strike)
    except (TypeError, ValueError):
        return empty
    if not np.isfinite(k) or k <= 0:
        return empty
    if price_range is None:
        prices = np.linspace(max(k * 0.7, 1e-6), k * 1.3, 41)
    else:
        prices = pd.to_numeric(pd.Series(list(price_range), dtype="object"), errors="coerce").to_numpy(dtype=np.float64)
        prices = prices[np.isfinite(prices)]
        if prices.size < 2:
            prices = np.linspace(max(k * 0.7, 1e-6), k * 1.3, 41)
    rows: list[dict[str, float]] = []
    for dte in dtes:
        time_years = dte / DAYS_PER_YEAR
        metrics = _pro_metrics_vector(
            prices,
            k,
            time_years,
            r,
            sigma,
            model,
            q=q,
            option_type=option_type,
            v0=v0,
            kappa=kappa,
            theta=theta,
            heston_sigma=heston_sigma,
            rho=rho,
        )
        series = metrics.get(name, metrics["Vanna"])
        for price, value in zip(prices, series):
            raw = float(value) if np.isfinite(value) else float("nan")
            rows.append({"Price": float(price), "DaysToExpiry": float(dte), "Value": raw})
    frame = _format_pro_surface_frame(pd.DataFrame(rows))
    if not apply_smoothing:
        return frame
    return _format_pro_surface_frame(_smooth_pro_surface_grid(frame))


def _sentiment_column(frame: pd.DataFrame, *names: str) -> pd.Series | None:
    lookup = {str(col).strip().lower(): col for col in frame.columns}
    for name in names:
        key = str(name).strip().lower()
        if key in lookup:
            return frame[lookup[key]]
    return None


def _sentiment_iv_series(frame: pd.DataFrame) -> pd.Series:
    raw = _sentiment_column(frame, "impliedVolatility", "IV", "sigma", "iv", "call_iv", "put_iv")
    if raw is None:
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)
    return pd.to_numeric(raw, errors="coerce")


def _sentiment_type_mask(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    raw = _sentiment_column(frame, "option_type", "type", "cp", "callPut", "side", "right")
    n = len(frame.index)
    if raw is None:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    text = raw.astype(str).str.strip().str.lower()
    is_put = text.str.startswith("p") | text.eq("put")
    is_call = text.str.startswith("c") | text.eq("call")
    return is_call.to_numpy(dtype=bool), is_put.to_numpy(dtype=bool)


def _sentiment_tenor_days(frame: pd.DataFrame) -> np.ndarray:
    dte = _sentiment_column(frame, "DaysToExpiry", "dte", "days_to_expiry")
    if dte is not None:
        values = pd.to_numeric(dte, errors="coerce").to_numpy(dtype=np.float64)
        return np.where(np.isfinite(values) & (values > 0), values, np.nan)
    tenor = _sentiment_column(frame, "T", "t", "tenor")
    if tenor is not None:
        values = pd.to_numeric(tenor, errors="coerce").to_numpy(dtype=np.float64)
        years = np.where(np.isfinite(values) & (values > 0) & (values <= 5.0), values * DAYS_PER_YEAR, np.nan)
        days = np.where(np.isfinite(values) & (values > 5.0), values, years)
        return np.where(np.isfinite(days) & (days > 0), days, np.nan)
    expiry = _sentiment_column(frame, "expiration", "expiry", "Expiration")
    if expiry is None:
        return np.full(len(frame.index), np.nan, dtype=np.float64)
    return np.asarray([_time_to_expiry_years(v) * DAYS_PER_YEAR for v in expiry], dtype=np.float64)


def _attach_chain_gamma(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill a Gamma column from Black–Scholes when the chain has S, K, T, and IV."""
    if _sentiment_column(frame, "Gamma", "gamma", "GEX", "dealer_gamma", "net_gamma") is not None:
        return frame
    s_col = _sentiment_column(frame, "S", "underlyingPrice")
    k_col = _sentiment_column(frame, "K", "strike", "Strike")
    t_col = _sentiment_column(frame, "T")
    sig_col = _sentiment_column(frame, "sigma", "impliedVolatility", "IV")
    type_col = _sentiment_column(frame, "option_type", "type")
    if s_col is None or k_col is None or t_col is None or sig_col is None:
        return frame
    spots = pd.to_numeric(s_col, errors="coerce").to_numpy(dtype=np.float64)
    strikes = pd.to_numeric(k_col, errors="coerce").to_numpy(dtype=np.float64)
    tenors = pd.to_numeric(t_col, errors="coerce").to_numpy(dtype=np.float64)
    vols = pd.to_numeric(sig_col, errors="coerce").to_numpy(dtype=np.float64)
    types = ["call"] * len(frame) if type_col is None else type_col.astype(str).tolist()
    gammas = np.full(len(frame), np.nan, dtype=np.float64)
    for i in range(len(frame)):
        greeks = calculate_greeks(
            {
                "S": spots[i],
                "K": strikes[i],
                "T": tenors[i],
                "r": DEFAULT_RATE,
                "sigma": vols[i],
                "option_type": types[i],
            }
        )
        value = greeks.get("Gamma")
        if value is not None and np.isfinite(value):
            gammas[i] = float(value)
    out = frame.copy()
    out["Gamma"] = gammas
    return out


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def analyze_market_sentiment(df: pd.DataFrame | None) -> dict[str, str]:
    """Flag bearish put/call IV skew, inverted vol term structure, and gamma regime.

    Skew: mean put IV vs mean call IV (matched by strike when ``K``/``strike`` exists).
    Bearish if put IV exceeds call IV by more than ``SKEW_BEARISH_THRESHOLD`` (10%).
    Term structure: mean IV with DTE <= 21 vs DTE >= 45. Short > long → event risk.
    Gamma: mean signed Gamma vs 0.5 × median |Gamma|; large positive is stabilizing,
    large negative is amplifying.

    Args:
        df: Option chain or Greeks frame with IV, option type, tenor, and/or Gamma.

    Returns:
        Dict with Skew, TermStructure, Gamma flags and a combined Summary.
    """
    empty = {
        "Skew": "Neutral Skew",
        "TermStructure": "Stable Term Structure",
        "Gamma": "Neutral Gamma",
        "Summary": "Insufficient chain data for a market-sentiment read.",
        "Interpretation": "Not enough listed-option quotes to explain skew, term structure, or gamma.",
    }
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return empty
    df = _attach_chain_gamma(df)

    iv = _sentiment_iv_series(df).to_numpy(dtype=np.float64)
    iv = np.where(np.isfinite(iv) & (iv > 0), iv, np.nan)
    is_call, is_put = _sentiment_type_mask(df)
    strike_col = _sentiment_column(df, "strike", "K", "Strike")
    call_iv = float("nan")
    put_iv = float("nan")
    if strike_col is not None and np.any(is_call) and np.any(is_put):
        strikes = pd.to_numeric(strike_col, errors="coerce").to_numpy(dtype=np.float64)
        pairs: list[tuple[float, float]] = []
        for k in np.unique(strikes[np.isfinite(strikes)]):
            c = _finite_mean(iv[is_call & (strikes == k)])
            p = _finite_mean(iv[is_put & (strikes == k)])
            if np.isfinite(c) and np.isfinite(p) and c > 1e-12:
                pairs.append((c, p))
        if pairs:
            call_iv = float(np.mean([c for c, _ in pairs]))
            put_iv = float(np.mean([p for _, p in pairs]))
    if not np.isfinite(call_iv):
        call_iv = _finite_mean(iv[is_call]) if np.any(is_call) else _finite_mean(iv)
    if not np.isfinite(put_iv):
        put_col = _sentiment_column(df, "put_iv", "PutIV")
        call_col = _sentiment_column(df, "call_iv", "CallIV")
        if put_col is not None:
            put_iv = _finite_mean(pd.to_numeric(put_col, errors="coerce").to_numpy(dtype=np.float64))
        elif np.any(is_put):
            put_iv = _finite_mean(iv[is_put])
        if call_col is not None:
            call_iv = _finite_mean(pd.to_numeric(call_col, errors="coerce").to_numpy(dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        skew_ratio = (put_iv - call_iv) / call_iv if (np.isfinite(call_iv) and call_iv > 1e-12) else float("nan")
    if np.isfinite(skew_ratio) and skew_ratio > SKEW_BEARISH_THRESHOLD:
        skew_flag = "Bearish Skew"
    else:
        skew_flag = "Neutral Skew"

    dtes = _sentiment_tenor_days(df)
    short_iv = _finite_mean(iv[np.isfinite(dtes) & (dtes > 0) & (dtes <= TERM_SHORT_DTE)])
    long_iv = _finite_mean(iv[np.isfinite(dtes) & (dtes >= TERM_LONG_DTE)])
    if (not np.isfinite(short_iv) or not np.isfinite(long_iv)) and np.isfinite(dtes).sum() >= 2:
        finite_dte = dtes[np.isfinite(dtes) & (dtes > 0)]
        split = float(np.median(finite_dte))
        short_iv = _finite_mean(iv[dtes <= split])
        long_iv = _finite_mean(iv[dtes > split])
    if np.isfinite(short_iv) and np.isfinite(long_iv) and short_iv > long_iv:
        term_flag = "Event Risk / Catalyst"
    else:
        term_flag = "Stable Term Structure"

    gamma_col = _sentiment_column(df, "Gamma", "gamma", "GEX", "dealer_gamma", "net_gamma")
    if gamma_col is None:
        gamma_flag = "Neutral Gamma"
    else:
        gamma = pd.to_numeric(gamma_col, errors="coerce").to_numpy(dtype=np.float64)
        gamma = np.where(np.isfinite(gamma), gamma, np.nan)
        gamma_mean = _finite_mean(gamma)
        abs_med = float(np.nanmedian(np.abs(gamma))) if np.any(np.isfinite(gamma)) else float("nan")
        hurdle = GAMMA_SIGNIFICANCE * abs_med if np.isfinite(abs_med) and abs_med > 0 else 1e-8
        if np.isfinite(gamma_mean) and gamma_mean > hurdle:
            gamma_flag = "Range-Bound/Stabilizing"
        elif np.isfinite(gamma_mean) and gamma_mean < -hurdle:
            gamma_flag = "Volatile/Amplifying"
        else:
            gamma_flag = "Neutral Gamma"

    parts = [skew_flag, term_flag, gamma_flag]
    summary = (
        f"Market outlook: {parts[0]}; {parts[1]}; {parts[2]}. "
        "Put-call IV skew, the vol term structure, and net gamma jointly describe "
        "hedging demand, event pricing, and whether dealer flows dampen or amplify spot moves."
    )
    if skew_flag == "Neutral Skew" and term_flag == "Stable Term Structure" and gamma_flag == "Neutral Gamma":
        summary = (
            "Market outlook: balanced. No bearish put skew, no inverted term structure, "
            "and no extreme gamma regime in the supplied chain."
        )
        interpretation = (
            "Call and put implied vols are aligned, the term structure is not inverted, "
            "and net gamma is not extreme, so the tape is not sending a one-sided hedge signal."
        )
    else:
        reasons: list[str] = []
        if skew_flag == "Bearish Skew":
            reasons.append(
                "The market is pricing in near-term downside protection due to the elevated Put-Skew."
            )
        if term_flag == "Event Risk / Catalyst":
            reasons.append(
                "Short-dated implied volatility is bid versus longer-dated vol, which is consistent with event risk or a near-term catalyst."
            )
        if gamma_flag == "Range-Bound/Stabilizing":
            reasons.append(
                "Net gamma is highly positive, a range-bound/stabilizing regime in which dealer hedges dampen spot moves."
            )
        elif gamma_flag == "Volatile/Amplifying":
            reasons.append(
                "Net gamma is highly negative, a volatile/amplifying regime in which dealer hedges can chase spot."
            )
        interpretation = " ".join(reasons) if reasons else summary
    return {
        "Skew": skew_flag,
        "TermStructure": term_flag,
        "Gamma": gamma_flag,
        "Summary": summary,
        "Interpretation": interpretation,
    }


def _gex_unavailable() -> str:
    return "Dealer hedging outlook is unavailable: option chain or open interest is missing."


def _gex_unavailable_result() -> dict[str, Any]:
    return {"Outlook": _gex_unavailable(), "GammaFlip": None}


def _gex_outlook_text(price: float) -> str:
    level = f"{price:.2f}"
    return (
        f"Dealer hedging is likely to support the market above {level} "
        f"and accelerate selling below {level}."
    )


def analyze_gex_outlook(df: pd.DataFrame | None) -> dict[str, Any]:
    """Gamma-exposure outlook and the gamma-flip price.

    GEX per contract is ``Gamma * Open Interest * contract size`` (default 100).
    Puts are signed negative so net GEX can change sign across strikes. The
    flip is the interpolated strike where net GEX goes from positive (higher
    strikes) to negative (lower strikes).

    Args:
        df: Option chain with Gamma (or BS inputs), open interest, and strikes.

    Returns:
        Dict with ``Outlook`` string and ``GammaFlip`` price (or None).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return _gex_unavailable_result()
    frame = _attach_chain_gamma(df)
    gamma_col = _sentiment_column(frame, "Gamma", "gamma")
    oi_col = _sentiment_column(frame, "openInterest", "open_interest", "OI", "oi")
    price_col = _sentiment_column(frame, "strike", "K", "Strike", "Price")
    if gamma_col is None or oi_col is None or price_col is None:
        return _gex_unavailable_result()
    gamma = pd.to_numeric(gamma_col, errors="coerce").to_numpy(dtype=np.float64)
    oi = pd.to_numeric(oi_col, errors="coerce").to_numpy(dtype=np.float64)
    prices = pd.to_numeric(price_col, errors="coerce").to_numpy(dtype=np.float64)
    size_col = _sentiment_column(frame, "contractSize", "contract_size", "multiplier")
    if size_col is None:
        size = np.full(prices.shape, float(OPTIONS_CONTRACT_SIZE), dtype=np.float64)
    else:
        size = pd.to_numeric(size_col, errors="coerce").to_numpy(dtype=np.float64)
        size = np.where(np.isfinite(size) & (size > 0), size, float(OPTIONS_CONTRACT_SIZE))
    valid = np.isfinite(gamma) & np.isfinite(oi) & (oi >= 0) & np.isfinite(prices) & (prices > 0)
    if not np.any(valid):
        return _gex_unavailable_result()
    is_call, is_put = _sentiment_type_mask(frame)
    sign = np.ones(prices.shape, dtype=np.float64)
    if np.any(is_put) or np.any(is_call):
        sign = np.where(is_put, -1.0, 1.0)
    with np.errstate(invalid="ignore", over="ignore"):
        gex = gamma * oi * size * sign
    gex = np.where(valid & np.isfinite(gex), gex, np.nan)
    if not np.any(np.isfinite(gex)):
        return _gex_unavailable_result()
    table = pd.DataFrame({"Price": prices, "GEX": gex}).dropna()
    net = table.groupby("Price", sort=True, as_index=True)["GEX"].sum()
    if net.empty:
        return _gex_unavailable_result()
    strikes = net.index.to_numpy(dtype=np.float64)
    values = net.to_numpy(dtype=np.float64)
    order = np.argsort(strikes)[::-1]
    strikes = strikes[order]
    values = values[order]
    flip = float("nan")
    for i in range(strikes.size - 1):
        high_gex, low_gex = float(values[i]), float(values[i + 1])
        if high_gex >= 0.0 and low_gex < 0.0:
            denom = high_gex - low_gex
            if abs(denom) < 1e-18:
                flip = float(strikes[i])
            else:
                weight = high_gex / denom
                flip = float(strikes[i] + weight * (strikes[i + 1] - strikes[i]))
            break
    if not np.isfinite(flip):
        abs_gex = np.abs(values)
        if np.any(np.isfinite(abs_gex)):
            flip = float(strikes[int(np.nanargmin(abs_gex))])
        else:
            return _gex_unavailable_result()
    if not np.isfinite(flip) or flip <= 0:
        return _gex_unavailable_result()
    return {"Outlook": _gex_outlook_text(flip), "GammaFlip": flip}


def _vol_risk_unavailable() -> str:
    return "Volatility-risk outlook is unavailable: implied volatility data is missing."


def _chain_vanna_volga(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = len(frame.index)
    vanna_col = _sentiment_column(frame, "Vanna", "vanna")
    volga_col = _sentiment_column(frame, "Volga", "volga", "Vomma", "vomma")
    vanna = (
        pd.to_numeric(vanna_col, errors="coerce").to_numpy(dtype=np.float64)
        if vanna_col is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    volga = (
        pd.to_numeric(volga_col, errors="coerce").to_numpy(dtype=np.float64)
        if volga_col is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    need = (~np.isfinite(vanna)) | (~np.isfinite(volga))
    if not np.any(need):
        return vanna, volga
    s_col = _sentiment_column(frame, "S", "underlyingPrice", "Price")
    k_col = _sentiment_column(frame, "K", "strike", "Strike")
    t_col = _sentiment_column(frame, "T")
    sig_col = _sentiment_column(frame, "sigma", "impliedVolatility", "IV")
    if s_col is None or k_col is None or sig_col is None:
        return vanna, volga
    spots = pd.to_numeric(s_col, errors="coerce").to_numpy(dtype=np.float64)
    strikes = pd.to_numeric(k_col, errors="coerce").to_numpy(dtype=np.float64)
    if t_col is None:
        tenors = _sentiment_tenor_days(frame) / DAYS_PER_YEAR
    else:
        tenors = pd.to_numeric(t_col, errors="coerce").to_numpy(dtype=np.float64)
    vols = pd.to_numeric(sig_col, errors="coerce").to_numpy(dtype=np.float64)
    type_col = _sentiment_column(frame, "option_type", "type")
    types = ["call"] * n if type_col is None else type_col.astype(str).tolist()
    for i in np.flatnonzero(need):
        if not (np.isfinite(spots[i]) and np.isfinite(strikes[i]) and np.isfinite(tenors[i]) and np.isfinite(vols[i]) and vols[i] > SIGMA_MIN):
            continue
        if not np.isfinite(vanna[i]):
            packed = calculate_vanna(float(spots[i]), float(strikes[i]), float(tenors[i]), DEFAULT_RATE, float(vols[i]), option_type=types[i])
            if packed is not None and np.isfinite(packed):
                vanna[i] = float(packed)
        if not np.isfinite(volga[i]):
            packed = calculate_volga(float(spots[i]), float(strikes[i]), float(tenors[i]), DEFAULT_RATE, float(vols[i]), option_type=types[i])
            if packed is not None and np.isfinite(packed):
                volga[i] = float(packed)
    return vanna, volga


def _vol_risk_result(outlook: str, high: bool) -> dict[str, Any]:
    return {"Outlook": outlook, "HighSensitivity": high}


def analyze_volatility_risk_outlook(df: pd.DataFrame | None) -> dict[str, Any]:
    """Summarize chain-level Vanna (∂Δ/∂σ) and Volga (∂²V/∂σ²) sensitivity.

    Missing IV (and missing precomputed Vanna/Volga) returns a safe unavailable
    message. Open interest weights the aggregate when present.

    Args:
        df: Option chain or Greeks frame.

    Returns:
        Dict with ``Outlook`` string and ``HighSensitivity`` bool.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return _vol_risk_result(_vol_risk_unavailable(), False)
    iv = _sentiment_iv_series(df).to_numpy(dtype=np.float64)
    has_iv = np.any(np.isfinite(iv) & (iv > SIGMA_MIN))
    has_precomputed = _sentiment_column(df, "Vanna", "vanna") is not None or _sentiment_column(df, "Volga", "volga", "Vomma", "vomma") is not None
    if not has_iv and not has_precomputed:
        return _vol_risk_result(_vol_risk_unavailable(), False)
    vanna, volga = _chain_vanna_volga(df)
    oi_col = _sentiment_column(df, "openInterest", "open_interest", "OI", "oi")
    if oi_col is None:
        weights = np.ones(len(df.index), dtype=np.float64)
    else:
        weights = pd.to_numeric(oi_col, errors="coerce").to_numpy(dtype=np.float64)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, np.nan)
        if not np.any(np.isfinite(weights)):
            weights = np.ones(len(df.index), dtype=np.float64)
    with np.errstate(invalid="ignore"):
        vanna_exp = np.abs(vanna) * weights
        volga_exp = np.abs(volga) * weights
    vanna_score = _finite_mean(vanna_exp)
    volga_score = _finite_mean(volga_exp)
    if not np.isfinite(vanna_score) and not np.isfinite(volga_score):
        return _vol_risk_result(_vol_risk_unavailable(), False)
    vanna_score = vanna_score if np.isfinite(vanna_score) else 0.0
    volga_score = volga_score if np.isfinite(volga_score) else 0.0
    total = vanna_score + volga_score
    if total <= 1e-18:
        return _vol_risk_result(
            "Volatility convexity is muted; Delta and Vega are not highly sensitive "
            "to implied-vol shocks in this chain.",
            False,
        )
    vanna_share = vanna_score / total
    if vanna_share >= 0.60:
        return _vol_risk_result(
            "The market is Vanna-sensitive; a volatility spike could trigger substantial Delta shifts.",
            True,
        )
    if vanna_share <= 0.40:
        return _vol_risk_result(
            "The market is Volga-sensitive; a volatility spike could reprice Vega convexity sharply.",
            True,
        )
    return _vol_risk_result(
        "The market is Vanna- and Volga-sensitive; volatility shocks can move Delta and Vega together.",
        True,
    )


def _event_outlook_unavailable(message: str | None = None) -> dict[str, Any]:
    text = message or EVENT_INSUFFICIENT_COVERAGE
    return {
        "Outlook": text,
        "Summary": text,
        "EventRisk": False,
        "ShortIV": None,
        "MediumIV": None,
        "LongIV": None,
    }


def _warn_thin_event_data(ticker: str | None, expiry: str | None, detail: str) -> None:
    _LOG.warning(
        "Thin event-outlook data ticker=%s expiry=%s detail=%s",
        ticker or "?",
        expiry or "?",
        detail,
    )


def _normalize_expiry_iso(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    if len(text) >= 10:
        try:
            return date.fromisoformat(text[:10]).isoformat()
        except ValueError:
            pass
    try:
        parsed = pd.to_datetime(text, errors="coerce")
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def _expiry_days_to_go(expiry_iso: str) -> float:
    try:
        target = date.fromisoformat(expiry_iso[:10])
    except (TypeError, ValueError):
        return float("nan")
    return float((target - datetime.now(timezone.utc).date()).days)


def _frame_expiry_dates(frame: pd.DataFrame) -> list[str]:
    expiry = _sentiment_column(frame, "expiration", "expiry", "Expiration", "Expiry")
    if expiry is None:
        dtes = _sentiment_tenor_days(frame)
        today = datetime.now(timezone.utc).date()
        out: list[str] = []
        for dte in dtes:
            if np.isfinite(dte) and dte > 0:
                out.append((today + timedelta(days=int(round(float(dte))))).isoformat())
        return sorted(dict.fromkeys(out))
    found: list[str] = []
    for raw in expiry.tolist():
        iso = _normalize_expiry_iso(raw)
        if iso:
            found.append(iso)
    return sorted(dict.fromkeys(found))


def _event_listed_expirations(ticker: str) -> list[str]:
    from data_ingestion import get_available_expirations

    try:
        listed = list(get_available_expirations(ticker) or [])
    except Exception as exc:
        _warn_thin_event_data(ticker, None, f"expiration lookup failed ({exc})")
        return []
    out: list[str] = []
    for item in listed:
        iso = _normalize_expiry_iso(item)
        if iso:
            out.append(iso)
    return sorted(dict.fromkeys(out))


def _event_fetch_chain(ticker: str, expiry: str) -> pd.DataFrame:
    from data_ingestion import fetch_option_chain

    try:
        frame = fetch_option_chain(ticker, expiry)
    except Exception as exc:
        _warn_thin_event_data(ticker, expiry, f"chain fetch failed ({exc})")
        return pd.DataFrame()
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    return frame


def _select_event_buckets(expiries: list[str], selected_expiry: Any = None) -> tuple[str | None, str | None]:
    dated = [(iso, _expiry_days_to_go(iso)) for iso in expiries]
    dated = [(iso, dte) for iso, dte in dated if np.isfinite(dte) and dte > 0]
    if len(dated) < 2:
        return None, None
    dated.sort(key=lambda item: item[1])
    long_iso = dated[-1][0]
    liquid_iso = min(dated, key=lambda item: abs(item[1] - EVENT_LIQUID_DTE))[0]
    selected_iso = _normalize_expiry_iso(selected_expiry)
    selected_dte = _expiry_days_to_go(selected_iso) if selected_iso else float("nan")
    listed = {iso for iso, _ in dated}
    too_far = (
        selected_iso is None
        or selected_iso not in listed
        or not np.isfinite(selected_dte)
        or selected_dte > EVENT_MEDIUM_DTE
        or selected_iso == long_iso
    )
    short_iso = liquid_iso if too_far else selected_iso
    if short_iso == long_iso:
        short_iso = dated[0][0]
    if short_iso == long_iso:
        return None, None
    return short_iso, long_iso


def _expiry_row_mask(frame: pd.DataFrame, expiry_iso: str) -> np.ndarray:
    n = len(frame.index)
    col = _sentiment_column(frame, "expiration", "expiry", "Expiration", "Expiry")
    if col is not None:
        labels = np.array([_normalize_expiry_iso(v) for v in col.tolist()], dtype=object)
        return labels == expiry_iso
    dtes = _sentiment_tenor_days(frame)
    target = _expiry_days_to_go(expiry_iso)
    if not np.isfinite(target):
        return np.zeros(n, dtype=bool)
    return np.isfinite(dtes) & (np.abs(dtes - target) <= 1.5)


def _pad_event_chain(frame: pd.DataFrame, ticker: str | None, expiry_iso: str) -> pd.DataFrame:
    if np.any(_expiry_row_mask(frame, expiry_iso)):
        return frame
    _warn_thin_event_data(ticker, expiry_iso, "missing chain rows; padding from API")
    if not ticker:
        return frame
    extra = _event_fetch_chain(ticker, expiry_iso)
    if extra.empty:
        _warn_thin_event_data(ticker, expiry_iso, "padding fetch returned no rows")
        return frame
    return pd.concat([frame, extra], ignore_index=True)


def _mean_iv_for_expiry(frame: pd.DataFrame, expiry_iso: str) -> float:
    iv = _sentiment_iv_series(frame).to_numpy(dtype=np.float64)
    iv = np.where(np.isfinite(iv) & (iv > SIGMA_MIN), iv, np.nan)
    mask = _expiry_row_mask(frame, expiry_iso) & np.isfinite(iv)
    return _finite_mean(iv[mask])


def analyze_event_outlook(
    df: pd.DataFrame | None,
    ticker: str | None = None,
    selected_expiry: Any = None,
) -> dict[str, Any]:
    """Flag a near-term catalyst from listed expiries and implied vol.

    Discovers available expirations for ``ticker`` (lazy MarketData lookup).
    Short-term IV uses the selected expiry when it is near-dated; otherwise the
    nearest liquid expiry around ``EVENT_LIQUID_DTE`` (30). Long-term IV uses
    the furthest listed expiry. Short IV > 1.2× the comparison tenor is event risk.

    Args:
        df: Option chain with implied volatility and tenor or expiration.
        ticker: Underlying symbol used to fetch the full expiration calendar.
        selected_expiry: Dashboard expiry; ignored when too far out to compare.

    Returns:
        Dict with ``Outlook``, ``Summary``, ``EventRisk``, and bucket IVs.
    """
    frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    symbol = str(ticker).strip().upper() if ticker else ""
    if not symbol:
        ticker_col = _sentiment_column(frame, "ticker", "underlying", "symbol") if not frame.empty else None
        if ticker_col is not None and len(ticker_col):
            symbol = str(ticker_col.iloc[0]).strip().upper()
    listed: list[str] = []
    if symbol:
        listed = _event_listed_expirations(symbol)
        if len(listed) < 2:
            _warn_thin_event_data(symbol, _normalize_expiry_iso(selected_expiry), "API returned fewer than 2 expirations")
            return _event_outlook_unavailable(EVENT_INSUFFICIENT_COVERAGE)
    else:
        listed = _frame_expiry_dates(frame)
        if len(listed) < 2:
            _warn_thin_event_data(None, _normalize_expiry_iso(selected_expiry), "fewer than 2 expirations")
            return _event_outlook_unavailable(EVENT_INSUFFICIENT_COVERAGE)
    short_iso, long_iso = _select_event_buckets(listed, selected_expiry)
    if short_iso is None or long_iso is None:
        _warn_thin_event_data(symbol or None, _normalize_expiry_iso(selected_expiry), "could not form short/long buckets")
        return _event_outlook_unavailable(EVENT_INSUFFICIENT_COVERAGE)
    if frame.empty:
        frame = pd.DataFrame()
    frame = _pad_event_chain(frame, symbol or None, short_iso)
    frame = _pad_event_chain(frame, symbol or None, long_iso)
    if frame.empty:
        _warn_thin_event_data(symbol or None, short_iso, "no chain after padding")
        return _event_outlook_unavailable(EVENT_INSUFFICIENT_COVERAGE)
    iv = _sentiment_iv_series(frame).to_numpy(dtype=np.float64)
    iv = np.where(np.isfinite(iv) & (iv > SIGMA_MIN), iv, np.nan)
    dtes = _sentiment_tenor_days(frame)
    valid = np.isfinite(iv) & np.isfinite(dtes) & (dtes > 0)
    short_iv = _mean_iv_for_expiry(frame, short_iso)
    long_iv = _mean_iv_for_expiry(frame, long_iso)
    medium_mask = valid & (dtes >= EVENT_SHORT_DTE) & (dtes <= EVENT_MEDIUM_DTE)
    medium_iv = _finite_mean(iv[medium_mask])
    compare_iv = medium_iv if np.isfinite(medium_iv) else long_iv
    if not np.isfinite(short_iv) or not np.isfinite(compare_iv) or compare_iv <= SIGMA_MIN:
        _warn_thin_event_data(symbol or None, short_iso, "missing short/long IV after padding")
        return _event_outlook_unavailable(EVENT_INSUFFICIENT_COVERAGE)
    trend_iv = long_iv if np.isfinite(long_iv) and long_iv > SIGMA_MIN else compare_iv
    with np.errstate(divide="ignore", invalid="ignore"):
        event = bool(short_iv > EVENT_IV_RATIO * compare_iv)
        lift_pct = float((short_iv / trend_iv - 1.0) * 100.0)
    if not event:
        return {
            "Outlook": "Stable Event Outlook",
            "Summary": (
                "No near-term catalyst is priced: short-term IV is not elevated "
                "versus the medium-term term structure."
            ),
            "EventRisk": False,
            "ShortIV": short_iv,
            "MediumIV": medium_iv if np.isfinite(medium_iv) else None,
            "LongIV": long_iv if np.isfinite(long_iv) else None,
        }
    return {
        "Outlook": "High Event Risk / Catalyst Detected.",
        "Summary": (
            f"The market is pricing in a catalyst for the {short_iso} expiry, "
            f"with IV elevated by {lift_pct:.0f}% compared to the long-term trend."
        ),
        "EventRisk": True,
        "ShortIV": short_iv,
        "MediumIV": medium_iv if np.isfinite(medium_iv) else None,
        "LongIV": long_iv if np.isfinite(long_iv) else None,
    }


def _portfolio_empty() -> dict[str, Any]:
    return {
        "Delta": 0.0,
        "Gamma": 0.0,
        "Theta": 0.0,
        "Vega": 0.0,
        "LargestPosition": None,
        "TickerConcentration": None,
        "ConcentratedTicker": None,
        "DaysToExpiration": None,
        "DeltaByTicker": [],
        "Positions": pd.DataFrame(),
    }


def _black_scholes_price(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    sigma: float,
    is_put: bool,
    dividend: float = 0.0,
) -> float:
    if not (
        np.isfinite(spot)
        and spot > 0
        and np.isfinite(strike)
        and strike > 0
        and np.isfinite(time_years)
        and time_years >= T_MIN
        and np.isfinite(rate)
        and np.isfinite(sigma)
        and sigma >= SIGMA_MIN
    ):
        return float("nan")
    sqrt_t = np.sqrt(time_years)
    d1 = (np.log(spot / strike) + (rate - dividend + 0.5 * sigma * sigma) * time_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc_q = np.exp(-dividend * time_years)
    disc_r = np.exp(-rate * time_years)
    if is_put:
        return float(strike * disc_r * norm.cdf(-d2) - spot * disc_q * norm.cdf(-d1))
    return float(spot * disc_q * norm.cdf(d1) - strike * disc_r * norm.cdf(d2))


def _position_signed_qty(side: Any, quantity: float) -> float:
    qty = float(quantity) if np.isfinite(quantity) else 0.0
    text = str(side or "").strip().lower()
    if text in {"short", "sell", "written", "write", "s"} or (text.startswith("short") or text.startswith("sell")):
        return -abs(qty)
    if text in {"long", "buy", "bought", "l"} or text.startswith("long") or text.startswith("buy"):
        return abs(qty)
    return qty


def _position_is_put(side: Any, option_type: Any) -> bool:
    for raw in (option_type, side):
        text = str(raw or "").strip().lower()
        if text.startswith("p") and text not in {"premium"}:
            return True
        if text.startswith("c") and text not in {"contracts"}:
            return False
    return False


def _spot_for_ticker(ticker: str, cache: dict[str, float]) -> float:
    symbol = str(ticker or "").strip().upper()
    if not symbol or symbol in {"?", "NAN", "NONE"}:
        return 0.0
    if symbol in cache:
        return cache[symbol]
    try:
        from data_ingestion import get_current_price

        raw = get_current_price(symbol)
        price = float(raw) if raw is not None else 0.0
    except Exception:
        price = 0.0
    if not np.isfinite(price) or price < 0:
        price = 0.0
    cache[symbol] = price
    return price


def _finite_or_zero(values: np.ndarray) -> np.ndarray:
    arr = np.array(values, dtype=np.float64, copy=True)
    return np.array(np.where(np.isfinite(arr), arr, 0.0), dtype=np.float64, copy=True)


def calculate_portfolio_risk(positions_df: pd.DataFrame | None) -> dict[str, Any]:
    """Net Delta, Gamma, Theta, Vega, concentration, and nearest DTE for a book.

    Each row is a position with strike, expiry, side, and quantity. Short/sell
    sides flip the signed quantity. Greeks are per-share Black–Scholes values
    times contracts times ``OPTIONS_CONTRACT_SIZE``. Concentration is the
    largest |premium| share of total |premium|. Liquidity is the smallest
    positive days-to-expiry in the book.

    Args:
        positions_df: Positions with Strike/Expiry/Side/Quantity and optional
            S, sigma, premium, EntryPrice, and option_type.

    Returns:
        Dict with net Greeks, concentration, ``DeltaByTicker``, and a
        ``Positions`` frame with filled Spot and Premium.
    """
    if positions_df is None or not isinstance(positions_df, pd.DataFrame) or positions_df.empty:
        return _portfolio_empty()
    working = positions_df.copy()
    n = len(working.index)
    strike_col = _sentiment_column(working, "Strike", "strike", "K")
    expiry_col = _sentiment_column(working, "Expiry", "expiry", "expiration", "Expiration")
    side_col = _sentiment_column(working, "Side", "side")
    qty_col = _sentiment_column(working, "Quantity", "quantity", "qty", "contracts")
    type_col = _sentiment_column(working, "option_type", "type", "Type")
    spot_col = _sentiment_column(working, "S", "spot", "underlyingPrice", "Price")
    sigma_col = _sentiment_column(working, "sigma", "impliedVolatility", "IV")
    rate_col = _sentiment_column(working, "r", "rate")
    prem_col = _sentiment_column(working, "premium", "lastPrice", "mark", "mid", "option_price")
    entry_col = _sentiment_column(working, "EntryPrice", "entry_price", "entry", "entryprice")
    t_col = _sentiment_column(working, "T")
    dte_col = _sentiment_column(working, "DaysToExpiry", "dte", "days_to_expiry")
    size_col = _sentiment_column(working, "contractSize", "multiplier", "contract_size")
    ticker_col = _sentiment_column(working, "Ticker", "ticker", "symbol", "underlying")

    def _num(col: pd.Series | None, default: float = float("nan")) -> np.ndarray:
        if col is None:
            return np.full(n, default, dtype=np.float64)
        return np.array(pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64), dtype=np.float64, copy=True)

    strikes = _num(strike_col)
    spots = _num(spot_col)
    sigmas = _num(sigma_col, DEFAULT_SIGMA)
    rates = _num(rate_col, DEFAULT_RATE)
    qtys = _num(qty_col, 0.0)
    premiums = _num(prem_col)
    entries = _num(entry_col)
    sizes = _num(size_col, float(OPTIONS_CONTRACT_SIZE))
    sizes = np.array(np.where(np.isfinite(sizes) & (sizes > 0), sizes, float(OPTIONS_CONTRACT_SIZE)), dtype=np.float64, copy=True)
    tenors = _num(t_col)
    dtes = _num(dte_col)
    sides = side_col.astype(str).tolist() if side_col is not None else [""] * n
    types = type_col.astype(str).tolist() if type_col is not None else [None] * n
    expiries = expiry_col.tolist() if expiry_col is not None else [None] * n
    if ticker_col is None:
        tickers = np.array(["?"] * n, dtype=object)
    else:
        tickers = np.array(ticker_col.astype(str).str.strip().str.upper().to_numpy(dtype=object), copy=True)
        tickers = np.array(np.where(pd.notna(ticker_col) & (tickers != "") & (tickers != "NAN"), tickers, "?"), copy=True)

    spots = np.array(spots, dtype=np.float64, copy=True)
    tenors = np.array(tenors, dtype=np.float64, copy=True)
    dtes = np.array(dtes, dtype=np.float64, copy=True)
    qtys = _finite_or_zero(qtys)
    entries = _finite_or_zero(entries)
    strikes = _finite_or_zero(strikes)
    spot_cache: dict[str, float] = {}
    filled_spots = np.array(spots, dtype=np.float64, copy=True)
    for i in range(n):
        if not np.isfinite(filled_spots[i]) or filled_spots[i] <= 0:
            filled_spots[i] = _spot_for_ticker(str(tickers[i]), spot_cache)
    spots = _finite_or_zero(filled_spots)
    missing_prem = np.array((~np.isfinite(premiums)) | (premiums < 0), copy=True)
    premiums = np.array(
        np.where(
            missing_prem,
            np.abs(qtys) * entries * float(OPTIONS_CONTRACT_SIZE),
            premiums,
        ),
        dtype=np.float64,
        copy=True,
    )
    premium_is_total = missing_prem
    premiums = _finite_or_zero(premiums)
    sigmas = _finite_or_zero(sigmas)
    rates = _finite_or_zero(rates)

    filled_tenors = np.array(tenors, dtype=np.float64, copy=True)
    missing_t = ~np.isfinite(filled_tenors) | (filled_tenors <= 0)
    if np.any(missing_t) and expiry_col is not None:
        for i in np.flatnonzero(missing_t):
            years = _time_to_expiry_years(expiries[i])
            if np.isfinite(years) and years > 0:
                filled_tenors[i] = years
    tenors = filled_tenors
    missing_dte = ~np.isfinite(dtes) | (dtes <= 0)
    dtes = np.array(
        np.where(missing_dte & np.isfinite(tenors) & (tenors > 0), tenors * DAYS_PER_YEAR, dtes),
        dtype=np.float64,
        copy=True,
    )
    if np.any(missing_dte) and expiry_col is not None:
        for i in np.flatnonzero(~np.isfinite(dtes) | (dtes <= 0)):
            iso = _normalize_expiry_iso(expiries[i])
            if iso:
                dtes[i] = _expiry_days_to_go(iso)
    tenors = _finite_or_zero(tenors)
    dtes = _finite_or_zero(dtes)

    net_delta = 0.0
    net_gamma = 0.0
    net_theta = 0.0
    net_vega = 0.0
    notionals = np.zeros(n, dtype=np.float64)
    delta_rows = np.zeros(n, dtype=np.float64)
    for i in range(n):
        signed = _position_signed_qty(sides[i], qtys[i])
        is_put = _position_is_put(sides[i], types[i])
        multiplier = float(sizes[i]) * signed
        greeks = calculate_greeks(
            {
                "S": spots[i],
                "K": strikes[i],
                "T": tenors[i],
                "r": rates[i],
                "sigma": sigmas[i],
                "option_type": "put" if is_put else "call",
            }
        )
        if greeks.get("Delta") is not None:
            contribution = float(greeks["Delta"]) * multiplier
            net_delta += contribution
            delta_rows[i] = contribution
        if greeks.get("Gamma") is not None:
            net_gamma += float(greeks["Gamma"]) * multiplier
        if greeks.get("Theta") is not None:
            net_theta += float(greeks["Theta"]) * multiplier
        if greeks.get("Vega") is not None:
            net_vega += float(greeks["Vega"]) * multiplier
        prem = premiums[i]
        if premium_is_total[i]:
            notionals[i] = abs(float(prem))
        else:
            notionals[i] = abs(float(prem) * abs(signed) * float(sizes[i]))

    total_prem = float(np.nansum(notionals))
    largest = None
    ticker_conc = None
    concentrated_ticker = None
    if total_prem > 1e-12:
        largest = float(100.0 * np.nanmax(notionals) / total_prem)
        ticker_prem = pd.Series(notionals, dtype=np.float64).groupby(tickers, dropna=False).sum()
        if not ticker_prem.empty:
            ticker_conc = float(100.0 * float(ticker_prem.max()) / total_prem)
            concentrated_ticker = str(ticker_prem.idxmax())
    delta_by_ticker = (
        pd.Series(delta_rows, dtype=np.float64)
        .groupby(tickers, dropna=False)
        .sum()
        .rename("Delta")
        .rename_axis("Ticker")
        .reset_index()
    )
    delta_by_ticker["Ticker"] = delta_by_ticker["Ticker"].astype(str)
    finite_dte = dtes[np.isfinite(dtes) & (dtes >= 0)]
    nearest = float(np.min(finite_dte)) if finite_dte.size else None
    filled = working.copy()
    filled.loc[:, "S"] = np.array(spots, dtype=np.float64, copy=True)
    filled.loc[:, "premium"] = np.array(premiums, dtype=np.float64, copy=True)
    if "EntryPrice" not in filled.columns:
        filled.loc[:, "EntryPrice"] = np.array(entries, dtype=np.float64, copy=True)
    if filled.empty:
        return _portfolio_empty()
    return {
        "Delta": float(net_delta),
        "Gamma": float(net_gamma),
        "Theta": float(net_theta),
        "Vega": float(net_vega),
        "LargestPosition": largest,
        "TickerConcentration": ticker_conc,
        "ConcentratedTicker": concentrated_ticker,
        "DaysToExpiration": nearest,
        "DeltaByTicker": delta_by_ticker.to_dict(orient="records"),
        "Positions": filled,
    }


# ---------------------------------------------------------------------------
# What-If Scenario Engine — spot / IV stress on DataRepository snapshots
# ---------------------------------------------------------------------------
# spot_shift / iv_shift are percent moves (e.g. -10 → −10% spot, +20 → +20% IV).
# Chain re-pricing uses vectorized Black–Scholes for real-time heatmaps. When
# ``model=heston`` is requested, stressed IV still drives the closed-form BS
# surface (same snappy path as Integrated Risk View); Heston MC is not run per
# contract.


def _stress_empty_impact(
    *,
    spot_shift: float = 0.0,
    iv_shift: float = 0.0,
    spot: float | None = None,
) -> dict[str, Any]:
    return {
        "delta_change": 0.0,
        "gamma_change": 0.0,
        "vega_change": 0.0,
        "net_pnl": 0.0,
        "base_delta": 0.0,
        "base_gamma": 0.0,
        "base_vega": 0.0,
        "stressed_delta": 0.0,
        "stressed_gamma": 0.0,
        "stressed_vega": 0.0,
        "total_gamma": 0.0,
        "negative_gamma": False,
        "spot": spot,
        "stressed_spot": spot,
        "spot_shift": float(spot_shift),
        "iv_shift": float(iv_shift),
        "n_contracts": 0,
    }


def _snapshot_chain_frame(
    ticker: str,
    repo: DataRepository | None = None,
) -> tuple[pd.DataFrame, float | None]:
    """Load the latest DataRepository snapshot (live cache, else newest history)."""
    repository = repo if repo is not None else DataRepository()
    symbol = str(ticker or "").strip().upper()
    payload = repository.load_from_cache(symbol)
    if not payload:
        stamps = repository.get_available_snapshots(symbol)
        if stamps:
            payload = repository.load_from_cache(symbol, stamps[-1])
    if payload:
        frame = repository._records_to_frame(payload.get("data")).copy()
        spot = payload.get("spot")
        if spot is None:
            spot = repository._spot_from_frame(frame)
        try:
            spot_f = float(spot) if spot is not None else None
        except (TypeError, ValueError):
            spot_f = None
        if spot_f is not None and (not np.isfinite(spot_f) or spot_f <= 0):
            spot_f = None
        return frame, spot_f
    try:
        frame = repository.get_data(symbol)
    except Exception:
        frame = pd.DataFrame()
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(), None
    working = frame.copy()
    return working, repository._spot_from_frame(working)


def _chain_stress_arrays(frame: pd.DataFrame, spot: float | None) -> dict[str, Any] | None:
    """Extract copied numeric chain arrays for vectorized stress re-pricing."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    working = frame.copy()
    n = len(working.index)
    s_col = _sentiment_column(working, "S", "underlyingPrice", "spot", "Price")
    k_col = _sentiment_column(working, "K", "strike", "Strike")
    t_col = _sentiment_column(working, "T")
    dte_col = _sentiment_column(working, "DaysToExpiry", "dte", "days_to_expiry")
    sig_col = _sentiment_column(working, "sigma", "impliedVolatility", "IV", "iv")
    rate_col = _sentiment_column(working, "r", "rate")
    oi_col = _sentiment_column(working, "openInterest", "open_interest", "OI", "oi")
    size_col = _sentiment_column(working, "contractSize", "contract_size", "multiplier")
    if k_col is None or sig_col is None:
        return None

    def _num(col: pd.Series | None, default: float = float("nan")) -> np.ndarray:
        if col is None:
            return np.full(n, default, dtype=np.float64)
        return np.array(pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64), dtype=np.float64, copy=True)

    strikes = _num(k_col)
    spots = _num(s_col)
    if spot is not None and np.isfinite(spot) and spot > 0:
        fill = float(spot)
        spots = np.array(np.where(np.isfinite(spots) & (spots > 0), spots, fill), dtype=np.float64, copy=True)
    tenors = _num(t_col)
    dtes = _num(dte_col)
    missing_t = ~np.isfinite(tenors) | (tenors <= 0)
    if np.any(missing_t):
        tenors = np.array(
            np.where(missing_t & np.isfinite(dtes) & (dtes > 0), dtes / DAYS_PER_YEAR, tenors),
            dtype=np.float64,
            copy=True,
        )
    sigmas = _num(sig_col)
    rates = _num(rate_col, DEFAULT_RATE)
    rates = np.array(np.where(np.isfinite(rates), rates, DEFAULT_RATE), dtype=np.float64, copy=True)
    oi = _num(oi_col, 0.0)
    oi = np.array(np.where(np.isfinite(oi) & (oi >= 0), oi, 0.0), dtype=np.float64, copy=True)
    if not np.any(oi > 0):
        oi = np.ones(n, dtype=np.float64)
    sizes = _num(size_col, float(OPTIONS_CONTRACT_SIZE))
    sizes = np.array(
        np.where(np.isfinite(sizes) & (sizes > 0), sizes, float(OPTIONS_CONTRACT_SIZE)),
        dtype=np.float64,
        copy=True,
    )
    is_call, is_put = _sentiment_type_mask(working)
    put_mask = np.array(is_put, dtype=bool, copy=True)
    if not np.any(is_call | is_put):
        put_mask = np.zeros(n, dtype=bool)
    valid = (
        np.isfinite(spots)
        & (spots > 0)
        & np.isfinite(strikes)
        & (strikes > 0)
        & np.isfinite(tenors)
        & (tenors >= T_MIN)
        & np.isfinite(sigmas)
        & (sigmas >= SIGMA_MIN)
    )
    if not np.any(valid):
        return None
    return {
        "spots": spots,
        "strikes": strikes,
        "tenors": tenors,
        "rates": rates,
        "sigmas": sigmas,
        "oi": oi,
        "sizes": sizes,
        "put_mask": put_mask,
        "valid": valid,
    }


def _bs_chain_price_greeks(
    spots: np.ndarray,
    strikes: np.ndarray,
    tenors: np.ndarray,
    rates: np.ndarray,
    sigmas: np.ndarray,
    put_mask: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorized BS price + Delta/Gamma/Vega for heterogeneous chain rows."""
    n = int(spots.shape[0])
    price = np.full(n, np.nan, dtype=np.float64)
    delta = np.full(n, np.nan, dtype=np.float64)
    gamma = np.full(n, np.nan, dtype=np.float64)
    vega = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or not np.any(valid):
        return {"price": price, "Delta": delta, "Gamma": gamma, "Vega": vega}

    s = spots[valid]
    k = strikes[valid]
    t = tenors[valid]
    r = rates[valid]
    sig = sigmas[valid]
    puts = put_mask[valid]
    sqrt_t = np.sqrt(t)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d1 = (np.log(s / k) + (r + 0.5 * sig * sig) * t) / (sig * sqrt_t)
        d2 = d1 - sig * sqrt_t
        n_d1 = norm.pdf(d1)
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        disc_r = np.exp(-r * t)
        call_px = s * nd1 - k * disc_r * nd2
        put_px = k * disc_r * norm.cdf(-d2) - s * norm.cdf(-d1)
        px = np.where(puts, put_px, call_px)
        dlt = np.where(puts, nd1 - 1.0, nd1)
        gam = n_d1 / (s * sig * sqrt_t)
        veg = s * n_d1 * sqrt_t / 100.0
    price[valid] = np.asarray(px, dtype=np.float64)
    delta[valid] = np.asarray(dlt, dtype=np.float64)
    gamma[valid] = np.asarray(gam, dtype=np.float64)
    vega[valid] = np.asarray(veg, dtype=np.float64)
    return {"price": price, "Delta": delta, "Gamma": gamma, "Vega": vega}


def _aggregate_chain_risk(
    greeks: Mapping[str, np.ndarray],
    oi: np.ndarray,
    sizes: np.ndarray,
    put_mask: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    """OI×multiplier nets; dealer gamma uses put-negative GEX sign convention."""
    weight = np.array(oi * sizes, dtype=np.float64, copy=True)
    weight = np.where(valid & np.isfinite(weight), weight, 0.0)
    delta = np.asarray(greeks["Delta"], dtype=np.float64)
    gamma = np.asarray(greeks["Gamma"], dtype=np.float64)
    vega = np.asarray(greeks["Vega"], dtype=np.float64)
    sign = np.where(put_mask, -1.0, 1.0)
    with np.errstate(invalid="ignore", over="ignore"):
        net_delta = float(np.nansum(np.where(valid, delta * weight, 0.0)))
        net_gamma = float(np.nansum(np.where(valid, gamma * weight, 0.0)))
        net_vega = float(np.nansum(np.where(valid, vega * weight, 0.0)))
        dealer_gamma = float(np.nansum(np.where(valid, gamma * weight * sign, 0.0)))
    return {
        "Delta": net_delta,
        "Gamma": net_gamma,
        "Vega": net_vega,
        "dealer_gamma": dealer_gamma,
    }


def _chain_net_pnl(
    base_price: np.ndarray,
    stress_price: np.ndarray,
    oi: np.ndarray,
    sizes: np.ndarray,
    valid: np.ndarray,
) -> float:
    """Long OI-weighted mark-to-market P&L of the chain book."""
    with np.errstate(invalid="ignore", over="ignore"):
        d_px = stress_price - base_price
        pnl = d_px * oi * sizes
        pnl = np.where(valid & np.isfinite(pnl), pnl, 0.0)
    return float(np.nansum(pnl))


def calculate_stress_scenario(
    ticker: str,
    spot_shift: float,
    iv_shift: float,
    *,
    repo: DataRepository | None = None,
    model: str = MODEL_BLACK_SCHOLES,
    rate: float = DEFAULT_RATE,
    frame: pd.DataFrame | None = None,
    spot: float | None = None,
) -> dict[str, Any]:
    """Re-price the DataRepository chain under spot% and IV% shocks.

    Takes the current snapshot, shifts every row's spot by ``spot_shift`` percent
    and IV by ``iv_shift`` percent, then recomputes the Greek surface
    (Delta / Gamma / Vega) with vectorized Black–Scholes. ``model=heston`` is
    accepted for API parity but still uses the BS closed form on the stressed
    IV surface for real-time use.

    Returns a Risk Impact dict with ``delta_change`` / ``gamma_change`` /
    ``vega_change``, ``net_pnl`` (OI-weighted chain mark P&L for heatmaps),
    and ``total_gamma`` (dealer-signed GEX at the stressed spot).
    """
    _ = model  # reserved; BS path is the real-time stress engine
    try:
        spot_pct = float(spot_shift)
        iv_pct = float(iv_shift)
        rate_f = float(rate)
    except (TypeError, ValueError):
        return _stress_empty_impact(spot_shift=0.0, iv_shift=0.0)

    if frame is None:
        chain, snap_spot = _snapshot_chain_frame(ticker, repo=repo)
        use_spot = spot if spot is not None else snap_spot
    else:
        chain = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        use_spot = spot
        if use_spot is None and not chain.empty:
            use_spot = DataRepository()._spot_from_frame(chain)

    arrays = _chain_stress_arrays(chain, use_spot)
    if arrays is None:
        return _stress_empty_impact(spot_shift=spot_pct, iv_shift=iv_pct, spot=use_spot)

    spots = np.array(arrays["spots"], dtype=np.float64, copy=True)
    strikes = arrays["strikes"]
    tenors = arrays["tenors"]
    rates = np.array(arrays["rates"], dtype=np.float64, copy=True)
    if np.isfinite(rate_f):
        rates = np.full(rates.shape, rate_f, dtype=np.float64)
    sigmas = np.array(arrays["sigmas"], dtype=np.float64, copy=True)
    oi = arrays["oi"]
    sizes = arrays["sizes"]
    put_mask = arrays["put_mask"]
    valid = arrays["valid"]

    spot_mult = 1.0 + spot_pct / 100.0
    iv_mult = max(1.0 + iv_pct / 100.0, 1e-6)
    stressed_spots = np.array(spots * spot_mult, dtype=np.float64, copy=True)
    stressed_sigmas = np.array(np.maximum(sigmas * iv_mult, SIGMA_MIN), dtype=np.float64, copy=True)

    base = _bs_chain_price_greeks(spots, strikes, tenors, rates, sigmas, put_mask, valid)
    stress = _bs_chain_price_greeks(
        stressed_spots, strikes, tenors, rates, stressed_sigmas, put_mask, valid
    )
    base_agg = _aggregate_chain_risk(base, oi, sizes, put_mask, valid)
    stress_agg = _aggregate_chain_risk(stress, oi, sizes, put_mask, valid)
    net_pnl = _chain_net_pnl(base["price"], stress["price"], oi, sizes, valid)

    base_spot = float(np.nanmedian(spots[valid])) if np.any(valid) else use_spot
    stressed_spot = float(base_spot * spot_mult) if base_spot is not None and np.isfinite(base_spot) else None
    total_gamma = float(stress_agg["dealer_gamma"])
    return {
        "delta_change": float(stress_agg["Delta"] - base_agg["Delta"]),
        "gamma_change": float(stress_agg["Gamma"] - base_agg["Gamma"]),
        "vega_change": float(stress_agg["Vega"] - base_agg["Vega"]),
        "net_pnl": float(net_pnl),
        "base_delta": float(base_agg["Delta"]),
        "base_gamma": float(base_agg["Gamma"]),
        "base_vega": float(base_agg["Vega"]),
        "stressed_delta": float(stress_agg["Delta"]),
        "stressed_gamma": float(stress_agg["Gamma"]),
        "stressed_vega": float(stress_agg["Vega"]),
        "total_gamma": total_gamma,
        "negative_gamma": bool(total_gamma < 0.0),
        "spot": float(base_spot) if base_spot is not None and np.isfinite(base_spot) else None,
        "stressed_spot": stressed_spot,
        "spot_shift": float(spot_pct),
        "iv_shift": float(iv_pct),
        "n_contracts": int(np.count_nonzero(valid)),
    }


def build_stress_pnl_matrix(
    ticker: str,
    spot_shifts: Iterable[float],
    iv_shifts: Iterable[float],
    *,
    repo: DataRepository | None = None,
    model: str = MODEL_BLACK_SCHOLES,
    rate: float = DEFAULT_RATE,
) -> dict[str, Any]:
    """Thin companion: Net P&L heatmap grid over spot% × IV% shock axes.

    Loads the snapshot once, then evaluates :func:`calculate_stress_scenario`
    for each cell. Returns ``spot_shifts``, ``iv_shifts``, and ``net_pnl`` as a
    2D list ``[i_spot][j_iv]`` ready for Plotly heatmaps.
    """
    spot_axis = [float(v) for v in spot_shifts]
    iv_axis = [float(v) for v in iv_shifts]
    chain, snap_spot = _snapshot_chain_frame(ticker, repo=repo)
    grid: list[list[float]] = []
    n_contracts = 0
    for spot_pct in spot_axis:
        row: list[float] = []
        for iv_pct in iv_axis:
            impact = calculate_stress_scenario(
                ticker,
                spot_pct,
                iv_pct,
                repo=repo,
                model=model,
                rate=rate,
                frame=chain,
                spot=snap_spot,
            )
            n_contracts = max(n_contracts, int(impact.get("n_contracts") or 0))
            row.append(float(impact.get("net_pnl") or 0.0))
        grid.append(row)
    return {
        "spot_shifts": spot_axis,
        "iv_shifts": iv_axis,
        "net_pnl": grid,
        "spot": snap_spot,
        "n_contracts": int(n_contracts),
    }


# ---------------------------------------------------------------------------
# Market Timelapse — historical surface frames from data_history/
# ---------------------------------------------------------------------------
# Prefer ``delta_surface`` (value key ``Delta``): same payload key persisted by
# DataRepository._compute_surfaces / Time Machine. Not vol_surface/gamma_surface.
TIMELAPSE_SURFACE_KEY = "delta_surface"
TIMELAPSE_VALUE_KEY = "Delta"


def snapshot_time_label_from_name(name: str) -> str:
    """Extract a slider label ``HH:MM:SS`` from a snapshot filename or stamp.

    Recognizes ``YYYY-MM-DD_HHMM``, optional ``_unix`` suffix (preferred for
    seconds), and compact ``YYYYmmDDHHMMSS`` capture stamps.
    """
    stem = Path(str(name or "")).stem
    parts = [p for p in stem.split("_") if p]
    # Unix epoch suffix (e.g. SPY_2026-09-13_0723_1789284208).
    for part in reversed(parts):
        if part.isdigit() and len(part) >= 10:
            try:
                moment = datetime.fromtimestamp(int(part))
                return moment.strftime("%H:%M:%S")
            except (OSError, OverflowError, ValueError):
                break
    # Compact capture stamp …HHMMSS (8+ digits ending in time).
    for part in reversed(parts):
        digits = "".join(ch for ch in part if ch.isdigit())
        if len(digits) >= 14:
            hh, mm, ss = digits[-6:-4], digits[-4:-2], digits[-2:]
            if hh.isdigit() and mm.isdigit() and ss.isdigit():
                return f"{int(hh):02d}:{int(mm):02d}:{int(ss):02d}"
        if len(digits) == 4 and digits.isdigit():
            return f"{int(digits[:2]):02d}:{int(digits[2:]):02d}:00"
    # Date-like token then HHMM: 2026-09-13_1434
    for i, part in enumerate(parts):
        if len(part) >= 10 and part[4] == "-" and part[7] == "-" and i + 1 < len(parts):
            tod = "".join(ch for ch in parts[i + 1] if ch.isdigit())
            if len(tod) >= 6:
                return f"{int(tod[0:2]):02d}:{int(tod[2:4]):02d}:{int(tod[4:6]):02d}"
            if len(tod) >= 4:
                return f"{int(tod[0:2]):02d}:{int(tod[2:4]):02d}:00"
    return stem or "00:00:00"


def delta_surface_to_grid(surface: Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flattened ``delta_surface`` dict → ``(z2d, strike_axis, dte_axis)``.

    Empty / unusable surfaces yield empty arrays. Layout matches Plotly Surface:
    ``z.shape == (len(dte_axis), len(strike_axis))``.
    """
    empty = (
        np.empty((0, 0), dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )
    if not isinstance(surface, Mapping):
        return empty
    strikes = surface.get("Strike") or surface.get("strike") or []
    dtes = surface.get("DaysToExpiry") or surface.get("dte") or []
    values = surface.get(TIMELAPSE_VALUE_KEY) or surface.get("delta") or []
    if not isinstance(strikes, list) or not isinstance(dtes, list) or not isinstance(values, list):
        return empty
    if not strikes or not dtes or not values or not (len(strikes) == len(dtes) == len(values)):
        return empty
    frame = pd.DataFrame(
        {
            "Strike": pd.to_numeric(pd.Series(strikes), errors="coerce"),
            "DaysToExpiry": pd.to_numeric(pd.Series(dtes), errors="coerce"),
            "Delta": pd.to_numeric(pd.Series([np.nan if v is None else v for v in values]), errors="coerce"),
        }
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["Strike", "DaysToExpiry"])
    if frame.empty or not frame["Delta"].notna().any():
        return empty
    pivot = frame.pivot_table(index="DaysToExpiry", columns="Strike", values="Delta", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    z2d = np.array(pivot.to_numpy(dtype=np.float64), copy=True)
    x_axis = np.array(pivot.columns.to_numpy(dtype=np.float64), copy=True)
    y_axis = np.array(pivot.index.to_numpy(dtype=np.float64), copy=True)
    return z2d, x_axis, y_axis


def timelapse_shared_z_range(grids: Iterable[Any]) -> tuple[float, float]:
    """Shared finite Z min/max across all timelapse frames (locks color / z-axis)."""
    lo = float("inf")
    hi = float("-inf")
    for grid in grids:
        arr = np.asarray(grid, dtype=np.float64)
        if arr.size == 0:
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        lo = min(lo, float(np.min(finite)))
        hi = max(hi, float(np.max(finite)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if lo == hi:
        pad = 1e-6 if lo == 0.0 else abs(lo) * 1e-6
        return lo - pad, hi + pad
    return lo, hi


def load_all_snapshots(
    ticker: str,
    *,
    history_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch every ``data_history/{TICKER}_*.json`` snapshot, sorted by timestamp.

    **Surface key:** ``delta_surface`` (``TIMELAPSE_SURFACE_KEY``), value ``Delta``.
    Each returned dict::

        path, stamp, label (HH:MM:SS), timestamp (sort key),
        z (2D ndarray), x (strikes), y (DTEs), surface_key

    Snapshots without a usable delta surface after in-memory backfill are skipped.
    Does not rewrite history files.
    """
    root = Path(history_dir) if history_dir is not None else Path(__file__).resolve().parent / "data_history"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(ticker or "").strip().upper()) or "_"
    prefix = f"{safe}_"
    paths = [p for p in root.glob(f"{prefix}*.json") if p.is_file()]
    if not paths and root.name != "data_history":
        # Also scan repo-level data_history when a custom dir is empty.
        fallback = Path(__file__).resolve().parent / "data_history"
        if fallback.is_dir() and fallback.resolve() != root.resolve():
            paths = [p for p in fallback.glob(f"{prefix}*.json") if p.is_file()]

    repo = DataRepository(history_dir=root)
    frames: list[dict[str, Any]] = []
    for path in paths:
        payload = repo._read_json(path)
        if not payload:
            continue
        # In-memory backfill only — never rewrite history files.
        if repo._surface_is_empty(payload.get(TIMELAPSE_SURFACE_KEY), TIMELAPSE_VALUE_KEY):
            payload = dict(payload)
            frame = repo._records_to_frame(payload.get("data"))
            if not frame.empty:
                payload.update(repo._compute_surfaces(frame))
        surface = payload.get(TIMELAPSE_SURFACE_KEY)
        z2d, x_axis, y_axis = delta_surface_to_grid(surface if isinstance(surface, Mapping) else None)
        if z2d.size == 0 or not np.isfinite(z2d).any():
            continue
        stamp = path.stem[len(prefix):] if path.stem.startswith(prefix) else path.stem
        label = snapshot_time_label_from_name(path.name)
        try:
            ts = float(payload.get("saved_at"))
        except (TypeError, ValueError):
            ts = float("nan")
        if not np.isfinite(ts):
            # Fall back to parsed stamp / mtime for ordering.
            ts = 0.0
            for part in reversed(stamp.split("_")):
                if part.isdigit() and len(part) >= 10:
                    ts = float(part)
                    break
            if ts <= 0.0:
                try:
                    ts = float(path.stat().st_mtime)
                except OSError:
                    ts = 0.0
        frames.append(
            {
                "path": str(path),
                "stamp": stamp,
                "label": label,
                "timestamp": float(ts),
                "z": z2d,
                "x": x_axis,
                "y": y_axis,
                "surface_key": TIMELAPSE_SURFACE_KEY,
            }
        )

    frames.sort(key=lambda item: (float(item["timestamp"]), str(item["stamp"]), str(item["path"])))
    # Disambiguate duplicate HH:MM:SS labels for Plotly frame names / slider steps.
    seen: dict[str, int] = {}
    for item in frames:
        base = str(item["label"])
        count = seen.get(base, 0)
        seen[base] = count + 1
        if count:
            item["label"] = f"{base}·{count + 1}"
        item["frame_name"] = str(item["label"])
    return frames


def test_simulated_pnl(
    delta: float,
    gamma: float,
    theta: float,
    vega: float,
    quantity: float,
    side: str,
    *,
    spot_shock: float = 1.0,
    day_shock: float = 1.0 / DAYS_PER_YEAR,
    vol_shock: float = 0.01,
    multiplier: float = float(OPTIONS_CONTRACT_SIZE),
) -> float:
    """Mock trade-ticket P&L from first-/second-order Greeks (Test Dashboard only).

    Formula (documented for the institutional mock ticket)::

        side_sign ∈ {+1 buy/long/call, -1 sell/short/put-as-sell}
        dV ≈ Δ·dS + ½·Γ·(dS)² + Θ·dT + ν·dσ
        P&L = side_sign · |quantity| · multiplier · dV

    Defaults: dS=1.0 spot point, dT=1/365 year, dσ=0.01 (1 vol point).
    Not a production pricing engine — sandbox simulation only.
    """
    try:
        dlt = float(delta)
        gam = float(gamma)
        tht = float(theta)
        veg = float(vega)
        qty = float(quantity)
        d_s = float(spot_shock)
        d_t = float(day_shock)
        d_v = float(vol_shock)
        mult = float(multiplier)
    except (TypeError, ValueError):
        return float("nan")
    if not all(np.isfinite(x) for x in (dlt, gam, tht, veg, qty, d_s, d_t, d_v, mult)):
        return float("nan")
    label = str(side or "").strip().lower()
    if label in {"sell", "short", "s"}:
        side_sign = -1.0
    else:
        side_sign = 1.0
    d_value = dlt * d_s + 0.5 * gam * (d_s ** 2) + tht * d_t + veg * d_v
    return float(side_sign * abs(qty) * mult * d_value)


test_simulated_pnl.__test__ = False  # not a pytest case; Test Dashboard helper only


# ---------------------------------------------------------------------------
# Event Impact Lab — event-driven Greek shock surfaces (isolated feature)
# ---------------------------------------------------------------------------
# All public helpers use the ``event_impact_`` prefix. Snapshot pairing uses
# DataRepository history only; no coupling to Time Machine / Advanced Metrics.

EVENT_IMPACT_DATA_PATH = Path(__file__).resolve().parent / "event_impact_data.json"
EVENT_IMPACT_EVENT_HOUR = 12  # noon local — same-calendar-day snaps can straddle


def event_impact_load_data(
    path: str | Path | None = None,
) -> list[dict[str, str]]:
    """Parse ``event_impact_data.json`` into ``[{date, event}, ...]``.

    Invalid rows are skipped. Missing / unreadable files yield ``[]``.
    """
    target = Path(path) if path is not None else EVENT_IMPACT_DATA_PATH
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        event_date = str(item.get("date") or "").strip()
        event_name = str(item.get("event") or "").strip()
        if not event_date or not event_name:
            continue
        out.append({"date": event_date, "event": event_name})
    return out


def event_impact_parse_stamp_datetime(stamp: str, saved_at: Any = None) -> datetime | None:
    """Best-effort datetime for a history stamp (prefers ``saved_at`` unix when valid)."""
    try:
        ts = float(saved_at)
    except (TypeError, ValueError):
        ts = float("nan")
    if np.isfinite(ts) and ts > 1e9:
        return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)

    text = str(stamp or "").strip()
    if not text:
        return None
    parts = text.replace(".json", "").split("_")
    date_part = next((p for p in parts if len(p) >= 10 and p[4] == "-" and p[7] == "-"), None)
    if date_part is None:
        return None
    try:
        base = datetime.strptime(date_part[:10], "%Y-%m-%d")
    except ValueError:
        return None
    # Prefer HHMM / HHMMSS token after the date; ignore trailing unix ids.
    for part in parts[parts.index(date_part) + 1 :]:
        digits = "".join(ch for ch in part if ch.isdigit())
        if len(digits) >= 10:
            continue
        if len(digits) >= 6:
            return base.replace(
                hour=int(digits[0:2]),
                minute=int(digits[2:4]),
                second=int(digits[4:6]),
            )
        if len(digits) >= 4:
            return base.replace(hour=int(digits[0:2]), minute=int(digits[2:4]), second=0)
    return base.replace(hour=EVENT_IMPACT_EVENT_HOUR, minute=0, second=0)


def event_impact_event_datetime(event_date: str | date | datetime) -> datetime | None:
    """Normalize an event date to a noon datetime for before/after pairing."""
    if isinstance(event_date, datetime):
        return event_date.replace(tzinfo=None)
    if isinstance(event_date, date):
        return datetime(event_date.year, event_date.month, event_date.day, EVENT_IMPACT_EVENT_HOUR)
    text = str(event_date or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            parsed = datetime.strptime(text[:10] if fmt != "%Y%m%d" else text[:8], fmt)
            return parsed.replace(hour=EVENT_IMPACT_EVENT_HOUR, minute=0, second=0)
        except ValueError:
            continue
    return None


def event_impact_list_snapshot_times(
    ticker: str,
    *,
    repo: DataRepository | None = None,
) -> list[dict[str, Any]]:
    """Return sorted snapshot metadata ``[{stamp, when, path}, ...]`` for ``ticker``."""
    repository = repo if repo is not None else DataRepository()
    symbol = str(ticker or "").strip().upper()
    rows: list[dict[str, Any]] = []
    for stamp in repository.get_available_snapshots(symbol):
        path = repository._resolve_snapshot_path(symbol, stamp)
        payload: dict[str, Any] = {}
        if path is not None:
            loaded = repository._read_json(path)
            if isinstance(loaded, dict):
                payload = loaded
        when = event_impact_parse_stamp_datetime(stamp, payload.get("saved_at"))
        if when is None:
            continue
        rows.append({"stamp": str(stamp), "when": when, "path": path})
    rows.sort(key=lambda item: (item["when"], item["stamp"]))
    return rows


def event_impact_find_bracketing_snapshots(
    ticker: str,
    event_date: str | date | datetime,
    *,
    repo: DataRepository | None = None,
) -> tuple[str | None, str | None]:
    """Closest snapshot strictly before and after the event instant."""
    event_when = event_impact_event_datetime(event_date)
    if event_when is None:
        return None, None
    rows = event_impact_list_snapshot_times(ticker, repo=repo)
    before = [row for row in rows if row["when"] < event_when]
    after = [row for row in rows if row["when"] > event_when]
    stamp_before = str(before[-1]["stamp"]) if before else None
    stamp_after = str(after[0]["stamp"]) if after else None
    return stamp_before, stamp_after


def _event_impact_payload_in_memory(
    repository: DataRepository,
    ticker: str,
    stamp: str,
) -> dict[str, Any]:
    """Load a snapshot and backfill surfaces in memory (no history rewrite)."""
    path = repository._resolve_snapshot_path(ticker, stamp)
    if path is None:
        return {}
    payload = repository._read_json(path)
    if not isinstance(payload, dict) or not payload:
        return {}
    working = dict(payload)
    missing = [
        key
        for key, val in repository.SURFACE_KEYS.items()
        if repository._surface_is_empty(working.get(key), val)
    ]
    if missing:
        frame = repository._records_to_frame(working.get("data")).copy()
        if not frame.empty:
            working.update(repository._compute_surfaces(frame.copy()))
    return working


def _event_impact_shared_axes(
    pts_a: pd.DataFrame,
    pts_b: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    empty = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))
    if pts_a is None or pts_b is None or pts_a.empty or pts_b.empty:
        return empty
    a = pts_a.copy()
    b = pts_b.copy()
    x_min = float(min(float(a["Strike"].min()), float(b["Strike"].min())))
    x_max = float(max(float(a["Strike"].max()), float(b["Strike"].max())))
    y_min = float(min(float(a["DaysToExpiry"].min()), float(b["DaysToExpiry"].min())))
    y_max = float(max(float(a["DaysToExpiry"].max()), float(b["DaysToExpiry"].max())))
    nx = int(max(2, min(VOL_SURFACE_STRIKE_POINTS, max(a["Strike"].nunique(), b["Strike"].nunique(), 8))))
    ny = int(max(2, min(VOL_SURFACE_DTE_POINTS, max(a["DaysToExpiry"].nunique(), b["DaysToExpiry"].nunique(), 8))))
    strike_axis = np.linspace(x_min, x_max if x_max > x_min else x_min + 1e-6, nx, dtype=np.float64)
    dte_axis = np.linspace(y_min, y_max if y_max > y_min else y_min + 1e-6, ny, dtype=np.float64)
    return strike_axis, dte_axis


def _event_impact_grid_abs_sum(grid: np.ndarray) -> float:
    arr = np.asarray(grid, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.sum(np.abs(finite)))


def event_impact_suggest_position(
    magnitude: float,
    *,
    gamma_magnitude: float = 0.0,
    vega_magnitude: float = 0.0,
) -> str:
    """Map shock magnitude / greek mix to a suggested options structure."""
    mag = float(magnitude) if np.isfinite(magnitude) else 0.0
    g_mag = float(gamma_magnitude) if np.isfinite(gamma_magnitude) else 0.0
    v_mag = float(vega_magnitude) if np.isfinite(vega_magnitude) else 0.0
    if mag <= 0.0:
        return "Low Shock: No action suggested"
    denom = g_mag + v_mag
    gamma_share = (g_mag / denom) if denom > 0 else 0.5
    if gamma_share >= 0.55:
        return "High Gamma Shock: Suggest Calendar Spreads"
    if gamma_share <= 0.45:
        return "High Vega Shock: Suggest Volatility Spreads"
    return "Mixed Shock: Suggest Defined-Risk Iron Condors"


def event_impact_calculate_shock(
    ticker: str,
    event_date: str | date | datetime,
    *,
    repo: DataRepository | None = None,
) -> dict[str, Any]:
    """Compute Shock Surface (Δ after − Δ before) and Gamma/Vega shock magnitude.

    Finds the two snapshots closest to ``event_date`` (one before, one after),
    interpolates Delta/Gamma/Vega onto a shared Strike×DTE grid, then::

        Shock Surface = Delta_After − Delta_Before
        Shock Magnitude = Σ|Γ_After − Γ_Before| + Σ|ν_After − ν_Before|

    All DataFrame operations use ``.copy()``. Returns a result dict with
    ``ok``, axes, long-format ``shock_frame``, magnitudes, and a position hint.
    """
    empty_frame = pd.DataFrame(columns=["Strike", "DaysToExpiry", "Shock"])
    result: dict[str, Any] = {
        "ok": False,
        "message": "",
        "ticker": str(ticker or "").strip().upper(),
        "event_date": str(event_date),
        "stamp_before": None,
        "stamp_after": None,
        "shock_frame": empty_frame.copy(),
        "strike_axis": np.empty(0, dtype=np.float64),
        "dte_axis": np.empty(0, dtype=np.float64),
        "shock_z": np.empty((0, 0), dtype=np.float64),
        "shock_magnitude": 0.0,
        "gamma_magnitude": 0.0,
        "vega_magnitude": 0.0,
        "suggestion": event_impact_suggest_position(0.0),
    }
    repository = repo if repo is not None else DataRepository()
    symbol = result["ticker"]
    if not symbol:
        result["message"] = "Ticker is required."
        return result

    stamp_before, stamp_after = event_impact_find_bracketing_snapshots(
        symbol, event_date, repo=repository
    )
    result["stamp_before"] = stamp_before
    result["stamp_after"] = stamp_after
    if not stamp_before or not stamp_after:
        result["message"] = (
            "Need one snapshot before and one after the event date "
            f"(before={stamp_before!r}, after={stamp_after!r})."
        )
        return result

    payload_before = _event_impact_payload_in_memory(repository, symbol, stamp_before)
    payload_after = _event_impact_payload_in_memory(repository, symbol, stamp_after)
    frame_before = repository._records_to_frame(payload_before.get("data")).copy()
    frame_after = repository._records_to_frame(payload_after.get("data")).copy()
    if frame_before.empty or frame_after.empty:
        result["message"] = "Bracketing snapshots have empty option chains."
        return result

    delta_before = repository._greek_points(frame_before.copy(), "Delta").copy()
    delta_after = repository._greek_points(frame_after.copy(), "Delta").copy()
    gamma_before = repository._greek_points(frame_before.copy(), "Gamma").copy()
    gamma_after = repository._greek_points(frame_after.copy(), "Gamma").copy()
    vega_before = repository._greek_points(frame_before.copy(), "Vega").copy()
    vega_after = repository._greek_points(frame_after.copy(), "Vega").copy()

    strike_axis, dte_axis = _event_impact_shared_axes(delta_before, delta_after)
    if strike_axis.size == 0 or dte_axis.size == 0:
        result["message"] = "Could not build a shared Strike × DTE grid."
        return result

    z_delta_before = repository._interpolate_delta_grid(
        delta_before.copy(), strike_axis, dte_axis, value_col="Delta"
    )
    z_delta_after = repository._interpolate_delta_grid(
        delta_after.copy(), strike_axis, dte_axis, value_col="Delta"
    )
    z_gamma_before = repository._interpolate_delta_grid(
        gamma_before.copy(), strike_axis, dte_axis, value_col="Gamma"
    )
    z_gamma_after = repository._interpolate_delta_grid(
        gamma_after.copy(), strike_axis, dte_axis, value_col="Gamma"
    )
    z_vega_before = repository._interpolate_delta_grid(
        vega_before.copy(), strike_axis, dte_axis, value_col="Vega"
    )
    z_vega_after = repository._interpolate_delta_grid(
        vega_after.copy(), strike_axis, dte_axis, value_col="Vega"
    )

    # Shock Surface = Delta_Surface_After − Delta_Surface_Before
    shock_z = np.array(z_delta_after - z_delta_before, dtype=np.float64, copy=True)
    gamma_mag = _event_impact_grid_abs_sum(z_gamma_after - z_gamma_before)
    vega_mag = _event_impact_grid_abs_sum(z_vega_after - z_vega_before)
    magnitude = float(gamma_mag + vega_mag)

    grid_x, grid_y = np.meshgrid(strike_axis, dte_axis)
    shock_frame = pd.DataFrame(
        {
            "Strike": np.asarray(grid_x, dtype=np.float64).ravel().copy(),
            "DaysToExpiry": np.asarray(grid_y, dtype=np.float64).ravel().copy(),
            "Shock": np.asarray(shock_z, dtype=np.float64).ravel().copy(),
        }
    ).copy()
    shock_frame.attrs["X"] = np.array(strike_axis, dtype=np.float64, copy=True)
    shock_frame.attrs["Y"] = np.array(dte_axis, dtype=np.float64, copy=True)
    shock_frame.attrs["Z"] = np.array(shock_z, dtype=np.float64, copy=True)

    result.update(
        {
            "ok": True,
            "message": "ok",
            "shock_frame": shock_frame.copy(),
            "strike_axis": np.array(strike_axis, dtype=np.float64, copy=True),
            "dte_axis": np.array(dte_axis, dtype=np.float64, copy=True),
            "shock_z": np.array(shock_z, dtype=np.float64, copy=True),
            "shock_magnitude": magnitude,
            "gamma_magnitude": gamma_mag,
            "vega_magnitude": vega_mag,
            "suggestion": event_impact_suggest_position(
                magnitude, gamma_magnitude=gamma_mag, vega_magnitude=vega_mag
            ),
        }
    )
    return result


# ---------------------------------------------------------------------------
# Market Regime & Reflexivity Engine
# ---------------------------------------------------------------------------
REGIME_STABLE_BULL = "Stable-Bull"
REGIME_SQUEEZE_UP = "Squeeze-Up"
REGIME_FRAGILE_BEAR = "Fragile-Bear"
REGIME_CRASH_CASCADE = "Crash-Cascade"
REGIME_STATES = (
    REGIME_STABLE_BULL,
    REGIME_SQUEEZE_UP,
    REGIME_FRAGILE_BEAR,
    REGIME_CRASH_CASCADE,
)
REGIME_VANNA_HIGH_QUANTILE = 0.65
REGIME_CASCADE_GEX_WEIGHT = 0.45
REGIME_CASCADE_VANNA_WEIGHT = 0.35
REGIME_CASCADE_MOM_WEIGHT = 0.20
REGIME_HEDGE_PUT_SPREADS = "Buy Put Spreads"


def regime_clip_probability(value: Any) -> float:
    """Bound cascade probability to ``[0, 1]``; non-finite → ``0.0``."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def regime_classify_states(
    gex: Any,
    vanna: Any,
    momentum: Any,
    *,
    vanna_high_threshold: float | None = None,
) -> np.ndarray:
    """Vectorized regime labels from GEX, Vanna, and price momentum.

    Rules (numpy boolean masks, no Python row loops):
    - Crash-Cascade: GEX < 0, momentum < 0, |Vanna| high
    - Squeeze-Up: GEX < 0, momentum ≥ 0
    - Fragile-Bear: momentum < 0 and not Crash-Cascade
    - Stable-Bull: otherwise (GEX ≥ 0 and momentum ≥ 0)

    ``vanna_high_threshold`` defaults to a robust scale of |Vanna|
    (``REGIME_VANNA_HIGH_QUANTILE`` of finite |Vanna|, floored at a tiny epsilon).
    """
    g = np.asarray(gex, dtype=np.float64).reshape(-1).copy()
    v = np.asarray(vanna, dtype=np.float64).reshape(-1).copy()
    m = np.asarray(momentum, dtype=np.float64).reshape(-1).copy()
    n = int(max(g.size, v.size, m.size))
    if g.size == 1 and n > 1:
        g = np.full(n, float(g[0]) if g.size else np.nan, dtype=np.float64)
    if v.size == 1 and n > 1:
        v = np.full(n, float(v[0]) if v.size else np.nan, dtype=np.float64)
    if m.size == 1 and n > 1:
        m = np.full(n, float(m[0]) if m.size else np.nan, dtype=np.float64)
    if not (g.size == v.size == m.size == n):
        raise ValueError("gex, vanna, and momentum must broadcast to the same length")

    abs_v = np.abs(v)
    finite_v = abs_v[np.isfinite(abs_v)]
    if vanna_high_threshold is None:
        if finite_v.size == 0:
            thr = 0.0
        else:
            thr = float(np.quantile(finite_v, REGIME_VANNA_HIGH_QUANTILE))
            if thr <= 0.0:
                thr = float(np.nanmax(finite_v)) if np.any(finite_v > 0) else 0.0
    else:
        thr = float(vanna_high_threshold)
    high_vanna = np.isfinite(abs_v) & (abs_v >= thr) & (thr > 0.0)
    gex_neg = np.isfinite(g) & (g < 0.0)
    mom_pos = np.isfinite(m) & (m >= 0.0)
    mom_neg = np.isfinite(m) & (m < 0.0)

    crash = gex_neg & mom_neg & high_vanna
    squeeze = gex_neg & mom_pos
    fragile = (~crash) & mom_neg
    # Remaining finite rows → Stable-Bull; unknown/NaN rows stay Stable-Bull as safe default.
    labels = np.full(n, REGIME_STABLE_BULL, dtype=object)
    labels[fragile] = REGIME_FRAGILE_BEAR
    labels[squeeze] = REGIME_SQUEEZE_UP
    labels[crash] = REGIME_CRASH_CASCADE
    return labels


def regime_cascade_probability(
    gex: Any,
    vanna: Any,
    momentum: Any,
    *,
    vanna_high_threshold: float | None = None,
    mom_ref: float = 0.05,
) -> np.ndarray:
    """Cascade risk in ``[0, 1]`` from negative GEX, elevated |Vanna|, and down-momentum.

    Weighted blend of unit scores; result is vectorized ``np.clip`` into ``[0, 1]``.
    Momentum uses a fixed reference scale (``mom_ref``, default 5% drop → score 1).
    """
    g = np.asarray(gex, dtype=np.float64).reshape(-1).copy()
    v = np.asarray(vanna, dtype=np.float64).reshape(-1).copy()
    m = np.asarray(momentum, dtype=np.float64).reshape(-1).copy()
    n = int(max(g.size, v.size, m.size, 1))
    if g.size == 1 and n > 1:
        g = np.full(n, float(g[0]), dtype=np.float64)
    if v.size == 1 and n > 1:
        v = np.full(n, float(v[0]), dtype=np.float64)
    if m.size == 1 and n > 1:
        m = np.full(n, float(m[0]), dtype=np.float64)

    abs_g = np.abs(g)
    abs_v = np.abs(v)
    neg_g = np.where(np.isfinite(g) & (g < 0.0), abs_g, 0.0)
    # Prefer relative scale when a threshold or span exists; else min-max.
    g_span = float(np.nanmax(neg_g) - np.nanmin(neg_g)) if neg_g.size else 0.0
    if g_span > 0.0:
        g_score = minmax_normalize_01(neg_g)
    else:
        g_ref = float(np.nanmax(neg_g)) if neg_g.size and float(np.nanmax(neg_g)) > 0 else 1.0
        g_score = np.clip(neg_g / g_ref, 0.0, 1.0)

    if vanna_high_threshold is not None and float(vanna_high_threshold) > 0.0:
        thr = float(vanna_high_threshold)
        v_score = np.clip(np.where(np.isfinite(abs_v), abs_v / thr, 0.0), 0.0, 1.0)
    else:
        v_span = float(np.nanmax(abs_v) - np.nanmin(np.where(np.isfinite(abs_v), abs_v, np.nan))) if abs_v.size else 0.0
        if np.isfinite(v_span) and v_span > 0.0:
            v_score = minmax_normalize_01(np.where(np.isfinite(abs_v), abs_v, 0.0))
        else:
            v_ref = float(np.nanmax(abs_v)) if abs_v.size and np.isfinite(np.nanmax(abs_v)) and float(np.nanmax(abs_v)) > 0 else 1.0
            v_score = np.clip(np.where(np.isfinite(abs_v), abs_v / v_ref, 0.0), 0.0, 1.0)

    mom_down = np.where(np.isfinite(m) & (m < 0.0), -m, 0.0)
    ref = float(mom_ref) if mom_ref and float(mom_ref) > 0 else 0.05
    m_score = np.clip(mom_down / ref, 0.0, 1.0)

    raw = (
        REGIME_CASCADE_GEX_WEIGHT * g_score
        + REGIME_CASCADE_VANNA_WEIGHT * v_score
        + REGIME_CASCADE_MOM_WEIGHT * m_score
    )
    return np.clip(np.asarray(raw, dtype=np.float64), 0.0, 1.0).copy()


def regime_reflexive_cascade_flag(gex: Any, vanna: Any, *, vanna_high_threshold: float | None = None) -> np.ndarray:
    """True where dealer GEX is negative and |Vanna| is high (Reflexive Cascade risk)."""
    g = np.asarray(gex, dtype=np.float64).reshape(-1).copy()
    v = np.asarray(vanna, dtype=np.float64).reshape(-1).copy()
    n = int(max(g.size, v.size, 1))
    if g.size == 1 and n > 1:
        g = np.full(n, float(g[0]), dtype=np.float64)
    if v.size == 1 and n > 1:
        v = np.full(n, float(v[0]), dtype=np.float64)
    abs_v = np.abs(v)
    finite_v = abs_v[np.isfinite(abs_v)]
    if vanna_high_threshold is None:
        thr = float(np.quantile(finite_v, REGIME_VANNA_HIGH_QUANTILE)) if finite_v.size else 0.0
        if thr <= 0.0 and finite_v.size:
            thr = float(np.nanmax(finite_v))
    else:
        thr = float(vanna_high_threshold)
    return (np.isfinite(g) & (g < 0.0) & np.isfinite(abs_v) & (abs_v >= thr) & (thr > 0.0)).copy()


def regime_net_dealer_gex(frame: pd.DataFrame) -> tuple[float, np.ndarray, np.ndarray]:
    """Net dealer-signed GEX plus per-row GEX and strike arrays (copied)."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return 0.0, np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    working = _attach_chain_gamma(frame.copy())
    gamma_col = _sentiment_column(working, "Gamma", "gamma")
    oi_col = _sentiment_column(working, "openInterest", "open_interest", "OI", "oi")
    price_col = _sentiment_column(working, "strike", "K", "Strike", "Price")
    if gamma_col is None or price_col is None:
        return 0.0, np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    gamma = np.array(pd.to_numeric(gamma_col, errors="coerce").to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    prices = np.array(pd.to_numeric(price_col, errors="coerce").to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    if oi_col is None:
        oi = np.ones(len(working.index), dtype=np.float64)
    else:
        oi = np.array(pd.to_numeric(oi_col, errors="coerce").to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
        oi = np.where(np.isfinite(oi) & (oi >= 0), oi, 0.0)
        if not np.any(oi > 0):
            oi = np.ones(len(working.index), dtype=np.float64)
    size_col = _sentiment_column(working, "contractSize", "contract_size", "multiplier")
    if size_col is None:
        size = np.full(prices.shape, float(OPTIONS_CONTRACT_SIZE), dtype=np.float64)
    else:
        size = np.array(pd.to_numeric(size_col, errors="coerce").to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
        size = np.where(np.isfinite(size) & (size > 0), size, float(OPTIONS_CONTRACT_SIZE))
    is_call, is_put = _sentiment_type_mask(working)
    sign = np.ones(prices.shape, dtype=np.float64)
    if np.any(is_put) or np.any(is_call):
        sign = np.where(is_put, -1.0, 1.0)
    valid = np.isfinite(gamma) & np.isfinite(prices) & (prices > 0) & np.isfinite(oi)
    with np.errstate(invalid="ignore", over="ignore"):
        gex = gamma * oi * size * sign
    gex = np.where(valid & np.isfinite(gex), gex, np.nan)
    net = float(np.nansum(gex)) if np.any(np.isfinite(gex)) else 0.0
    return net, np.array(gex, dtype=np.float64, copy=True), np.array(prices, dtype=np.float64, copy=True)


def regime_chain_vanna_exposure(frame: pd.DataFrame) -> tuple[float, np.ndarray]:
    """OI-weighted mean |Vanna| and per-row signed Vanna (copied)."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return 0.0, np.array([], dtype=np.float64)
    working = frame.copy()
    vanna, _volga = _chain_vanna_volga(working)
    vanna = np.array(vanna, dtype=np.float64, copy=True)
    oi_col = _sentiment_column(working, "openInterest", "open_interest", "OI", "oi")
    if oi_col is None:
        weights = np.ones(len(working.index), dtype=np.float64)
    else:
        weights = np.array(pd.to_numeric(oi_col, errors="coerce").to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, np.nan)
        if not np.any(np.isfinite(weights)):
            weights = np.ones(len(working.index), dtype=np.float64)
    with np.errstate(invalid="ignore"):
        abs_exp = np.abs(vanna) * weights
    score = _finite_mean(abs_exp)
    return (float(score) if np.isfinite(score) else 0.0), vanna


def regime_estimate_momentum(
    ticker: str,
    *,
    repo: DataRepository | None = None,
    spot: float | None = None,
) -> float:
    """Relative spot change from the two newest DataRepository snapshots.

    Returns ``(spot_new - spot_old) / spot_old``. Falls back to ``0.0`` when
    history is insufficient.
    """
    repository = repo if repo is not None else DataRepository()
    symbol = str(ticker or "").strip().upper()
    stamps = repository.get_available_snapshots(symbol)
    spots: list[float] = []
    for stamp in stamps[-8:]:
        payload = repository.load_from_cache(symbol, stamp) or {}
        raw = payload.get("spot")
        try:
            value = float(raw) if raw is not None else float("nan")
        except (TypeError, ValueError):
            value = float("nan")
        if not np.isfinite(value) or value <= 0:
            frame = repository._records_to_frame(payload.get("data")).copy()
            value = repository._spot_from_frame(frame) or float("nan")
        if np.isfinite(value) and value > 0:
            spots.append(float(value))
    if spot is not None:
        try:
            spot_f = float(spot)
        except (TypeError, ValueError):
            spot_f = float("nan")
        if np.isfinite(spot_f) and spot_f > 0:
            spots.append(spot_f)
    if len(spots) < 2:
        return 0.0
    old, new = float(spots[-2]), float(spots[-1])
    if old <= 0:
        return 0.0
    return float((new - old) / old)


def regime_build_map_frame(
    gex_rows: np.ndarray,
    vanna_rows: np.ndarray,
    momentum: float,
    *,
    vanna_high_threshold: float | None = None,
) -> pd.DataFrame:
    """Strike-level Regime Map points: X=GEX, Y=Vanna, Z=Momentum (+ Regime)."""
    g = np.asarray(gex_rows, dtype=np.float64).reshape(-1).copy()
    v = np.asarray(vanna_rows, dtype=np.float64).reshape(-1).copy()
    n = int(min(g.size, v.size))
    if n == 0:
        return pd.DataFrame(columns=["GEX", "Vanna", "Momentum", "Regime"]).copy()
    g = g[:n]
    v = v[:n]
    m = np.full(n, float(momentum), dtype=np.float64)
    valid = np.isfinite(g) & np.isfinite(v)
    g = g[valid]
    v = v[valid]
    m = m[valid]
    if g.size == 0:
        return pd.DataFrame(columns=["GEX", "Vanna", "Momentum", "Regime"]).copy()
    labels = regime_classify_states(g, v, m, vanna_high_threshold=vanna_high_threshold)
    return pd.DataFrame(
        {
            "GEX": np.array(g, dtype=np.float64, copy=True),
            "Vanna": np.array(v, dtype=np.float64, copy=True),
            "Momentum": np.array(m, dtype=np.float64, copy=True),
            "Regime": labels,
        }
    ).copy()


def regime_hedge_suggestion(regime: str, cascade_probability: float, reflexive: bool) -> str | None:
    """Hedge copy for fragile / cascade regimes; otherwise ``None``."""
    state = str(regime or "")
    p = regime_clip_probability(cascade_probability)
    if state in (REGIME_FRAGILE_BEAR, REGIME_CRASH_CASCADE) or reflexive or p >= 0.55:
        return REGIME_HEDGE_PUT_SPREADS
    return None


def _regime_empty_result(ticker: str, message: str) -> dict[str, Any]:
    return {
        "ticker": str(ticker or "").strip().upper(),
        "ok": False,
        "message": message,
        "regime": REGIME_STABLE_BULL,
        "cascade_probability": 0.0,
        "reflexive_cascade": False,
        "gex": 0.0,
        "vanna": 0.0,
        "momentum": 0.0,
        "hedge_suggestion": None,
        "map_frame": pd.DataFrame(columns=["GEX", "Vanna", "Momentum", "Regime"]).copy(),
        "vanna_high_threshold": 0.0,
    }


def calculate_market_reflexivity(
    ticker: str,
    *,
    repo: DataRepository | None = None,
    frame: pd.DataFrame | None = None,
    spot: float | None = None,
    momentum: float | None = None,
) -> dict[str, Any]:
    """Correlate dealer GEX, Vanna, and price momentum into a market regime.

    Classifies one of ``Stable-Bull``, ``Squeeze-Up``, ``Fragile-Bear``,
    ``Crash-Cascade``. When net GEX is negative and aggregate |Vanna| is high,
    sets ``reflexive_cascade`` and elevates cascade probability (always in
    ``[0, 1]``). Returns a Regime Map DataFrame for 3D scatter plotting.

    Args:
        ticker: Underlying symbol (DataRepository lookup when ``frame`` omitted).
        repo: Optional repository override.
        frame: Optional chain DataFrame (``.copy()`` used; caller frame untouched).
        spot: Optional spot override for momentum estimation.
        momentum: Optional relative momentum override (e.g. ``-0.02`` = −2%).

    Returns:
        Dict with regime label, cascade probability, map frame, and hedge hint.
    """
    symbol = str(ticker or "").strip().upper()
    if frame is None:
        chain, snap_spot = _snapshot_chain_frame(symbol, repo=repo)
        use_spot = spot if spot is not None else snap_spot
    else:
        chain = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        use_spot = spot
        if use_spot is None and not chain.empty:
            use_spot = DataRepository()._spot_from_frame(chain)

    if chain is None or not isinstance(chain, pd.DataFrame) or chain.empty:
        return _regime_empty_result(symbol, "No option chain available for regime analysis.")

    working = chain.copy()
    net_gex, gex_rows, _strikes = regime_net_dealer_gex(working)
    vanna_score, vanna_rows = regime_chain_vanna_exposure(working)
    if momentum is None:
        mom = regime_estimate_momentum(symbol, repo=repo, spot=use_spot)
    else:
        try:
            mom = float(momentum)
        except (TypeError, ValueError):
            mom = 0.0
        if not np.isfinite(mom):
            mom = 0.0

    abs_v = np.abs(vanna_rows[np.isfinite(vanna_rows)]) if vanna_rows.size else np.array([], dtype=np.float64)
    if abs_v.size:
        thr = float(np.quantile(abs_v, REGIME_VANNA_HIGH_QUANTILE))
        if thr <= 0.0:
            thr = float(np.nanmax(abs_v))
    else:
        thr = float(vanna_score) if vanna_score > 0 else 0.0

    labels = regime_classify_states(
        np.array([net_gex], dtype=np.float64),
        np.array([vanna_score], dtype=np.float64),
        np.array([mom], dtype=np.float64),
        vanna_high_threshold=thr if thr > 0 else None,
    )
    regime = str(labels[0])
    probs = regime_cascade_probability(
        np.array([net_gex], dtype=np.float64),
        np.array([vanna_score], dtype=np.float64),
        np.array([mom], dtype=np.float64),
        vanna_high_threshold=thr if thr > 0 else None,
    )
    cascade_p = regime_clip_probability(probs[0] if probs.size else 0.0)
    reflexive = bool(
        regime_reflexive_cascade_flag(
            np.array([net_gex], dtype=np.float64),
            np.array([vanna_score], dtype=np.float64),
            vanna_high_threshold=thr if thr > 0 else None,
        )[0]
    )
    if reflexive and regime == REGIME_FRAGILE_BEAR:
        regime = REGIME_CRASH_CASCADE
        cascade_p = regime_clip_probability(max(cascade_p, 0.7))
    elif reflexive:
        cascade_p = regime_clip_probability(max(cascade_p, 0.55))

    map_frame = regime_build_map_frame(
        gex_rows,
        vanna_rows,
        mom,
        vanna_high_threshold=thr if thr > 0 else None,
    )
    hedge = regime_hedge_suggestion(regime, cascade_p, reflexive)
    return {
        "ticker": symbol,
        "ok": True,
        "message": "ok",
        "regime": regime,
        "cascade_probability": cascade_p,
        "reflexive_cascade": reflexive,
        "gex": float(net_gex),
        "vanna": float(vanna_score),
        "momentum": float(mom),
        "hedge_suggestion": hedge,
        "map_frame": map_frame.copy(),
        "vanna_high_threshold": float(thr),
    }


# ---------------------------------------------------------------------------
# Liquidation Waterfall Engine (isolated; all helpers use test_ prefix)
# ---------------------------------------------------------------------------
TEST_LIQUIDATION_ZONE = "Liquidation Zone"
TEST_LIQUIDATION_STABLE = "Stable"
TEST_LIQUIDATION_UNSTABLE_MSG = "Liquidation Engine: GEX data unstable."
TEST_CASCADE_ZONE_THRESHOLD = 0.65
TEST_STABILITY_NEAR_PCT = 0.02
TEST_STABILITY_REF_PCT = 0.05


def test_audit_gex_normalized(gex: Any) -> dict[str, Any]:
    """Auditor: verify GEX values are finite after normalization.

    Non-finite input (NaN / ±inf) after ``minmax_normalize_01`` → halt render
    with ``TEST_LIQUIDATION_UNSTABLE_MSG``. Empty input is treated as unstable.
    """
    arr = np.asarray(gex, dtype=np.float64)
    if arr.size == 0:
        return {
            "ok": False,
            "message": TEST_LIQUIDATION_UNSTABLE_MSG,
            "gex_normalized": np.array([], dtype=np.float64),
            "halt_render": True,
        }
    if not np.all(np.isfinite(arr)):
        return {
            "ok": False,
            "message": TEST_LIQUIDATION_UNSTABLE_MSG,
            "gex_normalized": np.array(arr, dtype=np.float64, copy=True),
            "halt_render": True,
        }
    normalized = minmax_normalize_01(arr)
    if normalized.size == 0 or not np.all(np.isfinite(normalized)):
        return {
            "ok": False,
            "message": TEST_LIQUIDATION_UNSTABLE_MSG,
            "gex_normalized": np.array(normalized, dtype=np.float64, copy=True),
            "halt_render": True,
        }
    return {
        "ok": True,
        "message": "ok",
        "gex_normalized": np.array(normalized, dtype=np.float64, copy=True),
        "halt_render": False,
    }


def test_detect_gamma_flips(strikes: Any, gex: Any) -> np.ndarray:
    """Interpolated prices where net GEX crosses from positive to negative.

    Strikes are sorted ascending. A flip is recorded when ``gex[i] >= 0`` and
    ``gex[i + 1] < 0`` (GEX + → − as price falls through the level), matching
    the dealer long-gamma / short-gamma convention used by ``analyze_gex_outlook``.
    """
    k = np.asarray(strikes, dtype=np.float64).reshape(-1).copy()
    g = np.asarray(gex, dtype=np.float64).reshape(-1).copy()
    n = int(min(k.size, g.size))
    if n < 2:
        return np.array([], dtype=np.float64)
    k = k[:n]
    g = g[:n]
    valid = np.isfinite(k) & (k > 0) & np.isfinite(g)
    k = k[valid]
    g = g[valid]
    if k.size < 2:
        return np.array([], dtype=np.float64)
    order = np.argsort(k)
    k = k[order]
    g = g[order]
    # Aggregate duplicate strikes (vectorized unique + bincount-style sum).
    uniq_k, inv = np.unique(k, return_inverse=True)
    sums = np.zeros(uniq_k.shape, dtype=np.float64)
    np.add.at(sums, inv, g)
    if uniq_k.size < 2:
        return np.array([], dtype=np.float64)
    flips: list[float] = []
    for i in range(int(uniq_k.size) - 1):
        high_gex = float(sums[i + 1])
        low_gex = float(sums[i])
        # Ascending: lower strike then higher. Flip when higher strike is ≥0 and
        # lower strike is <0 (crossing + → − as price declines).
        if high_gex >= 0.0 and low_gex < 0.0:
            denom = high_gex - low_gex
            if abs(denom) < 1e-18:
                flip = float(uniq_k[i + 1])
            else:
                weight = high_gex / denom
                flip = float(uniq_k[i + 1] + weight * (uniq_k[i] - uniq_k[i + 1]))
            if np.isfinite(flip) and flip > 0:
                flips.append(flip)
    return np.array(flips, dtype=np.float64, copy=True)


def test_cascade_score(gex: Any) -> np.ndarray:
    """Per-strike cascade score in ``[0, 1]``; highly negative GEX → high score.

    Positive / zero GEX map to 0. Non-finite cells → 0. Always copies.
    """
    g = np.asarray(gex, dtype=np.float64)
    out = np.zeros(g.shape, dtype=np.float64)
    if g.size == 0:
        return out
    neg = np.where(np.isfinite(g) & (g < 0.0), -g, 0.0)
    span = float(np.nanmax(neg) - np.nanmin(neg)) if neg.size else 0.0
    if np.isfinite(span) and span > 0.0:
        scored = minmax_normalize_01(neg)
    else:
        ref = float(np.nanmax(neg)) if neg.size and float(np.nanmax(neg)) > 0 else 1.0
        scored = np.clip(neg / ref, 0.0, 1.0)
    scored = np.asarray(scored, dtype=np.float64)
    scored = np.where(np.isfinite(scored), scored, 0.0)
    return np.clip(scored, 0.0, 1.0).copy()


def test_classify_liquidation_zones(
    cascade: Any,
    *,
    threshold: float = TEST_CASCADE_ZONE_THRESHOLD,
) -> np.ndarray:
    """Label ``Liquidation Zone`` where cascade score ≥ threshold; else ``Stable``."""
    c = np.asarray(cascade, dtype=np.float64)
    thr = float(threshold) if np.isfinite(threshold) else TEST_CASCADE_ZONE_THRESHOLD
    labels = np.full(c.shape, TEST_LIQUIDATION_STABLE, dtype=object)
    hot = np.isfinite(c) & (c >= thr)
    labels[hot] = TEST_LIQUIDATION_ZONE
    return labels.copy()


def test_net_gex_by_strike(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate dealer-signed GEX by strike (copied frame; put-negative convention)."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=["Price", "GEX", "CascadeScore", "Zone"]).copy()
    working = _attach_chain_gamma(frame.copy())
    gamma_col = _sentiment_column(working, "Gamma", "gamma")
    oi_col = _sentiment_column(working, "openInterest", "open_interest", "OI", "oi")
    price_col = _sentiment_column(working, "strike", "K", "Strike", "Price")
    if gamma_col is None or price_col is None:
        return pd.DataFrame(columns=["Price", "GEX", "CascadeScore", "Zone"]).copy()
    gamma = np.array(pd.to_numeric(gamma_col, errors="coerce").to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    prices = np.array(pd.to_numeric(price_col, errors="coerce").to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    if oi_col is None:
        oi = np.ones(len(working.index), dtype=np.float64)
    else:
        oi = np.array(pd.to_numeric(oi_col, errors="coerce").to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
        oi = np.where(np.isfinite(oi) & (oi >= 0), oi, 0.0)
        if not np.any(oi > 0):
            oi = np.ones(len(working.index), dtype=np.float64)
    size_col = _sentiment_column(working, "contractSize", "contract_size", "multiplier")
    if size_col is None:
        size = np.full(prices.shape, float(OPTIONS_CONTRACT_SIZE), dtype=np.float64)
    else:
        size = np.array(pd.to_numeric(size_col, errors="coerce").to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
        size = np.where(np.isfinite(size) & (size > 0), size, float(OPTIONS_CONTRACT_SIZE))
    is_call, is_put = _sentiment_type_mask(working)
    sign = np.ones(prices.shape, dtype=np.float64)
    if np.any(is_put) or np.any(is_call):
        sign = np.where(is_put, -1.0, 1.0)
    valid = np.isfinite(gamma) & np.isfinite(prices) & (prices > 0) & np.isfinite(oi)
    with np.errstate(invalid="ignore", over="ignore"):
        gex = gamma * oi * size * sign
    gex = np.where(valid & np.isfinite(gex), gex, np.nan)
    table = pd.DataFrame({"Price": prices, "GEX": gex}).dropna().copy()
    if table.empty:
        return pd.DataFrame(columns=["Price", "GEX", "CascadeScore", "Zone"]).copy()
    net = table.groupby("Price", sort=True, as_index=False)["GEX"].sum().copy()
    scores = test_cascade_score(net["GEX"].to_numpy(dtype=np.float64))
    zones = test_classify_liquidation_zones(scores)
    net = net.copy()
    net["CascadeScore"] = scores
    net["Zone"] = zones
    return net.copy()


def test_build_gex_surface(frame: pd.DataFrame) -> dict[str, Any]:
    """GEX / cascade surface across full strike × tenor range.

    Returns ``price_axis`` (X), ``expiry_axis`` (Y = days to expiry), ``gex_z``,
    ``cascade_z``, and ``zone_mask`` for heatmap rendering.
    """
    empty = {
        "price_axis": np.array([], dtype=np.float64),
        "expiry_axis": np.array([], dtype=np.float64),
        "gex_z": np.zeros((0, 0), dtype=np.float64),
        "cascade_z": np.zeros((0, 0), dtype=np.float64),
        "zone_mask": np.zeros((0, 0), dtype=bool),
        "ok": False,
    }
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return empty
    working = _attach_chain_gamma(frame.copy())
    gamma_col = _sentiment_column(working, "Gamma", "gamma")
    oi_col = _sentiment_column(working, "openInterest", "open_interest", "OI", "oi")
    price_col = _sentiment_column(working, "strike", "K", "Strike", "Price")
    if gamma_col is None or price_col is None:
        return empty
    gamma = np.array(pd.to_numeric(gamma_col, errors="coerce").to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    prices = np.array(pd.to_numeric(price_col, errors="coerce").to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    dtes = np.array(_sentiment_tenor_days(working), dtype=np.float64, copy=True)
    if oi_col is None:
        oi = np.ones(len(working.index), dtype=np.float64)
    else:
        oi = np.array(pd.to_numeric(oi_col, errors="coerce").to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
        oi = np.where(np.isfinite(oi) & (oi >= 0), oi, 0.0)
        if not np.any(oi > 0):
            oi = np.ones(len(working.index), dtype=np.float64)
    size_col = _sentiment_column(working, "contractSize", "contract_size", "multiplier")
    if size_col is None:
        size = np.full(prices.shape, float(OPTIONS_CONTRACT_SIZE), dtype=np.float64)
    else:
        size = np.array(pd.to_numeric(size_col, errors="coerce").to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
        size = np.where(np.isfinite(size) & (size > 0), size, float(OPTIONS_CONTRACT_SIZE))
    is_call, is_put = _sentiment_type_mask(working)
    sign = np.ones(prices.shape, dtype=np.float64)
    if np.any(is_put) or np.any(is_call):
        sign = np.where(is_put, -1.0, 1.0)
    valid = (
        np.isfinite(gamma)
        & np.isfinite(prices)
        & (prices > 0)
        & np.isfinite(oi)
        & np.isfinite(dtes)
        & (dtes >= 0)
    )
    with np.errstate(invalid="ignore", over="ignore"):
        gex = gamma * oi * size * sign
    gex = np.where(valid & np.isfinite(gex), gex, np.nan)
    table = pd.DataFrame(
        {
            "Price": prices,
            "DTE": dtes,
            "GEX": gex,
        }
    ).dropna().copy()
    if table.empty:
        return empty
    pivot = (
        table.groupby(["DTE", "Price"], sort=True, as_index=False)["GEX"]
        .sum()
        .copy()
    )
    if pivot.empty:
        return empty
    wide = pivot.pivot(index="DTE", columns="Price", values="GEX").sort_index(axis=0).sort_index(axis=1)
    wide = wide.copy().fillna(0.0)
    price_axis = np.array(wide.columns.to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    expiry_axis = np.array(wide.index.to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    gex_z = np.array(wide.to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    # Sparse pivot NaNs are filled above; ±inf from overflow is left for the auditor.
    cascade_z = test_cascade_score(np.where(np.isfinite(gex_z), gex_z, 0.0))
    zone_mask = cascade_z >= float(TEST_CASCADE_ZONE_THRESHOLD)
    return {
        "price_axis": price_axis,
        "expiry_axis": expiry_axis,
        "gex_z": gex_z,
        "cascade_z": np.array(cascade_z, dtype=np.float64, copy=True),
        "zone_mask": np.array(zone_mask, dtype=bool, copy=True),
        "ok": True,
    }


def test_next_gamma_flip(spot: Any, flips: Any) -> float | None:
    """Nearest Gamma Flip at or below spot (waterfall downside); else nearest overall."""
    try:
        spot_f = float(spot)
    except (TypeError, ValueError):
        spot_f = float("nan")
    arr = np.asarray(flips, dtype=np.float64).reshape(-1).copy()
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0 or not np.isfinite(spot_f) or spot_f <= 0:
        return None
    below = arr[arr <= spot_f]
    if below.size:
        return float(below[np.argmax(below)])
    return float(arr[np.argmin(np.abs(arr - spot_f))])


def test_market_stability_distance(spot: Any, next_flip: Any) -> dict[str, Any]:
    """Distance from spot to next Gamma Flip; approach risk in ``[0, 1]``.

    ``approach_risk`` → 1 when within ``TEST_STABILITY_NEAR_PCT`` of the flip
    (aggressive red), → 0 when distance ≥ ``TEST_STABILITY_REF_PCT``.
    """
    try:
        spot_f = float(spot)
    except (TypeError, ValueError):
        spot_f = float("nan")
    try:
        flip_f = float(next_flip) if next_flip is not None else float("nan")
    except (TypeError, ValueError):
        flip_f = float("nan")
    if not np.isfinite(spot_f) or spot_f <= 0 or not np.isfinite(flip_f) or flip_f <= 0:
        return {
            "distance_abs": float("nan"),
            "distance_pct": float("nan"),
            "approach_risk": 0.0,
            "near_flip": False,
            "stability": 1.0,
        }
    dist_abs = float(abs(spot_f - flip_f))
    dist_pct = float(dist_abs / spot_f)
    near = dist_pct <= float(TEST_STABILITY_NEAR_PCT)
    ref = float(TEST_STABILITY_REF_PCT) if TEST_STABILITY_REF_PCT > 0 else 0.05
    approach = float(np.clip(1.0 - dist_pct / ref, 0.0, 1.0))
    if near:
        approach = max(approach, 0.85)
    stability = float(np.clip(1.0 - approach, 0.0, 1.0))
    return {
        "distance_abs": dist_abs,
        "distance_pct": dist_pct,
        "approach_risk": approach,
        "near_flip": bool(near),
        "stability": stability,
    }


def test_simulate_liquidation_greeks(
    ticker: str,
    flip_price: float,
    *,
    spot: float | None = None,
    repo: DataRepository | None = None,
    frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Portfolio / chain Greek reaction if spot hits the next Gamma Flip zone.

    Shifts spot by ``(flip - spot) / spot * 100`` percent via
    :func:`calculate_stress_scenario` (IV held flat).
    """
    try:
        flip_f = float(flip_price)
    except (TypeError, ValueError):
        flip_f = float("nan")
    if frame is None:
        chain, snap_spot = _snapshot_chain_frame(str(ticker or "").strip().upper(), repo=repo)
        use_spot = spot if spot is not None else snap_spot
    else:
        chain = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        use_spot = spot
        if use_spot is None and not chain.empty:
            use_spot = DataRepository()._spot_from_frame(chain)
    try:
        spot_f = float(use_spot) if use_spot is not None else float("nan")
    except (TypeError, ValueError):
        spot_f = float("nan")
    if not np.isfinite(flip_f) or flip_f <= 0 or not np.isfinite(spot_f) or spot_f <= 0:
        return {
            "ok": False,
            "message": "Flip or spot unavailable for liquidation simulation.",
            "spot_shift": 0.0,
            "impact": {},
        }
    spot_shift = float((flip_f - spot_f) / spot_f * 100.0)
    impact = calculate_stress_scenario(
        str(ticker or "").strip().upper(),
        spot_shift,
        0.0,
        repo=repo,
        frame=chain,
        spot=spot_f,
    )
    return {
        "ok": True,
        "message": "ok",
        "spot_shift": spot_shift,
        "flip_price": flip_f,
        "spot": spot_f,
        "impact": impact,
    }


def _test_liquidation_empty(ticker: str, message: str) -> dict[str, Any]:
    return {
        "ticker": str(ticker or "").strip().upper(),
        "ok": False,
        "message": message,
        "spot": None,
        "strike_frame": pd.DataFrame(columns=["Price", "GEX", "CascadeScore", "Zone"]).copy(),
        "gamma_flips": np.array([], dtype=np.float64),
        "next_gamma_flip": None,
        "stability": {
            "distance_abs": float("nan"),
            "distance_pct": float("nan"),
            "approach_risk": 0.0,
            "near_flip": False,
            "stability": 1.0,
        },
        "surface": {
            "price_axis": np.array([], dtype=np.float64),
            "expiry_axis": np.array([], dtype=np.float64),
            "gex_z": np.zeros((0, 0), dtype=np.float64),
            "cascade_z": np.zeros((0, 0), dtype=np.float64),
            "zone_mask": np.zeros((0, 0), dtype=bool),
            "ok": False,
        },
        "gex_audit": {
            "ok": False,
            "message": message,
            "gex_normalized": np.array([], dtype=np.float64),
            "halt_render": True,
        },
        "net_gex": 0.0,
        "liquidation_zone_count": 0,
    }


def test_calculate_liquidation_waterfall(
    ticker: str,
    *,
    repo: DataRepository | None = None,
    frame: pd.DataFrame | None = None,
    spot: float | None = None,
) -> dict[str, Any]:
    """Liquidation Waterfall: GEX surface, Gamma Flips, cascade / Liquidation Zones.

    Conceptual entry ``calculate_liquidation_waterfall(ticker)`` — implemented
    under the ``test_`` isolation prefix. Builds full-strike GEX, detects flips
    (GEX + → −), scores cascade risk per strike, and audits GEX finiteness before
    any heatmap render.

    Args:
        ticker: Underlying symbol (DataRepository when ``frame`` omitted).
        repo: Optional repository override.
        frame: Optional chain DataFrame (``.copy()`` used; caller untouched).
        spot: Optional spot override for stability distance.

    Returns:
        Dict with surface grids, flip list, stability metrics, and audit payload.
    """
    symbol = str(ticker or "").strip().upper()
    if frame is None:
        chain, snap_spot = _snapshot_chain_frame(symbol, repo=repo)
        use_spot = spot if spot is not None else snap_spot
    else:
        chain = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        use_spot = spot
        if use_spot is None and not chain.empty:
            use_spot = DataRepository()._spot_from_frame(chain)

    if chain is None or not isinstance(chain, pd.DataFrame) or chain.empty:
        return _test_liquidation_empty(symbol, "No option chain available for liquidation waterfall.")

    working = chain.copy()
    strike_frame = test_net_gex_by_strike(working)
    if strike_frame.empty:
        return _test_liquidation_empty(symbol, "GEX surface could not be built from chain.")

    prices = strike_frame["Price"].to_numpy(dtype=np.float64)
    gex_vals = strike_frame["GEX"].to_numpy(dtype=np.float64)
    flips = test_detect_gamma_flips(prices, gex_vals)
    next_flip = test_next_gamma_flip(use_spot, flips)
    stability = test_market_stability_distance(use_spot, next_flip)
    surface = test_build_gex_surface(working)
    # Auditor operates on the heatmap cascade / GEX grid (prefer surface GEX).
    audit_source = surface.get("gex_z") if surface.get("ok") else gex_vals
    gex_audit = test_audit_gex_normalized(audit_source)
    net_gex = float(np.nansum(gex_vals)) if np.any(np.isfinite(gex_vals)) else 0.0
    zone_count = int(np.count_nonzero(strike_frame["Zone"].to_numpy() == TEST_LIQUIDATION_ZONE))

    return {
        "ticker": symbol,
        "ok": True,
        "message": "ok",
        "spot": float(use_spot) if use_spot is not None and np.isfinite(float(use_spot)) else None,
        "strike_frame": strike_frame.copy(),
        "gamma_flips": np.array(flips, dtype=np.float64, copy=True),
        "next_gamma_flip": next_flip,
        "stability": stability,
        "surface": surface,
        "gex_audit": gex_audit,
        "net_gex": net_gex,
        "liquidation_zone_count": zone_count,
    }


# Public alias (no collision with other tabs); isolation entry remains test_*.
calculate_liquidation_waterfall = test_calculate_liquidation_waterfall


# ---------------------------------------------------------------------------
# Godlike Quant Strategist — re-exports (HMM regime + Monte Carlo PoP/PoT)
# ---------------------------------------------------------------------------
from quant_engine_regime import (  # noqa: E402
    HMM_MIN_OBSERVATIONS,
    HMM_REGIME_CRASH_CASCADE,
    HMM_REGIME_STABLE,
    HMM_REGIME_STATES,
    HMM_REGIME_TREND,
    build_regime_features,
    classify_market_regime,
    require_regime_before_trade,
)
from quant_engine_montecarlo import (  # noqa: E402
    HIGH_VANNA_WARNING,
    MATHEMATICIANS_RULE_MSG,
    MC_MIN_PATHS,
    audit_monte_carlo_path_count,
    build_vanna_volga_report,
    evaluate_position_montecarlo,
    mathematicians_rule_gate,
    probability_of_profit,
    probability_of_touching,
    simulate_gbm_paths,
    strategist_evaluate_position,
    test_audit_monte_carlo_paths,
)
from quant_engine_debate import (  # noqa: E402
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
    expert_a_momentum,
    expert_b_mean_reversion,
    expert_c_volatility_arb,
    run_council_debate,
)

# ---------------------------------------------------------------------------
# Social Sentiment Specialist — alpha + MC factor (auditor-normalized [0, 1])
# ---------------------------------------------------------------------------
from sentiment_engine import (  # noqa: E402
    SENTIMENT_UNSTABLE_MSG,
    audit_sentiment_normalized,
    clip_sentiment_01,
    extract_portfolio_sentiment,
    extract_ticker_sentiment,
    filter_social_posts,
    normalize_compound_to_01,
    sentiment_factor_for_montecarlo,
)


def calculate_sentiment_alpha(
    ticker: str,
    *,
    prices: Any = None,
    sentiment_scores: Any = None,
    force_stub: bool = True,
    spike_z: float = 1.0,
) -> dict[str, Any]:
    """Correlate sentiment spikes with subsequent price / realized-vol changes.

    Returns auditor-friendly normalized scores in ``[0, 1]`` plus Pearson
    correlations suitable for Monte Carlo Sentiment Factor consumption.
    ``force_stub=True`` by default so CI / offline paths stay deterministic.
    """
    symbol = str(ticker or "").strip().upper() or "SPY"
    sources_breakdown: dict[str, Any] = {}
    if sentiment_scores is None:
        payload = extract_ticker_sentiment(symbol, force_stub=force_stub)
        posts = payload.get("posts")
        if isinstance(posts, pd.DataFrame) and not posts.empty and "score_01" in posts.columns:
            scores = pd.to_numeric(posts["score_01"], errors="coerce").to_numpy(dtype=np.float64)
        else:
            scores = np.array([float(payload.get("score_01", 0.5))], dtype=np.float64)
        buzz = float(payload.get("buzz") or 0.0)
        uoa_flag = bool(payload.get("uoa_flag"))
        mention_count = int(payload.get("mention_count") or 0)
        source = str(payload.get("source") or "stub")
        sources_breakdown = dict(payload.get("sources") or {})
        reddit_block = payload.get("reddit") or {}
        fourchan_block = payload.get("fourchan") or {}
    else:
        scores = np.asarray(sentiment_scores, dtype=np.float64).ravel()
        buzz = float(np.clip(np.log1p(scores.size) / np.log1p(100.0), 0.0, 1.0))
        uoa_flag = False
        mention_count = int(scores.size)
        source = "injected"
        reddit_block = {}
        fourchan_block = {}

    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0 or not np.any(np.isfinite(scores)):
        scores = np.array([0.5], dtype=np.float64)

    # Price path: injected series, else synthetic path seeded by ticker (offline-safe).
    if prices is None:
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        n = max(int(scores.size), 16)
        rets = rng.normal(0.0, 0.01, size=n)
        # Embed a mild causal link: high sentiment → slight positive next return.
        sent = np.resize(scores, n)
        rets = rets + 0.02 * (sent - 0.5)
        px = 100.0 * np.exp(np.cumsum(rets))
    else:
        px = np.asarray(prices, dtype=np.float64).ravel()
        px = px[np.isfinite(px) & (px > 0.0)]
        if px.size < 3:
            rng = np.random.default_rng(0)
            px = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=max(scores.size, 16))))

    log_rets = np.diff(np.log(px))
    # Realized vol over a rolling window of 5 returns (annualized-ish daily).
    window = min(5, max(log_rets.size, 1))
    if log_rets.size >= window:
        rv = np.array(
            [
                float(np.std(log_rets[max(0, i - window + 1) : i + 1]) * np.sqrt(252.0))
                for i in range(log_rets.size)
            ],
            dtype=np.float64,
        )
    else:
        rv = np.array([], dtype=np.float64)

    n_align = min(scores.size, log_rets.size)
    if n_align < 2:
        # Pad scores to match returns when only a scalar aggregate is available.
        sent_series = np.resize(scores, max(log_rets.size, 2))
        n_align = min(sent_series.size, log_rets.size)
    else:
        sent_series = scores[:n_align]
    sent_aligned = np.asarray(sent_series[:n_align], dtype=np.float64)
    ret_aligned = np.asarray(log_rets[:n_align], dtype=np.float64)
    rv_aligned = np.asarray(rv[:n_align], dtype=np.float64) if rv.size >= n_align else rv

    finite_s = np.isfinite(sent_aligned)
    finite_r = np.isfinite(ret_aligned)
    mask = finite_s & finite_r
    if int(mask.sum()) >= 2:
        corr_price = float(np.corrcoef(sent_aligned[mask], ret_aligned[mask])[0, 1])
    else:
        corr_price = 0.0
    if not np.isfinite(corr_price):
        corr_price = 0.0

    if rv_aligned.size >= 2:
        m2 = np.isfinite(sent_aligned[: rv_aligned.size]) & np.isfinite(rv_aligned)
        if int(m2.sum()) >= 2:
            corr_vol = float(np.corrcoef(sent_aligned[: rv_aligned.size][m2], rv_aligned[m2])[0, 1])
        else:
            corr_vol = 0.0
    else:
        corr_vol = 0.0
    if not np.isfinite(corr_vol):
        corr_vol = 0.0

    mu = float(np.nanmean(sent_aligned)) if sent_aligned.size else 0.5
    sigma_s = float(np.nanstd(sent_aligned)) if sent_aligned.size else 0.0
    if np.isfinite(sigma_s) and sigma_s > 1e-12:
        spikes = sent_aligned >= (mu + float(spike_z) * sigma_s)
    else:
        spikes = np.zeros(sent_aligned.shape, dtype=bool)
    if int(spikes.sum()) > 0 and ret_aligned.size == sent_aligned.size:
        spike_ret = float(np.nanmean(ret_aligned[spikes]))
        spike_vol = (
            float(np.nanmean(rv_aligned[spikes[: rv_aligned.size]]))
            if rv_aligned.size == sent_aligned.size
            else float("nan")
        )
    else:
        spike_ret = 0.0
        spike_vol = float("nan")

    mean_score = clip_sentiment_01(float(np.nanmean(scores)) if scores.size else 0.5)
    # Alpha proxy: sentiment–return correlation scaled to [0, 1] via (ρ+1)/2, modulated by |spike ret|.
    alpha_raw = 0.5 * (corr_price + 1.0)
    if np.isfinite(spike_ret):
        alpha_raw = float(np.clip(alpha_raw + 2.0 * spike_ret, 0.0, 1.0))
    alpha_01 = clip_sentiment_01(alpha_raw)
    scores_01 = np.array([clip_sentiment_01(x) for x in scores.tolist()], dtype=np.float64)
    audit = audit_sentiment_normalized(scores_01)
    mc_factor = sentiment_factor_for_montecarlo(mean_score)

    # Correlation chart series (sentiment vs realized vol), length-aligned.
    chart_n = min(sent_aligned.size, rv_aligned.size) if rv_aligned.size else sent_aligned.size
    chart_sentiment = np.array(sent_aligned[:chart_n], dtype=np.float64, copy=True)
    if rv_aligned.size:
        chart_vol = np.array(rv_aligned[:chart_n], dtype=np.float64, copy=True)
    else:
        chart_vol = np.full(chart_n, 0.0, dtype=np.float64)

    return {
        "ok": True,
        "ticker": symbol,
        "score_01": mean_score,
        "alpha_01": alpha_01,
        "buzz": clip_sentiment_01(buzz),
        "mention_count": mention_count,
        "uoa_flag": uoa_flag,
        "corr_price": float(corr_price),
        "corr_vol": float(corr_vol),
        "spike_count": int(spikes.sum()) if spikes.size else 0,
        "spike_mean_return": spike_ret,
        "spike_mean_realized_vol": spike_vol if np.isfinite(spike_vol) else None,
        "scores_01": scores_01,
        "audit": audit,
        "sentiment_factor": mc_factor,
        "chart_sentiment": chart_sentiment,
        "chart_realized_vol": chart_vol,
        "source": source,
        "sources": sources_breakdown,
        "reddit": {
            "score_01": clip_sentiment_01(reddit_block.get("score_01", mean_score)),
            "buzz": clip_sentiment_01(reddit_block.get("buzz", 0.0)),
            "mention_count": int(reddit_block.get("mention_count") or 0),
            "source": str(reddit_block.get("source") or "empty"),
            "uoa_flag": bool(reddit_block.get("uoa_flag")),
        },
        "fourchan": {
            "score_01": clip_sentiment_01(fourchan_block.get("score_01", mean_score)),
            "buzz": clip_sentiment_01(fourchan_block.get("buzz", 0.0)),
            "mention_count": int(fourchan_block.get("mention_count") or 0),
            "source": str(fourchan_block.get("source") or "empty"),
            "uoa_flag": bool(fourchan_block.get("uoa_flag")),
        },
    }
