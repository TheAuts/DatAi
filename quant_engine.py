"""Black–Scholes–Merton Greeks for the DatAi MVP."""

from __future__ import annotations

from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Any, Iterable, Mapping

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
MODEL_BLACK_SCHOLES = "black-scholes"
MODEL_HESTON = "heston"
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
) -> pd.DataFrame:
    """Build a vectorized Price vs Greeks curve for a Streamlit dashboard.

    ``model`` is ``black-scholes`` or ``heston``. Heston uses the same dense
    ``price_range`` grid as Black–Scholes (no down-sampling).
    """
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
    """Vanna ∂Δ/∂σ. ``model`` is ``black-scholes`` or ``heston``."""
    spots = np.asarray(S, dtype=np.float64)
    key = tuple(np.atleast_1d(spots).tolist())
    vanna, _ = _second_order_cached(
        key, float(K), float(T), float(r), float(sigma), str(model), float(q), str(option_type), v0, float(kappa), float(theta), float(heston_sigma), float(rho)
    )
    return _pack_second_order(S, np.array(vanna, dtype=np.float64))


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
    """Volga ∂²V/∂σ². ``model`` is ``black-scholes`` or ``heston``."""
    spots = np.asarray(S, dtype=np.float64)
    key = tuple(np.atleast_1d(spots).tolist())
    _, volga = _second_order_cached(
        key, float(K), float(T), float(r), float(sigma), str(model), float(q), str(option_type), v0, float(kappa), float(theta), float(heston_sigma), float(rho)
    )
    return _pack_second_order(S, np.array(volga, dtype=np.float64))


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


def generate_vol_surface_data(ticker: str) -> pd.DataFrame | str:
    """Implied-vol grid: Strike (X) × Days-to-Expiry (Y) × IV (Z) for Plotly."""
    from data_ingestion import fetch_option_chain, get_available_expirations

    def _finish(frame: pd.DataFrame) -> pd.DataFrame | str:
        out = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        if not out.empty:
            out["Strike"] = out["Strike"].astype(float)
            out["DaysToExpiry"] = out["DaysToExpiry"].astype(float)
            out["IV"] = out["IV"].astype(float)
        print("generate_vol_surface_data shape:", out.shape)
        print(out.head())
        if len(out) < 10:
            return "Insufficient Data"
        return out

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
        return _finish(pd.DataFrame(columns=["Strike", "DaysToExpiry", "IV"]))
    if chain.empty:
        return _finish(pd.DataFrame(columns=["Strike", "DaysToExpiry", "IV"]))

    strikes = pd.to_numeric(chain.get("strike", chain.get("K")), errors="coerce")
    ivs = pd.to_numeric(chain.get("impliedVolatility", chain.get("sigma")), errors="coerce")
    if "T" in chain.columns:
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
            empty["Veta"] = earlier["Vega"] - base["Vega"]
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
    """Vomma (Volga) ∂²V/∂σ². ``model`` is ``black-scholes`` or ``heston``.

    Returns None/NaN when T, sigma, or S are non-positive or the divide is undefined.
    """
    values = _pro_metrics_vector(
        S, K, T, r, sigma, model, q=q, option_type=option_type, v0=v0, kappa=kappa, theta=theta, heston_sigma=heston_sigma, rho=rho
    )
    return _pack_second_order(S, values["Vomma"])


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
    """Zomma ∂Γ/∂σ. ``model`` is ``black-scholes`` or ``heston``.

    Returns None/NaN when T, sigma, or S are non-positive or sigma is near zero.
    """
    values = _pro_metrics_vector(
        S, K, T, r, sigma, model, q=q, option_type=option_type, v0=v0, kappa=kappa, theta=theta, heston_sigma=heston_sigma, rho=rho
    )
    return _pack_second_order(S, values["Zomma"])


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
    """Veta ∂Vega/∂t per calendar day. ``model`` is ``black-scholes`` or ``heston``.

    Returns None/NaN when T, sigma, or S are non-positive.
    """
    values = _pro_metrics_vector(
        S, K, T, r, sigma, model, q=q, option_type=option_type, v0=v0, kappa=kappa, theta=theta, heston_sigma=heston_sigma, rho=rho
    )
    return _pack_second_order(S, values["Veta"])


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
    """Ultima ∂Vomma/∂σ. ``model`` is ``black-scholes`` or ``heston``.

    Returns None/NaN when T, sigma, or S are non-positive or sigma is near zero.
    """
    values = _pro_metrics_vector(
        S, K, T, r, sigma, model, q=q, option_type=option_type, v0=v0, kappa=kappa, theta=theta, heston_sigma=heston_sigma, rho=rho
    )
    return _pack_second_order(S, values["Ultima"])


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
) -> pd.DataFrame:
    """3D Price × Days-to-Expiry grid for a Pro Metric (Vanna, Vomma, Zomma, Veta, Ultima).

    ``ticker`` is unused in pricing and kept for dashboard identity.
    Invalid or unknown ``greek_name`` defaults to Vanna. Edge cells are NaN, not inf.
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
    return pd.DataFrame(rows)
