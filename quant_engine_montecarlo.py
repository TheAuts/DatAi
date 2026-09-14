"""Monte Carlo PoP / PoT engine for DatAi Quant Strategist (Godlike Tier).

Every suggested position is simulated with ≥ ``MC_MIN_PATHS`` GBM paths and
reports Probability of Profit (PoP) and Probability of Touching the stop-loss
(PoT). Exposes auditor + Mathematician's Rule gates for the Strategist.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MC_MIN_PATHS = 10_000
MC_DEFAULT_STEPS = 64
MC_AUDIT_FAIL_MSG = "Monte Carlo path count below statistical significance threshold (≥ 10,000)."
HIGH_VANNA_WARNING = "High Vanna exposure: Delta will accelerate during IV spikes."
VANNA_EXPOSURE_THRESHOLD = 0.05
MATHEMATICIANS_RULE_MSG = (
    "Mathematician's Rule: FORBIDDEN — PoT exceeds PoP and position is not delta-hedged."
)


def audit_monte_carlo_path_count(n_paths: Any) -> dict[str, Any]:
    """Auditor: require Monte Carlo simulations use ≥ ``MC_MIN_PATHS`` paths."""
    try:
        n = int(n_paths)
    except (TypeError, ValueError):
        n = -1
    ok = n >= MC_MIN_PATHS
    return {
        "ok": ok,
        "n_paths": n,
        "min_paths": MC_MIN_PATHS,
        "message": "ok" if ok else MC_AUDIT_FAIL_MSG,
        "halt": not ok,
    }


def simulate_gbm_paths(
    spot: float,
    sigma: float,
    time_years: float,
    *,
    rate: float = 0.0,
    n_paths: int = MC_MIN_PATHS,
    n_steps: int = MC_DEFAULT_STEPS,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate geometric Brownian motion paths; shape ``(n_paths, n_steps + 1)``.

    Column 0 is ``spot``. Uses antithetic normals when ``n_paths`` is even.
    """
    s0 = float(spot)
    vol = max(float(sigma), 0.0)
    t = max(float(time_years), 0.0)
    r = float(rate)
    paths = max(int(n_paths), 1)
    steps = max(int(n_steps), 1)
    if not np.isfinite(s0) or s0 <= 0.0:
        raise ValueError("spot must be a positive finite float")
    if t <= 0.0 or vol <= 0.0:
        out = np.full((paths, steps + 1), s0, dtype=np.float64)
        return out

    dt = t / float(steps)
    drift = (r - 0.5 * vol * vol) * dt
    shock = vol * np.sqrt(dt)
    rng = np.random.default_rng(seed)
    half = paths // 2
    if paths >= 2 and half * 2 == paths:
        z = rng.standard_normal((half, steps))
        z = np.vstack([z, -z])
    else:
        z = rng.standard_normal((paths, steps))
    log_steps = drift + shock * z
    log_paths = np.cumsum(log_steps, axis=1)
    prices = s0 * np.exp(log_paths)
    return np.concatenate([np.full((paths, 1), s0, dtype=np.float64), prices], axis=1)


def _direction_sign(direction: str) -> float:
    d = str(direction or "long").strip().lower()
    if d in {"short", "sell", "-1", "put"}:
        return -1.0
    return 1.0


def probability_of_profit(
    paths: np.ndarray,
    *,
    entry: float,
    direction: str = "long",
    take_profit: float | None = None,
) -> float:
    """PoP ∈ [0, 1]: fraction of paths with terminal PnL > 0 (optional TP hit)."""
    p = np.asarray(paths, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        return 0.0
    terminal = p[:, -1]
    sign = _direction_sign(direction)
    entry_f = float(entry)
    pnl = sign * (terminal - entry_f)
    profitable = pnl > 0.0
    if take_profit is not None and np.isfinite(float(take_profit)):
        tp = float(take_profit)
        if sign > 0.0:
            hit_tp = np.any(p >= tp, axis=1)
        else:
            hit_tp = np.any(p <= tp, axis=1)
        profitable = profitable | hit_tp
    return float(np.mean(profitable.astype(np.float64)))


def probability_of_touching(
    paths: np.ndarray,
    stop_loss: float,
    *,
    direction: str = "long",
) -> float:
    """PoT ∈ [0, 1]: fraction of paths that touch ``stop_loss`` at any step."""
    p = np.asarray(paths, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        return 0.0
    sl = float(stop_loss)
    if not np.isfinite(sl):
        return 0.0
    sign = _direction_sign(direction)
    if sign > 0.0:
        touched = np.any(p <= sl, axis=1)
    else:
        touched = np.any(p >= sl, axis=1)
    return float(np.mean(touched.astype(np.float64)))


def clip_unit_interval(value: Any) -> float:
    """Bound a probability to ``[0, 1]``; non-finite → ``0.0``."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def mathematicians_rule_gate(
    pop: float,
    pot: float,
    *,
    delta_hedged: bool = False,
) -> dict[str, Any]:
    """Mathematician's Rule: forbid approval when PoT > PoP unless delta-hedged."""
    p_pop = clip_unit_interval(pop)
    p_pot = clip_unit_interval(pot)
    if p_pot > p_pop and not bool(delta_hedged):
        return {
            "approved": False,
            "forbidden": True,
            "pop": p_pop,
            "pot": p_pot,
            "delta_hedged": False,
            "message": MATHEMATICIANS_RULE_MSG,
        }
    return {
        "approved": True,
        "forbidden": False,
        "pop": p_pop,
        "pot": p_pot,
        "delta_hedged": bool(delta_hedged),
        "message": "ok",
    }


def build_vanna_volga_report(
    vanna: Any,
    volga: Any,
    *,
    vanna_threshold: float = VANNA_EXPOSURE_THRESHOLD,
) -> dict[str, Any]:
    """Include Vanna/Volga on every position report; emit exact high-Vanna warn."""
    try:
        v = float(vanna)
    except (TypeError, ValueError):
        v = float("nan")
    try:
        g = float(volga)
    except (TypeError, ValueError):
        g = float("nan")
    high = bool(np.isfinite(v) and abs(v) >= float(vanna_threshold))
    warning = HIGH_VANNA_WARNING if high else None
    return {
        "vanna": v if np.isfinite(v) else None,
        "volga": g if np.isfinite(g) else None,
        "high_vanna_exposure": high,
        "vanna_warning": warning,
    }


def evaluate_position_montecarlo(
    spot: float,
    stop_loss: float,
    *,
    sigma: float,
    time_years: float,
    rate: float = 0.0,
    direction: str = "long",
    take_profit: float | None = None,
    entry: float | None = None,
    n_paths: int = MC_MIN_PATHS,
    n_steps: int = MC_DEFAULT_STEPS,
    seed: int | None = None,
    vanna: float | None = None,
    volga: float | None = None,
    delta_hedged: bool = False,
    sentiment_factor: float | None = None,
) -> dict[str, Any]:
    """Simulate ≥ ``MC_MIN_PATHS`` paths and report PoP, PoT, Greeks, rule gate.

    Optional ``sentiment_factor`` ∈ [0, 1] (from Social Sentiment Specialist)
    nudges drift and vol via ``sentiment_factor_for_montecarlo``.
    """
    paths_req = max(int(n_paths), MC_MIN_PATHS)
    audit = audit_monte_carlo_path_count(paths_req)
    entry_px = float(entry) if entry is not None else float(spot)
    sigma_eff = float(sigma)
    rate_eff = float(rate)
    sentiment_payload: dict[str, float] | None = None
    if sentiment_factor is not None:
        try:
            from sentiment_engine import sentiment_factor_for_montecarlo

            sentiment_payload = sentiment_factor_for_montecarlo(sentiment_factor)
            sigma_eff = max(float(sigma) * float(sentiment_payload["vol_mult"]), 0.0)
            rate_eff = float(rate) + float(sentiment_payload["drift_bias"])
        except Exception:
            sentiment_payload = None
    paths = simulate_gbm_paths(
        float(spot),
        sigma_eff,
        float(time_years),
        rate=rate_eff,
        n_paths=paths_req,
        n_steps=int(n_steps),
        seed=seed,
    )
    pop = clip_unit_interval(
        probability_of_profit(paths, entry=entry_px, direction=direction, take_profit=take_profit)
    )
    pot = clip_unit_interval(probability_of_touching(paths, float(stop_loss), direction=direction))
    greeks = build_vanna_volga_report(vanna, volga)
    rule = mathematicians_rule_gate(pop, pot, delta_hedged=delta_hedged)
    result: dict[str, Any] = {
        "ok": bool(audit["ok"]),
        "n_paths": int(paths.shape[0]),
        "n_steps": int(paths.shape[1] - 1),
        "pop": pop,
        "pot": pot,
        "entry": entry_px,
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit) if take_profit is not None else None,
        "direction": "short" if _direction_sign(direction) < 0 else "long",
        "delta_hedged": bool(delta_hedged),
        "path_audit": audit,
        "mathematicians_rule": rule,
        "approved": bool(audit["ok"] and rule["approved"]),
        "paths": paths,
        "sigma_effective": sigma_eff,
        "rate_effective": rate_eff,
        "sentiment_factor": (
            float(sentiment_payload["sentiment_factor"]) if sentiment_payload is not None else None
        ),
        "sentiment_mc": sentiment_payload,
    }
    result.update(greeks)
    if greeks["vanna_warning"]:
        result["warning"] = greeks["vanna_warning"]
    return result


def strategist_evaluate_position(
    prices: Any,
    *,
    spot: float | None = None,
    stop_loss: float,
    sigma: float | None = None,
    time_years: float = 5.0 / 252.0,
    rate: float = 0.0,
    direction: str = "long",
    take_profit: float | None = None,
    vanna: float | None = None,
    volga: float | None = None,
    delta_hedged: bool = False,
    n_paths: int = MC_MIN_PATHS,
    seed: int | None = None,
    frame: pd.DataFrame | None = None,
    sentiment_factor: float | None = None,
) -> dict[str, Any]:
    """Full Godlike evaluation: HMM regime → MC PoP/PoT → Mathematician's Rule.

    Imports regime classification lazily so the Monte Carlo module stays usable
    without forcing HMM fit on every isolated PoP/PoT call.
    Optional ``sentiment_factor`` ∈ [0, 1] feeds the Monte Carlo Sentiment Factor.
    """
    from quant_engine_regime import classify_market_regime, require_regime_before_trade

    series = prices
    if frame is not None:
        working = frame.copy()
        series = working if series is None else series
    regime = classify_market_regime(series if series is not None else frame, frame=frame)
    regime_gate = require_regime_before_trade(regime)

    px = _infer_spot(series, frame=frame, spot=spot)
    vol = float(sigma) if sigma is not None else _infer_sigma(series, frame=frame)
    if not np.isfinite(px) or px <= 0.0:
        return {
            "ok": False,
            "approved": False,
            "message": "Valid spot required for Monte Carlo evaluation.",
            "regime": regime,
            "regime_gate": regime_gate,
            "pop": 0.0,
            "pot": 0.0,
        }

    # Default stop: 5% adverse move when caller passes NaN sentinel via None already required.
    sl = float(stop_loss)
    mc = evaluate_position_montecarlo(
        px,
        sl,
        sigma=vol,
        time_years=float(time_years),
        rate=float(rate),
        direction=direction,
        take_profit=take_profit,
        entry=px,
        n_paths=n_paths,
        seed=seed,
        vanna=vanna,
        volga=volga,
        delta_hedged=delta_hedged,
        sentiment_factor=sentiment_factor,
    )
    # Drop bulky path matrix from strategist summary (available via evaluate_*).
    paths = mc.pop("paths", None)
    approved = bool(regime_gate.get("approved") and mc.get("approved"))
    message = "ok"
    if not regime_gate.get("approved"):
        message = str(regime_gate.get("message") or "Regime gate failed.")
        approved = False
    elif not mc.get("approved"):
        message = str(mc.get("mathematicians_rule", {}).get("message") or mc.get("path_audit", {}).get("message") or "Rejected.")
    out: dict[str, Any] = {
        "ok": bool(regime.get("ok") and mc.get("ok")),
        "approved": approved,
        "message": message,
        "regime": regime.get("regime"),
        "regime_detail": regime,
        "regime_gate": regime_gate,
        "montecarlo": mc,
        "pop": mc.get("pop"),
        "pot": mc.get("pot"),
        "vanna": mc.get("vanna"),
        "volga": mc.get("volga"),
        "vanna_warning": mc.get("vanna_warning"),
        "n_paths": mc.get("n_paths"),
        "path_audit": mc.get("path_audit"),
        "mathematicians_rule": mc.get("mathematicians_rule"),
    }
    if mc.get("warning"):
        out["warning"] = mc["warning"]
    # Keep reference for advanced callers without forcing serialization cost in tests.
    out["_paths_ref"] = paths
    return out


def _infer_spot(prices: Any, *, frame: pd.DataFrame | None, spot: float | None) -> float:
    if spot is not None:
        try:
            s = float(spot)
            if np.isfinite(s) and s > 0.0:
                return s
        except (TypeError, ValueError):
            pass
    arr = _price_array(prices, frame=frame)
    if arr.size:
        last = float(arr[-1])
        if np.isfinite(last) and last > 0.0:
            return last
    return float("nan")


def _infer_sigma(prices: Any, *, frame: pd.DataFrame | None) -> float:
    arr = _price_array(prices, frame=frame)
    if arr.size < 3:
        return 0.20
    valid = arr[np.isfinite(arr) & (arr > 0.0)]
    if valid.size < 3:
        return 0.20
    rets = np.diff(np.log(valid))
    if rets.size == 0 or not np.any(np.isfinite(rets)):
        return 0.20
    return float(max(np.nanstd(rets) * np.sqrt(252.0), 1e-6))


def _price_array(prices: Any, *, frame: pd.DataFrame | None) -> np.ndarray:
    if frame is not None:
        working = frame.copy()
        numeric = working.select_dtypes(include=["number"])
        if not numeric.empty:
            return pd.to_numeric(numeric.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64).copy()
    if isinstance(prices, pd.Series):
        return pd.to_numeric(prices, errors="coerce").to_numpy(dtype=np.float64).copy()
    if isinstance(prices, pd.DataFrame):
        working = prices.copy()
        numeric = working.select_dtypes(include=["number"])
        if numeric.empty:
            return np.array([], dtype=np.float64)
        return pd.to_numeric(numeric.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64).copy()
    if prices is None:
        return np.array([], dtype=np.float64)
    return np.asarray(prices, dtype=np.float64).reshape(-1).copy()


# Auditor alias matching liquidation-engine ``test_audit_*`` naming.
def test_audit_monte_carlo_paths(n_paths: Any) -> dict[str, Any]:
    """Auditor hook: verify Monte Carlo path count ≥ 10,000."""
    return audit_monte_carlo_path_count(n_paths)
