"""HMM market-regime classification for DatAi Quant Strategist (Godlike Tier).

Classifies the current market as Stable | Trend | Crash-Cascade from a price
series via a 3-state Gaussian HMM (numpy EM + Viterbi). DataFrames are copied
before mutation; callers' inputs are never modified in place.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

HMM_REGIME_STABLE = "Stable"
HMM_REGIME_TREND = "Trend"
HMM_REGIME_CRASH_CASCADE = "Crash-Cascade"
HMM_REGIME_STATES: tuple[str, ...] = (
    HMM_REGIME_STABLE,
    HMM_REGIME_TREND,
    HMM_REGIME_CRASH_CASCADE,
)

# State index convention for the fitted HMM.
_STATE_STABLE = 0
_STATE_TREND = 1
_STATE_CRASH = 2

HMM_MIN_OBSERVATIONS = 30
HMM_EM_ITERS = 25
HMM_VOL_WINDOW = 5
_EPS = 1e-12


def _as_float_series(values: Any) -> np.ndarray:
    """Coerce prices/returns to a finite 1-D float64 copy."""
    if isinstance(values, pd.Series):
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64).copy()
    elif isinstance(values, pd.DataFrame):
        if values.empty:
            return np.array([], dtype=np.float64)
        col = values.iloc[:, 0]
        arr = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64).copy()
    else:
        arr = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    return arr


def _log_returns(prices: np.ndarray) -> np.ndarray:
    p = np.asarray(prices, dtype=np.float64).reshape(-1).copy()
    if p.size < 2:
        return np.array([], dtype=np.float64)
    valid = np.isfinite(p) & (p > 0.0)
    if int(np.count_nonzero(valid)) < 2:
        return np.array([], dtype=np.float64)
    # Keep only positive finite prices (contiguous in observation time).
    clean = p[valid]
    return np.diff(np.log(clean)).astype(np.float64, copy=True)


def build_regime_features(prices: Any, *, vol_window: int = HMM_VOL_WINDOW) -> np.ndarray:
    """Build (n, 2) features: log-return and rolling |return| vol proxy.

    Returns an empty (0, 2) array when the series is too short.
    """
    rets = _log_returns(_as_float_series(prices))
    if rets.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    w = max(int(vol_window), 1)
    abs_r = np.abs(rets)
    # Causal rolling mean of |r| via cumsum (O(n), fully vectorized).
    c = np.concatenate(([0.0], np.cumsum(abs_r, dtype=np.float64)))
    idx = np.arange(1, abs_r.size + 1, dtype=np.int64)
    lo = np.maximum(0, idx - w)
    vol = (c[idx] - c[lo]) / (idx - lo).astype(np.float64)
    feats = np.column_stack([rets, vol]).astype(np.float64, copy=True)
    finite = np.all(np.isfinite(feats), axis=1)
    return feats[finite].copy()


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    shift = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - shift)
    s = np.sum(e, axis=1, keepdims=True)
    return e / np.maximum(s, _EPS)


def _log_gaussian_pdf(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Per-row log N(x | mean, diag(var)); shape (n, k)."""
    # x: (n, d), mean/var: (k, d)
    n, d = x.shape
    k = mean.shape[0]
    out = np.zeros((n, k), dtype=np.float64)
    for j in range(k):
        diff = x - mean[j]
        v = np.maximum(var[j], _EPS)
        out[:, j] = -0.5 * (np.sum(np.log(2.0 * np.pi * v)) + np.sum((diff * diff) / v, axis=1))
    return out


def _forward_backward(
    log_emit: np.ndarray,
    log_start: np.ndarray,
    log_trans: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Scaled forward-backward in log space; returns gamma, xi_sum, loglik."""
    n, k = log_emit.shape
    log_alpha = np.full((n, k), -np.inf, dtype=np.float64)
    log_alpha[0] = log_start + log_emit[0]
    for t in range(1, n):
        for j in range(k):
            log_alpha[t, j] = log_emit[t, j] + np.logaddexp.reduce(log_alpha[t - 1] + log_trans[:, j])

    log_beta = np.zeros((n, k), dtype=np.float64)
    for t in range(n - 2, -1, -1):
        for i in range(k):
            log_beta[t, i] = np.logaddexp.reduce(log_trans[i] + log_emit[t + 1] + log_beta[t + 1])

    log_lik = float(np.logaddexp.reduce(log_alpha[-1]))
    log_gamma = log_alpha + log_beta
    log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    xi_sum = np.zeros((k, k), dtype=np.float64)
    for t in range(n - 1):
        log_xi = log_alpha[t][:, None] + log_trans + log_emit[t + 1][None, :] + log_beta[t + 1][None, :]
        log_xi -= np.logaddexp.reduce(log_xi.ravel())
        xi_sum += np.exp(log_xi)
    return gamma, xi_sum, log_lik


def _viterbi(log_emit: np.ndarray, log_start: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
    n, k = log_emit.shape
    dp = np.full((n, k), -np.inf, dtype=np.float64)
    ptr = np.zeros((n, k), dtype=np.int64)
    dp[0] = log_start + log_emit[0]
    for t in range(1, n):
        for j in range(k):
            scores = dp[t - 1] + log_trans[:, j]
            ptr[t, j] = int(np.argmax(scores))
            dp[t, j] = float(scores[ptr[t, j]]) + log_emit[t, j]
    states = np.zeros(n, dtype=np.int64)
    states[-1] = int(np.argmax(dp[-1]))
    for t in range(n - 2, -1, -1):
        states[t] = ptr[t + 1, states[t + 1]]
    return states


def _init_params(feats: np.ndarray, n_states: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Heuristic init: sort by vol proxy; assign terciles to Stable/Trend/Crash."""
    n, d = feats.shape
    order = np.argsort(feats[:, 1])  # ascending vol
    cuts = np.array_split(order, n_states)
    # Map terciles: low vol→Stable, mid→Trend, high→Crash.
    mean = np.zeros((n_states, d), dtype=np.float64)
    var = np.ones((n_states, d), dtype=np.float64)
    mapping = (_STATE_STABLE, _STATE_TREND, _STATE_CRASH)
    for bucket, state in zip(cuts, mapping):
        if bucket.size == 0:
            continue
        chunk = feats[bucket]
        mean[state] = np.mean(chunk, axis=0)
        var[state] = np.maximum(np.var(chunk, axis=0), 1e-6)
    # Nudge Crash toward negative mean return / high vol if collapsed.
    if mean[_STATE_CRASH, 0] > mean[_STATE_STABLE, 0]:
        mean[_STATE_CRASH, 0] = mean[_STATE_STABLE, 0] - abs(mean[_STATE_STABLE, 0]) - 1e-3
    if mean[_STATE_CRASH, 1] < mean[_STATE_TREND, 1]:
        mean[_STATE_CRASH, 1] = mean[_STATE_TREND, 1] * 1.5 + 1e-4
    start = np.full(n_states, 1.0 / n_states, dtype=np.float64)
    # Sticky transitions with elevated Crash self-loop.
    trans = np.full((n_states, n_states), 0.05 / max(n_states - 1, 1), dtype=np.float64)
    np.fill_diagonal(trans, 0.90)
    trans[_STATE_CRASH, _STATE_CRASH] = 0.85
    trans[_STATE_CRASH, _STATE_STABLE] = 0.10
    trans[_STATE_CRASH, _STATE_TREND] = 0.05
    trans = trans / trans.sum(axis=1, keepdims=True)
    return start, trans, mean, var


def fit_gaussian_hmm(
    feats: np.ndarray,
    *,
    n_states: int = 3,
    n_iter: int = HMM_EM_ITERS,
) -> dict[str, Any]:
    """Fit a diagonal-covariance Gaussian HMM via Baum–Welch EM."""
    x = np.asarray(feats, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("feats must be a (n, d) array with n >= 2")
    start, trans, mean, var = _init_params(x, n_states=n_states)
    log_lik = -np.inf
    for _ in range(max(int(n_iter), 1)):
        log_emit = _log_gaussian_pdf(x, mean, var)
        log_start = np.log(np.maximum(start, _EPS))
        log_trans = np.log(np.maximum(trans, _EPS))
        gamma, xi_sum, ll = _forward_backward(log_emit, log_start, log_trans)
        # M-step
        start = gamma[0].copy()
        start = start / max(float(start.sum()), _EPS)
        row = np.maximum(xi_sum.sum(axis=1, keepdims=True), _EPS)
        trans = xi_sum / row
        trans = trans / np.maximum(trans.sum(axis=1, keepdims=True), _EPS)
        weights = np.maximum(gamma.sum(axis=0), _EPS)
        mean = (gamma.T @ x) / weights[:, None]
        for j in range(n_states):
            diff = x - mean[j]
            var[j] = np.maximum((gamma[:, j][:, None] * (diff * diff)).sum(axis=0) / weights[j], 1e-6)
        if abs(ll - log_lik) < 1e-8:
            log_lik = ll
            break
        log_lik = ll
    return {
        "start": start.copy(),
        "trans": trans.copy(),
        "mean": mean.copy(),
        "var": var.copy(),
        "log_likelihood": float(log_lik),
        "n_states": int(n_states),
    }


def decode_regime_states(feats: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    """Viterbi state path for fitted ``model``; returns int labels in ``[0, k)``."""
    x = np.asarray(feats, dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    var = np.asarray(model["var"], dtype=np.float64)
    start = np.asarray(model["start"], dtype=np.float64)
    trans = np.asarray(model["trans"], dtype=np.float64)
    log_emit = _log_gaussian_pdf(x, mean, var)
    return _viterbi(log_emit, np.log(np.maximum(start, _EPS)), np.log(np.maximum(trans, _EPS)))


def state_index_to_label(state: int) -> str:
    """Map HMM state index → Godlike regime label."""
    idx = int(state)
    if idx == _STATE_TREND:
        return HMM_REGIME_TREND
    if idx == _STATE_CRASH:
        return HMM_REGIME_CRASH_CASCADE
    return HMM_REGIME_STABLE


def _empty_regime_result(message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "message": message,
        "regime": HMM_REGIME_STABLE,
        "regime_path": np.array([], dtype=object),
        "state_path": np.array([], dtype=np.int64),
        "posteriors": np.zeros((0, 3), dtype=np.float64),
        "features": np.zeros((0, 2), dtype=np.float64),
        "model": None,
        "n_observations": 0,
    }


def classify_market_regime(
    prices: Any,
    *,
    frame: pd.DataFrame | None = None,
    price_col: str | None = None,
    n_iter: int = HMM_EM_ITERS,
    min_observations: int = HMM_MIN_OBSERVATIONS,
) -> dict[str, Any]:
    """Classify current market via HMM as Stable | Trend | Crash-Cascade.

    Prefers ``prices`` when provided; otherwise reads ``price_col`` (or the first
    numeric column) from ``frame.copy()``.

    Returns:
        Dict with ``ok``, ``regime`` (current / last state), full ``regime_path``,
        Viterbi ``state_path``, soft ``posteriors``, and fitted ``model``.
    """
    series: Any
    if frame is not None:
        working = frame.copy()
        if working.empty:
            return _empty_regime_result("Empty price frame.")
        if price_col and price_col in working.columns:
            series = working[price_col]
        else:
            numeric = working.select_dtypes(include=["number"])
            if numeric.empty:
                return _empty_regime_result("No numeric price column in frame.")
            series = numeric.iloc[:, 0]
    else:
        series = prices

    feats = build_regime_features(series)
    n_obs = int(feats.shape[0])
    if n_obs < max(int(min_observations), 2):
        return _empty_regime_result(
            f"Insufficient observations for HMM ({n_obs} < {max(int(min_observations), 2)})."
        )

    model = fit_gaussian_hmm(feats, n_states=3, n_iter=n_iter)
    states = decode_regime_states(feats, model)
    labels = np.array([state_index_to_label(int(s)) for s in states], dtype=object)
    log_emit = _log_gaussian_pdf(feats, model["mean"], model["var"])
    posteriors = _softmax_rows(log_emit)
    current = str(labels[-1])
    if current not in HMM_REGIME_STATES:
        current = HMM_REGIME_STABLE
    return {
        "ok": True,
        "message": "ok",
        "regime": current,
        "regime_path": labels.copy(),
        "state_path": np.array(states, dtype=np.int64, copy=True),
        "posteriors": posteriors.copy(),
        "features": feats.copy(),
        "model": model,
        "n_observations": n_obs,
    }


def require_regime_before_trade(regime_result: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    """Gate: refuse trade approval until HMM regime is one of the three labels."""
    if not isinstance(regime_result, dict) or not regime_result.get("ok"):
        return {
            "approved": False,
            "regime": None,
            "message": "HMM regime classification required before trade approval.",
        }
    regime = str(regime_result.get("regime") or "")
    if regime not in HMM_REGIME_STATES:
        return {
            "approved": False,
            "regime": regime or None,
            "message": "HMM regime must be Stable, Trend, or Crash-Cascade before trade approval.",
        }
    return {
        "approved": True,
        "regime": regime,
        "message": f"Market classified as {regime}.",
    }
