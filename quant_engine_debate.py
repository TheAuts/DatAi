"""Council of Experts debate engine — Momentum, Mean Reversion, Volatility Arb.

Runs all three models on the current DataRepository snapshot and returns a
summary with stance, confidence (0–100), and mathematical justification.
DataFrames are always ``.copy()``'d; numpy used for IV rank / MC aggregates.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from quant_engine_montecarlo import (
    MC_MIN_PATHS,
    build_vanna_volga_report,
    evaluate_position_montecarlo,
    simulate_gbm_paths,
)

STANCE_BUY = "Buy"
STANCE_SELL = "Sell"
STANCE_NEUTRAL = "Neutral"
STANCES: tuple[str, ...] = (STANCE_BUY, STANCE_SELL, STANCE_NEUTRAL)

EXPERT_A = "A"
EXPERT_B = "B"
EXPERT_C = "C"
EXPERT_NAMES: dict[str, str] = {
    EXPERT_A: "Momentum",
    EXPERT_B: "Mean Reversion",
    EXPERT_C: "Volatility Arb",
}

IV_RANK_LOOKBACK = 30
MC_DEBATE_PATHS = MC_MIN_PATHS
MC_DEBATE_STEPS = 32
MC_DEBATE_HORIZON = 5.0 / 252.0


def clip_confidence(value: Any) -> float:
    """Bound confidence to ``[0, 100]``; non-finite → ``0.0``."""
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(c):
        return 0.0
    return float(np.clip(c, 0.0, 100.0))


def _empty_prediction(expert_id: str, message: str) -> dict[str, Any]:
    return {
        "expert_id": expert_id,
        "name": EXPERT_NAMES.get(expert_id, expert_id),
        "stance": STANCE_NEUTRAL,
        "confidence": 0.0,
        "justification": message,
        "signals": {},
    }


def _prediction(
    expert_id: str,
    stance: str,
    confidence: float,
    justification: str,
    *,
    signals: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    label = str(stance).strip()
    if label not in STANCES:
        label = STANCE_NEUTRAL
    return {
        "expert_id": expert_id,
        "name": EXPERT_NAMES.get(expert_id, expert_id),
        "stance": label,
        "confidence": clip_confidence(confidence),
        "justification": str(justification),
        "signals": dict(signals or {}),
    }


def _mean_chain_iv(frame: pd.DataFrame) -> float:
    from quant_engine import SIGMA_MIN, _finite_mean, _sentiment_iv_series

    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return float("nan")
    iv = _sentiment_iv_series(frame.copy()).to_numpy(dtype=np.float64)
    mask = np.isfinite(iv) & (iv > SIGMA_MIN)
    if not np.any(mask):
        return float("nan")
    return float(_finite_mean(iv[mask]))


def historical_iv_series(
    ticker: str,
    *,
    repo: Any = None,
    lookback: int = IV_RANK_LOOKBACK,
    current_iv: float | None = None,
) -> np.ndarray:
    """Mean IV per historical snapshot (oldest→newest), optionally appending current."""
    from quant_engine import DataRepository

    repository = repo if repo is not None else DataRepository()
    symbol = str(ticker or "").strip().upper()
    stamps = list(repository.get_available_snapshots(symbol) or [])
    if lookback > 0:
        stamps = stamps[-int(lookback) :]
    values: list[float] = []
    for stamp in stamps:
        payload = repository.load_from_cache(symbol, stamp)
        if not payload:
            continue
        frame = repository._records_to_frame(payload.get("data")).copy()
        iv = _mean_chain_iv(frame)
        if np.isfinite(iv):
            values.append(float(iv))
    arr = np.asarray(values, dtype=np.float64)
    if current_iv is not None and np.isfinite(float(current_iv)):
        cur = float(current_iv)
        if arr.size == 0 or not np.isclose(arr[-1], cur, rtol=1e-6, atol=1e-8):
            arr = np.concatenate([arr, np.array([cur], dtype=np.float64)])
    return arr.copy()


def compute_iv_rank_percentile(history: Any, current_iv: float) -> dict[str, float]:
    """IV Rank = (current − min) / (max − min); IV percentile = empirical CDF."""
    hist = np.asarray(history, dtype=np.float64).reshape(-1).copy()
    hist = hist[np.isfinite(hist)]
    try:
        cur = float(current_iv)
    except (TypeError, ValueError):
        cur = float("nan")
    if not np.isfinite(cur) or hist.size == 0:
        return {"iv_rank": float("nan"), "iv_percentile": float("nan"), "n": float(hist.size)}
    lo = float(np.min(hist))
    hi = float(np.max(hist))
    span = hi - lo
    if span <= 1e-12:
        rank = 50.0
    else:
        rank = 100.0 * float(np.clip((cur - lo) / span, 0.0, 1.0))
    pct = 100.0 * float(np.mean(hist <= cur))
    return {"iv_rank": float(rank), "iv_percentile": float(pct), "n": float(hist.size)}


def expert_a_momentum(
    ticker: str,
    *,
    frame: pd.DataFrame | None = None,
    spot: float | None = None,
    repo: Any = None,
) -> dict[str, Any]:
    """Expert A: GEX + price velocity + reflexivity → directional stance."""
    from quant_engine import (
        _snapshot_chain_frame,
        analyze_gex_outlook,
        calculate_market_reflexivity,
    )

    symbol = str(ticker or "").strip().upper()
    if frame is None:
        chain, snap_spot = _snapshot_chain_frame(symbol, repo=repo)
        use_spot = spot if spot is not None else snap_spot
    else:
        chain = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        use_spot = spot
    if chain is None or not isinstance(chain, pd.DataFrame) or chain.empty:
        return _empty_prediction(EXPERT_A, "Momentum: no chain snapshot available.")

    working = chain.copy()
    gex_report = analyze_gex_outlook(working)
    reflexivity = calculate_market_reflexivity(
        symbol, repo=repo, frame=working, spot=use_spot
    )
    net_gex = float(reflexivity.get("gex") or 0.0)
    momentum = float(reflexivity.get("momentum") or 0.0)
    regime = str(reflexivity.get("regime") or "")
    cascade = float(reflexivity.get("cascade_probability") or 0.0)
    reflexive = bool(reflexivity.get("reflexive_cascade"))
    flip = gex_report.get("GammaFlip")

    gex_sign = 1.0 if net_gex > 0 else (-1.0 if net_gex < 0 else 0.0)
    vel_score = float(np.tanh(momentum * 50.0))
    raw = 0.55 * gex_sign + 0.45 * vel_score
    if reflexive or cascade >= 0.7:
        raw = min(raw, -0.35) if gex_sign <= 0 else raw * 0.35

    if raw >= 0.25:
        stance = STANCE_BUY
    elif raw <= -0.25:
        stance = STANCE_SELL
    else:
        stance = STANCE_NEUTRAL

    confidence = clip_confidence(55.0 + 40.0 * abs(raw) + (10.0 if abs(net_gex) > 0 else 0.0))
    if stance == STANCE_NEUTRAL:
        confidence = clip_confidence(min(confidence, 45.0))

    try:
        flip_f = float(flip) if flip is not None else float("nan")
    except (TypeError, ValueError):
        flip_f = float("nan")
    flip_txt = f"{flip_f:.2f}" if np.isfinite(flip_f) else "n/a"
    justification = (
        f"GEX={net_gex:.4g}, velocity={momentum:+.4%}, regime={regime or 'n/a'}, "
        f"cascade_p={cascade:.2f}, gamma_flip={flip_txt}. "
        f"Score={raw:+.3f} → {stance}."
    )
    return _prediction(
        EXPERT_A,
        stance,
        confidence,
        justification,
        signals={
            "gex": net_gex,
            "momentum": momentum,
            "regime": regime,
            "cascade_probability": cascade,
            "reflexive_cascade": reflexive,
            "gamma_flip": flip_f if np.isfinite(flip_f) else None,
            "score": float(raw),
        },
    )


def expert_b_mean_reversion(
    ticker: str,
    *,
    frame: pd.DataFrame | None = None,
    repo: Any = None,
    lookback: int = IV_RANK_LOOKBACK,
) -> dict[str, Any]:
    """Expert B: IV Rank / percentile vs historical mean → fade extremes."""
    from quant_engine import _snapshot_chain_frame

    symbol = str(ticker or "").strip().upper()
    if frame is None:
        chain, _spot = _snapshot_chain_frame(symbol, repo=repo)
    else:
        chain = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if chain is None or not isinstance(chain, pd.DataFrame) or chain.empty:
        return _empty_prediction(EXPERT_B, "Mean Reversion: no chain snapshot available.")

    working = chain.copy()
    current_iv = _mean_chain_iv(working)
    history = historical_iv_series(
        symbol, repo=repo, lookback=lookback, current_iv=current_iv if np.isfinite(current_iv) else None
    )
    stats = compute_iv_rank_percentile(history, current_iv if np.isfinite(current_iv) else float("nan"))
    iv_rank = float(stats["iv_rank"])
    iv_pct = float(stats["iv_percentile"])
    n = int(stats["n"])

    if not np.isfinite(current_iv) or not np.isfinite(iv_rank) or n < 2:
        return _prediction(
            EXPERT_B,
            STANCE_NEUTRAL,
            15.0,
            "Mean Reversion: insufficient IV history for rank/percentile.",
            signals={"current_iv": current_iv, "iv_rank": iv_rank, "iv_percentile": iv_pct, "n": n},
        )

    hist_mean = float(np.mean(history))
    hist_std = float(np.std(history))
    z = (current_iv - hist_mean) / hist_std if hist_std > 1e-12 else 0.0

    if iv_rank >= 70.0 or z >= 1.0:
        stance = STANCE_SELL
        edge = max(iv_rank - 50.0, abs(z) * 20.0)
    elif iv_rank <= 30.0 or z <= -1.0:
        stance = STANCE_BUY
        edge = max(50.0 - iv_rank, abs(z) * 20.0)
    else:
        stance = STANCE_NEUTRAL
        edge = 20.0 - abs(iv_rank - 50.0) * 0.3

    confidence = clip_confidence(40.0 + float(edge) * 0.7)
    if stance == STANCE_NEUTRAL:
        confidence = clip_confidence(min(confidence, 50.0))

    justification = (
        f"IV={current_iv:.4f}, IV_Rank={iv_rank:.1f}, IV_Pct={iv_pct:.1f}, "
        f"μ={hist_mean:.4f}, σ={hist_std:.4f}, z={z:+.2f} (n={n}). "
        f"Mean-reversion stance → {stance}."
    )
    return _prediction(
        EXPERT_B,
        stance,
        confidence,
        justification,
        signals={
            "current_iv": current_iv,
            "iv_rank": iv_rank,
            "iv_percentile": iv_pct,
            "z_score": float(z),
            "hist_mean": hist_mean,
            "hist_std": hist_std,
            "n": n,
        },
    )


def expert_c_volatility_arb(
    ticker: str,
    *,
    frame: pd.DataFrame | None = None,
    spot: float | None = None,
    repo: Any = None,
    n_paths: int = MC_DEBATE_PATHS,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Expert C: Vanna/Volga + Monte Carlo path distribution → vol-arb stance."""
    from quant_engine import (
        DEFAULT_SIGMA,
        SIGMA_MIN,
        DataRepository,
        _chain_vanna_volga,
        _finite_mean,
        _snapshot_chain_frame,
        regime_chain_vanna_exposure,
    )

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
        return _empty_prediction(EXPERT_C, "Volatility Arb: no chain snapshot available.")

    working = chain.copy()
    vanna_rows, volga_rows = _chain_vanna_volga(working)
    vanna_score, _ = regime_chain_vanna_exposure(working)
    with np.errstate(invalid="ignore"):
        mean_vanna = float(_finite_mean(vanna_rows))
        mean_volga = float(_finite_mean(volga_rows))
    vv = build_vanna_volga_report(mean_vanna, mean_volga)

    current_iv = _mean_chain_iv(working)
    sigma = float(current_iv) if np.isfinite(current_iv) and current_iv > SIGMA_MIN else DEFAULT_SIGMA
    try:
        px = float(use_spot) if use_spot is not None else float("nan")
    except (TypeError, ValueError):
        px = float("nan")
    if not np.isfinite(px) or px <= 0:
        return _prediction(
            EXPERT_C,
            STANCE_NEUTRAL,
            10.0,
            "Volatility Arb: valid spot required for Monte Carlo distribution.",
            signals={"vanna": mean_vanna, "volga": mean_volga},
        )

    stop = px * 0.95
    mc = evaluate_position_montecarlo(
        px,
        stop,
        sigma=sigma,
        time_years=MC_DEBATE_HORIZON,
        direction="long",
        n_paths=max(int(n_paths), MC_MIN_PATHS),
        n_steps=MC_DEBATE_STEPS,
        seed=seed,
        vanna=mean_vanna,
        volga=mean_volga,
        delta_hedged=True,
    )
    paths = mc.get("paths")
    if paths is None:
        paths = simulate_gbm_paths(
            px,
            sigma,
            MC_DEBATE_HORIZON,
            n_paths=max(int(n_paths), MC_MIN_PATHS),
            n_steps=MC_DEBATE_STEPS,
            seed=seed,
        )
    terminal = np.asarray(paths, dtype=np.float64)[:, -1].copy()
    up_mass = float(np.mean(terminal > px))
    down_mass = float(np.mean(terminal < px))
    skew = float(up_mass - down_mass)
    pop = float(mc.get("pop") or 0.0)

    volga_signal = float(np.tanh(mean_volga * 5.0)) if np.isfinite(mean_volga) else 0.0
    vanna_signal = float(np.tanh(mean_vanna * 10.0)) if np.isfinite(mean_vanna) else 0.0
    dist_signal = float(np.clip(skew * 2.0, -1.0, 1.0))
    raw = 0.40 * volga_signal + 0.25 * vanna_signal + 0.35 * dist_signal

    if raw >= 0.20:
        stance = STANCE_BUY
    elif raw <= -0.20:
        stance = STANCE_SELL
    else:
        stance = STANCE_NEUTRAL

    confidence = clip_confidence(50.0 + 45.0 * abs(raw) + (5.0 if vv.get("high_vanna_exposure") else 0.0))
    if stance == STANCE_NEUTRAL:
        confidence = clip_confidence(min(confidence, 48.0))

    warn = vv.get("vanna_warning")
    justification = (
        f"Vanna={mean_vanna:.4g}, Volga={mean_volga:.4g}, "
        f"MC up_mass={up_mass:.3f}/down_mass={down_mass:.3f}, PoP={pop:.3f}, "
        f"n_paths={mc.get('n_paths')}. Score={raw:+.3f} → {stance}."
    )
    if warn:
        justification = f"{justification} {warn}"

    return _prediction(
        EXPERT_C,
        stance,
        confidence,
        justification,
        signals={
            "vanna": mean_vanna if np.isfinite(mean_vanna) else None,
            "volga": mean_volga if np.isfinite(mean_volga) else None,
            "vanna_score": float(vanna_score),
            "high_vanna_exposure": bool(vv.get("high_vanna_exposure")),
            "up_mass": up_mass,
            "down_mass": down_mass,
            "pop": pop,
            "n_paths": int(mc.get("n_paths") or 0),
            "score": float(raw),
        },
    )


def build_disagreement_report(predictions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Architect Disagreement Report when Buy vs Sell stances conflict."""
    preds = {k: dict(v) for k, v in predictions.items()}
    stances = {eid: str(preds.get(eid, {}).get("stance") or STANCE_NEUTRAL) for eid in (EXPERT_A, EXPERT_B, EXPERT_C)}
    directional = {eid: s for eid, s in stances.items() if s in (STANCE_BUY, STANCE_SELL)}
    unique = set(directional.values())
    disagrees = len(unique) > 1

    if not disagrees:
        return {
            "disagreement": False,
            "report": None,
            "stances": stances,
            "conflict_pairs": [],
        }

    pairs: list[tuple[str, str]] = []
    ids = list(directional.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if directional[a] != directional[b]:
                pairs.append((a, b))

    lines = [
        "Disagreement Report: Council experts conflict on direction.",
    ]
    for a, b in pairs:
        pa, pb = preds[a], preds[b]
        lines.append(
            f"- Expert {a} ({pa.get('name')}) {pa.get('stance')} "
            f"@ {clip_confidence(pa.get('confidence')):.0f} vs "
            f"Expert {b} ({pb.get('name')}) {pb.get('stance')} "
            f"@ {clip_confidence(pb.get('confidence')):.0f}."
        )
        lines.append(f"  A justification: {pa.get('justification')}")
        lines.append(f"  B justification: {pb.get('justification')}")
    lines.append(
        "Models differ because Momentum weights GEX/velocity/reflexivity, "
        "Mean Reversion weights IV Rank/percentile vs history, and "
        "Volatility Arb weights Vanna/Volga plus Monte Carlo path mass."
    )
    return {
        "disagreement": True,
        "report": "\n".join(lines),
        "stances": stances,
        "conflict_pairs": pairs,
    }


def run_council_debate(
    ticker: str,
    *,
    frame: pd.DataFrame | None = None,
    spot: float | None = None,
    repo: Any = None,
    lookback: int = IV_RANK_LOOKBACK,
    n_paths: int = MC_DEBATE_PATHS,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Run Experts A/B/C on the current snapshot; return summary + disagreement."""
    from quant_engine import DataRepository, _snapshot_chain_frame

    symbol = str(ticker or "").strip().upper()
    if frame is None:
        chain, snap_spot = _snapshot_chain_frame(symbol, repo=repo)
        use_spot = spot if spot is not None else snap_spot
    else:
        chain = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        use_spot = spot
        if use_spot is None and isinstance(chain, pd.DataFrame) and not chain.empty:
            use_spot = DataRepository()._spot_from_frame(chain)

    working = chain.copy() if isinstance(chain, pd.DataFrame) else pd.DataFrame()
    pred_a = expert_a_momentum(symbol, frame=working, spot=use_spot, repo=repo)
    pred_b = expert_b_mean_reversion(symbol, frame=working, repo=repo, lookback=lookback)
    pred_c = expert_c_volatility_arb(
        symbol, frame=working, spot=use_spot, repo=repo, n_paths=n_paths, seed=seed
    )
    predictions = {EXPERT_A: pred_a, EXPERT_B: pred_b, EXPERT_C: pred_c}
    disagreement = build_disagreement_report(predictions)
    ok = bool(isinstance(working, pd.DataFrame) and not working.empty)
    spot_out: float | None
    try:
        spot_out = float(use_spot) if use_spot is not None else None
        if spot_out is not None and not np.isfinite(spot_out):
            spot_out = None
    except (TypeError, ValueError):
        spot_out = None
    return {
        "ok": ok,
        "ticker": symbol,
        "spot": spot_out,
        "experts": predictions,
        "predictions": [predictions[EXPERT_A], predictions[EXPERT_B], predictions[EXPERT_C]],
        "disagreement": bool(disagreement["disagreement"]),
        "disagreement_report": disagreement.get("report"),
        "stances": disagreement.get("stances"),
        "conflict_pairs": disagreement.get("conflict_pairs"),
    }
