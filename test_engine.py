"""Unit tests for Black–Scholes and Heston Greeks."""

from __future__ import annotations

import math

import pytest

from quant_engine import (
    DEFAULT_HESTON_KAPPA,
    DEFAULT_HESTON_RHO,
    DEFAULT_HESTON_SIGMA,
    DEFAULT_HESTON_THETA,
    DEFAULT_HESTON_V0,
    calculate_greeks,
    calculate_heston_greeks,
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
