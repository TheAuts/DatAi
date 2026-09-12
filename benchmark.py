"""Micro-benchmark for generate_greek_curve."""

from __future__ import annotations

import time

import numpy as np

from quant_engine import generate_greek_curve


def run_benchmark() -> None:
    """Time vectorized Greek curves at several input sizes and print throughput."""
    for n in (1_000, 10_000, 100_000, 1_000_000):
        prices = np.linspace(50, 150, n)
        generate_greek_curve("SPY", 100.0, 0.25, prices)
        started = time.perf_counter()
        generate_greek_curve("SPY", 100.0, 0.25, prices)
        elapsed = time.perf_counter() - started
        print(f"n={n} time={elapsed:.6f}s rate={n / elapsed:,.0f}/s")


if __name__ == "__main__":
    run_benchmark()
