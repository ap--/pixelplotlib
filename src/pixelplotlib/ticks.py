"""Tick location and label formatting."""

from __future__ import annotations

import math

import numpy as np


def nice_ticks(lo: float, hi: float, max_ticks: int = 6) -> tuple[np.ndarray, float]:
    """Return "nice" tick values (multiples of 1, 2, 2.5 or 5 x 10^k) within ``[lo, hi]``."""
    if hi < lo:
        lo, hi = hi, lo
    span = hi - lo
    if not np.isfinite(span):
        return np.array([]), 1.0
    if span == 0:
        span = abs(lo) or 1.0
    max_ticks = max(int(max_ticks), 2)
    raw = span / (max_ticks - 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    step = 10 * mag
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if span / (m * mag) <= max_ticks - 1 + 1e-9:
            step = m * mag
            break
    eps = step * 1e-9
    start = math.ceil((lo - eps) / step) * step
    ticks = np.arange(start, hi + eps, step)
    ticks = np.where(np.abs(ticks) < eps, 0.0, ticks)
    return ticks, step


def log_ticks(lo: float, hi: float, max_ticks: int = 6) -> np.ndarray:
    """Powers of ten within ``[lo, hi]`` (both must be positive)."""
    if hi < lo:
        lo, hi = hi, lo
    e0 = math.ceil(math.log10(lo) - 1e-9)
    e1 = math.floor(math.log10(hi) + 1e-9)
    if e1 < e0:
        ticks, _ = nice_ticks(lo, hi, max_ticks)
        return ticks[ticks > 0]
    exps = np.arange(e0, e1 + 1)
    stride = max(1, math.ceil(len(exps) / max(max_ticks, 2)))
    exps = exps[exps % stride == 0] if stride > 1 else exps
    return 10.0 ** exps


def _sci(v: float, digits: int) -> str:
    mant, exp = f"{v:.{digits}e}".split("e")
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
    return f"{mant}E{int(exp)}"


def format_ticks(values, log: bool = False) -> list[str]:
    """Format tick values compactly with a consistent number of decimals."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return []
    if log:
        out = []
        for v in values:
            e = math.log10(v) if v > 0 else 0
            if abs(e - round(e)) < 1e-9 and (e < -3 or e > 4):
                out.append(f"1E{int(round(e))}")
            else:
                out.append(format_ticks([v])[0])
        return out
    big = np.nanmax(np.abs(values))
    nonzero = np.abs(values[values != 0])
    small = nonzero.min() if nonzero.size else 0.0
    if big >= 1e5 or (0 < small < 1e-3):
        for digits in range(0, 6):
            labels = [_sci(v, digits) for v in values]
            if np.allclose([float(lb.replace("E", "e")) for lb in labels], values,
                           rtol=1e-6, atol=0):
                break
        return [("0" if v == 0 else lb) for v, lb in zip(values, labels)]
    for decimals in range(0, 7):
        scaled = values * 10**decimals
        if np.allclose(scaled, np.round(scaled), atol=1e-6 * max(1.0, big * 10**decimals)):
            break
    labels = [f"{v:.{decimals}f}" for v in values]
    return ["0" if float(lb) == 0 else lb for lb in labels]
