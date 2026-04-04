"""core_pipeline — Protected Public Methodology
===============================================

FFD-based regime segmentation of macroeconomic release series.

This module exposes the methodological core needed to understand
and replicate the pipeline on user-supplied arrays.  Proprietary
data, ingestion logic, and the full internal robustness layer are
omitted intentionally.

Public defaults are provided for usability.  Exact study
calibration may differ and can be stated in the paper.

Reference: López de Prado (2018), Advances in Financial Machine
Learning, Wiley.

License: MIT
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional

from statsmodels.tsa.stattools import adfuller

__all__ = [
    "transform_series",
    "ffd_fixed_width",
    "estimate_d_stat95",
    "validate_regime",
    "build_failure_register",
    "build_reliability_register",
]

# ── Defaults (public usability; study calibration may differ) ────────────
D_GRID       = np.round(np.arange(0.0, 1.05, 0.05), 2)
THR_FFD      = 1e-2
M_BEFORE_MIN = 6
M_TRAIN_MIN  = 8
M_TEST_MIN   = 6
FRAC_TRAIN   = 0.80
ETA          = 0.05
TAU          = 0.05
ADF_ALPHA    = 0.05

# ── Series classification (populate before use) ─────────────────────────
LOG_SERIES:   frozenset = frozenset()
LEVEL_SERIES: frozenset = frozenset()


# ─────────────────────────────────────────────────────────────────────────
# 1. Transformation Layer
# ─────────────────────────────────────────────────────────────────────────

def transform_series(name: str, values: np.ndarray) -> np.ndarray:
    """Apply pre-FFD transformation (log for index-like, identity for rate-like)."""
    x = np.asarray(values, dtype=float)
    if name in LOG_SERIES:
        if np.any(x <= 0):
            raise ValueError(
                f"Log transform requires positive values; series='{name}'."
            )
        return np.log(x)
    if name in LEVEL_SERIES:
        return x
    raise ValueError(
        f"Series '{name}' not registered in LOG_SERIES or LEVEL_SERIES."
    )


# ─────────────────────────────────────────────────────────────────────────
# 2. FFD Layer — Fixed-Width Fractional Differencing
# ─────────────────────────────────────────────────────────────────────────

def _ffd_width(d: float, thr: float = THR_FFD, max_size: int = 10_000) -> int:
    """Effective filter width for FFD(d) given truncation threshold."""
    w = [1.0]
    for k in range(1, max_size):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thr:
            break
        w.append(w_k)
    return len(w)


def _ffd_weights(d: float, size: int, thr: float = THR_FFD) -> np.ndarray:
    """Recursive binomial weights, reversed for causal convolution."""
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thr:
            break
        w.append(w_k)
    return np.array(w[::-1])


def ffd_fixed_width(
    series: np.ndarray,
    d: float,
    thr: float = THR_FFD,
) -> np.ndarray:
    """Apply fixed-width fractional differencing FFD(d) to a 1-D array."""
    x     = np.asarray(series, dtype=float)
    width = _ffd_width(d, thr)
    if width > len(x):
        return np.array([])
    weights = _ffd_weights(d, width, thr)
    return np.array([
        np.dot(weights, x[t - width + 1 : t + 1])
        for t in range(width - 1, len(x))
    ])


# ─────────────────────────────────────────────────────────────────────────
# 3. Admissibility Layer — ADF-Based Minimal d*
# ─────────────────────────────────────────────────────────────────────────

def estimate_d_stat95(
    series: np.ndarray,
    d_grid: Optional[np.ndarray] = None,
    thr_ffd: float = THR_FFD,
    min_fd_len: int = M_BEFORE_MIN,
    alpha: float = ADF_ALPHA,
) -> float:
    """Find minimal admissible d*: smallest d where ADF rejects at level alpha."""
    x = np.asarray(series, dtype=float)
    if d_grid is None:
        d_grid = D_GRID
    for d in d_grid:
        fd = ffd_fixed_width(x, float(d), thr_ffd)
        if len(fd) < max(min_fd_len, 3):
            continue
        try:
            pval = adfuller(fd, maxlag=1, regression="c", autolag=None)[1]
            if np.isfinite(pval) and pval < alpha:
                return float(d)
        except Exception:
            continue
    return np.nan


# ─────────────────────────────────────────────────────────────────────────
# 4. Regime Validation Engine
# ─────────────────────────────────────────────────────────────────────────

def validate_regime(
    series_name: str,
    before_values: np.ndarray,
    during_values: np.ndarray,
    d_grid: Optional[np.ndarray] = None,
    frac_train: float = FRAC_TRAIN,
    eta: float = ETA,
    tau: float = TAU,
    thr_ffd: float = THR_FFD,
    pair_id: str = "",
) -> Dict:
    """Validate one (series, regime-window) pair."""
    if d_grid is None:
        d_grid = D_GRID

    n_before = len(before_values)
    n_during = len(during_values)

    record: Dict = {
        "pair_id":         pair_id,
        "series":          series_name,
        "n_before":        n_before,
        "n_during":        n_during,
        "n_train":         np.nan,
        "n_test":          np.nan,
        "d_global":        np.nan,
        "d_local":         np.nan,
        "delta_d":         np.nan,
        "mse_improvement": np.nan,
        "segmented":       False,
        "status":          None,
    }

    # -- Pre-gates ---------------------------------------------------------
    if n_before < M_BEFORE_MIN:
        record["status"] = "FAIL_BASELINE_TOO_SHORT"
        return record
    if n_during < (M_TRAIN_MIN + M_TEST_MIN):
        record["status"] = "FAIL_DURING_TOO_SHORT"
        return record

    # -- Transform ---------------------------------------------------------
    try:
        before_vals = transform_series(series_name, before_values)
        during_vals = transform_series(series_name, during_values)
    except ValueError:
        record["status"] = "FAIL_TRANSFORM"
        return record

    # -- Chronological train/test split ------------------------------------
    n = len(during_vals)
    if n < M_TRAIN_MIN + M_TEST_MIN:
        record["status"] = "FAIL_TEST_TOO_SHORT"
        return record
    k_train = int(np.floor(n * frac_train))
    k_train = max(M_TRAIN_MIN, min(k_train, n - M_TEST_MIN))
    k_test  = n - k_train
    record["n_train"] = k_train
    record["n_test"]  = k_test

    # -- Estimate d_global and d_local -------------------------------------
    d_global = estimate_d_stat95(
        before_vals, d_grid=d_grid, thr_ffd=thr_ffd,
        min_fd_len=M_BEFORE_MIN,
    )
    d_local = estimate_d_stat95(
        during_vals[:k_train], d_grid=d_grid, thr_ffd=thr_ffd,
        min_fd_len=M_TRAIN_MIN,
    )

    if not np.isfinite(d_global):
        record["status"] = "FAIL_ADF_INFEASIBLE"
        return record
    if not np.isfinite(d_local):
        record["status"] = "FAIL_ADF_INFEASIBLE"
        return record

    record.update({
        "d_global": float(d_global),
        "d_local":  float(d_local),
        "delta_d":  float(d_local - d_global),
    })

    # -- OOS scoring -------------------------------------------------------
    mse_g = _oos_mse(during_vals, d_global, k_train, thr_ffd)
    mse_l = _oos_mse(during_vals, d_local,  k_train, thr_ffd)

    if np.isfinite(mse_g) and np.isfinite(mse_l) and mse_g > 0:
        record["mse_improvement"] = float(1.0 - mse_l / mse_g)

    if not (np.isfinite(mse_g) and np.isfinite(mse_l)):
        record["status"] = "FAIL_OOS_UNDEFINED"
        return record

    # -- Segmentation gate -------------------------------------------------
    mse_ok   = mse_l <= (1 - eta) * mse_g
    delta_ok = abs(record["delta_d"]) >= tau

    if not mse_ok:
        record["status"] = "GATE_IMPROVEMENT_BELOW_THRESHOLD"
        return record
    if not delta_ok:
        record["status"] = "GATE_DELTA_D_BELOW_THRESHOLD"
        return record

    record["segmented"] = True
    record["status"]    = "SEGMENTED"
    return record


def _oos_mse(
    during_vals: np.ndarray,
    d: float,
    k_train: int,
    thr_ffd: float,
) -> float:
    """Internal OOS MSE computation for validate_regime."""
    x     = np.asarray(during_vals, dtype=float)
    width = _ffd_width(float(d), thr_ffd)
    if len(x) < width:
        return np.nan

    fd  = ffd_fixed_width(x, float(d), thr_ffd)
    fd0 = width - 1
    if k_train <= fd0:
        return np.nan

    fd_split = k_train - fd0
    if fd_split < M_TRAIN_MIN or (len(fd) - fd_split) < M_TEST_MIN:
        return np.nan

    train_fd = fd[:fd_split]
    test_fd  = fd[fd_split:]
    if len(train_fd) < 5 or len(test_fd) < 1:
        return np.nan

    hist   = list(train_fd)
    sq_err = []
    for y_true in test_fd:
        if len(hist) < 5 or np.std(hist[:-1]) <= 1e-12:
            y_pred = hist[-1]
        else:
            xa = np.asarray(hist[:-1])
            ya = np.asarray(hist[1:])
            X  = np.column_stack([np.ones_like(xa), xa])
            b  = np.linalg.lstsq(X, ya, rcond=None)[0]
            y_pred = float(b[0] + b[1] * hist[-1])
        sq_err.append((float(y_true) - y_pred) ** 2)
        hist.append(float(y_true))

    return float(np.mean(sq_err))


# ─────────────────────────────────────────────────────────────────────────
# 5. Failure Register
# ─────────────────────────────────────────────────────────────────────────

_FAILURE_MAP = {
    "BASELINE_TOO_SHORT":            "Baseline too short",
    "DURING_TOO_SHORT":              "Regime sample too short",
    "TEST_TOO_SHORT":                "Regime sample too short",
    "TRANSFORM":                     "Transform error",
    "ADF_INFEASIBLE":                "ADF infeasible",
    "OOS_UNDEFINED":                 "OOS score undefined",
    "IMPROVEMENT_BELOW_THRESHOLD":   "Predictive gain below threshold",
    "DELTA_D_BELOW_THRESHOLD":       "Memory shift below threshold",
}


def _classify_failure(status: str) -> str:
    if not isinstance(status, str):
        return "Unknown"
    s = status.upper()
    for key, label in _FAILURE_MAP.items():
        if key in s:
            return label
    return f"Other: {status[:60]}"


def build_failure_register(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate failure codes across non-segmented pairs."""
    failures = results.loc[~results["segmented"]].copy()
    if failures.empty:
        return pd.DataFrame(columns=["Reason", "N", "Share_pct"])
    failures["reason"] = failures["status"].apply(_classify_failure)
    summary = (
        failures.groupby("reason", as_index=False)
        .agg(N=("series", "count"))
        .sort_values("N", ascending=False)
        .reset_index(drop=True)
    )
    summary["Share_pct"] = 100.0 * summary["N"] / len(failures)
    return summary.rename(columns={"reason": "Reason"})


# ─────────────────────────────────────────────────────────────────────────
# 6. Reliability Register
# ─────────────────────────────────────────────────────────────────────────

def build_reliability_register(results: pd.DataFrame) -> pd.DataFrame:
    """Compact reliability warnings for estimable pairs."""
    rel = results.loc[results["d_global"].notna()].copy()

    rel["flag_boundary"] = (
        rel["d_global"].isin([0.0, 1.0])
        | (rel["d_local"].notna() & rel["d_local"].isin([0.0, 1.0]))
    )
    rel["flag_sample_bind"] = (
        (rel["n_train"].notna()
         & (rel["n_train"].astype(float) <= M_TRAIN_MIN))
        | (rel["n_test"].notna()
           & (rel["n_test"].astype(float) <= M_TEST_MIN))
    )
    rel["flag_stability"] = (
        rel["delta_d"].notna() & (rel["delta_d"].abs() > 0.8)
    )

    flag_cols = [
        "flag_boundary",
        "flag_sample_bind",
        "flag_stability",
    ]
    _labels = {
        "flag_boundary":    "boundary",
        "flag_sample_bind": "sample_bind",
        "flag_stability":   "stability",
    }
    rel["n_flags"]  = rel[flag_cols].sum(axis=1)
    rel["any_flag"] = rel["n_flags"] > 0

    flagged = rel.loc[rel["any_flag"]].copy()
    if not flagged.empty:
        flagged["Flags"] = flagged[flag_cols].apply(
            lambda r: ", ".join(
                _labels[c] for c in flag_cols if r[c]
            ),
            axis=1,
        )
    return flagged
