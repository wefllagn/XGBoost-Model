from __future__ import annotations

import logging
import re
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd

from .config import ModelConfig

logger = logging.getLogger(__name__)


def _normalize(name: str) -> str:
    """Lowercase, remove non-alphanumerics for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def auto_map_columns(df: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """
    Map input columns to the required schema and (if needed) derive the target.
    Required outputs: municipality, year, month, water_balance (+ optional exog).
    Priority for target:
      1) If 'changeinstorage(mm)' exists -> use that as 'water_balance'
      2) Else derive water_balance = precip - et - runoff - changeinstorage  (if all exist)
    """
    df = df.copy()

    # ---- Map municipality column ----
    muni_candidates = [
        cfg.municipality_col, "municipality", "muni", "muni_name", "city", "town",
    ]
    norm_cols = {_normalize(c): c for c in df.columns}

    municipality_col: Optional[str] = None
    for opt in muni_candidates:
        key = _normalize(opt)
        if key in norm_cols:
            municipality_col = norm_cols[key]
            break
    if municipality_col is None:
        raise ValueError(
            f"Could not find a municipality column among: {muni_candidates}. "
            f"Found columns: {list(df.columns)}"
        )
    df = df.rename(columns={municipality_col: "municipality"})

    # ---- Year/Month from existing 'date' or present columns ----
    if ("year" not in df.columns or "month" not in df.columns) and ("date" in df.columns):
        d = pd.to_datetime(df["date"])
        df["year"] = d.dt.year.astype(int)
        df["month"] = d.dt.month.astype(int)
    if "year" not in df.columns or "month" not in df.columns:
        raise ValueError("Need 'year' and 'month' columns (or a parsable 'date').")

    # ---- Identify hydrologic components ----
    def find_col(options: List[str]) -> Optional[str]:
        for opt in options:
            k = _normalize(opt)
            if k in norm_cols:
                return norm_cols[k]
        # try direct match on already-renamed frame
        for c in df.columns:
            if _normalize(c) in [_normalize(o) for o in options]:
                return c
        return None

    col_p = find_col(["precipitation(mm)", "precipitation", "ppt", "p"])
    col_et = find_col(["evapotranspiration(mm)", "evapotranspiration", "et"])
    col_q = find_col(["runoff(mm)", "runoff", "discharge", "q"])
    col_ds = find_col(["changeinstorage(mm)", "changeinstorage", "storage_change", "ds", "delta_s"])

    # ---- Choose/derive target ----
    if "water_balance" not in df.columns:
        if col_ds is not None:
            # If change in storage is provided, treat it as the target explicitly
            df["water_balance"] = pd.to_numeric(df[col_ds], errors="coerce")
            target_source = col_ds
            logger.info("Using target=changeinstorage(mm) from column '%s'.", target_source)
        elif all(c is not None for c in [col_p, col_et, col_q, col_ds]):
            df["water_balance"] = (
                pd.to_numeric(df[col_p], errors="coerce")
                - pd.to_numeric(df[col_et], errors="coerce")
                - pd.to_numeric(df[col_q], errors="coerce")
                - pd.to_numeric(df[col_ds], errors="coerce")
            )
            logger.info("Derived 'water_balance' = P - ET - runoff - ΔS.")
        else:
            missing = []
            if col_p is None: missing.append("precipitation")
            if col_et is None: missing.append("evapotranspiration")
            if col_q is None: missing.append("runoff")
            if col_ds is None: missing.append("change in storage")
            raise ValueError(
                "Cannot derive 'water_balance'. Missing components: "
                + ", ".join(missing)
            )

    # ---- Attach exogenous columns that actually exist ----
    present_exog: List[str] = []
    for ex in (cfg.exog_cols or []):
        k = _normalize(ex)
        if k in norm_cols:
            original = norm_cols[k]
            if ex not in df.columns:
                df[ex] = df[original]
            present_exog.append(ex)
        else:
            # soft exact match on current df columns
            for c in df.columns:
                if _normalize(c) == k:
                    if ex not in df.columns:
                        df[ex] = df[c]
                    present_exog.append(ex)
                    break

    if present_exog:
        logger.info("Using exogenous columns (filtered): %s", present_exog)
    else:
        logger.info("No valid exogenous columns found/selected.")

    # ---- Final required columns check ----
    required = ["municipality", "year", "month", "water_balance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after mapping: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    if col_ds is not None and "water_balance" in df.columns and df["water_balance"].equals(pd.to_numeric(df[col_ds], errors="coerce")):
        logger.info("Column mapping successful. target='changeinstorage(mm)'. Using required columns and %d exogenous.",
                    len(present_exog))
    else:
        logger.info("Column mapping successful. Using required columns and %d exogenous.",
                    len(present_exog))

    return df


def _expanding_zscore(s: pd.Series) -> pd.Series:
    """
    Leakage-safe z-score: (x - mean_{<=t-1}) / std_{<=t-1}
    If std is 0 or NaN, returns 0 for that timestamp.
    """
    mean = s.expanding(min_periods=2).mean().shift(1)
    std = s.expanding(min_periods=2).std().shift(1)
    z = (s - mean) / (std.replace(0.0, np.nan))
    return z.fillna(0.0)


def build_supervised_frame(
    df: pd.DataFrame,
    cfg: ModelConfig,
    lags: Iterable[int] | None = None,
    rolls: Iterable[int] | None = None,
) -> pd.DataFrame:
    """
    Build supervised features.
    Input df must have: municipality, year, month, water_balance, [exog], [date?]
    Adds:
      - Seasonal encodings (sin/cos of month)
      - Target lags: ensure {1,2,3,6,12,24}
      - Rolling means/stds on target for windows {3,6,12,24} (shifted one step)
      - rollsum_12 for target (shifted)
      - For each exog: lag1..3 and leakage-safe expanding z-score
    """
    df = df.copy()

    # Proper date for ordering
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df.sort_values(["municipality", "date"], inplace=True)

    # Final lag/roll sets (union with cfg defaults)
    base_lags = list(cfg.default_lags) if cfg.default_lags else [1, 2, 3]
    add_lags = [6, 12, 24]
    lag_set = sorted(set((lags or base_lags) + add_lags))

    base_rolls = list(cfg.default_rolls) if cfg.default_rolls else [3, 6]
    add_rolls = [12, 24]
    roll_set = sorted(set((rolls or base_rolls) + add_rolls))

    exog_cols = [c for c in (cfg.exog_cols or []) if c in df.columns]

    parts: List[pd.DataFrame] = []
    for muni, g in df.groupby("municipality", sort=False):
        g = g.copy()

        # seasonal features
        g["m"] = g["date"].dt.month
        g["sin_m"] = np.sin(2 * np.pi * g["m"] / 12.0)
        g["cos_m"] = np.cos(2 * np.pi * g["m"] / 12.0)

        # target lags
        for L in lag_set:
            g[f"lag_{L}"] = g["water_balance"].shift(L)

        # rolling mean/std (shifted to avoid leakage)
        for W in roll_set:
            g[f"rollmean_{W}"] = g["water_balance"].rolling(W, min_periods=1).mean().shift(1)
            g[f"rollstd_{W}"] = g["water_balance"].rolling(W, min_periods=1).std().shift(1)

        # rollsum_12 (shifted)
        g["rollsum_12"] = g["water_balance"].rolling(12, min_periods=1).sum().shift(1)

        # exogenous features: z-score (expanding, leakage-safe) and short lags
        for ex in exog_cols:
            # ensure numeric
            g[ex] = pd.to_numeric(g[ex], errors="coerce")

            # expanding z-score (leakage-safe)
            g[f"z_{ex}"] = _expanding_zscore(g[ex])

            # short lags 1..3
            for L in (1, 2, 3):
                g[f"{ex}_lag{L}"] = g[ex].shift(L)

            # optional rolling mean over a year for exog (shifted)
            g[f"{ex}_rollmean12"] = g[ex].rolling(12, min_periods=1).mean().shift(1)

        parts.append(g)

    out = pd.concat(parts, axis=0, ignore_index=True)

    # Drop rows that have NA due to lagging
    lag_cols = [c for c in out.columns if c.startswith("lag_")]
    if lag_cols:
        out = out.dropna(subset=lag_cols, how="any").reset_index(drop=True)

    return out


def make_future_calendar(last_year: int, last_month: int, horizon: int) -> List[Tuple[int, int]]:
    """Generate (year, month) pairs for 'horizon' months after the last known (year, month)."""
    y, m = int(last_year), int(last_month)
    fut: List[Tuple[int, int]] = []
    for _ in range(horizon):
        m += 1
        if m > 12:
            m = 1
            y += 1
        fut.append((y, m))
    return fut
