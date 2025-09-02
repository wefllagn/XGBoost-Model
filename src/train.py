# src/train.py
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DBConfig, ModelConfig
from .db import ensure_pred_table, get_engine, read_water_table, upsert_predictions
from .features import auto_map_columns, build_supervised_frame
from .model import XGBWaterBalanceModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("train")


def _unique_sorted(values: Iterable[str]) -> List[str]:
    return sorted(set(values))


def _dbg_stats(tag: str, d: pd.DataFrame) -> None:
    if "water_balance" not in d.columns:
        logger.info("[DEBUG] %s: no 'water_balance' column to summarize.", tag)
        return
    wb = d["water_balance"]
    print(
        f"\n[DEBUG] {tag}: rows={len(d)} "
        f"min={wb.min()} max={wb.max()} mean={wb.mean()} "
        f"nonzero={int((wb != 0).sum())}"
    )


def run(
    municipality: Optional[str],
    horizon: int,
    no_write: bool,
    exog_cols_cli: Optional[List[str]],
    targets_cli: Optional[List[str]],
) -> None:
    """
    Train & forecast one or multiple target variables.
    - If --targets is omitted, defaults to the historical behavior (changeinstorage(mm)).
    - Per-target, per-municipality metrics are printed.
    - If multiple targets are requested and --no-write is False, DB upsert is skipped
      (existing schema doesn’t track a 'target' column).
    """
    db_cfg = DBConfig()

    # Resolve targets
    if targets_cli and len([t for t in targets_cli if t.strip()]) > 0:
        targets = [t.strip() for t in targets_cli if t.strip()]
    else:
        # Historical default
        targets = ["changeinstorage(mm)"]

    # Resolve exogenous columns from CLI or ENV (applied per-target, with the target removed)
    import os
    env_cols = os.getenv("EXOG_COLS", "")
    exog_from_env = [c.strip() for c in env_cols.split(",") if c.strip()] if env_cols.strip() else []
    exog_from_cli = [c.strip() for c in (exog_cols_cli or []) if c.strip()]
    base_exog = exog_from_cli or exog_from_env  # may be empty

    engine = get_engine(db_cfg)
    df_raw = read_water_table(engine, db_cfg)

    # Determine municipality list (Benguet-only by default)
    BENGUET_ID = 1401100000
    # We'll compute muni_list from a minimally mapped frame to access columns reliably.
    # Use a temporary config with default target so auto_map can normalize basic columns.
    temp_cfg = ModelConfig()
    df_temp = auto_map_columns(df_raw, temp_cfg)

    if municipality and municipality.upper() != "ALL":
        muni_list = [municipality]
    else:
        if "provincialID" not in df_temp.columns:
            raise ValueError("Column 'provincialID' not found. Can't filter Benguet.")
        muni_list = (
            df_temp.loc[df_temp["provincialID"] == BENGUET_ID, "municipality"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        muni_list = sorted(muni_list)

    if not muni_list:
        sample = (
            df_temp[
                ["municipality"]
                + ([c for c in ["province", "provincialID"] if c in df_temp.columns])
            ]
            .drop_duplicates()
            .sort_values("municipality")
            .head(40)
        )
        logger.error(
            "No municipalities matched provincialID=%s. "
            "Here are some rows we see:\n%s",
            BENGUET_ID,
            sample.to_string(index=False),
        )
        return

    model_version = f"xgb_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    all_console_rows: List[Tuple[str, str, int, int, float, str]] = []  # (target, muni, y, m, yhat, ver)
    all_metrics: List[Tuple[str, str, float, float]] = []  # (target, muni, rmse, mae)

    # ===== Loop over TARGETS =====
    for target in targets:
        # Exogenous for this target (avoid leaking the target back in as a feature)
        exog_effective = [c for c in base_exog if c != target] if base_exog else None

        model_cfg = ModelConfig(
            # Important: pass target to features so it’s mapped into canonical 'water_balance'
            target_col=target,
            exog_cols=exog_effective,
        )

        # Map columns & build supervised frame for THIS target
        df_mapped = auto_map_columns(df_raw, model_cfg)
        _dbg_stats("overall (after auto_map_columns)", df_mapped)

        df_super = build_supervised_frame(df_mapped, model_cfg)
        _dbg_stats("after build_supervised_frame", df_super)
        print(
            "[DEBUG] Sample supervised rows:\n",
            df_super[["municipality", "year", "month", "water_balance"]]
            .sort_values(["municipality", "year", "month"])
            .head(12)
            .to_string(index=False),
        )

        # Train per municipality
        trainer = XGBWaterBalanceModel(model_cfg)

        for muni in muni_list:
            try:
                result = trainer.train_one(df_super, muni)
                logger.info(
                    "Trained %s | target=%s | RMSE=%.4f | MAE=%.4f | features=%d",
                    muni,
                    target,
                    result.rmse,
                    result.mae,
                    len(result.features_used),
                )
                all_metrics.append((target, muni, result.rmse, result.mae))

                # Forecast horizon
                fc = trainer.recursive_forecast(df_super, result.model, muni, horizon)
                # fc: List[Tuple[muni, year, month, yhat]]
                for (m, y, mo, yhat) in fc:
                    all_console_rows.append((target, m, y, mo, yhat, model_version))

            except Exception as e:
                logger.exception("Failed training for %s (target=%s): %s", muni, target, e)

    # ===== Output (console / DB) =====
    if all_console_rows:
        # Print first N rows per target for readability
        from itertools import islice
        print("\n=== Forecasts (console only) ===")
        # group by target
        targets_seen = _unique_sorted([t for (t, *_rest) in all_console_rows])
        for t in targets_seen:
            print(f"\n--- Target: {t} ---")
            rows_t = [r for r in all_console_rows if r[0] == t]
            for row in islice(rows_t, 12):
                _, muni, yy, mm, yhat, ver = row
                print(f"{muni:15s}  {yy}-{mm:02d}  y_pred={yhat:.4f}  model={ver}")

    # DB write behavior:
    # Existing upsert schema doesn’t track 'target'. To avoid schema breakage,
    # we’ll only write when there is a SINGLE target. Otherwise, skip with a warning.
    uniq_targets = _unique_sorted(targets)
    if not no_write:
        if len(uniq_targets) == 1:
            # Use existing upsert schema (municipality, year, month, y_pred, model_version)
            rows_single = [
                (muni, y, mo, yhat, model_version)
                for (_t, muni, y, mo, yhat, _ver) in all_console_rows
            ]
            written = upsert_predictions(engine, db_cfg, rows_single)
            logger.info("Upserted %d prediction rows into %s", written, db_cfg.pred_table)
        else:
            logger.warning(
                "Multiple targets requested (%s) but DB schema doesn’t include a 'target' column. "
                "Skipping DB write. Use --no-write (already implied here) or change schema.",
                ", ".join(uniq_targets),
            )

    # ===== Metrics summary =====
    if all_metrics:
        df_met = pd.DataFrame(all_metrics, columns=["target", "municipality", "rmse", "mae"])

        # Per-target summary (mean over municipalities)
        per_target = (
            df_met.groupby("target", as_index=False)[["rmse", "mae"]]
            .mean()
            .sort_values("rmse")
        )

        # Per-target, per-municipality (sorted within each target)
        print("\n=== Metrics per target (avg across municipalities) ===")
        print(per_target.to_string(index=False))

        print("\n=== Metrics per municipality and target ===")
        for t in _unique_sorted(df_met["target"].tolist()):
            sub = df_met[df_met["target"] == t].sort_values("rmse")
            logger.info("\nTarget: %s\n%s", t, sub.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train XGBoost models and forecast future months for one or more targets."
    )
    p.add_argument(
        "--municipality",
        type=str,
        default="ALL",
        help='Municipality name or "ALL" (ignored if you’re running Benguet-wide).',
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=60,
        help="Forecast horizon in months (default: 60).",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write predictions to DB.",
    )
    p.add_argument(
        "--exog-cols",
        type=str,
        default="",
        help="Comma-separated exogenous columns (overrides .env).",
    )
    p.add_argument(
        "--targets",
        type=str,
        default="",
        help=(
            "Comma-separated target columns to forecast, e.g. "
            '"precipitation(mm),evapotranspiration(mm),runoff(mm),soilmoisture(mm),changeinstorage(mm)". '
            "If omitted, defaults to 'changeinstorage(mm)'."
        ),
    )
    args = p.parse_args()

    exog_cols_cli = [c.strip() for c in args.exog_cols.split(",")] if args.exog_cols.strip() else None
    targets_cli = [c.strip() for c in args.targets.split(",")] if args.targets.strip() else None

    return argparse.Namespace(
        municipality=args.municipality,
        horizon=args.horizon,
        no_write=args.no_write,
        exog_cols=exog_cols_cli,
        targets=targets_cli,
    )


if __name__ == "__main__":
    ns = parse_args()
    run(ns.municipality, ns.horizon, ns.no_write, ns.exog_cols, ns.targets)
