from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DBConfig, ModelConfig
from .db import get_engine, read_water_table, upsert_predictions  # ensure_pred_table not needed when --no-write
from .features import auto_map_columns, build_supervised_frame
from .model import XGBWaterBalanceModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("train")


def _unique_sorted(values: Iterable[str]) -> List[str]:
    return sorted(set(values))


def run(municipality: Optional[str], horizon: int, no_write: bool, exog_cols_cli: Optional[List[str]]):
    # ---- config ----
    db_cfg = DBConfig()
    model_cfg = ModelConfig(
        exog_cols=[c.strip() for c in (exog_cols_cli or []) if c.strip()] or None
    )

    # If not passed via CLI, pull from EXOG_COLS env
    import os
    if not exog_cols_cli:
        env_cols = os.getenv("EXOG_COLS", "")
        if env_cols.strip():
            model_cfg.exog_cols = [c.strip() for c in env_cols.split(",")]

    # ---- load ----
    engine = get_engine(db_cfg)
    df_raw = read_water_table(engine, db_cfg)
    df = auto_map_columns(df_raw, model_cfg)

    # ---- debug: target variance ----
    def _dbg_stats(tag: str, d: pd.DataFrame):
        print(
            f"\n[DEBUG] {tag}: rows={len(d)} "
            f"min={d['water_balance'].min()} "
            f"max={d['water_balance'].max()} "
            f"mean={d['water_balance'].mean()} "
            f"nonzero={(d['water_balance']!=0).sum()}"
        )

    _dbg_stats("overall (after auto_map_columns)", df)

    # ---- choose municipalities (Benguet only when ALL) ----
    BENGUET_ID = 1401100000
    if municipality and municipality.upper() != "ALL":
        # Train a single municipality 
        muni_list = [municipality]
    else:
        if "provincialID" not in df.columns:
            raise ValueError("Column 'provincialID' not found. Can't filter Benguet.")
        muni_list = (
            df.loc[df["provincialID"] == BENGUET_ID, "municipality"]
              .dropna().astype(str).unique().tolist()
        )
        muni_list = sorted(muni_list)

    if not muni_list:
        sample_cols = ["municipality"] + [c for c in ["province", "provincialID"] if c in df.columns]
        sample = df[sample_cols].drop_duplicates().sort_values("municipality").head(40)
        logger.error(
            "No municipalities matched provincialID=%s. Here are some rows we see:\n%s",
            BENGUET_ID, sample.to_string(index=False)
        )
        return

    # ---- features ----
    df_super = build_supervised_frame(df, model_cfg)

    _dbg_stats("after build_supervised_frame", df_super)
    print(
        "[DEBUG] Sample supervised rows:\n",
        df_super[["municipality", "year", "month", "water_balance"]]
        .sort_values(["municipality", "year", "month"])
        .head(12).to_string(index=False)
    )

    # ---- train/forecast ----
    trainer = XGBWaterBalanceModel(model_cfg)
    model_version = f"xgb_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    all_rows: List[Tuple[str, int, int, float, str]] = []
    metrics: List[Tuple[str, float, float]] = []

    for muni in muni_list:
        try:
            result = trainer.train_one(df_super, muni)
            logger.info(
                "Trained %s | RMSE=%.4f | MAE=%.4f | features=%d",
                muni, result.rmse, result.mae, len(result.features_used)
            )
            metrics.append((muni, result.rmse, result.mae))

            # Forecast horizon
            fc = trainer.recursive_forecast(df_super, result.model, muni, horizon)
            # (municipality, year, month, yhat, version)
            all_rows.extend([(m, y, mo, yhat, model_version) for (m, y, mo, yhat) in fc])
        except Exception as e:
            logger.exception("Failed training for %s: %s", muni, e)

    # ---- output ----
    if no_write:
        from itertools import islice
        print("\n=== Forecasts (console only) ===")
        for row in islice(all_rows, 20):
            muni, yy, mm, yhat, ver = row
            print(f"{muni:15s}  {yy}-{mm:02d}  y_pred={yhat:.4f}  model={ver}")
    else:
        written = upsert_predictions(engine, db_cfg, all_rows)
        logger.info("Upserted %d prediction rows into %s", written, db_cfg.pred_table)

    if metrics:
        summary = pd.DataFrame(metrics, columns=["municipality", "rmse", "mae"]).sort_values("rmse")
        logger.info("\n%s", summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost water balance model and forecast future months.")
    p.add_argument("--municipality", type=str, default="ALL", help='Municipality name or "ALL" (default: ALL).')
    p.add_argument("--horizon", type=int, default=60, help="Forecast horizon in months (default: 60).")
    p.add_argument("--no-write", action="store_true", help="Do not write predictions to DB.")
    p.add_argument("--exog-cols", type=str, default="", help="Comma-separated exogenous columns (overrides .env).")
    args = p.parse_args()

    exog_cols_cli = [c.strip() for c in args.exog_cols.split(",")] if args.exog_cols.strip() else None
    return argparse.Namespace(
        municipality=args.municipality,
        horizon=args.horizon,
        no_write=args.no_write,
        exog_cols=exog_cols_cli,
    )


if __name__ == "__main__":
    ns = parse_args()
    run(ns.municipality, ns.horizon, ns.no_write, ns.exog_cols)
