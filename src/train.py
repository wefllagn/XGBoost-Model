from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from .config import DBConfig, ModelConfig
from .db import get_engine, read_water_table, upsert_predictions
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
    train_window: str,
    augmentation: str,
    aug_scale: float,
    aug_multiplier: int,
) -> None:
    db_cfg = DBConfig()

    # Resolve targets
    if targets_cli and any(t.strip() for t in targets_cli):
        targets = [t.strip() for t in targets_cli if t.strip()]
    else:
        targets = ["changeinstorage(mm)"]

    # Resolve exogenous columns from CLI or ENV
    import os
    env_cols = os.getenv("EXOG_COLS", "")
    exog_from_env = [c.strip() for c in env_cols.split(",") if c.strip()] if env_cols.strip() else []
    exog_from_cli = [c.strip() for c in (exog_cols_cli or []) if c.strip()]
    base_exog = exog_from_cli or exog_from_env

    engine = get_engine(db_cfg)
    df_raw = read_water_table(engine, db_cfg)

    # Municipality list (Benguet-only default)
    BENGUET_ID = 1401100000
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
            df_temp[["municipality"] + ([c for c in ["province", "provincialID"] if c in df_temp.columns])]
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
    all_console_rows: List[Tuple[str, str, int, int, float, str]] = []
    all_metrics: List[Tuple[str, str, float, float]] = []

    # Loop over targets
    for target in targets:
        exog_effective = [c for c in base_exog if c != target] if base_exog else None
        model_cfg = ModelConfig(target_col=target, exog_cols=exog_effective)

        # Map & window (after mapping so 'year' exists)
        df_mapped = auto_map_columns(df_raw, model_cfg)
        if train_window == "last7":
            cutoff_year = int(df_mapped["year"].max()) - 6
            df_mapped = df_mapped[df_mapped["year"] >= cutoff_year]
            logger.info(
                "Using last 7 years for target %s (>= %s). Rows kept: %d",
                target, cutoff_year, len(df_mapped)
            )
        elif train_window == "all":
            logger.info(
                "Using full history for target %s (since %s). Rows: %d",
                target, int(df_mapped["year"].min()), len(df_mapped)
            )

        _dbg_stats("overall (after auto_map_columns + window)", df_mapped)
        df_super = build_supervised_frame(df_mapped, model_cfg)
        _dbg_stats("after build_supervised_frame", df_super)
        print(
            "[DEBUG] Sample supervised rows:\n",
            df_super[["municipality", "year", "month", "water_balance"]]
            .sort_values(["municipality", "year", "month"])
            .head(12)
            .to_string(index=False),
        )

        # Instantiate model with augmentation config
        trainer = XGBWaterBalanceModel(
            model_cfg,
            augmentation_mode=augmentation,
            augmentation_scale=aug_scale,
            augmentation_multiplier=aug_multiplier,
        )

        for muni in muni_list:
            try:
                result = trainer.train_one(df_super, muni)
                logger.info("Trained %s | target=%s | RMSE=%.4f | MAE=%.4f | AIC=%.2f | BIC=%.2f | features=%d",
                    muni, target, result.rmse, result.mae, result.aic, result.bic, len(result.features_used)
                )
                all_metrics.append((target, muni, result.rmse, result.mae))

                fc = trainer.recursive_forecast(df_super, result.model, muni, horizon)
                for (m, y, mo, yhat) in fc:
                    all_console_rows.append((target, m, y, mo, yhat, model_version))
            except Exception as e:
                logger.exception("Failed training for %s (target=%s): %s", muni, target, e)

    # Console forecasts
    if all_console_rows:
        from itertools import islice
        print("\n=== Forecasts (console only) ===")
        for t in _unique_sorted([t for (t, *_rest) in all_console_rows]):
            print(f"\n--- Target: {t} ---")
            rows_t = [r for r in all_console_rows if r[0] == t]
            for row in islice(rows_t, 12):
                _, muni, yy, mm, yhat, ver = row
                print(f"{muni:15s}  {yy}-{mm:02d}  y_pred={yhat:.4f}  model={ver}")

    # Metrics summary
    if all_metrics:
        df_met = pd.DataFrame(all_metrics, columns=["target", "municipality", "rmse", "mae"])
        print("\n=== Metrics per target (avg across municipalities) ===")
        per_target = (
            df_met.groupby("target", as_index=False)[["rmse", "mae"]]
            .mean()
            .sort_values("rmse")
        )
        print(per_target.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost models and forecast hydrological targets.")
    p.add_argument("--municipality", type=str, default="ALL", help='Municipality name or "ALL".')
    p.add_argument("--horizon", type=int, default=60, help="Forecast horizon in months.")
    p.add_argument("--no-write", action="store_true", help="Do not write predictions to DB.")
    p.add_argument("--exog-cols", type=str, default="", help="Comma-separated exogenous columns.")
    p.add_argument("--targets", type=str, default="", help="Comma-separated target columns.")
    p.add_argument("--train-window", type=str, default="all", choices=["all", "last7"],
                   help="Training window to use.")
    p.add_argument("--augmentation", type=str, default="none",
                   choices=["none", "noise", "seasonal_bootstrap"],
                   help="Data augmentation mode.")
    p.add_argument("--aug-scale", type=float, default=0.05,
                   help="Augmentation noise scale (fraction of col std).")
    p.add_argument("--aug-multiplier", type=int, default=1,
                   help="How many total copies (1=no extra, 2=double, etc.).")
    args = p.parse_args()

    exog_cols_cli = [c.strip() for c in args.exog_cols.split(",")] if args.exog_cols.strip() else None
    targets_cli = [c.strip() for c in args.targets.split(",")] if args.targets.strip() else None

    return argparse.Namespace(
        municipality=args.municipality,
        horizon=args.horizon,
        no_write=args.no_write,
        exog_cols=exog_cols_cli,
        targets=targets_cli,
        train_window=args.train_window,
        augmentation=args.augmentation,
        aug_scale=args.aug_scale,
        aug_multiplier=args.aug_multiplier,
    )


if __name__ == "__main__":
    ns = parse_args()
    run(
        ns.municipality,
        ns.horizon,
        ns.no_write,
        ns.exog_cols,
        ns.targets,
        ns.train_window,
        ns.augmentation,
        ns.aug_scale,
        ns.aug_multiplier,
    )
