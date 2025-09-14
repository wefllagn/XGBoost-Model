from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

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


CORDILLERA_PROVINCES = {
    "Abra",
    "Apayao",
    "Benguet",
    "Ifugao",
    "Kalinga",
    "Mountain Province",
}


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


def _select_municipalities(
    df_raw: pd.DataFrame,
    model_cfg: ModelConfig,
    province: str,
    municipality: Optional[str],
) -> Dict[str, List[str]]:
    """
    Return mapping: province -> list of municipalities to train.
    Filtering is done using the mapped (auto_map_columns) frame so column names are consistent.
    """
    df_temp = auto_map_columns(df_raw, model_cfg)

    # Basic column checks
    needed = {"province", "municipality"}
    missing = [c for c in needed if c not in df_temp.columns]
    if missing:
        raise ValueError(f"Expected columns missing after mapping: {missing}")

    # Normalize strings
    df_temp["province"] = df_temp["province"].astype(str).str.strip()
    df_temp["municipality"] = df_temp["municipality"].astype(str).str.strip()

    # If a specific municipality is requested, train just that, inferring its province
    if municipality and municipality.upper() != "ALL":
        muni = municipality.strip()
        rows = df_temp[df_temp["municipality"].str.casefold() == muni.casefold()]
        if rows.empty:
            found = _unique_sorted(df_temp["municipality"].unique())
            raise ValueError(
                f"Municipality '{municipality}' not found. "
                f"Available examples: {found[:20]}{' ...' if len(found) > 20 else ''}"
            )
        provs = _unique_sorted(rows["province"].unique())
        # If multiple provinces somehow, keep all distinct province buckets
        province_to_munis = {}
        for p in provs:
            ms = (
                rows.loc[rows["province"] == p, "municipality"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            province_to_munis[p] = sorted(ms)
        return province_to_munis

    # Province filter
    if province and province.upper() != "ALL":
        prov = province.strip()
        dfp = df_temp[df_temp["province"].str.casefold() == prov.casefold()]
        if dfp.empty:
            found = _unique_sorted(df_temp["province"].unique())
            raise ValueError(
                f"Province '{province}' not found. "
                f"Available: {found}"
            )
        munis = (
            dfp["municipality"].dropna().astype(str).unique().tolist()
        )
        return {dfp["province"].iloc[0]: sorted(munis)}

    # Province == ALL: take all Cordillera provinces present in the data
    present_provs = _unique_sorted(df_temp["province"].unique())
    car_provs = [p for p in present_provs if p in CORDILLERA_PROVINCES]
    if not car_provs:
        raise ValueError(
            "No Cordillera provinces found in data. "
            f"Data has provinces: {present_provs}"
        )

    province_to_munis: Dict[str, List[str]] = {}
    for p in car_provs:
        munis = (
            df_temp.loc[df_temp["province"] == p, "municipality"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        province_to_munis[p] = sorted(munis)

    return province_to_munis


def run(
    municipality: Optional[str],
    province: str,
    horizon: int,
    no_write: bool,
    exog_cols_cli: Optional[List[str]],
    targets_cli: Optional[List[str]],
    train_window: str,
    augmentation: str,
    aug_scale: float,
    aug_multiplier: int,
    max_depth: Optional[int] = None,
    n_estimators: Optional[int] = None,
    learning_rate: Optional[float] = None,
    subsample: Optional[float] = None,
    colsample_bytree: Optional[float] = None,
    reg_lambda: Optional[float] = None,
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
    exog_from_env = (
        [c.strip() for c in env_cols.split(",") if c.strip()]
        if env_cols.strip()
        else []
    )
    exog_from_cli = [c.strip() for c in (exog_cols_cli or []) if c.strip()]
    base_exog = exog_from_cli or exog_from_env

    engine = get_engine(db_cfg)
    df_raw = read_water_table(engine, db_cfg)

    model_version = f"xgb_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    all_console_rows: List[Tuple[str, str, int, int, float, str]] = []
    all_metrics: List[Tuple[str, str, float, float, float, float, str]] = []  # + province

    # Loop over targets
    for target in targets:
        exog_effective = [c for c in base_exog if c != target] if base_exog else None
        model_cfg = ModelConfig(target_col=target, exog_cols=exog_effective)

        # Determine which provinces/municipalities to train for this target
        province_to_munis = _select_municipalities(df_raw, model_cfg, province, municipality)

        # Map & window (after mapping so 'year' exists)
        df_mapped = auto_map_columns(df_raw, model_cfg)

        # --- Ensure we're training on THIS target ---
        if target not in df_mapped.columns:
            raise ValueError(
                f"Requested target '{target}' not found in data columns: {list(df_mapped.columns)}"
            )
        # Force-map to the requested target as water_balance
        df_mapped["water_balance"] = pd.to_numeric(df_mapped[target], errors="coerce")

        # Debug differences per target
        logger.info(
            "Target %s stats -> water_balance: count=%s min=%s max=%s mean=%s",
            target,
            df_mapped["water_balance"].count(),
            df_mapped["water_balance"].min(),
            df_mapped["water_balance"].max(),
            df_mapped["water_balance"].mean(),
        )

        # Train window
        if train_window == "last7":
            cutoff_year = int(df_mapped["year"].max()) - 6
            df_mapped = df_mapped[df_mapped["year"] >= cutoff_year]
            logger.info(
                "Using last 7 years for target %s (>= %s). Rows kept: %d",
                target,
                cutoff_year,
                len(df_mapped),
            )
        elif train_window == "all":
            logger.info(
                "Using full history for target %s (since %s). Rows: %d",
                target,
                int(df_mapped["year"].min()),
                len(df_mapped),
            )

        _dbg_stats("overall (after auto_map_columns + window)", df_mapped)
        df_super = build_supervised_frame(df_mapped, model_cfg)
        _dbg_stats("after build_supervised_frame", df_super)

        # nice sample print
        print(
            "[DEBUG] Sample supervised rows:\n",
            df_super[["province", "municipality", "year", "month", "water_balance"]]
            .sort_values(["province", "municipality", "year", "month"])
            .head(12)
            .to_string(index=False),
        )

        # Instantiate model with augmentation config
        trainer = XGBWaterBalanceModel(
            model_cfg,
            augmentation_mode=augmentation,
            augmentation_scale=aug_scale,
            augmentation_multiplier=aug_multiplier,
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
        )


        # Province -> municipalities
        for prov, muni_list in province_to_munis.items():
            logger.info("=== Training Province: %s | Target: %s ===", prov, target)
            for muni in muni_list:
                try:
                    # filter df_super by this municipality (train_one expects full df_super but selects muni)
                    result = trainer.train_one(df_super, muni)
                    logger.info(
                        "Trained %s | province=%s | target=%s | RMSE=%.4f | MAE=%.4f | R2=%.2f | MSE=%.2f | features=%d",
                        muni,
                        prov,
                        target,
                        result.rmse,
                        result.mae,
                        result.r2,
                        result.mse,
                        len(result.features_used),
                    )
                    all_metrics.append(
                        (target, muni, result.rmse, result.mae, result.mse, result.r2, prov)
                    )

                    fc = trainer.recursive_forecast(df_super, result.model, muni, horizon)
                    for (m, y, mo, yhat) in fc:
                        all_console_rows.append((target, m, y, mo, yhat, model_version))
                except Exception as e:
                    logger.exception(
                        "Failed training for %s (province=%s, target=%s): %s",
                        muni,
                        prov,
                        target,
                        e,
                    )

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
        df_met = pd.DataFrame(
            all_metrics,
            columns=["target", "municipality", "rmse", "mae", "mse", "r2", "province"],
        )

        print("\n=== Metrics per target (avg across municipalities) ===")
        per_target = (
            df_met.groupby("target", as_index=False)[["rmse", "mae", "mse", "r2"]]
            .mean()
            .sort_values("rmse")
        )
        print(per_target.to_string(index=False))

        # Also print per-province summary (handy!)
        print("\n=== Metrics per province & target (avg across municipalities) ===")
        per_prov_target = (
            df_met.groupby(["province", "target"], as_index=False)[["rmse", "mae", "mse", "r2"]]
            .mean()
            .sort_values(["province", "rmse"])
        )
        print(per_prov_target.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost models and forecast hydrological targets.")
    p.add_argument("--province", type=str, default="ALL",
                   help='Province name (e.g., "Benguet") or "ALL" for all Cordillera provinces.')
    p.add_argument("--municipality", type=str, default="ALL",
                   help='Municipality name (overrides province if set) or "ALL".')
    p.add_argument("--horizon", type=int, default=60, help="Forecast horizon in months.")
    p.add_argument("--no-write", action="store_true", help="Do not write predictions to DB.")
    p.add_argument("--exog-cols", type=str, default="", help="Comma-separated exogenous columns.")
    p.add_argument("--targets", type=str, default="", help="Comma-separated target columns.")
    p.add_argument(
        "--train-window",
        type=str,
        default="all",
        choices=["all", "last7"],
        help="Training window to use.",
    )
    p.add_argument(
        "--augmentation",
        type=str,
        default="none",
        choices=["none", "noise", "seasonal_bootstrap"],
        help="Data augmentation mode.",
    )
    p.add_argument("--aug-scale", type=float, default=0.05,
                   help="Augmentation noise scale (fraction of col std).")
    p.add_argument("--aug-multiplier", type=int, default=1,
                   help="How many total copies (1=no extra, 2=double, etc.).")
    p.add_argument("--max-depth", type=int, default=None, help="XGBoost max_depth")
    p.add_argument("--n-estimators", type=int, default=None, help="Number of trees")
    p.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    p.add_argument("--subsample", type=float, default=None, help="Subsample fraction")
    p.add_argument("--colsample-bytree", type=float, default=None, help="Column subsample fraction")
    p.add_argument("--reg-lambda", type=float, default=None, help="L2 regularization lambda")


    args = p.parse_args()

    exog_cols_cli = [c.strip() for c in args.exog_cols.split(",")] if args.exog_cols.strip() else None
    targets_cli = [c.strip() for c in args.targets.split(",")] if args.targets.strip() else None

    return argparse.Namespace(
        province=args.province,
        municipality=args.municipality,
        horizon=args.horizon,
        no_write=args.no_write,
        exog_cols=exog_cols_cli,
        targets=targets_cli,
        train_window=args.train_window,
        augmentation=args.augmentation,
        aug_scale=args.aug_scale,
        aug_multiplier=args.aug_multiplier,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
    )

    
if __name__ == "__main__":
    ns = parse_args()
    run(
        ns.municipality,
        ns.province,
        ns.horizon,
        ns.no_write,
        ns.exog_cols,
        ns.targets,
        ns.train_window,
        ns.augmentation,
        ns.aug_scale,
        ns.aug_multiplier,
        ns.max_depth,
        ns.n_estimators,
        ns.learning_rate,
        ns.subsample,
        ns.colsample_bytree,
        ns.reg_lambda,
    )
