from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import os
import pandas as pd
import numpy as np
from tabulate import tabulate

from .config import DBConfig, ModelConfig
from .db import get_engine, read_water_table
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
    "Mt.Province",
}

ALL_FEATURES = {
    "soil_moisture(mm)",
    "precipitation(mm)",
    "runoff(mm)",
    "evapotranspiration(mm)"
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
    """Return mapping: province -> list of municipalities to train."""
    df_temp = auto_map_columns(df_raw, model_cfg)

    needed = {"province", "municipality"}
    missing = [c for c in needed if c not in df_temp.columns]
    if missing:
        raise ValueError(f"Expected columns missing after mapping: {missing}")

    df_temp["province"] = df_temp["province"].astype(str).str.strip()
    df_temp["municipality"] = df_temp["municipality"].astype(str).str.strip()

    province_to_munis: Dict[str, List[str]] = {}
    for p in CORDILLERA_PROVINCES:
        dfp = df_temp[df_temp["province"].str.casefold() == p.casefold()]
        if not dfp.empty:
            munis = (
                dfp["municipality"].dropna().astype(str).unique().tolist()
            )
            province_to_munis[p] = sorted(munis)

    return province_to_munis


def run(
    municipality: Optional[str],
    province: str,
    horizon: int,
    no_write: bool,
    exog_cols_cli: Optional[List[str]],
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

    # Always train on all features
    targets = ALL_FEATURES

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
    all_metrics: List[Tuple[str, str, float, float, float, float, str]] = []

    # Loop over all targets
    for target in targets:
        exog_effective = [c for c in base_exog if c != target] if base_exog else None
        model_cfg = ModelConfig(target_col=target, exog_cols=exog_effective)

        province_to_munis = _select_municipalities(
            df_raw, model_cfg, province, municipality
        )

        df_mapped = auto_map_columns(df_raw, model_cfg)

        if target not in df_mapped.columns:
            logger.info("Using target=%s from column '%s'.", target, target)
            continue

        df_mapped["water_balance"] = pd.to_numeric(
            df_mapped[target], errors="coerce"
        )

        logger.info(
            "Target %s stats -> count=%s min=%s max=%s mean=%s",
            target,
            df_mapped["water_balance"].count(),
            df_mapped["water_balance"].min(),
            df_mapped["water_balance"].max(),
            df_mapped["water_balance"].mean(),
        )

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

        _dbg_stats("overall", df_mapped)
        df_super = build_supervised_frame(df_mapped, model_cfg)
        _dbg_stats("after build_supervised_frame", df_super)

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

        for prov, muni_list in province_to_munis.items():
            logger.info("=== Training Province: %s | Target: %s ===", prov, target)
            for muni in muni_list:
                try:
                    result = trainer.train_one(df_super, muni)
                    logger.info(
                        "Trained %s | province=%s | target=%s | RMSE=%.4f | MAE=%.4f | R2=%.2f | MSE=%.2f",
                        muni, prov, target,
                        result.rmse, result.mae, result.r2, result.mse,
                    )
                    all_metrics.append(
                        (target, muni, result.rmse, result.mae, result.mse, result.r2, prov)
                    )
                    fc = trainer.recursive_forecast(df_super, result.model, muni, horizon)
                    for (m, y, mo, yhat) in fc:
                        all_console_rows.append((target, m, y, mo, yhat, model_version))
                except Exception as e:
                    logger.warning("Skipped %s (%s, %s): %s", muni, prov, target, e)

    # === Save forecasts to CSV ===
    if all_console_rows:

        df_fc_long = pd.DataFrame(
            all_console_rows,
            columns=["target", "municipality", "year", "month", "y_pred", "model_version"],
        )

        meta_cfg = ModelConfig(target_col="changeinstorage(mm)", exog_cols=None)
        df_meta = auto_map_columns(df_raw, meta_cfg)
        df_meta = build_supervised_frame(df_meta, meta_cfg)

        # Keep only the first row per municipality to fetch IDs/province
        meta_cols = ["municipality", "municipalID", "provincialID", "province"]
        present_meta_cols = [c for c in meta_cols if c in df_meta.columns]
        df_meta_unique = (
            df_meta[present_meta_cols]
            .dropna(subset=["municipality"])
            .drop_duplicates(subset=["municipality"])
        )

        # Merge metadata into forecast rows
        df_fc_long = df_fc_long.merge(df_meta_unique, on="municipality", how="left")

        # Build the Hydrohub-style date string: M/15/YYYY (matches your sample)
        df_fc_long["date"] = (
            df_fc_long["month"].astype(int).astype(str)
            + "/15/"
            + df_fc_long["year"].astype(int).astype(str)
        )

        # Pivot to wide so each target becomes its own column like the source CSV
        # Ensure column names exactly match your dataset headers
        target_name_map = {
            "changeinstorage(mm)": "changeinstorage(mm)",
            "runoff(mm)": "runoff(mm)",
            "soilmoisture(mm)": "soilmoisture(mm)",
            "precipitation(mm)": "precipitation(mm)",
            "evapotranspiration(mm)": "evapotranspiration(mm)",
            # If you ever used pretty names earlier, map them back to exact headers here.
            "Soil Moisture": "soilmoisture(mm)",
            "Precipitation": "precipitation(mm)",
            "Runoff": "runoff(mm)",
            "Evapotranspiration": "evapotranspiration(mm)",
        }
        df_fc_long["target"] = df_fc_long["target"].map(target_name_map).fillna(df_fc_long["target"])

        # Now pivot
        id_cols = ["municipalID", "municipality", "provincialID", "province", "date"]
        available_id_cols = [c for c in id_cols if c in df_fc_long.columns]
        df_fc_wide = (
            df_fc_long
            .pivot_table(
                index=available_id_cols + ["year", "month"],  # keep year/month for stable sort
                columns="target",
                values="y_pred",
                aggfunc="mean"  # just in case
            )
            .reset_index()
        )

        # Drop the pandas MultiIndex column name
        df_fc_wide.columns.name = None

        # Sort by province/municipality/date nicely
        sort_cols = [c for c in ["province", "municipality", "year", "month"] if c in df_fc_wide.columns]
        if sort_cols:
            df_fc_wide = df_fc_wide.sort_values(sort_cols)

        # Remove helper sort columns (year, month) from final output
        if "year" in df_fc_wide.columns: df_fc_wide = df_fc_wide.drop(columns=["year"])
        if "month" in df_fc_wide.columns: df_fc_wide = df_fc_wide.drop(columns=["month"])

        # Ensure all forecast columns exist (even if a target wasn’t trained)
        forecast_cols = [
            "changeinstorage(mm)",
            "runoff(mm)",
            "soilmoisture(mm)",
            "precipitation(mm)",
            "evapotranspiration(mm)",
        ]
        for col in forecast_cols:
            if col not in df_fc_wide.columns:
                df_fc_wide[col] = np.nan

        # Reorder columns to match your Hydrohub format
        ordered_cols = []
        for c in ["municipalID", "municipality", "provincialID", "province", "date"]:
            if c in df_fc_wide.columns:
                ordered_cols.append(c)
        ordered_cols += forecast_cols
        df_fc_wide = df_fc_wide[ordered_cols]

        # Write to forecasts/ folder
        os.makedirs("forecasts", exist_ok=True)
        out_csv = os.path.join("forecasts", f"forecasts_{model_version}.csv")
        df_fc_wide.to_csv(out_csv, index=False)
        print(f"\nFull forecasts saved to {out_csv}")

    # === Metrics summary ===
    if all_metrics:
        df_met = pd.DataFrame(
            all_metrics,
            columns=["target", "municipality", "rmse", "mae", "mse", "r2", "province"],
        )

        per_prov = (
            df_met.groupby("province", as_index=False)[["rmse", "mae", "mse", "r2"]]
            .mean()
        )

        # Ensure all 6 provinces appear
        for prov in CORDILLERA_PROVINCES:
            if prov not in per_prov["province"].values:
                per_prov = pd.concat(
                    [per_prov,
                     pd.DataFrame([{"province": prov,
                                    "rmse": float("nan"),
                                    "mae": float("nan"),
                                    "mse": float("nan"),
                                    "r2": float("nan")}])],
                    ignore_index=True,
                )

        per_prov = per_prov.sort_values("rmse", na_position="last")

        overall = (
            df_met[["rmse", "mae", "mse", "r2"]].mean()
            .to_frame().T.assign(province="OVERALL")
        )

        table = pd.concat([per_prov, overall], ignore_index=True)

        print("\n=== Metrics Summary ===")
        print(tabulate(table, headers="keys", tablefmt="pretty", floatfmt=".4f"))
        # Save the metrics table to a text file
        out_txt = f"metrics_{model_version}.txt"
        with open(out_txt, "w") as f:
            f.write("=== Metrics Summary ===\n")
            f.write(tabulate(table, headers="keys", tablefmt="pretty", floatfmt=".4f"))
            f.write("\n")
        print(f"Metrics summary saved to {out_txt}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost models on all provinces & features.")
    p.add_argument("--run", action="store_true", help="Shortcut to run everything with defaults.")
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--exog-cols", type=str, default="")
    p.add_argument("--train-window", type=str, default="all", choices=["all", "last7"])
    p.add_argument("--augmentation", type=str, default="none", choices=["none", "noise", "seasonal_bootstrap"])
    p.add_argument("--aug-scale", type=float, default=0.05)
    p.add_argument("--aug-multiplier", type=int, default=1)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--subsample", type=float, default=None)
    p.add_argument("--colsample-bytree", type=float, default=None)
    p.add_argument("--reg-lambda", type=float, default=None)

    args = p.parse_args()

    if args.run:
        return argparse.Namespace(
            municipality="ALL",
            province="ALL",
            horizon=args.horizon,
            no_write=args.no_write,
            exog_cols=None,
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

    return args


if __name__ == "__main__":
    ns = parse_args()
    run(
        ns.municipality,
        ns.province,
        ns.horizon,
        ns.no_write,
        ns.exog_cols,
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
