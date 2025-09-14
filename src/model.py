from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
import xgboost as xgb

from .config import ModelConfig


@dataclass
class TrainResult:
    municipality: str
    model: xgb.XGBRegressor
    rmse: float
    mae: float
    r2: float
    mse: float
    features_used: List[str]


def _time_series_split(
    df_muni: pd.DataFrame,
    y_col: str = "water_balance",
    valid_frac: float = 0.2,
) -> int:
    """Chronological split: last valid_frac portion is validation."""
    n = len(df_muni)
    n_val = max(1, int(round(n * valid_frac)))
    split = max(1, n - n_val)
    return split


def _augment_training_rows(
    df_tr: pd.DataFrame,
    y_col: str,
    mode: str,
    scale: float,
    multiplier: int,
) -> pd.DataFrame:
    """
    Time-series-safe augmentation applied ONLY to training rows.
      - "none": no augmentation
      - "noise": add Gaussian noise
      - "seasonal_bootstrap": resample within same calendar month + small noise
    """
    if mode == "none" or len(df_tr) == 0 or multiplier <= 1:
        return df_tr

    num_cols = list(df_tr.select_dtypes(include=[np.number]).columns)
    PROTECTED = {"year", "month", "day", "provincialID", "municipalID", "municipalityID"}
    PROTECTED |= {c for c in df_tr.columns if c.lower().endswith("id")}
    PROTECTED.add(y_col)
    feat_cols = [c for c in num_cols if c not in PROTECTED]

    def _add_noise_blockwise(df_src: pd.DataFrame, ref_for_std: pd.DataFrame) -> pd.DataFrame:
        out = df_src.copy()
        if feat_cols:
            base = out[feat_cols].astype(np.float64).to_numpy(copy=True)
            ref_std = (
                ref_for_std[feat_cols]
                .astype(np.float64).std(ddof=0)
                .replace(0, 1.0).astype(np.float64)
                .to_numpy(copy=False)
            )
            noise = np.random.normal(0.0, scale, size=base.shape)
            bumped = base + noise * ref_std
            for j, col in enumerate(feat_cols):
                out[col] = bumped[:, j].astype(np.float64)
        # target
        y_base = out[y_col].astype(np.float64).to_numpy(copy=True)
        y_std = float(ref_for_std[y_col].astype(np.float64).std(ddof=0) or 1.0)
        y_bumped = y_base + np.random.normal(0.0, scale, size=len(out)) * y_std
        out[y_col] = pd.Series(y_bumped, index=out.index, dtype=np.float64)
        return out

    augmented = [df_tr]

    if mode == "noise":
        for _ in range(multiplier - 1):
            augmented.append(_add_noise_blockwise(df_tr, ref_for_std=df_tr))

    elif mode == "seasonal_bootstrap":
        if "month" not in df_tr.columns:
            for _ in range(multiplier - 1):
                augmented.append(_add_noise_blockwise(df_tr, ref_for_std=df_tr))
        else:
            rng = np.random.default_rng()
            for _ in range(multiplier - 1):
                dups = []
                for m in range(1, 13):
                    pool = df_tr[df_tr["month"] == m]
                    if len(pool) == 0:
                        continue
                    take = max(1, int(np.ceil(len(pool) * 0.5)))
                    picks = pool.sample(n=take, replace=True, random_state=rng.integers(1_000_000))
                    dups.append(_add_noise_blockwise(picks, ref_for_std=pool))
                if dups:
                    augmented.append(pd.concat(dups, ignore_index=True))

    return pd.concat(augmented, ignore_index=True)


class XGBWaterBalanceModel:
    def __init__(
        self,
        cfg: ModelConfig,
        augmentation_mode: str = "none",
        augmentation_scale: float = 0.05,
        augmentation_multiplier: int = 1,
        max_depth: int | None = None,
        n_estimators: int | None = None,
        learning_rate: float | None = None,
        subsample: float | None = None,
        colsample_bytree: float | None = None,
        reg_lambda: float | None = None,
    ):
        self.cfg = cfg
        self.augmentation_mode = augmentation_mode
        self.augmentation_scale = augmentation_scale
        self.augmentation_multiplier = augmentation_multiplier

        # store overrides only if provided
        self._hp_overrides = {}
        if max_depth is not None:        self._hp_overrides["max_depth"] = max_depth
        if n_estimators is not None:     self._hp_overrides["n_estimators"] = n_estimators
        if learning_rate is not None:    self._hp_overrides["learning_rate"] = learning_rate
        if subsample is not None:        self._hp_overrides["subsample"] = subsample
        if colsample_bytree is not None: self._hp_overrides["colsample_bytree"] = colsample_bytree
        if reg_lambda is not None:       self._hp_overrides["reg_lambda"] = reg_lambda

    def _base_estimator(self, province: str | None = None) -> xgb.XGBRegressor:
        params = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )

        if province in ["Abra", "Apayao"]:
            params.update({
                "max_depth": 10,
                "n_estimators": 1500,
                "learning_rate": 0.03,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            })



        return xgb.XGBRegressor(**params)

    def _fit_plain(
        self,
        model: xgb.XGBRegressor,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
    ) -> Tuple[xgb.XGBRegressor, float, float, float, float]:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        preds = model.predict(X_va)
        rmse = root_mean_squared_error(y_va, preds)
        mae = mean_absolute_error(y_va, preds)
        mse = mean_squared_error(y_va, preds)
        n = int(len(y_va))
        if n >= 2:
            ss_res = np.sum((y_va - preds) ** 2)
            ss_tot = np.sum((y_va - np.mean(y_va)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
        else:
            r2 = float("nan")
        return model, rmse, mae, r2, mse

    def train_one(self, df_super: pd.DataFrame, municipality: str) -> TrainResult:
        g = df_super[df_super["municipality"] == municipality].copy()
        if len(g) < 24:
            raise ValueError(f"Not enough rows to train {municipality}: {len(g)}")

        y_col = "water_balance"
        drop_cols = {"water_balance", "date", "municipality"}
        numeric_cols = [c for c in g.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(g[c])]
        X_cols = numeric_cols

        split = _time_series_split(g, y_col=y_col, valid_frac=0.2)
        g_tr = g.iloc[:split].copy()
        g_va = g.iloc[split:].copy()

        # augment train only
        if self.augmentation_mode != "none":
            g_tr = _augment_training_rows(
                g_tr,
                y_col=y_col,
                mode=self.augmentation_mode,
                scale=self.augmentation_scale,
                multiplier=self.augmentation_multiplier,
            )

        X_tr, y_tr = g_tr[X_cols], g_tr[y_col]
        X_va, y_va = g_va[X_cols], g_va[y_col]

        # pick province if available in the data
        prov = g["province"].iloc[0] if "province" in g.columns else None
        prov = g["province"].iloc[0] if "province" in g.columns else None
        model = self._base_estimator(province=prov)

        model, rmse, mae, r2, mse = self._fit_plain(model, X_tr, y_tr, X_va, y_va)
        r2 = model.score(X_va, y_va)

        return TrainResult(
            municipality=municipality,
            model=model,
            rmse=rmse,
            mae=mae,
            r2=r2,
            mse=mse,
            features_used=X_cols,
        )

    def recursive_forecast(
        self,
        df_super: pd.DataFrame,
        model: xgb.XGBRegressor,
        municipality: str,
        horizon: int,
    ) -> List[Tuple[str, int, int, float]]:
        g = df_super[df_super["municipality"] == municipality].copy()
        g = g.sort_values(["year", "month"])
        last = g.iloc[-1:].copy()

        drop_cols = {"water_balance", "date", "municipality"}
        X_cols = [c for c in g.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(g[c])]

        results = []
        y, m = int(last["year"].values[0]), int(last["month"].values[0])
        for _ in range(horizon):
            yhat = float(model.predict(last[X_cols])[0])
            m += 1
            if m > 12:
                y += 1
                m = 1
            results.append((municipality, y, m, yhat))
        return results
