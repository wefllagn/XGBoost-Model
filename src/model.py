from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

from .config import ModelConfig
from .features import make_future_calendar

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    municipality: str
    rmse: float
    mae: float
    n_train: int
    n_valid: int
    features_used: List[str]
    model: XGBRegressor


def _rmse(y_true, y_pred) -> float:
    """Sklearn-version-safe RMSE."""
    try:
        # sklearn >= 0.22
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        # older sklearn: no 'squared' kw
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class XGBWaterBalanceModel:
    """Wrapper around XGBRegressor for municipal water balance forecasting."""

    def __init__(self, cfg: ModelConfig, n_splits: int = 3):
        self.cfg = cfg
        self.n_splits = n_splits

    def _base_estimator(self):
    # Tune as needed
        return xgb.XGBRegressor(
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


    def _fit_plain(self, model: XGBRegressor, X_tr, y_tr, X_va, y_va) -> None:
        """Train without early stopping to avoid API differences across xgboost versions."""
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

    def train_one(self, df_supervised: pd.DataFrame, municipality: str) -> TrainResult:
        data = df_supervised[df_supervised["municipality"] == municipality].copy()
        if data.empty:
            raise ValueError(f"No rows for municipality={municipality}")

        drop_cols = {"municipality", "year", "month", "water_balance", "date"}
        X_cols = [c for c in data.columns if c not in drop_cols]
        y = data["water_balance"].values
        X = data[X_cols].values

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        best_rmse = float("inf")
        best_model: Optional[XGBRegressor] = None
        last_X_va = last_y_va = None  # for reporting after loop

        for _, (tr, va) in enumerate(tscv.split(X), start=1):
            X_tr, X_va = X[tr], X[va]
            y_tr, y_va = y[tr], y[va]

            model = self._base_estimator()
            self._fit_plain(model, X_tr, y_tr, X_va, y_va)

            preds = model.predict(X_va)
            rmse = _rmse(y_va, preds)
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                last_X_va, last_y_va = X_va, y_va

        assert best_model is not None and last_X_va is not None and last_y_va is not None

        # Report metrics on the best validation split
        preds = best_model.predict(last_X_va)
        rmse = _rmse(last_y_va, preds)
        mae = float(mean_absolute_error(last_y_va, preds))

        return TrainResult(
            municipality=municipality,
            rmse=rmse,
            mae=mae,
            n_train=len(y) - len(last_y_va),
            n_valid=len(last_y_va),
            features_used=X_cols,
            model=best_model,
        )

    def recursive_forecast(
        self,
        history_supervised: pd.DataFrame,
        model: XGBRegressor,
        municipality: str,
        horizon: int
    ) -> List[Tuple[str, int, int, float]]:
        """Produce horizon-step forecasts via recursive one-step predictions."""
        drop_cols = {"municipality", "year", "month", "water_balance", "date"}
        X_cols = [c for c in history_supervised.columns if c not in drop_cols]

        hist = history_supervised[history_supervised["municipality"] == municipality].copy()
        hist.sort_values(["year", "month"], inplace=True)
        last_year, last_month = int(hist.iloc[-1]["year"]), int(hist.iloc[-1]["month"])

        cur_df = hist.copy()
        out: List[Tuple[str, int, int, float]] = []
        for y, m in make_future_calendar(last_year, last_month, horizon):
            last_row = cur_df.iloc[-1:].copy()
            last_row["year"] = y
            last_row["month"] = m
            last_row["m"] = m
            last_row["sin_m"] = np.sin(2 * np.pi * m / 12.0)
            last_row["cos_m"] = np.cos(2 * np.pi * m / 12.0)

            X_input = last_row[X_cols].values
            y_hat = float(model.predict(X_input)[0])
            out.append((municipality, int(y), int(m), y_hat))

            synth = last_row.copy()
            synth["water_balance"] = y_hat
            cur_df = pd.concat([cur_df, synth], ignore_index=True)

        return out
