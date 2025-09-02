
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class DBConfig:
    """Database connection configuration."""
    dialect: str = os.getenv("DB_DIALECT", "mysql")  # mysql or postgresql
    driver: str = os.getenv("DB_DRIVER", "pymysql")  # pymysql or psycopg2
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "3306"))
    user: str = os.getenv("DB_USER", "root")
    password: str = os.getenv("DB_PASSWORD", "")
    name: str = os.getenv("DB_NAME", "hydrohub")
    water_table: str = os.getenv("WATER_TABLE", "WATERBALMUNI")
    pred_table: str = os.getenv("PRED_TABLE", "PREDWATERBALMUNI")

    def sqlalchemy_url(self) -> str:
        return f"{self.dialect}+{self.driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass(frozen=True)
class ModelConfig:
    """Model and feature engineering configuration."""
    target_col: str = "water_balance"
    municipality_col: str = "municipality"
    year_col: str = "year"
    month_col: str = "month"
    exog_cols: Optional[List[str]] = None  # e.g. ['precipitation', 'evapotranspiration', ...]
    seed: int = int(os.getenv("RANDOM_STATE", "42"))
    default_lags: List[int] = (1, 2, 3, 6, 12)
    default_rolls: List[int] = (3, 6, 12)
