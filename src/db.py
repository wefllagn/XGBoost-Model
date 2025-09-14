
from __future__ import annotations

import logging
import os
from typing import Iterable, List, Tuple

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import DBConfig

logger = logging.getLogger(__name__)


def get_engine(cfg: DBConfig) -> Engine:
    """Create and return a SQLAlchemy engine."""
    url = cfg.sqlalchemy_url()
    logger.info("Creating SQLAlchemy engine for %s", url.replace(cfg.password, "***") if cfg.password else url)
    return create_engine(url, pool_pre_ping=True)


def read_water_table(engine: Engine, cfg: DBConfig) -> pd.DataFrame:
    """Read historical water balance data from the configured table."""
    try:
        sql = text(f"SELECT * FROM `{cfg.water_table}`") if 'mysql' in cfg.dialect else text(f'SELECT * FROM "{cfg.water_table}"')
        df = pd.read_sql(sql, con=engine)
        logger.info("Loaded %d rows from %s", len(df), cfg.water_table)
        return df
    except Exception as e:
        logger.warning("DB access failed: %s. Falling back to CSV.", e, exc_info=False)
        csv_path = os.path.join(os.path.dirname(__file__), "Hydrohub.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV fallback file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info("Loaded %d rows from fallback CSV: %s", len(df), csv_path)  
        return df



def ensure_pred_table(engine: Engine, cfg: DBConfig) -> None:
    """Create predictions table if missing."""
    if 'mysql' in cfg.dialect:
        ddl = f"""
        CREATE TABLE IF NOT EXISTS `{cfg.pred_table}` (
            municipality VARCHAR(255) NOT NULL,
            year INT NOT NULL,
            month INT NOT NULL,
            y_pred DOUBLE NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (municipality, year, month)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
    else:
        ddl = f"""
        CREATE TABLE IF NOT EXISTS "{cfg.pred_table}" (
            municipality TEXT NOT NULL,
            year INT NOT NULL,
            month INT NOT NULL,
            y_pred DOUBLE PRECISION NOT NULL,
            model_version TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (municipality, year, month)
        );
        """
    with engine.begin() as conn:
        conn.exec_driver_sql(ddl)
    logger.info("Ensured predictions table exists: %s", cfg.pred_table)


from sqlalchemy import text

def upsert_predictions(engine, db_cfg, rows):
    """Upsert list of prediction rows into PREDWATERBALMUNI.
    Accepts list of tuples OR list of dicts.
    Tuple order expected: (municipality, year, month, y_pred, model_version)
    """
    if not rows:
        return 0

    # Normalize to list[dict]
    keys = ["municipality", "year", "month", "y_pred", "model_version"]
    dict_rows = []
    for r in rows:
        if isinstance(r, dict):
            dict_rows.append(r)
        else:
            dict_rows.append(dict(zip(keys, r)))

    sql = f"""
    INSERT INTO `{db_cfg.pred_table}` (
        municipality, year, month, y_pred, model_version
    ) VALUES (
        :municipality, :year, :month, :y_pred, :model_version
    )
    ON DUPLICATE KEY UPDATE
        y_pred = VALUES(y_pred),
        model_version = VALUES(model_version),
        created_at = NOW();
    """

    with engine.begin() as conn:
        conn.execute(text(sql), dict_rows)
    return len(dict_rows)

