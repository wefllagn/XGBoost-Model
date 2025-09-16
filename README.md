# HydroHub XGBoost Time Series Model (Municipal Water Balance)

The repository includes an **XGBoost**-based time series model to forecast municipal **water balance** in the Cordillera Administrative Region (CAR).

- Connects to the SQL database (default: **MySQL** via `pymysql` + SQLAlchemy)
- Loads historical municipal water balance (and optional exogenous features)
- Builds supervised learning features (lags, rolling means, seasonal encodings)
- Trains `xgboost.XGBRegressor` with time-aware validation
- Evaluates with **RMSE** and **MAE**
- Forecasts the next **N months** (default: 60 = 5 years)
- Upserts predictions into a table (default: `PREDWATERBALMUNI`)

## 1) How to set up

**You must have:**

- Python 3.10+
- A SQL database with at least one table of historical water balance by municipality

Install deps:

```bash
pip install -r requirements.txt
```

---

## 2) Environment variables

Create a `.env` file in the project root:

# Paste this

DB_DIALECT=mysql
DB_DRIVER=pymysql
DB_HOST=hydrohub-acetone-hydrohub.d.aivencloud.com
DB_PORT=13519
DB_USER=avnadmin
DB_PASSWORD=AVNS_1fPYqBgVs5UCT9N0YmL
DB_NAME=HydroHub
WATER_TABLE=WATERBALMUNI
PRED_TABLE=PREDWATERBALMUNI

> **Note:** never commit `.env` to version control.

---

## 3) Train & Forecast

- Predict all features without any hyperparameters

```bash
python -m src.train --horizon 60 --targets "precipitation(mm),evapotranspiration(mm),runoff(mm),soilmoisture(mm),changeinstorage(mm)" --no-write
```

- Predict using the last 7 years of the data

```bash
python -m src.train --targets "changeinstorage(mm)" --horizon 60 --train-window last7 --no-write
```

- Predict with data augmentation

```bash
python -m src.train --horizon 60 --targets "precipitation(mm),evapotranspiration(mm),runoff(mm),soilmoisture(mm),changeinstorage(mm)" --augmentation noise --aug-scale 0.05 --aug-multiplier 2 --no-write
```

- Predict using 7 years of data with augmentation

```bash
python -m src.train --train-window last7 --targets "precipitation(mm),evapotranspiration(mm),runoff(mm),soilmoisture(mm),changeinstorage(mm)" --augmentation noise --aug-scale 0.05 --aug-multiplier 2 --no-write
```

- Train all provinces

```bash
python -m src.train --province ALL --targets "changeinstorage(mm)" --horizon 60 --no-write
```

**USEFUL FLAGS**

`--municipality "La Trinidad"` to filter which muni to train.

`--exog-cols "precipitation(mm),evapotranspiration(mm),..."` to include drivers.

`--no-write` to avoid DB writes (console only).

---

## 4) Notes

- The pipeline uses **time-based splits** (`TimeSeriesSplit`) with early stopping to avoid leakage.
- If you add or rename columns, pass `--exog-cols` at runtime or update `.env`.
- For PostgreSQL, set `DB_DIALECT=postgresql` and `DB_DRIVER=psycopg2` in `.env` and install `psycopg2-binary`.
