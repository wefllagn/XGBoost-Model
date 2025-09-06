
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

```
# Choose mysql or postgresql
DB_DIALECT=mysql  # or postgresql

# If mysql: use 'mysql+pymysql'
# If postgresql: use 'postgresql+psycopg2'
DB_DRIVER=pymysql

DB_HOST=your-hostname
DB_PORT=13519
DB_USER=avnadmin
DB_PASSWORD=your-strong-password
DB_NAME=hydrohub

# Tables
WATER_TABLE=WATERBALMUNI
PRED_TABLE=PREDWATERBALMUNI

# Optional: comma-separated list of exogenous feature columns if present
EXOG_COLS=precipitation,evapotranspiration,runoff,soil_moisture,storage_change

# Random seed for reproducibility
RANDOM_STATE=42
```

> **Note:** never commit `.env` to version control. 

---

## 3) Train & Forecast

Train **per municipality** and write **5-year** forecasts:


```bash
python -m src.train --horizon 60 
```

**EXECUTE THIS FOR RESULTS TO WRITE IN CONSOLE**
Evaluate only (no write):

```bash
python -m src.train --horizon 60 --no-write
```

Predict all features
```bash
python -m src.train --horizon 60 --targets "precipitation(mm),evapotranspiration(mm),runoff(mm),soilmoisture(mm),changeinstorage(mm)" --no-write
```

predict all features with augmentation for all municipalities
```bash
python -m src.train `
  --horizon 60 `
  --targets "precipitation(mm),evapotranspiration(mm),runoff(mm),soilmoisture(mm),changeinstorage(mm)"`
  --augmentation noise `
  --aug-scale 0.05 `
  --aug-multiplier 2 `
  --municipality "ALL" `
  --no-write
  
```

---

**USEFUL FLAGS**

```--municipality "La Trinidad"``` to filter which muni to train.

```--exog-cols "precipitation(mm),evapotranspiration(mm),..."``` to include drivers.

```--no-write``` to avoid DB writes (console only).

## 4) Notes

- The pipeline uses **time-based splits** (`TimeSeriesSplit`) with early stopping to avoid leakage.
- If you add or rename columns, pass `--exog-cols` at runtime or update `.env`.
- For PostgreSQL, set `DB_DIALECT=postgresql` and `DB_DRIVER=psycopg2` in `.env` and install `psycopg2-binary`.

