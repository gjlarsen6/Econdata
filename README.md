# tep-ml — Macroeconomic Forecasting & Process Monitoring

A production-ready pipeline that refreshes U.S. economic data from public APIs, retrains LightGBM forecasting models, and prints a unified summary table of 12-month outlooks — designed to run weekly with a single command.

Optional modules extend coverage to industry-level employment and GDP data (BLS, BEA, World Bank) and Venture Capital activity by sector (Crunchbase — AI, Fintech, Healthcare).

The project also includes a separate Tennessee Eastman Process (TEP) fault-detection benchmark that compares four classification models.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Setup](#setup)
4. [Running the Weekly Refresh](#running-the-weekly-refresh)
5. [Sector API Integration](#sector-api-integration)
6. [Crunchbase VC Integration](#crunchbase-vc-integration)
7. [LightGBM Forecasting Models](#lightgbm-forecasting-models)
8. [Model Scripts](#model-scripts)
9. [Shared Utilities — macro_utils.py](#shared-utilities--macro_utilspy)
10. [Summary Table Output](#summary-table-output)
11. [TEP Fault Detection — train.py](#tep-fault-detection--trainpy)
12. [Output Files Reference](#output-files-reference)
13. [Data Files Reference](#data-files-reference)
14. [Scheduling with Cron](#scheduling-with-cron)
15. [Adding New Series](#adding-new-series)

---

## Project Overview

The macroeconomic side of this project answers one question every week: **where are key economic indicators heading over the next 12 months?** It does this by:

1. Fetching the latest observations from the FRED API (and optionally from BLS, BEA, and World Bank)
2. Retraining gradient-boosted tree models on the updated data
3. Generating 12-month recursive forecasts with 80% prediction intervals
4. Displaying everything in a single tabular summary

Four model groups cover distinct areas of the economy:

| Group | Series Modeled | Source Directory |
|---|---|---|
| Business Environment | Industrial production, capacity utilization, nonfarm payroll | `data/BusinessEnvironment/` |
| Consumer Demand | Disposable income, PCE, core inflation, retail sales, consumer sentiment | `data/ConsumerDemand/` |
| Cost of Capital | Fed funds rate, prime rate, yield curve spreads | `data/CostOfCapital/` |
| Risk & Leading Indicators | Recession probability, consumer sentiment | `data/RiskLeadingInd/` |

Optional sector modules extend coverage to industry-level data from BLS (employment by sector), BEA (GDP by industry), and the World Bank (sector shares of GDP).

An additional Crunchbase module (`--crunchbase`) tracks weekly VC investment activity across three segments — AI, Fintech, and Healthcare — building a time-series dataset suitable for LightGBM forecasting once sufficient history accumulates.

---

## Directory Structure

```
tep-ml/
│
├── fred_refresh.py            # Main orchestrator — run this weekly
├── sector_apis.py             # Sector API fetchers (BLS, BEA, World Bank, Trading Economics)
├── sector_model.py            # Generic LightGBM trainer for sector data
├── crunchbase_apis.py         # Crunchbase VC data fetcher (AI, Fintech, Healthcare)
├── vc_model.py                # LightGBM trainer for VC segment weekly data
├── macro_utils.py             # Shared feature engineering, training, forecasting, plotting
├── business_env_model.py      # Business Environment model (INDPRO, TCU, PAYEMS)
├── consumer_demand_model.py   # Consumer Demand model (6 series)
├── cost_of_capital_model.py   # Cost of Capital model (DFF, DPRIME, T10Y3M, T10Y2Y)
├── risk_model.py              # Risk model (recession probability, consumer sentiment)
├── data_summary.py            # Data inventory and feature engineering recommendations
├── train.py                   # TEP binary fault detection (LR, RF, LightGBM, MLP)
│
├── .env.example               # API key template — copy to .env and fill in
│
├── data/
│   ├── fred_ingestion_map_full_production.json   # FRED series catalogue
│   ├── BusinessEnvironment/   # INDPRO, TCU, PAYEMS, CAPUTLB50001SQ CSVs
│   ├── ConsumerDemand/        # DSPIC96, PCE, PCEPILFE, RSAFS, RRSFS, UMCSENT CSVs
│   ├── CostOfCapital/         # DFF, DPRIME, FEDFUNDS, PRIME, T10Y2Y, T10Y3M CSVs
│   ├── RiskLeadingInd/        # RECPROUSM156N, UMCSENT CSVs
│   ├── SectorAPIs/            # JSON API documentation for each sector data source
│   ├── Sector/                # Created automatically by sector refresh
│   │   ├── BLS/               #   Employment by industry (monthly)
│   │   ├── BEA/               #   GDP by industry (quarterly)
│   │   └── WorldBank/         #   Sector % of GDP (annual → monthly)
│   ├── VentureCapital/        # Created automatically by --crunchbase refresh
│   │   ├── crunchbase_vc_ingestion_map.json  #   API specification
│   │   ├── dim_category.csv   #   Resolved Crunchbase category UUIDs per segment
│   │   ├── dim_organization.csv #  Top-ranked orgs per segment (updated weekly)
│   │   ├── fact_funding_round.csv #  All fetched funding rounds (append-only)
│   │   ├── agg_ai_weekly.csv       #  AI segment weekly metrics — modeled by vc_model.py
│   │   ├── agg_fintech_weekly.csv  #  Fintech segment weekly metrics
│   │   └── agg_healthcare_weekly.csv #  Healthcare segment weekly metrics
│   └── TEP_*.csv              # Tennessee Eastman Process datasets (used by train.py only)
│
└── outputs/
    ├── results_*.json         # Forecast + validation results per model group
    ├── refresh_log.json       # Run history (last 52 weeks)
    ├── *_model_*.joblib       # Serialized LightGBM models
    └── *.png                  # Forecast dashboards, validation plots, feature importance
```

All CSV data files share a common format:

```
observation_date,SERIES_ID
1967-01-01,100.4
1967-02-01,100.8
...
```

---

## Setup

### Prerequisites

```bash
pip install lightgbm pandas numpy scikit-learn matplotlib requests tabulate joblib python-dotenv
```

### API Key Configuration

Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```ini
# Mandatory
FRED_API_KEY=your_fred_api_key_here

# Optional — only needed with the --sector flag
BEA_API_KEY=your_bea_api_key_here
BLS_API_KEY=your_bls_api_key_here       # BLS works without a key at lower limits
TE_CLIENT_KEY=your_te_client_key_here
TE_CLIENT_SECRET=your_te_client_secret_here

# Optional — only needed with the --crunchbase flag
CRUNCHBASE_API_KEY=your_crunchbase_api_key_here
```

**Where to get keys:**

| Key | Source | Cost |
|---|---|---|
| `FRED_API_KEY` | https://fred.stlouisfed.org/docs/api/api_key.html | Free |
| `BEA_API_KEY` | https://apps.bea.gov/api/signup/ | Free |
| `BLS_API_KEY` | https://www.bls.gov/developers/home.htm | Free (optional) |
| `TE_CLIENT_KEY/SECRET` | https://tradingeconomics.com/api/ | Commercial |
| `CRUNCHBASE_API_KEY` | https://data.crunchbase.com/docs/welcome-to-crunchbase-data | Paid (Fundamentals plan or above) |

You may also export keys directly as environment variables instead of using `.env`.

---

## Running the Weekly Refresh

```bash
python3 fred_refresh.py
```

This single command runs the full pipeline:

1. Fetches the latest observations for all 18 FRED series
2. Merges new rows into existing CSVs (never overwrites historical data)
3. Retrains all four LightGBM model groups
4. Prints a unified summary table with forecasts and validation metrics
5. Appends a run log entry to `outputs/refresh_log.json`

### Command-Line Options

```
python3 fred_refresh.py [--sector SOURCE ...] [--crunchbase] [--skip-models]
```

| Option | Description |
|---|---|
| *(no flags)* | FRED data refresh + all four model retrains |
| `--sector bls` | Also refresh BLS employment data and train BLS models |
| `--sector bea` | Also refresh BEA GDP-by-industry data (requires `BEA_API_KEY`) |
| `--sector worldbank` | Also refresh World Bank sector indicators |
| `--sector bls bea worldbank` | Multiple sources in one run |
| `--sector all` | All available sector sources |
| `--crunchbase` | Also refresh Crunchbase VC data (AI, Fintech, Healthcare) and train VC models (requires `CRUNCHBASE_API_KEY`) |
| `--skip-models` | Refresh data only — skip all model retraining |

### Examples

```bash
# Standard weekly run
python3 fred_refresh.py

# Include free sector APIs (no extra keys needed)
python3 fred_refresh.py --sector bls worldbank

# Data-only refresh — useful for testing or after an API outage
python3 fred_refresh.py --skip-models

# Full run with all sector APIs
python3 fred_refresh.py --sector all

# Refresh sector data without retraining FRED models
python3 fred_refresh.py --sector bls --skip-models

# Enable Crunchbase VC data collection (data only, no model training yet)
python3 fred_refresh.py --crunchbase --skip-models

# Full run: FRED + all sector APIs + Crunchbase VC
python3 fred_refresh.py --sector all --crunchbase
```

### What the Pipeline Does Step by Step

The step count shown in the log (`[1/N]`) adjusts automatically based on which flags are active: each of `--sector` and `--crunchbase` adds 2 steps (data refresh + model training).

```
[1/N]  Refresh FRED data (18 series)
         → Incremental fetch: only pulls data since last run (30-day lookback for revisions)
         → Rate-limited: 0.6 s between API calls (stays under 120 req/min)
         → Prints a per-series status table

[2/N]  Refresh sector API data  (only with --sector)
         → BLS: POST multi-series employment data
         → BEA: GET GDP by industry (all years, pivots to per-industry CSVs)
         → World Bank: GET annual indicators, forward-fills to monthly
         → Prints a per-series status table

[3/N]  Refresh Crunchbase VC data  (only with --crunchbase)
         → Resolves category UUIDs for AI, Fintech, Healthcare via autocomplete
         → Searches funding rounds (trailing 90 days) and top organizations per segment
         → Computes weekly aggregate metrics; appends to agg_{segment}_weekly.csv
         → Also updates dim_category.csv, dim_organization.csv, fact_funding_round.csv
         → Rate-limited: 0.31 s between calls (~33 calls/run, under 200/min limit)
         → Prints a per-segment status table

[N-2]  Retrain FRED LightGBM models
         → Runs 4 model scripts sequentially as subprocesses
         → Each script saves .joblib models, PNGs, and a results JSON

[N-1]  Train sector models  (only with --sector + data exists)
         → Discovers all CSVs in data/Sector/
         → Auto-creates models for any new series without an existing .joblib

[N-1]  Train VC models  (only with --crunchbase + data exists)
         → Resamples weekly agg CSVs to monthly cadence for LightGBM compatibility
         → Skips gracefully if fewer than 54 monthly rows (needs ~13 months of history)
         → Once sufficient data exists: trains, saves .joblib and results_vc_*.json

[N]    Print unified summary table
         → Reads all results JSONs (FRED + sector + VC as available)
         → Appends run log entry (keeps last 52 weeks)
```

### Interpreting the Summary Table

```
╭─────────────────────────────┬────────────┬─────────────┬──────────────┬────────────┬────────┬──────────┬─────────┬──────────┬──────────┬──────────┬──────────┬───────┬─────────────────────╮
│ Group                       │ Series     │ Label       │ Unit         │ Last Date  │ Last V │ Val MAE  │ Val R²  │ +1M      │ +3M      │ +6M      │ +12M     │ Trend │ Run At              │
├─────────────────────────────┼────────────┼─────────────┼──────────────┼────────────┼────────┼──────────┼─────────┼──────────┼──────────┼──────────┼──────────┼───────┼─────────────────────┤
│ Business Environment        │ INDPRO     │ Indl Prod.. │ Index 2017=  │ 2026-02    │ 102.55 │ 0.856    │ -0.5213 │ 101.29   │ 101.12   │ 100.87   │ 100.41   │  ↓   │ 2026-04-08T08:46:30 │
...
```

| Column | Meaning |
|---|---|
| `Last Date` | Most recent observation in the local CSV |
| `Last Value` | Value at that date |
| `Val MAE` | Mean absolute error on the 24-month held-out validation set |
| `Val R²` | R² on the held-out validation set (negative is normal for trending series — use MAE instead) |
| `+1M / +3M / +6M / +12M` | Forecast midpoint at each horizon |
| `Trend` | ↑ rising / ↓ falling / → stable (< 0.5% change over 12-month forecast) |

---

## Sector API Integration

The `sector_apis.py` module fetches industry-level data from four external sources and saves it to `data/Sector/` using the same CSV format as FRED data.

### BLS — Bureau of Labor Statistics

**No API key required.** Fetches monthly employment levels for six sectors:

| Series ID | Description |
|---|---|
| CES3000000001 | Manufacturing Employment |
| CES4000000001 | Trade, Transportation & Utilities Employment |
| CES5500000001 | Financial Activities Employment |
| CES6000000001 | Professional & Business Services Employment |
| CEU6500000001 | Education & Health Services Employment |
| CEU7000000001 | Leisure & Hospitality Employment |

Saved to: `data/Sector/BLS/<series_id>.csv`

### BEA — Bureau of Economic Analysis

**Requires `BEA_API_KEY`.** Fetches quarterly GDP by industry:

| Column Name | Description |
|---|---|
| BEA_Manufacturing | Gross Output — Manufacturing |
| BEA_Finance_Insurance_RE | Gross Output — Finance, Insurance & Real Estate |
| BEA_Wholesale_Retail_Trade | Gross Output — Wholesale & Retail Trade |
| BEA_Professional_Biz_Svcs | Gross Output — Professional & Business Services |

Saved to: `data/Sector/BEA/<industry>.csv` (quarterly dates)

### World Bank

**No API key required.** Fetches annual U.S. sector shares of GDP, then forward-fills to monthly frequency:

| Indicator | Description |
|---|---|
| NV.IND.MANF.ZS | Manufacturing Value Added (% of GDP) |
| NV.SRV.TOTL.ZS | Services Value Added (% of GDP) |
| NV.IND.TOTL.ZS | Industry Value Added (% of GDP) |

Saved to: `data/Sector/WorldBank/<indicator>.csv` (monthly, forward-filled from annual)

### Trading Economics

Requires a commercial plan (`TE_CLIENT_KEY` + `TE_CLIENT_SECRET`). A stub is implemented in `sector_apis.py`; extend `refresh_trading_economics()` with specific indicators once credentials are available.

### S&P Global / ISM PMI

No public API is available for either source. They are documented in `data/SectorAPIs/` for reference but not implemented.

---

## Crunchbase VC Integration

The `crunchbase_apis.py` module fetches weekly Venture Capital activity snapshots from the Crunchbase Data API v4 and saves them to `data/VentureCapital/`. The `vc_model.py` script trains LightGBM forecasting models on the accumulated time series once sufficient history is available.

### Enabling the Integration

Add your Crunchbase API key to `.env`:

```ini
CRUNCHBASE_API_KEY=your_crunchbase_api_key_here
```

Then run with the `--crunchbase` flag:

```bash
python3 fred_refresh.py --crunchbase
```

### Segments Tracked

Three canonical VC sectors are tracked, each resolved from a set of Crunchbase category queries:

| Segment | Category Resolution Queries |
|---|---|
| **AI** | artificial intelligence, generative ai, machine learning, ai infrastructure, computer vision, natural language processing |
| **Fintech** | fintech, payments, digital banking, insurtech, embedded finance, lending |
| **Healthcare** | health care, healthcare, health tech, biotech, medical device, digital health |

### Data Collected Per Run

Each weekly run (Monday snapshot date) appends or updates six files in `data/VentureCapital/`:

| File | Key Column | Description |
|---|---|---|
| `dim_category.csv` | `category_uuid` | Crunchbase category UUIDs resolved per segment (additive) |
| `dim_organization.csv` | `org_uuid` | Top-ranked organizations per segment; updated to latest snapshot |
| `fact_funding_round.csv` | `round_uuid` | All fetched funding rounds from trailing 90 days (append-only) |
| `agg_ai_weekly.csv` | `observation_date` | AI segment weekly metrics snapshot — **modeled by vc_model.py** |
| `agg_fintech_weekly.csv` | `observation_date` | Fintech segment weekly metrics snapshot |
| `agg_healthcare_weekly.csv` | `observation_date` | Healthcare segment weekly metrics snapshot |

The `agg_*_weekly.csv` files contain one row per Monday with these metrics:

| Metric | Description |
|---|---|
| `company_count` | Number of distinct organizations in the segment |
| `round_count` | Funding rounds announced in the trailing 90 days |
| `capital_raised_usd` | Total capital raised in the trailing 90 days (USD) |
| `median_round_size_usd` | Median funding round size in the trailing 90 days (USD) |
| `lead_investor_count` | Rounds with an identified lead investor in the trailing 90 days |

All data is additive — existing rows are never deleted. Running twice in the same week produces one row per segment (idempotent on the Monday snapshot date).

### LightGBM Modeling — `vc_model.py`

`vc_model.py` mirrors `sector_model.py` exactly. It can be run directly:

```bash
python3 vc_model.py --segment ai
python3 vc_model.py --segment ai fintech healthcare
python3 vc_model.py --segment all
```

**Data accumulation requirement:** The macro_utils feature engineering pipeline uses monthly lags up to 12 months. Weekly data is resampled to monthly (last weekly row per calendar month), and the model requires at least 54 monthly rows for a meaningful train/validation split. This means approximately **13 months of weekly collection** must occur before models train for the first time.

During the accumulation period, `vc_model.py` prints a clear skip message and exits cleanly. The `--crunchbase` flag is safe to enable from the very first run — data collection proceeds even when models cannot yet be trained.

Once sufficient data is available, the script produces:
- `outputs/vc_model_{segment}_{metric}.joblib` — one model per metric per segment
- `outputs/results_vc_{segment}.json` — forecasts + validation metrics for the summary table
- `outputs/vc_{segment}_{dashboard,validation,importance}.png` — standard three-plot set

### Rate Limits

The Crunchbase API allows 200 calls per minute. `crunchbase_apis.py` sleeps 0.31 s between every call. A typical weekly run makes approximately 33 calls (18 autocomplete + ~9 round search pages + ~6 org search pages), consuming ~10 seconds of enforced sleep time.

---

## LightGBM Forecasting Models

### Architecture

Each economic series gets **three** LightGBM regressors:

| Model | Objective | Purpose |
|---|---|---|
| `mid` | `regression` (median) | Point forecast — the central estimate |
| `lo` | `quantile` α = 0.10 | Lower bound of the 80% prediction interval |
| `hi` | `quantile` α = 0.90 | Upper bound of the 80% prediction interval |

Only the `mid` model is serialized to disk (`.joblib`). All three are used during inference inside each model script.

### Hyperparameters (universal across all models)

```python
n_estimators    = 2000      # Maximum trees; early stopping usually kicks in sooner
num_leaves      = 31        # Moderate complexity — avoids overfitting short series
learning_rate   = 0.03      # Slow learning rate balanced by early stopping
subsample       = 0.8       # Row sampling per tree
colsample_bytree= 0.8       # Feature sampling per tree
min_child_samples = 10      # Minimum leaf size
reg_alpha       = 0.1       # L1 regularization
reg_lambda      = 0.1       # L2 regularization
early_stopping  = 50        # Stop if validation loss doesn't improve for 50 rounds
```

### Feature Engineering

For every series in a model group, `macro_utils.engineer_features()` generates:

| Feature Type | Details |
|---|---|
| **Lags** | 1, 2, 3, 6, and 12 months back |
| **Rolling mean** | 3-month, 6-month, 12-month windows (shifted by 1 to avoid leakage) |
| **Rolling std** | 3-month, 6-month, 12-month windows (shifted by 1) |
| **Momentum** | Month-over-month % change (shifted by 1) |
| **Year-over-year** | 12-month % change (shifted by 1) |
| **Seasonality** | Calendar month (1–12), calendar quarter (1–4) |
| **Cross-term** | Product of lag-1 values for the first two series (captures co-movement) |

A typical model group with 4 series produces 50+ features. Rows with any NaN feature are dropped before training.

### Training and Validation

```
Full historical data
├── Training set: all rows except the last 24 months
└── Validation set: last 24 months (held out, never seen during training)
```

The 24-month validation window is used for:
- Early stopping during training
- Computing MAE, RMSE, and R² reported in the summary table

### Recursive Forecasting

`macro_utils.joint_recursive_forecast()` generates all 12 monthly forecasts by iteratively feeding predictions back as inputs:

```
Step 1: Predict month T+1 using actual data up to T
Step 2: Append prediction to working dataframe; re-engineer features
Step 3: Predict month T+2 using actual data + step-1 prediction
...
Step 12: Predict month T+12 using actual data + steps 1–11 predictions
```

All series in a group are forecast jointly at each step so cross-series lag features remain coherent.

Predictions are clipped to economically plausible ranges (e.g., capacity utilization 50–100%, consumer sentiment 20–120) before being committed to the working dataframe.

### Prediction Intervals

The 80% prediction interval `[lo, hi]` comes from the 10th and 90th quantile regressors. After clipping, the interval is enforced to be non-negative in width: `lo = min(lo, mid)`, `hi = max(hi, mid)`.

---

## Model Scripts

### business_env_model.py

Covers the supply side of the economy.

| Series | Description | Unit | Clip Range |
|---|---|---|---|
| INDPRO | Industrial Production Index | Index 2017=100 | 0 – 200 |
| TCU | Total Capacity Utilization | % | 50 – 100 |
| PAYEMS | Nonfarm Payroll Employment | thousands of persons | 0 – 300,000 |
| CAPUTLB50001SQ | Manufacturing Capacity Utilization (quarterly) | % | feature-only |

Data starts January 1967. CAPUTLB50001SQ is quarterly and is forward-filled to monthly before merging.

**Outputs:** `business_env_model_{INDPRO,TCU,PAYEMS}.joblib`, `business_env_{dashboard,validation,importance}.png`, `results_business_env.json`

---

### consumer_demand_model.py

Covers household spending and sentiment.

| Series | Description | Unit | Clip Range |
|---|---|---|---|
| DSPIC96 | Real Disposable Personal Income | billions of chained 2017 $ | 5,000 – 30,000 |
| PCE | Personal Consumption Expenditures | billions of $ | 5,000 – 40,000 |
| PCEPILFE | Core PCE Price Index (ex food & energy) | Index 2017=100 | 10 – 200 |
| RSAFS | Nominal Retail & Food Services Sales | millions of $ | 50,000 – 1,500,000 |
| RRSFS | Real Retail & Food Services Sales | millions of chained $ | 50,000 – 500,000 |
| UMCSENT | U. of Michigan Consumer Sentiment | Index 1966:Q1=100 | 20 – 120 |

Data starts January 1992 (when RSAFS/RRSFS become available). UMCSENT gaps of up to 2 months are interpolated. Dashboard uses a 2-column layout for 6 panels.

**Outputs:** `consumer_demand_model_{series}.joblib` (×6), `consumer_demand_{dashboard,validation,importance}.png`, `results_consumer_demand.json`

---

### cost_of_capital_model.py

Covers interest rates and yield curve dynamics. All series are daily in FRED and are resampled to monthly means.

| Series | Description | Unit | Clip Range | Notes |
|---|---|---|---|---|
| DFF | Effective Federal Funds Rate | % | 0 – 25 | Daily → monthly mean |
| DPRIME | Bank Prime Loan Rate | % | 0 – 30 | Daily → monthly mean |
| T10Y3M | Yield Curve: 10Y minus 3M | pct pts | −5 to 8 | Daily → monthly mean |
| T10Y2Y | Yield Curve: 10Y minus 2Y | pct pts | −3 to 5 | Only available from 2021-04; trained on subset |
| FEDFUNDS | Monthly Fed Funds Rate | % | — | Feature only |
| PRIME | Prime Rate (event-based) | % | — | Feature only; forward-filled |

T10Y2Y has only ~60 months of data (April 2021 onward) and is trained on that subset. Its validation and forecasts are produced independently of the main model fit.

**Outputs:** `cost_of_capital_model_{DFF,DPRIME,T10Y3M,T10Y2Y}.joblib`, `cost_of_capital_{dashboard,validation,importance}.png`, `results_cost_of_capital.json`

---

### risk_model.py

Covers recession risk and consumer confidence. More elaborate than the other scripts — produces four plot types including a risk snapshot gauge.

| Series | Description | Unit | Notes |
|---|---|---|---|
| RECPROUSM156N | Chauvet-Piger Recession Probability | % | Threshold annotation at 20% |
| UMCSENT | U. of Michigan Consumer Sentiment | Index 1966:Q1=100 | Threshold annotation at 70 |

Data starts January 1978. A cross-interaction feature (recession probability × sentiment) captures co-movement between risk and confidence.

**Outputs:** `risk_model_{recpro,sentiment}.joblib`, `risk_{forecast_dashboard,validation,feature_importance,snapshot}.png`, `results_risk.json`

---

### vc_model.py

Generic model script that trains LightGBM forecasters for Crunchbase VC segment data. Invoked automatically by `fred_refresh.py` when `--crunchbase` is used, or directly:

```bash
python3 vc_model.py --segment ai
python3 vc_model.py --segment ai fintech healthcare
python3 vc_model.py --segment all
```

For each segment it:
1. Loads the `agg_{segment}_weekly.csv` from `data/VentureCapital/`
2. Resamples weekly observations to monthly cadence (last weekly row per month)
3. Skips if fewer than 54 monthly rows are available (prints a helpful message about the accumulation timeline)
4. Excludes any metric columns with fewer than 10 non-NaN values
5. Auto-computes clip ranges from the 1st–99th percentile of each series' historical values (± 10% buffer)
6. Trains and forecasts using the same `macro_utils` pipeline as all other model scripts
7. Saves per-segment results JSONs (`results_vc_{segment}.json`) that feed into the summary table

**New VC segments** (a CSV without a corresponding `.joblib`) trigger automatic model creation on the next `fred_refresh.py --crunchbase` run once enough data has accumulated.

---

### sector_model.py

Generic model script that trains LightGBM forecasters for any sector source. Invoked automatically by `fred_refresh.py` when `--sector` is used, or directly:

```bash
python3 sector_model.py --source bls
python3 sector_model.py --source bls bea worldbank
python3 sector_model.py --source all
```

For each source, it:
1. Discovers all CSVs in `data/Sector/{Source}/`
2. Outer-merges them on date; forward-fills gaps of up to 3 months
3. Excludes series with fewer than 60 non-NaN observations
4. Auto-computes clip ranges from the 1st–99th percentile of each series' historical values (± 10% buffer)
5. Trains and forecasts using the same `macro_utils` pipeline as the FRED models
6. Saves per-source results JSONs that feed directly into the summary table

**New series** (a CSV without a corresponding `.joblib`) trigger automatic model creation on the next `fred_refresh.py --sector` run.

---

## Shared Utilities — macro_utils.py

`macro_utils.py` is imported by all model scripts and `sector_model.py`. Key functions:

### `engineer_features(df, series_cols)`

Adds all lag, rolling, momentum, YoY, seasonality, and cross-term features to `df`. Returns a new DataFrame with the original columns plus all engineered columns.

### `fit_model(X_tr, y_tr, X_vl, y_vl, objective, alpha)`

Trains a single `lgb.LGBMRegressor` with early stopping. `objective` is `"regression"` for the median model or `"quantile"` with `alpha=0.10 / 0.90` for the interval models.

### `train_series_models(series_cols, X_tr, y_tr_dict, X_vl, y_vl_dict)`

Trains three regressors (mid, lo, hi) for every series in `series_cols`. Returns `{col: {"mid": model, "lo": model, "hi": model}}`.

### `joint_recursive_forecast(df_base, models, feature_cols, series_cols, clip_ranges, horizon=12)`

Generates the 12-month forecast as described above. Returns `{col: DataFrame(dates, mid, lo, hi)}`.

### `save_model_results(group_name, df_hist, fc_dict, y_val_dict, val_preds_dict, series_info, save_path)`

Serializes group results to JSON. The output file is what `fred_refresh.py` reads to build the summary table.

### Plotting Functions

| Function | Output |
|---|---|
| `plot_forecast_dashboard` | Historical data (last 10 years) + 12-month forecast bands per series |
| `plot_validation_performance` | Time series and scatter (actual vs predicted) on the 24-month holdout |
| `plot_feature_importance` | Horizontal bar chart of top-12 features by LightGBM Gain |

---

## Summary Table Output

After every run, a formatted table is printed to stdout. The title reflects which modules were active:

| Flags used | Table title |
|---|---|
| *(none)* | `MACRO MODEL SUMMARY — FRED Weekly Refresh` |
| `--sector` | `MACRO + SECTOR MODEL SUMMARY — FRED Weekly Refresh` |
| `--crunchbase` | `MACRO + VC MODEL SUMMARY — FRED Weekly Refresh` |
| `--sector --crunchbase` | `MACRO + SECTOR + VC MODEL SUMMARY — FRED Weekly Refresh` |

VC rows appear in the table only after `vc_model.py` has successfully trained (requires ~13 months of weekly data). During the accumulation period the table shows FRED and sector rows only.

```
==============================================================================================================================
  MACRO + SECTOR + VC MODEL SUMMARY — FRED Weekly Refresh
  Generated: 2026-04-08 09:15:22
==============================================================================================================================
╭──────────────────────────┬────────────────┬────────────────────────────┬──────────┬──────────┬───────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬───────╮
│ Group                    │ Series         │ Label                      │ Unit     │ Last Date│ Last Value│ Val MAE  │ Val R²   │ +1M      │ +3M      │ +6M      │ +12M     │ Trend │
├──────────────────────────┼────────────────┼────────────────────────────┼──────────┼──────────┼───────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────┤
│ BLS Employment           │ CES3000000001  │ Manufacturing Employment   │          │ 2026-03  │ 12,930.00 │ 42.320   │ 0.8811   │ 12,944   │ 12,921   │ 12,893   │ 12,847   │  ↓   │
│ Business Environment     │ INDPRO         │ Industrial Production      │ Index .. │ 2026-02  │ 102.55    │ 0.856    │ -0.5213  │ 101.29   │ 101.12   │ 100.87   │ 100.41   │  ↓   │
│ Business Environment     │ PAYEMS         │ Nonfarm Payroll            │ K persons│ 2026-03  │ 159,369.00│ 158.303  │ 0.9923   │ 159,465  │ 159,615  │ 159,822  │ 160,184  │  ↑   │
...
╰──────────────────────────┴────────────────┴────────────────────────────┴──────────┴──────────┴───────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴───────╯

  Columns: +1M/+3M/+6M/+12M = forecast midpoints at those horizons
  Trend  : ↑ rising  ↓ falling  → stable (<0.5% change over 12 months)
  Val R² : negative values are normal for trending non-stationary series
           (MAE is the more meaningful metric for those)
```

The table rows are sorted alphabetically by Group, then Series. All FRED groups and sector groups appear together.

---

## TEP Fault Detection — train.py

A standalone benchmark (independent of the FRED pipeline) that trains and compares four binary classifiers on the Tennessee Eastman Process dataset.

### Dataset

The TEP dataset simulates a chemical plant with 52 measured variables. The task is to detect whether the plant is in a faulty state. Four large CSV files in the `data/` directory provide fault-free and faulty observations for training and testing. LightGBM evaluation additionally filters to faulty samples with `sample ≥ 160`, matching the TEP benchmark standard for post-fault-onset detection.

### Models Compared

| Model | Key Settings |
|---|---|
| Logistic Regression | `solver=saga`, `class_weight=balanced`, StandardScaler |
| Random Forest | 100 trees, `max_depth=20`, `class_weight=balanced_subsample` |
| **LightGBM** | 1,000 estimators, `num_leaves=63`, `scale_pos_weight`, early stopping |
| MLP (Neural Network) | Layers (128, 64), `solver=adam`, early stopping |

### Metrics Reported

- Accuracy
- Fault Detection Rate (recall on faulty class)
- False Alarm Rate (FP / (FP + TN))
- Precision, F1 Score, ROC-AUC
- Training time (seconds)

### Running

```bash
python3 train.py
```

Outputs: `outputs/{metrics_comparison,roc_curves,confusion_matrices,training_times}.png` and `outputs/lgbm_model.joblib`.

---

## Output Files Reference

### Model Files (`outputs/`)

| File | Description |
|---|---|
| `business_env_model_{INDPRO,TCU,PAYEMS}.joblib` | Business Environment LightGBM regressors |
| `consumer_demand_model_{series}.joblib` (×6) | Consumer Demand LightGBM regressors |
| `cost_of_capital_model_{DFF,DPRIME,T10Y3M,T10Y2Y}.joblib` | Cost of Capital LightGBM regressors |
| `risk_model_{recpro,sentiment}.joblib` | Risk model LightGBM regressors |
| `sector_{source}_model_{series}.joblib` | Sector model regressors (created on first sector run) |
| `vc_model_{segment}_{metric}.joblib` | VC segment model regressors (created after ~13 months of data collection) |
| `lgbm_model.joblib` | TEP fault detection LightGBM classifier |

All models are saved with `joblib.dump()` and can be loaded with `joblib.load()`.

### Results JSONs (`outputs/`)

| File | Contents |
|---|---|
| `results_business_env.json` | Forecasts + validation metrics for Business Environment |
| `results_consumer_demand.json` | Forecasts + validation metrics for Consumer Demand |
| `results_cost_of_capital.json` | Forecasts + validation metrics for Cost of Capital |
| `results_risk.json` | Forecasts + validation metrics for Risk & Leading Indicators |
| `results_sector_bls.json` | BLS Employment forecasts (created after first `--sector bls` run) |
| `results_sector_bea.json` | BEA GDP forecasts (created after first `--sector bea` run) |
| `results_sector_worldbank.json` | World Bank sector forecasts (created after first `--sector worldbank` run) |
| `results_vc_ai.json` | AI VC segment forecasts (created after ~13 months of `--crunchbase` runs) |
| `results_vc_fintech.json` | Fintech VC segment forecasts |
| `results_vc_healthcare.json` | Healthcare VC segment forecasts |

Results JSON structure:
```json
{
  "group": "Business Environment",
  "run_at": "2026-04-08T09:15:22",
  "series": [
    {
      "series_id": "INDPRO",
      "label": "Industrial Production",
      "unit": "Index 2017=100",
      "last_date": "2026-02",
      "last_value": 102.551,
      "validation": { "mae": 0.856, "rmse": 1.037, "r2": -0.521 },
      "forecast": [
        { "month": "2026-03", "mid": 101.29, "lo": 99.33, "hi": 101.84 },
        ...12 months total...
      ]
    }
  ]
}
```

### Run Log (`outputs/refresh_log.json`)

Appended after every `fred_refresh.py` run. The last 52 entries are kept.

```json
[
  {
    "run_at": "2026-04-08T09:15:22",
    "series_ok": 18,
    "series_errors": 0,
    "models_ok": 4,
    "models_failed": 0,
    "sector_sources_enabled": ["bls"],
    "sector_series_refreshed": 6,
    "sector_model_ok": true,
    "crunchbase_enabled": true,
    "crunchbase_segments_refreshed": 3,
    "vc_model_ok": null,
    "refresh_details": [...],
    "sector_refresh_details": [...],
    "crunchbase_refresh_details": [...]
  }
]
```

`vc_model_ok` is `null` when `--crunchbase` was not used, `false` when the model script was skipped (insufficient data), and `true` once models have trained successfully.

### Visualizations (`outputs/`)

| File | Description |
|---|---|
| `business_env_dashboard.png` | 12-month forecasts with 80% bands for INDPRO, TCU, PAYEMS |
| `business_env_validation.png` | Actual vs predicted on 24-month hold-out |
| `business_env_importance.png` | Top-12 feature importance by Gain |
| `consumer_demand_*.png` | Same three plots for Consumer Demand (2-column layout) |
| `cost_of_capital_*.png` | Same three plots for Cost of Capital |
| `risk_forecast_dashboard.png` | Recession probability + consumer sentiment forecasts |
| `risk_validation.png` | Validation for risk series |
| `risk_feature_importance.png` | Top-15 features for risk models |
| `risk_snapshot.png` | Distribution gauges with current value and 12-month forecast |
| `sector_bls_*.png` | Dashboard, validation, importance for BLS series |
| `sector_bea_*.png` | Dashboard, validation, importance for BEA series |
| `sector_worldbank_*.png` | Dashboard, validation, importance for World Bank series |
| `vc_ai_*.png` | Dashboard, validation, importance for AI VC segment (after model trains) |
| `vc_fintech_*.png` | Dashboard, validation, importance for Fintech VC segment |
| `vc_healthcare_*.png` | Dashboard, validation, importance for Healthcare VC segment |
| `metrics_comparison.png` | TEP model comparison (grouped bar chart) |
| `roc_curves.png` | TEP ROC curves for all four classifiers |
| `confusion_matrices.png` | TEP confusion matrices |
| `training_times.png` | TEP training time comparison |

---

## Data Files Reference

### FRED Series Fetched by `fred_refresh.py`

| Series ID | Description | Frequency | Data Directory |
|---|---|---|---|
| INDPRO | Industrial Production Index | Monthly | BusinessEnvironment |
| TCU | Total Capacity Utilization | Monthly | BusinessEnvironment |
| CAPUTLB50001SQ | Mfg Capacity Utilization | Quarterly | BusinessEnvironment |
| PAYEMS | Nonfarm Payroll | Monthly | BusinessEnvironment |
| DSPIC96 | Real Disposable Personal Income | Monthly | ConsumerDemand |
| PCE | Personal Consumption Expenditures | Monthly | ConsumerDemand |
| PCEPILFE | Core PCE Price Index | Monthly | ConsumerDemand |
| RSAFS | Retail & Food Services Sales | Monthly | ConsumerDemand |
| RRSFS | Real Retail & Food Services Sales | Monthly | ConsumerDemand |
| UMCSENT | U. of Michigan Consumer Sentiment | Monthly | ConsumerDemand + RiskLeadingInd |
| DFF | Effective Federal Funds Rate | Daily | CostOfCapital |
| DPRIME | Bank Prime Loan Rate | Daily | CostOfCapital |
| FEDFUNDS | Federal Funds Rate | Monthly | CostOfCapital |
| PRIME | Prime Rate | Event | CostOfCapital |
| T10Y2Y | Yield Curve 10Y-2Y | Daily | CostOfCapital |
| T10Y3M | Yield Curve 10Y-3M | Daily | CostOfCapital |
| RECPROUSM156N | Chauvet-Piger Recession Probability | Monthly | RiskLeadingInd |

UMCSENT is saved to both `ConsumerDemand/` and `RiskLeadingInd/` on every refresh. Daily series (DFF, DPRIME, T10Y2Y, T10Y3M) are stored in daily format; model scripts resample to monthly means internally.

---

## Scheduling with Cron

To run the refresh automatically every Monday at 8 AM:

```cron
0 8 * * 1 cd /path/to/tep-ml && /path/to/python3 fred_refresh.py >> logs/refresh.log 2>&1
```

With sector APIs (BLS and World Bank are free):

```cron
0 8 * * 1 cd /path/to/tep-ml && /path/to/python3 fred_refresh.py --sector bls worldbank >> logs/refresh.log 2>&1
```

With Crunchbase VC data:

```cron
0 8 * * 1 cd /path/to/tep-ml && /path/to/python3 fred_refresh.py --crunchbase >> logs/refresh.log 2>&1
```

Full pipeline — FRED + free sector APIs + Crunchbase VC:

```cron
0 8 * * 1 cd /path/to/tep-ml && /path/to/python3 fred_refresh.py --sector bls worldbank --crunchbase >> logs/refresh.log 2>&1
```

FRED data typically releases on weekday mornings. Running on Monday morning captures most prior-week releases. The Crunchbase snapshot date is pinned to the Monday of the current ISO week, so the job is idempotent if re-run later in the same week.

---

## Adding New Series

### Adding a new FRED series

1. Add the series to `fred_ingestion_map_full_production.json` under `"series"`.
2. Add an entry to `SERIES_FILE_MAP` in `fred_refresh.py` mapping `series_id → (directory, filename)`.
3. Add the series to the appropriate model script (`PREDICT_COLS` or as a feature-only column) and define a clip range.

### Adding a new sector data source

1. Implement a `fetch_*` function and a `refresh_*` orchestrator in `sector_apis.py` following the BLS/BEA/World Bank patterns.
2. Add an entry to `SOURCE_CONFIG` in `sector_model.py`.
3. Wire the new source into `refresh_sector_data()` in `fred_refresh.py`.

### Adding a new Crunchbase VC segment

1. Add the new segment key and its category resolution queries to `SEGMENTS` and `SEGMENT_QUERIES` in `crunchbase_apis.py`.
2. Add a corresponding entry to `VC_SOURCE_CONFIG` in `vc_model.py` with a `csv_file`, `label_map`, `results_file`, `model_prefix`, and `plot_prefix`.
3. The next `fred_refresh.py --crunchbase` run will create the new `agg_{segment}_weekly.csv` automatically. Models are created once ≥ 54 monthly rows have accumulated.

### Adding a completely new model group

Create a new model script following the 8-step structure of `business_env_model.py`:
1. Load CSVs
2. Merge on date
3. Engineer features (`macro_utils.engineer_features`)
4. Train/val split
5. Train (`macro_utils.train_series_models`)
6. Validate
7. Forecast (`macro_utils.joint_recursive_forecast`)
8. Save models + plots + results JSON (`macro_utils.save_model_results`)

Then add the script to `MODEL_SCRIPTS` and its results file to `RESULTS_FILES` in `fred_refresh.py`.
