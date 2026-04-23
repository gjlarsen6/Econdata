# econdata — Macroeconomic Forecasting & Process Monitoring

A production-ready pipeline that refreshes U.S. economic data from public APIs, retrains LightGBM forecasting models, and prints a unified summary table of 12-month outlooks — designed to run weekly with a single command.

**Phase 0** extensions (no additional API keys required) add market risk indicators (VIX, credit spreads, USD index), commodity prices (WTI oil, gold), the full Treasury yield curve (8 tenors), and a composite Financial Stress Index with market regime classification — all sourced from FRED.

**Phase 1** adds a financial news ingestion pipeline — daily briefings, top-stories ranking, and threshold-based impact alerts — sourced from NewsAPI, Marketaux, and Finnhub. At least one news API key is required; all three have free tiers. Enable with `--news daily`.

**Phase 2** adds yfinance/FMP article enrichment (`enrichment_apis.py`) and a LightGBM ML layer (`news_model.py`) that trains on 30+ days of Phase 1 data to forecast sector sentiment and article volume trends. Enable enrichment with `--enrich`; run `news_model.py` directly after data has accumulated.

Optional modules extend coverage to industry-level employment and GDP data (BLS, BEA, World Bank) and Venture Capital activity by sector (Crunchbase — AI, Fintech, Healthcare).

---

## Usage

Quick-reference workflow — run these scripts in order for a full data refresh, model build, and insight query cycle.

### Data Refresh & Model Training

Steps 1–5 are **flags on the same script** and can be combined into a single call. Only `news_model.py` (step 6) must be run as a separate command.

| Step | Flag / Command | What it adds | Frequency |
|------|----------------|--------------|-----------|
| 1 | `python3 fred_refresh.py` | *(base)* Fetch all FRED macro series + retrain core LightGBM models (7 groups) | Weekly |
| 2 | `+ --sector` | Also fetch BLS/BEA/World Bank sector data + retrain sector models *(optional)* | Weekly |
| 3 | `+ --crunchbase` | Also fetch Crunchbase VC data + retrain VC model *(optional, requires API key)* | Weekly |
| 4 | `+ --news daily` | Also ingest latest financial news articles (Phase 1) *(requires ≥1 news API key)* | Daily |
| 5 | `+ --enrich` | Also enrich news articles with yfinance/FMP signals (Phase 2) *(optional)* | Daily |
| 6 | `python3 news_model.py` | Separate script — train sentiment & volume forecast models on 30+ days of news data (Phase 2) | Weekly (after ≥30 days of news) |

> **Full refresh in one command** (combine any flags you need):
> ```bash
> python3 fred_refresh.py --sector --crunchbase --news daily --enrich && python3 news_model.py
> ```

### Generate Reports

| Step | Command | Purpose |
|------|---------|---------|
| 7 | `python3 reports.py` | Generate a timestamped Markdown forecast report in `reports/` |

### Query Models & Get Insights

| Step | Command | Purpose |
|------|---------|---------|
| 8 | `python3 api.py` | Start the REST API server on `http://localhost:8100` |
| 9 | `curl http://localhost:8100/api/summary` | List all available endpoints and their status |
| 10 | `curl http://localhost:8100/api/macro/{group_id}` | Fetch 12-month forecast for a core macro group |
| 11 | `curl http://localhost:8100/api/macro/{group_id}/{series_id}` | Fetch a single macro series forecast |
| 12 | `curl http://localhost:8100/api/financial-news/briefing/latest` | Get today's AI-generated news briefing (Phase 1) |
| 13 | `curl http://localhost:8100/api/financial-news/alerts` | Get threshold-based news impact alerts (Phase 1) |
| 14 | `curl http://localhost:8100/api/financial-news/sentiment` | Get 12-month sector sentiment forecast (Phase 2) |
| 15 | `curl http://localhost:8100/api/financial-news/volume` | Get 12-month article volume forecast by sector (Phase 2) |
| 16 | `python3 test_api.py` | Run full API test suite to verify all endpoints |

> **Tip:** Replace `{group_id}` with IDs like `business_environment`, `consumer_demand`, `cost_of_capital`, `market_risk`, etc. Replace `{series_id}` with a series column name (e.g., `INDPRO`, `MACRO_SENT`). See [REST API — api.py](#rest-api--apipy) for the full endpoint list.

---

## Table of Contents

1. [Usage](#usage)
2. [Project Overview](#project-overview)
3. [Directory Structure](#directory-structure)
4. [Setup](#setup)
5. [Running the Weekly Refresh](#running-the-weekly-refresh)
6. [Sector API Integration](#sector-api-integration)
7. [Crunchbase VC Integration](#crunchbase-vc-integration)
8. [LightGBM Forecasting Models](#lightgbm-forecasting-models)
9. [Model Scripts](#model-scripts)
10. [Shared Utilities — macro_utils.py](#shared-utilities--macro_utilspy)
11. [Summary Table Output](#summary-table-output)
12. [Reports — reports.py](#reports--reportspy)
13. [REST API — api.py](#rest-api--apipy)
14. [API Testing](#api-testing)
15. [TEP Fault Detection — train.py](#tep-fault-detection--trainpy)
16. [Output Files Reference](#output-files-reference)
17. [Data Files Reference](#data-files-reference)
18. [Scheduling with Cron](#scheduling-with-cron)
19. [Adding New Series](#adding-new-series)
20. [Phase Roadmap](#phase-roadmap)

---

## Project Overview

The macroeconomic side of this project answers one question every week: **where are key economic indicators heading over the next 12 months?** It does this by:

1. Fetching the latest observations from the FRED API (and optionally from BLS, BEA, and World Bank)
2. Retraining gradient-boosted tree models on the updated data
3. Generating 12-month recursive forecasts with 80% prediction intervals
4. Displaying everything in a single tabular summary

Eight model groups cover distinct areas of the economy:

| Group | Series Modeled | Source Directory |
|---|---|---|
| Business Environment | Industrial production, capacity utilization, nonfarm payroll | `data/BusinessEnvironment/` |
| Consumer Demand | Disposable income, PCE, core inflation, retail sales, consumer sentiment | `data/ConsumerDemand/` |
| Cost of Capital | Fed funds rate, prime rate, yield curve spreads | `data/CostOfCapital/` |
| Risk & Leading Indicators | Recession probability, consumer sentiment | `data/RiskLeadingInd/` |
| Market Risk *(Phase 0)* | VIX, HY credit spread, IG credit spread, USD index | `data/MarketRisk/` |
| Commodities *(Phase 0)* | WTI crude oil, gold | `data/Commodities/` |
| Yield Curve *(Phase 0)* | 8 Treasury tenors: 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y | `data/YieldCurve/` |
| Industrial / ISM / Credit *(Phase 1)* | IP sector breakdowns, ISM PMI sub-indices, sector capacity utilization, commercial loans, PPI | `data/IndustrialProduction/`, `data/ISMIndicators/`, `data/CapacityUtilSector/`, `data/CreditIndicators/` |

Phase 0 also produces a **Financial Stress Index (FSI)** and a **Market Regime** label (`expansion / slowdown / contraction / stress / recovery`) from `composite_model.py` using the expanded FRED data — no additional API keys required.

Optional sector modules extend coverage to industry-level data from BLS (10 employment sectors, hourly earnings by sector, weekly hours by sector, JOLTS job openings by sector), BEA (10 GDP-by-industry series), World Bank (sector shares of GDP), and S&P 500 sector ETFs via yfinance.

An additional Crunchbase module (`--crunchbase`) tracks weekly VC investment activity across three segments — AI, Fintech, and Healthcare — building a time-series dataset suitable for LightGBM forecasting once sufficient history accumulates.

---

## Directory Structure

```
econdata/
│
├── fred_refresh.py            # Main orchestrator — run this weekly
├── sector_apis.py             # Sector API fetchers (BLS, BEA, World Bank, ETFs, Trading Economics)
├── sector_model.py            # Generic LightGBM trainer for sector data (BLS, BEA, WB, wages, hours, JOLTS, ETFs)
├── industrial_model.py        # Phase 1: LightGBM trainer for IP, ISM PMI, capacity util, credit/PPI
├── crunchbase_apis.py         # Crunchbase VC data fetcher (AI, Fintech, Healthcare)
├── vc_model.py                # LightGBM trainer for VC segment weekly data
├── macro_utils.py             # Shared feature engineering, training, forecasting, plotting
├── business_env_model.py      # Business Environment model (INDPRO, TCU, PAYEMS)
├── consumer_demand_model.py   # Consumer Demand model (6 series)
├── cost_of_capital_model.py   # Cost of Capital model (DFF, DPRIME, T10Y3M, T10Y2Y)
├── risk_model.py              # Risk model (recession probability, consumer sentiment)
├── market_model.py            # Phase 0: Market Risk (VIX, HY/IG spreads, USD) + Commodities (WTI, gold)
├── yield_curve_model.py       # Phase 0: 8-tenor Treasury yield curve (DGS1MO → DGS30)
├── composite_model.py         # Phase 0: Financial Stress Index + Market Regime classifier
├── news_apis.py               # Phase 1: News ingestion + normalization (NewsAPI, Marketaux, Finnhub)
├── briefing.py                # Phase 1: Daily briefing generator + rule-based impact scoring
├── enrichment_apis.py         # Phase 2: yfinance + FMP article enrichment (--enrich flag)
├── news_model.py              # Phase 2: Monthly sentiment/volume aggregation + LightGBM forecasting
├── data_summary.py            # Data inventory and feature engineering recommendations
├── api.py                     # Read-only REST API server (FastAPI, port 8100)
├── test_api.py                # Comprehensive API test suite (40+ checks incl. Phase 0 + security)
├── test_api_quick.py          # Quick smoke test (5 checks, one line each)
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
│   ├── MarketRisk/            # VIXCLS, BAMLH0A0HYM2, BAMLC0A0CM, DTWEXBGS CSVs (Phase 0)
│   ├── Commodities/           # DCOILWTICO, GOLDAMGBD228NLBM CSVs (Phase 0)
│   ├── YieldCurve/            # DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS5, DGS10, DGS30 CSVs (Phase 0)
│   ├── SectorAPIs/            # JSON API documentation for each sector data source
│   ├── IndustrialProduction/  # IPMAN, IPUTIL, IPMINE, IPCONGD, IPBUSEQ, IPMAT, IPDCONGD, IPNCONGD (Phase 1)
│   ├── ISMIndicators/         # NAPM, NMFCI, NAPMPROD, NAPMNEWO, NAPMEMPL, NAPMVNDR (Phase 1)
│   ├── CapacityUtilSector/    # MCUMFN, CAPUTLG211S, CAPUTLB58SQ (Phase 1)
│   ├── CreditIndicators/      # BUSLOANS, REALLN, CONSUMER, WPU05, WPU10 (Phase 1)
│   ├── Sector/                # Created automatically by sector refresh
│   │   ├── BLS/               #   10 employment series by NAICS sector (monthly)
│   │   ├── BEA/               #   10 GDP-by-industry series (quarterly)
│   │   ├── WorldBank/         #   Sector % of GDP (annual → monthly)
│   │   ├── BLS_Wages/         #   Avg hourly earnings by sector — 6 series (--sector bls_wages)
│   │   ├── BLS_Hours/         #   Avg weekly hours by sector — 3 series (--sector bls_hours)
│   │   ├── JOLTS/             #   Job openings by sector — 7 series (--sector jolts)
│   │   └── ETF/               #   S&P 500 sector ETF monthly close — 11 tickers (--sector etf)
│   ├── FinancialNews/         # Created automatically by --news refresh (Phase 1)
│   │   ├── raw/
│   │   │   └── news_articles.csv        # append-only master — all articles, deduped by normalized URL
│   │   ├── daily/
│   │   │   └── {YYYY-MM-DD}.csv         # per-day partition written after each refresh
│   │   └── enriched/
│   │       └── news_enriched.csv        # Phase 2: articles + yfinance/FMP signals (written by --enrich)
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
    ├── regime_history.json    # FSI time series + current regime label (Phase 0)
    ├── daily_briefing_{YYYY-MM-DD}.json  # Daily news briefing (Phase 1, written by briefing.py)
    ├── results_news_sentiment.json       # Phase 2: Sector sentiment forecasts (or cold_start status)
    ├── results_news_volume.json          # Phase 2: Article volume forecasts (or cold_start status)
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
pip install lightgbm pandas numpy scikit-learn matplotlib requests tabulate joblib python-dotenv fastapi uvicorn
pip install filelock        # Phase 1: prevents CSV corruption when daily + realtime crons overlap
pip install newsapi-python  # Phase 1: official NewsAPI client (fetch_newsapi uses NewsApiClient)
pip install yfinance        # Phase 2: article enrichment with market signals (no API key required)
```

**newsapi-python** replaces raw `requests` calls to NewsAPI. `news_apis.py` imports it as:

```python
from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key="YOUR_KEY")

# Fetch everything matching a query from a given date
result = newsapi.get_everything(
    q="federal reserve",
    from_param="2024-01-01T00:00:00Z",
    language="en",
    sort_by="publishedAt",
    page_size=10,
)
articles = result.get("articles", [])

# Fetch top headlines
top = newsapi.get_top_headlines(category="business", language="en", country="us")

# List available sources
sources = newsapi.get_sources()
```

If `newsapi-python` is not installed, `news_apis.py` automatically falls back to raw `requests` calls so the pipeline continues to work.

### API Key Configuration

Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```ini
# Mandatory
FRED_API_KEY=your_fred_api_key_here

# Phase 0 — no additional keys required (uses FRED_API_KEY above)

# Optional — only needed with the --sector flag
BEA_API_KEY=your_bea_api_key_here
BLS_API_KEY=your_bls_api_key_here       # BLS works without a key at lower limits
TE_CLIENT_KEY=your_te_client_key_here
TE_CLIENT_SECRET=your_te_client_secret_here

# Optional — only needed with the --crunchbase flag
CRUNCHBASE_API_KEY=your_crunchbase_api_key_here

# Phase 1 — at least one required for --news flag
NEWS_API_KEY=your_newsapi_key_here
MARKETAUX_API_KEY=your_marketaux_key_here
FINNHUB_API_KEY=your_finnhub_key_here

# Phase 2 — financial data enrichment (yfinance needs no key; FMP is optional supplement)
FMP_API_KEY=your_fmp_key_here
```

**Where to get keys:**

| Key | Source | Cost | Required For |
|---|---|---|---|
| `FRED_API_KEY` | https://fred.stlouisfed.org/docs/api/api_key.html | Free | Everything (mandatory) |
| `BEA_API_KEY` | https://apps.bea.gov/api/signup/ | Free | `--sector bea` |
| `BLS_API_KEY` | https://www.bls.gov/developers/home.htm | Free (optional) | `--sector bls` (works without key at lower limits) |
| `TE_CLIENT_KEY/SECRET` | https://tradingeconomics.com/api/ | Commercial | `--sector tradingeconomics` |
| `CRUNCHBASE_API_KEY` | https://data.crunchbase.com/docs/welcome-to-crunchbase-data | Paid | `--crunchbase` |
| `NEWS_API_KEY` | https://newsapi.org/register | Free tier available | Phase 1 `--news` — at least one required |
| `MARKETAUX_API_KEY` | https://www.marketaux.com/ | Free tier available | Phase 1 `--news` — optional additional source |
| `FINNHUB_API_KEY` | https://finnhub.io/register | Free tier available | Phase 1 `--news` — also used for intraday realtime updates |
| `FMP_API_KEY` | https://financialmodelingprep.com/developer/docs/ | Free tier available | Phase 2 `--enrich` — optional FMP fundamentals supplement (yfinance runs without this key) |

You may also export keys directly as environment variables instead of using `.env`.

---

## Running the Weekly Refresh

```bash
python3 fred_refresh.py
```

This single command runs the full pipeline:

1. Fetches the latest observations for all 53 FRED series (18 core + 14 Phase 0 + 22 Phase 1 industrial/ISM/credit)
2. Merges new rows into existing CSVs (never overwrites historical data)
3. Retrains all eight LightGBM model groups (4 core + 3 Phase 0 + 1 Phase 1 industrial)
4. Prints a **Market Conditions Snapshot** (VIX, FSI, regime, WTI, gold, 10Y−2Y slope)
5. Prints a unified summary table with forecasts and validation metrics
6. Appends a run log entry to `outputs/refresh_log.json`

### Command-Line Options

```
python3 fred_refresh.py [--sector SOURCE ...] [--crunchbase] [--news MODE] [--skip-models]
```

| Option | Description |
|---|---|
| *(no flags)* | FRED data refresh + all eight model retrains (includes industrial_model.py) |
| `--sector bls` | Also refresh BLS employment data (10 sectors) and train BLS models |
| `--sector bea` | Also refresh BEA GDP-by-industry data (10 industries; requires `BEA_API_KEY`) |
| `--sector worldbank` | Also refresh World Bank sector indicators |
| `--sector bls_wages` | Also refresh BLS avg hourly earnings by sector (6 series) |
| `--sector bls_hours` | Also refresh BLS avg weekly hours by sector (3 series) |
| `--sector jolts` | Also refresh JOLTS job openings by sector (7 series) |
| `--sector etf` | Also refresh S&P 500 sector ETF monthly prices via yfinance (11 tickers; no API key required) |
| `--sector bls bea worldbank` | Multiple sources in one run |
| `--sector all` | All available sector sources (BLS, BEA, WorldBank, wages, hours, JOLTS, ETF) |
| `--crunchbase` | Also refresh Crunchbase VC data (AI, Fintech, Healthcare) and train VC models (requires `CRUNCHBASE_API_KEY`) |
| `--news daily` | Ingest news from all configured sources (once/day); generate daily briefing JSON *(Phase 1)* |
| `--news realtime` | Ingest Finnhub only — safe to run every 15 min as a cron *(Phase 1)* |
| `--news all` | All sources with 7-day backfill window — use for manual catch-up runs *(Phase 1)* |
| `--enrich` | Enrich articles with yfinance signals (PE, momentum, 52w range) + FMP fundamentals for S&P 500 tickers — writes `news_enriched.csv` *(Phase 2)* |
| `--skip-models` | Refresh data only — skip all model retraining and briefing generation |

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

# Phase 1: Daily news pull (requires at least one news API key)
python3 fred_refresh.py --news daily

# Phase 1: Intraday Finnhub update only (safe for 15-min cron)
python3 fred_refresh.py --news realtime

# Phase 1: One-time backfill — all sources, 7-day lookback, no model retraining
python3 fred_refresh.py --news all --skip-models

# Full weekly pipeline: FRED + news
python3 fred_refresh.py --news daily

# Full pipeline: FRED + sector + VC + news
python3 fred_refresh.py --sector all --crunchbase --news daily

# Phase 2: Enrich today's articles with yfinance signals
python3 fred_refresh.py --news daily --enrich

# Phase 2: Train sentiment/volume ML models (after ≥30 days of --news daily runs)
python3 news_model.py
```

### What the Pipeline Does Step by Step

The step count shown in the log (`[1/N]`) adjusts automatically based on which flags are active: each of `--sector` and `--crunchbase` adds 2 steps (data refresh + model training).

```
[1/N]  Refresh FRED data (53 series — 18 core + 14 Phase 0 + 22 Phase 1 industrial/ISM/credit)
         → Incremental fetch: only pulls data since last run (30-day lookback for revisions)
         → Rate-limited: 0.6 s between API calls (stays under 120 req/min)
         → Prints a per-series status table

[2/N]  Refresh sector API data  (only with --sector)
         → BLS employment: POST multi-series for 10 NAICS employment sectors
         → BLS wages: POST avg hourly earnings by sector (6 series; --sector bls_wages)
         → BLS hours: POST avg weekly hours by sector (3 series; --sector bls_hours)
         → JOLTS: POST job openings by sector (7 series; --sector jolts)
         → BEA: GET GDP by industry (all years, pivots to 10 per-industry CSVs)
         → World Bank: GET annual indicators, forward-fills to monthly
         → ETF: fetch monthly close prices for 11 S&P 500 sector ETFs via yfinance (--sector etf)
         → Prints a per-series status table

[3/N]  Refresh Crunchbase VC data  (only with --crunchbase)
         → Resolves category UUIDs for AI, Fintech, Healthcare via autocomplete
         → Searches funding rounds (trailing 90 days) and top organizations per segment
         → Computes weekly aggregate metrics; appends to agg_{segment}_weekly.csv
         → Also updates dim_category.csv, dim_organization.csv, fact_funding_round.csv
         → Rate-limited: 0.31 s between calls (~33 calls/run, under 200/min limit)
         → Prints a per-segment status table

[N-2]  Retrain FRED LightGBM models
         → Runs 8 model scripts sequentially as subprocesses:
              4 core + market_model, yield_curve_model, composite_model, industrial_model
         → Each script saves .joblib models, PNGs, and a results JSON
         → composite_model.py additionally writes regime_history.json
         → industrial_model.py runs all 4 groups (IP, ISM PMI, capacity util, credit/PPI);
              skips any group whose data directory is empty

[N-1]  Train sector models  (only with --sector + data exists)
         → Discovers all CSVs in data/Sector/{BLS,BEA,WorldBank,BLS_Wages,BLS_Hours,JOLTS,ETF}/
         → Auto-creates models for any new series without an existing .joblib

[N-1]  Train VC models  (only with --crunchbase + data exists)
         → Resamples weekly agg CSVs to monthly cadence for LightGBM compatibility
         → Skips gracefully if fewer than 54 monthly rows (needs ~13 months of history)
         → Once sufficient data exists: trains, saves .joblib and results_vc_*.json

[N]    Print output
         → Prints Market Conditions Snapshot (VIX, FSI, regime label, WTI, gold, 10Y−2Y slope)
         → Prints unified summary table from all results JSONs (FRED + sector + VC as available)
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

**No API key required** (a key raises rate limits but is not mandatory). Three subsets of BLS data are available, each enabled by a different `--sector` value.

#### Employment by Sector (`--sector bls`)

10 monthly employment series covering full NAICS private-sector scope:

| Series ID | Description |
|---|---|
| CES1000000001 | Mining & Logging Employment |
| CES2000000001 | Construction Employment |
| CES3000000001 | Manufacturing Employment |
| CES4000000001 | Trade, Transportation & Utilities Employment |
| CES5000000001 | Information Employment |
| CES5500000001 | Financial Activities Employment |
| CES6000000001 | Professional & Business Services Employment |
| CES9000000001 | Government Employment |
| CEU6500000001 | Education & Health Services Employment |
| CEU7000000001 | Leisure & Hospitality Employment |

Saved to: `data/Sector/BLS/<series_id>.csv`

#### Average Hourly Earnings by Sector (`--sector bls_wages`)

Sector-level wage data — a direct input-cost inflation signal. Series suffix `008`.

| Series ID | Description |
|---|---|
| CES2000000008 | Construction Avg Hourly Earnings |
| CES3000000008 | Manufacturing Avg Hourly Earnings |
| CES4000000008 | Trade/Transport Avg Hourly Earnings |
| CES5000000008 | Information Avg Hourly Earnings |
| CES5500000008 | Financial Avg Hourly Earnings |
| CES6000000008 | Professional Services Avg Hourly Earnings |

Saved to: `data/Sector/BLS_Wages/<series_id>.csv`

#### Average Weekly Hours by Sector (`--sector bls_hours`)

Hours lead payrolls — firms cut hours before headcount. Series suffix `007`.

| Series ID | Description |
|---|---|
| CES2000000007 | Construction Avg Weekly Hours |
| CES3000000007 | Manufacturing Avg Weekly Hours |
| CES6000000007 | Professional Services Avg Weekly Hours |

Saved to: `data/Sector/BLS_Hours/<series_id>.csv`

#### JOLTS — Job Openings by Sector (`--sector jolts`)

Job openings turn before payrolls — a forward-looking labor market signal.

| Series ID | Description |
|---|---|
| JTS2300JOL | Job Openings: Construction |
| JTS3000JOL | Job Openings: Manufacturing |
| JTS4000JOL | Job Openings: Trade/Transport |
| JTS5500JOL | Job Openings: Financial |
| JTS6000JOL | Job Openings: Professional Services |
| JTS6500JOL | Job Openings: Education & Health |
| JTS7000JOL | Job Openings: Leisure & Hospitality |

Saved to: `data/Sector/JOLTS/<series_id>.csv`

### BEA — Bureau of Economic Analysis

**Requires `BEA_API_KEY`.** Fetches quarterly GDP by industry in a single API call; all 10 industries are pivoted from one response at zero additional API cost:

| Column Name | Description |
|---|---|
| BEA_Agriculture | Gross Output — Agriculture |
| BEA_Utilities | Gross Output — Utilities |
| BEA_Construction | Gross Output — Construction |
| BEA_Manufacturing | Gross Output — Manufacturing |
| BEA_Wholesale_Retail_Trade | Gross Output — Wholesale & Retail Trade |
| BEA_Information | Gross Output — Information |
| BEA_Finance_Insurance_RE | Gross Output — Finance, Insurance & Real Estate |
| BEA_Professional_Biz_Svcs | Gross Output — Professional & Business Services |
| BEA_Healthcare | Gross Output — Healthcare & Social Assistance |
| BEA_Arts_Hospitality | Gross Output — Arts, Entertainment & Hospitality |

Saved to: `data/Sector/BEA/<industry>.csv` (quarterly dates)

### S&P 500 Sector ETFs (`--sector etf`)

**No API key required** (uses `yfinance`). Monthly close prices for all 11 GICS sector ETFs provide a market-implied, forward-looking view of sector strength that leads lagging government data by months.

| Ticker | Sector |
|---|---|
| XLK | Technology |
| XLF | Financials |
| XLV | Healthcare |
| XLE | Energy |
| XLI | Industrials |
| XLP | Consumer Staples |
| XLY | Consumer Discretionary |
| XLU | Utilities |
| XLRE | Real Estate |
| XLB | Materials |
| XLC | Communication Services |

Saved to: `data/Sector/ETF/<ticker>.csv`

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

### market_model.py *(Phase 0)*

Covers financial market risk indicators and commodity prices. All FRED series are daily; the script resamples to monthly means via `groupby("date").mean()`.

**Market Risk group** (saved to `results_market_risk.json`):

| Series | Description | Unit | Data Start |
|---|---|---|---|
| VIXCLS | CBOE Volatility Index (VIX) | Index | 1990-01 |
| BAMLH0A0HYM2 | ICE BofA US High Yield Option-Adjusted Spread | % | 1996-12 |
| BAMLC0A0CM | ICE BofA US Corporate (IG) Option-Adjusted Spread | % | 1996-12 |
| DTWEXBGS | Nominal Broad USD Index | Index | 2006-01 |

**Commodities group** (saved to `results_commodities.json`):

| Series | Description | Unit | Data Start |
|---|---|---|---|
| DCOILWTICO | WTI Crude Oil Price | $/bbl | 1986-01 |
| GOLDAMGBD228NLBM | Gold London Fixing Price | $/troy oz | 1968-01 |

Cross-features: `hy_ig_diff` (HY−IG spread), `vix_x_hy` (VIX × HY), `gold_oil_ratio`. Data window starts 1996-01 to align with credit spread availability.

**Outputs:** `market_model_{series}.joblib` (×6), `market_risk_{dashboard,validation,importance}.png`, `commodities_{dashboard,validation,importance}.png`, `results_market_risk.json`, `results_commodities.json`

---

### yield_curve_model.py *(Phase 0)*

Trains a joint LightGBM model on 8 Treasury yield tenors simultaneously. Daily FRED rates are resampled to monthly means.

| Series | Tenor | Data Start |
|---|---|---|
| DGS1MO | 1-month | 2001-07 |
| DGS3MO | 3-month | 1982-01 |
| DGS6MO | 6-month | 1982-01 |
| DGS1 | 1-year | 1962-01 |
| DGS2 | 2-year | 1976-06 |
| DGS5 | 5-year | 1962-01 |
| DGS10 | 10-year | 1962-01 |
| DGS30 | 30-year | 1977-02 |

Cross-features: `slope_10y_2y` (DGS10−DGS2), `curvature` (2×DGS5−DGS1−DGS10), `spread_10y_3m`, `spread_2y_1m`. The joint data window starts 2001-07 when DGS1MO becomes available.

**Outputs:** `yield_curve_model_{series}.joblib` (×8), `yield_curve_{dashboard,validation,importance}.png`, `results_yield_curve.json`

---

### composite_model.py *(Phase 0)*

Computes a **Financial Stress Index (FSI)** and classifies the current **Market Regime**. No additional FRED keys required — reads from files already written by the Phase 0 data fetch.

**FSI computation:**

FSI is the expanding-window percentile rank average of 5 components:

| Component | Source |
|---|---|
| VIX | `data/MarketRisk/VIXCLS.csv` |
| HY spread | `data/MarketRisk/BAMLH0A0HYM2.csv` |
| IG spread | `data/MarketRisk/BAMLC0A0CM.csv` |
| Recession probability | `data/RiskLeadingInd/RECPROUSM156N.csv` |
| Inverted yield curve | `data/CostOfCapital/T10Y2Y.csv` (negated) |

**Market Regime thresholds:**

| FSI Range | Trend Condition | Label |
|---|---|---|
| < 0.25 | any | `expansion` |
| 0.25 – 0.45 | any | `slowdown` |
| 0.45 – 0.65 | trending down | `recovery` |
| 0.45 – 0.65 | flat or rising | `contraction` |
| ≥ 0.65 | any | `stress` |

Trains a LightGBM regressor on the FSI time series for a 12-month forecast.

**Outputs:** `composite_model_fsi.joblib`, `composite_{dashboard,validation,importance}.png`, `results_financial_stress.json` (GroupResponse format with FSI series), `regime_history.json` (FSI history + current regime label + regime forecast)

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

### industrial_model.py *(Phase 1)*

Discovery-based LightGBM trainer for four groups of FRED-sourced industrial indicators. Runs automatically on every `fred_refresh.py` call (no flag required); results appear in the unified summary table. Can also be run directly:

```bash
python3 industrial_model.py --source industrial_production
python3 industrial_model.py --source ism_pmi capacity_util_sector
python3 industrial_model.py --source all
```

| Source Key | Data Directory | Series | API Endpoint |
|---|---|---|---|
| `industrial_production` | `data/IndustrialProduction/` | IPMAN, IPUTIL, IPMINE, IPCONGD, IPBUSEQ, IPMAT, IPDCONGD, IPNCONGD | `/api/industrial/production` |
| `ism_pmi` | `data/ISMIndicators/` | NAPM, NMFCI, NAPMPROD, NAPMNEWO, NAPMEMPL, NAPMVNDR | `/api/industrial/ism-pmi` |
| `capacity_util_sector` | `data/CapacityUtilSector/` | MCUMFN, CAPUTLG211S, CAPUTLB58SQ | `/api/industrial/capacity-utilization` |
| `credit_indicators` | `data/CreditIndicators/` | BUSLOANS, REALLN, CONSUMER, WPU05, WPU10 | `/api/industrial/credit` |

Groups skip gracefully if their data directory is empty. ISM New Orders (`NAPMNEWO`) leads GDP direction by 2–3 months and is the highest signal-to-noise sub-index.

**Outputs:** `industrial_{group}_model_{series}.joblib`, `industrial_{group}_{dashboard,validation,importance}.png`, `results_industrial_{group}.json`

---

### news_apis.py *(Phase 1)*

Fetches and normalizes financial news from up to three sources. Called by `fred_refresh.py` when `--news` is used, or from a cron directly.

**Sources and rate limit budget:**

| Source | Key | Free Tier | Mode |
|---|---|---|---|
| NewsAPI | `NEWS_API_KEY` | 100 req/day | `daily` / `all` — 6 queries × 10 articles |
| Marketaux | `MARKETAUX_API_KEY` | 100 req/day | `daily` / `all` — 3 ticker-group requests |
| Finnhub | `FINNHUB_API_KEY` | 60 req/min | all modes — single request per run |

At least one key is required. All sources with a key run automatically.

**Key functions:**

- `normalize_url(url)` — strips query params and lowercases scheme/host for reliable deduplication
- `classify_sector(headline, ticker, entities)` — keyword-match to 8 sectors; default `"macro"`
- `classify_macro_tag(headline)` — keyword-match to 8 macro tags (`rate_cuts`, `earnings`, `ipo`, etc.)
- `normalize_article(raw, source_api)` — maps source-specific fields to the standard 14-column schema
- `load_existing_urls(csv_path)` — returns set of normalized URLs from `news_articles.csv` for dedup
- `save_articles(articles, csv_path, daily_dir, date)` — appends new-only rows; uses `FileLock` when installed; writes daily partition CSV
- `refresh_news(api_keys, mode, since_hours)` — orchestrator; `all` mode uses 168h lookback

**`news_articles.csv` schema:** `timestamp`, `ingested_at`, `source_api`, `source_name`, `url`, `headline`, `summary`, `sector`, `ticker`, `entities` (JSON list), `sentiment`, `sentiment_label`, `macro_tag`, `market_impact_score` (backfilled by `briefing.py`)

---

### briefing.py *(Phase 1)*

Generates the daily briefing JSON. No ML required — works from day 1 with keyword-based impact scoring.

Run directly:
```bash
python3 briefing.py
python3 briefing.py --date 2026-04-14
```

**Market impact scoring** (`score_impact()`):

| Signal | Weight | Notes |
|---|---|---|
| Sentiment magnitude (`abs(score)`) | 25% | API-provided; 0 if missing |
| Source authority | 20% | Reuters/Bloomberg=1.0, default 0.5 |
| Ticker prominence | 20% | S&P 500=1.0, other ticker=0.5, none=0.2 |
| Volume spike in sector | 20% | >2σ above 30d avg; defaults to 0.5 if <7 days history |
| Macro tag type | 15% | `rate_cuts`/`earnings`=1.0, no tag=0.4 |

**Key functions:**

- `score_impact(df)` — adds `market_impact_score` column [0.0–1.0] to the article DataFrame
- `load_articles(csv_path, date)` — loads CSV, optionally filters to a single date; returns empty DataFrame if absent
- `compute_sector_mood(df)` — `{sector: avg_sentiment}` for non-null sentiment rows
- `generate_macro_signals(df, regime_path)` — up to 6 bullets in priority order: regime/FSI first, then tag volume, then sentiment outliers
- `generate_alerts(df, threshold=0.75)` — up to 5 high-impact headlines formatted as `"{source}: {headline}"`
- `generate_daily_briefing(...)` — full pipeline; calls `output_dir.mkdir(parents=True, exist_ok=True)`
- `main()` — CLI entry; writes `outputs/daily_briefing_{YYYY-MM-DD}.json`

**Outputs:** `outputs/daily_briefing_{YYYY-MM-DD}.json`

---

### enrichment_apis.py *(Phase 2)*

Fetches financial signals from yfinance and FMP for ticker-tagged articles. Called by `fred_refresh.py` when `--enrich` is used, or directly:

```bash
python3 enrichment_apis.py
python3 enrichment_apis.py --fmp-key YOUR_FMP_KEY
```

**Signal sources:**

| Source | Key | Signals Fetched |
|---|---|---|
| yfinance | None required | `trailing_pe`, `market_cap`, `eps_trailing`, `week52_high`, `week52_low`, `price_momentum_30d` |
| FMP | `FMP_API_KEY` | `ev_ebitda`, `revenue_growth_yoy`, `debt_to_equity` — S&P 500 tickers only, capped at 83/day (free tier) |

**Key functions:**

- `fetch_yfinance_signals(ticker)` — returns signal dict; sleeps 0.5 s between calls; returns `{}` on any exception
- `fetch_fmp_fundamentals(api_key, ticker)` — returns fundamentals dict from most recent annual row; returns `{}` if key missing or error
- `load_raw_articles()` — reads `news_articles.csv`; returns `[]` if file absent
- `enrich_articles(articles, api_keys)` — deduplicates tickers, fetches yfinance for all, FMP for S&P 500 only (budget-capped); merges signals back into each article
- `save_enriched(articles)` — writes `data/FinancialNews/enriched/news_enriched.csv` (full overwrite)

**Outputs:** `data/FinancialNews/enriched/news_enriched.csv`

---

### news_model.py *(Phase 2)*

Aggregates `news_articles.csv` to monthly metrics, trains LightGBM models on sector sentiment and article volume, and writes 12-month forecasts. Mirrors the `sector_model.py` structure exactly (uses the same `macro_utils` pipeline).

Run directly after ≥30 days of `--news daily` data:
```bash
python3 news_model.py
```

**Cold-start guard:** If fewer than `MIN_SENTIMENT_DAYS = 30` distinct calendar days are found in `news_articles.csv`, both results files are written with `{"status": "cold_start", "days_collected": N, "min_required": 30, "series": []}`. The API returns this status directly rather than a 404, so the endpoint is always available once `news_model.py` has run once.

**Monthly aggregation:** Daily articles → grouped by month → per-sector avg sentiment and article counts:

| Series | Description | Unit |
|---|---|---|
| `MACRO_SENT` | Avg monthly sentiment — macro sector | avg score [-1, 1] |
| `EQUITIES_SENT` | Avg monthly sentiment — equities sector | avg score [-1, 1] |
| `FINTECH_SENT` | Avg monthly sentiment — fintech sector | avg score [-1, 1] |
| `VC_SENT` | Avg monthly sentiment — VC sector | avg score [-1, 1] |
| `TOTAL_VOL` | Total article count per month | articles/month |
| `MACRO_VOL` | Macro sector article count per month | articles/month |
| `EQUITIES_VOL` | Equities sector article count per month | articles/month |
| `FINTECH_VOL` | Fintech sector article count per month | articles/month |

Clip ranges: sentiment columns are clipped to `[-1.0, 1.0]`; volume columns use the `(p1 × 0.9, p99 × 1.1)` formula.

**Outputs:** `outputs/results_news_sentiment.json`, `outputs/results_news_volume.json`

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

### Market Conditions Snapshot

Before the main table, a compact snapshot of Phase 0 market signals is printed:

```
══════════════════════════════════════════════════════════════════
  MARKET CONDITIONS SNAPSHOT
  VIX: 18.4   Dollar: 106.2
  FSI: 0.214 (expansion)   WTI: $72.4/bbl   Gold: $3,180/oz   10Y−2Y: +0.42%
══════════════════════════════════════════════════════════════════
```

All values are read from the most-recently written Phase 0 results files. The block is silently omitted if the Phase 0 models haven't been run yet.

### Main Summary Table

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

## REST API — api.py

`api.py` is a read-only FastAPI server that exposes all pre-computed LightGBM forecast results as queryable HTTP endpoints. It reads directly from the `outputs/results_*.json` files produced by the pipeline — no database, no retraining, no authentication required.

### Starting the Server

```bash
# Direct execution (recommended)
python3 api.py

# Development mode with auto-reload
uvicorn api:app --reload --port 8100 --no-server-header
```

The server binds to `127.0.0.1:8100` (localhost only). Interactive documentation is available at:
- **Swagger UI:** `http://localhost:8100/docs`
- **ReDoc:** `http://localhost:8100/redoc`

### Security Controls

The API enforces the following hardening measures:

| Control | Detail |
|---|---|
| **Localhost-only binding** | Server binds to `127.0.0.1` — not reachable from other hosts |
| **CORS policy** | Only `localhost` and `127.0.0.1` origins are allowed; all external origins are blocked |
| **Security headers** | Every response includes `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Cache-Control: no-store`, `Content-Security-Policy: default-src 'none'`, `Referrer-Policy: no-referrer` |
| **Server banner suppressed** | `server: uvicorn` header is not emitted |
| **Input validation** | `series_id` path parameters are validated against `^[A-Z0-9_]{1,30}$`; invalid inputs return HTTP 400 |
| **No internal error leakage** | Parse errors return a generic HTTP 500 message; full detail is logged server-side only |
| **Read-only** | No write, delete, or model-triggering endpoints exist |

### Response Format

All group endpoints return the same structure:

```json
{
  "group": "Business Environment",
  "run_at": "2026-04-08T08:46:30",
  "series_count": 3,
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

Single-series endpoints return the inner `series` object directly (without the group wrapper).

Optional endpoints (sector, VC) return HTTP 404 with an instructive message when their backing results file has not yet been generated, including the exact `fred_refresh.py` command needed to produce it.

### Endpoint Reference

#### Meta

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Liveness check — returns `{"status": "ok", "ts": "<ISO timestamp>"}`. Always available; does not require any results files. |
| GET | `/api/summary` | Lists every endpoint, its description, the series IDs it covers, and whether its backing results file currently exists on disk. Use this as a discovery endpoint. |
| GET | `/docs` | Interactive Swagger UI — try any endpoint directly from the browser |
| GET | `/redoc` | ReDoc documentation |

#### Core Model Endpoints (always available after pipeline run)

| Method | Path | Series | What It Returns |
|---|---|---|---|
| GET | `/api/business-env` | INDPRO, TCU, PAYEMS | 12-month LightGBM forecasts for Industrial Production Index (2017=100), Total Capacity Utilization (%), and Nonfarm Payroll Employment (thousands). Includes last observed value, 24-month MAE/RMSE/R² validation metrics, and monthly median + 80% prediction interval for 12 months. |
| GET | `/api/business-env/{series_id}` | — | Single series from Business Environment. Valid `series_id`: `INDPRO`, `TCU`, `PAYEMS` |
| GET | `/api/consumer-demand` | DSPIC96, PCE, PCEPILFE, RSAFS, RRSFS, UMCSENT | 12-month forecasts for Real Disposable Personal Income (B chained $), Personal Consumption Expenditures (B $), Core PCE Price Index (ex food & energy, 2017=100), Nominal Retail & Food Services Sales (M $), Real Retail & Food Services Sales (M chained $), and U. of Michigan Consumer Sentiment (1966:Q1=100). |
| GET | `/api/consumer-demand/{series_id}` | — | Single series from Consumer Demand. Valid `series_id`: `DSPIC96`, `PCE`, `PCEPILFE`, `RSAFS`, `RRSFS`, `UMCSENT` |
| GET | `/api/cost-of-capital` | DFF, DPRIME, T10Y3M, T10Y2Y | 12-month forecasts for the Federal Funds Effective Rate (%), Bank Prime Loan Rate (%), 10Y−3M Treasury yield-curve spread (%pts), and 10Y−2Y yield-curve spread (%pts). Yield-curve spreads are key recession leading indicators — negative values signal inversion. |
| GET | `/api/cost-of-capital/{series_id}` | — | Single series from Cost of Capital. Valid `series_id`: `DFF`, `DPRIME`, `T10Y3M`, `T10Y2Y` |
| GET | `/api/risk` | RECPROUSM156N, UMCSENT | 12-month forecasts for the Chauvet-Piger Smoothed Recession Probability (%) and U. of Michigan Consumer Sentiment (1966:Q1=100) as leading risk indicators. Recession probability above 20% is historically associated with elevated downturn risk. |
| GET | `/api/risk/{series_id}` | — | Single series from Risk. Valid `series_id`: `RECPROUSM156N`, `UMCSENT` |

#### Phase 0 Endpoints (available after standard pipeline run)

These return HTTP 404 until `fred_refresh.py` has completed a run that includes the Phase 0 model scripts (`market_model.py`, `yield_curve_model.py`, `composite_model.py`).

| Method | Path | Series | What It Returns |
|---|---|---|---|
| GET | `/api/market/vix` | VIXCLS | CBOE VIX — last value, 24-month validation, and 12-month forecast. |
| GET | `/api/market/spreads` | BAMLH0A0HYM2, BAMLC0A0CM | ICE BofA US High Yield and Investment Grade option-adjusted spreads (%). |
| GET | `/api/market/dollar` | DTWEXBGS | Nominal Broad USD Index — last value and 12-month forecast. |
| GET | `/api/commodities/oil` | DCOILWTICO | WTI crude oil price ($/bbl) — last value and 12-month forecast. |
| GET | `/api/commodities/gold` | GOLDAMGBD228NLBM | Gold London fixing price ($/troy oz) — last value and 12-month forecast. |
| GET | `/api/market/yield-curve` | DGS1MO … DGS30 | Full 8-tenor Treasury yield curve — last value and 12-month forecast per tenor. |
| GET | `/api/market/yield-curve/{series_id}` | — | Single tenor. Valid `series_id`: `DGS1MO`, `DGS3MO`, `DGS6MO`, `DGS1`, `DGS2`, `DGS5`, `DGS10`, `DGS30`. Returns HTTP 400 for invalid IDs. |
| GET | `/api/market/stress` | FSI | Financial Stress Index time series — last value and 12-month forecast. |
| GET | `/api/market/regime` | — | Current regime label, current FSI value, full FSI history, and 12-month regime forecast. See [Regime Response format](#regime-response-format) below. |
| GET | `/api/series/{series_id}/history` | — | Raw observation history for any of the 31 FRED series in the pipeline. Returns `series_id`, `row_count`, and `observations` list with `observation_date` + `value` pairs. Returns HTTP 404 for unknown IDs, HTTP 400 for invalid IDs. |

##### Regime Response Format

```json
{
  "current_regime": "expansion",
  "current_fsi": 0.214,
  "as_of": "2026-03",
  "history": [
    { "date": "1996-12", "fsi": 0.183, "regime": "expansion" },
    ...
  ],
  "forecast": [
    { "month": "2026-04", "fsi_mid": 0.219, "fsi_lo": 0.181, "fsi_hi": 0.258, "regime": "expansion" },
    ...12 months total...
  ]
}
```

---

#### Phase 1 Financial News Endpoints *(requires `--news daily` run)*

These return HTTP 404 until `fred_refresh.py --news daily` has completed at least once. All three return HTTP 500 with a log entry if the briefing JSON is corrupt. The `briefing` response includes a `stale: true` flag if the most recent briefing file is older than 36 hours (cron failure indicator).

| Method | Path | What It Returns |
|---|---|---|
| GET | `/api/financial-news/briefing` | Today's full briefing: date, article count, top 10 stories by impact score, sector mood scores (avg sentiment per sector), macro signal bullets (regime, tag volume, sentiment outliers), high-impact alerts, and `stale` flag. |
| GET | `/api/financial-news/top-stories` | Top 10 articles sorted by `market_impact_score` — includes headline, source, sector, ticker, sentiment, and score. |
| GET | `/api/financial-news/alerts` | Headlines where `market_impact_score ≥ 0.75`, formatted as `"{source}: {headline}"`, sorted by score. Returns `[]` if none exceed threshold. |

**Impact score formula** (rule-based, no ML required):

| Signal | Weight |
|---|---|
| Sentiment magnitude (`abs(score)`) | 25% |
| Source authority (Reuters/Bloomberg > blogs) | 20% |
| Ticker prominence (S&P 500 > unknown ticker > none) | 20% |
| Volume spike in sector (>2σ above 30d avg) | 20% |
| Macro tag type (`rate_cuts`/`earnings` > general) | 15% |

---

#### Phase 2 Financial News ML Endpoints *(requires `python3 news_model.py` after ≥30 days of data)*

These return HTTP 404 until `news_model.py` has been run at least once. After the first run, they always return HTTP 200 — returning either forecast data or a `cold_start` status JSON if insufficient data has been collected.

| Method | Path | What It Returns |
|---|---|---|
| GET | `/api/financial-news/sentiment` | 12-month LightGBM forecasts for four sector sentiment series (`MACRO_SENT`, `EQUITIES_SENT`, `FINTECH_SENT`, `VC_SENT`). Returns `{"status": "cold_start", "days_collected": N, "min_required": 30, "series": []}` until ≥30 days of news data are collected. |
| GET | `/api/financial-news/sentiment/{series_id}` | Single sentiment series. Valid `series_id`: `MACRO_SENT`, `EQUITIES_SENT`, `FINTECH_SENT`, `VC_SENT`. Returns HTTP 404 if model is still in cold_start or series is not found. |
| GET | `/api/financial-news/volume` | 12-month forecasts for article volume series (`TOTAL_VOL`, `MACRO_VOL`, `EQUITIES_VOL`, `FINTECH_VOL`). Returns `cold_start` status JSON until ≥30 days of data. |

**Cold-start response format:**
```json
{
  "status": "cold_start",
  "days_collected": 12,
  "min_required": 30,
  "series": []
}
```

**Forecast response format** (same `GroupResponse` structure as all other model groups):
```json
{
  "group": "Financial News — Sector Sentiment",
  "run_at": "2026-05-01T08:00:00",
  "series": [
    {
      "series_id": "MACRO_SENT",
      "label": "Macro Sector Sentiment",
      "unit": "avg score [-1,1]",
      "last_date": "2026-04",
      "last_value": -0.142,
      "validation": { "mae": 0.05, "rmse": 0.07, "r2": 0.41 },
      "forecast": [
        { "month": "2026-05", "mid": -0.11, "lo": -0.18, "hi": -0.04 },
        ...12 months total...
      ]
    }
  ]
}
```

---

#### Optional Sector Endpoints

These return HTTP 404 until the pipeline has been run with the corresponding `--sector` flag.

| Method | Path | Requires | What It Returns |
|---|---|---|---|
| GET | `/api/sector/bls` | `--sector bls` | 12-month forecasts for BLS employment across 10 NAICS sectors. |
| GET | `/api/sector/bea` | `--sector bea` + `BEA_API_KEY` | 12-month forecasts for BEA GDP-by-Industry gross output (10 industries). |
| GET | `/api/sector/worldbank` | `--sector worldbank` | 12-month forecasts for World Bank U.S. sector share-of-GDP (3 series). |
| GET | `/api/sector/bls-wages` | `--sector bls_wages` | 12-month forecasts for BLS average hourly earnings by sector (6 series). |
| GET | `/api/sector/bls-hours` | `--sector bls_hours` | 12-month forecasts for BLS average weekly hours by sector (3 series). |
| GET | `/api/sector/jolts` | `--sector jolts` | 12-month forecasts for JOLTS job openings by sector (7 series). |
| GET | `/api/sector/etf` | `--sector etf` | 12-month forecasts for S&P 500 sector ETF monthly close prices (11 tickers). |

#### Phase 1 Industrial Endpoints

Generated automatically by `industrial_model.py` on every `fred_refresh.py` run. Return HTTP 404 until FRED data has been fetched at least once.

| Method | Path | Series | What It Returns |
|---|---|---|---|
| GET | `/api/industrial/production` | IPMAN, IPUTIL, IPMINE, IPCONGD, IPBUSEQ, IPMAT, IPDCONGD, IPNCONGD | 12-month forecasts for 8 IP sector breakdowns. |
| GET | `/api/industrial/ism-pmi` | NAPM, NMFCI, NAPMPROD, NAPMNEWO, NAPMEMPL, NAPMVNDR | 12-month forecasts for ISM manufacturing and services PMI + sub-indices. |
| GET | `/api/industrial/capacity-utilization` | MCUMFN, CAPUTLG211S, CAPUTLB58SQ | 12-month forecasts for sector-level capacity utilization rates. |
| GET | `/api/industrial/credit` | BUSLOANS, REALLN, CONSUMER, WPU05, WPU10 | 12-month forecasts for commercial/consumer loans and PPI commodity prices. |

#### Optional Venture Capital Endpoints

These return HTTP 404 until approximately 13 months of weekly Crunchbase data has been collected and `vc_model.py` has successfully trained.

| Method | Path | Requires | What It Returns |
|---|---|---|---|
| GET | `/api/vc/ai` | `--crunchbase` (≥13 months) | 12-month forecasts for the Crunchbase AI segment: company count, 90-day rolling round count, capital raised (USD), median round size (USD), and lead investor count. |
| GET | `/api/vc/fintech` | `--crunchbase` (≥13 months) | Same five metrics for the Crunchbase Fintech segment. |
| GET | `/api/vc/healthcare` | `--crunchbase` (≥13 months) | Same five metrics for the Crunchbase Healthcare segment. |

### No Restart Required After Pipeline Runs

Because `api.py` reads results files from disk on every request, running `fred_refresh.py` while the server is up automatically makes fresh forecasts available on the next API call — no server restart needed.

---

## API Testing

Two test scripts are provided to verify the API is running correctly and all security controls are in place.

### Quick Smoke Test — `test_api_quick.py`

Hits the five most important endpoints and prints one pass/fail line each. Run this after starting the server or after any `fred_refresh.py` run.

```bash
python3 test_api_quick.py
```

Expected output:

```
[PASS] GET /api/summary                          — 15 endpoints listed
[PASS] GET /api/business-env                     — 3 series, run_at=2026-04-08T08:46:30
[PASS] GET /api/consumer-demand                  — 6 series, run_at=2026-04-08T08:47:01
[PASS] GET /api/cost-of-capital                  — 4 series, run_at=2026-04-08T08:47:12
[PASS] GET /api/risk                             — 2 series, run_at=2026-04-08T08:47:19

All 5 checks passed.
```

Use `--url` to target a different host or port:

```bash
python3 test_api_quick.py --url http://localhost:8100
```

Exits with code `0` on success, `1` if any check fails.

### Comprehensive Test Suite — `test_api.py`

Tests all endpoints, prints detailed response values for every series, and verifies all security controls. Covers 45+ checks total (Phase 0, Phase 1, and Phase 2 tests SKIP gracefully if results files haven't been generated yet).

```bash
python3 test_api.py
```

**What is tested:**

| Category | Checks |
|---|---|
| `/api/health` | Returns `status="ok"` and `ts` field (hard fail — always required) |
| `/api/summary` | Returns ≥31 endpoints; lists available series per endpoint |
| Business Environment | Group endpoint (series count + IDs), individual series for INDPRO, TCU, PAYEMS |
| Consumer Demand | Group endpoint, individual series for PCE and UMCSENT |
| Cost of Capital | Group endpoint, individual series for DFF and T10Y3M |
| Risk | Group endpoint, individual series for RECPROUSM156N |
| Phase 0: Market Risk | `/api/market/vix`, `/api/market/spreads`, `/api/market/dollar` — SKIP if not yet generated |
| Phase 0: Commodities | `/api/commodities/oil`, `/api/commodities/gold` — SKIP if not yet generated |
| Phase 0: Yield Curve | Full group + individual DGS10 — SKIP if not yet generated |
| Phase 0: FSI + Regime | `/api/market/stress`, `/api/market/regime` — SKIP if not yet generated |
| Phase 0: Raw history | `/api/series/VIXCLS/history`, `/api/series/DGS10/history` — SKIP if not generated |
| Phase 0: Error cases | `INVALID!!` series_id → HTTP 400; `DOESNOTEXIST` → HTTP 404 |
| Phase 1: Financial News | `/api/financial-news/briefing`, `/api/financial-news/top-stories`, `/api/financial-news/alerts` — SKIP if no briefing yet; verifies `stale` flag and required fields |
| Phase 2: News ML | `/api/financial-news/sentiment`, `/api/financial-news/volume` — SKIP if results not yet generated; accepts and verifies both `cold_start` and full forecast responses |
| Sector endpoints | Each returns SKIP with an instructive message if results file not yet generated |
| VC endpoints | Each returns SKIP with an instructive message if results file not yet generated |
| Security headers | All five required headers present and correct on every response |
| Server banner | `server` header must be absent |
| Input validation | Special characters in `series_id` → HTTP 400 |
| Input validation | `series_id` exceeding 30 characters → HTTP 400 |
| Unknown series | Nonexistent `series_id` → HTTP 404 with available-series list |
| CORS — blocked | External origin (`http://evil.com`) receives no CORS header |
| CORS — allowed | `localhost` origin receives correct `Access-Control-Allow-Origin` header |

**For each series, the output shows:**

```
INDPRO  Industrial Production
  unit      : Index 2017=100
  last obs  : 2026-02 = 102.6
  validation: MAE=0.8558  R²=-0.5213
  forecast  : 2026-03 mid=101.3 [99.33–101.8]  →  2027-02 mid=98.03 [95.1–98.41]  (12 months)
```

Use `--url` to target a non-default host:

```bash
python3 test_api.py --url http://localhost:8100
```

Exits with code `0` if all required checks pass (SKIPs do not count as failures), `1` if any required check fails.

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

## Reports — reports.py

`reports.py` reads all model result files from `outputs/` and generates a comprehensive Markdown forecast report, saved as a timestamped file in the `reports/` directory.

```bash
python3 reports.py
# → reports/report_20260423_105200.md
```

The `reports/` directory is created automatically on the first run.

### What the Report Contains

Each report includes:

- **Table of Contents** with anchor links to all 8 model group sections
- **Per-group forecast table** — for every series in the group: last known value, +1M / +3M / +6M / +12M point forecasts, and a trend label (↑ Rising / ↓ Declining / → Stable / ⚠ Rising)
- **Signal paragraph** — a narrative interpretation of what the forecasts mean for that group, with live values injected from the model output
- **Low-confidence warning** — automatically added for any group where a series R² < −5, flagging that forecasts are extrapolating outside the training distribution
- **Overall Macro Narrative** — a synthesized view across all groups covering recession probability, growth trajectory, Fed policy, credit vs. equity divergence, consumer sentiment, and commodity prices
- **Bottom Line** — one-paragraph summary calibrated to the current recession probability level

### Report Naming

Files are named `report_<YYYYMMDD_HHMMSS>.md` so every run produces a new file. Previous reports are preserved, allowing comparison across refresh cycles.

### Viewing Reports

Reports are standard Markdown and render in any Markdown viewer:

- **VS Code** — open the file and press `Cmd+Shift+V` (Mac) or `Ctrl+Shift+V` (Windows) for the preview pane
- **GitHub** — reports pushed to the repo render automatically
- **Terminal** — `cat reports/report_*.md | head -100` for a quick read

### Workflow Integration

Run `reports.py` immediately after a model refresh to capture the latest forecasts:

```bash
python3 fred_refresh.py --sector --news daily --enrich && \
python3 news_model.py && \
python3 reports.py
```

---

## Output Files Reference

### Model Files (`outputs/`)

| File | Description |
|---|---|
| `business_env_model_{INDPRO,TCU,PAYEMS}.joblib` | Business Environment LightGBM regressors |
| `consumer_demand_model_{series}.joblib` (×6) | Consumer Demand LightGBM regressors |
| `cost_of_capital_model_{DFF,DPRIME,T10Y3M,T10Y2Y}.joblib` | Cost of Capital LightGBM regressors |
| `risk_model_{recpro,sentiment}.joblib` | Risk model LightGBM regressors |
| `market_model_{series}.joblib` (×6) | Phase 0: Market Risk + Commodities regressors |
| `yield_curve_model_{series}.joblib` (×8) | Phase 0: Yield curve tenor regressors |
| `composite_model_fsi.joblib` | Phase 0: FSI LightGBM regressor |
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
| `results_market_risk.json` | Phase 0: Forecasts for VIX, HY spread, IG spread, USD index |
| `results_commodities.json` | Phase 0: Forecasts for WTI crude oil and gold |
| `results_yield_curve.json` | Phase 0: Forecasts for all 8 Treasury yield tenors |
| `results_financial_stress.json` | Phase 0: Financial Stress Index (FSI) forecast (GroupResponse format) |
| `regime_history.json` | Phase 0: Full FSI history + current regime label + 12-month regime forecast |
| `results_sector_bls.json` | BLS Employment forecasts (created after first `--sector bls` run) |
| `results_sector_bea.json` | BEA GDP forecasts (created after first `--sector bea` run) |
| `results_sector_worldbank.json` | World Bank sector forecasts (created after first `--sector worldbank` run) |
| `results_vc_ai.json` | AI VC segment forecasts (created after ~13 months of `--crunchbase` runs) |
| `results_vc_fintech.json` | Fintech VC segment forecasts |
| `results_vc_healthcare.json` | Healthcare VC segment forecasts |
| `daily_briefing_{YYYY-MM-DD}.json` | Phase 1: Daily news briefing — top stories, sector mood, macro signals, alerts, stale flag. Created by `briefing.py` after each `--news daily` run. Only the most recent file is served by the API. |
| `results_news_sentiment.json` | Phase 2: 12-month LightGBM forecasts for `MACRO_SENT`, `EQUITIES_SENT`, `FINTECH_SENT`, `VC_SENT`. Contains `{"status": "cold_start", ...}` until ≥30 days of data. Created by `news_model.py`. |
| `results_news_volume.json` | Phase 2: 12-month forecasts for `TOTAL_VOL`, `MACRO_VOL`, `EQUITIES_VOL`, `FINTECH_VOL`. Same cold-start behavior. Created by `news_model.py`. |

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
| `market_risk_*.png` | Phase 0: Dashboard, validation, importance for VIX/spreads/USD |
| `commodities_*.png` | Phase 0: Dashboard, validation, importance for WTI and gold |
| `yield_curve_*.png` | Phase 0: Dashboard, validation, importance for 8-tenor yield curve |
| `composite_*.png` | Phase 0: Dashboard, validation, importance for FSI model |
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
| VIXCLS | CBOE Volatility Index (VIX) | Daily | MarketRisk |
| BAMLH0A0HYM2 | ICE BofA US High Yield OAS | Daily | MarketRisk |
| BAMLC0A0CM | ICE BofA US Corporate (IG) OAS | Daily | MarketRisk |
| DTWEXBGS | Nominal Broad USD Index | Daily | MarketRisk |
| DCOILWTICO | WTI Crude Oil Price | Daily | Commodities |
| GOLDAMGBD228NLBM | Gold London Fixing Price | Daily | Commodities |
| DGS1MO | Treasury 1-Month Yield | Daily | YieldCurve |
| DGS3MO | Treasury 3-Month Yield | Daily | YieldCurve |
| DGS6MO | Treasury 6-Month Yield | Daily | YieldCurve |
| DGS1 | Treasury 1-Year Yield | Daily | YieldCurve |
| DGS2 | Treasury 2-Year Yield | Daily | YieldCurve |
| DGS5 | Treasury 5-Year Yield | Daily | YieldCurve |
| DGS10 | Treasury 10-Year Yield | Daily | YieldCurve |
| DGS30 | Treasury 30-Year Yield | Daily | YieldCurve |

UMCSENT is saved to both `ConsumerDemand/` and `RiskLeadingInd/` on every refresh. Daily series are stored in daily format; model scripts resample to monthly means internally.

---

## Scheduling with Cron

To run the refresh automatically every Monday at 8 AM:

```cron
0 8 * * 1 cd /path/to/econdata && /path/to/python3 fred_refresh.py >> logs/refresh.log 2>&1
```

With sector APIs (BLS and World Bank are free):

```cron
0 8 * * 1 cd /path/to/econdata && /path/to/python3 fred_refresh.py --sector bls worldbank >> logs/refresh.log 2>&1
```

With Crunchbase VC data:

```cron
0 8 * * 1 cd /path/to/econdata && /path/to/python3 fred_refresh.py --crunchbase >> logs/refresh.log 2>&1
```

Full pipeline — FRED + free sector APIs + Crunchbase VC:

```cron
0 8 * * 1 cd /path/to/econdata && /path/to/python3 fred_refresh.py --sector bls worldbank --crunchbase >> logs/refresh.log 2>&1
```

**Phase 1 — Financial News (recommended schedule):**

```cron
# Daily full news pull at 8 AM (all configured sources)
0 8 * * * cd /path/to/econdata && /path/to/python3 fred_refresh.py --news daily >> logs/news.log 2>&1

# Intraday Finnhub updates — 4× per day (noon, 4pm, 8pm, midnight)
0 0,12,16,20 * * * cd /path/to/econdata && /path/to/python3 fred_refresh.py --news realtime >> logs/news.log 2>&1
```

Note: `--news daily` and `--news realtime` are safe to overlap — `news_apis.py` uses a file lock on the CSV to prevent concurrent write corruption.

**Phase 2 — Enrichment + ML (after ≥30 days of news data):**

```cron
# Daily enrichment run (after news pull)
30 8 * * * cd /path/to/econdata && /path/to/python3 fred_refresh.py --news daily --enrich >> logs/news.log 2>&1

# Weekly ML model retrain (every Monday, after enrichment)
0 9 * * 1 cd /path/to/econdata && /path/to/python3 news_model.py >> logs/news_model.log 2>&1
```

`news_model.py` exits cleanly with a cold-start message until 30 days of news data exist — safe to schedule from day 1. Rate limits: NewsAPI and Marketaux are capped at 100 req/day (daily only); Finnhub supports up to 60 req/min (used for realtime).

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

---

## Phase Roadmap

| Phase | Status | Description | New API Keys Required |
|---|---|---|---|
| **Phase 0** | ✅ Complete | Free FRED extensions: VIX, credit spreads, USD index, WTI oil, gold, 8-tenor yield curve, FSI, Market Regime | None |
| **Phase 1** | ✅ Complete | News ingestion pipeline (`news_apis.py` + `briefing.py`): daily briefings, top-stories ranking, rule-based impact alerts. Two new files + targeted changes to `fred_refresh.py`, `api.py`, `test_api.py`. | At least one of `NEWS_API_KEY`, `MARKETAUX_API_KEY`, `FINNHUB_API_KEY` |
| **Phase 2** | ✅ Complete | Enrichment + Sentiment ML — `enrichment_apis.py` attaches yfinance/FMP signals to ticker-tagged articles; `news_model.py` trains LightGBM on 30+ days of sentiment/volume data; `/api/financial-news/sentiment`, `/api/financial-news/sentiment/{id}`, `/api/financial-news/volume` endpoints with cold-start handling. | `FMP_API_KEY` (optional — yfinance needs no key) |

Phase 1 requires at least one news API key (the others are optional additional sources) and works from day 1. Phase 2 enrichment (`--enrich`) requires `yfinance` installed; FMP fundamentals are optional. Phase 2 ML (`news_model.py`) requires Phase 1 having run for ≥30 days but exits cleanly with a cold-start message until then. All phase-specific keys are pre-documented in `.env.example`.
