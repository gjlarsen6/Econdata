# Financial Data Expansion Plan

## Context

The existing `econdata` system is a macroeconomic forecasting pipeline that pulls structured time-series data from FRED, BLS, BEA, World Bank, and Crunchbase, trains LightGBM forecasting models, and exposes results via a FastAPI server on port 8100. The architecture is modular: fetcher modules (`sector_apis.py`, `crunchbase_apis.py`) feed CSVs → model scripts → JSON outputs → REST API.

This plan expands the system in two phases:

- **Phase 0** — Free FRED extensions: VIX, credit spreads, commodities, full yield curve, and composite analytics (Financial Stress Index, Market Regime). Zero new API keys required.
- **Phase 1** — News ingestion pipeline: ingest, normalize, enrich, and model financial news data across macro, equities, fintech, and VC sectors — adding a qualitative/sentiment signal layer.
- **Phase 2** — Sentiment ML (after ≥30 days of news data accumulate).

---

## Minimal Viable Paths

```
Just want VIX + credit spread + commodity forecasts?
  → Do Phase 0 only (7 steps, no new API keys)

Just want a daily news briefing (no ML)?
  → Do Phase 1 only (steps 1–5)

Full sentiment trend forecasting?
  → Phase 0 + Phase 1 + collect 30+ days of news data + Phase 2
```

---

## Architecture Overview

```
Layer 0 (Free FRED)     → fred_ingestion_map (extended)
                          market_model.py      VIX, spreads, dollar → results_market_risk.json
                          yield_curve_model.py 8 treasury tenors    → results_yield_curve.json
                          composite_model.py   FSI, regime          → results_financial_stress.json

Layer 1 (News Ingest)   → news_apis.py        Fetch from Finnhub, News API, Marketaux → CSV
Layer 2 (Enrichment)    → enrichment_apis.py  yfinance fundamentals, FMP signals
Layer 3 (AI/ML)         → news_model.py       Sentiment forecasting, market impact scoring
Layer 4 (Output)        → briefing.py         Daily briefing JSON, real-time alerts, top-10 events
```

### Dependency Graph

```
fred_refresh.py
├── [existing] sector_apis.py       → sector_model.py    → results_sector_*.json
├── [existing] crunchbase_apis.py   → vc_model.py        → results_vc_*.json
├── [Ph0] fred_ingestion_map (++)   → market_model.py    → results_market_risk.json
│                                   → yield_curve_model.py → results_yield_curve.json
│                                   → composite_model.py → results_financial_stress.json
│                                                          → regime_history.json
└── [Ph1] news_apis.py              → briefing.py        → daily_briefing_{date}.json
          ↓ (optional enrichment)
       enrichment_apis.py (yfinance + FMP) → news_enriched.csv
          ↓ (after ≥30 days)
       news_model.py → results_news_sentiment.json, results_news_volume.json, news_briefing.json

api.py reads all results_*.json → serves all endpoints
```

---

## Phase 0 — Free FRED Extensions

### New FRED Series (no new API keys)

Add the following to `data/fred_ingestion_map_full_production.json`. All use the existing `FRED_API_KEY`.

#### Market Risk Group → `data/MarketRisk/`

| Series ID | Description | Frequency | Resample |
|---|---|---|---|
| `VIXCLS` | CBOE Volatility Index (fear gauge) | Daily | Monthly mean |
| `BAMLH0A0HYM2` | ICE BofA US High Yield OAS | Daily | Monthly mean |
| `BAMLC0A0CM` | ICE BofA US Corp IG OAS | Daily | Monthly mean |
| `DTWEXBGS` | USD Broad Nominal Index (dollar strength) | Daily | Monthly mean |

#### Commodities Group → `data/Commodities/`

| Series ID | Description | Frequency | Resample |
|---|---|---|---|
| `DCOILWTICO` | WTI Crude Oil Price ($/bbl) | Daily | Monthly mean |
| `GOLDAMGBD228NLBM` | Gold Price, London AM fixing ($/troy oz) | Daily | Monthly mean |

#### Yield Curve Group → `data/YieldCurve/`

| Series ID | Tenor | Frequency | Resample |
|---|---|---|---|
| `DGS1MO` | 1-Month | Daily | Monthly mean |
| `DGS3MO` | 3-Month | Daily | Monthly mean |
| `DGS6MO` | 6-Month | Daily | Monthly mean |
| `DGS1` | 1-Year | Daily | Monthly mean |
| `DGS2` | 2-Year | Daily | Monthly mean |
| `DGS5` | 5-Year | Daily | Monthly mean |
| `DGS10` | 10-Year | Daily | Monthly mean |
| `DGS30` | 30-Year | Daily | Monthly mean |

### New Phase 0 Model Files

#### `market_model.py`
Follows pattern of `sector_model.py` and `risk_model.py`. Reuses `macro_utils.py` pipeline.

```
Trains: VIXCLS, BAMLH0A0HYM2, BAMLC0A0CM, DTWEXBGS, DCOILWTICO, GOLDAMGBD228NLBM
Cross-features: HY_spread - IG_spread (HY-IG differential), VIX × recession_prob
Date range: per series (BAMLH0A0HYM2 starts 1996; VIXCLS 1990; commodities 1980s)
Outputs: outputs/results_market_risk.json, outputs/results_commodities.json
```

#### `yield_curve_model.py`
Trains all 8 treasury tenors jointly. Cross-features: 10Y−2Y slope, 2×5Y−1Y−10Y curvature, 10Y−3M spread.

```
Trains: DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS5, DGS10, DGS30
Cross-features: slope (DGS10−DGS2), curvature (2×DGS5−DGS1−DGS10), butterfly
Output: outputs/results_yield_curve.json
```

#### `composite_model.py`
Computes derived signals from already-fetched data. No new API calls.

**Financial Stress Index (FSI):**
```python
# All inputs normalized to [0, 1] using historical percentile rank
FSI = mean([
    normalize(VIXCLS),           # higher VIX = more stress
    normalize(BAMLH0A0HYM2),     # wider HY spread = more stress
    normalize(BAMLC0A0CM),       # wider IG spread = more stress
    normalize(RECPROUSM156N),    # higher recession prob = more stress
    normalize(-T10Y2Y),          # deeper inversion = more stress
])
# Thresholds: < 0.3 normal | 0.3–0.6 elevated | > 0.6 crisis
```

**Market Regime Classifier (rule-based, deterministic):**
```python
# Inputs: FSI, INDPRO 3-month trend, UMCSENT level, RECPROUSM156N
regime = classify_regime(fsi, indpro_trend, umcsent, recession_prob)
# Labels: "expansion" | "slowdown" | "contraction" | "stress" | "recovery"
```

```
Outputs: outputs/results_financial_stress.json   # FSI time series + 12-month forecast
         outputs/regime_history.json              # regime label per month + current
```

**Reuses from `macro_utils.py`:** `engineer_features()`, `train_series_models()`, `joint_recursive_forecast()`, `save_model_results()`

### Phase 0 Data Storage

```
data/
├── MarketRisk/
│   ├── VIXCLS.csv
│   ├── BAMLH0A0HYM2.csv
│   ├── BAMLC0A0CM.csv
│   └── DTWEXBGS.csv
├── Commodities/
│   ├── DCOILWTICO.csv
│   └── GOLDAMGBD228NLBM.csv
└── YieldCurve/
    ├── DGS1MO.csv  DGS3MO.csv  DGS6MO.csv  DGS1.csv
    ├── DGS2.csv    DGS5.csv    DGS10.csv   DGS30.csv

outputs/
├── results_market_risk.json      # GroupResponse: VIXCLS, BAMLH0A0HYM2, BAMLC0A0CM, DTWEXBGS
├── results_commodities.json      # GroupResponse: DCOILWTICO, GOLDAMGBD228NLBM
├── results_yield_curve.json      # GroupResponse: all 8 tenors + slope/curvature metrics
├── results_financial_stress.json # FSI time series + forecast
└── regime_history.json           # monthly regime labels + current
```

### Phase 0 API Endpoints (add to `api.py`)

| Endpoint | Source File | Description |
|---|---|---|
| `GET /api/health` | — | Liveness check: `{"status":"ok","ts":"..."}` (no file I/O) |
| `GET /api/market/vix` | results_market_risk.json | VIX + 12-month forecast |
| `GET /api/market/spreads` | results_market_risk.json | HY + IG credit spreads + forecasts |
| `GET /api/market/dollar` | results_market_risk.json | USD broad index + forecast |
| `GET /api/commodities/oil` | results_commodities.json | WTI crude oil + forecast |
| `GET /api/commodities/gold` | results_commodities.json | Gold price + forecast |
| `GET /api/market/yield-curve` | results_yield_curve.json | Full yield curve: spot + forecasts + slope/curvature |
| `GET /api/market/stress` | results_financial_stress.json | FSI value + trend + 12-month forecast |
| `GET /api/market/regime` | regime_history.json | Current regime label + history |
| `GET /api/series/{series_id}/history` | data/{group}/{series_id}.csv | Raw observations for any fetched series |

**New Pydantic schemas for `api.py`:**
```python
class YieldCurveResponse(BaseModel):
    run_at: str
    spot: dict[str, float]          # tenor → latest yield
    slope_10y2y: float
    curvature: float
    forecasts: dict[str, list]      # tenor → [{month, mid, lo, hi}]

class StressResponse(BaseModel):
    run_at: str
    current_fsi: float
    regime: str                     # "normal" | "elevated" | "crisis"
    history: list[dict]             # [{date, fsi}]
    forecast: list[dict]            # [{month, mid, lo, hi}]

class RegimeResponse(BaseModel):
    run_at: str
    current_regime: str
    history: list[dict]             # [{date, regime}]
```

### `fred_refresh.py` Changes for Phase 0

- Add new group blocks for `MarketRisk`, `Commodities`, `YieldCurve` (same pattern as existing groups, lines ~312–363)
- Add subprocess calls: `python market_model.py`, `python yield_curve_model.py`, `python composite_model.py`
- Update summary table to include: current VIX, FSI level, regime label, WTI price, gold price, 10Y−2Y slope

---

## Phase 1 — News Ingestion Pipeline

### API Sources and Rate Limit Budget

| API | Env Key | Free Tier | Polling Strategy |
|---|---|---|---|
| **Finnhub** | `FINNHUB_API_KEY` | 60 req/min | Up to 4×/day safe (`--news realtime`) |
| **News API** | `NEWS_API_KEY` | 100 req/day | 1×/day only (`--news daily`) |
| **Marketaux** | `MARKETAUX_API_KEY` | 100 req/day | 1×/day only (`--news daily`) |
| **FMP** | `FMP_API_KEY` | 250 req/day | Covers ~83 tickers/day |
| **yfinance** | *(no key)* | No hard limit | 0.5s sleep between tickers |

**`--news` flag modes (add to `fred_refresh.py`):**
```bash
python fred_refresh.py --news daily      # News API + Marketaux + Finnhub (once/day)
python fred_refresh.py --news realtime   # Finnhub only (safe for frequent cron runs)
python fred_refresh.py --news all        # All sources (use for manual/backfill runs)
python fred_refresh.py --news all --skip-models  # Ingest only, no model retraining
```

**Cron schedule recommendation:**
```cron
0 8 * * *        python fred_refresh.py --news daily         # full daily pull
0 12,16,20 * * * python fred_refresh.py --news realtime      # intraday Finnhub updates
0 2 * * 1        python fred_refresh.py --sector all --crunchbase  # weekly macro refresh
```

### Standard News Article Schema

Every ingested article is normalized to this schema before storage:

```json
{
  "timestamp": "2026-04-15T09:30:00Z",
  "ingested_at": "2026-04-15T09:35:00Z",
  "source_api": "marketaux",
  "source_name": "Reuters",
  "url": "https://...",
  "sector": "equities",
  "ticker": "NVDA",
  "headline": "Nvidia beats Q1 earnings...",
  "summary": "...",
  "sentiment": 0.72,
  "sentiment_label": "positive",
  "macro_tag": "earnings",
  "market_impact_score": null,
  "entities": ["NVDA", "semiconductors"]
}
```

**Sectors:** `macro`, `equities`, `fintech`, `vc`, `energy`, `banking`, `crypto`, `real_estate`
**Macro tags:** `rate_cuts`, `earnings`, `layoffs`, `inflation`, `gdp`, `regulation`, `funding_round`, `ipo`

**Classification method:** keyword + entity matching (fast, deterministic). No NLP libraries (transformers/sentence-transformers) — too slow for free-tier use.

### Phase 1 New Files

#### `news_apis.py`
Follows pattern of `sector_apis.py` and `crunchbase_apis.py`.

```
Functions:
  fetch_newsapi(api_key, topics, from_date) → list[dict]
  fetch_marketaux(api_key, tickers, sectors, from_date) → list[dict]
  fetch_finnhub(api_key, tickers, from_date) → list[dict]
  normalize_article(raw, source_api) → dict   # maps raw fields to standard schema
  refresh_news(api_keys: dict, mode="daily", since_hours=24) → None
    → Calls appropriate sources per mode, normalizes, dedupes by URL, saves CSV
```

**Storage:**
```
data/FinancialNews/raw/news_articles.csv        # append-only, all articles (deduped by URL)
data/FinancialNews/daily/{YYYY-MM-DD}.csv       # daily partition
data/FinancialNews/sentiment_daily.csv          # aggregated daily sentiment per sector
data/FinancialNews/sector_volume_daily.csv      # article count per sector per day
```

#### `briefing.py`
Generates daily briefing output. **Works immediately — no ML required.**

```
Functions:
  generate_daily_briefing(date=None) → dict
    → Reads news_articles.csv (today's rows), scores by market_impact_score
    → Returns: top_stories (10), sector_mood (avg sentiment), macro_signals, alerts
  format_briefing_text(briefing: dict) → str  # plain text version
  main() → None  # writes outputs/daily_briefing_{YYYY-MM-DD}.json
```

### Phase 1 API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/financial-news/briefing` | Today's top stories, sector mood, macro signals, alerts |
| `GET /api/financial-news/top-stories` | Top-10 highest market-impact articles |
| `GET /api/financial-news/alerts` | High-impact articles: `severity: "critical" \| "high" \| "medium"` |

**New Pydantic schemas:**
```python
class NewsArticle(BaseModel):
    timestamp: str
    source_name: str
    sector: str
    ticker: str | None
    headline: str
    sentiment: float | None
    macro_tag: str | None
    market_impact_score: float | None

class BriefingResponse(BaseModel):
    date: str
    generated_at: str
    top_stories: list[NewsArticle]
    sector_mood: dict[str, float]   # sector → avg sentiment
    macro_signals: list[str]        # text bullets
    alerts: list[str]               # high-impact alerts
```

---

## Phase 2 — Enrichment + Sentiment ML

### Cold Start Handling

The news sentiment ML model requires ≥30 days of data before training is meaningful.

```python
# In news_model.py
MIN_SENTIMENT_DAYS = 30

def check_readiness(sentiment_df: pd.DataFrame) -> bool:
    return len(sentiment_df) >= MIN_SENTIMENT_DAYS
```

**Pipeline behavior during cold start:**
```
if not news_model.check_readiness(sentiment_df):
    print(f"[NEWS] {len(sentiment_df)} days collected — need {MIN_SENTIMENT_DAYS} for ML training")
    # Still write news_briefing.json with top stories (no forecast section)
    # API returns cold_start status for forecast fields
```

**API response during cold start:**
```json
{
  "status": "cold_start",
  "days_collected": 12,
  "min_required": 30,
  "series": []
}
```

### `enrichment_apis.py`
Uses **yfinance** (no key, no hard rate limit) as primary enrichment source. FMP as supplement for fundamentals.

```
Functions:
  fetch_yfinance_signals(ticker) → dict
    → trailing_pe, market_cap, eps_trailing, week52_high, week52_low, price_momentum_30d
    → Sleep 0.5s between calls to avoid soft rate limits
  fetch_fmp_fundamentals(api_key, ticker) → dict
    → ev_ebitda, revenue_growth_yoy, debt_to_equity
    → Budget: max 83 unique tickers/day on free tier
  enrich_articles(articles: list[dict], api_keys: dict) → list[dict]
    → Attaches yfinance signals to all ticker-tagged articles
    → Attaches FMP fundamentals where budget allows (priority: S&P 500 tickers)
  save_enriched(articles) → None  # writes data/FinancialNews/enriched/news_enriched.csv
```

**Note:** Alpha Vantage (25 req/day free) is too limited for production use — replaced by yfinance. Add `ALPHA_VANTAGE_API_KEY` to `.env.example` as optional fallback only.

### `news_model.py`
ML model for financial news signals. Follows pattern of `sector_model.py` and `vc_model.py`.

```
Functions:
  load_sentiment_series() → DataFrame  # from sentiment_daily.csv (requires ≥30 rows)
  load_volume_series() → DataFrame     # from sector_volume_daily.csv
  train_and_forecast(series_df, series_id) → dict  # uses macro_utils.py pipeline
  score_market_impact(articles_df) → DataFrame  # rule-based scoring (see weights below)
  generate_top10(articles_df) → list[dict]      # highest market_impact_score
  main() → None  # trains models, writes outputs below
```

**Outputs:** `outputs/results_news_sentiment.json`, `outputs/results_news_volume.json`, `outputs/news_briefing.json`

**Reuses from `macro_utils.py`:** `engineer_features()`, `train_series_models()`, `joint_recursive_forecast()`, `save_model_results()`

### Market Impact Scoring

| Signal | Weight | Source |
|---|---|---|
| Sentiment magnitude (abs score) | 25% | Finnhub / Marketaux built-in |
| Source authority (Reuters/Bloomberg > blog) | 20% | `source_name` allowlist |
| Entity prominence (S&P 500 ticker vs unknown) | 20% | ticker lookup |
| Volume spike in sector (>2σ above 30d avg) | 20% | `sector_volume_daily.csv` |
| Macro tag type (rate_cuts, earnings > general) | 15% | keyword classifier |

### Phase 2 API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/financial-news/sentiment` | Sector sentiment trends + 12-day LightGBM forecast |
| `GET /api/financial-news/sentiment/{id}` | Single sector (MACRO_SENT, EQUITIES_SENT, etc.) |
| `GET /api/financial-news/volume` | Article volume trends by sector |

**Series IDs:** `MACRO_SENT`, `EQUITIES_SENT`, `FINTECH_SENT`, `VC_SENT`

---

## All New Endpoints Summary

| Phase | Endpoint | Data Required |
|---|---|---|
| 0 | `GET /api/health` | None |
| 0 | `GET /api/market/vix` | VIXCLS.csv + results_market_risk.json |
| 0 | `GET /api/market/spreads` | BAMLH*.csv + results_market_risk.json |
| 0 | `GET /api/market/dollar` | DTWEXBGS.csv + results_market_risk.json |
| 0 | `GET /api/commodities/oil` | DCOILWTICO.csv + results_commodities.json |
| 0 | `GET /api/commodities/gold` | GOLDAMGBD228NLBM.csv + results_commodities.json |
| 0 | `GET /api/market/yield-curve` | DGS*.csv + results_yield_curve.json |
| 0 | `GET /api/market/stress` | results_financial_stress.json |
| 0 | `GET /api/market/regime` | regime_history.json |
| 0 | `GET /api/series/{series_id}/history` | Any data/{group}/{id}.csv |
| 1 | `GET /api/financial-news/briefing` | news_articles.csv (same-day) |
| 1 | `GET /api/financial-news/top-stories` | news_articles.csv + impact scores |
| 1 | `GET /api/financial-news/alerts` | news_articles.csv (severity filtered) |
| 2 | `GET /api/financial-news/sentiment` | results_news_sentiment.json (≥30d data) |
| 2 | `GET /api/financial-news/sentiment/{id}` | Same |
| 2 | `GET /api/financial-news/volume` | results_news_volume.json |

---

## Files to Create / Modify

### New Files

| File | Phase | Purpose |
|---|---|---|
| `market_model.py` | 0 | LightGBM for VIX, credit spreads, dollar, oil, gold |
| `yield_curve_model.py` | 0 | Joint 8-tenor Treasury yield curve model |
| `composite_model.py` | 0 | Financial Stress Index + Market Regime classifier |
| `news_apis.py` | 1 | News ingestion + normalization (3 APIs) |
| `briefing.py` | 1 | Daily briefing generator (no ML required) |
| `enrichment_apis.py` | 2 | yfinance + FMP article enrichment |
| `news_model.py` | 2 | Sentiment aggregation + LightGBM forecasting |

### Modified Files

| File | Changes |
|---|---|
| `data/fred_ingestion_map_full_production.json` | Add 14 new FRED series across 3 new groups |
| `fred_refresh.py` | New group fetch blocks, subprocess calls for new models, `--news` flag |
| `api.py` | New endpoints, new Pydantic schemas, cold-start handling in news routes |
| `.env.example` | Add Phase 1 news + enrichment API key entries |

### New Python Dependencies

```bash
pip install yfinance   # Phase 2 enrichment — no API key required
# requests already installed
# No transformers/sentence-transformers — keyword matching only
```

---

## `.env.example` Additions

```
# Phase 0: No new keys required — all data uses existing FRED_API_KEY

# Phase 1 — News Ingestion (at least one key required for --news flag)
NEWS_API_KEY=your_newsapi_key_here
MARKETAUX_API_KEY=your_marketaux_key_here
FINNHUB_API_KEY=your_finnhub_key_here

# Phase 2 — Enrichment
FMP_API_KEY=your_fmp_key_here
# yfinance: no API key required
# ALPHA_VANTAGE_API_KEY=optional_fallback_only  (25 req/day — very limited)
```

---

## Pipeline Command Usage

```bash
# Phase 0: Run with no new keys — fetches all new FRED series automatically
python fred_refresh.py

# Phase 1: Daily news pull
python fred_refresh.py --news daily

# Phase 1: Intraday Finnhub update only (safe to run every 15 min)
python fred_refresh.py --news realtime

# Phase 1: One-time backfill / manual run (uses all 3 news APIs)
python fred_refresh.py --news all --skip-models

# Phase 2: News model training (after ≥30 days of data)
python news_model.py

# Full weekly pipeline: macro + sector + VC + news
python fred_refresh.py --sector all --crunchbase --news daily
```

---

## Implementation Order

### Phase 0 — Free FRED Extensions (no new API keys)
1. Add 14 FRED series to `fred_ingestion_map_full_production.json`
2. Update `fred_refresh.py` — add MarketRisk/Commodities/YieldCurve fetch blocks + model subprocess calls
3. Write `market_model.py` — reuse macro_utils.py pipeline
4. Write `yield_curve_model.py` — 8-tenor joint model
5. Write `composite_model.py` — FSI + regime classifier
6. Add Phase 0 endpoints + schemas to `api.py` (including `/api/health`)
7. Update `test_api.py` to cover new endpoints

### Phase 1 — News Ingestion
1. Update `.env.example` with news API key entries
2. Write `news_apis.py` — fetch + normalize + dedupe + save CSVs
3. Add `--news` flag + rate-limit-aware refresh block to `fred_refresh.py`
4. Write `briefing.py` — daily briefing generator
5. Add Phase 1 endpoints + Pydantic schemas to `api.py`

### Phase 2 — Enrichment + Sentiment ML (after ≥30 days of news data)
1. Write `enrichment_apis.py` — yfinance primary, FMP supplement
2. Write `news_model.py` — with cold-start guard (`MIN_SENTIMENT_DAYS = 30`)
3. Add Phase 2 endpoints to `api.py` with cold-start response handling
4. `pip install yfinance`

---

## Verification

```bash
# --- Phase 0 ---
python fred_refresh.py   # fetches new FRED series, trains market/yield/composite models
python api.py &          # start server

curl http://localhost:8100/api/health
curl http://localhost:8100/api/market/vix
curl http://localhost:8100/api/market/spreads
curl http://localhost:8100/api/market/dollar
curl http://localhost:8100/api/commodities/oil
curl http://localhost:8100/api/commodities/gold
curl http://localhost:8100/api/market/yield-curve
curl http://localhost:8100/api/market/stress
curl http://localhost:8100/api/market/regime
curl http://localhost:8100/api/series/VIXCLS/history

python test_api.py       # all checks should pass including new endpoints

# --- Phase 1 (after setting at least one news API key) ---
python fred_refresh.py --news daily --skip-models
ls data/FinancialNews/raw/news_articles.csv       # should exist with rows

python api.py &
curl http://localhost:8100/api/financial-news/briefing
curl http://localhost:8100/api/financial-news/top-stories
curl http://localhost:8100/api/financial-news/alerts

# --- Phase 2 (after ≥30 days of news data) ---
python news_model.py
ls outputs/results_news_sentiment.json   # should exist
ls outputs/news_briefing.json            # should exist

curl http://localhost:8100/api/financial-news/sentiment
curl http://localhost:8100/api/financial-news/volume

# Cold-start check (before 30 days of data):
# /api/financial-news/sentiment should return:
# {"status": "cold_start", "days_collected": N, "min_required": 30, "series": []}

python test_api.py       # full suite including Phase 1 + 2 endpoints
```

---

## Optional / Advanced APIs (future phases)

| API | Env Key | Use Case | Free Tier |
|---|---|---|---|
| **Newscatcher** | `NEWSCATCHER_API_KEY` | Multi-sector segmentation, structured pipelines | Limited |
| **Stock News API** | `STOCKNEWS_API_KEY` | Equities + earnings, clean headlines | Limited |
| **PredictLeads** | `PREDICTLEADS_API_KEY` | Startup/funding signals for VC tracking | Paid only |
| **Polygon.io** | `POLYGON_API_KEY` | Real-time market data, options | Free tier (delayed) |
| **CoinGecko** | *(no key)* | Crypto prices and market cap | Free (no key) |
