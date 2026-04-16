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
pip install filelock   # Phase 1 — prevents CSV corruption when daily + realtime crons overlap
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
1. Write `news_apis.py` — constants, classify helpers, 3 fetchers, normalize_article, save_articles, refresh_news
2. Write `briefing.py` — score_impact, load_articles, compute_sector_mood, generate_macro_signals, generate_alerts, generate_daily_briefing, main
3. Update `fred_refresh.py` — `--news` argparse, env key loads, news refresh step in main()
4. Update `api.py` — NewsArticle + BriefingResponse models, _latest_briefing_path, 3 route handlers + descriptors
5. Update `test_api.py` — 3 new test functions; update summary threshold 25→28

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
python fred_refresh.py --news daily              # ingest + generate briefing
# OR data-only:
python fred_refresh.py --news daily --skip-models

ls data/FinancialNews/raw/news_articles.csv       # should exist with rows
ls outputs/daily_briefing_$(date +%Y-%m-%d).json  # should exist

# verify via API
python api.py &
curl http://localhost:8100/api/financial-news/briefing
curl http://localhost:8100/api/financial-news/top-stories
curl http://localhost:8100/api/financial-news/alerts

python test_api.py   # Phase 1 tests PASS or SKIP (SKIP if no data yet)

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
