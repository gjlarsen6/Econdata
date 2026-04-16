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

> **Status:** Ready to implement. Phase 0 complete. All Phase 1 work is contained in two new files (`news_apis.py`, `briefing.py`) plus targeted additions to `fred_refresh.py` and `api.py`.

### API Sources and Rate Limit Budget

| API | Env Key | Free Tier | Polling Strategy |
|---|---|---|---|
| **News API** | `NEWS_API_KEY` | 100 req/day | 1×/day only (`--news daily`) |
| **Marketaux** | `MARKETAUX_API_KEY` | 100 req/day | 1×/day only (`--news daily`) |
| **Finnhub** | `FINNHUB_API_KEY` | 60 req/min | Up to 4×/day safe (`--news realtime`) — cron runs 4× at 0,12,16,20h |

At least one key is required. If multiple are set, all active sources run per mode.

**`--news` flag modes:**
```bash
python fred_refresh.py --news daily      # News API + Marketaux + Finnhub (once/day)
python fred_refresh.py --news realtime   # Finnhub only (safe for 15-min cron)
python fred_refresh.py --news all        # All sources, backfill window (since_hours=168 / 7 days)
python fred_refresh.py --news all --skip-models  # Ingest only, no model retraining
```

**Recommended cron schedule:**
```cron
0 8 * * *        python fred_refresh.py --news daily          # daily full pull
0 0,12,16,20 * * * python fred_refresh.py --news realtime      # intraday Finnhub (4×/day)
0 2 * * 1        python fred_refresh.py --sector all --crunchbase  # weekly macro
```

---

### Data Storage Layout

```
data/FinancialNews/
├── raw/
│   └── news_articles.csv        # append-only master — ALL articles, deduped by URL
└── daily/
    └── {YYYY-MM-DD}.csv         # partition per day (written after each refresh)

outputs/
└── daily_briefing_{YYYY-MM-DD}.json   # generated by briefing.py
```

**`news_articles.csv` column schema (ordered):**

| Column | Type | Notes |
|---|---|---|
| `timestamp` | ISO-8601 str | Article publish time (UTC) |
| `ingested_at` | ISO-8601 str | When this pipeline fetched it |
| `source_api` | str | `"newsapi"` \| `"marketaux"` \| `"finnhub"` |
| `source_name` | str | Publisher name (e.g. `"Reuters"`) |
| `url` | str | Canonical URL — **dedup key** |
| `headline` | str | Article title |
| `summary` | str | First 500 chars of body/description |
| `sector` | str | See sector values below |
| `ticker` | str \| empty | Primary ticker mentioned (e.g. `"NVDA"`) |
| `entities` | JSON str | `["NVDA","semiconductors"]` — serialized list; consumers must call `json.loads()` on this column |
| `sentiment` | float \| empty | API-provided score in [−1, 1] |
| `sentiment_label` | str \| empty | `"positive"` \| `"neutral"` \| `"negative"` |
| `macro_tag` | str \| empty | See macro_tag values below |
| `market_impact_score` | float \| empty | Computed by `score_impact()` in `briefing.py` — **empty at ingest time**; backfilled each time `briefing.py` runs |

**Valid `sector` values:** `macro`, `equities`, `fintech`, `vc`, `energy`, `banking`, `crypto`, `real_estate`

**Valid `macro_tag` values:** `rate_cuts`, `earnings`, `layoffs`, `inflation`, `gdp`, `regulation`, `funding_round`, `ipo`

---

### File 1: `news_apis.py`

Follows the pattern of `sector_apis.py` and `crunchbase_apis.py`. All functions return `list[dict]` using the standard schema above; errors are logged and skipped (never crash the pipeline).

#### Constants

```python
NEWS_DATA_DIR = BASE_DIR / "data" / "FinancialNews"
RAW_CSV       = NEWS_DATA_DIR / "raw" / "news_articles.csv"

# Keyword → sector mapping (checked in order, first match wins)
SECTOR_KEYWORDS: dict[str, list[str]] = {
    "crypto":       ["bitcoin", "ethereum", "crypto", "defi", "blockchain"],
    "energy":       ["oil", "wti", "natural gas", "opec", "crude", "energy"],
    "vc":           ["venture capital", "funding round", "series a", "seed round", "startup"],
    "fintech":      ["fintech", "payments", "digital banking", "insurtech"],
    "banking":      ["federal reserve", "fed", "bank", "interest rate", "mortgage"],
    "real_estate":  ["real estate", "housing", "reit", "home price"],
    "equities":     ["earnings", "stock", "shares", "ipo", "market cap", "s&p"],
    "macro":        ["gdp", "inflation", "cpi", "pce", "unemployment", "recession"],
}

# Keyword → macro_tag mapping
MACRO_TAG_KEYWORDS: dict[str, list[str]] = {
    "rate_cuts":     ["rate cut", "rate hike", "fed funds", "fomc", "basis points"],
    "earnings":      ["earnings", "eps", "revenue beat", "quarterly results"],
    "layoffs":       ["layoffs", "job cuts", "workforce reduction", "downsizing"],
    "inflation":     ["inflation", "cpi", "pce", "price index", "deflation"],
    "gdp":           ["gdp", "gross domestic", "economic growth", "recession"],
    "regulation":    ["regulation", "sec", "compliance", "lawsuit", "antitrust"],
    "funding_round": ["series a", "series b", "seed round", "raised", "funding"],
    "ipo":           ["ipo", "initial public offering", "direct listing", "spac"],
}

# Source authority scores for impact scoring (used in briefing.py)
SOURCE_AUTHORITY: dict[str, float] = {
    "reuters": 1.0, "bloomberg": 1.0, "wall street journal": 0.95,
    "financial times": 0.95, "cnbc": 0.85, "associated press": 0.85,
    "marketwatch": 0.75, "seeking alpha": 0.55,
}
```

#### Functions

```python
def classify_sector(headline: str, ticker: str, entities: list[str]) -> str:
    """Return sector label via keyword matching. Default: 'macro'."""

def classify_macro_tag(headline: str) -> str | None:
    """Return macro_tag via keyword matching. Returns None if no match."""

def normalize_article(raw: dict, source_api: str) -> dict | None:
    """
    Map source-specific fields to standard schema.
    Returns None if headline or url is missing (skip the article).

    Field mapping per source:
      newsapi:   raw["title"]→headline, raw["description"]→summary,
                 raw["publishedAt"]→timestamp, raw["source"]["name"]→source_name
      marketaux: raw["title"]→headline, raw["description"]→summary,
                 raw["published_at"]→timestamp, raw["source"]→source_name,
                 raw["entities"][0]["symbol"]→ticker, raw["sentiment_score"]→sentiment
      finnhub:   raw["headline"]→headline, raw["summary"]→summary,
                 datetime.fromtimestamp(raw["datetime"])→timestamp,
                 raw["source"]→source_name, raw["related"]→ticker
    """

def fetch_newsapi(api_key: str, from_dt: datetime) -> list[dict]:
    """
    GET https://newsapi.org/v2/everything
    Queries: ['federal reserve', 'stock market', 'economic outlook',
              'inflation rate', 'earnings report', 'venture capital']
    One request per query (6 total). page_size=10 each.
    Sleep 1s between calls. On HTTP error: log warning and return [].
    """

def fetch_marketaux(api_key: str, from_dt: datetime) -> list[dict]:
    """
    GET https://api.marketaux.com/v1/news/all
    Params: filter_entities=true, language=en, published_after=from_dt.isoformat()
    One request per sector group: ['AAPL,MSFT,NVDA', 'JPM,BAC,GS', 'TSLA,XOM,CVX'].
    Sleep 1s between calls. On HTTP error: log warning and return [].
    """

def fetch_finnhub(api_key: str, from_dt: datetime) -> list[dict]:
    """
    GET https://finnhub.io/api/v1/news?category=general
    Params: token=api_key, from=from_dt.strftime('%Y-%m-%d')
    Single request. On HTTP error: log warning and return [].
    """

def normalize_url(url: str) -> str:
    """
    Normalize a URL for dedup comparison: lowercase scheme+host, strip query params
    and trailing slashes. Example: 'https://Reuters.com/article/foo?utm_source=x'
    → 'https://reuters.com/article/foo'
    """

def load_existing_urls(csv_path: Path) -> set[str]:
    """
    Read news_articles.csv and return the set of normalized URL values.
    Returns set() if file absent. Applies normalize_url() to each stored URL
    so near-duplicate URLs (tracking params, http vs https) are caught.
    """

def save_articles(articles: list[dict], csv_path: Path, daily_dir: Path, date: str) -> int:
    """
    Append articles to news_articles.csv (create with header if absent).
    Also write/overwrite data/FinancialNews/daily/{date}.csv.
    Returns count of rows written.

    Use a file lock (e.g. filelock.FileLock on csv_path.with_suffix('.lock')) before
    appending to prevent corruption if --news daily and --news realtime overlap.
    """

def refresh_news(api_keys: dict[str, str], mode: str = "daily",
                 since_hours: int = 24) -> dict[str, int]:
    """
    Orchestrator called by fred_refresh.py.
    mode='daily':    runs fetch_newsapi + fetch_marketaux + fetch_finnhub; since_hours=24
    mode='realtime': runs fetch_finnhub only; since_hours=since_hours (default 24)
    mode='all':      runs all 3 sources with since_hours=168 (7-day backfill window)
                     Distinct from 'daily': wider lookback, intended for manual/catch-up runs.

    Returns stats dict: {'fetched': N, 'new': M, 'sources': [...]}
    Dedupes by normalized URL against existing news_articles.csv.
    Creates NEWS_DATA_DIR/raw/ and NEWS_DATA_DIR/daily/ if absent.
    """
```

---

### File 2: `briefing.py`

Generates the daily briefing JSON. **No ML required — works from day 1.**

#### Market Impact Scoring

```python
def score_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'market_impact_score' column [0.0–1.0] using weighted rule formula:

    score = (
        0.25 * abs(sentiment)                          # sentiment magnitude
      + 0.20 * source_authority_score(source_name)     # source prestige
      + 0.20 * ticker_prominence(ticker)               # S&P 500 ticker = 1.0, unknown = 0.2
      + 0.20 * volume_spike_score(sector, date, df)    # >2σ above 30d sector avg
      + 0.15 * macro_tag_weight(macro_tag)             # rate_cuts/earnings=1.0, others=0.6
    )

    source_authority_score: lookup SOURCE_AUTHORITY dict (from news_apis.py), default 0.5
    ticker_prominence: S&P500_TICKERS set → 1.0; any non-empty ticker → 0.5; empty → 0.2
    volume_spike_score: requires ≥7 days of history; defaults to 0.5 if insufficient
    macro_tag_weight: {'rate_cuts': 1.0, 'earnings': 1.0, 'ipo': 0.8,
                       'funding_round': 0.7, None: 0.4}

    NOTE: This is the Phase 1 baseline scorer — rule-based, no ML required.
    Phase 2 does NOT replace this formula. Phase 2 adds yfinance/FMP enrichment
    signals (PE ratio, price momentum, EV/EBITDA) as additional input context for
    the briefing, but the weighted rule formula above remains the production scorer.
    """
```

#### Functions

```python
def load_articles(csv_path: Path, date: str | None = None) -> pd.DataFrame:
    """
    Load news_articles.csv. If date provided, filter to that date's rows only.
    Returns empty DataFrame (with correct columns) if file absent — no crash.
    """

def compute_sector_mood(df: pd.DataFrame) -> dict[str, float]:
    """
    Return {sector: avg_sentiment} for articles with non-null sentiment.
    Rounds to 3 decimal places. Omits sectors with zero articles.
    """

def generate_macro_signals(df: pd.DataFrame, regime_path: Path) -> list[str]:
    """
    Return list of plain-text bullet strings. Sources (filled in priority order):
    1. If regime_history.json exists: prepend current regime + FSI value (always first, max 1)
    2. For each macro_tag with ≥3 articles today: add a volume bullet (up to 3)
    3. For any sector_mood value outside [−0.3, 0.3]: add a sentiment bullet (fills remaining slots)
    Max 6 bullets total; earlier priorities are never crowded out by later ones.
    Returns [] if df is empty.
    """

def generate_alerts(df: pd.DataFrame, threshold: float = 0.75) -> list[str]:
    """
    Return headlines where market_impact_score >= threshold.
    Format: "{source_name}: {headline}" sorted by score descending.
    Max 5 alerts. Returns [] if no articles exceed threshold.
    """

def generate_daily_briefing(date: str | None = None,
                             data_dir: Path | None = None,
                             output_dir: Path | None = None,
                             regime_path: Path | None = None) -> dict:
    """
    Main briefing generator.
    date defaults to today (YYYY-MM-DD).
    Returns BriefingResponse-compatible dict:
    {
      "date": "2026-04-15",
      "generated_at": "2026-04-15T09:35:00",
      "article_count": 47,
      "top_stories": [...10 NewsArticle dicts sorted by market_impact_score...],
      "sector_mood": {"macro": 0.12, "equities": -0.08, ...},
      "macro_signals": ["FSI: 0.21 (expansion)", "3 earnings articles today", ...],
      "alerts": ["Reuters: Fed signals pause in rate hikes", ...]
    }
    Calls: load_articles → score_impact → compute_sector_mood →
           generate_macro_signals → generate_alerts
    """

def main() -> None:
    """
    CLI entry point.
    Calls output_dir.mkdir(parents=True, exist_ok=True) before writing.
    Writes outputs/daily_briefing_{YYYY-MM-DD}.json.
    Prints one-line summary: "[BRIEFING] 2026-04-15 — 47 articles, 3 alerts"
    """
```

---

### `fred_refresh.py` Changes

**1. Argparse (add alongside `--sector`, `--crunchbase`):**
```python
parser.add_argument(
    "--news",
    choices=["daily", "realtime", "all"],
    metavar="MODE",
    help="Refresh financial news (daily|realtime|all). Requires at least one news API key.",
)
parser.add_argument(
    "--skip-models",
    action="store_true",
    help="Skip model retraining and briefing generation after data ingestion.",
)
```

> **Note:** If `--skip-models` is already defined in the existing `fred_refresh.py`, omit the second block. Verify before adding.

**2. Env key loading (add after existing key loads):**
```python
NEWS_API_KEY      = os.getenv("NEWS_API_KEY", "")
MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY", "")
FINNHUB_API_KEY   = os.getenv("FINNHUB_API_KEY", "")
```

**3. Import (top of file, alongside sector_apis):**
```python
import news_apis
import briefing as briefing_mod
```

**4. News refresh step in `main()` — insert after FRED data refresh step, before model retraining:**
```python
if args.news:
    api_keys = {
        "newsapi":    NEWS_API_KEY,
        "marketaux":  MARKETAUX_API_KEY,
        "finnhub":    FINNHUB_API_KEY,
    }
    available = [k for k, v in api_keys.items() if v]
    if not available:
        log.warning("[NEWS] No news API keys set — skipping news refresh. Set at least one of NEWS_API_KEY, MARKETAUX_API_KEY, FINNHUB_API_KEY in .env")
    else:
        log.info("[%s/%s] Refresh financial news (%s) — sources: %s", step, n_steps, args.news, available)
        stats = news_apis.refresh_news(api_keys, mode=args.news)
        log.info("  fetched=%d  new=%d  sources=%s", stats["fetched"], stats["new"], stats["sources"])

        if not args.skip_models:
            log.info("[%s/%s] Generate daily briefing", step + 1, n_steps)
            briefing_mod.main()
```

**5. Step count:** Each `--news` run adds 1 step (2 if briefing generation is included).

---

### `api.py` Changes

**1. Pydantic models (add after existing Phase 0 models):**
```python
class NewsArticle(BaseModel):
    timestamp: str
    source_name: str
    sector: str
    ticker: str | None = None
    headline: str
    sentiment: float | None = None
    sentiment_label: str | None = None
    macro_tag: str | None = None
    market_impact_score: float | None = None

class BriefingResponse(BaseModel):
    date: str
    generated_at: str
    article_count: int
    top_stories: list[NewsArticle]
    sector_mood: dict[str, float]
    macro_signals: list[str]
    alerts: list[str]
    stale: bool = False   # True if briefing file is older than BRIEFING_STALE_HOURS (36h)
```

**2. Helper to find latest briefing (add near `_load_optional`):**
```python
BRIEFING_STALE_HOURS = 36  # briefing older than this is flagged stale in responses

def _latest_briefing_path() -> Path | None:
    """Return the most recent daily_briefing_*.json in OUTPUT_DIR, or None."""
    candidates = sorted(OUTPUT_DIR.glob("daily_briefing_*.json"), reverse=True)
    return candidates[0] if candidates else None

def _briefing_is_stale(path: Path) -> bool:
    """Return True if the briefing file's mtime is older than BRIEFING_STALE_HOURS."""
    age = datetime.utcnow() - datetime.utcfromtimestamp(path.stat().st_mtime)
    return age.total_seconds() > BRIEFING_STALE_HOURS * 3600
```

**3. OPTIONAL_RESULTS additions** — briefing is loaded dynamically (not a fixed path), so **no entry needed** in OPTIONAL_RESULTS. Handle directly in route handlers.

**4. _ENDPOINT_DESCRIPTORS additions:**
```python
# Phase 1 — Financial News
{
    "path": "/api/financial-news/briefing",
    "description": "Today's top stories, sector mood scores, macro signal bullets, and high-impact alerts. Generated by briefing.py after --news refresh.",
    "group": "Financial News (Phase 1)",
    "key": None,
    "optional": True,
},
{
    "path": "/api/financial-news/top-stories",
    "description": "Top 10 highest market-impact articles from today's news ingestion, sorted by impact score.",
    "group": "Financial News (Phase 1)",
    "key": None,
    "optional": True,
},
{
    "path": "/api/financial-news/alerts",
    "description": "High-impact articles (market_impact_score ≥ 0.75) from today's news. Returns [] if none exceed threshold.",
    "group": "Financial News (Phase 1)",
    "key": None,
    "optional": True,
},
```

**5. Route handlers:**
```python
@app.get("/api/financial-news/briefing", response_model=BriefingResponse,
         tags=["Financial News (Phase 1)"],
         summary="Daily financial news briefing")
def get_briefing() -> BriefingResponse:
    path = _latest_briefing_path()
    if path is None:
        raise HTTPException(status_code=404,
            detail="No briefing found. Run: python fred_refresh.py --news daily")
    try:
        data = json.loads(path.read_text())
        data["stale"] = _briefing_is_stale(path)
        return BriefingResponse(**data)
    except Exception:
        log.exception("Failed to parse briefing %s", path)
        raise HTTPException(status_code=500, detail="Failed to parse briefing file.")


@app.get("/api/financial-news/top-stories", response_model=list[NewsArticle],
         tags=["Financial News (Phase 1)"],
         summary="Top 10 market-impact articles today")
def get_top_stories() -> list[NewsArticle]:
    # Delegate to briefing — extract top_stories from latest briefing
    path = _latest_briefing_path()
    if path is None:
        raise HTTPException(status_code=404,
            detail="No briefing found. Run: python fred_refresh.py --news daily")
    try:
        data = json.loads(path.read_text())
        return [NewsArticle(**a) for a in data.get("top_stories", [])]
    except Exception:
        log.exception("Failed to parse briefing %s", path)
        raise HTTPException(status_code=500, detail="Failed to parse briefing file.")


@app.get("/api/financial-news/alerts", response_model=list[str],
         tags=["Financial News (Phase 1)"],
         summary="High-impact news alerts (score ≥ 0.75)")
def get_alerts() -> list[str]:
    path = _latest_briefing_path()
    if path is None:
        raise HTTPException(status_code=404,
            detail="No briefing found. Run: python fred_refresh.py --news daily")
    try:
        data = json.loads(path.read_text())
        return data.get("alerts", [])
    except Exception:
        log.exception("Failed to parse briefing %s", path)
        raise HTTPException(status_code=500, detail="Failed to parse briefing file.")
```

---

### `test_api.py` Changes

Add after the existing Phase 0 tests in `main()`:

```python
# ── Phase 1: Financial News ────────────────────────────────────────────────────
test_briefing()
test_top_stories()
test_alerts()
```

New test functions to add:

```python
def test_briefing() -> None:
    """GET /api/financial-news/briefing — optional, SKIPs on 404."""
    label = "GET /api/financial-news/briefing"
    r = get(f"{BASE}/api/financial-news/briefing")
    if r.status_code == 404:
        skip(label, "no briefing yet — run: python fred_refresh.py --news daily")
        return
    assert r.status_code == 200, f"{label} expected 200, got {r.status_code}"
    d = r.json()
    for key in ("date", "generated_at", "article_count", "top_stories",
                "sector_mood", "macro_signals", "alerts", "stale"):
        assert key in d, f"{label} missing key: {key}"
    assert isinstance(d["top_stories"], list), f"{label} top_stories not a list"
    assert isinstance(d["sector_mood"], dict), f"{label} sector_mood not a dict"
    assert isinstance(d["stale"], bool), f"{label} stale must be bool"
    ok(label, f"date={d['date']}  articles={d['article_count']}  stale={d['stale']}")

def test_top_stories() -> None:
    """GET /api/financial-news/top-stories — optional, SKIPs on 404."""
    label = "GET /api/financial-news/top-stories"
    r = get(f"{BASE}/api/financial-news/top-stories")
    if r.status_code == 404:
        skip(label, "no briefing yet")
        return
    assert r.status_code == 200, f"{label} expected 200, got {r.status_code}"
    stories = r.json()
    assert isinstance(stories, list), f"{label} expected list"
    if stories:
        first = stories[0]
        for key in ("headline", "sector", "source_name", "timestamp"):
            assert key in first, f"{label} story missing key: {key}"
    ok(label, f"count={len(stories)}")

def test_alerts() -> None:
    """GET /api/financial-news/alerts — optional, SKIPs on 404. Always returns 200."""
    label = "GET /api/financial-news/alerts"
    r = get(f"{BASE}/api/financial-news/alerts")
    if r.status_code == 404:
        skip(label, "no briefing yet")
        return
    assert r.status_code == 200, f"{label} expected 200, got {r.status_code}"
    alerts = r.json()
    assert isinstance(alerts, list), f"{label} expected list[str]"
    if alerts:
        assert isinstance(alerts[0], str), f"{label} items must be str"
    ok(label, f"alerts={len(alerts)}")
```

Update `test_summary()` threshold: `>= 25` → `>= 28` (3 new descriptors added).

---

### Implementation Order

1. Write `news_apis.py` — constants, `normalize_url`, classify helpers, 3 fetchers, `normalize_article`, `load_existing_urls`, `save_articles` (with filelock), `refresh_news`
2. Write `briefing.py` — `score_impact`, `load_articles`, `compute_sector_mood`, `generate_macro_signals` (priority-ordered), `generate_alerts`, `generate_daily_briefing`, `main` (mkdir outputs/)
3. Update `fred_refresh.py` — argparse (`--news`, verify `--skip-models`), env keys, imports, news refresh step in main()
4. Update `api.py` — Pydantic models (incl. `stale` field), `_latest_briefing_path`, `_briefing_is_stale`, endpoint descriptors, 3 route handlers (all with try/except)
5. Update `test_api.py` — `test_briefing`, `test_top_stories`, `test_alerts`; update threshold to ≥28

### Critical Files to Modify

| File | Change |
|---|---|
| `news_apis.py` | **New file** — all news fetching and normalization |
| `briefing.py` | **New file** — briefing generation and impact scoring |
| `fred_refresh.py` | Add `--news` argparse, env key loads, refresh step in main() |
| `api.py` | NewsArticle + BriefingResponse models, _latest_briefing_path helper, 3 route handlers, 3 endpoint descriptors |
| `test_api.py` | 3 new test functions + 3 calls in main(); threshold 25→28 |

---

