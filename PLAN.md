# Finnhub API Expansion Plan

## Context

The current Finnhub integration makes a single call to `/api/v1/news?category=general`. The Finnhub API also provides:
- `/api/v1/news` with 4 categories (general, forex, crypto, merger)
- `/api/v1/company-news` for ticker-specific articles (same response shape)
- `/api/v1/news-sentiment` for per-symbol sentiment scores (bullish/bearish %, company news score, buzz stats)

This plan expands the integration to use all three endpoints, following the exact same patterns already established for Marketaux in `news_apis.py`.

**Only file modified:** `news_apis.py`
`fred_refresh.py` needs no changes — it only reads `fetched`, `new`, `sources` from `refresh_news()`.

---

## Critical File

`news_apis.py`

Existing patterns to reuse exactly:
- `SENTIMENT_AGG_JSON` / `SENTIMENT_AGG_PARSED_JSON` constant naming (lines 38-39)
- `parse_marketaux_sentiment_agg()` — parse function structure
- `fetch_marketaux_sentiment_agg()` — fetch → parse → save raw + parsed → log each row
- `sentiment_agg: dict = {}` initializer pattern in `refresh_news()`
- `time.sleep(1)` between calls (use `0.5` for Finnhub — 20 total calls per run, within free tier)

---

## Step-by-Step Implementation (sequential order)

### Step 1 — Add constants (after existing `SENTIMENT_AGG_PARSED_JSON` line ~39)

```python
FINNHUB_SENT_JSON        = NEWS_DATA_DIR / "raw" / "finnhub_sentiment.json"
FINNHUB_SENT_PARSED_JSON = NEWS_DATA_DIR / "raw" / "finnhub_sentiment_parsed.json"

FINNHUB_MARKET_CATEGORIES = ["general", "forex", "crypto", "merger"]
FINNHUB_DEFAULT_SYMBOLS   = ["AAPL", "MSFT", "NVDA", "JPM", "TSLA", "XOM", "AMZN", "GOOGL"]
```

### Step 2 — Add private helpers (just before the Finnhub fetchers section)

```python
def _safe_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None

def _safe_int(val) -> int | None:
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None
```

### Step 3 — Expand `fetch_finnhub()` (replace existing single-category function)

Loop over `FINNHUB_MARKET_CATEGORIES`, one request per category, `time.sleep(0.5)` between calls. Same params (`token`, `from` as `%Y-%m-%d`), same `normalize_article("finnhub")` normalization — no changes needed to the normalizer.

```python
def fetch_finnhub(api_key: str, from_dt: datetime) -> list[dict]:
    """
    GET https://finnhub.io/api/v1/news for 4 market news categories.
    One request per category, 0.5s sleep between calls.
    """
    articles: list[dict] = []
    from_str = from_dt.strftime("%Y-%m-%d")
    for category in FINNHUB_MARKET_CATEGORIES:
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/news",
                params={"category": category, "token": api_key, "from": from_str},
                timeout=15,
            )
            resp.raise_for_status()
            raw_list = resp.json()
            if isinstance(raw_list, list):
                for raw in raw_list:
                    art = normalize_article(raw, "finnhub")
                    if art:
                        articles.append(art)
        except Exception as exc:
            log.warning("[NEWS] Finnhub market news category=%r failed: %s", category, exc)
        time.sleep(0.5)
    log.info("[NEWS] Finnhub market news: %d articles fetched", len(articles))
    return articles
```

### Step 4 — Add `parse_finnhub_sentiment(payload: list[dict]) -> list[dict]`

Mirrors `parse_marketaux_sentiment_agg()`. Input: list of raw per-symbol API dicts. Output: one clean dict per symbol.

```python
# Input per symbol (raw API response):
# {"symbol": "AAPL", "sentiment": {"bullishPercent": 0.6, "bearishPercent": 0.4},
#  "companyNewsScore": 0.72, "sectorAverageNewsScore": 0.61,
#  "buzz": {"articlesInLastWeek": 142, "weeklyAverage": 110.5}}

# Output per symbol:
{
    "symbol":             str,
    "bullish_pct":        float | None,   # sentiment.bullishPercent
    "bearish_pct":        float | None,   # sentiment.bearishPercent
    "company_news_score": float | None,   # companyNewsScore
    "sector_avg_score":   float | None,   # sectorAverageNewsScore
    "articles_last_week": int   | None,   # buzz.articlesInLastWeek
    "weekly_avg":         float | None,   # buzz.weeklyAverage
}
```

Uses `_safe_float()` / `_safe_int()` helpers from Step 2.

### Step 5 — Add `fetch_finnhub_company_news(api_key, from_dt, symbols=None) -> list[dict]`

Calls `/api/v1/company-news?symbol=X&from=YYYY-MM-DD&to=YYYY-MM-DD&token=KEY` for each ticker in `FINNHUB_DEFAULT_SYMBOLS`. Response shape is identical to `/api/v1/news`, so `normalize_article("finnhub")` is reused unchanged (`related` field contains the symbol). `time.sleep(0.5)` between calls. Duplicates deduped automatically by `save_articles()` at write time.

### Step 6 — Add `fetch_finnhub_sentiment(api_key, symbols=None) -> list[dict]`

Calls `/api/v1/news-sentiment?symbol=X&token=KEY` per ticker. Collects raw dicts, calls `parse_finnhub_sentiment()`, logs each row, saves:
- `FINNHUB_SENT_JSON` — raw list of per-symbol API responses
- `FINNHUB_SENT_PARSED_JSON` — parsed list of clean dicts

Returns the parsed list (mirrors `fetch_marketaux_sentiment_agg` pattern).

### Step 7 — Wire into `refresh_news()`

```python
# Initialize before the if-blocks:
finnhub_sentiment: list[dict] = []

# Replace the existing finnhub block:
if finnhub_key:
    active_sources.append("finnhub")
    articles.extend(fetch_finnhub(finnhub_key, from_dt))
    articles.extend(fetch_finnhub_company_news(finnhub_key, from_dt))
    finnhub_sentiment = fetch_finnhub_sentiment(finnhub_key)

# Add to return dict:
return {
    "fetched":           fetched,
    "new":               new_count,
    "sources":           active_sources,
    "sentiment_agg":     sentiment_agg,       # existing Marketaux key
    "finnhub_sentiment": finnhub_sentiment,   # new
}
```

---

## Rate Limit Note

Free tier: 60 calls/minute. With 0.5s sleep: 4 category + 8 company news + 8 sentiment = **20 calls per refresh** — well within limits.

---

## New Output Files

| File | Contents |
|---|---|
| `data/FinancialNews/raw/finnhub_sentiment.json` | Raw per-symbol API responses (list of dicts) |
| `data/FinancialNews/raw/finnhub_sentiment_parsed.json` | Clean parsed records: `{symbol, bullish_pct, bearish_pct, company_news_score, sector_avg_score, articles_last_week, weekly_avg}` |

---

## Verification

```bash
# 1. Syntax check
python3 -c "import ast; ast.parse(open('news_apis.py').read()); print('OK')"

# 2. Parser unit test (no network needed)
python3 -c "
from news_apis import parse_finnhub_sentiment
sample = [{'symbol':'AAPL','sentiment':{'bullishPercent':0.6,'bearishPercent':0.4},
           'companyNewsScore':0.72,'sectorAverageNewsScore':0.61,
           'buzz':{'articlesInLastWeek':142,'weeklyAverage':110.5}}]
print(parse_finnhub_sentiment(sample))
"

# 3. Live run
source .env && python3 fred_refresh.py --news daily

# 4. Check output files
ls data/FinancialNews/raw/finnhub_sentiment*.json
cat data/FinancialNews/raw/finnhub_sentiment_parsed.json
```
