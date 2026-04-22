"""
news_apis.py — Financial news ingestion and normalization (Phase 1).

Fetches from up to three sources (NewsAPI, Marketaux, Finnhub),
normalizes to a common schema, deduplicates by normalized URL, and
appends to data/FinancialNews/raw/news_articles.csv.

Called by fred_refresh.py with --news daily|realtime|all.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests

try:
    from newsapi import NewsApiClient as _NewsApiClient  # type: ignore[import]
except ImportError:
    _NewsApiClient = None  # type: ignore[assignment,misc]

try:
    from filelock import FileLock as _FileLock
except ImportError:
    _FileLock = None  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)

BASE_DIR          = Path(__file__).parent
NEWS_DATA_DIR     = BASE_DIR / "data" / "FinancialNews"
RAW_CSV           = NEWS_DATA_DIR / "raw" / "news_articles.csv"
SENTIMENT_AGG_JSON        = NEWS_DATA_DIR / "raw" / "marketaux_sentiment_agg.json"
SENTIMENT_AGG_PARSED_JSON = NEWS_DATA_DIR / "raw" / "marketaux_sentiment_parsed.json"
FINNHUB_SENT_JSON         = NEWS_DATA_DIR / "raw" / "finnhub_sentiment.json"
FINNHUB_SENT_PARSED_JSON  = NEWS_DATA_DIR / "raw" / "finnhub_sentiment_parsed.json"

MARKETAUX_AGG_URL = "https://api.marketaux.com/v1/entity/stats/aggregation"

FINNHUB_MARKET_CATEGORIES = ["general", "forex", "crypto", "merger"]
FINNHUB_DEFAULT_SYMBOLS   = ["AAPL", "MSFT", "NVDA", "JPM", "TSLA", "XOM", "AMZN", "GOOGL"]

# Ordered CSV column schema
CSV_COLUMNS = [
    "timestamp", "ingested_at", "source_api", "source_name",
    "url", "headline", "summary", "sector", "ticker", "entities",
    "sentiment", "sentiment_label", "macro_tag", "market_impact_score",
]

# Keyword → sector mapping (checked in order; first match wins)
SECTOR_KEYWORDS: dict[str, list[str]] = {
    "crypto":      ["bitcoin", "ethereum", "crypto", "defi", "blockchain"],
    "energy":      ["oil", "wti", "natural gas", "opec", "crude", "energy"],
    "vc":          ["venture capital", "funding round", "series a", "seed round", "startup"],
    "fintech":     ["fintech", "payments", "digital banking", "insurtech"],
    "banking":     ["federal reserve", "fed", "bank", "interest rate", "mortgage"],
    "real_estate": ["real estate", "housing", "reit", "home price"],
    "equities":    ["earnings", "stock", "shares", "ipo", "market cap", "s&p"],
    "macro":       ["gdp", "inflation", "cpi", "pce", "unemployment", "recession"],
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

# Source authority scores (used by briefing.py score_impact)
SOURCE_AUTHORITY: dict[str, float] = {
    "reuters":             1.0,
    "bloomberg":           1.0,
    "wall street journal": 0.95,
    "financial times":     0.95,
    "cnbc":                0.85,
    "associated press":    0.85,
    "marketwatch":         0.75,
    "seeking alpha":       0.55,
}


# ── URL normalization ──────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """
    Normalize a URL for dedup comparison: lowercase scheme+host, strip query
    params and fragment, strip trailing slash.
    'https://Reuters.com/article/foo?utm_source=x' → 'https://reuters.com/article/foo'
    """
    try:
        p = urlparse(url.strip())
        return urlunparse((
            p.scheme.lower(),
            p.netloc.lower(),
            p.path.rstrip("/"),
            "",   # params
            "",   # query — drop all tracking params
            "",   # fragment
        ))
    except Exception:
        return url.strip().lower()


# ── Classification helpers ─────────────────────────────────────────────────────

def classify_sector(headline: str, ticker: str, entities: list[str]) -> str:
    """Return sector label via keyword matching. Default: 'macro'."""
    text = " ".join([headline, ticker] + entities).lower()
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return sector
    return "macro"


def classify_macro_tag(headline: str) -> str | None:
    """Return macro_tag via keyword matching. Returns None if no match."""
    text = headline.lower()
    for tag, keywords in MACRO_TAG_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return tag
    return None


# ── Article normalization ──────────────────────────────────────────────────────

def normalize_article(raw: dict, source_api: str) -> dict | None:
    """
    Map source-specific fields to the standard schema.
    Returns None if headline or url is missing (skip the article).
    entities column is stored as a JSON-serialized list — consumers must
    call json.loads() when reading it back.
    """
    try:
        if source_api == "newsapi":
            headline        = (raw.get("title") or "").strip()
            url             = (raw.get("url") or "").strip()
            summary         = (raw.get("description") or "")[:500]
            timestamp       = raw.get("publishedAt", "")
            source_name     = (raw.get("source") or {}).get("name", "")
            ticker          = ""
            entities        = []
            sentiment       = None
            sentiment_label = None

        elif source_api == "marketaux":
            headline    = (raw.get("title") or "").strip()
            url         = (raw.get("url") or "").strip()
            summary     = (raw.get("description") or "")[:500]
            timestamp   = raw.get("published_at", "")
            source_name = raw.get("source", "")
            ent_list    = raw.get("entities") or []

            # Primary ticker = entity with highest match_score
            if ent_list:
                primary = max(ent_list, key=lambda e: e.get("match_score") or 0.0)
                ticker = primary.get("symbol", "")
            else:
                ticker = ""
            entities = [e.get("symbol", "") for e in ent_list if e.get("symbol")]

            # Sentiment: match-score-weighted average of per-entity sentiment_score.
            # The API returns sentiment_score on each entity, not at article level.
            scored = [
                (e.get("sentiment_score", 0.0) or 0.0, e.get("match_score", 1.0) or 1.0)
                for e in ent_list
                if e.get("sentiment_score") is not None
            ]
            if scored:
                total_weight = sum(w for _, w in scored)
                sentiment = sum(s * w for s, w in scored) / total_weight if total_weight else None
            else:
                sentiment = None

            if sentiment is not None:
                sentiment_label = (
                    "positive" if sentiment > 0.05
                    else "negative" if sentiment < -0.05
                    else "neutral"
                )
            else:
                sentiment_label = None

        elif source_api == "finnhub":
            headline    = (raw.get("headline") or "").strip()
            url         = (raw.get("url") or "").strip()
            summary     = (raw.get("summary") or "")[:500]
            ts_epoch    = raw.get("datetime")
            timestamp   = (
                datetime.utcfromtimestamp(ts_epoch).isoformat() + "Z"
                if ts_epoch else ""
            )
            source_name     = raw.get("source", "")
            ticker          = raw.get("related", "")
            entities        = [ticker] if ticker else []
            sentiment       = None
            sentiment_label = None

        else:
            return None

        if not headline or not url:
            return None

        sector    = classify_sector(headline, ticker, entities)
        macro_tag = classify_macro_tag(headline)

        return {
            "timestamp":           timestamp,
            "ingested_at":         datetime.utcnow().isoformat() + "Z",
            "source_api":          source_api,
            "source_name":         source_name,
            "url":                 url,
            "headline":            headline,
            "summary":             summary,
            "sector":              sector,
            "ticker":              ticker,
            "entities":            json.dumps(entities),
            "sentiment":           sentiment,
            "sentiment_label":     sentiment_label,
            "macro_tag":           macro_tag,
            "market_impact_score": "",   # backfilled by briefing.py score_impact()
        }

    except Exception as exc:
        log.warning("[NEWS] normalize_article(%s) error: %s", source_api, exc)
        return None


# ── API fetchers ───────────────────────────────────────────────────────────────

def fetch_newsapi(api_key: str, from_dt: datetime) -> list[dict]:
    """
    Fetch from NewsAPI /v2/everything using the newsapi-python client when
    available, falling back to raw requests if the library is not installed.

    6 queries × 10 articles each. Sleeps 1s between calls.
    On error: logs warning and returns partial results.
    """
    queries = [
        "federal reserve", "stock market", "economic outlook",
        "inflation rate", "earnings report", "venture capital",
    ]
    articles: list[dict] = []
    from_str = from_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    if _NewsApiClient is not None:
        # ── newsapi-python client path ────────────────────────────────────────
        client = _NewsApiClient(api_key=api_key)
        for q in queries:
            try:
                result = client.get_everything(
                    q=q,
                    from_param=from_str,
                    language="en",
                    sort_by="publishedAt",
                    page_size=10,
                )
                for raw in (result or {}).get("articles", []):
                    art = normalize_article(raw, "newsapi")
                    if art:
                        articles.append(art)
            except Exception as exc:
                log.warning("[NEWS] NewsAPI (client) query %r failed: %s", q, exc)
            time.sleep(1)
    else:
        # ── Raw requests fallback ─────────────────────────────────────────────
        log.warning("[NEWS] newsapi-python not installed — falling back to raw requests")
        for q in queries:
            try:
                resp = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q":        q,
                        "from":     from_str,
                        "pageSize": 10,
                        "language": "en",
                        "sortBy":   "publishedAt",
                        "apiKey":   api_key,
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                for raw in resp.json().get("articles", []):
                    art = normalize_article(raw, "newsapi")
                    if art:
                        articles.append(art)
            except Exception as exc:
                log.warning("[NEWS] NewsAPI (requests) query %r failed: %s", q, exc)
            time.sleep(1)

    log.info("[NEWS] NewsAPI: %d articles fetched", len(articles))
    return articles


def fetch_marketaux(api_key: str, from_dt: datetime) -> list[dict]:
    """
    GET https://api.marketaux.com/v1/news/all
    Fetches US financial news with entity filtering across 3 pages (10 articles
    each). published_after uses %Y-%m-%dT%H:%M format as required by the API.
    On HTTP error: logs warning and returns partial results.
    """
    articles: list[dict] = []
    from_str = from_dt.strftime("%Y-%m-%dT%H:%M")
    for page in range(1, 4):
        try:
            resp = requests.get(
                "https://api.marketaux.com/v1/news/all",
                params={
                    "countries":       "us",
                    "filter_entities": "true",
                    "limit":           10,
                    "page":            page,
                    "published_after": from_str,
                    "api_token":       api_key,
                },
                timeout=15,
            )
            resp.raise_for_status()
            batch = resp.json().get("data", [])
            for raw in batch:
                art = normalize_article(raw, "marketaux")
                if art:
                    articles.append(art)
            if len(batch) < 10:
                break   # no more pages
        except Exception as exc:
            log.warning("[NEWS] Marketaux page %d failed: %s", page, exc)
        time.sleep(1)
    log.info("[NEWS] Marketaux: %d articles fetched", len(articles))
    return articles


def _safe_float(val: object) -> float | None:
    try:
        return float(val) if val is not None else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _safe_int(val: object) -> int | None:
    try:
        return int(val) if val is not None else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def fetch_finnhub(api_key: str, from_dt: datetime) -> list[dict]:
    """
    GET https://finnhub.io/api/v1/news for all 4 market news categories:
    general, forex, crypto, merger. One request per category, 0.5s sleep
    between calls. On HTTP error per category: logs warning and continues.
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


def parse_finnhub_sentiment(payload: list[dict]) -> list[dict]:
    """
    Parse a list of Finnhub /api/v1/news-sentiment responses (one per symbol).

    Each element of payload is the raw dict returned by the API:
        {
            "symbol": "AAPL",
            "sentiment": {"bullishPercent": 0.6, "bearishPercent": 0.4},
            "companyNewsScore": 0.72,
            "sectorAverageNewsScore": 0.61,
            "buzz": {"articlesInLastWeek": 142, "weeklyAverage": 110.5}
        }

    Returns a list of clean dicts with keys:
        symbol, bullish_pct, bearish_pct, company_news_score,
        sector_avg_score, articles_last_week, weekly_avg
    """
    rows = []
    for item in payload:
        sentiment_block = item.get("sentiment") or {}
        buzz_block      = item.get("buzz") or {}
        rows.append({
            "symbol":             item.get("symbol", ""),
            "bullish_pct":        _safe_float(sentiment_block.get("bullishPercent")),
            "bearish_pct":        _safe_float(sentiment_block.get("bearishPercent")),
            "company_news_score": _safe_float(item.get("companyNewsScore")),
            "sector_avg_score":   _safe_float(item.get("sectorAverageNewsScore")),
            "articles_last_week": _safe_int(buzz_block.get("articlesInLastWeek")),
            "weekly_avg":         _safe_float(buzz_block.get("weeklyAverage")),
        })
    return rows


def fetch_finnhub_company_news(
    api_key: str,
    from_dt: datetime,
    symbols: list[str] | None = None,
) -> list[dict]:
    """
    GET https://finnhub.io/api/v1/company-news for a list of tickers.
    Response shape is identical to /api/v1/news, so normalize_article("finnhub")
    is reused unchanged. Duplicate articles are deduped at write time by
    save_articles(). One request per symbol, 0.5s sleep between calls.
    """
    if symbols is None:
        symbols = FINNHUB_DEFAULT_SYMBOLS

    articles: list[dict] = []
    from_str = from_dt.strftime("%Y-%m-%d")
    to_str   = datetime.utcnow().strftime("%Y-%m-%d")

    for symbol in symbols:
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    "symbol": symbol,
                    "from":   from_str,
                    "to":     to_str,
                    "token":  api_key,
                },
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
            log.warning("[NEWS] Finnhub company news symbol=%r failed: %s", symbol, exc)
        time.sleep(0.5)

    log.info("[NEWS] Finnhub company news: %d articles fetched across %d symbols",
             len(articles), len(symbols))
    return articles


def fetch_finnhub_sentiment(
    api_key: str,
    symbols: list[str] | None = None,
) -> list[dict]:
    """
    GET https://finnhub.io/api/v1/news-sentiment for a list of tickers.
    Saves raw API responses to FINNHUB_SENT_JSON and parsed records to
    FINNHUB_SENT_PARSED_JSON. One request per symbol, 0.5s sleep between calls.
    Returns the parsed list of dicts, or [] on complete failure.
    """
    if symbols is None:
        symbols = FINNHUB_DEFAULT_SYMBOLS

    raw_responses: list[dict] = []
    for symbol in symbols:
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/news-sentiment",
                params={"symbol": symbol, "token": api_key},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                raw_responses.append(data)
        except Exception as exc:
            log.warning("[NEWS] Finnhub sentiment symbol=%r failed: %s", symbol, exc)
        time.sleep(0.5)

    parsed = parse_finnhub_sentiment(raw_responses)

    for row in parsed:
        log.info(
            "[NEWS] Finnhub sentiment — symbol=%s  bullish=%.2f  bearish=%.2f  score=%.4f",
            row["symbol"],
            row["bullish_pct"] or 0.0,
            row["bearish_pct"] or 0.0,
            row["company_news_score"] or 0.0,
        )

    try:
        FINNHUB_SENT_JSON.parent.mkdir(parents=True, exist_ok=True)
        FINNHUB_SENT_JSON.write_text(json.dumps(raw_responses, indent=2), encoding="utf-8")
        FINNHUB_SENT_PARSED_JSON.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        log.info("[NEWS] Finnhub sentiment: %d symbols saved → %s",
                 len(parsed), FINNHUB_SENT_PARSED_JSON)
    except Exception as exc:
        log.warning("[NEWS] Could not save Finnhub sentiment JSON: %s", exc)

    return parsed


# ── CSV helpers ────────────────────────────────────────────────────────────────

def load_existing_urls(csv_path: Path) -> set[str]:
    """
    Read news_articles.csv and return the set of normalized URL values.
    Applies normalize_url() so near-duplicate URLs (tracking params,
    http vs https) are caught. Returns set() if file absent.
    """
    if not csv_path.exists():
        return set()
    urls: set[str] = set()
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw_url = row.get("url", "")
                if raw_url:
                    urls.add(normalize_url(raw_url))
    except Exception as exc:
        log.warning("[NEWS] Could not read existing URLs from %s: %s", csv_path, exc)
    return urls


def save_articles(articles: list[dict], csv_path: Path,
                  daily_dir: Path, date: str) -> int:
    """
    Append new-only articles to news_articles.csv (create with header if absent).
    Also write/overwrite data/FinancialNews/daily/{date}.csv.
    Uses a FileLock (if filelock is installed) to prevent concurrent write
    corruption when --news daily and --news realtime crons overlap.
    Returns count of rows written.
    """
    if not articles:
        return 0

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)

    def _do_write() -> int:
        existing_urls = load_existing_urls(csv_path)
        new_articles  = [
            a for a in articles
            if normalize_url(a.get("url", "")) not in existing_urls
        ]
        if not new_articles:
            return 0

        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(new_articles)

        # Daily partition — always overwrite with today's fresh articles
        daily_path    = daily_dir / f"{date}.csv"
        today_articles = [
            a for a in new_articles
            if (a.get("timestamp") or "").startswith(date)
        ]
        with open(daily_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(today_articles)

        return len(new_articles)

    if _FileLock is not None:
        lock_path = csv_path.with_suffix(".lock")
        with _FileLock(str(lock_path), timeout=30):
            return _do_write()
    return _do_write()


# ── Marketaux sentiment aggregation ───────────────────────────────────────────

def parse_marketaux_sentiment_agg(payload: dict) -> list[dict]:
    """
    Parse a Marketaux /v1/entity/stats/aggregation response.

    Expected structure:
        {
            "meta": {"returned": 1, "limit": 10},
            "data": [
                {"key": "us", "total_documents": 1581, "sentiment_avg": 0.2347}
            ]
        }

    Returns a list of clean dicts with keys:
        key               — aggregation group value (e.g. "us", "Technology")
        total_documents   — article count in the window
        sentiment_avg     — average sentiment score [-1, 1]
        sentiment_label   — "positive" / "negative" / "neutral"
    """
    rows = []
    for item in payload.get("data", []):
        key        = item.get("key", "")
        total_docs = int(item.get("total_documents") or 0)
        sent_avg   = item.get("sentiment_avg")
        if sent_avg is not None:
            sent_avg = float(sent_avg)
            label = (
                "positive" if sent_avg > 0.05
                else "negative" if sent_avg < -0.05
                else "neutral"
            )
        else:
            label = None
        rows.append({
            "key":             key,
            "total_documents": total_docs,
            "sentiment_avg":   sent_avg,
            "sentiment_label": label,
        })
    return rows


def fetch_marketaux_sentiment_agg(
    api_key: str,
    from_dt: datetime,
    group_by: str = "country",
    limit: int = 10,
) -> dict:
    """
    GET https://api.marketaux.com/v1/entity/stats/aggregation

    Fetches pre-aggregated sentiment stats, parses the response, and writes
    two files:
      - SENTIMENT_AGG_JSON        — raw API response (meta + data)
      - SENTIMENT_AGG_PARSED_JSON — parsed list of {key, total_documents,
                                     sentiment_avg, sentiment_label}

    Response structure:
        {"meta": {"returned": N, "limit": 10},
         "data": [{"key": "us", "total_documents": 1581, "sentiment_avg": 0.23}]}

    Parameters
    ----------
    group_by : Aggregation dimension — "country", "industry", "entity_type", etc.
    limit    : Max rows returned. Default 10.

    Returns the raw payload dict, or {} on any error (never raises).
    """
    from_str = from_dt.strftime("%Y-%m-%dT%H:%M")
    try:
        resp = requests.get(
            MARKETAUX_AGG_URL,
            params={
                "countries":       "us",
                "group_by":        group_by,
                "limit":           limit,
                "published_after": from_str,
                "api_token":       api_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        log.warning("[NEWS] Marketaux sentiment aggregation failed: %s", exc)
        return {}

    # Parse into clean records
    parsed = parse_marketaux_sentiment_agg(payload)
    for row in parsed:
        log.info(
            "[NEWS] Marketaux sentiment agg — key=%s  docs=%d  sentiment_avg=%.4f (%s)",
            row["key"], row["total_documents"],
            row["sentiment_avg"] if row["sentiment_avg"] is not None else 0.0,
            row["sentiment_label"],
        )

    # Persist raw + parsed
    try:
        SENTIMENT_AGG_JSON.parent.mkdir(parents=True, exist_ok=True)
        SENTIMENT_AGG_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        SENTIMENT_AGG_PARSED_JSON.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        log.info(
            "[NEWS] Marketaux sentiment agg (%s): %d rows saved → %s",
            group_by, len(parsed), SENTIMENT_AGG_PARSED_JSON,
        )
    except Exception as exc:
        log.warning("[NEWS] Could not save sentiment agg JSON: %s", exc)

    return payload


# ── Orchestrator ───────────────────────────────────────────────────────────────

def refresh_news(api_keys: dict[str, str], mode: str = "daily",
                 since_hours: int = 24) -> dict:
    """
    Orchestrator called by fred_refresh.py.

    mode='daily':    all 3 sources; since_hours=24
    mode='realtime': Finnhub only; since_hours=since_hours (default 24)
    mode='all':      all 3 sources; since_hours=168 (7-day backfill window)
                     Distinct from 'daily': wider lookback for manual catch-up runs.

    Returns: {'fetched': N, 'new': M, 'sources': [...active source names...]}
    Deduplicates by normalized URL against existing news_articles.csv.
    Creates NEWS_DATA_DIR/raw/ and NEWS_DATA_DIR/daily/ if absent.
    """
    if mode == "all":
        since_hours = 168

    from_dt  = datetime.utcnow() - timedelta(hours=since_hours)
    date_str = datetime.utcnow().strftime("%Y-%m-%d")

    articles: list[dict]  = []
    active_sources: list[str] = []

    newsapi_key   = api_keys.get("newsapi", "")
    marketaux_key = api_keys.get("marketaux", "")
    finnhub_key   = api_keys.get("finnhub", "")

    sentiment_agg: dict = {}

    if mode in ("daily", "all"):
        if newsapi_key:
            active_sources.append("newsapi")
            articles.extend(fetch_newsapi(newsapi_key, from_dt))
        if marketaux_key:
            active_sources.append("marketaux")
            articles.extend(fetch_marketaux(marketaux_key, from_dt))
            sentiment_agg = fetch_marketaux_sentiment_agg(marketaux_key, from_dt)

    finnhub_sentiment: list[dict] = []

    if finnhub_key:
        active_sources.append("finnhub")
        articles.extend(fetch_finnhub(finnhub_key, from_dt))
        articles.extend(fetch_finnhub_company_news(finnhub_key, from_dt))
        finnhub_sentiment = fetch_finnhub_sentiment(finnhub_key)

    fetched   = len(articles)
    daily_dir = NEWS_DATA_DIR / "daily"
    new_count = save_articles(articles, RAW_CSV, daily_dir, date_str)

    log.info(
        "[NEWS] refresh_news(%s): fetched=%d  new=%d  sources=%s",
        mode, fetched, new_count, active_sources,
    )
    return {
        "fetched":           fetched,
        "new":               new_count,
        "sources":           active_sources,
        "sentiment_agg":     sentiment_agg,
        "finnhub_sentiment": finnhub_sentiment,
    }
