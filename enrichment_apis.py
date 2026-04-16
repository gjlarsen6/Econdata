"""
enrichment_apis.py — Phase 2 article enrichment: yfinance + FMP signals.

Attaches financial signals to ticker-tagged news articles from news_articles.csv
and writes the enriched output to data/FinancialNews/enriched/news_enriched.csv.

Standalone:
    python3 enrichment_apis.py
    python3 enrichment_apis.py --fmp-key YOUR_KEY

Via fred_refresh.py:
    python3 fred_refresh.py --news daily --enrich
"""

from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

BASE_DIR     = Path(__file__).parent
RAW_CSV      = BASE_DIR / "data" / "FinancialNews" / "raw" / "news_articles.csv"
ENRICHED_CSV = BASE_DIR / "data" / "FinancialNews" / "enriched" / "news_enriched.csv"

FMP_BUDGET     = 83    # max unique tickers/day on FMP free tier
YFINANCE_SLEEP = 0.5   # seconds between yfinance calls
FMP_BASE       = "https://financialmodelingprep.com/api/v3/key-metrics"

# S&P 500 tickers that get FMP priority (mirrors briefing.py _SP500_TICKERS)
try:
    from briefing import _SP500_TICKERS
except ImportError:
    _SP500_TICKERS: frozenset[str] = frozenset([
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA",
        "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
        "ABBV", "PEP", "KO", "AVGO", "COST", "BAC", "WMT", "LLY", "MCD",
        "CSCO", "ACN", "CRM", "DHR", "TMO", "NEE", "ADBE", "NKE", "DIS",
        "ORCL", "VZ", "INTC", "NFLX", "IBM", "GS", "MS", "C", "WFC",
        "USB", "AXP", "SCHW", "BLK", "GE", "CAT", "DE", "MMM", "BA",
        "RTX", "LMT", "NOC", "GD", "HON",
    ])

# Graceful import — yfinance is optional
try:
    import yfinance as _yf
except ImportError:
    _yf = None  # type: ignore[assignment]

# Columns added by enrichment (None where unavailable)
ENRICHED_EXTRA_COLS: list[str] = [
    "trailing_pe", "market_cap", "eps_trailing",
    "week52_high", "week52_low", "price_momentum_30d",
    "ev_ebitda", "revenue_growth_yoy", "debt_to_equity",
]


# ── yfinance signals ──────────────────────────────────────────────────────────

def fetch_yfinance_signals(ticker: str) -> dict:
    """
    Fetch key signals for ticker using yfinance.

    Returns a dict with keys: trailing_pe, market_cap, eps_trailing,
    week52_high, week52_low, price_momentum_30d (30d price % change).
    Returns {} on any exception or if yfinance is not installed.
    Sleeps YFINANCE_SLEEP seconds before each call to respect soft rate limits.
    """
    if _yf is None:
        return {}
    time.sleep(YFINANCE_SLEEP)
    try:
        tk   = _yf.Ticker(ticker)
        info = tk.info
        hist = tk.history(period="35d")
        momentum: float | None = None
        if len(hist) >= 2:
            p_now  = float(hist["Close"].iloc[-1])
            p_30d  = float(hist["Close"].iloc[0])
            momentum = round((p_now - p_30d) / (abs(p_30d) + 1e-9) * 100, 4)
        return {
            "trailing_pe":        info.get("trailingPE"),
            "market_cap":         info.get("marketCap"),
            "eps_trailing":       info.get("trailingEps"),
            "week52_high":        info.get("fiftyTwoWeekHigh"),
            "week52_low":         info.get("fiftyTwoWeekLow"),
            "price_momentum_30d": momentum,
        }
    except Exception as exc:
        log.debug("[ENRICH] yfinance %s: %s", ticker, exc)
        return {}


# ── FMP fundamentals ──────────────────────────────────────────────────────────

def fetch_fmp_fundamentals(api_key: str, ticker: str) -> dict:
    """
    Fetch fundamental metrics from FMP for ticker (most recent annual row).

    Returns a dict with keys: ev_ebitda, revenue_growth_yoy, debt_to_equity.
    Returns {} on any exception, non-200 response, or missing api_key.
    """
    if not api_key:
        return {}
    try:
        resp = requests.get(
            f"{FMP_BASE}/{ticker}",
            params={"apikey": api_key, "limit": 1},
            timeout=10,
        )
        if resp.status_code != 200:
            return {}
        rows = resp.json()
        if not rows or not isinstance(rows, list):
            return {}
        row = rows[0]
        return {
            "ev_ebitda":          row.get("enterpriseValueOverEBITDA"),
            "revenue_growth_yoy": row.get("revenueGrowth"),
            "debt_to_equity":     row.get("debtToEquity"),
        }
    except Exception as exc:
        log.debug("[ENRICH] FMP %s: %s", ticker, exc)
        return {}


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_raw_articles() -> list[dict]:
    """
    Load news_articles.csv and return as a list of row dicts.
    Returns [] if the file is missing or unreadable.
    """
    if not RAW_CSV.exists():
        log.warning("[ENRICH] %s not found — run --news daily first", RAW_CSV)
        return []
    try:
        with open(RAW_CSV, newline="", encoding="utf-8") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    except Exception as exc:
        log.warning("[ENRICH] Failed to read %s: %s", RAW_CSV, exc)
        return []


# ── Enrichment orchestrator ───────────────────────────────────────────────────

def enrich_articles(articles: list[dict], api_keys: dict) -> list[dict]:
    """
    Attach yfinance and FMP signals to ticker-tagged articles.

    Pipeline:
      1. Collect unique non-empty tickers from articles.
      2. fetch_yfinance_signals() for every unique ticker.
      3. fetch_fmp_fundamentals() for S&P 500 tickers only, capped at FMP_BUDGET.
      4. Merge signals back into each article dict.

    api_keys: {"fmp": FMP_API_KEY}
    Articles with no ticker get None for all enrichment columns.
    Returns list of article dicts with ENRICHED_EXTRA_COLS added.
    """
    if not articles:
        return []

    fmp_key = (api_keys.get("fmp") or "").strip()

    # Collect unique non-empty tickers in article order
    unique_tickers: list[str] = []
    seen: set[str] = set()
    for art in articles:
        t = (art.get("ticker") or "").strip().upper()
        if t and t not in seen:
            seen.add(t)
            unique_tickers.append(t)

    log.info("[ENRICH] Fetching yfinance signals for %d unique tickers ...",
             len(unique_tickers))
    yf_cache: dict[str, dict] = {}
    for ticker in unique_tickers:
        yf_cache[ticker] = fetch_yfinance_signals(ticker)

    # FMP: S&P 500 tickers only, budget-capped
    fmp_cache: dict[str, dict] = {}
    sp500_tickers = [t for t in unique_tickers if t in _SP500_TICKERS][:FMP_BUDGET]
    if fmp_key and sp500_tickers:
        log.info("[ENRICH] Fetching FMP fundamentals for %d S&P 500 tickers ...",
                 len(sp500_tickers))
        for ticker in sp500_tickers:
            fmp_cache[ticker] = fetch_fmp_fundamentals(fmp_key, ticker)

    # Merge signals into each article
    enriched: list[dict] = []
    for art in articles:
        t = (art.get("ticker") or "").strip().upper()
        signals = {**yf_cache.get(t, {}), **fmp_cache.get(t, {})}
        merged = dict(art)
        for col in ENRICHED_EXTRA_COLS:
            merged[col] = signals.get(col)
        enriched.append(merged)

    return enriched


# ── Save enriched CSV ─────────────────────────────────────────────────────────

def save_enriched(articles: list[dict]) -> None:
    """
    Write enriched articles to ENRICHED_CSV (full overwrite on each run).
    Creates data/FinancialNews/enriched/ if it does not exist.
    """
    if not articles:
        log.warning("[ENRICH] No articles to save.")
        return
    ENRICHED_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(articles[0].keys())
    with open(ENRICHED_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(articles)
    log.info("[ENRICH] Saved %d enriched articles → %s", len(articles), ENRICHED_CSV)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Enrich news articles with yfinance and FMP financial signals"
    )
    parser.add_argument("--fmp-key", default=None,
                        help="FMP API key (overrides FMP_API_KEY env var)")
    args = parser.parse_args()

    fmp_key = args.fmp_key or os.getenv("FMP_API_KEY", "").strip()
    if not fmp_key:
        log.warning("[ENRICH] FMP_API_KEY not set — FMP enrichment skipped")

    articles = load_raw_articles()
    if not articles:
        print("[ENRICH] No articles found — run: python fred_refresh.py --news daily")
        return

    enriched = enrich_articles(articles, {"fmp": fmp_key})
    save_enriched(enriched)
    print(f"[ENRICH] {len(enriched)} articles enriched  →  {ENRICHED_CSV}")


if __name__ == "__main__":
    main()
