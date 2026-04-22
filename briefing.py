"""
briefing.py — Daily financial news briefing generator (Phase 1).

No ML required — computes a rule-based market_impact_score and generates
a human-readable daily briefing JSON from ingested news articles.

Run directly:
    python3 briefing.py
    python3 briefing.py --date 2026-04-14

Or called automatically by fred_refresh.py after --news daily.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data" / "FinancialNews"
OUTPUT_DIR  = BASE_DIR / "outputs"
RAW_CSV     = DATA_DIR / "raw" / "news_articles.csv"
REGIME_PATH = BASE_DIR / "outputs" / "regime_history.json"

# CSV column schema (must match news_apis.CSV_COLUMNS)
CSV_COLUMNS = [
    "timestamp", "ingested_at", "source_api", "source_name",
    "url", "headline", "summary", "sector", "ticker", "entities",
    "sentiment", "sentiment_label", "macro_tag", "market_impact_score",
]

# S&P 500 ticker set for ticker_prominence scoring (representative subset)
_SP500_TICKERS: frozenset[str] = frozenset([
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA",
    "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "PEP", "KO", "AVGO", "COST", "BAC", "WMT", "LLY", "MCD",
    "CSCO", "ACN", "CRM", "DHR", "TMO", "NEE", "ADBE", "NKE", "DIS",
    "ORCL", "VZ", "INTC", "NFLX", "IBM", "GS", "MS", "C", "WFC",
    "USB", "AXP", "SCHW", "BLK", "GE", "CAT", "DE", "MMM", "BA",
    "RTX", "LMT", "NOC", "GD", "HON",
    "SPY", "QQQ", "IWM", "DIA",   # major ETFs often cited like tickers
])

# Import SOURCE_AUTHORITY from news_apis to avoid duplication
try:
    from news_apis import SOURCE_AUTHORITY
except ImportError:
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


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _source_authority_score(source_name: str) -> float:
    name = (source_name or "").lower()
    for key, score in SOURCE_AUTHORITY.items():
        if key in name:
            return score
    return 0.5


def _ticker_prominence(ticker: str) -> float:
    t = (ticker or "").strip().upper()
    if not t:
        return 0.2
    return 1.0 if t in _SP500_TICKERS else 0.5


def _macro_tag_weight(macro_tag: str | None) -> float:
    weights = {
        "rate_cuts":     1.0,
        "earnings":      1.0,
        "ipo":           0.8,
        "gdp":           0.7,
        "inflation":     0.7,
        "funding_round": 0.7,
        "layoffs":       0.6,
        "regulation":    0.6,
    }
    return weights.get(macro_tag or "", 0.4)


# ── Impact scoring ────────────────────────────────────────────────────────────

def score_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add/overwrite 'market_impact_score' column [0.0–1.0] using a weighted rule formula:

      score = (
          0.25 * abs(sentiment)                        # sentiment magnitude
        + 0.20 * source_authority_score(source_name)   # source prestige
        + 0.20 * ticker_prominence(ticker)             # S&P 500 = 1.0, unknown = 0.2
        + 0.20 * volume_spike_score(sector, df)        # >2σ above 30d sector avg
        + 0.15 * macro_tag_weight(macro_tag)           # rate_cuts/earnings = 1.0
      )

    volume_spike_score requires ≥7 days of history in the DataFrame; defaults
    to 0.5 if insufficient.

    NOTE: Phase 1 baseline scorer — rule-based, no ML required.
    Phase 2 adds yfinance/FMP enrichment signals on top of this formula
    but does NOT replace it.
    """
    if df.empty:
        df = df.copy()
        df["market_impact_score"] = pd.Series(dtype=float)
        return df

    df = df.copy()
    df["sentiment"] = pd.to_numeric(df.get("sentiment"), errors="coerce")

    # Volume spike: how many distinct dates exist across all articles?
    unique_dates = pd.to_datetime(df["timestamp"], errors="coerce").dt.date.nunique()
    sector_spike: dict[str, float] = {}

    if unique_dates >= 7 and "sector" in df.columns:
        df["_date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
        sector_daily = (
            df.groupby(["sector", "_date"])
            .size()
            .reset_index()
            .rename(columns={0: "cnt"})
        )
        for sector_key, grp in sector_daily.groupby("sector"):
            sector_name = str(sector_key)
            cnt_series  = grp["cnt"].astype(float)
            mu          = float(cnt_series.mean())  # type: ignore[arg-type]
            sd          = float(cnt_series.std(ddof=0))  # type: ignore[arg-type]
            today_count = float(cnt_series.iloc[-1]) if len(grp) > 0 else mu  # type: ignore[arg-type]
            if sd > 0:
                spike = min((today_count - mu) / sd, 3) / 3
            else:
                spike = 0.0
            sector_spike[sector_name] = round(max(0.0, min(1.0, 0.5 + spike * 0.5)), 4)
        df.drop(columns=["_date"], inplace=True, errors="ignore")

    def _row_score(row: pd.Series) -> float:
        sent_raw  = row["sentiment"]
        sent_mag  = min(abs(float(sent_raw)), 1.0) if pd.notna(sent_raw) else 0.0
        authority = _source_authority_score(str(row.get("source_name") or ""))
        ticker_p  = _ticker_prominence(str(row.get("ticker") or ""))
        vol_spike = sector_spike.get(str(row.get("sector") or ""), 0.5)
        tag_w     = _macro_tag_weight(row.get("macro_tag"))  # type: ignore[arg-type]
        score = (0.25 * sent_mag
               + 0.20 * authority
               + 0.20 * ticker_p
               + 0.20 * vol_spike
               + 0.15 * tag_w)
        return round(min(score, 1.0), 4)

    df["market_impact_score"] = df.apply(_row_score, axis=1)
    return df


# ── Data loading ───────────────────────────────────────────────────────────────

def load_articles(csv_path: Path, date: str | None = None) -> pd.DataFrame:
    """
    Load news_articles.csv. If date provided, filter to that date's rows only.
    Returns empty DataFrame (with correct columns) if file absent — no crash.

    Note: the 'entities' column contains JSON-serialized lists. Consumers
    must call json.loads() on each value when they need the parsed list.
    """
    if not csv_path.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)
    try:
        df = pd.read_csv(csv_path, dtype=str)
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df["sentiment"]           = pd.to_numeric(df["sentiment"], errors="coerce")
        df["market_impact_score"] = pd.to_numeric(df["market_impact_score"], errors="coerce")
        if date:
            df = df[df["timestamp"].str.startswith(date, na=False)].reset_index(drop=True)
        return df
    except Exception as exc:
        log.warning("[BRIEFING] load_articles failed: %s", exc)
        return pd.DataFrame(columns=CSV_COLUMNS)


# ── Briefing components ────────────────────────────────────────────────────────

def compute_sector_mood(df: pd.DataFrame) -> dict[str, float]:
    """
    Return {sector: avg_sentiment} for articles with non-null sentiment.
    Rounds to 3 decimal places. Omits sectors with zero valid articles.
    """
    if df.empty or "sentiment" not in df.columns:
        return {}
    valid = df[df["sentiment"].notna() & df["sector"].notna() & (df["sector"] != "")]
    if valid.empty:
        return {}
    grouped: pd.Series = valid.groupby("sector")["sentiment"].mean()  # type: ignore[assignment]
    return {str(k): round(float(grouped[k]), 3) for k in grouped.index}  # type: ignore[call-overload]


def generate_macro_signals(df: pd.DataFrame, regime_path: Path) -> list[str]:
    """
    Return list of plain-text bullet strings (max 6), filled in priority order:
      1. Current regime + FSI value from regime_history.json (always first, max 1)
      2. macro_tags with ≥3 articles today: volume bullets (up to 3)
      3. sector_mood values outside [−0.3, 0.3]: sentiment bullets (fills remaining)
    Earlier priorities are never crowded out by later ones.
    Returns [] if df is empty and no regime data is available.
    """
    bullets: list[str] = []
    MAX_BULLETS = 6

    # Priority 1: regime + FSI (always first if available)
    try:
        if regime_path.exists():
            rh     = json.loads(regime_path.read_text())
            regime = rh.get("current_regime", "")
            fsi    = rh.get("current_fsi")
            if regime and fsi is not None:
                bullets.append(f"Market regime: {regime.capitalize()} (FSI={fsi:.3f})")
    except Exception:
        pass

    if df.empty:
        return bullets

    # Priority 2: macro_tag volume (up to 3 bullets, leaving at least 1 for sentiment)
    if "macro_tag" in df.columns:
        tag_counts = df[df["macro_tag"].notna()]["macro_tag"].value_counts()
        for tag, count in tag_counts.items():
            if len(bullets) >= MAX_BULLETS - 1:
                break
            if count >= 3:
                bullets.append(f"{count} {tag.replace('_', ' ')} articles today")

    # Priority 3: sector sentiment outliers (fill remaining slots)
    mood = compute_sector_mood(df)
    for sector, avg in sorted(mood.items(), key=lambda x: abs(x[1]), reverse=True):
        if len(bullets) >= MAX_BULLETS:
            break
        if abs(avg) > 0.3:
            direction = "positive" if avg > 0 else "negative"
            bullets.append(f"{sector.capitalize()} sentiment {direction} ({avg:+.3f})")

    return bullets[:MAX_BULLETS]


def generate_alerts(df: pd.DataFrame, threshold: float = 0.75) -> list[str]:
    """
    Return headlines where market_impact_score >= threshold.
    Format: "{source_name}: {headline}" sorted by score descending.
    Max 5 alerts. Returns [] if no articles exceed threshold.
    """
    if df.empty or "market_impact_score" not in df.columns:
        return []
    high = df[df["market_impact_score"] >= threshold].copy()
    if high.empty:
        return []
    high = high.sort_values("market_impact_score", ascending=False).head(5)
    alerts = []
    for _, row in high.iterrows():
        src = (row.get("source_name") or "").strip()
        hed = (row.get("headline") or "").strip()
        if hed:
            alerts.append(f"{src}: {hed}" if src else hed)
    return alerts


def generate_daily_briefing(
    date: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    regime_path: Path | None = None,
) -> dict:
    """
    Main briefing generator. Returns a BriefingResponse-compatible dict.
    date defaults to today (YYYY-MM-DD).
    output_dir.mkdir(parents=True, exist_ok=True) is called before writing.

    Pipeline: load_articles → score_impact → compute_sector_mood
              → generate_macro_signals → generate_alerts
    """
    date        = date or datetime.utcnow().strftime("%Y-%m-%d")
    data_dir    = data_dir or DATA_DIR
    output_dir  = output_dir or OUTPUT_DIR
    regime_path = regime_path or REGIME_PATH

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "raw" / "news_articles.csv"
    df = load_articles(csv_path, date=date)
    df = score_impact(df)

    sector_mood   = compute_sector_mood(df)
    macro_signals = generate_macro_signals(df, regime_path)
    alerts        = generate_alerts(df)

    # Top 10 stories by impact score
    if not df.empty and "market_impact_score" in df.columns:
        top_df = (
            df[df["market_impact_score"].notna()]
            .sort_values("market_impact_score", ascending=False)
            .head(10)
        )
    else:
        top_df = df.head(10)

    top_stories: list[dict] = []
    for _, row in top_df.iterrows():
        sent = row.get("sentiment")
        mis  = row.get("market_impact_score")
        top_stories.append({
            "timestamp":           (row.get("timestamp") or ""),
            "source_name":         (row.get("source_name") or ""),
            "sector":              (row.get("sector") or ""),
            "ticker":              (row.get("ticker") or None),
            "headline":            (row.get("headline") or ""),
            "sentiment":           float(sent) if pd.notna(sent) else None,
            "sentiment_label":     (row.get("sentiment_label") or None),
            "macro_tag":           (row.get("macro_tag") or None),
            "market_impact_score": float(mis) if pd.notna(mis) else None,
        })

    return {
        "date":          date,
        "generated_at":  datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "article_count": len(df),
        "top_stories":   top_stories,
        "sector_mood":   sector_mood,
        "macro_signals": macro_signals,
        "alerts":        alerts,
    }


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    """
    CLI entry point.
    Calls output_dir.mkdir(parents=True, exist_ok=True) before writing.
    Writes outputs/daily_briefing_{YYYY-MM-DD}.json.
    Prints: "[BRIEFING] 2026-04-15 — 47 articles, 3 alerts"
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Generate daily financial news briefing")
    parser.add_argument("--date", default=None,
                        help="Date to generate briefing for (YYYY-MM-DD, default: today)")
    args = parser.parse_args([] if __name__ != "__main__" else None)

    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    briefing  = generate_daily_briefing(date=args.date, output_dir=output_dir)
    date_str  = briefing["date"]
    out_path  = output_dir / f"daily_briefing_{date_str}.json"
    out_path.write_text(json.dumps(briefing, indent=2))

    print(
        f"[BRIEFING] {date_str} — {briefing['article_count']} articles, "
        f"{len(briefing['alerts'])} alerts  →  {out_path}"
    )


if __name__ == "__main__":
    main()
