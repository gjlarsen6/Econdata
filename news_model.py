"""
news_model.py — Phase 2 ML forecasting for financial news sentiment and volume.

Aggregates news_articles.csv into monthly metrics, trains LightGBM models on
sector sentiment and article volume series, and writes 12-month forecasts.

Cold-start guard: requires ≥30 daily rows in news_articles.csv before training.
Until then, results files contain {"status": "cold_start", ...}.

Run directly:
    python3 news_model.py

Or called automatically by fred_refresh.py after news data accumulates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from macro_utils import (
    engineer_features,
    joint_recursive_forecast,
    save_model_results,
    train_series_models,
)

log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
RAW_CSV    = BASE_DIR / "data" / "FinancialNews" / "raw" / "news_articles.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

SENTIMENT_RESULTS = OUTPUT_DIR / "results_news_sentiment.json"
VOLUME_RESULTS    = OUTPUT_DIR / "results_news_volume.json"

# Cold-start threshold: minimum daily rows before attempting any modeling
MIN_SENTIMENT_DAYS = 30
# Minimum monthly rows needed for the ML pipeline (feature engineering needs lags)
MIN_MONTHLY_ROWS   = 6
HORIZON            = 12   # months to forecast

# ── Series definitions ────────────────────────────────────────────────────────
# (col, label, color_placeholder, unit)  — matches save_model_results series_info format

SENTIMENT_SERIES_INFO: list[tuple[str, str, str, str]] = [
    ("MACRO_SENT",    "Macro Sector Sentiment",    "#1f77b4", "avg score [-1,1]"),
    ("EQUITIES_SENT", "Equities Sector Sentiment", "#ff7f0e", "avg score [-1,1]"),
    ("FINTECH_SENT",  "Fintech Sector Sentiment",  "#2ca02c", "avg score [-1,1]"),
    ("VC_SENT",       "VC Sector Sentiment",        "#d62728", "avg score [-1,1]"),
]

VOLUME_SERIES_INFO: list[tuple[str, str, str, str]] = [
    ("TOTAL_VOL",    "Total Article Volume",    "#1f77b4", "articles/month"),
    ("MACRO_VOL",    "Macro Sector Volume",     "#ff7f0e", "articles/month"),
    ("EQUITIES_VOL", "Equities Sector Volume",  "#2ca02c", "articles/month"),
    ("FINTECH_VOL",  "Fintech Sector Volume",   "#d62728", "articles/month"),
]

SENTIMENT_COLS = [info[0] for info in SENTIMENT_SERIES_INFO]
VOLUME_COLS    = [info[0] for info in VOLUME_SERIES_INFO]

# Sector → column name mapping (matches news_apis.py SECTOR_KEYWORDS classification)
SECTOR_SENT_MAP = {
    "macro":     "MACRO_SENT",
    "equities":  "EQUITIES_SENT",
    "fintech":   "FINTECH_SENT",
    "vc":        "VC_SENT",
}
SECTOR_VOL_MAP = {
    "macro":     "MACRO_VOL",
    "equities":  "EQUITIES_VOL",
    "fintech":   "FINTECH_VOL",
}


# ── Data loading and aggregation ──────────────────────────────────────────────

def _load_daily_df() -> pd.DataFrame:
    """
    Load news_articles.csv. Returns empty DataFrame if file absent.
    The 'timestamp' column is parsed; the 'sentiment' column is numeric.
    """
    if not RAW_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(RAW_CSV, dtype=str)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "sentiment" in df.columns:
            df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
        return df
    except Exception as exc:
        log.warning("[NEWS_MODEL] Failed to load %s: %s", RAW_CSV, exc)
        return pd.DataFrame()


def check_readiness(daily_df: pd.DataFrame) -> bool:
    """Return True if we have ≥ MIN_SENTIMENT_DAYS distinct calendar days."""
    if daily_df.empty or "timestamp" not in daily_df.columns:
        return False
    ts_col: pd.Series = daily_df["timestamp"]  # type: ignore[assignment]
    unique_days = ts_col.dt.date.nunique()
    return unique_days >= MIN_SENTIMENT_DAYS


def aggregate_to_monthly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily news_articles rows into monthly metrics.

    Produces columns:
      date                — month-start timestamp (normalized to month-start)
      MACRO_SENT          — avg monthly sentiment for macro sector
      EQUITIES_SENT       — avg monthly sentiment for equities sector
      FINTECH_SENT        — avg monthly sentiment for fintech sector
      VC_SENT             — avg monthly sentiment for vc sector
      TOTAL_VOL           — total article count per month
      MACRO_VOL           — macro sector article count per month
      EQUITIES_VOL        — equities sector article count per month
      FINTECH_VOL         — fintech sector article count per month

    Returns empty DataFrame if input is empty or has no valid timestamps.
    """
    if daily_df.empty:
        return pd.DataFrame()

    df = daily_df.copy()
    df = df[df["timestamp"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    # Normalize to month-start
    df["_month"] = df["timestamp"].apply(  # type: ignore[union-attr]
        lambda x: x.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if pd.notna(x) else pd.NaT
    )

    all_months = sorted(df["_month"].dropna().unique())  # type: ignore[union-attr]
    if not all_months:
        return pd.DataFrame()

    # Build output row-by-row to avoid pandas type issues
    rows: list[dict] = []
    for month in all_months:
        mdf = df[df["_month"] == month]
        row: dict = {"date": month, "TOTAL_VOL": float(len(mdf))}

        for sector, col in SECTOR_SENT_MAP.items():
            subset = mdf[(mdf["sector"] == sector) & pd.notna(mdf["sentiment"])]
            if len(subset) > 0:
                row[col] = float(subset["sentiment"].mean())  # type: ignore[arg-type]
            else:
                row[col] = 0.0

        for sector, col in SECTOR_VOL_MAP.items():
            row[col] = float(len(mdf[mdf["sector"] == sector]))

        rows.append(row)

    combined = pd.DataFrame(rows)

    return combined


# ── Cold-start output ─────────────────────────────────────────────────────────

def _write_cold_start_json(days_collected: int) -> None:
    """Write cold-start JSON to both results files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "status":         "cold_start",
        "days_collected": days_collected,
        "min_required":   MIN_SENTIMENT_DAYS,
        "series":         [],
    }
    for path in (SENTIMENT_RESULTS, VOLUME_RESULTS):
        path.write_text(json.dumps(payload, indent=2))
    log.info(
        "[NEWS_MODEL] Cold start — %d/%d days collected. "
        "Written cold_start status to results files.",
        days_collected, MIN_SENTIMENT_DAYS,
    )


# ── ML training pipeline ──────────────────────────────────────────────────────

def train_and_save(
    monthly_df: pd.DataFrame,
    series_cols: list[str],
    series_info: list[tuple[str, str, str, str]],
    results_path: Path,
    group_name: str,
) -> None:
    """
    Run the full LightGBM pipeline for one metric group (sentiment or volume).

    Steps: feature engineering → train/val split → model training →
           12-month recursive forecast → save_model_results().

    If there are fewer than MIN_MONTHLY_ROWS usable rows after feature engineering,
    writes a cold_start JSON to results_path and returns.
    """
    df: pd.DataFrame = monthly_df[["date"] + series_cols].copy()  # type: ignore[assignment]

    # Drop any series columns that are entirely NaN
    valid_cols = [c for c in series_cols if bool(pd.notna(df[c]).any())]
    if not valid_cols:
        log.warning("[NEWS_MODEL] %s: all columns are NaN — skipping.", group_name)
        payload = {
            "status": "cold_start", "days_collected": 0,
            "min_required": MIN_SENTIMENT_DAYS, "series": [],
        }
        results_path.write_text(json.dumps(payload, indent=2))
        return

    # Feature engineering
    df_feat: pd.DataFrame = engineer_features(df, valid_cols).dropna()  # type: ignore[assignment]

    if len(df_feat) < MIN_MONTHLY_ROWS:
        log.warning(
            "[NEWS_MODEL] %s: only %d usable monthly rows after feature engineering "
            "(need ≥%d). Writing cold_start.",
            group_name, len(df_feat), MIN_MONTHLY_ROWS,
        )
        payload = {
            "status": "cold_start", "days_collected": len(df),
            "min_required": MIN_SENTIMENT_DAYS, "series": [],
        }
        results_path.write_text(json.dumps(payload, indent=2))
        return

    # Train/val split — last min(24, rows//4) rows for validation
    val_size = min(24, max(1, len(df_feat) // 4))
    feature_cols = [c for c in df_feat.columns if c != "date" and c not in valid_cols]

    if len(df_feat) > val_size + 1:
        X_tr   = df_feat[feature_cols].iloc[:-val_size]
        X_vl   = df_feat[feature_cols].iloc[-val_size:]
        y_tr   = {col: df_feat[col].iloc[:-val_size] for col in valid_cols}
        y_vl   = {col: df_feat[col].iloc[-val_size:]  for col in valid_cols}
    else:
        # Too few rows for a split — use all for training, last row for "val"
        X_tr   = df_feat[feature_cols]
        X_vl   = df_feat[feature_cols].iloc[-1:]
        y_tr   = {col: df_feat[col]         for col in valid_cols}
        y_vl   = {col: df_feat[col].iloc[-1:] for col in valid_cols}

    log.info("[NEWS_MODEL] %s: %d train rows, %d val rows, %d features",
             group_name, len(X_tr), len(X_vl), len(feature_cols))

    # Clip ranges: (p1 * 0.9, p99 * 1.1) for volume; [-1, 1] for sentiment
    clip_ranges: dict[str, tuple[float, float]] = {}
    for col in valid_cols:
        vals = df[col].dropna()  # type: ignore[union-attr]
        if "_SENT" in col:
            clip_ranges[col] = (-1.0, 1.0)
        else:
            lo = float(np.percentile(vals, 1))  * 0.9
            hi = float(np.percentile(vals, 99)) * 1.1
            clip_ranges[col] = (max(0.0, lo), max(hi, lo + 1.0))

    # Train models
    models = train_series_models(valid_cols, X_tr, y_tr, X_vl, y_vl)

    # Validation predictions
    val_preds: dict[str, np.ndarray] = {
        col: models[col]["mid"].predict(X_vl) for col in valid_cols if col in models
    }

    # 12-month recursive forecast
    df_base = df[["date"] + valid_cols].copy()
    fc_dict = joint_recursive_forecast(
        df_base, models, feature_cols, valid_cols, clip_ranges, horizon=HORIZON
    )

    # Filter series_info to only valid_cols
    valid_info = [info for info in series_info if info[0] in valid_cols]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_model_results(
        group_name, df_feat, fc_dict, y_vl, val_preds,
        valid_info, str(results_path),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log.info("[NEWS_MODEL] Loading news_articles.csv ...")
    daily_df = _load_daily_df()

    if daily_df.empty:
        days_collected = 0
        log.warning("[NEWS_MODEL] news_articles.csv not found or empty.")
        _write_cold_start_json(days_collected)
        print(f"[NEWS_MODEL] 0/{MIN_SENTIMENT_DAYS} days — cold start "
              "(run: python fred_refresh.py --news daily)")
        return

    ts_col: pd.Series = daily_df["timestamp"] if "timestamp" in daily_df.columns else pd.Series(dtype=object)  # type: ignore[assignment]
    days_collected = int(ts_col.dt.date.nunique()) if "timestamp" in daily_df.columns else 0

    if not check_readiness(daily_df):
        log.warning(
            "[NEWS_MODEL] Cold start — %d distinct days collected, need %d.",
            days_collected, MIN_SENTIMENT_DAYS,
        )
        _write_cold_start_json(days_collected)
        print(f"[NEWS_MODEL] {days_collected}/{MIN_SENTIMENT_DAYS} days collected — "
              "cold start. Keep running --news daily to accumulate data.")
        return

    log.info("[NEWS_MODEL] Aggregating %d articles to monthly metrics ...", len(daily_df))
    monthly_df = aggregate_to_monthly(daily_df)
    log.info("[NEWS_MODEL] Monthly rows: %d", len(monthly_df))

    if monthly_df.empty:
        _write_cold_start_json(days_collected)
        return

    # ── Sentiment group ───────────────────────────────────────────────────────
    log.info("[NEWS_MODEL] Training sentiment models (%s) ...", SENTIMENT_COLS)
    train_and_save(
        monthly_df, SENTIMENT_COLS, SENTIMENT_SERIES_INFO,
        SENTIMENT_RESULTS, "Financial News — Sector Sentiment",
    )

    # ── Volume group ──────────────────────────────────────────────────────────
    log.info("[NEWS_MODEL] Training volume models (%s) ...", VOLUME_COLS)
    train_and_save(
        monthly_df, VOLUME_COLS, VOLUME_SERIES_INFO,
        VOLUME_RESULTS, "Financial News — Article Volume",
    )

    print(
        f"[NEWS_MODEL] Done  —  {days_collected} days of data  "
        f"→  {SENTIMENT_RESULTS.name}, {VOLUME_RESULTS.name}"
    )


if __name__ == "__main__":
    main()
