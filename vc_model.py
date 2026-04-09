"""
vc_model.py — LightGBM forecasting model for Venture Capital weekly segment data.

Discovers agg_{segment}_weekly.csv files in data/VentureCapital/, resamples
weekly observations to monthly cadence, trains one LightGBM model group per
segment (ai, fintech, healthcare) using the same macro_utils pipeline as all
other model scripts, and writes results JSONs to outputs/ for the unified
summary table.

Skips gracefully when fewer than MIN_ROWS_REQUIRED monthly rows are available —
this is expected for the first ~13 months of weekly data collection.

Usage (invoked as subprocess from fred_refresh.py):
    python3 vc_model.py --segment ai
    python3 vc_model.py --segment ai fintech healthcare
    python3 vc_model.py --segment all
"""

import argparse
import os
import sys
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).parent
VC_DATA_DIR = BASE_DIR / "data" / "VentureCapital"
OUTPUT_DIR  = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
os.makedirs("outputs", exist_ok=True)

from macro_utils import (
    engineer_features,
    joint_recursive_forecast,
    plot_feature_importance,
    plot_forecast_dashboard,
    plot_validation_performance,
    print_forecast_table,
    print_validation_metrics,
    save_model_results,
    train_series_models,
)

VALIDATION_MONTHS = 24
FORECAST_HORIZON  = 12
MIN_ROWS_REQUIRED = VALIDATION_MONTHS + 30   # 54 — same threshold as sector_model.py

VC_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71",
    "#f39c12", "#9b59b6", "#1abc9c",
]

# ── Source configuration ──────────────────────────────────────────────────────

# Columns in the agg files that are time-series targets to forecast
AGG_METRIC_COLS = [
    "company_count",
    "round_count",
    "capital_raised_usd",
    "median_round_size_usd",
    "lead_investor_count",
]

VC_SOURCE_CONFIG: dict[str, dict] = {
    "ai": {
        "csv_file":     "agg_ai_weekly.csv",
        "group_name":   "VC — AI Segment",
        "label_map": {
            "company_count":         "AI Company Count",
            "round_count":           "AI Round Count (90d)",
            "capital_raised_usd":    "AI Capital Raised USD",
            "median_round_size_usd": "AI Median Round Size USD",
            "lead_investor_count":   "AI Lead Investor Count",
        },
        "results_file": "results_vc_ai.json",
        "model_prefix": "vc_model_ai",
        "plot_prefix":  "vc_ai",
    },
    "fintech": {
        "csv_file":     "agg_fintech_weekly.csv",
        "group_name":   "VC — Fintech Segment",
        "label_map": {
            "company_count":         "Fintech Company Count",
            "round_count":           "Fintech Round Count (90d)",
            "capital_raised_usd":    "Fintech Capital Raised USD",
            "median_round_size_usd": "Fintech Median Round Size USD",
            "lead_investor_count":   "Fintech Lead Investor Count",
        },
        "results_file": "results_vc_fintech.json",
        "model_prefix": "vc_model_fintech",
        "plot_prefix":  "vc_fintech",
    },
    "healthcare": {
        "csv_file":     "agg_healthcare_weekly.csv",
        "group_name":   "VC — Healthcare Segment",
        "label_map": {
            "company_count":         "Healthcare Company Count",
            "round_count":           "Healthcare Round Count (90d)",
            "capital_raised_usd":    "Healthcare Capital Raised USD",
            "median_round_size_usd": "Healthcare Median Round Size USD",
            "lead_investor_count":   "Healthcare Lead Investor Count",
        },
        "results_file": "results_vc_healthcare.json",
        "model_prefix": "vc_model_healthcare",
        "plot_prefix":  "vc_healthcare",
    },
}

# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VC LightGBM model trainer for AI, Fintech, Healthcare segments"
    )
    p.add_argument(
        "--segment", nargs="+",
        choices=["ai", "fintech", "healthcare", "all"],
        default=["all"],
        help="Which VC segment(s) to model. Use 'all' to process every segment.",
    )
    return p.parse_args()

# ── CSV loading and resampling ────────────────────────────────────────────────

def load_vc_agg_csv(path: Path, series_cols: list[str]) -> pd.DataFrame:
    """
    Load agg_{segment}_weekly.csv and resample from weekly to monthly cadence.

    The macro_utils feature engineering pipeline (lags, rolling means) is
    designed for monthly data.  Weekly agg rows are coerced to monthly by
    taking the last weekly observation in each calendar month — this gives the
    most recent weekly snapshot per month and aligns with the monthly
    observation_date convention used across all other CSVs in this project.

    Returns a DataFrame with a 'date' column (month-start timestamps) and
    one column per series_col, sorted ascending.
    """
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["observation_date"])
    for col in series_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("observation_date").reset_index(drop=True)

    # Assign each row to its calendar month (month-start timestamp)
    df["date"] = df["observation_date"].dt.to_period("M").dt.to_timestamp()

    # Keep last weekly row per month for each series column
    available_cols = [c for c in series_cols if c in df.columns]
    if not available_cols:
        return pd.DataFrame()

    monthly = (
        df.groupby("date")[available_cols]
        .last()
        .reset_index()
    )
    monthly = monthly.dropna(subset=available_cols, how="all")
    return monthly.sort_values("date").reset_index(drop=True)

# ── Series info builder ───────────────────────────────────────────────────────

def build_series_info(
    series_cols: list[str],
    label_map: dict[str, str],
) -> list[tuple]:
    """
    Build series_info tuples required by macro_utils plotting/results functions:
        (col, label, color, unit, threshold, threshold_label)
    """
    return [
        (
            col,
            label_map.get(col, col.replace("_", " ")),
            VC_COLORS[i % len(VC_COLORS)],
            "",     # unit — not applicable for VC metrics in general
            None,   # threshold
            None,   # threshold_label
        )
        for i, col in enumerate(series_cols)
    ]

# ── Auto clip ranges (same approach as sector_model.py) ──────────────────────

def build_clip_ranges(df: pd.DataFrame, series_cols: list[str]) -> dict:
    """
    Compute clip ranges from data percentiles to avoid extreme forecast values.
    Uses (1st pct × 0.9, 99th pct × 1.1) with a small margin.
    """
    ranges = {}
    for col in series_cols:
        vals = df[col].dropna()
        if vals.empty:
            continue
        lo     = float(np.percentile(vals, 1))
        hi     = float(np.percentile(vals, 99))
        margin = abs(hi - lo) * 0.1
        ranges[col] = (lo - margin, hi + margin)
    return ranges

# ── Main training pipeline ────────────────────────────────────────────────────

def run_vc_segment(cfg: dict, segment: str) -> bool:
    """
    Load, validate, engineer features, train, forecast, and save results for
    one VC segment.  Returns True on success, False if skipped or failed.
    Mirrors run_sector_group() in sector_model.py step-for-step.
    """
    group_name   = cfg["group_name"]
    model_prefix = cfg["model_prefix"]
    plot_prefix  = cfg["plot_prefix"]
    results_file = cfg["results_file"]
    label_map    = cfg["label_map"]
    csv_path     = VC_DATA_DIR / cfg["csv_file"]

    print("=" * 65)
    print(f"{group_name.upper()} — LightGBM Forecast Model")
    print("=" * 65)

    # Load and resample to monthly
    series_cols = list(label_map.keys())
    df = load_vc_agg_csv(csv_path, series_cols)

    if df.empty:
        print(f"  SKIP: {cfg['csv_file']} not found or empty.")
        return False

    # Only model columns that exist with sufficient non-NaN data
    valid_cols = [
        c for c in series_cols
        if c in df.columns and df[c].notna().sum() >= 10
    ]
    if not valid_cols:
        print(f"  SKIP: no valid series columns found in {cfg['csv_file']}.")
        return False

    if len(df) < MIN_ROWS_REQUIRED:
        print(
            f"  SKIP: only {len(df)} monthly rows (need >= {MIN_ROWS_REQUIRED}).\n"
            f"  Note: weekly data accumulates over time. Re-run once >= {MIN_ROWS_REQUIRED} "
            f"months (~{MIN_ROWS_REQUIRED // 4} quarters) have been collected."
        )
        return False

    last_date = df["date"].max()
    print(
        f"\n  Rows: {len(df)}  |  "
        f"{df['date'].min().strftime('%Y-%m')} → {last_date.strftime('%Y-%m')}"
    )
    for col in valid_cols:
        val   = df[col].dropna().iloc[-1] if df[col].notna().any() else float("nan")
        label = label_map.get(col, col)
        print(f"  Latest {label}: {val:,.2f}")

    # ── 1. Feature engineering ────────────────────────────────────────────────
    print("\nEngineering features...")
    df_feat   = engineer_features(df, valid_cols).dropna().reset_index(drop=True)
    feat_cols = [c for c in df_feat.columns if c not in ["date"] + valid_cols]
    print(f"  {len(feat_cols)} features  |  {len(df_feat)} usable rows")

    if len(df_feat) < MIN_ROWS_REQUIRED:
        print(f"  SKIP: after dropna only {len(df_feat)} rows remain.")
        return False

    # ── 2. Train / val split ──────────────────────────────────────────────────
    split_idx = len(df_feat) - VALIDATION_MONTHS
    X_train   = df_feat[feat_cols].iloc[:split_idx]
    X_val     = df_feat[feat_cols].iloc[split_idx:]
    y_train   = {col: df_feat[col].iloc[:split_idx] for col in valid_cols}
    y_val     = {col: df_feat[col].iloc[split_idx:]  for col in valid_cols}
    dates_val = df_feat["date"].iloc[split_idx:].values
    print(f"  Train: {split_idx}  |  Validation: {VALIDATION_MONTHS} months")

    # ── 3. Train models ───────────────────────────────────────────────────────
    print("\nTraining models (mid + 10th/90th quantiles)...")
    models = train_series_models(valid_cols, X_train, y_train, X_val, y_val)

    # ── 4. Validate ───────────────────────────────────────────────────────────
    val_preds   = {col: models[col]["mid"].predict(X_val) for col in valid_cols}
    series_info = build_series_info(valid_cols, label_map)

    print("\nValidation Metrics (last 24 months):")
    print_validation_metrics(y_val, val_preds, series_info)

    # ── 5. Forecast ───────────────────────────────────────────────────────────
    clip_ranges = build_clip_ranges(df, valid_cols)
    print(f"\nGenerating {FORECAST_HORIZON}-month joint recursive forecast...")
    fc = joint_recursive_forecast(
        df, models, feat_cols, valid_cols, clip_ranges, FORECAST_HORIZON
    )
    print(f"\n{FORECAST_HORIZON}-Month Forecast (from {last_date.strftime('%Y-%m')}):")
    print_forecast_table(fc, series_info)

    # ── 6. Save models ────────────────────────────────────────────────────────
    for col in valid_cols:
        path = OUTPUT_DIR / f"{model_prefix}_{col}.joblib"
        joblib.dump(models[col]["mid"], str(path))
        print(f"  Saved {path}")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    ncols = min(2, len(valid_cols))
    plot_forecast_dashboard(
        df, fc, series_info, last_date,
        title=f"{group_name} Forecast Dashboard — {FORECAST_HORIZON}-Month Outlook",
        save_path=str(OUTPUT_DIR / f"{plot_prefix}_dashboard.png"),
        ncols=ncols,
    )
    plot_validation_performance(
        dates_val, y_val, val_preds, series_info,
        title=f"{group_name} — Validation Performance (Last {VALIDATION_MONTHS} Months)",
        save_path=str(OUTPUT_DIR / f"{plot_prefix}_validation.png"),
    )
    plot_feature_importance(
        models, feat_cols, series_info,
        title=f"{group_name} — Feature Importance (Gain)",
        save_path=str(OUTPUT_DIR / f"{plot_prefix}_importance.png"),
    )

    # ── 8. Save results JSON ──────────────────────────────────────────────────
    save_model_results(
        group_name, df, fc, y_val, val_preds,
        series_info, str(OUTPUT_DIR / results_file),
    )
    print("\nDone.")
    return True

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args     = parse_args()
    segments = args.segment
    if "all" in segments:
        segments = list(VC_SOURCE_CONFIG.keys())

    any_failed = False

    for segment_key in segments:
        cfg = VC_SOURCE_CONFIG.get(segment_key)
        if cfg is None:
            print(f"Unknown segment: {segment_key}", file=sys.stderr)
            any_failed = True
            continue

        print(f"\n[VC {segment_key.upper()}] Processing segment ...")
        try:
            ok = run_vc_segment(cfg, segment_key)
            if not ok:
                any_failed = True
        except Exception as exc:
            print(f"[{segment_key}] ERROR: {exc}", file=sys.stderr)
            traceback.print_exc()
            any_failed = True

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
