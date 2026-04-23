"""
sector_model.py — Generic LightGBM forecasting model for sector API data.

Discovers CSVs in data/Sector/{BLS,BEA,WorldBank}/, groups them by source,
trains one LightGBM model group per source (using the same macro_utils pipeline
as business_env_model.py), and writes results JSONs to outputs/ for the
unified summary table.

Usage (invoked as subprocess from fred_refresh.py):
    python3 sector_model.py --source bls
    python3 sector_model.py --source bls bea worldbank
    python3 sector_model.py --source all
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR        = Path(__file__).parent
SECTOR_DATA_DIR = BASE_DIR / "data" / "Sector"
OUTPUT_DIR      = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

os.makedirs("outputs", exist_ok=True)

from macro_utils import (
    engineer_features, train_series_models, joint_recursive_forecast,
    print_validation_metrics, print_forecast_table,
    plot_forecast_dashboard, plot_validation_performance, plot_feature_importance,
    save_model_results,
)

VALIDATION_MONTHS = 24
FORECAST_HORIZON  = 12
MIN_ROWS_REQUIRED = VALIDATION_MONTHS + 30   # need enough data for train+val

SECTOR_COLORS = [
    "#3498db", "#e74c3c", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]

# ── Source configuration ──────────────────────────────────────────────────────

from sector_apis import (
    BLS_SERIES_IDS, BEA_INDUSTRY_MAP, WB_INDICATORS,
    BLS_WAGES_SERIES, BLS_HOURS_SERIES, JOLTS_SERIES, SECTOR_ETFS,
)

SOURCE_CONFIG: dict[str, dict] = {
    "bls": {
        "dir":          "BLS",
        "group_name":   "BLS Employment",
        "label_map":    BLS_SERIES_IDS,
        "results_file": "results_sector_bls.json",
        "model_prefix": "sector_bls_model",
        "plot_prefix":  "sector_bls",
    },
    "bea": {
        "dir":          "BEA",
        "group_name":   "BEA GDP by Industry",
        "label_map":    {v: v.replace("_", " ") for v in BEA_INDUSTRY_MAP.values()},
        "results_file": "results_sector_bea.json",
        "model_prefix": "sector_bea_model",
        "plot_prefix":  "sector_bea",
    },
    "worldbank": {
        "dir":          "WorldBank",
        "group_name":   "World Bank Sector GDP",
        "label_map":    WB_INDICATORS,
        "results_file": "results_sector_worldbank.json",
        "model_prefix": "sector_worldbank_model",
        "plot_prefix":  "sector_worldbank",
    },
    # Priority 3 — BLS subgroups
    "bls_wages": {
        "dir":          "BLS_Wages",
        "group_name":   "BLS Avg Hourly Earnings by Sector",
        "label_map":    BLS_WAGES_SERIES,
        "results_file": "results_sector_bls_wages.json",
        "model_prefix": "sector_bls_wages_model",
        "plot_prefix":  "sector_bls_wages",
    },
    "bls_hours": {
        "dir":          "BLS_Hours",
        "group_name":   "BLS Avg Weekly Hours by Sector",
        "label_map":    BLS_HOURS_SERIES,
        "results_file": "results_sector_bls_hours.json",
        "model_prefix": "sector_bls_hours_model",
        "plot_prefix":  "sector_bls_hours",
    },
    "jolts": {
        "dir":          "JOLTS",
        "group_name":   "JOLTS Job Openings by Sector",
        "label_map":    JOLTS_SERIES,
        "results_file": "results_sector_jolts.json",
        "model_prefix": "sector_jolts_model",
        "plot_prefix":  "sector_jolts",
    },
    # Priority 4 — Sector ETFs
    "etf": {
        "dir":          "ETF",
        "group_name":   "S&P 500 Sector ETFs",
        "label_map":    SECTOR_ETFS,
        "results_file": "results_sector_etf.json",
        "model_prefix": "sector_etf_model",
        "plot_prefix":  "sector_etf",
    },
}

# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sector LightGBM model trainer"
    )
    p.add_argument(
        "--source", nargs="+",
        choices=["bls", "bea", "worldbank", "bls_wages", "bls_hours", "jolts", "etf", "all"],
        default=["all"],
        help="Which sector source(s) to model. Use 'all' to process every source.",
    )
    return p.parse_args()

# ── CSV discovery & loading ───────────────────────────────────────────────────

def discover_sector_csvs(source_dir: Path) -> list[tuple[str, Path]]:
    """Return [(stem, path)] for all *.csv files in source_dir, sorted."""
    if not source_dir.exists():
        return []
    return sorted(
        [(p.stem, p) for p in source_dir.glob("*.csv")],
        key=lambda x: x[0],
    )

def load_sector_csv(path: Path, series_col: str) -> pd.DataFrame:
    """
    Load a sector CSV.  Normalises observation_date to month-start timestamps
    (same dt.to_period("M").dt.to_timestamp() technique used throughout the
    existing model scripts).
    """
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df["date"]    = df["observation_date"].dt.to_period("M").dt.to_timestamp()
    df[series_col] = pd.to_numeric(df[series_col], errors="coerce")
    return df[["date", series_col]].dropna().sort_values("date").reset_index(drop=True)

# ── Group DataFrame builder ───────────────────────────────────────────────────

def build_sector_group_df(csv_files: list[tuple[str, Path]],
                           min_rows: int = 60) -> tuple[pd.DataFrame, list[str]]:
    """
    Load all CSVs for one source, outer-merge on 'date', forward-fill small
    gaps, and drop rows where more than half the series are NaN.

    Returns (merged_df, series_cols_with_enough_data).
    """
    if not csv_files:
        return pd.DataFrame(), []

    merged: pd.DataFrame | None = None
    series_cols: list[str] = []

    for stem, path in csv_files:
        df = load_sector_csv(path, stem)
        if df.empty:
            continue
        series_cols.append(stem)
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="date", how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame(), []

    merged = merged.sort_values("date").reset_index(drop=True)

    # Forward-fill short gaps (up to 3 months — handles occasional missing releases)
    for col in series_cols:
        merged[col] = merged[col].ffill(limit=3)

    # Drop rows where more than half the series are missing
    threshold = len(series_cols) / 2
    merged = merged.dropna(thresh=int(len(series_cols) - threshold + 1),
                           subset=series_cols).reset_index(drop=True)

    # Exclude series with too few non-NaN values to model
    valid_cols = [c for c in series_cols if merged[c].notna().sum() >= min_rows]
    if not valid_cols:
        return pd.DataFrame(), []

    return merged, valid_cols

# ── Auto clip ranges ──────────────────────────────────────────────────────────

def build_clip_ranges(df: pd.DataFrame, series_cols: list[str]) -> dict:
    """
    Compute clip ranges from data as (1st percentile × 0.9, 99th percentile × 1.1).
    Avoids hard-coding domain-specific bounds for sector series.
    """
    ranges = {}
    for col in series_cols:
        vals    = df[col].dropna()
        if vals.empty:
            continue
        lo      = float(np.percentile(vals, 1))
        hi      = float(np.percentile(vals, 99))
        margin  = abs(hi - lo) * 0.1
        ranges[col] = (lo - margin, hi + margin)
    return ranges

# ── series_info builder ───────────────────────────────────────────────────────

def build_series_info(series_cols: list[str],
                       label_map: dict[str, str] | None = None) -> list[tuple]:
    """
    Build series_info tuples required by macro_utils plotting/results functions:
        (col, label, color, unit, threshold, threshold_label)
    """
    label_map = label_map or {}
    return [
        (
            col,
            label_map.get(col, col.replace("_", " ")),
            SECTOR_COLORS[i % len(SECTOR_COLORS)],
            "",       # unit — unknown for sector data in general
            None,     # threshold
            None,     # threshold_label
        )
        for i, col in enumerate(series_cols)
    ]

# ── Main training pipeline (mirrors business_env_model.py structure) ──────────

def run_sector_group(cfg: dict,
                      df: pd.DataFrame,
                      series_cols: list[str]) -> bool:
    """
    Train LightGBM models for one sector source group.
    Returns True on success, False if skipped or failed.
    """
    group_name   = cfg["group_name"]
    model_prefix = cfg["model_prefix"]
    plot_prefix  = cfg["plot_prefix"]
    results_file = cfg["results_file"]
    label_map    = cfg.get("label_map", {})

    print("=" * 65)
    print(f"{group_name.upper()} — LightGBM Forecast Model")
    print("=" * 65)

    # ── Guard: enough data? ────────────────────────────────────────────────────
    if len(df) < MIN_ROWS_REQUIRED:
        print(f"  SKIP: only {len(df)} rows — need ≥{MIN_ROWS_REQUIRED} for train/val split.")
        return False

    last_date = df["date"].max()
    print(f"\n  Rows: {len(df)}  |  "
          f"{df['date'].min().strftime('%Y-%m')} → {last_date.strftime('%Y-%m')}")
    for col in series_cols:
        val = df[col].dropna().iloc[-1] if df[col].notna().any() else float("nan")
        label = label_map.get(col, col)
        print(f"  Latest {label}: {val:,.3f}")

    # ── 1. Feature engineering ────────────────────────────────────────────────
    print("\nEngineering features...")
    df_feat  = engineer_features(df, series_cols).dropna().reset_index(drop=True)
    feat_cols = [c for c in df_feat.columns if c not in ["date"] + series_cols]
    print(f"  {len(feat_cols)} features  |  {len(df_feat)} usable rows")

    if len(df_feat) < MIN_ROWS_REQUIRED:
        print(f"  SKIP: after dropna only {len(df_feat)} rows remain.")
        return False

    # ── 2. Train / val split ──────────────────────────────────────────────────
    split_idx = len(df_feat) - VALIDATION_MONTHS
    X_train   = df_feat[feat_cols].iloc[:split_idx]
    X_val     = df_feat[feat_cols].iloc[split_idx:]
    y_train   = {col: df_feat[col].iloc[:split_idx] for col in series_cols}
    y_val     = {col: df_feat[col].iloc[split_idx:]  for col in series_cols}
    dates_val = df_feat["date"].iloc[split_idx:].values
    print(f"  Train: {split_idx}  |  Validation: {VALIDATION_MONTHS} months")

    # ── 3. Train models ───────────────────────────────────────────────────────
    print("\nTraining models (mid + 10th/90th quantiles)...")
    models = train_series_models(series_cols, X_train, y_train, X_val, y_val)

    # ── 4. Validate ───────────────────────────────────────────────────────────
    val_preds  = {col: models[col]["mid"].predict(X_val) for col in series_cols}
    series_info = build_series_info(series_cols, label_map)

    print("\nValidation Metrics (last 24 months):")
    print_validation_metrics(y_val, val_preds, series_info)

    # ── 5. Forecast ───────────────────────────────────────────────────────────
    clip_ranges = build_clip_ranges(df, series_cols)
    print(f"\nGenerating {FORECAST_HORIZON}-month joint recursive forecast...")
    fc = joint_recursive_forecast(df, models, feat_cols, series_cols,
                                   clip_ranges, FORECAST_HORIZON)
    print(f"\n{FORECAST_HORIZON}-Month Forecast (from {last_date.strftime('%Y-%m')}):")
    print_forecast_table(fc, series_info)

    # ── 6. Save models ────────────────────────────────────────────────────────
    for col in series_cols:
        path = OUTPUT_DIR / f"{model_prefix}_{col}.joblib"
        joblib.dump(models[col]["mid"], str(path))
        print(f"  Saved {path}")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    ncols = min(2, len(series_cols))
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
    args = parse_args()

    sources = args.source
    if "all" in sources:
        sources = list(SOURCE_CONFIG.keys())

    any_failed = False

    for source_key in sources:
        cfg = SOURCE_CONFIG.get(source_key)
        if cfg is None:
            print(f"Unknown source: {source_key}", file=sys.stderr)
            any_failed = True
            continue

        source_dir = SECTOR_DATA_DIR / cfg["dir"]
        csv_files  = discover_sector_csvs(source_dir)

        if not csv_files:
            print(f"\n[{cfg['group_name']}] No CSV files found in {source_dir} — skipping.")
            continue

        print(f"\n[{cfg['group_name']}] Found {len(csv_files)} CSV file(s): "
              f"{[s for s, _ in csv_files]}")

        df, series_cols = build_sector_group_df(csv_files)

        if df.empty or not series_cols:
            print(f"[{cfg['group_name']}] Insufficient data — skipping.")
            continue

        try:
            ok = run_sector_group(cfg, df, series_cols)
            if not ok:
                any_failed = True
        except Exception as exc:
            print(f"[{cfg['group_name']}] ERROR: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            any_failed = True

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
