"""
industrial_model.py — LightGBM forecasting model for FRED industrial/ISM/credit series.

Handles four model groups, each using discovery-based CSV loading from its own
data directory (same pattern as sector_model.py):

  IndustrialProduction — IP sector breakdowns (IPMAN, IPUTIL, IPMINE, …)
  ISMIndicators        — ISM PMI composite + sub-indices (NAPM, NAPMNEWO, …)
  CapacityUtilSector   — Sector-level capacity utilization (MCUMFN, …)
  CreditIndicators     — Commercial loans + PPI commodity prices (BUSLOANS, …)

Usage (invoked as subprocess from fred_refresh.py):
    python3 industrial_model.py --source industrial_production
    python3 industrial_model.py --source ism_pmi capacity_util_sector
    python3 industrial_model.py --source all
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

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
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
MIN_ROWS_REQUIRED = VALIDATION_MONTHS + 30

GROUP_COLORS = [
    "#3498db", "#e74c3c", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]

# ── Source configuration ──────────────────────────────────────────────────────

SOURCE_CONFIG: dict[str, dict] = {
    "industrial_production": {
        "dir":          DATA_DIR / "IndustrialProduction",
        "group_name":   "Industrial Production by Sector",
        "label_map": {
            "IPMAN":    "IP: Manufacturing",
            "IPUTIL":   "IP: Utilities",
            "IPMINE":   "IP: Mining",
            "IPCONGD":  "IP: Consumer Goods",
            "IPBUSEQ":  "IP: Business Equipment",
            "IPMAT":    "IP: Materials",
            "IPDCONGD": "IP: Durable Consumer Goods",
            "IPNCONGD": "IP: Nondurable Consumer Goods",
        },
        "results_file": "results_industrial_production.json",
        "model_prefix": "industrial_production_model",
        "plot_prefix":  "industrial_production",
    },
    "ism_pmi": {
        "dir":          DATA_DIR / "ISMIndicators",
        "group_name":   "ISM PMI Leading Indicators",
        "label_map": {
            "NAPM":     "ISM Manufacturing PMI",
            "NMFCI":    "ISM Services PMI",
            "NAPMPROD": "ISM Mfg: Production",
            "NAPMNEWO": "ISM Mfg: New Orders",
            "NAPMEMPL": "ISM Mfg: Employment",
            "NAPMVNDR": "ISM: Vendor Deliveries",
        },
        "results_file": "results_industrial_ism_pmi.json",
        "model_prefix": "industrial_ism_model",
        "plot_prefix":  "industrial_ism",
    },
    "capacity_util_sector": {
        "dir":          DATA_DIR / "CapacityUtilSector",
        "group_name":   "Capacity Utilization by Sector",
        "label_map": {
            "MCUMFN":      "Capacity Util: Manufacturing",
            "CAPUTLG211S": "Capacity Util: Mining",
            "CAPUTLB58SQ": "Capacity Util: Durable Goods",
        },
        "results_file": "results_industrial_capacity_util.json",
        "model_prefix": "industrial_capacity_util_model",
        "plot_prefix":  "industrial_capacity_util",
    },
    "credit_indicators": {
        "dir":          DATA_DIR / "CreditIndicators",
        "group_name":   "Credit & PPI Sector Indicators",
        "label_map": {
            "BUSLOANS": "C&I Loans",
            "REALLN":   "Real Estate Loans",
            "CONSUMER": "Consumer Loans",
            "WPU05":    "PPI: Fuels & Related",
            "WPU10":    "PPI: Farm Products",
        },
        "results_file": "results_industrial_credit.json",
        "model_prefix": "industrial_credit_model",
        "plot_prefix":  "industrial_credit",
    },
}

# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Industrial/ISM/Credit LightGBM model trainer")
    p.add_argument(
        "--source", nargs="+",
        choices=list(SOURCE_CONFIG.keys()) + ["all"],
        default=["all"],
        help="Which group(s) to model. Use 'all' to process every group.",
    )
    return p.parse_args()

# ── CSV helpers (mirror sector_model.py) ─────────────────────────────────────

def discover_csvs(source_dir: Path) -> list[tuple[str, Path]]:
    if not source_dir.exists():
        return []
    return sorted(
        [(p.stem, p) for p in source_dir.glob("*.csv")],
        key=lambda x: x[0],
    )


def load_csv(path: Path, series_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df["date"] = df["observation_date"].dt.to_period("M").dt.to_timestamp()
    df[series_col] = pd.to_numeric(df[series_col], errors="coerce")
    return df[["date", series_col]].dropna().sort_values("date").reset_index(drop=True)


def build_group_df(csv_files: list[tuple[str, Path]],
                   min_rows: int = 60) -> tuple[pd.DataFrame, list[str]]:
    if not csv_files:
        return pd.DataFrame(), []

    merged: pd.DataFrame | None = None
    series_cols: list[str] = []

    for stem, path in csv_files:
        df = load_csv(path, stem)
        if df.empty:
            continue
        series_cols.append(stem)
        merged = df if merged is None else merged.merge(df, on="date", how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame(), []

    merged = merged.sort_values("date").reset_index(drop=True)

    for col in series_cols:
        merged[col] = merged[col].ffill(limit=3)

    threshold = len(series_cols) / 2
    merged = merged.dropna(
        thresh=int(len(series_cols) - threshold + 1), subset=series_cols
    ).reset_index(drop=True)

    valid_cols = [c for c in series_cols if merged[c].notna().sum() >= min_rows]
    return (merged, valid_cols) if valid_cols else (pd.DataFrame(), [])


def build_clip_ranges(df: pd.DataFrame, series_cols: list[str]) -> dict:
    ranges = {}
    for col in series_cols:
        vals = df[col].dropna()
        if vals.empty:
            continue
        lo, hi = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
        margin = abs(hi - lo) * 0.1
        ranges[col] = (lo - margin, hi + margin)
    return ranges


def build_series_info(series_cols: list[str],
                       label_map: dict[str, str]) -> list[tuple]:
    return [
        (col, label_map.get(col, col.replace("_", " ")),
         GROUP_COLORS[i % len(GROUP_COLORS)], "", None, None)
        for i, col in enumerate(series_cols)
    ]

# ── Main training pipeline ────────────────────────────────────────────────────

def run_group(cfg: dict, df: pd.DataFrame, series_cols: list[str]) -> bool:
    group_name   = cfg["group_name"]
    model_prefix = cfg["model_prefix"]
    plot_prefix  = cfg["plot_prefix"]
    results_file = cfg["results_file"]
    label_map    = cfg.get("label_map", {})

    print("=" * 65)
    print(f"{group_name.upper()} — LightGBM Forecast Model")
    print("=" * 65)

    if len(df) < MIN_ROWS_REQUIRED:
        print(f"  SKIP: only {len(df)} rows — need ≥{MIN_ROWS_REQUIRED}.")
        return False

    last_date = df["date"].max()
    print(f"\n  Rows: {len(df)}  |  "
          f"{df['date'].min().strftime('%Y-%m')} → {last_date.strftime('%Y-%m')}")
    for col in series_cols:
        val   = df[col].dropna().iloc[-1] if df[col].notna().any() else float("nan")
        label = label_map.get(col, col)
        print(f"  Latest {label}: {val:,.3f}")

    print("\nEngineering features...")
    df_feat   = engineer_features(df, series_cols).dropna().reset_index(drop=True)
    feat_cols = [c for c in df_feat.columns if c not in ["date"] + series_cols]
    print(f"  {len(feat_cols)} features  |  {len(df_feat)} usable rows")

    if len(df_feat) < MIN_ROWS_REQUIRED:
        print(f"  SKIP: after dropna only {len(df_feat)} rows remain.")
        return False

    split_idx = len(df_feat) - VALIDATION_MONTHS
    X_train   = df_feat[feat_cols].iloc[:split_idx]
    X_val     = df_feat[feat_cols].iloc[split_idx:]
    y_train   = {col: df_feat[col].iloc[:split_idx] for col in series_cols}
    y_val     = {col: df_feat[col].iloc[split_idx:]  for col in series_cols}
    dates_val = df_feat["date"].iloc[split_idx:].values
    print(f"  Train: {split_idx}  |  Validation: {VALIDATION_MONTHS} months")

    print("\nTraining models (mid + 10th/90th quantiles)...")
    models = train_series_models(series_cols, X_train, y_train, X_val, y_val)

    val_preds   = {col: models[col]["mid"].predict(X_val) for col in series_cols}
    series_info = build_series_info(series_cols, label_map)

    print("\nValidation Metrics (last 24 months):")
    print_validation_metrics(y_val, val_preds, series_info)

    clip_ranges = build_clip_ranges(df, series_cols)
    print(f"\nGenerating {FORECAST_HORIZON}-month joint recursive forecast...")
    fc = joint_recursive_forecast(df, models, feat_cols, series_cols,
                                   clip_ranges, FORECAST_HORIZON)
    print(f"\n{FORECAST_HORIZON}-Month Forecast (from {last_date.strftime('%Y-%m')}):")
    print_forecast_table(fc, series_info)

    for col in series_cols:
        path = OUTPUT_DIR / f"{model_prefix}_{col}.joblib"
        joblib.dump(models[col]["mid"], str(path))
        print(f"  Saved {path}")

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

        source_dir = cfg["dir"]
        csv_files  = discover_csvs(source_dir)

        if not csv_files:
            print(f"\n[{cfg['group_name']}] No CSV files found in {source_dir} — skipping.")
            continue

        print(f"\n[{cfg['group_name']}] Found {len(csv_files)} CSV file(s): "
              f"{[s for s, _ in csv_files]}")

        df, series_cols = build_group_df(csv_files)

        if df.empty or not series_cols:
            print(f"[{cfg['group_name']}] Insufficient data — skipping.")
            continue

        try:
            ok = run_group(cfg, df, series_cols)
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
