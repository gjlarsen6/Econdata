"""
Consumer Demand LightGBM Forecasting Model
Series (all monthly):
  DSPIC96  — Real Disposable Personal Income
  PCE      — Personal Consumption Expenditures
  PCEPILFE — Core PCE Price Index (ex food & energy)
  RSAFS    — Nominal Retail & Food Services Sales
  RRSFS    — Real Retail & Food Services Sales
  UMCSENT  — U. of Michigan Consumer Sentiment
Dataset starts 1992-01 (when RSAFS/RRSFS begin).
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from macro_utils import (
    engineer_features, train_series_models, joint_recursive_forecast,
    print_validation_metrics, print_forecast_table,
    plot_forecast_dashboard, plot_validation_performance, plot_feature_importance,
    save_model_results,
)

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

PREDICT_COLS      = ["DSPIC96", "PCE", "PCEPILFE", "RSAFS", "RRSFS", "UMCSENT"]
VALIDATION_MONTHS = 24
FORECAST_HORIZON  = 12
CLIP_RANGES = {
    "DSPIC96":  (5_000.0,  30_000.0),
    "PCE":      (5_000.0,  40_000.0),
    "PCEPILFE": (10.0,     200.0),
    "RSAFS":    (50_000.0, 1_500_000.0),
    "RRSFS":    (50_000.0, 500_000.0),
    "UMCSENT":  (20.0,     120.0),
}
SERIES_INFO = [
    # (col, label, color, unit, threshold, threshold_label)
    ("DSPIC96",  "Real Disposable Income",    "#27ae60", "B chained $", None, None),
    ("PCE",      "Personal Consumption",       "#2980b9", "B $",         None, None),
    ("PCEPILFE", "Core PCE Price Index",       "#e74c3c", "2017=100",    None, None),
    ("RSAFS",    "Nominal Retail Sales",       "#f39c12", "M $",         None, None),
    ("RRSFS",    "Real Retail Sales",          "#16a085", "M chained $", None, None),
    ("UMCSENT",  "Consumer Sentiment",         "#8e44ad", "1966:Q1=100", 70.0, "70 — weakness threshold"),
]

# ── 1. Load ───────────────────────────────────────────────────────────────────

print("=" * 65)
print("CONSUMER DEMAND — LightGBM Forecast Model")
print("=" * 65)
print("\nLoading data...")

def load_monthly(path, col):
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df["date"] = df["observation_date"].dt.to_period("M").dt.to_timestamp()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["date", col]].dropna()

frames = {
    "DSPIC96":  load_monthly("data/ConsumerDemand/DSPIC96.csv",                          "DSPIC96"),
    "PCE":      load_monthly("data/ConsumerDemand/PCE.csv",                              "PCE"),
    "PCEPILFE": load_monthly("data/ConsumerDemand/PersConsume_noFoodEnergyPCEPILFE.csv", "PCEPILFE"),
    "RSAFS":    load_monthly("data/ConsumerDemand/RSAFS.csv",                            "RSAFS"),
    "RRSFS":    load_monthly("data/ConsumerDemand/RealRetailandFoodSalesRRSFS.csv",      "RRSFS"),
    "UMCSENT":  load_monthly("data/ConsumerDemand/UMCSENT.csv",                          "UMCSENT"),
}

df = frames["DSPIC96"]
for col, frame in list(frames.items())[1:]:
    df = df.merge(frame, on="date", how="inner")
df = df.sort_values("date").reset_index(drop=True)

# Restrict to 1992 onwards (RSAFS/RRSFS start) and fill small UMCSENT gaps
df = df[df["date"] >= "1992-01-01"].copy()
df["UMCSENT"] = df["UMCSENT"].interpolate(method="linear", limit=2)
df = df.dropna().reset_index(drop=True)
last_date = df["date"].max()

print(f"  Rows: {len(df)}  |  {df['date'].min().strftime('%Y-%m')} → {last_date.strftime('%Y-%m')}")
for col, lbl, *_ in SERIES_INFO:
    print(f"  Latest {lbl}: {df[col].iloc[-1]:,.2f}")

# ── 2. Feature engineering ────────────────────────────────────────────────────

print("\nEngineering features...")
df_feat = engineer_features(df, PREDICT_COLS).dropna().reset_index(drop=True)
FEAT_COLS = [c for c in df_feat.columns if c not in ["date"] + PREDICT_COLS]
print(f"  {len(FEAT_COLS)} features  |  {len(df_feat)} usable rows")

# ── 3. Train / val split ──────────────────────────────────────────────────────

split_idx = len(df_feat) - VALIDATION_MONTHS
X_train = df_feat[FEAT_COLS].iloc[:split_idx]
X_val   = df_feat[FEAT_COLS].iloc[split_idx:]
y_train = {col: df_feat[col].iloc[:split_idx] for col in PREDICT_COLS}
y_val   = {col: df_feat[col].iloc[split_idx:]  for col in PREDICT_COLS}
dates_val = df_feat["date"].iloc[split_idx:].values

print(f"  Train: {split_idx}  |  Validation: {VALIDATION_MONTHS} months")

# ── 4. Train ──────────────────────────────────────────────────────────────────

print("\nTraining models (mid + 10th/90th quantiles)...")
models = train_series_models(PREDICT_COLS, X_train, y_train, X_val, y_val)

# ── 5. Validate ───────────────────────────────────────────────────────────────

val_preds = {col: models[col]["mid"].predict(X_val) for col in PREDICT_COLS}
print("\nValidation Metrics (last 24 months):")
print_validation_metrics(y_val, val_preds, SERIES_INFO)

# ── 6. Forecast ───────────────────────────────────────────────────────────────

print(f"\nGenerating {FORECAST_HORIZON}-month joint recursive forecast...")
fc = joint_recursive_forecast(df, models, FEAT_COLS, PREDICT_COLS, CLIP_RANGES, FORECAST_HORIZON)

print(f"\n{FORECAST_HORIZON}-Month Forecast (from {last_date.strftime('%Y-%m')}):")
print_forecast_table(fc, SERIES_INFO)

# ── 7. Save models ────────────────────────────────────────────────────────────

for col in PREDICT_COLS:
    path = f"outputs/consumer_demand_model_{col}.joblib"
    joblib.dump(models[col]["mid"], path)
    print(f"  Saved {path}")

# ── 8. Plots ──────────────────────────────────────────────────────────────────

print("\nGenerating plots...")
plot_forecast_dashboard(
    df, fc, SERIES_INFO, last_date,
    title="Consumer Demand Forecast Dashboard — 12-Month Outlook",
    save_path="outputs/consumer_demand_dashboard.png",
    ncols=2,
)
plot_validation_performance(
    dates_val, y_val, val_preds, SERIES_INFO,
    title="Consumer Demand — Validation Performance (Last 24 Months)",
    save_path="outputs/consumer_demand_validation.png",
)
plot_feature_importance(
    models, FEAT_COLS, SERIES_INFO,
    title="Consumer Demand — Feature Importance (Gain)",
    save_path="outputs/consumer_demand_importance.png",
)

save_model_results(
    "Consumer Demand", df, fc, y_val, val_preds,
    SERIES_INFO, "outputs/results_consumer_demand.json",
)
print("\nDone.")
