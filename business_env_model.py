"""
Business Environment LightGBM Forecasting Model
Series: INDPRO, TCU, PAYEMS  (monthly, predicted)
        CAPUTLB50001SQ        (quarterly → monthly via ffill, used as feature)
Outputs: 9 month-ahead forecasts with 80% prediction intervals.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from tabulate import tabulate
from macro_utils import (
    engineer_features, train_series_models, joint_recursive_forecast,
    print_validation_metrics, print_forecast_table,
    plot_forecast_dashboard, plot_validation_performance, plot_feature_importance,
    save_model_results,
)

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

PREDICT_COLS   = ["INDPRO", "TCU", "PAYEMS"]
FEATURE_COLS_EXTRA = ["CAPUTLB50001SQ"]          # quarterly series, used as feature only
ALL_SERIES     = PREDICT_COLS + FEATURE_COLS_EXTRA
VALIDATION_MONTHS = 24
FORECAST_HORIZON  = 12
CLIP_RANGES = {
    "INDPRO":  (0.0, 200.0),
    "TCU":     (50.0, 100.0),
    "PAYEMS":  (0.0, 300_000.0),
}
SERIES_INFO = [
    # (col, label, color, unit, threshold, threshold_label)
    ("INDPRO",  "Industrial Production",  "#2ecc71", "Index 2017=100", None,  None),
    ("TCU",     "Capacity Utilization",   "#e67e22", "%",              75.0,  "75% — typical floor"),
    ("PAYEMS",  "Nonfarm Payroll",         "#9b59b6", "K persons",      None,  None),
]

# ── 1. Load ───────────────────────────────────────────────────────────────────

print("=" * 65)
print("BUSINESS ENVIRONMENT — LightGBM Forecast Model")
print("=" * 65)
print("\nLoading data...")

def load_monthly(path, col):
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df["date"] = df["observation_date"].dt.to_period("M").dt.to_timestamp()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["date", col]].dropna()

indpro  = load_monthly("data/BusinessEnvironment/INDPRO.csv",               "INDPRO")
tcu     = load_monthly("data/BusinessEnvironment/TCU_capacityutilization.csv","TCU")
payems  = load_monthly("data/BusinessEnvironment/Payroll_PAYEMS.csv",        "PAYEMS")
caput_q = load_monthly("data/BusinessEnvironment/CAPUTLB50001SQ.csv",        "CAPUTLB50001SQ")

# Forward-fill quarterly to monthly
monthly_idx = pd.DataFrame(
    {"date": pd.date_range("1967-01-01", indpro["date"].max(), freq="MS")}
)
caput = (monthly_idx
         .merge(caput_q, on="date", how="left")
         .assign(CAPUTLB50001SQ=lambda d: d["CAPUTLB50001SQ"].ffill()))

# Merge all on monthly index
df = (indpro
      .merge(tcu,    on="date", how="inner")
      .merge(payems, on="date", how="inner")
      .merge(caput,  on="date", how="left")
      .sort_values("date").reset_index(drop=True))

df = df[df["date"] >= "1967-01-01"].dropna().reset_index(drop=True)
last_date = df["date"].max()

print(f"  Rows: {len(df)}  |  {df['date'].min().strftime('%Y-%m')} → {last_date.strftime('%Y-%m')}")
for col, lbl, *_ in SERIES_INFO:
    print(f"  Latest {lbl}: {df[col].iloc[-1]:,.2f}")

# ── 2. Feature engineering ────────────────────────────────────────────────────

print("\nEngineering features...")
df_feat = engineer_features(df, ALL_SERIES).dropna().reset_index(drop=True)
FEAT_COLS = [c for c in df_feat.columns if c not in ["date"] + ALL_SERIES]
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
fc = joint_recursive_forecast(df, models, FEAT_COLS, ALL_SERIES, CLIP_RANGES, FORECAST_HORIZON)

print(f"\n{FORECAST_HORIZON}-Month Forecast (from {last_date.strftime('%Y-%m')}):")
print_forecast_table(fc, SERIES_INFO)

# ── 7. Save models ────────────────────────────────────────────────────────────

for col in PREDICT_COLS:
    path = f"outputs/business_env_model_{col}.joblib"
    joblib.dump(models[col]["mid"], path)
    print(f"  Saved {path}")

# ── 8. Plots ──────────────────────────────────────────────────────────────────

print("\nGenerating plots...")
plot_forecast_dashboard(
    df, fc, SERIES_INFO, last_date,
    title="Business Environment Forecast Dashboard — 12-Month Outlook",
    save_path="outputs/business_env_dashboard.png",
)
plot_validation_performance(
    dates_val, y_val, val_preds, SERIES_INFO,
    title="Business Environment — Validation Performance (Last 24 Months)",
    save_path="outputs/business_env_validation.png",
)
plot_feature_importance(
    models, FEAT_COLS, SERIES_INFO,
    title="Business Environment — Feature Importance (Gain)",
    save_path="outputs/business_env_importance.png",
)

save_model_results(
    "Business Environment", df, fc, y_val, val_preds,
    SERIES_INFO, "outputs/results_business_env.json",
)
print("\nDone.")
