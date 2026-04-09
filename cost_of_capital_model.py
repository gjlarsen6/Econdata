"""
Cost of Capital LightGBM Forecasting Model
Raw frequencies → resampled to monthly:
  DFF    (daily → monthly mean)    Federal Funds Effective Rate
  DPRIME (daily → monthly mean)    Bank Prime Loan Rate
  T10Y3M (daily → monthly mean)    Yield Curve: 10Y minus 3M  (from 1982)
  T10Y2Y (daily → monthly mean)    Yield Curve: 10Y minus 2Y  (from 2021-04 only)
  FEDFUNDS (monthly)               Fed Funds Monthly Avg — used as extra feature
  PRIME    (event → ffill monthly) Prime Rate — used as extra feature

T10Y2Y has only ~60 months of history; its model is trained on that subset only.
Main dataset starts 1982-01 (T10Y3M earliest date).
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from macro_utils import (
    engineer_features, fit_model, joint_recursive_forecast,
    print_validation_metrics, print_forecast_table,
    plot_forecast_dashboard, plot_validation_performance, plot_feature_importance,
)

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

PREDICT_COLS      = ["DFF", "DPRIME", "T10Y3M", "T10Y2Y"]
ALL_SERIES        = PREDICT_COLS + ["FEDFUNDS", "PRIME"]   # extras as features
VALIDATION_MONTHS = 24
FORECAST_HORIZON  = 12
CLIP_RANGES = {
    "DFF":    (0.0,  25.0),
    "DPRIME": (0.0,  30.0),
    "T10Y3M": (-5.0,  8.0),
    "T10Y2Y": (-3.0,  5.0),
}
SERIES_INFO = [
    # (col, label, color, unit, threshold, threshold_label)
    ("DFF",    "Fed Funds Rate",           "#c0392b", "%",  None,   None),
    ("DPRIME", "Bank Prime Loan Rate",     "#e67e22", "%",  None,   None),
    ("T10Y3M", "Yield Curve 10Y−3M",       "#2980b9", "%pts", 0.0, "0 — inversion line"),
    ("T10Y2Y", "Yield Curve 10Y−2Y",       "#8e44ad", "%pts", 0.0, "0 — inversion line"),
]

# ── 1. Load & resample to monthly ─────────────────────────────────────────────

print("=" * 65)
print("COST OF CAPITAL — LightGBM Forecast Model")
print("=" * 65)
print("\nLoading and resampling data to monthly...")

def daily_to_monthly_mean(path, col):
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = df["observation_date"].dt.to_period("M").dt.to_timestamp()
    return df.groupby("date")[col].mean().reset_index()

def event_to_monthly(path, col, end_date):
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("observation_date")
    full_idx = pd.date_range(df["observation_date"].iloc[0], end_date, freq="D")
    daily = df.set_index("observation_date").reindex(full_idx).ffill()
    daily.index.name = "observation_date"
    daily["date"] = daily.index.to_period("M").to_timestamp()
    return daily.groupby("date")[col].last().reset_index()

dff      = daily_to_monthly_mean("data/CostOfCapital/DFF.csv",    "DFF")
dprime   = daily_to_monthly_mean("data/CostOfCapital/DPRIME.csv", "DPRIME")
t10y3m   = daily_to_monthly_mean("data/CostOfCapital/T10Y3M.csv", "T10Y3M")
t10y2y   = daily_to_monthly_mean("data/CostOfCapital/T10Y2Y.csv", "T10Y2Y")
fedfunds = pd.read_csv("data/CostOfCapital/FEDFUNDS.csv", parse_dates=["observation_date"])
fedfunds["date"] = fedfunds["observation_date"].dt.to_period("M").dt.to_timestamp()
fedfunds["FEDFUNDS"] = pd.to_numeric(fedfunds["FEDFUNDS"], errors="coerce")
fedfunds = fedfunds[["date", "FEDFUNDS"]].dropna()

end_date = dff["date"].max() + pd.DateOffset(months=1)
prime = event_to_monthly("data/CostOfCapital/PRIME.csv", "PRIME", end_date)

# Merge — main frame starts at T10Y3M's earliest (1982-01)
df = (dff
      .merge(dprime,   on="date", how="inner")
      .merge(t10y3m,   on="date", how="inner")
      .merge(fedfunds, on="date", how="left")
      .merge(prime,    on="date", how="left")
      .merge(t10y2y,   on="date", how="left")   # NaN before 2021
      .sort_values("date").reset_index(drop=True))

df["FEDFUNDS"] = df["FEDFUNDS"].ffill()
df["PRIME"]    = df["PRIME"].ffill()
df = df[df["date"] >= "1982-01-01"].copy().reset_index(drop=True)
last_date = df["date"].max()
t10y2y_start = df.dropna(subset=["T10Y2Y"])["date"].min()

print(f"  Main dataset: {len(df)} months  |  {df['date'].min().strftime('%Y-%m')} → {last_date.strftime('%Y-%m')}")
print(f"  T10Y2Y available from: {t10y2y_start.strftime('%Y-%m')} ({df['T10Y2Y'].notna().sum()} months)")
for col, lbl, *_ in SERIES_INFO:
    val = df[col].dropna().iloc[-1]
    print(f"  Latest {lbl}: {val:.2f}%")

# ── 2. Feature engineering ────────────────────────────────────────────────────

print("\nEngineering features...")
df_feat = engineer_features(df, ALL_SERIES).reset_index(drop=True)
FEAT_COLS = [c for c in df_feat.columns if c not in ["date"] + ALL_SERIES]
# Drop rows only where the main (non-T10Y2Y) series have NaN features
main_cols = ["DFF", "DPRIME", "T10Y3M", "FEDFUNDS", "PRIME"]
df_feat = df_feat.dropna(subset=[c for c in df_feat.columns
                                  if any(c.startswith(m + "_") for m in main_cols)
                                  and "T10Y2Y" not in c]).reset_index(drop=True)
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
# T10Y2Y uses only the rows where it has data (2021+)

print("\nTraining models (mid + 10th/90th quantiles)...")
models = {}
for col in ["DFF", "DPRIME", "T10Y3M"]:
    print(f"    {col}: {split_idx} train rows  →  ", end="", flush=True)
    models[col] = {
        "mid": fit_model(X_train, y_train[col], X_val, y_val[col]),
        "lo":  fit_model(X_train, y_train[col], X_val, y_val[col], "quantile", 0.10),
        "hi":  fit_model(X_train, y_train[col], X_val, y_val[col], "quantile", 0.90),
    }
    print(f"best iter {models[col]['mid'].best_iteration_}")

# T10Y2Y — subset to rows where target is not NaN
t2y_mask_tr = y_train["T10Y2Y"].notna()
t2y_mask_vl = y_val["T10Y2Y"].notna()
Xtr_t2y = X_train[t2y_mask_tr]
ytr_t2y = y_train["T10Y2Y"][t2y_mask_tr]
Xvl_t2y = X_val[t2y_mask_vl]
yvl_t2y = y_val["T10Y2Y"][t2y_mask_vl]
print(f"    T10Y2Y: {len(Xtr_t2y)} train rows (2021+)  →  ", end="", flush=True)
if len(Xtr_t2y) >= 10 and len(Xvl_t2y) >= 2:
    models["T10Y2Y"] = {
        "mid": fit_model(Xtr_t2y, ytr_t2y, Xvl_t2y, yvl_t2y),
        "lo":  fit_model(Xtr_t2y, ytr_t2y, Xvl_t2y, yvl_t2y, "quantile", 0.10),
        "hi":  fit_model(Xtr_t2y, ytr_t2y, Xvl_t2y, yvl_t2y, "quantile", 0.90),
    }
    print(f"best iter {models['T10Y2Y']['mid'].best_iteration_}")
else:
    print("insufficient data — skipped")

# ── 5. Validate ───────────────────────────────────────────────────────────────

val_preds = {}
for col in models:
    if col == "T10Y2Y":
        preds = np.full(len(X_val), np.nan)
        preds[t2y_mask_vl.values] = models[col]["mid"].predict(Xvl_t2y)
        val_preds[col] = preds[t2y_mask_vl.values]
        y_val_t2y = {col: y_val[col][t2y_mask_vl]}
    else:
        val_preds[col] = models[col]["mid"].predict(X_val)

y_val_for_metrics = {**y_val}
if "T10Y2Y" in models:
    y_val_for_metrics["T10Y2Y"] = y_val["T10Y2Y"][t2y_mask_vl]

print("\nValidation Metrics (last 24 months):")
print_validation_metrics(y_val_for_metrics, val_preds, SERIES_INFO)

# ── 6. Forecast ───────────────────────────────────────────────────────────────

print(f"\nGenerating {FORECAST_HORIZON}-month joint recursive forecast...")
fc = joint_recursive_forecast(df, models, FEAT_COLS, ALL_SERIES, CLIP_RANGES, FORECAST_HORIZON)

print(f"\n{FORECAST_HORIZON}-Month Forecast (from {last_date.strftime('%Y-%m')}):")
print_forecast_table(fc, SERIES_INFO)

# ── 7. Save models ────────────────────────────────────────────────────────────

for col, m in models.items():
    path = f"outputs/cost_of_capital_model_{col}.joblib"
    joblib.dump(m["mid"], path)
    print(f"  Saved {path}")

# ── 8. Plots ──────────────────────────────────────────────────────────────────

# Only include T10Y2Y in plots if the model was trained
plot_series_info = [s for s in SERIES_INFO if s[0] in models]
val_preds_plot   = {col: models[col]["mid"].predict(X_val)
                    for col in ["DFF", "DPRIME", "T10Y3M"] if col in models}
if "T10Y2Y" in models:
    full_preds = np.full(len(X_val), np.nan)
    full_preds[t2y_mask_vl.values] = models["T10Y2Y"]["mid"].predict(Xvl_t2y)
    val_preds_plot["T10Y2Y"] = full_preds
    pass  # T10Y2Y y_val already has NaN in pre-2021 rows — no change needed

print("\nGenerating plots...")
plot_forecast_dashboard(
    df, fc, plot_series_info, last_date,
    title="Cost of Capital Forecast Dashboard — 12-Month Outlook",
    save_path="outputs/cost_of_capital_dashboard.png",
    ncols=2,
)
# Validation — only use rows where val data is non-NaN for scatter plots
plot_validation_performance(
    dates_val, y_val, val_preds_plot, plot_series_info,
    title="Cost of Capital — Validation Performance (Last 24 Months)",
    save_path="outputs/cost_of_capital_validation.png",
)
plot_feature_importance(
    models, FEAT_COLS, plot_series_info,
    title="Cost of Capital — Feature Importance (Gain)",
    save_path="outputs/cost_of_capital_importance.png",
)

from macro_utils import save_model_results
save_model_results(
    "Cost of Capital", df, fc, y_val, val_preds_plot,
    plot_series_info, "outputs/results_cost_of_capital.json",
)
print("\nDone.")
