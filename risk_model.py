"""
RiskLeadingInd LightGBM Model
Trains LightGBM regressors on recession probability and consumer sentiment,
generates 12-month recursive forecasts with 80% prediction intervals,
and produces a multi-panel dashboard of results.

Models trained:
  - RECPROUSM156N : Chauvet-Piger recession probability (%)
  - UMCSENT       : U. of Michigan consumer sentiment index

Outputs:
  outputs/risk_forecast_dashboard.png   — main forecast visualization
  outputs/risk_validation.png           — actual vs predicted on held-out set
  outputs/risk_feature_importance.png   — top feature importances per model
  outputs/risk_model_recpro.joblib      — saved recession probability model
  outputs/risk_model_sentiment.joblib   — saved consumer sentiment model
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from tabulate import tabulate
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

RECPRO_COL       = "RECPROUSM156N"
SENT_COL         = "UMCSENT"
FORECAST_HORIZON = 12     # months ahead
VALIDATION_MONTHS = 24    # hold out last 24 months
HISTORY_DISPLAY  = 120    # months of history to show in forecast plot (10 years)
CLIP_RANGES      = {RECPRO_COL: (0.0, 100.0), SENT_COL: (20.0, 120.0)}

COLORS = {
    RECPRO_COL: "#c0392b",   # red
    SENT_COL:   "#2980b9",   # blue
    "forecast":  "#7f8c8d",  # grey
    "ci":        0.15,       # alpha for confidence band fill
}

# ── 1. Load & merge ───────────────────────────────────────────────────────────

print("Loading RiskLeadingInd data...")
recpro = pd.read_csv("data/RiskLeadingInd/RECPROUSM156N.csv", parse_dates=["observation_date"])
sent   = pd.read_csv("data/RiskLeadingInd/UMCSENT.csv",       parse_dates=["observation_date"])

for df_ in (recpro, sent):
    df_["date"] = df_["observation_date"].dt.to_period("M").dt.to_timestamp()

df = pd.merge(
    recpro[["date", RECPRO_COL]],
    sent[["date", SENT_COL]],
    on="date", how="outer"
).sort_values("date").reset_index(drop=True)

# UMCSENT becomes consistently monthly from 1978; restrict to that window
df = df[df["date"] >= "1978-01-01"].copy()
df[SENT_COL]   = pd.to_numeric(df[SENT_COL], errors="coerce")
df[RECPRO_COL] = pd.to_numeric(df[RECPRO_COL], errors="coerce")
df[SENT_COL]   = df[SENT_COL].interpolate(method="linear", limit=3)
df = df.dropna().reset_index(drop=True)

last_date  = df["date"].max()
last_recpro = df[RECPRO_COL].iloc[-1]
last_sent   = df[SENT_COL].iloc[-1]

print(f"  Rows: {len(df)}  |  {df['date'].min().strftime('%Y-%m')} → {last_date.strftime('%Y-%m')}")
print(f"  Latest recession probability : {last_recpro:.2f}%")
print(f"  Latest consumer sentiment    : {last_sent:.1f}")

# ── 2. Feature engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col in [RECPRO_COL, SENT_COL]:
        for lag in [1, 2, 3, 6, 12]:
            d[f"{col}_lag{lag}"] = d[col].shift(lag)
        for w in [3, 6, 12]:
            rolled = d[col].shift(1).rolling(w)
            d[f"{col}_roll{w}_mean"] = rolled.mean()
            d[f"{col}_roll{w}_std"]  = rolled.std()
        d[f"{col}_mom"] = d[col].shift(1).pct_change(1) * 100
        d[f"{col}_yoy"] = d[col].shift(1).pct_change(12) * 100
    d["month"]          = d["date"].dt.month
    d["quarter"]        = d["date"].dt.quarter
    d["recpro_x_sent"]  = d[f"{RECPRO_COL}_lag1"] * d[f"{SENT_COL}_lag1"]
    d["sent_norm_recpro"] = d[f"{SENT_COL}_lag1"] / (d[f"{RECPRO_COL}_lag1"] + 1)
    return d

df_feat = engineer_features(df).dropna().reset_index(drop=True)
FEATURE_COLS = [c for c in df_feat.columns if c not in ["date", RECPRO_COL, SENT_COL]]
print(f"  Features: {len(FEATURE_COLS)}")

# ── 3. Train / val split ──────────────────────────────────────────────────────

split_idx = len(df_feat) - VALIDATION_MONTHS
X_train = df_feat[FEATURE_COLS].iloc[:split_idx]
X_val   = df_feat[FEATURE_COLS].iloc[split_idx:]
y_train = {col: df_feat[col].iloc[:split_idx] for col in [RECPRO_COL, SENT_COL]}
y_val   = {col: df_feat[col].iloc[split_idx:]  for col in [RECPRO_COL, SENT_COL]}
dates_val = df_feat["date"].iloc[split_idx:].values

print(f"\nTrain: {split_idx} months  |  Validation: {VALIDATION_MONTHS} months")

# ── 4. Train models ───────────────────────────────────────────────────────────

LGB_PARAMS = dict(
    n_estimators=2000, num_leaves=31, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, random_state=42, verbose=-1,
)

def fit(X_tr, y_tr, X_vl, y_vl, objective="regression", alpha=None):
    params = {**LGB_PARAMS, "objective": objective}
    if alpha is not None:
        params["alpha"] = alpha
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
          callbacks=[lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(-1)])
    return m

models = {}
for col in [RECPRO_COL, SENT_COL]:
    label = "Recession Probability" if col == RECPRO_COL else "Consumer Sentiment"
    print(f"\nTraining {label} models (median + 10th/90th quantiles)...")
    models[col] = {
        "mid": fit(X_train, y_train[col], X_val, y_val[col]),
        "lo":  fit(X_train, y_train[col], X_val, y_val[col], "quantile", 0.10),
        "hi":  fit(X_train, y_train[col], X_val, y_val[col], "quantile", 0.90),
    }
    bi = models[col]["mid"].best_iteration_
    print(f"  Best iteration: {bi}")

# ── 5. Validation metrics ─────────────────────────────────────────────────────

val_rows = []
val_preds = {}
for col in [RECPRO_COL, SENT_COL]:
    preds = models[col]["mid"].predict(X_val)
    val_preds[col] = preds
    mae  = mean_absolute_error(y_val[col], preds)
    rmse = np.sqrt(mean_squared_error(y_val[col], preds))
    r2   = r2_score(y_val[col], preds)
    label = "Recession Prob (%)" if col == RECPRO_COL else "Consumer Sentiment"
    val_rows.append({"Series": label,
                     "MAE": f"{mae:.3f}", "RMSE": f"{rmse:.3f}", "R²": f"{r2:.4f}"})

print("\n" + "=" * 60)
print("VALIDATION METRICS  (last 24 months held out)")
print("=" * 60)
print(tabulate(val_rows, headers="keys", tablefmt="rounded_outline"))

# ── 6. Joint recursive forecast ───────────────────────────────────────────────

print(f"\nGenerating {FORECAST_HORIZON}-month recursive forecast...")

# Working data — include the full df so all lags resolve correctly
work = df.copy()
forecast = {col: {"dates": [], "mid": [], "lo": [], "hi": []} for col in [RECPRO_COL, SENT_COL]}

for step in range(FORECAST_HORIZON):
    next_date = work["date"].iloc[-1] + pd.DateOffset(months=1)

    # Append placeholder so engineer_features can compute lag/rolling from work tail
    placeholder = pd.DataFrame([{
        "date": next_date, RECPRO_COL: np.nan, SENT_COL: np.nan
    }])
    temp = pd.concat([work, placeholder], ignore_index=True)
    temp_feat = engineer_features(temp)
    feat_row  = temp_feat[FEATURE_COLS].iloc[[-1]]

    step_preds = {}
    for col in [RECPRO_COL, SENT_COL]:
        lo_v, hi_v = CLIP_RANGES[col]
        pm = float(np.clip(models[col]["mid"].predict(feat_row)[0], lo_v, hi_v))
        pl = float(np.clip(models[col]["lo"].predict(feat_row)[0],  lo_v, hi_v))
        ph = float(np.clip(models[col]["hi"].predict(feat_row)[0],  lo_v, hi_v))
        # Ensure lo <= mid <= hi
        pl = min(pl, pm)
        ph = max(ph, pm)
        step_preds[col] = (pm, pl, ph)
        forecast[col]["dates"].append(next_date)
        forecast[col]["mid"].append(pm)
        forecast[col]["lo"].append(pl)
        forecast[col]["hi"].append(ph)

    # Commit predictions into working data for next step's lags
    work = pd.concat([work, pd.DataFrame([{
        "date": next_date,
        RECPRO_COL: step_preds[RECPRO_COL][0],
        SENT_COL:   step_preds[SENT_COL][0],
    }])], ignore_index=True)

fc = {col: pd.DataFrame(forecast[col]) for col in [RECPRO_COL, SENT_COL]}

# ── 7. Print forecast table ───────────────────────────────────────────────────

fc_rows = []
for i, row in fc[RECPRO_COL].iterrows():
    fc_rows.append({
        "Month":                  row["dates"].strftime("%Y-%m"),
        "Recession Prob Mid (%)": f"{row['mid']:.2f}",
        "Recession Prob 80% CI":  f"[{row['lo']:.2f}, {row['hi']:.2f}]",
        "Sentiment Mid":          f"{fc[SENT_COL]['mid'].iloc[i]:.1f}",
        "Sentiment 80% CI":       f"[{fc[SENT_COL]['lo'].iloc[i]:.1f}, {fc[SENT_COL]['hi'].iloc[i]:.1f}]",
    })

print(f"\n{FORECAST_HORIZON}-MONTH FORECAST  (from {last_date.strftime('%Y-%m')})")
print(tabulate(fc_rows, headers="keys", tablefmt="rounded_outline", stralign="right"))

# ── 8. Save models ────────────────────────────────────────────────────────────

joblib.dump(models[RECPRO_COL]["mid"], "outputs/risk_model_recpro.joblib")
joblib.dump(models[SENT_COL]["mid"],   "outputs/risk_model_sentiment.joblib")
print("\nSaved outputs/risk_model_recpro.joblib")
print("Saved outputs/risk_model_sentiment.joblib")

# ── 9. Plot helpers ───────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linestyle": "--",
})

def add_forecast_band(ax, fc_df, color, alpha=0.18):
    ax.fill_between(fc_df["dates"], fc_df["lo"], fc_df["hi"],
                    color=color, alpha=alpha, label="80% prediction interval")
    ax.plot(fc_df["dates"], fc_df["mid"], color=color,
            linewidth=2, linestyle="--", label="Forecast (median)")
    ax.axvline(last_date, color="grey", linewidth=1.2,
               linestyle=":", alpha=0.8, label="Last known date")

# ── 9a. Main forecast dashboard ───────────────────────────────────────────────

hist_start = last_date - pd.DateOffset(months=HISTORY_DISPLAY)
df_disp    = df[df["date"] >= hist_start]

fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
fig.suptitle("Risk & Sentiment Forecast Dashboard — 12-Month Outlook",
             fontsize=14, fontweight="bold", y=1.01)

# Panel 1: Recession Probability
ax = axes[0]
ax.plot(df_disp["date"], df_disp[RECPRO_COL],
        color=COLORS[RECPRO_COL], linewidth=2, label="Actual (Chauvet-Piger)")
add_forecast_band(ax, fc[RECPRO_COL], COLORS[RECPRO_COL])
# Risk zone shading
ax.axhspan(20, 100, alpha=0.04, color=COLORS[RECPRO_COL])
ax.axhline(20, color=COLORS[RECPRO_COL], linewidth=0.8, linestyle=":", alpha=0.5,
           label="Elevated risk threshold (20%)")
ax.set_ylabel("Recession Probability (%)", fontsize=10)
ax.set_title("Recession Probability (RECPROUSM156N)", fontsize=11, fontweight="bold")
ax.set_ylim(-2, max(df_disp[RECPRO_COL].max(), fc[RECPRO_COL]["hi"].max()) * 1.15 + 5)
ax.legend(fontsize=8, loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
# Annotate current & last forecast value
ax.annotate(f"Now: {last_recpro:.2f}%", xy=(last_date, last_recpro),
            xytext=(10, 10), textcoords="offset points",
            fontsize=8, color=COLORS[RECPRO_COL], fontweight="bold")
ax.annotate(f"→ {fc[RECPRO_COL]['mid'].iloc[-1]:.2f}%",
            xy=(fc[RECPRO_COL]["dates"].iloc[-1], fc[RECPRO_COL]["mid"].iloc[-1]),
            xytext=(-50, 10), textcoords="offset points",
            fontsize=8, color=COLORS[RECPRO_COL])

# Panel 2: Consumer Sentiment
ax = axes[1]
ax.plot(df_disp["date"], df_disp[SENT_COL],
        color=COLORS[SENT_COL], linewidth=2, label="Actual (U. of Michigan)")
add_forecast_band(ax, fc[SENT_COL], COLORS[SENT_COL])
# Weakness threshold
ax.axhspan(20, 70, alpha=0.04, color=COLORS[SENT_COL])
ax.axhline(70, color=COLORS[SENT_COL], linewidth=0.8, linestyle=":", alpha=0.5,
           label="Weakness threshold (70)")
ax.set_ylabel("Sentiment Index (1966:Q1=100)", fontsize=10)
ax.set_title("Consumer Sentiment (UMCSENT)", fontsize=11, fontweight="bold")
sent_min = min(df_disp[SENT_COL].min(), fc[SENT_COL]["lo"].min()) - 3
sent_max = max(df_disp[SENT_COL].max(), fc[SENT_COL]["hi"].max()) + 3
ax.set_ylim(sent_min, sent_max)
ax.legend(fontsize=8, loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
ax.annotate(f"Now: {last_sent:.1f}", xy=(last_date, last_sent),
            xytext=(10, -15), textcoords="offset points",
            fontsize=8, color=COLORS[SENT_COL], fontweight="bold")
ax.annotate(f"→ {fc[SENT_COL]['mid'].iloc[-1]:.1f}",
            xy=(fc[SENT_COL]["dates"].iloc[-1], fc[SENT_COL]["mid"].iloc[-1]),
            xytext=(-40, 10), textcoords="offset points",
            fontsize=8, color=COLORS[SENT_COL])

fig.tight_layout()
fig.savefig("outputs/risk_forecast_dashboard.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved outputs/risk_forecast_dashboard.png")

# ── 9b. Validation performance ────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Model Validation Performance (Last 24 Months Held Out)",
             fontsize=13, fontweight="bold")

for row_idx, (col, label, color) in enumerate([
    (RECPRO_COL, "Recession Probability (%)", COLORS[RECPRO_COL]),
    (SENT_COL,   "Consumer Sentiment",        COLORS[SENT_COL]),
]):
    actual = y_val[col].values
    pred   = val_preds[col]
    dates  = pd.to_datetime(dates_val)

    # Time series comparison
    ax = axes[row_idx][0]
    ax.plot(dates, actual, color=color, linewidth=2, label="Actual")
    ax.plot(dates, pred, color=color, linewidth=1.5, linestyle="--",
            alpha=0.8, label="Predicted")
    ax.set_title(f"{label} — Time Series", fontsize=10, fontweight="bold")
    ax.set_ylabel(label, fontsize=9)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # Scatter: actual vs predicted
    ax = axes[row_idx][1]
    ax.scatter(actual, pred, alpha=0.65, color=color, edgecolors="white",
               linewidths=0.5, s=50)
    mn, mx = min(actual.min(), pred.min()), max(actual.max(), pred.max())
    pad = (mx - mn) * 0.05
    ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad],
            "k--", linewidth=1, alpha=0.5, label="Perfect fit")
    r2 = r2_score(actual, pred)
    mae = mean_absolute_error(actual, pred)
    ax.set_title(f"{label} — Actual vs Predicted\nR²={r2:.4f}  MAE={mae:.3f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Actual", fontsize=9)
    ax.set_ylabel("Predicted", fontsize=9)
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("outputs/risk_validation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved outputs/risk_validation.png")

# ── 9c. Feature importance ────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("LightGBM Feature Importance (Gain)", fontsize=13, fontweight="bold")

for ax, (col, label, color) in zip(axes, [
    (RECPRO_COL, "Recession Probability Model", COLORS[RECPRO_COL]),
    (SENT_COL,   "Consumer Sentiment Model",    COLORS[SENT_COL]),
]):
    imp = pd.Series(
        models[col]["mid"].booster_.feature_importance(importance_type="gain"),
        index=FEATURE_COLS
    ).sort_values(ascending=True).tail(15)  # top 15

    bars = ax.barh(imp.index, imp.values, color=color, alpha=0.80, edgecolor="white")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Importance (Gain)", fontsize=9)
    # Add value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{w:,.0f}", va="center", fontsize=7)
    ax.set_xlim(0, imp.values.max() * 1.15)
    ax.tick_params(axis="y", labelsize=8)

fig.tight_layout()
fig.savefig("outputs/risk_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved outputs/risk_feature_importance.png")

# ── 9d. Current risk snapshot gauge ──────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Current Risk Snapshot  ({last_date.strftime('%B %Y')})",
             fontsize=13, fontweight="bold")

for ax, (col, label, color, threshold, threshold_label, unit) in zip(axes, [
    (RECPRO_COL, "Recession Probability", COLORS[RECPRO_COL],
     20, "Elevated (20%)", "%"),
    (SENT_COL, "Consumer Sentiment", COLORS[SENT_COL],
     70, "Weakness (70)", ""),
]):
    hist_vals = df[col].values
    bins = np.linspace(hist_vals.min(), hist_vals.max(), 40)
    ax.hist(hist_vals, bins=bins, color=color, alpha=0.35,
            edgecolor="white", label="Historical distribution")
    ax.axvline(df[col].iloc[-1], color=color, linewidth=2.5,
               label=f"Current: {df[col].iloc[-1]:.1f}{unit}")
    ax.axvline(fc[col]["mid"].iloc[-1], color=color, linewidth=2,
               linestyle="--",
               label=f"12m forecast: {fc[col]['mid'].iloc[-1]:.1f}{unit}")
    ax.axvspan(fc[col]["lo"].iloc[-1], fc[col]["hi"].iloc[-1],
               alpha=0.15, color=color, label="12m forecast 80% CI")
    ax.axvline(threshold, color="grey", linewidth=1.2,
               linestyle=":", label=threshold_label)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel(f"{label} ({unit})" if unit else label, fontsize=9)
    ax.set_ylabel("Frequency (months since 1978)", fontsize=9)
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("outputs/risk_snapshot.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved outputs/risk_snapshot.png")

print("\n" + "=" * 60)
print("COMPLETE — all outputs written to outputs/")

# ── Save results JSON for summary table ───────────────────────────────────────
import json
from datetime import datetime as _dt

_risk_series_info = [
    (RECPRO_COL, "Recession Probability", COLORS[RECPRO_COL], "%",   20.0, "Elevated (20%)"),
    (SENT_COL,   "Consumer Sentiment",    COLORS[SENT_COL],   "",    70.0, "Weakness (70)"),
]
_risk_val_preds = {
    RECPRO_COL: val_preds[RECPRO_COL],
    SENT_COL:   val_preds[SENT_COL],
}

from macro_utils import save_model_results as _smr
_smr("Risk & Leading Indicators", df, fc,
     {RECPRO_COL: y_val[RECPRO_COL], SENT_COL: y_val[SENT_COL]},
     _risk_val_preds, _risk_series_info,
     "outputs/results_risk.json")
print("=" * 60)
