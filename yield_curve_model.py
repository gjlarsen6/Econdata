"""
yield_curve_model.py — LightGBM forecasts for the full U.S. Treasury yield curve.

Trains on 8 constant-maturity Treasury yield series:
  DGS1MO  1-Month
  DGS3MO  3-Month
  DGS6MO  6-Month
  DGS1    1-Year
  DGS2    2-Year
  DGS5    5-Year
  DGS10   10-Year
  DGS30   30-Year

All series are daily in FRED; resampled to monthly mean before modeling.
Joint dataset starts from DGS1MO availability (~2001-07).
Cross-features: slope (DGS10−DGS2), curvature (2×DGS5−DGS1−DGS10), 10Y−3M spread.

Outputs:
  outputs/results_yield_curve.json     — GroupResponse format, all 8 tenors
  outputs/yield_curve_dashboard.png
  outputs/yield_curve_importance.png
  outputs/yield_curve_validation.png
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tabulate import tabulate
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

from macro_utils import save_model_results as _save_model_results

# ── Constants ──────────────────────────────────────────────────────────────────

FORECAST_HORIZON  = 12
VALIDATION_MONTHS = 24
HISTORY_DISPLAY   = 120

TENOR_COLS = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]

# (label, color, unit, threshold, threshold_label)
TENOR_META = {
    "DGS1MO":  ("1-Month Treasury",   "#1a1a2e", "%", None, None),
    "DGS3MO":  ("3-Month Treasury",   "#16213e", "%", None, None),
    "DGS6MO":  ("6-Month Treasury",   "#0f3460", "%", None, None),
    "DGS1":    ("1-Year Treasury",    "#533483", "%", None, None),
    "DGS2":    ("2-Year Treasury",    "#e94560", "%", None, None),
    "DGS5":    ("5-Year Treasury",    "#e85d04", "%", None, None),
    "DGS10":   ("10-Year Treasury",   "#2196f3", "%", None, None),
    "DGS30":   ("30-Year Treasury",   "#4caf50", "%", None, None),
}

CLIP_RANGES = {col: (-0.5, 20.0) for col in TENOR_COLS}

LGB_PARAMS = dict(
    n_estimators=2000, num_leaves=31, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, random_state=42, verbose=-1,
)

plt.rcParams.update({
    "font.family": "sans-serif", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linestyle": "--",
})

# ── 1. Load and resample ───────────────────────────────────────────────────────

print("=" * 60)
print("Treasury Yield Curve Model")
print("=" * 60)
print("\nLoading data...")


def _load_daily_csv(path: str, series_id: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df["date"] = df["observation_date"].dt.to_period("M").dt.to_timestamp()
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df.groupby("date")[series_id].mean().reset_index()


frames: list[pd.DataFrame] = []
available_cols: list[str] = []

for sid in TENOR_COLS:
    path = f"data/YieldCurve/{sid}.csv"
    try:
        mdf = _load_daily_csv(path, sid)
        if mdf[sid].notna().sum() < 12:
            print(f"  SKIP  {sid}: insufficient data")
            continue
        frames.append(mdf)
        available_cols.append(sid)
        last_val = mdf[sid].dropna().iloc[-1]
        print(f"  OK    {sid:<8}  {mdf['date'].min().strftime('%Y-%m')} → "
              f"{mdf['date'].max().strftime('%Y-%m')}  latest={last_val:.3f}%")
    except FileNotFoundError:
        print(f"  SKIP  {sid}: {path} not found (run fred_refresh.py first)")

if len(available_cols) < 2:
    print("\nERROR: fewer than 2 tenor series available — run fred_refresh.py first.")
    sys.exit(1)

# Outer merge
df = frames[0]
for f in frames[1:]:
    df = pd.merge(df, f, on="date", how="outer")
df = df.sort_values("date").reset_index(drop=True)

# Interpolate short gaps (≤2 months — handles DGS30 discontinuity gaps)
for col in available_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").interpolate(
        method="linear", limit=2
    )

# Restrict to window with reasonably complete curve (DGS1MO starts ~2001-07)
min_col_start = max(
    df.loc[df[c].notna(), "date"].min()
    for c in available_cols
)
df = df[df["date"] >= min_col_start].copy().reset_index(drop=True)

last_date = df["date"].max()
print(f"\n  Combined: {len(df)} monthly rows  "
      f"{df['date'].min().strftime('%Y-%m')} → {last_date.strftime('%Y-%m')}")

# ── 2. Feature engineering ─────────────────────────────────────────────────────

print("\nEngineering features...")


def _engineer(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for col in cols:
        for lag in [1, 2, 3, 6, 12]:
            d[f"{col}_lag{lag}"] = d[col].shift(lag)
        for w in [3, 6, 12]:
            rolled = d[col].shift(1).rolling(w, min_periods=max(1, w // 2))
            d[f"{col}_roll{w}_mean"] = rolled.mean()
            d[f"{col}_roll{w}_std"]  = rolled.std()
        d[f"{col}_mom"] = d[col].shift(1).pct_change(1) * 100
    d["month"]   = d["date"].dt.month
    d["quarter"] = d["date"].dt.quarter
    # Curve shape cross-features (computed from lagged values to avoid lookahead)
    if "DGS10" in cols and "DGS2" in cols:
        d["slope_10y_2y"] = d["DGS10_lag1"] - d["DGS2_lag1"]
    if "DGS5" in cols and "DGS1" in cols and "DGS10" in cols:
        d["curvature"]    = 2 * d["DGS5_lag1"] - d["DGS1_lag1"] - d["DGS10_lag1"]
    if "DGS10" in cols and "DGS3MO" in cols:
        d["spread_10y_3m"] = d["DGS10_lag1"] - d["DGS3MO_lag1"]
    if "DGS2" in cols and "DGS1MO" in cols:
        d["spread_2y_1m"]  = d["DGS2_lag1"]  - d["DGS1MO_lag1"]
    return d


df_feat      = _engineer(df, available_cols)
lag1_cols    = [f"{c}_lag1" for c in available_cols]
df_feat      = df_feat.dropna(subset=[c for c in lag1_cols if c in df_feat.columns])
df_feat      = df_feat.reset_index(drop=True)
FEATURE_COLS = [c for c in df_feat.columns if c not in ["date"] + available_cols]
print(f"  Feature columns: {len(FEATURE_COLS)}")

# ── 3. Train / val split ───────────────────────────────────────────────────────

split_idx = len(df_feat) - VALIDATION_MONTHS
X_train   = df_feat[FEATURE_COLS].iloc[:split_idx]
X_val     = df_feat[FEATURE_COLS].iloc[split_idx:]
y_train   = {col: df_feat[col].iloc[:split_idx] for col in available_cols}
y_val     = {col: df_feat[col].iloc[split_idx:] for col in available_cols}
dates_val = df_feat["date"].iloc[split_idx:].values

print(f"  Train: {split_idx} months  |  Validation: {VALIDATION_MONTHS} months")

# ── 4. Train LightGBM models ───────────────────────────────────────────────────

print("\nTraining yield curve models...")


def _fit(Xtr, ytr, Xvl, yvl, objective="regression", alpha=None):
    params = {**LGB_PARAMS, "objective": objective}
    if alpha is not None:
        params["alpha"] = alpha
    m = lgb.LGBMRegressor(**params)
    m.fit(Xtr, ytr, eval_set=[(Xvl, yvl)],
          callbacks=[lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(-1)])
    return m


models: dict = {}
for col in available_cols:
    tr_mask = y_train[col].notna()
    vl_mask = y_val[col].notna()
    if tr_mask.sum() < 12:
        print(f"  SKIP {col}: only {tr_mask.sum()} training rows")
        continue
    label = TENOR_META.get(col, (col,))[0]
    print(f"  {label:<22} ({col})  train={tr_mask.sum()} ...", end="", flush=True)
    models[col] = {
        "mid": _fit(X_train[tr_mask], y_train[col][tr_mask],
                    X_val[vl_mask],   y_val[col][vl_mask]),
        "lo":  _fit(X_train[tr_mask], y_train[col][tr_mask],
                    X_val[vl_mask],   y_val[col][vl_mask], "quantile", 0.10),
        "hi":  _fit(X_train[tr_mask], y_train[col][tr_mask],
                    X_val[vl_mask],   y_val[col][vl_mask], "quantile", 0.90),
    }
    print(f"  best_iter={models[col]['mid'].best_iteration_}")

# ── 5. Validation metrics ──────────────────────────────────────────────────────

val_preds: dict = {}
val_rows:  list = []
for col in available_cols:
    if col not in models:
        continue
    preds = models[col]["mid"].predict(X_val)
    val_preds[col] = preds
    valid  = y_val[col].notna()
    mae    = mean_absolute_error(y_val[col][valid], preds[valid])
    rmse   = np.sqrt(mean_squared_error(y_val[col][valid], preds[valid]))
    r2     = r2_score(y_val[col][valid], preds[valid])
    val_rows.append({
        "Tenor": TENOR_META.get(col, (col,))[0],
        "MAE":  f"{mae:.4f}", "RMSE": f"{rmse:.4f}", "R²": f"{r2:.4f}",
    })

print("\n" + "=" * 60)
print("VALIDATION METRICS  (last 24 months held out)")
print("=" * 60)
print(tabulate(val_rows, headers="keys", tablefmt="rounded_outline"))

# ── 6. Recursive forecast ──────────────────────────────────────────────────────

print(f"\nGenerating {FORECAST_HORIZON}-month recursive forecasts...")

work  = df[["date"] + available_cols].copy()
fcast = {col: {"dates": [], "mid": [], "lo": [], "hi": []}
         for col in available_cols if col in models}

for _ in range(FORECAST_HORIZON):
    next_date   = work["date"].iloc[-1] + pd.DateOffset(months=1)
    placeholder = {col: np.nan for col in available_cols}
    placeholder["date"] = next_date
    temp      = pd.concat([work, pd.DataFrame([placeholder])], ignore_index=True)
    temp_feat = _engineer(temp, available_cols)
    feat_row  = temp_feat[FEATURE_COLS].iloc[[-1]]

    step_preds: dict = {}
    for col in available_cols:
        if col not in models:
            continue
        lo_v, hi_v = CLIP_RANGES.get(col, (-np.inf, np.inf))
        pm = float(np.clip(models[col]["mid"].predict(feat_row)[0], lo_v, hi_v))
        pl = float(np.clip(models[col]["lo"].predict(feat_row)[0],  lo_v, hi_v))
        ph = float(np.clip(models[col]["hi"].predict(feat_row)[0],  lo_v, hi_v))
        pl, ph = min(pl, pm), max(ph, pm)
        step_preds[col] = (pm, pl, ph)
        fcast[col]["dates"].append(next_date)
        fcast[col]["mid"].append(pm)
        fcast[col]["lo"].append(pl)
        fcast[col]["hi"].append(ph)

    new_row = {"date": next_date}
    for col in available_cols:
        new_row[col] = step_preds[col][0] if col in step_preds else work[col].iloc[-1]
    work = pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)

fc = {col: pd.DataFrame(fcast[col]) for col in fcast}

# ── 7. Print current curve snapshot ───────────────────────────────────────────

latest = df.iloc[-1]
fc_12m_mid = {col: fc[col]["mid"].iloc[-1] for col in fc}

rows_snap = []
for col in available_cols:
    if col not in fc:
        continue
    rows_snap.append({
        "Tenor":       TENOR_META.get(col, (col,))[0],
        "Current (%)": f"{latest[col]:.3f}" if pd.notna(latest.get(col)) else "—",
        "+12M Mid (%)": f"{fc_12m_mid[col]:.3f}",
    })
print(f"\nYield Curve Snapshot  ({last_date.strftime('%Y-%m')})")
print(tabulate(rows_snap, headers="keys", tablefmt="rounded_outline", stralign="right"))

# ── 8. Plots ───────────────────────────────────────────────────────────────────

hist_start = last_date - pd.DateOffset(months=HISTORY_DISPLAY)
df_disp    = df[df["date"] >= hist_start]

# Panel dashboard — 4 tenors per row
n_plot   = len([c for c in available_cols if c in fc])
nrows    = -(-n_plot // 2)
fig, axes = plt.subplots(nrows, 2, figsize=(14, 4 * nrows), squeeze=False)
fig.suptitle("U.S. Treasury Yield Curve — 12-Month Forecast",
             fontsize=13, fontweight="bold", y=1.01)

for idx, col in enumerate([c for c in available_cols if c in fc]):
    meta  = TENOR_META.get(col, (col, "#888888", "%", None, None))
    label, color = meta[0], meta[1]
    ax    = axes[idx // 2][idx % 2]
    ax.plot(df_disp["date"], df_disp[col], color=color, linewidth=2, label="Actual")
    ax.fill_between(fc[col]["dates"], fc[col]["lo"], fc[col]["hi"],
                    color=color, alpha=0.15, label="80% PI")
    ax.plot(fc[col]["dates"], fc[col]["mid"], color=color,
            linewidth=2, linestyle="--", label="Forecast")
    ax.axvline(last_date, color="grey", linewidth=1.2, linestyle=":", alpha=0.7)
    curr = df[col].dropna().iloc[-1]
    ax.annotate(f"Now: {curr:.3f}%", xy=(last_date, curr),
                xytext=(8, 6), textcoords="offset points",
                fontsize=8, color=color, fontweight="bold")
    ax.set_title(f"{col} — {label}", fontsize=10, fontweight="bold")
    ax.set_ylabel("Yield (%)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

for j in range(n_plot, nrows * 2):
    axes[j // 2][j % 2].set_visible(False)
fig.tight_layout()
fig.savefig("outputs/yield_curve_dashboard.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n  Saved outputs/yield_curve_dashboard.png")

# Feature importance
n_imp  = len([c for c in available_cols if c in models])
nrows_i = -(-n_imp // 2)
fig, axes = plt.subplots(nrows_i, 2, figsize=(14, 4 * nrows_i), squeeze=False)
fig.suptitle("Yield Curve — Feature Importance (Gain)", fontsize=13, fontweight="bold")
for idx, col in enumerate([c for c in available_cols if c in models]):
    meta = TENOR_META.get(col, (col, "#888888"))
    label, color = meta[0], meta[1]
    ax   = axes[idx // 2][idx % 2]
    imp  = pd.Series(
        models[col]["mid"].booster_.feature_importance(importance_type="gain"),
        index=FEATURE_COLS,
    ).sort_values(ascending=True).tail(12)
    bars = ax.barh(imp.index, imp.values, color=color, alpha=0.80, edgecolor="white")
    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{w:,.0f}", va="center", fontsize=7)
    ax.set_xlim(0, imp.values.max() * 1.18)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlabel("Importance (Gain)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
for j in range(n_imp, nrows_i * 2):
    axes[j // 2][j % 2].set_visible(False)
fig.tight_layout()
fig.savefig("outputs/yield_curve_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved outputs/yield_curve_importance.png")

# ── 9. Save results JSON ───────────────────────────────────────────────────────

series_info_tuples = [
    (c,) + TENOR_META.get(c, (c, "#888888", "%", None, None))
    for c in available_cols if c in fc
]

_save_model_results(
    "Treasury Yield Curve", df, fc,
    {c: y_val[c] for c in available_cols if c in fc and c in y_val},
    {c: val_preds[c] for c in available_cols if c in fc and c in val_preds},
    series_info_tuples,
    "outputs/results_yield_curve.json",
)

print("\n" + "=" * 60)
print("COMPLETE — all outputs written to outputs/")
print("=" * 60)
