"""
market_model.py — LightGBM forecasts for Market Risk and Commodity series.

Trains on:
  Market Risk:  VIXCLS, BAMLH0A0HYM2, BAMLC0A0CM, DTWEXBGS
  Commodities:  DCOILWTICO, NASDAQQGLDI

All series are daily in FRED; resampled to monthly mean before modeling.
Cross-features: HY-IG spread differential, VIX × HY spread, gold/oil ratio.

Outputs:
  outputs/results_market_risk.json       — VIX, credit spreads, USD index
  outputs/results_commodities.json       — WTI crude oil, gold
  outputs/market_risk_dashboard.png
  outputs/commodities_dashboard.png
  outputs/market_risk_importance.png
  outputs/commodities_importance.png
  outputs/market_risk_validation.png
  outputs/commodities_validation.png
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
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
VALIDATION_MONTHS = 12  # BAML credit spread series only available from 2023 on FRED
HISTORY_DISPLAY   = 120  # months for dashboard plots

MARKET_RISK_COLS = ["VIXCLS", "BAMLH0A0HYM2", "BAMLC0A0CM", "DTWEXBGS"]
COMMODITY_COLS   = ["DCOILWTICO", "NASDAQQGLDI"]
ALL_SERIES       = MARKET_RISK_COLS + COMMODITY_COLS

# (label, color, unit, threshold, threshold_label)
SERIES_META = {
    "VIXCLS":           ("VIX Volatility Index",     "#8e44ad", "Index", 30.0,  "Stress (30)"),
    "BAMLH0A0HYM2":     ("HY Credit Spread (OAS)",   "#c0392b", "%",     5.0,   "Stress (5%)"),
    "BAMLC0A0CM":       ("IG Credit Spread (OAS)",   "#e67e22", "%",     None,  None),
    "DTWEXBGS":         ("USD Broad Index",           "#2980b9", "Index", None,  None),
    "DCOILWTICO":       ("WTI Crude Oil",             "#27ae60", "$/bbl", None,  None),
    "NASDAQQGLDI": ("Gold Price Index (NASDAQ)", "#f39c12", "Index", None,  None),
}

CLIP_RANGES = {
    "VIXCLS":            (5.0,    90.0),
    "BAMLH0A0HYM2":      (1.0,    25.0),
    "BAMLC0A0CM":        (0.1,    10.0),
    "DTWEXBGS":          (80.0,  145.0),
    "DCOILWTICO":        (5.0,   200.0),
    "NASDAQQGLDI":  (200.0, 5000.0),
}

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
print("Market Risk + Commodities Model")
print("=" * 60)
print("\nLoading data...")


def _load_daily_csv(path: str, series_id: str) -> pd.DataFrame:
    """Load daily CSV, resample to monthly mean, return (date, series_id) DataFrame."""
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df["date"] = df["observation_date"].dt.to_period("M").dt.to_timestamp()
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return (
        df.groupby("date")[series_id]
        .mean()
        .reset_index()
    )


frames: list[pd.DataFrame] = []
available_series: list[str] = []

for sid in ALL_SERIES:
    group = "MarketRisk" if sid in MARKET_RISK_COLS else "Commodities"
    path  = f"data/{group}/{sid}.csv"
    try:
        mdf = _load_daily_csv(path, sid)
        if mdf[sid].notna().sum() < 24:
            print(f"  SKIP  {sid}: only {mdf[sid].notna().sum()} non-null rows")
            continue
        frames.append(mdf)
        available_series.append(sid)
        last_val = mdf[sid].dropna().iloc[-1]
        print(f"  OK    {sid:<22}  {mdf['date'].min().strftime('%Y-%m')} → "
              f"{mdf['date'].max().strftime('%Y-%m')}  latest={last_val:.2f}")
    except FileNotFoundError:
        print(f"  SKIP  {sid}: {path} not found (run fred_refresh.py first)")

if len(available_series) < 2:
    print("\nERROR: fewer than 2 series available — run fred_refresh.py to fetch data.")
    sys.exit(1)

# Outer merge so each series keeps its full history
df = frames[0]
for f in frames[1:]:
    df = pd.merge(df, f, on="date", how="outer")

df = df.sort_values("date").reset_index(drop=True)

# Restrict to window where credit spreads are available (1996-12-31 onward)
# Earlier rows have many NaNs which LightGBM handles but degrade features.
df = df[df["date"] >= "1996-01-01"].copy().reset_index(drop=True)

# Interpolate short gaps (≤2 months)
for col in available_series:
    df[col] = pd.to_numeric(df[col], errors="coerce").interpolate(
        method="linear", limit=2
    )

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
        d[f"{col}_yoy"] = d[col].shift(1).pct_change(12) * 100
    d["month"]   = d["date"].dt.month
    d["quarter"] = d["date"].dt.quarter
    # Market cross-features
    if "BAMLH0A0HYM2" in cols and "BAMLC0A0CM" in cols:
        d["hy_ig_diff"]  = d["BAMLH0A0HYM2_lag1"] - d["BAMLC0A0CM_lag1"]
    if "VIXCLS" in cols and "BAMLH0A0HYM2" in cols:
        d["vix_x_hy"]    = d["VIXCLS_lag1"] * d["BAMLH0A0HYM2_lag1"]
    if "DCOILWTICO" in cols and "NASDAQQGLDI" in cols:
        d["gold_oil_ratio"] = d["NASDAQQGLDI_lag1"] / (d["DCOILWTICO_lag1"] + 1e-9)
    return d


df_feat = _engineer(df, available_series).dropna(
    subset=[f"{c}_lag1" for c in available_series if f"{c}_lag1" in
            _engineer(df, available_series).columns]
).reset_index(drop=True)

FEATURE_COLS = [c for c in df_feat.columns if c not in ["date"] + available_series]
print(f"  Feature columns: {len(FEATURE_COLS)}")

# ── 3. Train / val split ───────────────────────────────────────────────────────

split_idx = len(df_feat) - VALIDATION_MONTHS
X_train   = df_feat[FEATURE_COLS].iloc[:split_idx]
X_val     = df_feat[FEATURE_COLS].iloc[split_idx:]
y_train   = {col: df_feat[col].iloc[:split_idx] for col in available_series}
y_val     = {col: df_feat[col].iloc[split_idx:]  for col in available_series}
dates_val = df_feat["date"].iloc[split_idx:].values

print(f"  Train: {split_idx} months  |  Validation: {VALIDATION_MONTHS} months")

# ── 4. Train LightGBM models ───────────────────────────────────────────────────

print("\nTraining models...")


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
for col in available_series:
    label = SERIES_META.get(col, (col,))[0]
    # Drop rows where this target is NaN
    tr_mask = y_train[col].notna()
    vl_mask = y_val[col].notna()
    if tr_mask.sum() < 24:
        print(f"  SKIP  {col}: only {tr_mask.sum()} training rows")
        continue
    print(f"  {label:<32} ({col})  train={tr_mask.sum()} ...", end="", flush=True)
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
for col in available_series:
    if col not in models:
        continue
    preds = models[col]["mid"].predict(X_val)
    val_preds[col] = preds
    valid   = y_val[col].notna()
    actual  = y_val[col][valid].values
    pred    = preds[valid]
    label   = SERIES_META.get(col, (col,))[0]
    mae     = mean_absolute_error(actual, pred)
    rmse    = np.sqrt(mean_squared_error(actual, pred))
    r2      = r2_score(actual, pred)
    val_rows.append({"Series": label[:30], "MAE": f"{mae:.3f}",
                     "RMSE": f"{rmse:.3f}", "R²": f"{r2:.4f}"})

print("\n" + "=" * 60)
print("VALIDATION METRICS  (last 24 months held out)")
print("=" * 60)
print(tabulate(val_rows, headers="keys", tablefmt="rounded_outline"))

# ── 6. Recursive forecast ──────────────────────────────────────────────────────

print(f"\nGenerating {FORECAST_HORIZON}-month recursive forecasts...")

work  = df[["date"] + available_series].copy()
fcast = {col: {"dates": [], "mid": [], "lo": [], "hi": []}
         for col in available_series if col in models}

for _ in range(FORECAST_HORIZON):
    next_date   = work["date"].iloc[-1] + pd.DateOffset(months=1)
    placeholder = {col: np.nan for col in available_series}
    placeholder["date"] = next_date
    temp      = pd.concat([work, pd.DataFrame([placeholder])], ignore_index=True)
    temp_feat = _engineer(temp, available_series)
    feat_row  = temp_feat[FEATURE_COLS].iloc[[-1]]

    step_preds: dict = {}
    for col in available_series:
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
    for col in available_series:
        new_row[col] = step_preds[col][0] if col in step_preds else work[col].iloc[-1]
    work = pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)

fc = {col: pd.DataFrame(fcast[col]) for col in fcast}

# ── 7. Plots ───────────────────────────────────────────────────────────────────

hist_start = last_date - pd.DateOffset(months=HISTORY_DISPLAY)
df_disp    = df[df["date"] >= hist_start]


def _plot_dashboard(cols: list[str], title: str, save_path: str) -> None:
    plot_cols = [c for c in cols if c in fc]
    if not plot_cols:
        return
    n     = len(plot_cols)
    nrows = -(-n // 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4.5 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    for idx, col in enumerate(plot_cols):
        meta  = SERIES_META.get(col, (col, "#888888", "", None, None))
        label, color, unit, thresh, thresh_lbl = meta
        ax    = axes[idx // 2][idx % 2]
        ax.plot(df_disp["date"], df_disp[col], color=color, linewidth=2, label="Actual")
        ax.fill_between(fc[col]["dates"], fc[col]["lo"], fc[col]["hi"],
                        color=color, alpha=0.15, label="80% PI")
        ax.plot(fc[col]["dates"], fc[col]["mid"], color=color,
                linewidth=2, linestyle="--", label="Forecast")
        ax.axvline(last_date, color="grey", linewidth=1.2, linestyle=":", alpha=0.7)
        if thresh is not None:
            ax.axhline(thresh, color="grey", linewidth=0.8, linestyle=":", alpha=0.6,
                       label=thresh_lbl)
        curr = df[col].dropna().iloc[-1]
        ax.annotate(f"Now: {curr:,.2f}", xy=(last_date, curr),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=8, color=color, fontweight="bold")
        ax.set_title(f"{col} — {label}", fontsize=10, fontweight="bold")
        ax.set_ylabel(unit or label, fontsize=9)
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    for j in range(n, nrows * 2):
        axes[j // 2][j % 2].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def _plot_importance(cols: list[str], title: str, save_path: str) -> None:
    plot_cols = [c for c in cols if c in models]
    if not plot_cols:
        return
    n     = len(plot_cols)
    nrows = -(-n // 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4.5 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for idx, col in enumerate(plot_cols):
        meta  = SERIES_META.get(col, (col, "#888888", "", None, None))
        label, color = meta[0], meta[1]
        ax    = axes[idx // 2][idx % 2]
        imp   = pd.Series(
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
    for j in range(n, nrows * 2):
        axes[j // 2][j % 2].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


_plot_dashboard(
    [c for c in MARKET_RISK_COLS if c in fc],
    "Market Risk Forecast Dashboard — VIX, Credit Spreads, USD Index",
    "outputs/market_risk_dashboard.png",
)
_plot_dashboard(
    [c for c in COMMODITY_COLS if c in fc],
    "Commodities Forecast Dashboard — WTI Crude Oil, Gold",
    "outputs/commodities_dashboard.png",
)
_plot_importance(
    [c for c in MARKET_RISK_COLS if c in models],
    "Market Risk — Feature Importance (Gain)",
    "outputs/market_risk_importance.png",
)
_plot_importance(
    [c for c in COMMODITY_COLS if c in models],
    "Commodities — Feature Importance (Gain)",
    "outputs/commodities_importance.png",
)

# ── 8. Save results JSON ───────────────────────────────────────────────────────

def _series_info_tuples(cols: list[str]) -> list[tuple]:
    return [
        (c,) + SERIES_META.get(c, (c, "#888888", "", None, None))
        for c in cols if c in fc
    ]


mr_cols = [c for c in MARKET_RISK_COLS if c in fc]
if mr_cols:
    _save_model_results(
        "Market Risk", df, fc,
        {c: y_val[c] for c in mr_cols if c in y_val},
        {c: val_preds[c] for c in mr_cols if c in val_preds},
        _series_info_tuples(mr_cols),
        "outputs/results_market_risk.json",
    )

cm_cols = [c for c in COMMODITY_COLS if c in fc]
if cm_cols:
    _save_model_results(
        "Commodities", df, fc,
        {c: y_val[c] for c in cm_cols if c in y_val},
        {c: val_preds[c] for c in cm_cols if c in val_preds},
        _series_info_tuples(cm_cols),
        "outputs/results_commodities.json",
    )

print("\n" + "=" * 60)
print("COMPLETE — all outputs written to outputs/")
print("=" * 60)
