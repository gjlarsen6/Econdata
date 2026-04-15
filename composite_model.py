"""
composite_model.py — Financial Stress Index (FSI) and Market Regime Classifier.

Computes derived signals from already-fetched FRED data — no new API calls.

FSI = mean of 5 percentile-rank-normalized components:
  1. VIXCLS          (from data/MarketRisk/)
  2. BAMLH0A0HYM2    (from data/MarketRisk/)
  3. BAMLC0A0CM      (from data/MarketRisk/)
  4. RECPROUSM156N   (from data/RiskLeadingInd/)
  5. -T10Y2Y         (inverted: deeper inversion = more stress)

Scale: 0 (calm) → 1 (extreme stress).
Thresholds: < 0.25 expansion | 0.25–0.45 slowdown | 0.45–0.65 contraction | > 0.65 stress.

Outputs:
  outputs/results_financial_stress.json   — FSI time series + 12-month LightGBM forecast
  outputs/regime_history.json             — monthly regime labels + current state
  outputs/fsi_dashboard.png
"""

import json
import os
import sys
import warnings
from datetime import datetime
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

plt.rcParams.update({
    "font.family": "sans-serif", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linestyle": "--",
})

FORECAST_HORIZON  = 12
VALIDATION_MONTHS = 24

LGB_PARAMS = dict(
    n_estimators=2000, num_leaves=31, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, random_state=42, verbose=-1,
)

# ── 1. Load component series ───────────────────────────────────────────────────

print("=" * 60)
print("Financial Stress Index + Market Regime Classifier")
print("=" * 60)
print("\nLoading component series...")


def _load_daily_monthly(path: str, col: str) -> pd.DataFrame:
    """Load daily FRED CSV, resample to monthly mean."""
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df["date"] = df["observation_date"].dt.to_period("M").dt.to_timestamp()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.groupby("date")[col].mean().reset_index()


def _load_monthly(path: str, col: str) -> pd.DataFrame:
    """Load already-monthly FRED CSV."""
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df["date"] = df["observation_date"].dt.to_period("M").dt.to_timestamp()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["date", col]].drop_duplicates("date").sort_values("date")


COMPONENT_PATHS = {
    "VIXCLS":          ("data/MarketRisk/VIXCLS.csv",          "daily"),
    "BAMLH0A0HYM2":    ("data/MarketRisk/BAMLH0A0HYM2.csv",    "daily"),
    "BAMLC0A0CM":      ("data/MarketRisk/BAMLC0A0CM.csv",      "daily"),
    "RECPROUSM156N":   ("data/RiskLeadingInd/RECPROUSM156N.csv","monthly"),
    "T10Y2Y":          ("data/CostOfCapital/T10Y2Y.csv",        "daily"),
}

frames: dict[str, pd.DataFrame] = {}
for col, (path, freq) in COMPONENT_PATHS.items():
    try:
        if freq == "daily":
            frames[col] = _load_daily_monthly(path, col)
        else:
            frames[col] = _load_monthly(path, col)
        last_val = frames[col][col].dropna().iloc[-1]
        print(f"  OK    {col:<22}  latest={last_val:.3f}")
    except FileNotFoundError:
        print(f"  SKIP  {col}: {path} not found (run fred_refresh.py first)")

if len(frames) < 3:
    print("\nERROR: Need at least VIXCLS, BAMLH0A0HYM2, and RECPROUSM156N to compute FSI.")
    sys.exit(1)

# Outer merge all components
df = list(frames.values())[0]
for f in list(frames.values())[1:]:
    df = pd.merge(df, f, on="date", how="outer")

df = df.sort_values("date").reset_index(drop=True)

# Interpolate short gaps (≤2 months)
for col in frames:
    df[col] = pd.to_numeric(df[col], errors="coerce").interpolate(
        method="linear", limit=2
    )

# Restrict to when credit spreads are available (limits to ~1996-12-31)
if "BAMLH0A0HYM2" in df.columns:
    first_spread = df.loc[df["BAMLH0A0HYM2"].notna(), "date"].min()
    df = df[df["date"] >= first_spread].copy().reset_index(drop=True)

last_date = df["date"].max()
print(f"\n  Combined: {len(df)} monthly rows  "
      f"{df['date'].min().strftime('%Y-%m')} → {last_date.strftime('%Y-%m')}")

# ── 2. Compute Financial Stress Index ─────────────────────────────────────────

print("\nComputing Financial Stress Index...")


def _pct_rank(series: pd.Series) -> pd.Series:
    """Rolling historical percentile rank using expanding window."""
    return series.expanding(min_periods=12).rank(pct=True)


available_components: list[str] = []
component_cols: list[str] = []

# Build component signals (all normalized to [0,1] where 1 = more stress)
components: dict[str, pd.Series] = {}

if "VIXCLS" in df.columns:
    components["vix_norm"]    = _pct_rank(df["VIXCLS"])
    available_components.append("VIXCLS")

if "BAMLH0A0HYM2" in df.columns:
    components["hy_norm"]     = _pct_rank(df["BAMLH0A0HYM2"])
    available_components.append("BAMLH0A0HYM2")

if "BAMLC0A0CM" in df.columns:
    components["ig_norm"]     = _pct_rank(df["BAMLC0A0CM"])
    available_components.append("BAMLC0A0CM")

if "RECPROUSM156N" in df.columns:
    components["recpro_norm"] = _pct_rank(df["RECPROUSM156N"])
    available_components.append("RECPROUSM156N")

if "T10Y2Y" in df.columns:
    # Invert: more negative (deeper inversion) = more stress
    components["inversion_norm"] = _pct_rank(-df["T10Y2Y"])
    available_components.append("T10Y2Y")

n_components = len(components)
print(f"  Components: {n_components}")

# FSI = equal-weighted average of normalized components
comp_df = pd.DataFrame(components)
df["FSI"] = comp_df.mean(axis=1)

# Drop rows without FSI (early rows where expanding window hasn't warmed up)
df = df.dropna(subset=["FSI"]).reset_index(drop=True)
last_date = df["date"].max()

fsi_current = df["FSI"].iloc[-1]
print(f"  FSI range: {df['FSI'].min():.3f} – {df['FSI'].max():.3f}")
print(f"  Current FSI: {fsi_current:.3f}")

# ── 3. Market Regime Classification ───────────────────────────────────────────

def _classify_regime(fsi: float, fsi_prev3: float) -> str:
    """Deterministic rule-based regime from FSI value + 3-month trend."""
    if fsi > 0.65:
        return "stress"
    if fsi > 0.45:
        # Distinguish contraction vs recovery by recent trend
        return "contraction" if fsi >= fsi_prev3 else "recovery"
    if fsi > 0.25:
        return "slowdown"
    return "expansion"


df["fsi_prev3"] = df["FSI"].shift(3).fillna(df["FSI"])
df["regime"]    = df.apply(
    lambda r: _classify_regime(r["FSI"], r["fsi_prev3"]), axis=1
)

current_regime = df["regime"].iloc[-1]
print(f"  Current regime: {current_regime}")

# Regime summary
regime_counts = df["regime"].value_counts()
print("  Historical regime distribution:")
for regime, count in regime_counts.items():
    pct = count / len(df) * 100
    print(f"    {regime:<14}  {count:3d} months  ({pct:.1f}%)")

# ── 4. Feature engineering for FSI forecast ───────────────────────────────────

print("\nEngineering FSI features...")


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for lag in [1, 2, 3, 6, 12]:
        d[f"FSI_lag{lag}"] = d["FSI"].shift(lag)
    for w in [3, 6, 12]:
        rolled = d["FSI"].shift(1).rolling(w, min_periods=max(1, w // 2))
        d[f"FSI_roll{w}_mean"] = rolled.mean()
        d[f"FSI_roll{w}_std"]  = rolled.std()
    d["FSI_mom"]   = d["FSI"].shift(1).pct_change(1) * 100
    d["FSI_trend"] = d["FSI_roll3_mean"] - d["FSI_roll12_mean"]  # short vs long MA
    d["month"]     = d["date"].dt.month
    d["quarter"]   = d["date"].dt.quarter
    # Include raw component lags as features
    for col in available_components:
        if col in d.columns:
            d[f"{col}_lag1"] = d[col].shift(1)
    return d


df_feat = _engineer(df).dropna(subset=["FSI_lag1"]).reset_index(drop=True)
FEATURE_COLS = [c for c in df_feat.columns
                if c not in ["date", "FSI", "fsi_prev3", "regime"] + list(frames.keys())]
print(f"  Feature columns: {len(FEATURE_COLS)}")

# ── 5. Train / val split ───────────────────────────────────────────────────────

split_idx = len(df_feat) - VALIDATION_MONTHS
X_train   = df_feat[FEATURE_COLS].iloc[:split_idx]
X_val     = df_feat[FEATURE_COLS].iloc[split_idx:]
y_train   = df_feat["FSI"].iloc[:split_idx]
y_val     = df_feat["FSI"].iloc[split_idx:]
dates_val = df_feat["date"].iloc[split_idx:].values

print(f"  Train: {split_idx} months  |  Validation: {VALIDATION_MONTHS} months")

# ── 6. Train LightGBM on FSI ──────────────────────────────────────────────────

print("\nTraining FSI forecast model...")


def _fit(Xtr, ytr, Xvl, yvl, objective="regression", alpha=None):
    params = {**LGB_PARAMS, "objective": objective}
    if alpha is not None:
        params["alpha"] = alpha
    m = lgb.LGBMRegressor(**params)
    m.fit(Xtr, ytr, eval_set=[(Xvl, yvl)],
          callbacks=[lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(-1)])
    return m


fsi_models = {
    "mid": _fit(X_train, y_train, X_val, y_val),
    "lo":  _fit(X_train, y_train, X_val, y_val, "quantile", 0.10),
    "hi":  _fit(X_train, y_train, X_val, y_val, "quantile", 0.90),
}
print(f"  Best iteration: {fsi_models['mid'].best_iteration_}")

# Validation
val_pred = fsi_models["mid"].predict(X_val)
mae  = mean_absolute_error(y_val, val_pred)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
r2   = r2_score(y_val, val_pred)
print(f"  Validation — MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

# ── 7. Recursive FSI forecast ──────────────────────────────────────────────────

print(f"\nGenerating {FORECAST_HORIZON}-month FSI forecast...")

work_df    = df[["date", "FSI"] + list(frames.keys())].copy()
work_df["fsi_prev3"] = work_df["FSI"].shift(3).fillna(work_df["FSI"])
work_df["regime"]    = work_df.apply(
    lambda r: _classify_regime(r["FSI"], r["fsi_prev3"]), axis=1
)

fcast = {"dates": [], "mid": [], "lo": [], "hi": [], "regime": []}

for _ in range(FORECAST_HORIZON):
    next_date   = work_df["date"].iloc[-1] + pd.DateOffset(months=1)
    placeholder = {"date": next_date, "FSI": np.nan, "fsi_prev3": np.nan, "regime": ""}
    for col in frames:
        placeholder[col] = work_df[col].iloc[-1]  # carry forward last known
    temp      = pd.concat([work_df, pd.DataFrame([placeholder])], ignore_index=True)
    temp_feat = _engineer(temp)
    feat_row  = temp_feat[FEATURE_COLS].iloc[[-1]]

    pm = float(np.clip(fsi_models["mid"].predict(feat_row)[0], 0.0, 1.0))
    pl = float(np.clip(fsi_models["lo"].predict(feat_row)[0],  0.0, 1.0))
    ph = float(np.clip(fsi_models["hi"].predict(feat_row)[0],  0.0, 1.0))
    pl, ph = min(pl, pm), max(ph, pm)

    # Compute regime for forecast step
    fsi_prev3 = work_df["FSI"].iloc[-3] if len(work_df) >= 3 else pm
    fc_regime = _classify_regime(pm, float(fsi_prev3))

    fcast["dates"].append(next_date)
    fcast["mid"].append(pm)
    fcast["lo"].append(pl)
    fcast["hi"].append(ph)
    fcast["regime"].append(fc_regime)

    new_row = {"date": next_date, "FSI": pm, "fsi_prev3": fsi_prev3, "regime": fc_regime}
    for col in frames:
        new_row[col] = placeholder[col]
    work_df = pd.concat([work_df, pd.DataFrame([new_row])], ignore_index=True)

fc_df = pd.DataFrame(fcast)

# Print FSI forecast table
print(f"\nFSI Forecast  (from {last_date.strftime('%Y-%m')})")
print(tabulate(
    [{"Month": r["dates"].strftime("%Y-%m"), "FSI Mid": f"{r['mid']:.3f}",
      "80% CI": f"[{r['lo']:.3f}, {r['hi']:.3f}]", "Regime": r["regime"]}
     for _, r in fc_df.iterrows()],
    headers="keys", tablefmt="rounded_outline", stralign="right",
))

# ── 8. Plots ───────────────────────────────────────────────────────────────────

REGIME_COLORS = {
    "expansion":   "#27ae60",
    "slowdown":    "#f39c12",
    "contraction": "#e74c3c",
    "stress":      "#8e44ad",
    "recovery":    "#2980b9",
}

hist_start = last_date - pd.DateOffset(months=120)
df_disp    = df[df["date"] >= hist_start].copy()

fig, axes = plt.subplots(2, 1, figsize=(13, 10))
fig.suptitle("Financial Stress Index — 12-Month Forecast Dashboard",
             fontsize=13, fontweight="bold", y=1.01)

# Panel 1: FSI history + forecast
ax = axes[0]
ax.plot(df_disp["date"], df_disp["FSI"], color="#2c3e50", linewidth=2, label="FSI (historical)")
ax.fill_between(fc_df["dates"], fc_df["lo"], fc_df["hi"],
                color="#2c3e50", alpha=0.15, label="80% PI")
ax.plot(fc_df["dates"], fc_df["mid"], color="#2c3e50",
        linewidth=2, linestyle="--", label="FSI Forecast")
ax.axvline(last_date, color="grey", linewidth=1.2, linestyle=":", alpha=0.7)
ax.axhline(0.25, color="#f39c12", linewidth=0.8, linestyle=":", alpha=0.7, label="Slowdown (0.25)")
ax.axhline(0.45, color="#e74c3c", linewidth=0.8, linestyle=":", alpha=0.7, label="Contraction (0.45)")
ax.axhline(0.65, color="#8e44ad", linewidth=0.8, linestyle=":", alpha=0.7, label="Stress (0.65)")
ax.annotate(f"Current FSI: {fsi_current:.3f} ({current_regime})",
            xy=(last_date, fsi_current),
            xytext=(10, 8), textcoords="offset points",
            fontsize=9, color="#2c3e50", fontweight="bold")
ax.set_ylim(-0.05, 1.05)
ax.set_ylabel("Financial Stress Index", fontsize=10)
ax.set_title("Financial Stress Index", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="upper left", ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

# Panel 2: Regime history (colored spans)
ax = axes[1]
ax.plot(df_disp["date"], df_disp["FSI"], color="#2c3e50", linewidth=1.5, alpha=0.6)
prev_regime = None
span_start  = None
for _, row in df_disp.iterrows():
    r = row["regime"]
    if r != prev_regime:
        if prev_regime is not None and span_start is not None:
            ax.axvspan(span_start, row["date"],
                       alpha=0.2, color=REGIME_COLORS.get(prev_regime, "#888888"))
        span_start  = row["date"]
        prev_regime = r
if prev_regime and span_start is not None:
    ax.axvspan(span_start, df_disp["date"].iloc[-1],
               alpha=0.2, color=REGIME_COLORS.get(prev_regime, "#888888"),
               label=prev_regime)
# Add legend patches
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=c, alpha=0.6, label=r)
           for r, c in REGIME_COLORS.items()]
ax.legend(handles=patches, fontsize=8, loc="upper left", ncol=5)
ax.set_ylabel("FSI", fontsize=10)
ax.set_title("Market Regime History", fontsize=11, fontweight="bold")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

fig.tight_layout()
fig.savefig("outputs/fsi_dashboard.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved outputs/fsi_dashboard.png")

# ── 9. Save results_financial_stress.json ─────────────────────────────────────

fc_for_save = {"FSI": pd.DataFrame({
    "dates": fc_df["dates"],
    "mid":   fc_df["mid"],
    "lo":    fc_df["lo"],
    "hi":    fc_df["hi"],
})}

fsi_series_info = [
    ("FSI", "Financial Stress Index", "#2c3e50", "Score (0–1)", 0.45, "Stress threshold (0.45)")
]

from macro_utils import save_model_results as _smr
_smr(
    "Financial Stress Index", df, fc_for_save,
    {"FSI": y_val},
    {"FSI": val_pred},
    fsi_series_info,
    "outputs/results_financial_stress.json",
)

# ── 10. Save regime_history.json ──────────────────────────────────────────────

history_rows = [
    {"date": row["date"].strftime("%Y-%m-%d"),
     "fsi":  round(float(row["FSI"]), 4),
     "regime": row["regime"]}
    for _, row in df.iterrows()
    if pd.notna(row["FSI"])
]

# Compute regime distribution for response
regime_dist: dict[str, float] = {}
for r in ["expansion", "slowdown", "contraction", "stress", "recovery"]:
    count = sum(1 for h in history_rows if h["regime"] == r)
    regime_dist[r] = round(count / max(len(history_rows), 1), 4)

# Forecast regime
forecast_regimes = [
    {"month": row["dates"].strftime("%Y-%m"),
     "fsi":   round(float(row["mid"]), 4),
     "regime": row["regime"]}
    for _, row in fc_df.iterrows()
]

regime_payload = {
    "generated_at":     datetime.now().isoformat(timespec="seconds"),
    "current_regime":   current_regime,
    "current_fsi":      round(float(fsi_current), 4),
    "regime_distribution": regime_dist,
    "forecast":         forecast_regimes,
    "history":          history_rows[-120:],  # last 10 years in response
}

with open("outputs/regime_history.json", "w") as fh:
    json.dump(regime_payload, fh, indent=2)
print("  Saved outputs/regime_history.json")

print("\n" + "=" * 60)
print("COMPLETE — all outputs written to outputs/")
print("=" * 60)
