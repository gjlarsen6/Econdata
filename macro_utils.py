"""
Shared utilities for macroeconomic LightGBM forecasting models.
Used by business_env_model.py, consumer_demand_model.py, cost_of_capital_model.py.
"""

import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabulate import tabulate

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

LGB_PARAMS = dict(
    n_estimators=2000, num_leaves=31, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, random_state=42, verbose=-1,
)

# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, series_cols: list) -> pd.DataFrame:
    """Add lag, rolling, momentum, and time features for all series."""
    d = df.copy()
    for col in series_cols:
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
    if len(series_cols) >= 2:
        d[f"cross_{series_cols[0]}_{series_cols[1]}"] = (
            d[f"{series_cols[0]}_lag1"] * d[f"{series_cols[1]}_lag1"]
        )
    return d

# ── Model training ────────────────────────────────────────────────────────────

def fit_model(X_tr, y_tr, X_vl, y_vl, objective="regression", alpha=None):
    params = {**LGB_PARAMS, "objective": objective}
    if alpha is not None:
        params["alpha"] = alpha
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
          callbacks=[lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(-1)])
    return m

def train_series_models(series_cols, X_tr, y_tr_dict, X_vl, y_vl_dict,
                        train_masks=None):
    """
    Train median + 10th/90th quantile models for each series.
    train_masks: optional dict {col: bool_array} to subset training rows per series.
    """
    models = {}
    for col in series_cols:
        if train_masks and col in train_masks:
            mask = train_masks[col]
            Xtr, ytr = X_tr[mask], y_tr_dict[col][mask]
            mask_v = train_masks.get(f"{col}_val", slice(None))
            Xvl, yvl = X_vl[mask_v] if not isinstance(mask_v, slice) else X_vl, \
                       y_vl_dict[col][mask_v] if not isinstance(mask_v, slice) else y_vl_dict[col]
        else:
            Xtr, ytr, Xvl, yvl = X_tr, y_tr_dict[col], X_vl, y_vl_dict[col]

        print(f"    {col}: {len(Xtr)} train rows  →  ", end="", flush=True)
        models[col] = {
            "mid": fit_model(Xtr, ytr, Xvl, yvl),
            "lo":  fit_model(Xtr, ytr, Xvl, yvl, "quantile", 0.10),
            "hi":  fit_model(Xtr, ytr, Xvl, yvl, "quantile", 0.90),
        }
        print(f"best iter {models[col]['mid'].best_iteration_}")
    return models

# ── Recursive forecasting ─────────────────────────────────────────────────────

def joint_recursive_forecast(df_base, models, feature_cols, series_cols,
                              clip_ranges, horizon=12):
    """
    Jointly forecast all series for `horizon` months using recursive prediction.
    df_base : DataFrame with 'date' + series_cols columns (full historical data)
    models  : {col: {"mid", "lo", "hi"}}
    Returns : {col: DataFrame(date, mid, lo, hi)}
    """
    work = df_base[["date"] + series_cols].copy()
    out  = {col: {"dates": [], "mid": [], "lo": [], "hi": []}
            for col in series_cols if col in models}

    for _ in range(horizon):
        next_date   = work["date"].iloc[-1] + pd.DateOffset(months=1)
        placeholder = {col: np.nan for col in series_cols}
        placeholder["date"] = next_date
        temp      = pd.concat([work, pd.DataFrame([placeholder])], ignore_index=True)
        temp_feat = engineer_features(temp, series_cols)
        feat_row  = temp_feat[feature_cols].iloc[[-1]]

        step_preds = {}
        for col in series_cols:
            if col not in models:
                continue
            lo_v, hi_v = clip_ranges.get(col, (-np.inf, np.inf))
            pm = float(np.clip(models[col]["mid"].predict(feat_row)[0], lo_v, hi_v))
            pl = float(np.clip(models[col]["lo"].predict(feat_row)[0],  lo_v, hi_v))
            ph = float(np.clip(models[col]["hi"].predict(feat_row)[0],  lo_v, hi_v))
            pl, ph = min(pl, pm), max(ph, pm)
            step_preds[col] = (pm, pl, ph)
            out[col]["dates"].append(next_date)
            out[col]["mid"].append(pm)
            out[col]["lo"].append(pl)
            out[col]["hi"].append(ph)

        new_row = {"date": next_date}
        for col in series_cols:
            new_row[col] = step_preds[col][0] if col in step_preds else work[col].iloc[-1]
        work = pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)

    return {col: pd.DataFrame(out[col]) for col in out}

# ── Console output ────────────────────────────────────────────────────────────

def print_validation_metrics(y_val_dict, val_preds_dict, series_info):
    rows = []
    for col, label, *_ in series_info:
        if col not in val_preds_dict:
            continue
        actual, pred = y_val_dict[col].values, val_preds_dict[col]
        rows.append({
            "Series": label,
            "MAE":  f"{mean_absolute_error(actual, pred):.3f}",
            "RMSE": f"{np.sqrt(mean_squared_error(actual, pred)):.3f}",
            "R²":   f"{r2_score(actual, pred):.4f}",
        })
    print(tabulate(rows, headers="keys", tablefmt="rounded_outline"))

def print_forecast_table(fc_dict, series_info):
    for col, label, color, unit, *_ in series_info:
        if col not in fc_dict:
            continue
        fc = fc_dict[col]
        rows = [{
            "Month":   r["dates"].strftime("%Y-%m"),
            f"Mid ({unit})": f"{r['mid']:,.2f}",
            "80% CI":  f"[{r['lo']:,.2f},  {r['hi']:,.2f}]",
        } for _, r in fc.iterrows()]
        print(f"\n  {label} ({col})")
        print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))

# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_forecast_dashboard(df_hist, fc_dict, series_info, last_date,
                             title, save_path, history_months=120, ncols=1):
    n     = len(series_info)
    nrows = -(-n // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(13 * ncols, 4.2 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    hist_start = last_date - pd.DateOffset(months=history_months)
    df_disp    = df_hist[df_hist["date"] >= hist_start]

    for idx, (col, label, color, unit, threshold, thr_label) in enumerate(series_info):
        ax = axes[idx // ncols][idx % ncols]

        ax.plot(df_disp["date"], df_disp[col],
                color=color, linewidth=2, label="Actual")

        if col in fc_dict:
            fc = fc_dict[col]
            ax.fill_between(fc["dates"], fc["lo"], fc["hi"],
                            color=color, alpha=0.15, label="80% PI")
            ax.plot(fc["dates"], fc["mid"], color=color,
                    linewidth=2, linestyle="--", label="Forecast")
            curr = df_hist[col].dropna().iloc[-1]
            ax.annotate(f"Now: {curr:,.1f}{unit}",
                        xy=(last_date, curr), xytext=(8, 8),
                        textcoords="offset points", fontsize=8,
                        color=color, fontweight="bold")
            ax.annotate(f"→ {fc['mid'].iloc[-1]:,.1f}{unit}",
                        xy=(fc["dates"].iloc[-1], fc["mid"].iloc[-1]),
                        xytext=(-55, 8), textcoords="offset points",
                        fontsize=8, color=color)

        ax.axvline(last_date, color="grey", linewidth=1.2,
                   linestyle=":", alpha=0.7, label="Last known")
        if threshold is not None:
            ax.axhline(threshold, color="grey", linewidth=0.8,
                       linestyle=":", alpha=0.6, label=thr_label)

        ax.set_title(f"{col} — {label}", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"{label}" + (f" ({unit})" if unit else ""), fontsize=9)
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # Hide unused subplots
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_validation_performance(dates_val, y_val_dict, val_preds_dict,
                                 series_info, title, save_path):
    n    = sum(1 for col, *_ in series_info if col in val_preds_dict)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    dates = pd.to_datetime(dates_val)
    row   = 0

    for col, label, color, unit, *_ in series_info:
        if col not in val_preds_dict:
            continue
        actual = y_val_dict[col].values
        pred   = val_preds_dict[col]

        # Time series comparison
        ax = axes[row][0]
        ax.plot(dates, actual, color=color, linewidth=2, label="Actual")
        ax.plot(dates, pred,   color=color, linewidth=1.5,
                linestyle="--", alpha=0.85, label="Predicted")
        ax.set_title(f"{label} — Time Series", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"{unit}" if unit else label, fontsize=9)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

        # Scatter
        ax = axes[row][1]
        ax.scatter(actual, pred, alpha=0.65, color=color,
                   edgecolors="white", linewidths=0.4, s=45)
        mn = min(actual.min(), pred.min())
        mx = max(actual.max(), pred.max())
        pad = (mx - mn) * 0.05
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad],
                "k--", linewidth=1, alpha=0.5, label="Perfect fit")
        r2  = r2_score(actual, pred)
        mae = mean_absolute_error(actual, pred)
        ax.set_title(f"{label}\nR²={r2:.4f}  MAE={mae:.3f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Actual", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.legend(fontsize=8)
        row += 1

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_feature_importance(models_dict, feature_cols, series_info,
                             title, save_path, top_n=12):
    n     = sum(1 for col, *_ in series_info if col in models_dict)
    ncols = 2
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(14, 4.5 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    idx = 0

    for col, label, color, *_ in series_info:
        if col not in models_dict:
            continue
        ax = axes[idx // ncols][idx % ncols]
        imp = pd.Series(
            models_dict[col]["mid"].booster_.feature_importance(importance_type="gain"),
            index=feature_cols,
        ).sort_values(ascending=True).tail(top_n)

        bars = ax.barh(imp.index, imp.values,
                       color=color, alpha=0.82, edgecolor="white")
        for bar in bars:
            w = bar.get_width()
            ax.text(w * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{w:,.0f}", va="center", fontsize=7)
        ax.set_xlim(0, imp.values.max() * 1.18)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Importance (Gain)", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        idx += 1

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── Result persistence ────────────────────────────────────────────────────────

def save_model_results(group_name: str, df_hist,
                        fc_dict: dict, y_val_dict: dict,
                        val_preds_dict: dict, series_info: list,
                        save_path: str):
    """
    Serialise model outputs to JSON so fred_refresh.py can build the summary table.
    series_info items: (col, label, color, unit, threshold, threshold_label)
    """
    payload = {
        "group":  group_name,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "series": [],
    }

    for item in series_info:
        col, label, _, unit = item[0], item[1], item[2], item[3]
        if col not in fc_dict:
            continue

        fc        = fc_dict[col]
        valid_idx = df_hist[col].last_valid_index()
        last_val  = float(df_hist[col].iloc[valid_idx]) if valid_idx is not None else None
        last_date = df_hist["date"].iloc[valid_idx].strftime("%Y-%m") if valid_idx is not None else None

        val_metrics = {}
        if col in val_preds_dict and col in y_val_dict:
            actual = np.asarray(y_val_dict[col], dtype=float)
            pred   = np.asarray(val_preds_dict[col], dtype=float)
            mask   = ~np.isnan(actual) & ~np.isnan(pred)
            if mask.sum() > 1:
                val_metrics = {
                    "mae":  round(float(mean_absolute_error(actual[mask], pred[mask])), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(actual[mask], pred[mask]))), 4),
                    "r2":   round(float(r2_score(actual[mask], pred[mask])), 4),
                }

        forecast_list = [
            {
                "month": row["dates"].strftime("%Y-%m"),
                "mid":   round(float(row["mid"]), 4),
                "lo":    round(float(row["lo"]),  4),
                "hi":    round(float(row["hi"]),  4),
            }
            for _, row in fc.iterrows()
        ]

        payload["series"].append({
            "series_id":  col,
            "label":      label,
            "unit":       unit if unit else "",
            "last_date":  last_date,
            "last_value": round(last_val, 4) if last_val is not None else None,
            "validation": val_metrics,
            "forecast":   forecast_list,
        })

    with open(save_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  Results saved → {save_path}")
