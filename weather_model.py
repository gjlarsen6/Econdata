"""
weather_model.py — LightGBM climate forecasting model.

Aggregates daily city-level Visual Crossing weather data to monthly state-level
climate indices, then further aggregates to US Census regions + national.
Trains LightGBM models using macro_utils.py patterns for 12-month forecasting of
climate variables with economic relevance.

Three model groups × up to five geographies (northeast, midwest, south, west, national):

  temperature_energy       — temp_mean, HDD, CDD, temperature anomaly
  precipitation_disruption — total precip, precip days, snow, extreme-precip days
  extremes_composite       — extreme heat/cold days, wind speed, cloud cover

Usage (invoked as subprocess from fred_refresh.py or standalone):
    python3 weather_model.py --geo national
    python3 weather_model.py --geo all --source temperature_energy
    python3 weather_model.py --geo all --source all
    python3 weather_model.py --agg-only
    python3 weather_model.py --agg-only --force-agg --states CA,TX,FL
"""

from __future__ import annotations

import argparse
import logging
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

from macro_utils import (
    engineer_features, train_series_models, joint_recursive_forecast,
    print_validation_metrics, print_forecast_table,
    plot_forecast_dashboard, plot_validation_performance, plot_feature_importance,
    save_model_results,
)
from connectors.weather import WEATHER_DIR, _city_file_stem, _year_file_path

VALIDATION_MONTHS = 24
FORECAST_HORIZON  = 12
MIN_ROWS_REQUIRED = VALIDATION_MONTHS + 30

AGG_DIR        = DATA_DIR / "Weather" / "Aggregated" / "state"
BASELINE_START = 2000
BASELINE_END   = 2019
HDD_BASE       = 65   # °F standard base
CDD_BASE       = 65   # °F standard base
MIN_DAY_COUNT  = 20   # minimum valid daily obs for a month to be kept
MIN_CITIES     = 2    # minimum cities contributing to a state-month

GROUP_COLORS = [
    "#3498db", "#e74c3c", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]

# Extended column map — NOT shared with connectors/weather.py (_COL_MAP there uses 5 cols)
_WEATHER_MODEL_COL_MAP: dict[str, str] = {
    "Maximum Temperature": "max_temp",
    "Minimum Temperature": "min_temp",
    "Temperature":         "mean_temp",
    "Precipitation":       "precip",
    "Snow Depth":          "snow_depth",
    "Wind Speed":          "wind_speed",
    "Cloud Cover":         "cloud_cover",
}

# US Census Bureau 4 geographic regions + national
CENSUS_REGIONS: dict[str, list[str]] = {
    "northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "midwest":   ["IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"],
    "south":     ["AL", "AR", "DE", "FL", "GA", "KY", "LA", "MD", "MS",
                  "NC", "OK", "SC", "TN", "TX", "VA", "WV", "DC"],
    "west":      ["AK", "AZ", "CA", "CO", "HI", "ID", "MT", "NV", "NM",
                  "OR", "UT", "WA", "WY"],
    "national":  [],  # populated dynamically in build_national_df
}

SOURCE_CONFIG: dict[str, dict] = {
    "temperature_energy": {
        "series": ["temp_mean", "hdd", "cdd", "temp_anom"],
        "label_map": {
            "temp_mean":  "Mean Temperature (°F)",
            "hdd":        "Heating Degree Days",
            "cdd":        "Cooling Degree Days",
            "temp_anom":  "Temperature Anomaly (°F)",
        },
        "units": {
            "temp_mean": "°F", "hdd": "degree-days",
            "cdd": "degree-days", "temp_anom": "°F",
        },
        "group_name":   "Weather: Temperature & Energy Demand",
        "model_prefix": "weather_temperature",
        "results_file": "results_weather_temperature_{geo}.json",
        "plot_prefix":  "weather_temperature_{geo}",
    },
    "precipitation_disruption": {
        "series": ["precip_total", "precip_days", "snow_total", "extreme_precip_days"],
        "label_map": {
            "precip_total":        "Monthly Precipitation (in)",
            "precip_days":         "Precipitation Days",
            "snow_total":          "Snow Depth (in)",
            "extreme_precip_days": "Extreme Precip Days (>1in)",
        },
        "units": {
            "precip_total": "inches", "precip_days": "days",
            "snow_total": "inches", "extreme_precip_days": "days",
        },
        "group_name":   "Weather: Precipitation & Disruption",
        "model_prefix": "weather_precipitation",
        "results_file": "results_weather_precipitation_{geo}.json",
        "plot_prefix":  "weather_precipitation_{geo}",
    },
    "extremes_composite": {
        "series": ["extreme_heat_days", "extreme_cold_days", "wind_mean", "cloud_cover_mean"],
        "label_map": {
            "extreme_heat_days":  "Extreme Heat Days (>90°F)",
            "extreme_cold_days":  "Extreme Cold Days (<32°F)",
            "wind_mean":          "Mean Wind Speed (mph)",
            "cloud_cover_mean":   "Mean Cloud Cover (%)",
        },
        "units": {
            "extreme_heat_days": "days", "extreme_cold_days": "days",
            "wind_mean": "mph", "cloud_cover_mean": "%",
        },
        "group_name":   "Weather: Extreme Events & Renewables",
        "model_prefix": "weather_extremes",
        "results_file": "results_weather_extremes_{geo}.json",
        "plot_prefix":  "weather_extremes_{geo}",
    },
}


# ── Phase 1: Raw data loading ─────────────────────────────────────────────────

def load_city_daily(
    state: str,
    city: str,
    weather_dir: Path = WEATHER_DIR,
) -> pd.DataFrame:
    """Load all year CSV files for (state, city) into a single daily DataFrame.

    Uses _WEATHER_MODEL_COL_MAP (7 columns) rather than the 5-column map in
    connectors/weather.py.  Reuses _year_file_path / _city_file_stem from that
    module for consistent path resolution.

    Returns columns: date (datetime), max_temp, min_temp, mean_temp,
                     precip, snow_depth, wind_speed, cloud_cover.
    Returns empty DataFrame if no files are found or readable.
    """
    city_dir = weather_dir / state.upper() / city
    if not city_dir.exists():
        return pd.DataFrame()

    year_files = sorted(city_dir.glob("*.csv"))
    if not year_files:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in year_files:
        try:
            raw = pd.read_csv(path, low_memory=False)
            available = {src: dst for src, dst in _WEATHER_MODEL_COL_MAP.items()
                         if src in raw.columns}
            if not available:
                continue
            df = raw[["Date time", *available.keys()]].copy().rename(columns=available)
            df["date"] = pd.to_datetime(
                df["Date time"], format="%m/%d/%Y", errors="coerce"
            ).dt.normalize()
            df = df.drop(columns=["Date time"]).dropna(subset=["date"])
            for col in available.values():
                df[col] = pd.to_numeric(df[col], errors="coerce")
            frames.append(df)
        except Exception as exc:
            log.debug("load_city_daily: skipping %s — %s", path.name, exc)

    if not frames:
        return pd.DataFrame()

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return combined


def discover_state_cities(
    state: str,
    weather_dir: Path = WEATHER_DIR,
) -> list[str]:
    """Return sorted list of city directory names under weather_dir/STATE/."""
    state_dir = weather_dir / state.upper()
    if not state_dir.exists():
        return []
    return sorted(d.name for d in state_dir.iterdir() if d.is_dir())


# ── Phase 2: Daily → monthly aggregation ─────────────────────────────────────

def aggregate_city_to_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Resample one city's daily data to monthly climate indices.

    Input columns:  date, max_temp, min_temp, mean_temp, precip,
                    snow_depth, wind_speed, cloud_cover
    Output columns: date (month-start), temp_mean, hdd, cdd, precip_total,
                    precip_days, snow_total, extreme_heat_days, extreme_cold_days,
                    extreme_precip_days, wind_mean, cloud_cover_mean, day_count

    Months with day_count < MIN_DAY_COUNT are dropped (partial-month guard).
    Missing columns are silently skipped — handles 21-column vs 25-column files.
    """
    if df_daily.empty:
        return pd.DataFrame()

    df = df_daily.set_index("date").sort_index()

    def _col(name: str) -> pd.Series | None:
        return df[name] if name in df.columns else None

    mean_t  = _col("mean_temp")
    max_t   = _col("max_temp")
    min_t   = _col("min_temp")
    precip  = _col("precip")
    snow    = _col("snow_depth")
    wind    = _col("wind_speed")
    cloud   = _col("cloud_cover")

    if mean_t is None:
        return pd.DataFrame()

    parts: dict[str, pd.Series] = {
        "temp_mean": mean_t.resample("MS").mean(),
        "hdd":       mean_t.resample("MS").apply(
            lambda g: (HDD_BASE - g).clip(lower=0).sum()
        ),
        "cdd":       mean_t.resample("MS").apply(
            lambda g: (g - CDD_BASE).clip(lower=0).sum()
        ),
        "day_count": mean_t.resample("MS").count(),
    }

    if precip is not None:
        parts["precip_total"]        = precip.resample("MS").sum()
        parts["precip_days"]         = precip.resample("MS").apply(
            lambda g: (g > 0.01).sum()
        )
        parts["extreme_precip_days"] = precip.resample("MS").apply(
            lambda g: (g > 1.0).sum()
        )

    if snow is not None:
        parts["snow_total"] = snow.resample("MS").sum()

    if max_t is not None:
        parts["extreme_heat_days"] = max_t.resample("MS").apply(
            lambda g: (g > 90).sum()
        )

    if min_t is not None:
        parts["extreme_cold_days"] = min_t.resample("MS").apply(
            lambda g: (g < 32).sum()
        )

    if wind is not None:
        parts["wind_mean"] = wind.resample("MS").mean()

    if cloud is not None:
        parts["cloud_cover_mean"] = cloud.resample("MS").mean()

    monthly = pd.DataFrame(parts).reset_index()
    monthly = monthly[monthly["day_count"] >= MIN_DAY_COUNT].reset_index(drop=True)
    return monthly


def aggregate_state_monthly(
    state: str,
    weather_dir: Path = WEATHER_DIR,
    min_cities: int = MIN_CITIES,
) -> pd.DataFrame:
    """Aggregate all cities in a state to one monthly state-level DataFrame.

    Takes an unweighted mean across cities for each month.
    Months with fewer than min_cities valid cities are dropped.
    Returns same columns as aggregate_city_to_monthly() plus city_count.
    """
    cities = discover_state_cities(state, weather_dir)
    if not cities:
        log.warning("aggregate_state: no cities found for %s", state.upper())
        return pd.DataFrame()

    all_monthly: list[pd.DataFrame] = []
    for city in cities:
        daily = load_city_daily(state, city, weather_dir)
        if daily.empty:
            continue
        monthly = aggregate_city_to_monthly(daily)
        if not monthly.empty:
            all_monthly.append(monthly)

    if not all_monthly:
        log.warning("aggregate_state: no valid monthly data for %s", state.upper())
        return pd.DataFrame()

    # Tag each city then concatenate
    tagged: list[pd.DataFrame] = []
    for i, m in enumerate(all_monthly):
        tmp = m.copy()
        tmp["_city_idx"] = i
        tagged.append(tmp)
    concat = pd.concat(tagged, ignore_index=True)

    feat_cols = [c for c in concat.columns
                 if c not in ("date", "day_count", "_city_idx")]

    monthly_state = concat.groupby("date")[feat_cols].mean().reset_index()
    city_counts = (
        concat.groupby("date")["_city_idx"].nunique().rename("city_count")
    )
    monthly_state = monthly_state.merge(city_counts, on="date", how="left")
    monthly_state = (
        monthly_state[monthly_state["city_count"] >= min_cities]
        .reset_index(drop=True)
    )
    return monthly_state


def add_temperature_anomaly(df_state: pd.DataFrame) -> pd.DataFrame:
    """Add temp_anom = temp_mean − 2000-2019 monthly baseline mean.

    Baseline is computed from rows where year ∈ [BASELINE_START, BASELINE_END].
    If fewer than 10 baseline years exist for a calendar month, temp_anom is NaN.
    """
    if df_state.empty or "temp_mean" not in df_state.columns:
        return df_state

    df = df_state.copy()
    baseline = df[df["date"].dt.year.between(BASELINE_START, BASELINE_END)].copy()
    baseline["month_num"] = baseline["date"].dt.month

    baseline_year_counts = (
        baseline.groupby("month_num")["date"]
        .apply(lambda g: g.dt.year.nunique())
    )
    valid_months = baseline_year_counts[baseline_year_counts >= 10].index
    monthly_baseline = (
        baseline[baseline["month_num"].isin(valid_months)]
        .groupby("month_num")["temp_mean"]
        .mean()
    )

    df["temp_anom"] = df["date"].dt.month.map(monthly_baseline)
    df["temp_anom"] = df["temp_mean"] - df["temp_anom"]
    return df


def save_state_csv(
    df: pd.DataFrame,
    state: str,
    agg_dir: Path = AGG_DIR,
) -> Path:
    """Save monthly state DataFrame to agg_dir/{STATE}.csv. Returns path."""
    agg_dir.mkdir(parents=True, exist_ok=True)
    path = agg_dir / f"{state.upper()}.csv"
    df.to_csv(path, index=False)
    return path


def load_state_csv(state: str, agg_dir: Path = AGG_DIR) -> pd.DataFrame:
    """Load previously saved monthly state CSV. Returns empty DataFrame if absent."""
    path = agg_dir / f"{state.upper()}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])


# ── Phase 3: State → regional aggregation ─────────────────────────────────────

def build_region_df(
    region_name: str,
    states: list[str],
    agg_dir: Path = AGG_DIR,
) -> pd.DataFrame:
    """Load state CSVs for all states in the region; unweighted mean per month.

    Months where fewer than half the available states have data are dropped.
    Returns DataFrame with same feature columns plus region_state_count.
    """
    state_frames: list[pd.DataFrame] = []
    for state in states:
        df = load_state_csv(state, agg_dir)
        if not df.empty:
            state_frames.append(df)

    if not state_frames:
        log.warning("build_region_df: no state CSVs found for region '%s'", region_name)
        return pd.DataFrame()

    feat_cols = [c for c in state_frames[0].columns
                 if c not in ("date", "day_count", "city_count")]

    tagged: list[pd.DataFrame] = []
    for i, sdf in enumerate(state_frames):
        available = ["date"] + [c for c in feat_cols if c in sdf.columns]
        tmp = sdf[available].copy()
        tmp["_state_idx"] = i
        tagged.append(tmp)
    concat = pd.concat(tagged, ignore_index=True)

    agg_cols = [c for c in feat_cols if c in concat.columns]
    regional = concat.groupby("date")[agg_cols].mean().reset_index()
    state_counts = (
        concat.groupby("date")["_state_idx"].nunique().rename("region_state_count")
    )
    regional = regional.merge(state_counts, on="date", how="left")

    min_states = max(1, len(state_frames) // 2)
    regional = (
        regional[regional["region_state_count"] >= min_states]
        .reset_index(drop=True)
    )
    return regional


def build_national_df(agg_dir: Path = AGG_DIR) -> pd.DataFrame:
    """National monthly DataFrame: unweighted mean across all states in CENSUS_REGIONS."""
    all_states: list[str] = []
    for states in CENSUS_REGIONS.values():
        all_states.extend(states)
    # deduplicate preserving order
    seen: set[str] = set()
    unique_states = [s for s in all_states if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]
    return build_region_df("national", unique_states, agg_dir)


# ── Phase 4: Weather-specific feature engineering ─────────────────────────────

def engineer_weather_features(
    df: pd.DataFrame,
    series_cols: list[str],
) -> pd.DataFrame:
    """Wrap macro_utils.engineer_features() with weather-specific additions.

    1. Call engineer_features(df, series_cols) for lags/rolling/momentum.
    2. Replace integer month/quarter with cyclical sin/cos encoding:
         sin_month = sin(2π × month / 12)
         cos_month = cos(2π × month / 12)
    3. Add cross-climate features when both lag1 components are present:
         hdd_x_precip = hdd_lag1 × precip_total_lag1
         cdd_x_wind   = cdd_lag1 × wind_mean_lag1
    4. Drop integer 'month' and 'quarter' columns (superseded by sin/cos).
    """
    df_feat = engineer_features(df, series_cols)

    if "month" in df_feat.columns:
        df_feat["sin_month"] = np.sin(2 * np.pi * df_feat["month"] / 12)
        df_feat["cos_month"] = np.cos(2 * np.pi * df_feat["month"] / 12)
        df_feat = df_feat.drop(columns=["month"])
    if "quarter" in df_feat.columns:
        df_feat = df_feat.drop(columns=["quarter"])

    if "hdd_lag1" in df_feat.columns and "precip_total_lag1" in df_feat.columns:
        df_feat["hdd_x_precip"] = df_feat["hdd_lag1"] * df_feat["precip_total_lag1"]
    if "cdd_lag1" in df_feat.columns and "wind_mean_lag1" in df_feat.columns:
        df_feat["cdd_x_wind"] = df_feat["cdd_lag1"] * df_feat["wind_mean_lag1"]

    return df_feat


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_clip_ranges(df: pd.DataFrame, series_cols: list[str]) -> dict:
    ranges = {}
    for col in series_cols:
        vals = df[col].dropna()
        if vals.empty:
            continue
        lo = float(np.percentile(vals, 1))
        hi = float(np.percentile(vals, 99))
        margin = abs(hi - lo) * 0.1
        ranges[col] = (lo - margin, hi + margin)
    return ranges


def build_series_info(series_cols: list[str], source_cfg: dict) -> list[tuple]:
    label_map = source_cfg.get("label_map", {})
    units     = source_cfg.get("units", {})
    return [
        (col,
         label_map.get(col, col.replace("_", " ")),
         GROUP_COLORS[i % len(GROUP_COLORS)],
         units.get(col, ""),
         None,
         None)
        for i, col in enumerate(series_cols)
    ]


# ── Phase 5: Model training ────────────────────────────────────────────────────

def run_weather_group(
    source_key: str,
    source_cfg: dict,
    geo_name: str,
    df: pd.DataFrame,
) -> bool:
    """Train LightGBM models for one (source, geography) pair.

    Mirrors industrial_model.run_group() exactly.
    Returns True on success, False if insufficient data.
    """
    group_name   = source_cfg["group_name"]
    model_prefix = source_cfg["model_prefix"]
    plot_prefix  = source_cfg["plot_prefix"].format(geo=geo_name)
    results_file = source_cfg["results_file"].format(geo=geo_name)
    series_cols  = [c for c in source_cfg["series"] if c in df.columns]

    print("=" * 65)
    print(f"{group_name.upper()} [{geo_name.upper()}] — LightGBM Forecast Model")
    print("=" * 65)

    if not series_cols:
        print(f"  SKIP: none of {source_cfg['series']} present in DataFrame.")
        return False

    if len(df) < MIN_ROWS_REQUIRED:
        print(f"  SKIP: only {len(df)} rows — need ≥{MIN_ROWS_REQUIRED}.")
        return False

    # Restrict to date + target series only — prevents extra columns (city_count,
    # region_state_count, other weather variables) from bleeding into feat_cols
    # and confusing joint_recursive_forecast.
    df_model = df[["date"] + series_cols].dropna(subset=series_cols).reset_index(drop=True)

    last_date = df_model["date"].max()
    print(f"\n  Rows: {len(df_model)}  |  "
          f"{df_model['date'].min().strftime('%Y-%m')} → {last_date.strftime('%Y-%m')}")
    for col in series_cols:
        val   = df_model[col].dropna().iloc[-1] if df_model[col].notna().any() else float("nan")
        label = source_cfg["label_map"].get(col, col)
        print(f"  Latest {label}: {val:,.3f}")

    print("\nEngineering features...")
    df_feat   = engineer_weather_features(df_model, series_cols).dropna().reset_index(drop=True)
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
    series_info = build_series_info(series_cols, source_cfg)

    print("\nValidation Metrics (last 24 months):")
    print_validation_metrics(y_val, val_preds, series_info)

    clip_ranges = build_clip_ranges(df_model, series_cols)
    print(f"\nGenerating {FORECAST_HORIZON}-month joint recursive forecast...")
    fc = joint_recursive_forecast(df_model, models, feat_cols, series_cols,
                                   clip_ranges, FORECAST_HORIZON,
                                   feature_engineer=engineer_weather_features)
    print(f"\n{FORECAST_HORIZON}-Month Forecast (from {last_date.strftime('%Y-%m')}):")
    print_forecast_table(fc, series_info)

    for col in series_cols:
        path = OUTPUT_DIR / f"{model_prefix}_{geo_name}_{col}.joblib"
        joblib.dump(models[col]["mid"], str(path))
        print(f"  Saved {path}")

    print("\nGenerating plots...")
    ncols = min(2, len(series_cols))
    plot_forecast_dashboard(
        df_model, fc, series_info, last_date,
        title=f"{group_name} [{geo_name.upper()}] — {FORECAST_HORIZON}-Month Outlook",
        save_path=str(OUTPUT_DIR / f"{plot_prefix}_dashboard.png"),
        ncols=ncols,
    )
    plot_validation_performance(
        dates_val, y_val, val_preds, series_info,
        title=(f"{group_name} [{geo_name.upper()}] — "
               f"Validation (Last {VALIDATION_MONTHS} Months)"),
        save_path=str(OUTPUT_DIR / f"{plot_prefix}_validation.png"),
    )
    plot_feature_importance(
        models, feat_cols, series_info,
        title=f"{group_name} [{geo_name.upper()}] — Feature Importance (Gain)",
        save_path=str(OUTPUT_DIR / f"{plot_prefix}_importance.png"),
    )

    save_model_results(
        group_name, df_model, fc, y_val, val_preds,
        series_info, str(OUTPUT_DIR / results_file),
    )
    print("\nDone.")
    return True


# ── Phase 6: Aggregation pipeline ─────────────────────────────────────────────

def run_aggregation(
    states: list[str] | None = None,
    weather_dir: Path = WEATHER_DIR,
    agg_dir: Path = AGG_DIR,
    force: bool = False,
) -> dict[str, bool]:
    """Aggregate all states (or a subset) from raw daily → monthly state CSVs.

    If states is None, discovers all state directories from weather_dir.
    If force=False, skips states where agg_dir/{STATE}.csv already exists.
    Returns {state: success_bool} for all attempted states.
    """
    if states is None:
        if not weather_dir.exists():
            log.error("run_aggregation: WEATHER_DIR not found: %s", weather_dir)
            return {}
        states = sorted(d.name for d in weather_dir.iterdir() if d.is_dir())

    results: dict[str, bool] = {}
    skipped = ok = failed = 0

    for state in states:
        csv_path = agg_dir / f"{state.upper()}.csv"
        if csv_path.exists() and not force:
            log.debug("run_aggregation: skipping %s — CSV exists", state.upper())
            skipped += 1
            results[state] = True
            continue

        log.info("run_aggregation: aggregating %s ...", state.upper())
        df = aggregate_state_monthly(state, weather_dir)
        if df.empty:
            log.warning("run_aggregation: no data for %s — skipping", state.upper())
            results[state] = False
            failed += 1
            continue

        df = add_temperature_anomaly(df)
        path = save_state_csv(df, state, agg_dir)
        log.info("run_aggregation: saved %s (%d months)", path, len(df))
        results[state] = True
        ok += 1

    log.info(
        "run_aggregation complete — ok=%d  skipped=%d  failed=%d",
        ok, skipped, failed,
    )
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Weather LightGBM climate forecasting model trainer."
    )
    p.add_argument(
        "--geo", nargs="+",
        choices=list(CENSUS_REGIONS.keys()) + ["all"],
        default=["national"],
        help="Geographies to train (default: national).",
    )
    p.add_argument(
        "--source", nargs="+",
        choices=list(SOURCE_CONFIG.keys()) + ["all"],
        default=["all"],
        help="Weather source groups to train (default: all).",
    )
    p.add_argument(
        "--agg-only",
        action="store_true",
        default=False,
        help="Run aggregation step only; skip model training.",
    )
    p.add_argument(
        "--force-agg",
        action="store_true",
        default=False,
        help="Re-aggregate even if state CSVs already exist.",
    )
    p.add_argument(
        "--agg-dir",
        type=Path,
        default=AGG_DIR,
        metavar="PATH",
        help=f"Override aggregated state CSV directory (default: {AGG_DIR}).",
    )
    p.add_argument(
        "--weather-dir",
        type=Path,
        default=WEATHER_DIR,
        metavar="PATH",
        help=f"Override weather database root (default: {WEATHER_DIR}).",
    )
    p.add_argument(
        "--states",
        default=None,
        metavar="XX,YY",
        help="Comma-separated state codes to include in aggregation (default: all).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    agg_states: list[str] | None = None
    if args.states:
        agg_states = [s.strip().upper() for s in args.states.split(",")]

    # ── Step 1: Aggregate raw daily → monthly state CSVs ─────────────────────
    agg_results = run_aggregation(
        states=agg_states,
        weather_dir=args.weather_dir,
        agg_dir=args.agg_dir,
        force=args.force_agg,
    )
    if not any(agg_results.values()):
        log.error("No state CSVs available — cannot train models.")
        return 1

    if args.agg_only:
        log.info("--agg-only: aggregation complete, skipping model training.")
        return 0

    # ── Step 2: Train models for each (geo, source) pair ─────────────────────
    geos    = list(CENSUS_REGIONS.keys()) if "all" in args.geo    else args.geo
    sources = list(SOURCE_CONFIG.keys())  if "all" in args.source else args.source

    any_failed   = False
    summary_rows: list[dict] = []

    for geo_name in geos:
        if geo_name == "national":
            df = build_national_df(args.agg_dir)
        else:
            df = build_region_df(geo_name, CENSUS_REGIONS[geo_name], args.agg_dir)

        if df.empty:
            log.warning("No data for geography '%s' — skipping.", geo_name)
            any_failed = True
            continue

        for source_key in sources:
            cfg = SOURCE_CONFIG.get(source_key)
            if cfg is None:
                log.error("Unknown source: %s", source_key)
                any_failed = True
                continue

            try:
                ok = run_weather_group(source_key, cfg, geo_name, df)
                summary_rows.append({
                    "Geography": geo_name,
                    "Source":    source_key,
                    "Status":    "OK" if ok else "SKIP",
                })
                if not ok:
                    any_failed = True
            except Exception as exc:
                import traceback
                log.error("[%s / %s] ERROR: %s", geo_name, source_key, exc)
                traceback.print_exc()
                summary_rows.append({
                    "Geography": geo_name,
                    "Source":    source_key,
                    "Status":    "FAILED",
                })
                any_failed = True

    if summary_rows:
        try:
            from tabulate import tabulate
            print("\n" + "─" * 50)
            print("  WEATHER MODEL TRAINING SUMMARY")
            print("─" * 50)
            print(tabulate(summary_rows, headers="keys", tablefmt="rounded_outline"))
            print("─" * 50)
        except ImportError:
            for row in summary_rows:
                print(f"  {row['Geography']:12s}  {row['Source']:30s}  {row['Status']}")

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
