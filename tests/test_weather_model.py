"""
tests/test_weather_model.py — Unit tests for weather_model.py.

Run with:
    python3 -m pytest tests/test_weather_model.py -v
    python3 -m pytest tests/test_weather_model.py -v -k "not integration"
    python3 -m pytest tests/test_weather_model.py -v -m integration
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from weather_model import (
    SOURCE_CONFIG,
    aggregate_city_to_monthly,
    add_temperature_anomaly,
    build_region_df,
    engineer_weather_features,
    run_weather_group,
    save_state_csv,
    load_state_csv,
    run_aggregation,
    build_national_df,
)


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _make_daily(
    n_days: int = 365,
    year: int = 2010,
    mean_temp: float = 55.0,
    max_temp: float = 65.0,
    min_temp: float = 45.0,
    precip: float = 0.05,
    snow_depth: float = 0.0,
    wind_speed: float = 8.0,
    cloud_cover: float = 40.0,
) -> pd.DataFrame:
    """Synthetic daily DataFrame for one city-year with constant values."""
    dates = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")
    return pd.DataFrame({
        "date":        dates,
        "max_temp":    max_temp,
        "min_temp":    min_temp,
        "mean_temp":   mean_temp,
        "precip":      precip,
        "snow_depth":  snow_depth,
        "wind_speed":  wind_speed,
        "cloud_cover": cloud_cover,
    })


def _make_state_monthly(
    n_months: int = 120,
    start: str = "2000-01-01",
    temp_mean: float = 55.0,
    city_count: int = 3,
) -> pd.DataFrame:
    """Synthetic monthly state-level DataFrame with seasonal variation.

    Uses a sine-wave seasonal cycle so that year-over-year % change features
    (computed by macro_utils.engineer_features) are never NaN from 0/0 division.
    """
    dates = pd.date_range(start, periods=n_months, freq="MS")
    months = np.arange(n_months)
    seasonal = 12.0 * np.sin(2 * np.pi * months / 12)  # ±12°F swing
    return pd.DataFrame({
        "date":                dates,
        "temp_mean":           temp_mean + seasonal,
        "hdd":                 np.maximum(1.0, 200.0 - seasonal * 10),
        "cdd":                 np.maximum(1.0, 50.0 + seasonal * 8),
        "temp_anom":           seasonal * 0.3,
        "precip_total":        np.maximum(0.5, 3.0 + seasonal * 0.1),
        "precip_days":         np.maximum(1.0, 10.0 + seasonal * 0.5),
        "snow_total":          np.maximum(0.0, -seasonal * 0.5),
        "extreme_heat_days":   np.maximum(0.0, seasonal * 0.3),
        "extreme_cold_days":   np.maximum(0.0, -seasonal * 0.4 + 3.0),
        "extreme_precip_days": np.maximum(0.0, 0.5 + seasonal * 0.05),
        "wind_mean":           np.maximum(1.0, 8.0 - seasonal * 0.2),
        "cloud_cover_mean":    np.maximum(5.0, 40.0 + seasonal * 2.0),
        "city_count":          float(city_count),
    })


# ── TestAggregateCityToMonthly ────────────────────────────────────────────────

class TestAggregateCityToMonthly:

    def test_hdd_computation(self):
        # All days at 55°F: HDD per day = max(0, 65-55) = 10; Jan has 31 days → 310
        daily = _make_daily(365, year=2010, mean_temp=55.0)
        monthly = aggregate_city_to_monthly(daily)
        jan = monthly[monthly["date"].dt.month == 1].iloc[0]
        assert abs(jan["hdd"] - 31 * 10.0) < 1.0

    def test_cdd_computation(self):
        # All days at 75°F: CDD per day = max(0, 75-65) = 10; Jul has 31 days → 310
        daily = _make_daily(365, year=2010, mean_temp=75.0)
        monthly = aggregate_city_to_monthly(daily)
        jul = monthly[monthly["date"].dt.month == 7].iloc[0]
        assert abs(jul["cdd"] - 31 * 10.0) < 1.0

    def test_hdd_zero_when_hot(self):
        # At 80°F there should be no heating demand
        daily = _make_daily(365, year=2010, mean_temp=80.0)
        monthly = aggregate_city_to_monthly(daily)
        assert (monthly["hdd"] == 0).all()

    def test_cdd_zero_when_cold(self):
        # At 50°F there should be no cooling demand
        daily = _make_daily(365, year=2010, mean_temp=50.0)
        monthly = aggregate_city_to_monthly(daily)
        assert (monthly["cdd"] == 0).all()

    def test_extreme_heat_days(self):
        # All days in a month above 90°F
        daily = _make_daily(31, year=2010, max_temp=95.0)
        monthly = aggregate_city_to_monthly(daily)
        assert monthly.iloc[0]["extreme_heat_days"] == 31

    def test_extreme_cold_days(self):
        # All January days below 32°F
        daily = _make_daily(31, year=2010, min_temp=20.0)
        monthly = aggregate_city_to_monthly(daily)
        assert monthly.iloc[0]["extreme_cold_days"] == 31

    def test_extreme_precip_days(self):
        # 5 days with precip > 1.0, rest with 0.0
        dates = pd.date_range("2010-01-01", periods=31, freq="D")
        precip_vals = [2.0] * 5 + [0.0] * 26
        daily = pd.DataFrame({
            "date":        dates,
            "max_temp":    55.0, "min_temp": 45.0, "mean_temp": 50.0,
            "precip":      precip_vals,
            "snow_depth":  0.0, "wind_speed": 8.0, "cloud_cover": 40.0,
        })
        monthly = aggregate_city_to_monthly(daily)
        assert monthly.iloc[0]["extreme_precip_days"] == 5

    def test_partial_month_filtered(self):
        # 10 days < MIN_DAY_COUNT=20 → month is dropped
        daily = _make_daily(10, year=2010)
        monthly = aggregate_city_to_monthly(daily)
        assert len(monthly) == 0

    def test_output_columns_complete(self):
        daily = _make_daily(365, year=2010)
        monthly = aggregate_city_to_monthly(daily)
        expected = {
            "date", "temp_mean", "hdd", "cdd",
            "precip_total", "precip_days", "snow_total",
            "extreme_heat_days", "extreme_cold_days", "extreme_precip_days",
            "wind_mean", "cloud_cover_mean", "day_count",
        }
        assert expected.issubset(set(monthly.columns))

    def test_date_is_month_start(self):
        daily = _make_daily(365, year=2010)
        monthly = aggregate_city_to_monthly(daily)
        assert (monthly["date"].dt.day == 1).all()

    def test_empty_input_returns_empty(self):
        monthly = aggregate_city_to_monthly(pd.DataFrame())
        assert len(monthly) == 0


# ── TestAddTemperatureAnomaly ─────────────────────────────────────────────────

class TestAddTemperatureAnomaly:

    def _baseline_df(
        self,
        temp: float = 55.0,
        years_start: int = 2000,
        n_years: int = 20,
    ) -> pd.DataFrame:
        dates = pd.date_range(f"{years_start}-01-01", periods=n_years * 12, freq="MS")
        return pd.DataFrame({"date": dates, "temp_mean": temp})

    def test_baseline_mean_correct(self):
        df = self._baseline_df(temp=55.0)
        result = add_temperature_anomaly(df)
        baseline_rows = result[result["date"].dt.year.between(2000, 2019)]
        assert baseline_rows["temp_anom"].abs().max() < 0.1

    def test_anomaly_near_zero_in_baseline(self):
        df = self._baseline_df(temp=55.0)
        result = add_temperature_anomaly(df)
        baseline_rows = result[result["date"].dt.year.between(2000, 2019)]
        assert not baseline_rows["temp_anom"].isna().all()
        assert baseline_rows["temp_anom"].abs().max() < 0.01

    def test_insufficient_baseline_gives_nan(self):
        # Only 5 years of baseline data → all temp_anom should be NaN
        df = self._baseline_df(temp=55.0, years_start=2015, n_years=5)
        result = add_temperature_anomaly(df)
        assert result["temp_anom"].isna().all()

    def test_post_baseline_anomaly_nonzero(self):
        # Baseline at 55°F; future year at 65°F → anomaly ≈ +10
        baseline = self._baseline_df(temp=55.0, years_start=2000, n_years=20)
        future = pd.DataFrame({
            "date":      pd.date_range("2023-01-01", periods=12, freq="MS"),
            "temp_mean": 65.0,
        })
        df = pd.concat([baseline, future], ignore_index=True)
        result = add_temperature_anomaly(df)
        future_rows = result[result["date"].dt.year == 2023]
        assert (future_rows["temp_anom"].dropna() > 5.0).all()

    def test_empty_df_passthrough(self):
        result = add_temperature_anomaly(pd.DataFrame())
        assert result.empty

    def test_missing_temp_mean_passthrough(self):
        df = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=12, freq="MS")})
        result = add_temperature_anomaly(df)
        assert "temp_anom" not in result.columns


# ── TestBuildRegionDf ─────────────────────────────────────────────────────────

class TestBuildRegionDf:

    def test_state_mean_aggregation(self, tmp_path):
        state_a = _make_state_monthly(n_months=60, temp_mean=60.0)
        state_b = _make_state_monthly(n_months=60, temp_mean=40.0)
        save_state_csv(state_a, "AA", tmp_path)
        save_state_csv(state_b, "BB", tmp_path)

        result = build_region_df("test", ["AA", "BB"], tmp_path)
        assert not result.empty
        assert abs(result["temp_mean"].mean() - 50.0) < 0.5

    def test_missing_state_csv_skipped(self, tmp_path):
        state_a = _make_state_monthly(n_months=60)
        save_state_csv(state_a, "AA", tmp_path)
        # "BB" CSV does not exist — should be gracefully skipped

        result = build_region_df("test", ["AA", "BB"], tmp_path)
        assert not result.empty
        assert (result["region_state_count"] == 1).all()

    def test_all_missing_returns_empty(self, tmp_path):
        result = build_region_df("test", ["ZZ", "YY"], tmp_path)
        assert result.empty

    def test_region_state_count_column_present(self, tmp_path):
        state_a = _make_state_monthly(n_months=60)
        save_state_csv(state_a, "AA", tmp_path)
        result = build_region_df("test", ["AA"], tmp_path)
        assert "region_state_count" in result.columns

    def test_majority_missing_min_threshold(self, tmp_path):
        # 3 states declared, only 1 CSV exists: min_states = max(1, 3//2) = 1
        # So 1 contributing state is enough to keep the month
        state_a = _make_state_monthly(n_months=12)
        save_state_csv(state_a, "AA", tmp_path)
        result = build_region_df("test", ["AA", "BB", "CC"], tmp_path)
        assert not result.empty


# ── TestEngineerWeatherFeatures ───────────────────────────────────────────────

class TestEngineerWeatherFeatures:

    def _df_and_cols(self) -> tuple[pd.DataFrame, list[str]]:
        df = _make_state_monthly(n_months=120)
        series_cols = ["temp_mean", "hdd", "cdd", "precip_total", "wind_mean"]
        return df, series_cols

    def test_sin_cos_month_added(self):
        df, cols = self._df_and_cols()
        result = engineer_weather_features(df, cols)
        assert "sin_month" in result.columns
        assert "cos_month" in result.columns

    def test_raw_month_column_dropped(self):
        df, cols = self._df_and_cols()
        result = engineer_weather_features(df, cols)
        assert "month" not in result.columns

    def test_raw_quarter_column_dropped(self):
        df, cols = self._df_and_cols()
        result = engineer_weather_features(df, cols)
        assert "quarter" not in result.columns

    def test_cross_feature_hdd_x_precip(self):
        df, cols = self._df_and_cols()
        result = engineer_weather_features(df, cols)
        assert "hdd_x_precip" in result.columns

    def test_cross_feature_cdd_x_wind(self):
        df, cols = self._df_and_cols()
        result = engineer_weather_features(df, cols)
        assert "cdd_x_wind" in result.columns

    def test_standard_lag1_present(self):
        df, cols = self._df_and_cols()
        result = engineer_weather_features(df, cols)
        assert "temp_mean_lag1" in result.columns
        assert "hdd_lag1" in result.columns

    def test_sin_cos_values_in_range(self):
        df, cols = self._df_and_cols()
        result = engineer_weather_features(df, cols).dropna()
        assert result["sin_month"].between(-1.0, 1.0).all()
        assert result["cos_month"].between(-1.0, 1.0).all()


# ── TestSourceConfig ──────────────────────────────────────────────────────────

class TestSourceConfig:

    def test_all_keys_have_required_fields(self):
        required = {
            "series", "label_map", "units", "group_name",
            "model_prefix", "results_file", "plot_prefix",
        }
        for key, cfg in SOURCE_CONFIG.items():
            missing = required - set(cfg.keys())
            assert not missing, f"{key} missing fields: {missing}"

    def test_series_ids_in_label_map(self):
        for key, cfg in SOURCE_CONFIG.items():
            for series in cfg["series"]:
                assert series in cfg["label_map"], \
                    f"{key}: '{series}' missing from label_map"

    def test_series_ids_in_units(self):
        for key, cfg in SOURCE_CONFIG.items():
            for series in cfg["series"]:
                assert series in cfg["units"], \
                    f"{key}: '{series}' missing from units"

    def test_results_file_has_geo_placeholder(self):
        for key, cfg in SOURCE_CONFIG.items():
            assert "{geo}" in cfg["results_file"], \
                f"{key}: results_file missing {{geo}} placeholder"

    def test_plot_prefix_has_geo_placeholder(self):
        for key, cfg in SOURCE_CONFIG.items():
            assert "{geo}" in cfg["plot_prefix"], \
                f"{key}: plot_prefix missing {{geo}} placeholder"


# ── TestRunWeatherGroup ───────────────────────────────────────────────────────

class TestRunWeatherGroup:

    def test_insufficient_data_returns_false(self):
        df = _make_state_monthly(n_months=10)  # < MIN_ROWS_REQUIRED
        ok = run_weather_group(
            "temperature_energy",
            SOURCE_CONFIG["temperature_energy"],
            "national",
            df,
        )
        assert ok is False

    def test_empty_df_returns_false(self):
        ok = run_weather_group(
            "temperature_energy",
            SOURCE_CONFIG["temperature_energy"],
            "national",
            pd.DataFrame(),
        )
        assert ok is False

    def test_produces_results_json(self, tmp_path, monkeypatch):
        import weather_model
        monkeypatch.setattr(weather_model, "OUTPUT_DIR", tmp_path)

        df = _make_state_monthly(n_months=120)
        df = add_temperature_anomaly(df)
        ok = run_weather_group(
            "temperature_energy",
            SOURCE_CONFIG["temperature_energy"],
            "national",
            df,
        )
        assert ok is True
        assert (tmp_path / "results_weather_temperature_national.json").exists()

    def test_results_json_schema(self, tmp_path, monkeypatch):
        import json
        import weather_model
        monkeypatch.setattr(weather_model, "OUTPUT_DIR", tmp_path)

        df = _make_state_monthly(n_months=120)
        df = add_temperature_anomaly(df)
        run_weather_group(
            "temperature_energy",
            SOURCE_CONFIG["temperature_energy"],
            "national",
            df,
        )
        data = json.loads(
            (tmp_path / "results_weather_temperature_national.json").read_text()
        )
        assert "group" in data
        assert "series" in data
        assert len(data["series"]) > 0
        first = data["series"][0]
        assert "forecast" in first
        assert len(first["forecast"]) == 12


# ── TestSaveLoadStateCsv ──────────────────────────────────────────────────────

class TestSaveLoadStateCsv:

    def test_roundtrip(self, tmp_path):
        df = _make_state_monthly(n_months=24)
        save_state_csv(df, "CA", tmp_path)
        loaded = load_state_csv("CA", tmp_path)
        assert len(loaded) == 24
        assert "temp_mean" in loaded.columns

    def test_load_missing_returns_empty(self, tmp_path):
        loaded = load_state_csv("ZZ", tmp_path)
        assert loaded.empty

    def test_state_code_uppercased(self, tmp_path):
        df = _make_state_monthly(n_months=12)
        save_state_csv(df, "ca", tmp_path)  # lowercase input
        loaded = load_state_csv("CA", tmp_path)
        assert not loaded.empty


# ── Integration tests (require live weather files) ────────────────────────────

@pytest.mark.integration
class TestIntegration:

    def test_aggregate_ca_state(self):
        from weather_model import WEATHER_DIR
        if not (WEATHER_DIR / "CA").exists():
            pytest.skip("CA weather directory not found")
        from weather_model import aggregate_state_monthly
        df = aggregate_state_monthly("CA")
        assert not df.empty
        assert "temp_mean" in df.columns
        assert len(df) >= 12

    def test_add_anomaly_ca(self):
        from weather_model import WEATHER_DIR
        if not (WEATHER_DIR / "CA").exists():
            pytest.skip("CA weather directory not found")
        from weather_model import aggregate_state_monthly
        df = aggregate_state_monthly("CA")
        result = add_temperature_anomaly(df)
        assert "temp_anom" in result.columns
        # 2000-2019 anomalies should average near zero
        baseline = result[result["date"].dt.year.between(2000, 2019)]
        if not baseline.empty:
            assert baseline["temp_anom"].dropna().abs().mean() < 5.0

    def test_national_model_runs(self, tmp_path):
        from weather_model import WEATHER_DIR
        if not WEATHER_DIR.exists():
            pytest.skip("Weather directory not found")
        # Aggregate a small subset then attempt model training
        agg_dir = tmp_path / "state"
        run_aggregation(states=["CA"], agg_dir=agg_dir)
        df = build_national_df(agg_dir)
        if df.empty:
            pytest.skip("No regional data after aggregating CA")
        # Validates the pipeline assembles regional data without exceptions
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
