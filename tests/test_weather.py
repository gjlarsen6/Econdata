"""
tests/test_weather.py — Unit and integration tests for the weather connector
and refresh script.

Run with:
    python3 -m pytest tests/test_weather.py -v
    python3 -m pytest tests/test_weather.py -v -k "not integration"  # skip live-data tests
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Make project root importable when running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from connectors.weather import (
    WeatherConnector,
    _city_file_stem,
    _parse_date,
    _read_year_file,
    _year_file_path,
    get_weather_for_date,
    WEATHER_DIR,
)
from weather_refresh import fetch_vc_year, refresh_city, _city_file_stem as rs_stem

# ── Fixtures ──────────────────────────────────────────────────────────────────

KNOWN_STATE = "CA"
KNOWN_CITY  = "Bakersfield"
KNOWN_YEAR  = 2000
KNOWN_FILE  = WEATHER_DIR / KNOWN_STATE / KNOWN_CITY / f"{KNOWN_YEAR}_BakersfieldCA.csv"

# Synthetic single-year weather CSV (21-column format matching existing files)
_SYNTH_ROWS = [
    "Synth City ST,01/01/2025,30.0,50.0,40.0,25.0,60.0,,10.0,,180.0,,0.0,0.0,,10.0,20.0,1013.0,34.0,-118.0,Clear",
    "Synth City ST,01/02/2025,32.0,52.0,42.0,26.0,61.0,,11.0,,185.0,,0.0,0.0,,10.0,22.0,1014.0,34.0,-118.0,Partially cloudy",
    "Synth City ST,01/03/2025,28.0,48.0,38.0,24.0,58.0,,9.0,,175.0,,0.1,4.0,,9.0,30.0,1012.0,34.0,-118.0,Rain",
    # Note: Jan 4 (Saturday) deliberately omitted to test carry-forward
    "Synth City ST,01/05/2025,31.0,51.0,41.0,25.5,59.0,,10.5,,182.0,,0.0,0.0,,10.0,18.0,1013.5,34.0,-118.0,Clear",
]
_SYNTH_HEADER = (
    "Address,Date time,Minimum Temperature,Maximum Temperature,Temperature,"
    "Dew Point,Relative Humidity,Heat Index,Wind Speed,Wind Gust,Wind Direction,"
    "Wind Chill,Precipitation,Precipitation Cover,Snow Depth,Visibility,"
    "Cloud Cover,Sea Level Pressure,Latitude,Longitude,Conditions"
)
_SYNTH_CSV = "\n".join([_SYNTH_HEADER] + _SYNTH_ROWS) + "\n"


def _write_synth_file(city_dir: Path, year: int = 2025) -> Path:
    """Write a synthetic weather CSV into tmp_path city directory."""
    city_dir.mkdir(parents=True, exist_ok=True)
    stem = city_dir.name.replace(" ", "") + city_dir.parent.name  # citySTATE
    path = city_dir / f"{year}_{stem}.csv"
    path.write_text(_SYNTH_CSV)
    return path


# ── Path construction ─────────────────────────────────────────────────────────

class TestPathHelpers:
    def test_stem_simple(self):
        assert _city_file_stem("Bakersfield", "CA") == "BakersfieldCA"

    def test_stem_multiword(self):
        assert _city_file_stem("Coeur D Alene", "ID") == "CoeurDAleneID"

    def test_stem_lowercase_state(self):
        assert _city_file_stem("Boise", "id") == "BoiseID"

    def test_year_file_path(self):
        p = _year_file_path("CA", "Bakersfield", 2000)
        assert p == WEATHER_DIR / "CA" / "Bakersfield" / "2000_BakersfieldCA.csv"

    def test_year_file_path_custom_dir(self, tmp_path):
        p = _year_file_path("ID", "Boise", 2025, weather_dir=tmp_path)
        assert p == tmp_path / "ID" / "Boise" / "2025_BoiseID.csv"

    def test_weather_refresh_stem_matches_connector(self):
        # Ensure weather_refresh._city_file_stem and connectors.weather._city_file_stem
        # are consistent (they import from the same source)
        assert rs_stem("Bakersfield", "CA") == _city_file_stem("Bakersfield", "CA")


# ── Date parsing ──────────────────────────────────────────────────────────────

class TestParsDate:
    def test_iso_format(self):
        from datetime import datetime
        assert _parse_date("2000-01-15") == datetime(2000, 1, 15)

    def test_mm_dd_yyyy_format(self):
        from datetime import datetime
        assert _parse_date("01/15/2000") == datetime(2000, 1, 15)

    def test_datetime_passthrough(self):
        from datetime import datetime
        dt = datetime(2000, 6, 1)
        assert _parse_date(dt) is dt

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            _parse_date("15-01-2000")


# ── _read_year_file ───────────────────────────────────────────────────────────

class TestReadYearFile:
    def test_reads_expected_columns(self, tmp_path):
        path = tmp_path / "2025_SynthST.csv"
        path.write_text(_SYNTH_CSV)
        df = _read_year_file(path)
        assert set(df.columns) == {
            "date", "weather_max_temp", "weather_min_temp",
            "weather_precip", "weather_wind_speed", "weather_conditions",
        }

    def test_sorted_by_date(self, tmp_path):
        path = tmp_path / "2025_SynthST.csv"
        path.write_text(_SYNTH_CSV)
        df = _read_year_file(path)
        assert list(df["date"]) == sorted(df["date"])

    def test_correct_values(self, tmp_path):
        path = tmp_path / "2025_SynthST.csv"
        path.write_text(_SYNTH_CSV)
        df = _read_year_file(path)
        row = df.iloc[0]
        assert row["weather_max_temp"] == pytest.approx(50.0)
        assert row["weather_min_temp"] == pytest.approx(30.0)
        assert row["weather_conditions"] == "Clear"

    def test_handles_25_column_file(self, tmp_path):
        # Add the 4 extra columns present in newer VC API responses
        header_25 = _SYNTH_HEADER + ",Weather Type,Resolved Address,Name,Info"
        rows_25 = [r + ",,Bakersfield CA,," for r in _SYNTH_ROWS]
        csv_25 = "\n".join([header_25] + rows_25) + "\n"
        path = tmp_path / "2025_SynthST.csv"
        path.write_text(csv_25)
        df = _read_year_file(path)
        # Should still return only the 5 standard columns + date
        assert "Weather Type" not in df.columns
        assert len(df) == len(_SYNTH_ROWS)


# ── get_weather_for_date ──────────────────────────────────────────────────────

class TestGetWeatherForDate:
    def test_returns_five_keys(self, tmp_path):
        city_dir = tmp_path / "ST" / "SynthCity"
        _write_synth_file(city_dir)
        result = get_weather_for_date("ST", "SynthCity", "2025-01-01",
                                      weather_dir=tmp_path)
        assert set(result.keys()) == {
            "weather_max_temp", "weather_min_temp", "weather_precip",
            "weather_wind_speed", "weather_conditions",
        }

    def test_correct_values_on_exact_match(self, tmp_path):
        city_dir = tmp_path / "ST" / "SynthCity"
        _write_synth_file(city_dir)
        result = get_weather_for_date("ST", "SynthCity", "2025-01-03",
                                      weather_dir=tmp_path)
        assert result["weather_max_temp"] == pytest.approx(48.0)
        assert result["weather_conditions"] == "Rain"

    def test_accepts_mm_dd_yyyy(self, tmp_path):
        city_dir = tmp_path / "ST" / "SynthCity"
        _write_synth_file(city_dir)
        result = get_weather_for_date("ST", "SynthCity", "01/01/2025",
                                      weather_dir=tmp_path)
        assert result["weather_max_temp"] == pytest.approx(50.0)

    def test_carry_forward_for_missing_date(self, tmp_path):
        # Jan 4 (Saturday) is not in the synthetic file — should carry Jan 3
        city_dir = tmp_path / "ST" / "SynthCity"
        _write_synth_file(city_dir)
        jan3 = get_weather_for_date("ST", "SynthCity", "2025-01-03", weather_dir=tmp_path)
        jan4 = get_weather_for_date("ST", "SynthCity", "2025-01-04", weather_dir=tmp_path)
        assert jan4["weather_max_temp"] == jan3["weather_max_temp"]
        assert jan4["weather_conditions"] == jan3["weather_conditions"]

    def test_missing_city_returns_empty(self, tmp_path):
        result = get_weather_for_date("XX", "NoSuchCity", "2025-01-01",
                                      weather_dir=tmp_path)
        assert result == {}

    def test_missing_year_returns_empty(self, tmp_path):
        city_dir = tmp_path / "ST" / "SynthCity"
        _write_synth_file(city_dir, year=2025)
        # Request 2024 — file doesn't exist
        result = get_weather_for_date("ST", "SynthCity", "2024-07-04",
                                      weather_dir=tmp_path)
        assert result == {}

    @pytest.mark.integration
    def test_existing_file(self):
        """Integration: reads a real 2000_BakersfieldCA.csv file."""
        if not KNOWN_FILE.exists():
            pytest.skip(f"Live data file not found: {KNOWN_FILE}")
        result = get_weather_for_date(KNOWN_STATE, KNOWN_CITY, "2000-01-01")
        assert "weather_max_temp" in result
        assert result["weather_conditions"] is not None
        assert isinstance(result["weather_max_temp"], float)


# ── WeatherConnector ──────────────────────────────────────────────────────────

class TestWeatherConnector:
    def _make_connector(self, tmp_path: Path, years: list[int] = [2025]) -> WeatherConnector:
        city_dir = tmp_path / "ST" / "SynthCity"
        for year in years:
            _write_synth_file(city_dir, year=year)
        return WeatherConnector("ST", "SynthCity", weather_dir=tmp_path)

    def test_load_returns_expected_columns(self, tmp_path):
        wc = self._make_connector(tmp_path)
        df = wc.load()
        assert set(df.columns) == {
            "date", "weather_max_temp", "weather_min_temp",
            "weather_precip", "weather_wind_speed", "weather_conditions",
        }

    def test_load_missing_city_returns_empty(self, tmp_path):
        wc = WeatherConnector("ZZ", "NoCity", weather_dir=tmp_path)
        df = wc.load()
        assert df.empty

    def test_load_concatenates_multiple_years(self, tmp_path):
        wc = self._make_connector(tmp_path, years=[2024, 2025])
        df = wc.load()
        years_in_data = df["date"].dt.year.unique()
        assert 2025 in years_in_data
        # Both years present; duplicates removed
        assert df["date"].nunique() == len(df)

    def test_enrich_adds_columns(self, tmp_path):
        wc = self._make_connector(tmp_path)
        events = pd.DataFrame({
            "date":  ["2025-01-01", "2025-01-02", "2025-01-03"],
            "value": [10, 20, 30],
        })
        out = wc.enrich(events)
        assert "weather_max_temp"   in out.columns
        assert "weather_conditions" in out.columns
        assert len(out) == len(events)

    def test_enrich_carry_forward_fills_gaps(self, tmp_path):
        wc = self._make_connector(tmp_path)
        # Jan 4 is missing from synthetic data
        events = pd.DataFrame({
            "date":  ["2025-01-03", "2025-01-04", "2025-01-05"],
            "value": [1, 2, 3],
        })
        out = wc.enrich(events)
        # Jan 4 should carry Jan 3's conditions
        jan3_cond = out.loc[out["date"] == "2025-01-03", "weather_conditions"].iloc[0]
        jan4_cond = out.loc[out["date"] == "2025-01-04", "weather_conditions"].iloc[0]
        assert jan4_cond == jan3_cond

    def test_enrich_no_nans_after_ffill(self, tmp_path):
        wc = self._make_connector(tmp_path)
        events = pd.DataFrame({
            "date":  ["2025-01-01", "2025-01-02", "2025-01-03",
                      "2025-01-04", "2025-01-05"],
            "value": range(5),
        })
        out = wc.enrich(events)
        # All rows on/after 2025-01-01 should have weather data (no NaN after ffill)
        assert out["weather_max_temp"].notna().all()

    @pytest.mark.integration
    def test_integration_load(self):
        """Integration: loads real Bakersfield data."""
        if not KNOWN_FILE.exists():
            pytest.skip(f"Live data file not found: {KNOWN_FILE}")
        wc = WeatherConnector(KNOWN_STATE, KNOWN_CITY)
        df = wc.load()
        assert len(df) > 300
        assert df["date"].dt.year.min() <= KNOWN_YEAR


# ── weather_refresh fetch and write ──────────────────────────────────────────

class TestFetchVcYear:
    def test_request_parameters(self):
        """fetch_vc_year sends correct params to the VC API."""
        captured: dict = {}

        class MockResponse:
            status_code = 200
            text = _SYNTH_CSV

        def mock_get(url, params=None, timeout=None):
            captured["url"]    = url
            captured["params"] = params
            return MockResponse()

        with patch("weather_refresh.requests.get", side_effect=mock_get):
            df = fetch_vc_year("CA", "Bakersfield", 2025, "test_key")

        assert df is not None
        p = captured["params"]
        assert p["aggregateHours"]  == 24
        assert p["unitGroup"]       == "us"
        assert p["contentType"]     == "csv"
        assert p["locationMode"]    == "single"
        assert p["locations"]       == "Bakersfield,CA"
        assert p["key"]             == "test_key"
        assert "2025-01-01" in p["startDateTime"]
        assert "2025-12-31" in p["endDateTime"]

    def test_returns_none_on_http_error(self):
        class BadResponse:
            status_code = 403
            text = "Forbidden"

        with patch("weather_refresh.requests.get", return_value=BadResponse()):
            result = fetch_vc_year("CA", "Bakersfield", 2025, "bad_key")
        assert result is None

    def test_returns_none_on_network_error(self):
        with patch("weather_refresh.requests.get",
                   side_effect=Exception("connection refused")):
            result = fetch_vc_year("CA", "Bakersfield", 2025, "key")
        assert result is None

    def test_partial_year_end_date(self):
        """Current year end date should not exceed today."""
        from datetime import datetime as dt_cls
        captured: dict = {}

        class MockResponse:
            status_code = 200
            text = _SYNTH_CSV

        def mock_get(url, params=None, timeout=None):
            captured["params"] = params
            return MockResponse()

        today = dt_cls.now().date()
        with patch("weather_refresh.requests.get", side_effect=mock_get):
            fetch_vc_year("CA", "Bakersfield", today.year, "key")

        end_str = captured["params"]["endDateTime"][:10]   # "YYYY-MM-DD"
        assert end_str <= today.strftime("%Y-%m-%d")


class TestRefreshCity:
    def test_skips_existing_file(self, tmp_path):
        # Pre-create the file
        path = _year_file_path("CA", "Bakersfield", 2025, weather_dir=tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("dummy")

        results = refresh_city("CA", "Bakersfield", "key",
                               from_year=2025, to_year=2025,
                               overwrite=False, weather_dir=tmp_path)
        assert results[0]["status"] == "skipped"

    def test_overwrites_when_flag_set(self, tmp_path):
        path = _year_file_path("CA", "Bakersfield", 2025, weather_dir=tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("old content")

        def mock_fetch(state, city, year, api_key):
            return pd.read_csv(__import__("io").StringIO(_SYNTH_CSV))

        with patch("weather_refresh.fetch_vc_year", side_effect=mock_fetch), \
             patch("weather_refresh.time.sleep"):
            results = refresh_city("CA", "Bakersfield", "key",
                                   from_year=2025, to_year=2025,
                                   overwrite=True, weather_dir=tmp_path)

        assert results[0]["status"] == "ok"
        assert path.read_text() != "old content"

    def test_writes_file_to_correct_path(self, tmp_path):
        def mock_fetch(state, city, year, api_key):
            return pd.read_csv(__import__("io").StringIO(_SYNTH_CSV))

        with patch("weather_refresh.fetch_vc_year", side_effect=mock_fetch), \
             patch("weather_refresh.time.sleep"):
            results = refresh_city("CA", "Bakersfield", "key",
                                   from_year=2025, to_year=2025,
                                   weather_dir=tmp_path)

        assert results[0]["status"] == "ok"
        assert results[0]["rows"] > 0
        out_path = _year_file_path("CA", "Bakersfield", 2025, weather_dir=tmp_path)
        assert out_path.exists()

    def test_records_error_on_fetch_failure(self, tmp_path):
        with patch("weather_refresh.fetch_vc_year", return_value=None), \
             patch("weather_refresh.time.sleep"):
            results = refresh_city("CA", "Bakersfield", "key",
                                   from_year=2025, to_year=2025,
                                   weather_dir=tmp_path)

        assert results[0]["status"] == "error"

    def test_multi_year_range(self, tmp_path):
        def mock_fetch(state, city, year, api_key):
            return pd.read_csv(__import__("io").StringIO(_SYNTH_CSV))

        with patch("weather_refresh.fetch_vc_year", side_effect=mock_fetch), \
             patch("weather_refresh.time.sleep"):
            results = refresh_city("CA", "Bakersfield", "key",
                                   from_year=2023, to_year=2025,
                                   weather_dir=tmp_path)

        assert len(results) == 3
        assert all(r["status"] == "ok" for r in results)
        for year in [2023, 2024, 2025]:
            assert _year_file_path("CA", "Bakersfield", year, weather_dir=tmp_path).exists()

    def test_written_file_is_readable_by_connector(self, tmp_path):
        """Round-trip: refresh_city writes a file that WeatherConnector can load."""
        def mock_fetch(state, city, year, api_key):
            return pd.read_csv(__import__("io").StringIO(_SYNTH_CSV))

        with patch("weather_refresh.fetch_vc_year", side_effect=mock_fetch), \
             patch("weather_refresh.time.sleep"):
            refresh_city("ST", "SynthCity", "key",
                         from_year=2025, to_year=2025, weather_dir=tmp_path)

        wc = WeatherConnector("ST", "SynthCity", weather_dir=tmp_path)
        df = wc.load()
        assert len(df) > 0
        assert "weather_max_temp" in df.columns
