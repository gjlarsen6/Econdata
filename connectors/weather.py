"""
weather.py — Weather enrichment connector.

Ported and modernised from:
  olddatapipeline/orig_ClickAI_Connect_2c-OFA_WeatherTOSales.py
    — findWeatherEvent() column selection + date matching

Reads the local weather database at:
  /gjlarsen/Documents/ClickAI/data/Weather/US_orig/{STATE}/{CityName}/YYYY_{City}{STATE}.csv

Adds five columns to an events DataFrame:
  weather_max_temp     — Maximum Temperature (°F)
  weather_min_temp     — Minimum Temperature (°F)
  weather_precip       — Precipitation (inches)
  weather_wind_speed   — Wind Speed (mph)
  weather_conditions   — Conditions text (e.g. "Partially cloudy")

Missing dates (days between VC observations) are filled forward from the
most recent prior date — fixing the silent empty-cell bug in findWeatherEvent().
"""

from __future__ import annotations

import bisect
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from .base_connector import BaseConnector

log = logging.getLogger(__name__)

WEATHER_DIR = Path("/Users/gjlarsen/Documents/ClickAI/data/Weather/US_orig")

# Columns extracted from VC files and their output names
_COL_MAP: dict[str, str] = {
    "Maximum Temperature": "weather_max_temp",
    "Minimum Temperature": "weather_min_temp",
    "Precipitation":       "weather_precip",
    "Wind Speed":          "weather_wind_speed",
    "Conditions":          "weather_conditions",
}


# ── Path helpers (shared with weather_refresh.py) ─────────────────────────────

def _city_file_stem(city: str, state: str) -> str:
    """Strip spaces from city name and append uppercase state code.

    "Coeur D Alene", "ID"  →  "CoeurDAleneID"
    "Bakersfield",   "CA"  →  "BakersfieldCA"
    """
    return city.replace(" ", "") + state.upper()


def _year_file_path(
    state: str, city: str, year: int, weather_dir: Path = WEATHER_DIR
) -> Path:
    """Return the expected path for a single year's weather file."""
    return (
        weather_dir
        / state.upper()
        / city
        / f"{year}_{_city_file_stem(city, state)}.csv"
    )


# ── Date parsing ──────────────────────────────────────────────────────────────

def _parse_date(date: str | datetime) -> datetime:
    """Accept "YYYY-MM-DD", "MM/DD/YYYY", or datetime; return datetime."""
    if isinstance(date, datetime):
        return date
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(date, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unrecognised date format: {date!r}")


# ── Single-date lookup ────────────────────────────────────────────────────────

def get_weather_for_date(
    state: str,
    city: str,
    date: str | datetime,
    weather_dir: Path = WEATHER_DIR,
) -> dict:
    """Return weather signals for a single date.

    Parameters
    ----------
    state : str
        2-letter postal code, e.g. ``"CA"``.
    city : str
        City name exactly as it appears in the directory, e.g. ``"Bakersfield"``.
    date : str | datetime
        Target date.  Accepts ``"YYYY-MM-DD"``, ``"MM/DD/YYYY"``, or datetime.
    weather_dir : Path
        Root of the weather database.

    Returns
    -------
    dict
        Keys: ``weather_max_temp``, ``weather_min_temp``, ``weather_precip``,
        ``weather_wind_speed``, ``weather_conditions``.
        Returns ``{}`` if the year file does not exist.
        If the exact date is absent, the most recent prior date is used
        (carry-forward).
    """
    dt = _parse_date(date)
    path = _year_file_path(state, city, dt.year, weather_dir)
    if not path.exists():
        log.debug("weather: no file for %s/%s/%d — %s", state, city, dt.year, path)
        return {}

    try:
        df = _read_year_file(path)
    except Exception as exc:
        log.warning("weather: failed to read %s — %s", path, exc)
        return {}

    if df.empty:
        return {}

    target = pd.Timestamp(dt).normalize()
    dates_sorted = df["date"].tolist()   # already sorted by _read_year_file

    idx = bisect.bisect_right(dates_sorted, target) - 1
    if idx < 0:
        return {}

    row = df.iloc[idx]
    return {
        "weather_max_temp":   row.get("weather_max_temp"),
        "weather_min_temp":   row.get("weather_min_temp"),
        "weather_precip":     row.get("weather_precip"),
        "weather_wind_speed": row.get("weather_wind_speed"),
        "weather_conditions": row.get("weather_conditions"),
    }


# ── Internal file reader ──────────────────────────────────────────────────────

def _read_year_file(path: Path) -> pd.DataFrame:
    """Read one year CSV, select and rename the 5 target columns, sort by date."""
    df = pd.read_csv(path, low_memory=False)

    # Select only columns that actually exist in this file (handles 21 vs 25-col files)
    available = {src: dst for src, dst in _COL_MAP.items() if src in df.columns}
    if not available:
        log.warning("weather: no recognised columns in %s", path)
        return pd.DataFrame()

    df = df[["Date time", *available.keys()]].copy()
    df.rename(columns=available, inplace=True)
    df["date"] = pd.to_datetime(df["Date time"], format="%m/%d/%Y", errors="coerce").dt.normalize()
    df = df.drop(columns=["Date time"]).dropna(subset=["date"]).sort_values("date")
    df = df.reset_index(drop=True)
    return df


# ── WeatherConnector ──────────────────────────────────────────────────────────

class WeatherConnector(BaseConnector):
    """Enriches an events DataFrame with local Visual Crossing weather signals.

    Parameters
    ----------
    state : str
        2-letter postal code, e.g. ``"CA"``.
    city : str
        City name exactly as it appears in the directory.
    date_col : str
        Column in the events DataFrame containing the event date.
    weather_dir : Path
        Root of the weather database.
    """

    def __init__(
        self,
        state: str,
        city: str,
        date_col: str = "date",
        weather_dir: Path = WEATHER_DIR,
    ) -> None:
        self.state = state.upper()
        self.city = city
        self.date_col = date_col
        self.weather_dir = weather_dir

    # ── load ──────────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Read all available year files for (state, city) and return a single DataFrame.

        Returns columns: date, weather_max_temp, weather_min_temp, weather_precip,
        weather_wind_speed, weather_conditions.

        Handles mixed 21-column (2000-2019) and 25-column (2020+) files gracefully
        by selecting columns by name.
        """
        city_dir = self.weather_dir / self.state / self.city
        if not city_dir.exists():
            log.warning("weather: city directory not found — %s", city_dir)
            return pd.DataFrame(columns=["date", *_COL_MAP.values()])

        year_files = sorted(city_dir.glob("*.csv"))
        if not year_files:
            log.warning("weather: no CSV files in %s", city_dir)
            return pd.DataFrame(columns=["date", *_COL_MAP.values()])

        frames: list[pd.DataFrame] = []
        for path in year_files:
            try:
                df = _read_year_file(path)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                log.warning("weather: skipping %s — %s", path.name, exc)

        if not frames:
            return pd.DataFrame(columns=["date", *_COL_MAP.values()])

        combined = pd.concat(frames, ignore_index=True)
        combined = (
            combined
            .drop_duplicates(subset="date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        log.info(
            "weather: loaded %d days for %s/%s (%d files)",
            len(combined), self.state, self.city, len(frames),
        )
        return combined

    # ── enrich ────────────────────────────────────────────────────────────────

    def enrich(self, events: pd.DataFrame) -> pd.DataFrame:
        """Join weather signals onto *events* by date.

        New columns added:
          ``weather_max_temp``, ``weather_min_temp``, ``weather_precip``,
          ``weather_wind_speed``, ``weather_conditions``

        Days with no weather record are filled forward from the most recent
        available date (replaces the silent empty-cell behaviour in
        findWeatherEvent()).
        """
        weather = self.load()

        df = events.copy()
        df["_merge_date"] = pd.to_datetime(df[self.date_col]).dt.normalize()

        merged = df.merge(
            weather.rename(columns={"date": "_merge_date"}),
            on="_merge_date",
            how="left",
        ).sort_values("_merge_date")

        for col in _COL_MAP.values():
            if col in merged.columns:
                merged[col] = merged[col].ffill()

        return merged.drop(columns=["_merge_date"]).reset_index(drop=True)
