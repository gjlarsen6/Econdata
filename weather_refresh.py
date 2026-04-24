"""
weather_refresh.py — Visual Crossing Weather Database Updater

Fetches new or missing year files from weather.visualcrossing.com and writes
them into the existing local weather database:
  /gjlarsen/Documents/ClickAI/data/Weather/US_orig/{STATE}/{CityName}/YYYY_{City}{STATE}.csv

Existing 2000-2019 files are left untouched unless --overwrite is passed.
Default behaviour adds years 2020 through the current year.

Usage:
    python3 weather_refresh.py --state CA --city Bakersfield
    python3 weather_refresh.py --state ID
    python3 weather_refresh.py --all
    python3 weather_refresh.py --state CA --city Bakersfield --from-year 2020 --to-year 2026
    python3 weather_refresh.py --state CA --city Bakersfield --overwrite

Requires VISUAL_CROSSING_API_KEY in the environment or a .env file in this directory.

Rate limits:
    Free tier  : ~1,000 observations/day  (~2-3 city-years per day)
    Basic tier : ~10,000 observations/day (~27 city-years per day)
    The script logs a warning when approaching the configured daily limit.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from connectors.weather import WEATHER_DIR, _city_file_stem, _year_file_path

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent

VC_API_BASE = (
    "https://weather.visualcrossing.com/VisualCrossingWebServices"
    "/rest/services/weatherdata/history"
)
VC_RATE_SLEEP   = 1.0   # seconds between API calls
VC_FREE_LIMIT   = 1000  # daily observation limit (free tier)
VC_WARN_PCT     = 0.80  # warn at this fraction of the daily limit

_current_year = datetime.now().year


# ── API key ───────────────────────────────────────────────────────────────────

def _load_vc_api_key() -> str:
    """Load VISUAL_CROSSING_API_KEY from env or .env file."""
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("VISUAL_CROSSING_API_KEY"):
                    _, _, val = line.partition("=")
                    os.environ["VISUAL_CROSSING_API_KEY"] = val.strip().strip('"').strip("'")

    key = os.getenv("VISUAL_CROSSING_API_KEY", "").strip()
    if not key:
        log.error(
            "VISUAL_CROSSING_API_KEY not set.\n"
            "  Set it as an environment variable:  "
            "export VISUAL_CROSSING_API_KEY=your_key\n"
            "  Or add VISUAL_CROSSING_API_KEY=your_key to a .env file in this directory.\n"
            "  Free keys: https://www.visualcrossing.com/weather-api"
        )
        sys.exit(1)
    return key


# ── Core fetch ────────────────────────────────────────────────────────────────

def fetch_vc_year(
    state: str,
    city: str,
    year: int,
    api_key: str,
) -> pd.DataFrame | None:
    """Fetch one full calendar year of daily weather from the VC API.

    Parameters
    ----------
    state : str   2-letter postal code, e.g. ``"CA"``
    city  : str   City name, e.g. ``"Bakersfield"``
    year  : int   Calendar year to fetch
    api_key : str Visual Crossing API key

    Returns
    -------
    pd.DataFrame | None
        DataFrame with the raw VC columns on success; ``None`` on error.
    """
    # Cap end date at today for the current year — VC won't return future dates
    end_date = min(datetime(year, 12, 31), datetime.now()).strftime("%Y-%m-%d")
    start_date = f"{year}-01-01"

    params = {
        "aggregateHours":              24,
        "startDateTime":               f"{start_date}T00:00:00",
        "endDateTime":                 f"{end_date}T00:00:00",
        "collectStationContributions": "false",
        "maxStations":                 -1,
        "maxDistance":                 -1,
        "includeNormals":              "false",
        "contentType":                 "csv",
        "unitGroup":                   "us",
        "locationMode":                "single",
        "key":                         api_key,
        "locations":                   f"{city},{state.upper()}",
    }

    t0 = time.time()
    try:
        resp = requests.get(VC_API_BASE, params=params, timeout=30)
    except Exception as exc:
        log.warning("[VC] %s/%s/%d  request failed — %s", state, city, year, exc)
        return None

    elapsed = time.time() - t0

    if resp.status_code != 200:
        log.warning(
            "[VC] %s/%s/%d  HTTP %d — %s",
            state, city, year, resp.status_code, resp.text[:200],
        )
        return None

    try:
        df = pd.read_csv(StringIO(resp.text))
    except Exception as exc:
        log.warning("[VC] %s/%s/%d  CSV parse failed — %s", state, city, year, exc)
        return None

    if df.empty or "Date time" not in df.columns:
        log.warning("[VC] %s/%s/%d  empty or missing 'Date time' column", state, city, year)
        return None

    log.info(
        "[VC] %s/%s/%d  %d rows  (%.1fs)",
        state, city, year, len(df), elapsed,
    )
    return df


# ── Refresh a single city ─────────────────────────────────────────────────────

def refresh_city(
    state: str,
    city: str,
    api_key: str,
    from_year: int = 2020,
    to_year: int = _current_year,
    overwrite: bool = False,
    weather_dir: Path = WEATHER_DIR,
    _row_counter: list[int] | None = None,
    daily_limit: int = VC_FREE_LIMIT,
) -> list[dict]:
    """Fetch missing (or all) year files for one city.

    Parameters
    ----------
    state, city, api_key : str
    from_year, to_year   : int   Inclusive year range to process
    overwrite            : bool  Re-fetch even if the local file already exists
    weather_dir          : Path  Root of the weather database
    _row_counter         : list[int]  Mutable counter shared with refresh_all for
                                      rate-limit tracking (single-element list)
    daily_limit          : int   Warn threshold for daily observation count

    Returns
    -------
    list[dict]  One entry per year: {state, city, year, rows, status, elapsed}
    """
    if _row_counter is None:
        _row_counter = [0]

    results: list[dict] = []

    for year in range(from_year, to_year + 1):
        path = _year_file_path(state, city, year, weather_dir)
        record: dict = {
            "state": state.upper(), "city": city, "year": year,
            "rows": 0, "status": "skipped", "elapsed": 0.0,
        }

        if path.exists() and not overwrite:
            log.debug("[VC] skip  %s/%s/%d — file exists", state, city, year)
            results.append(record)
            continue

        if _row_counter[0] >= daily_limit * VC_WARN_PCT:
            log.warning(
                "[VC] approaching daily limit (%d / %d rows fetched) — "
                "consider running again tomorrow or upgrading your VC tier",
                _row_counter[0], daily_limit,
            )

        t0 = time.time()
        df = fetch_vc_year(state, city, year, api_key)
        record["elapsed"] = round(time.time() - t0, 2)

        if df is None:
            record["status"] = "error"
            results.append(record)
            time.sleep(VC_RATE_SLEEP)
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

        record["rows"]   = len(df)
        record["status"] = "ok"
        _row_counter[0] += len(df)
        results.append(record)

        time.sleep(VC_RATE_SLEEP)

    return results


# ── Refresh all cities ────────────────────────────────────────────────────────

def refresh_all(
    api_key: str,
    from_year: int = 2020,
    to_year: int = _current_year,
    overwrite: bool = False,
    weather_dir: Path = WEATHER_DIR,
    daily_limit: int = VC_FREE_LIMIT,
) -> list[dict]:
    """Refresh every city found in the weather database.

    Walks ``weather_dir/{state}/{city}/`` and calls ``refresh_city()`` for
    each city directory found.  A shared row counter tracks total observations
    fetched against the daily limit.

    This is designed for paid tier usage or multi-day runs.  On the free tier
    (~1000 obs/day) only 2-3 city-years can be fetched before hitting the limit.
    """
    if not weather_dir.exists():
        log.error("[VC] weather_dir not found: %s", weather_dir)
        return []

    row_counter = [0]
    all_results: list[dict] = []

    state_dirs = sorted(d for d in weather_dir.iterdir() if d.is_dir())
    log.info("[VC] refresh_all: %d state directories found", len(state_dirs))

    for state_dir in state_dirs:
        state = state_dir.name
        city_dirs = sorted(d for d in state_dir.iterdir() if d.is_dir())

        for city_dir in city_dirs:
            city = city_dir.name
            results = refresh_city(
                state, city, api_key,
                from_year=from_year, to_year=to_year,
                overwrite=overwrite, weather_dir=weather_dir,
                _row_counter=row_counter, daily_limit=daily_limit,
            )
            all_results.extend(results)

    ok    = sum(1 for r in all_results if r["status"] == "ok")
    skip  = sum(1 for r in all_results if r["status"] == "skipped")
    err   = sum(1 for r in all_results if r["status"] == "error")
    total = sum(r["rows"] for r in all_results)
    log.info(
        "[VC] refresh_all complete — ok=%d  skipped=%d  errors=%d  total_rows=%d",
        ok, skip, err, total,
    )
    return all_results


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(results: list[dict]) -> None:
    if not results:
        print("No results.")
        return

    ok    = [r for r in results if r["status"] == "ok"]
    skip  = [r for r in results if r["status"] == "skipped"]
    err   = [r for r in results if r["status"] == "error"]

    print(f"\n{'─'*60}")
    print(f"  Fetched : {len(ok):>4}  year-files  "
          f"({sum(r['rows'] for r in ok):,} rows)")
    print(f"  Skipped : {len(skip):>4}  (already exist)")
    print(f"  Errors  : {len(err):>4}")
    if err:
        for r in err:
            print(f"    ✗ {r['state']}/{r['city']}/{r['year']}")
    print(f"{'─'*60}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Update the local Visual Crossing weather database with new year files. "
            "Requires VISUAL_CROSSING_API_KEY in environment or .env file."
        )
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--all",    action="store_true",
                        help="Refresh every city in the database")
    target.add_argument("--state",  metavar="XX",
                        help="2-letter state code (e.g. CA)")

    parser.add_argument("--city",       metavar="NAME",
                        help="City name (requires --state; omit to refresh all "
                             "cities in the state)")
    parser.add_argument("--from-year",  type=int, default=2020,
                        metavar="YYYY",
                        help="First year to fetch (default: 2020)")
    parser.add_argument("--to-year",    type=int, default=_current_year,
                        metavar="YYYY",
                        help=f"Last year to fetch (default: {_current_year})")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Re-fetch files that already exist locally")
    parser.add_argument("--daily-limit", type=int, default=VC_FREE_LIMIT,
                        metavar="N",
                        help=f"Daily observation limit for rate warnings "
                             f"(default: {VC_FREE_LIMIT})")
    parser.add_argument("--weather-dir", type=Path, default=WEATHER_DIR,
                        metavar="PATH",
                        help=f"Root of the weather database (default: {WEATHER_DIR})")

    args = parser.parse_args()

    if args.city and not args.state:
        parser.error("--city requires --state")

    api_key = _load_vc_api_key()

    kwargs = dict(
        from_year=args.from_year,
        to_year=args.to_year,
        overwrite=args.overwrite,
        weather_dir=args.weather_dir,
        daily_limit=args.daily_limit,
    )

    if args.all:
        results = refresh_all(api_key, **kwargs)

    elif args.city:
        results = refresh_city(args.state, args.city, api_key, **kwargs)

    else:
        # --state only: refresh all cities in that state
        state_dir = args.weather_dir / args.state.upper()
        if not state_dir.exists():
            log.error("State directory not found: %s", state_dir)
            sys.exit(1)
        results = []
        row_counter = [0]
        for city_dir in sorted(d for d in state_dir.iterdir() if d.is_dir()):
            results.extend(
                refresh_city(
                    args.state, city_dir.name, api_key,
                    _row_counter=row_counter, **kwargs,
                )
            )

    _print_summary(results)


if __name__ == "__main__":
    main()
