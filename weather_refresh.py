"""
weather_refresh.py — Visual Crossing Weather Database Updater

Incrementally updates the local weather database at:
  data/Weather/US/{STATE}/{CityName}/YYYY_{CityNoSpaces}{STATE}.csv

Behaviour per year-file
-----------------------
* Past year, file absent   → fetch full year (Jan 1 – Dec 31), write new file
* Past year, file present  → check last date in file
    - last date == Dec 31  → skip (complete)
    - last date < Dec 31   → fetch gap (last_date+1 – Dec 31), append to file
* Current year, file absent → fetch Jan 1 – today, write new file
* Current year, file present → fetch gap (last_date+1 – today), append to file
* --overwrite              → always re-fetch full year and overwrite file

This means the script is safe to run daily (appends only the new day) or after
an extended gap (catches up all missing days automatically).  Historical files
that are already complete are never re-fetched.

Usage:
    python3 weather_refresh.py --state CA --city Bakersfield
    python3 weather_refresh.py --state CA
    python3 weather_refresh.py --all
    python3 weather_refresh.py --all --from-year 2022
    python3 weather_refresh.py --state CA --city Bakersfield --overwrite

Requires VISUAL_CROSSING_API_KEY in the environment or a .env file.

Rate limits:
    Free tier  : ~1,000 observations/day  (~2–3 city-years per day)
    Basic tier : ~10,000 observations/day (~27 city-years per day)
    The script logs a warning when approaching the configured daily limit.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from connectors.weather import WEATHER_DIR, _year_file_path

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
VC_RATE_SLEEP = 1.0    # seconds between API calls
VC_FREE_LIMIT = 1000   # daily observation limit (free tier)
VC_WARN_PCT   = 0.80   # warn at this fraction of the daily limit

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


# ── Date helpers ──────────────────────────────────────────────────────────────

def _year_end(year: int) -> date:
    """Last calendar day of the year."""
    return date(year, 12, 31)


def _last_date_in_file(path: Path) -> date | None:
    """Return the most recent date found in a year-file, or None if unreadable.

    Expects a 'Date time' column in MM/DD/YYYY format.
    """
    try:
        df = pd.read_csv(path, usecols=["Date time"])
        if df.empty:
            return None
        parsed = pd.to_datetime(df["Date time"], format="%m/%d/%Y", errors="coerce").dropna()
        if parsed.empty:
            return None
        return parsed.max().date()
    except Exception as exc:
        log.warning("[VC] could not read last date from %s — %s", path, exc)
        return None


# ── File helpers ──────────────────────────────────────────────────────────────

def _append_to_year_file(path: Path, new_df: pd.DataFrame) -> int:
    """Append new rows to an existing year-file CSV.

    Deduplicates on 'Date time' (new API rows take priority, preserving
    any revisions), sorts chronologically, and writes the merged result
    back to disk.

    Returns the number of net-new rows added.
    """
    existing = pd.read_csv(path)
    before   = len(existing)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date time"], keep="last")
    combined["_sort"] = pd.to_datetime(
        combined["Date time"], format="%m/%d/%Y", errors="coerce"
    )
    combined = (
        combined.sort_values("_sort")
        .drop(columns=["_sort"])
        .reset_index(drop=True)
    )
    combined.to_csv(path, index=False)
    return len(combined) - before


# ── Core fetch ────────────────────────────────────────────────────────────────

def fetch_vc_range(
    state: str,
    city: str,
    start: date,
    end: date,
    api_key: str,
) -> pd.DataFrame | None:
    """Fetch daily weather for an arbitrary date range from the VC API.

    Parameters
    ----------
    state   : 2-letter postal code, e.g. ``"CA"``
    city    : City name exactly as it appears in the directory
    start   : First date to fetch (inclusive)
    end     : Last date to fetch (inclusive); capped at today automatically
    api_key : Visual Crossing API key

    Returns
    -------
    pd.DataFrame | None
        DataFrame with raw VC columns on success; empty DataFrame if the
        date range contains no data yet; ``None`` on network/parse error.
    """
    today = date.today()
    end   = min(end, today)

    if start > end:
        return pd.DataFrame()

    params = {
        "aggregateHours":              24,
        "startDateTime":               f"{start.isoformat()}T00:00:00",
        "endDateTime":                 f"{end.isoformat()}T00:00:00",
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
        log.warning(
            "[VC] %s/%s %s→%s  request failed — %s",
            state, city, start, end, exc,
        )
        return None

    elapsed = time.time() - t0

    if resp.status_code != 200:
        log.warning(
            "[VC] %s/%s %s→%s  HTTP %d — %s",
            state, city, start, end, resp.status_code, resp.text[:200],
        )
        return None

    try:
        df = pd.read_csv(StringIO(resp.text))
    except Exception as exc:
        log.warning(
            "[VC] %s/%s %s→%s  CSV parse failed — %s",
            state, city, start, end, exc,
        )
        return None

    if df.empty or "Date time" not in df.columns:
        log.warning(
            "[VC] %s/%s %s→%s  empty or missing 'Date time' column",
            state, city, start, end,
        )
        return None

    log.info(
        "[VC] %s/%s %s→%s  %d rows  (%.1fs)",
        state, city, start, end, len(df), elapsed,
    )
    return df


# ── Refresh a single city ─────────────────────────────────────────────────────

def refresh_city(
    state: str,
    city: str,
    api_key: str,
    from_year: int = 2020,
    to_year: int | None = None,
    overwrite: bool = False,
    weather_dir: Path = WEATHER_DIR,
    _row_counter: list[int] | None = None,
    daily_limit: int = VC_FREE_LIMIT,
) -> list[dict]:
    """Incrementally update year-files for one city.

    For each year in ``[from_year, to_year]``:

    * ``overwrite=True``           → re-fetch full year, overwrite file
    * file absent                  → fetch full year, create file
    * file present, already current → skip
    * file present, stale          → fetch gap only, append to file

    A file is considered "current" when its last row date equals Dec 31 for
    past years, or equals yesterday / today for the current year.

    Returns
    -------
    list[dict]
        One entry per year: ``{state, city, year, rows, status, elapsed}``
        Status values: ``"skipped"`` | ``"ok"`` | ``"appended"`` | ``"error"``
    """
    if _row_counter is None:
        _row_counter = [0]
    resolved_to_year = to_year if to_year is not None else datetime.now().year

    today   = date.today()
    results: list[dict] = []

    for year in range(from_year, resolved_to_year + 1):
        path        = _year_file_path(state, city, year, weather_dir)
        target_end  = min(_year_end(year), today)
        record: dict = {
            "state": state.upper(), "city": city, "year": year,
            "rows": 0, "status": "skipped", "elapsed": 0.0,
        }

        # ── Determine fetch start and whether to append ────────────────────────
        if overwrite or not path.exists():
            fetch_start  = date(year, 1, 1)
            append_mode  = False
        else:
            last = _last_date_in_file(path)

            if last is None:
                # Unreadable — treat as absent and re-fetch from start
                log.warning(
                    "[VC] %s/%s/%d  could not read dates — re-fetching full year",
                    state, city, year,
                )
                fetch_start = date(year, 1, 1)
                append_mode = False
            elif last >= target_end:
                # File is fully current
                log.debug(
                    "[VC] skip  %s/%s/%d — current through %s",
                    state, city, year, last,
                )
                results.append(record)
                continue
            else:
                # File exists but is missing days after `last`
                fetch_start = last + timedelta(days=1)
                append_mode = True

        if fetch_start > target_end:
            results.append(record)
            continue

        # ── Rate-limit warning ─────────────────────────────────────────────────
        if _row_counter[0] >= daily_limit * VC_WARN_PCT:
            log.warning(
                "[VC] approaching daily limit (%d / %d rows fetched) — "
                "consider running again tomorrow or upgrading your VC tier",
                _row_counter[0], daily_limit,
            )

        # ── Fetch ──────────────────────────────────────────────────────────────
        t0 = time.time()
        df = fetch_vc_range(state, city, fetch_start, target_end, api_key)
        record["elapsed"] = round(time.time() - t0, 2)

        if df is None:
            record["status"] = "error"
            results.append(record)
            time.sleep(VC_RATE_SLEEP)
            continue

        if df.empty:
            # No new rows available yet (e.g. today's data not yet published)
            log.debug(
                "[VC] %s/%s/%d  no new data available for %s → %s",
                state, city, year, fetch_start, target_end,
            )
            results.append(record)
            continue

        # ── Write ──────────────────────────────────────────────────────────────
        path.parent.mkdir(parents=True, exist_ok=True)

        if append_mode and path.exists():
            added = _append_to_year_file(path, df)
            record["rows"]   = added
            record["status"] = "appended"
            log.info(
                "[VC] %s/%s/%d  appended %d rows (through %s)",
                state, city, year, added, target_end,
            )
        else:
            df.to_csv(path, index=False)
            record["rows"]   = len(df)
            record["status"] = "ok"

        _row_counter[0] += record["rows"]
        results.append(record)
        time.sleep(VC_RATE_SLEEP)

    return results


# ── Refresh all cities ────────────────────────────────────────────────────────

def refresh_all(
    api_key: str,
    from_year: int = 2020,
    to_year: int | None = None,
    overwrite: bool = False,
    weather_dir: Path = WEATHER_DIR,
    daily_limit: int = VC_FREE_LIMIT,
) -> list[dict]:
    """Incrementally refresh every city found in the weather database.

    Walks ``weather_dir/{state}/{city}/`` and calls ``refresh_city()`` for
    each city directory found.  A shared row counter tracks total observations
    fetched against the daily limit.

    Designed to run daily: cities already current are skipped in milliseconds;
    only cities with missing days make API calls.
    """
    resolved_to_year = to_year if to_year is not None else datetime.now().year

    if not weather_dir.exists():
        log.error("[VC] weather_dir not found: %s", weather_dir)
        return []

    row_counter  = [0]
    all_results: list[dict] = []

    state_dirs = sorted(d for d in weather_dir.iterdir() if d.is_dir())
    log.info("[VC] refresh_all: %d state directories found", len(state_dirs))

    for state_dir in state_dirs:
        state     = state_dir.name
        city_dirs = sorted(d for d in state_dir.iterdir() if d.is_dir())

        for city_dir in city_dirs:
            city = city_dir.name
            results = refresh_city(
                state, city, api_key,
                from_year=from_year, to_year=resolved_to_year,
                overwrite=overwrite, weather_dir=weather_dir,
                _row_counter=row_counter, daily_limit=daily_limit,
            )
            all_results.extend(results)

    ok       = sum(1 for r in all_results if r["status"] == "ok")
    appended = sum(1 for r in all_results if r["status"] == "appended")
    skipped  = sum(1 for r in all_results if r["status"] == "skipped")
    err      = sum(1 for r in all_results if r["status"] == "error")
    total    = sum(r["rows"] for r in all_results)
    log.info(
        "[VC] refresh_all complete — new=%d  appended=%d  skipped=%d  errors=%d  rows=%d",
        ok, appended, skipped, err, total,
    )
    return all_results


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(results: list[dict]) -> None:
    if not results:
        print("No results.")
        return

    ok       = [r for r in results if r["status"] == "ok"]
    appended = [r for r in results if r["status"] == "appended"]
    skipped  = [r for r in results if r["status"] == "skipped"]
    err      = [r for r in results if r["status"] == "error"]

    print(f"\n{'─'*60}")
    print(f"  New files : {len(ok):>4}  ({sum(r['rows'] for r in ok):,} rows written)")
    print(f"  Appended  : {len(appended):>4}  ({sum(r['rows'] for r in appended):,} rows added to existing files)")
    print(f"  Skipped   : {len(skipped):>4}  (already current)")
    print(f"  Errors    : {len(err):>4}")
    if err:
        for r in err:
            print(f"    ✗ {r['state']}/{r['city']}/{r['year']}")
    print(f"{'─'*60}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally update the local Visual Crossing weather database. "
            "Appends missing days to existing year-files; creates new year-files "
            "when needed.  Requires VISUAL_CROSSING_API_KEY in environment or .env."
        )
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--all",   action="store_true",
                        help="Refresh every city in the database")
    target.add_argument("--state", metavar="XX",
                        help="2-letter state code (e.g. CA)")

    parser.add_argument("--city",        metavar="NAME",
                        help="City name (requires --state; omit to refresh all "
                             "cities in the state)")
    parser.add_argument("--from-year",   type=int, default=2020, metavar="YYYY",
                        help="First year to consider (default: 2020)")
    parser.add_argument("--to-year",     type=int, default=_current_year, metavar="YYYY",
                        help=f"Last year to consider (default: {_current_year})")
    parser.add_argument("--overwrite",   action="store_true",
                        help="Re-fetch and overwrite files that already exist")
    parser.add_argument("--daily-limit", type=int, default=VC_FREE_LIMIT, metavar="N",
                        help=f"Daily observation limit for rate warnings "
                             f"(default: {VC_FREE_LIMIT})")
    parser.add_argument("--weather-dir", type=Path, default=WEATHER_DIR, metavar="PATH",
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
        results    = []
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
