# Code Review: Improvements & Bug Fixes Plan

## Context
Comprehensive review of the econdata Python data pipeline covering fred_refresh.py, briefing.py, api.py, weather_refresh.py, connectors/, and model files. Issues are grouped by severity. The goal is to fix real bugs first, then eliminate silent failure modes, then improve maintainability.

---

## CRITICAL

### 1. .env file should not be committed to git
- **File**: `.gitignore`
- **Issue**: `.env` is not listed in `.gitignore`, meaning API keys (FRED, NewsAPI, Marketaux, Finnhub, FMP, Visual Crossing, Anthropic) are at risk of being pushed to a remote repo.
- **Fix**: Add `.env` to `.gitignore`.

---

## HIGH — Real Bugs

### 2. Weather deduplication keeps stale rows (`weather_refresh.py:143`)
- **Issue**: `drop_duplicates(keep="first")` retains the old row when the API returns a revised value for a date that already exists in the file. Should be `keep="last"` so revisions win.
- **Fix**: Change `keep="first"` → `keep="last"` in the `_append_rows()` / dedup logic.

### 3. Module-level `datetime.now().year` evaluated at import time (`fred_refresh.py:27`, `weather_refresh.py:73`)
- **Issue**: `_current_year = datetime.now().year` is frozen at import time. If the process starts on Dec 31 and runs past midnight, or if the module is imported early and used late, `to_year` will be wrong.
- **Fix**: Replace module-level constant with a `_get_current_year()` function called at runtime, or use `default=None` in argparse and resolve inside `main()`.

### 4. Row count calculation can be negative and is silently masked (`fred_refresh.py:253–255`)
- **Issue**: `new_count = len(combined) - len(existing)` can be negative when deduplication reduces rows. The `max(new_count, 0)` on line 255 hides this. When FRED revises historical data and reduces row count, the pipeline silently misreports.
- **Fix**: Calculate `new_count` as the number of rows in `combined` with dates strictly greater than the previous max date, not by diffing lengths.

### 5. Unlimited `ffill()` can propagate stale data indefinitely (`connectors/weather.py:267`, `connectors/market_bias.py:171–172`)
- **Issue**: `merged[col].ffill()` with no `limit=` argument will carry forward the last known value across arbitrarily large gaps. A city missing months of weather data silently uses December values in June.
- **Fix**: Use `ffill(limit=7)` for weather (reasonable max gap for retries) and `ffill(limit=5)` for market signals (trading days in a week). Log a warning when fills exceed threshold.

### 6. No retry logic on FRED API transient failures (`fred_refresh.py:194–210`)
- **Issue**: Single `requests.get()` with no retry. A transient 429 or 503 fails the entire series silently. At 50+ series, this is a real operational risk.
- **Fix**: Wrap the FRED fetch in a simple retry loop (3 attempts, exponential backoff) before marking the series as failed.

---

## MEDIUM — Silent Failure Modes

### 7. Bare `except Exception: pass` blocks swallow errors silently
- **Files**:
  - `fred_refresh.py:768` — loading model result JSON
  - `fred_refresh.py:800` — loading regime/FSI JSON
  - `fred_refresh.py:1426` — parsing run log JSON
  - `api.py:874, 911, 1030, 1553, 1876` — various endpoint handlers
- **Fix**: Replace `pass` with `log.warning("...", exc)` at minimum. For api.py, return a structured error response.

### 8. Corrupted weather files indistinguishable from missing files (`weather_refresh.py:112–127`)
- **Issue**: `_last_date_in_file()` returns `None` for both "file doesn't exist" and "file is corrupted/unreadable." Both paths trigger a full re-fetch, but a corrupted file is silently overwritten with no warning.
- **Fix**: Distinguish the two cases — log `WARNING` for parse failures vs. `DEBUG` for missing files.

### 9. Exception handling in connectors is too broad
- **Files**: `connectors/market_bias.py:139–141`, `connectors/lunar.py` (no logging at all)
- **Fix**: Catch specific exceptions (e.g., `requests.HTTPError`, `pd.errors.ParserError`) and add `log.warning` in `lunar.py`.

### 10. Shared VC row counter is not safe for concurrent use (`weather_refresh.py:322`)
- **Issue**: `_row_counter[0]` is a mutable list used as a shared counter. If `refresh_all()` is ever called from multiple threads/processes, the count can race. Currently single-process but worth making safe.
- **Fix**: Use `threading.Lock` around counter increment, or document that concurrent use is unsupported.

### 11. Outputs / run log grows unbounded
- **File**: `outputs/refresh_log.json`, `fred_refresh.py:1429`
- **Issue**: Run log keeps only `[-52:]` (last 52 runs) but this logic runs only on read, not on write. JSON file grows until read and re-written.
- **Fix**: Enforce trim-on-write, not trim-on-read.

---

## LOW — Code Quality & Maintainability

### 12. Duplicate `.env` / API key loading across files
- **Files**: `fred_refresh.py:78–102`, `weather_refresh.py` (copy-paste of same logic)
- **Fix**: Extract to `utils/config.py` — single `load_env_key(var_name, fatal=True)` function used everywhere.

### 13. Hardcoded absolute path in connector (`connectors/weather.py:35`)
- **Issue**: `WEATHER_DIR = Path("/Users/gjlarsen/aiprojects/econdata/data/Weather/US")` breaks on any other machine.
- **Fix**: `Path(__file__).parent.parent / "data" / "Weather" / "US"`

### 14. `briefing.py` argparse broken when imported as module (`briefing.py:366`)
- **Issue**: `parser.parse_args([] if __name__ != "__main__" else None)` always passes empty args when imported.
- **Fix**: Guard argparse inside `if __name__ == "__main__":` block only.

### 15. `BRIEFING_STALE_HOURS` defined in two places out of sync
- **Files**: `briefing.py` and `api.py:854`
- **Fix**: Define once in a shared constants module and import in both places.

### 16. Model output clipping ranges are hardcoded with no warning on hit
- **Files**: `business_env_model.py:29–33`, `market_model.py:61–68`
- **Fix**: Add a `log.warning` when a prediction is clipped.

### 17. Duplicate `os.makedirs("outputs", exist_ok=True)` across model files
- **Fix**: Call once in the orchestrator (`fred_refresh.py`) before dispatching model runs.

### 18. Large hardcoded `SERIES_FILE_MAP` in code (`fred_refresh.py:57–120`)
- **Fix** (low priority): Move to `config/series.json` loaded at startup.

---

## Verification

After implementing fixes, validate with:
1. `python3 fred_refresh.py --sector` — confirm row counts are non-negative and logged correctly
2. `python3 weather_refresh.py --state CA --city Bakersfield` — confirm `keep="last"` dedup behavior
3. `python3 briefing.py` — confirm no argparse errors when run as script
4. Force a FRED API timeout — confirm retry logic fires and logs correctly
5. Check `.gitignore` includes `.env` before next commit
