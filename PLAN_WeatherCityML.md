# Plan: City-Level Weather ML Models

## Problem

The current pipeline collapses 1,487 cities into unweighted state averages before
training any model.  That aggregation step discards real spatial variation:

- Coastal vs. inland temperature gradients (Los Angeles vs. Bakersfield)
- Elevation-driven temperature inversions
- Urban heat-island effects
- Micro-climate precipitation differences (windward vs. leeward)

The city monthly data already exists (produced by `aggregate_city_to_monthly()`
but never saved or modelled).  The fix is to add a parallel city-level track
alongside the existing state → region → national track.

---

## Architecture

### Current pipeline (unchanged)
```
daily city CSVs
  → aggregate_city_to_monthly()   (per city, in memory only)
  → aggregate_state_monthly()     (unweighted mean across cities)
  → save_state_csv()              → data/Weather/Aggregated/state/{STATE}.csv
  → build_region_df()             (unweighted mean across states)
  → build_national_df()
  → run_weather_group()           → models + results for state/region/national
```

### New parallel city track
```
daily city CSVs
  → aggregate_city_to_monthly()   (same function, already exists)
  → add_temperature_anomaly()     (city-specific baseline, already exists)
  → save_city_csv()               → data/Weather/Aggregated/city/{STATE}/{City}.csv  [NEW]
  → load_city_csv()               [NEW]
  → run_weather_group()           → outputs/city/{STATE}/{City}/  [NEW output path]
```

The model training machinery (`run_weather_group`, `engineer_weather_features`,
`joint_recursive_forecast`) is **identical** at both levels — only the input
DataFrame and output paths differ.

---

## Changes Required

### 1. `weather_model.py` — six additions, no rewrites

#### 1a. Add `CITY_AGG_DIR` constant (after `AGG_DIR` line ~64)
```python
CITY_AGG_DIR = DATA_DIR / "Weather" / "Aggregated" / "city"
```

#### 1b. Add `save_city_csv()` and `load_city_csv()` (after `load_state_csv`)
```python
def save_city_csv(df, state, city, city_agg_dir=CITY_AGG_DIR):
    out = city_agg_dir / state.upper() / city
    out.mkdir(parents=True, exist_ok=True)
    path = out / "monthly.csv"
    df.to_csv(path, index=False)
    return path

def load_city_csv(state, city, city_agg_dir=CITY_AGG_DIR):
    path = city_agg_dir / state.upper() / city / "monthly.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])
```

#### 1c. Add `run_aggregation_cities()` (after `run_aggregation`)
Mirrors `run_aggregation()` but iterates every city in every state (or a subset)
and saves one `monthly.csv` per city.

```python
def run_aggregation_cities(states=None, weather_dir=WEATHER_DIR,
                            city_agg_dir=CITY_AGG_DIR, force=False):
    if states is None:
        states = sorted(d.name for d in weather_dir.iterdir() if d.is_dir())
    ok = skipped = failed = 0
    for state in states:
        for city in discover_state_cities(state, weather_dir):
            path = city_agg_dir / state.upper() / city / "monthly.csv"
            if path.exists() and not force:
                skipped += 1
                continue
            daily = load_city_daily(state, city, weather_dir)
            if daily.empty:
                failed += 1
                continue
            monthly = aggregate_city_to_monthly(daily)
            if monthly.empty:
                failed += 1
                continue
            monthly = add_temperature_anomaly(monthly)
            save_city_csv(monthly, state, city, city_agg_dir)
            ok += 1
    log.info("city aggregation — ok=%d skipped=%d failed=%d", ok, skipped, failed)
```

#### 1d. Extend `--agg-level` CLI option
Add `--agg-level` argument with choices `state` (default), `city`, `all`.
When `--agg-level city` or `all`: call `run_aggregation_cities()` in addition to
or instead of `run_aggregation()`.

#### 1e. Add `--geo city` support in `main()`
When `geo_name == "city"`:
- Determine which cities to model: filtered by `--states` and optional `--city`,
  or all cities if no filter.
- For each `(state, city)`: `df = load_city_csv(state, city)`
- Output directory: `OUTPUT_DIR / "city" / state / city`
- Call `run_weather_group()` with a city-scoped copy of each source config
  (adjusted `model_prefix`, `results_file`, `plot_prefix`).

#### 1f. Add `--city` CLI argument
`--city CityName` — restricts `--geo city` to a single city (requires `--state`).

---

## CLI Interface (after changes)

```bash
# Aggregate city monthly CSVs only (new)
python3 weather_model.py --agg-only --agg-level city
python3 weather_model.py --agg-only --agg-level all           # state + city
python3 weather_model.py --agg-only --agg-level city --states CA,TX

# Train city-level models
python3 weather_model.py --geo city --states CA               # all CA cities
python3 weather_model.py --geo city --states CA --city Bakersfield  # single city
python3 weather_model.py --geo city --source temperature_energy --states TX

# Existing behaviour unchanged
python3 weather_model.py --geo national
python3 weather_model.py --geo all --source all
```

---

## Output Structure

```
outputs/
  # existing (unchanged)
  results_weather_temperature_national.json
  weather_temperature_national_*.png
  ...

  # new city outputs
  city/
    CA/
      Bakersfield/
        results_weather_temperature.json
        results_weather_precipitation.json
        results_weather_extremes.json
        weather_temperature_dashboard.png
        weather_temperature_validation.png
        weather_temperature_importance.png
        weather_temperature_temp_mean.joblib
        ...
    TX/
      Houston/
        ...

data/Weather/Aggregated/
  state/             (existing, unchanged)
    CA.csv  ...
  city/              (new)
    CA/
      Bakersfield/
        monthly.csv
      Los Angeles/
        monthly.csv
    TX/
      Houston/
        monthly.csv
```

---

## Scale & Performance

| Item | Count |
|---|---|
| Total cities | 1,487 |
| City monthly CSVs to generate | 1,487 |
| City × source group training runs | 4,461 (1,487 × 3) |
| Estimated aggregation time | ~5 min (pure pandas, no API) |
| Estimated training time | ~2–4 hrs (full run, all cities) |
| Model files per city | 9 (3 sources × 3 quantiles) |
| Total model files | ~13,400 |

Cities with fewer than `MIN_ROWS_REQUIRED` (54) months of data are auto-skipped
by the existing `run_weather_group()` logic. Most cities have 27 years × 12 = 324
months available, so skips will be rare.

---

## Implementation Steps (in order)

1. Add `CITY_AGG_DIR` constant to `weather_model.py`
2. Add `save_city_csv()` and `load_city_csv()` after `load_state_csv()`
3. Add `run_aggregation_cities()` after `run_aggregation()`
4. Add `--agg-level` CLI argument; wire into `main()` to call city aggregation
5. Add `--city` CLI argument
6. Add `--geo city` handling in `main()`: iterate cities, build city-scoped
   output paths, call `run_weather_group()`
7. Run full city aggregation: `python3 weather_model.py --agg-only --agg-level city`
8. Smoke-test a single city: `python3 weather_model.py --geo city --states CA --city Bakersfield`
9. Commit

---

## What Is NOT Changing

- `aggregate_city_to_monthly()` — no changes; already correct
- `run_weather_group()` — no changes; already geography-agnostic
- `engineer_weather_features()` — no changes
- State / region / national aggregation and training — no changes
- `connectors/weather.py`, `weather_refresh.py`, `macro_utils.py` — untouched
- `reports.py` — not changed in this plan (city forecasts are too numerous for
  a static report table; a future city-lookup feature is a separate task)
