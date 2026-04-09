"""
sector_apis.py — Sector API fetchers for BLS, BEA, World Bank, and Trading Economics.

Each fetcher saves data to data/Sector/{Source}/<series_id>.csv in the same
[observation_date, <series_id>] format used by all FRED CSVs in this project.
Data is additive — existing rows are never deleted, new data is merged in.

Usage (called from fred_refresh.py):
    import sector_apis
    keys = sector_apis.load_sector_api_keys()
    results = sector_apis.refresh_bls(api_key=keys["BLS_API_KEY"])
    results = sector_apis.refresh_bea(api_key=keys["BEA_API_KEY"])
    results = sector_apis.refresh_worldbank()
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent
SECTOR_DATA_DIR = BASE_DIR / "data" / "Sector"

BLS_API_BASE = "https://api.bls.gov/publicAPI/v2"
BEA_API_BASE = "https://apps.bea.gov/api/data/"
WB_API_BASE  = "https://api.worldbank.org/v2"
TE_API_BASE  = "https://api.tradingeconomics.com"

# ── BLS Series Catalogue ──────────────────────────────────────────────────────

BLS_SERIES_IDS: dict[str, str] = {
    "CES3000000001": "Manufacturing Employment",
    "CES4000000001": "Trade Transport Utilities Employment",
    "CES5500000001": "Financial Activities Employment",
    "CES6000000001": "Professional Business Services Employment",
    "CEU6500000001": "Education Health Employment",
    "CEU7000000001": "Leisure Hospitality Employment",
}

# ── BEA Industry Map ──────────────────────────────────────────────────────────
# Maps IndustryID from BEA response → CSV column name / file stem.
# IndustryID values come from the GDPbyIndustry dataset; check BEA docs for
# updated codes if the API returns different IDs.

BEA_INDUSTRY_MAP: dict[str, str] = {
    "31G":   "BEA_Manufacturing",
    "52-53": "BEA_Finance_Insurance_RE",
    "44-45": "BEA_Wholesale_Retail_Trade",
    "54-56": "BEA_Professional_Biz_Svcs",
}

# ── World Bank Indicators ─────────────────────────────────────────────────────

WB_INDICATORS: dict[str, str] = {
    "NV.IND.MANF.ZS": "Manufacturing pct GDP",
    "NV.SRV.TOTL.ZS":  "Services pct GDP",
    "NV.IND.TOTL.ZS":  "Industry pct GDP",
}

# ── Key loader ────────────────────────────────────────────────────────────────

def load_sector_api_keys() -> dict[str, str | None]:
    """
    Load optional sector API keys from environment (or .env file if already
    loaded by fred_refresh.py).  Does NOT sys.exit() — callers decide whether
    to skip or abort when a key is missing.
    """
    return {
        "BEA_API_KEY":     os.getenv("BEA_API_KEY", "").strip() or None,
        "BLS_API_KEY":     os.getenv("BLS_API_KEY", "").strip() or None,
        "TE_CLIENT_KEY":   os.getenv("TE_CLIENT_KEY", "").strip() or None,
        "TE_CLIENT_SECRET":os.getenv("TE_CLIENT_SECRET", "").strip() or None,
    }

# ── CSV helpers (mirror fred_refresh.py patterns exactly) ─────────────────────

def _sector_csv_path(source: str, filename: str) -> Path:
    """Return Path to data/Sector/{source}/{filename}, creating parent dirs."""
    p = SECTOR_DATA_DIR / source / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _load_sector_existing(path: Path, series_col: str) -> pd.DataFrame:
    """Load existing sector CSV or return an empty DataFrame."""
    if not path.exists():
        return pd.DataFrame(columns=["observation_date", series_col])
    df = pd.read_csv(path, parse_dates=["observation_date"])
    if series_col not in df.columns:
        df[series_col] = float("nan")
    return df[["observation_date", series_col]]

def _merge_and_save_sector(existing: pd.DataFrame, new_rows: pd.DataFrame,
                            series_col: str, path: Path) -> int:
    """
    Concat existing + new, prefer newer values on date overlap (handles
    revisions), sort by observation_date, save CSV.
    Returns net-new row count (0 if only revisions, not new dates).
    """
    if new_rows.empty:
        existing.to_csv(path, index=False)
        return 0
    combined = (
        pd.concat([existing, new_rows], ignore_index=True)
        .drop_duplicates(subset=["observation_date"], keep="last")
        .sort_values("observation_date")
        .reset_index(drop=True)
    )
    net_new = max(len(combined) - len(existing), 0)
    combined.to_csv(path, index=False)
    return net_new

# ── BLS Fetcher ───────────────────────────────────────────────────────────────

def fetch_bls_series(series_ids: list[str],
                     start_year: int,
                     end_year: int,
                     api_key: str | None = None) -> dict[str, pd.DataFrame]:
    """
    POST to BLS /timeseries/data/ for up to 25 series.
    Returns {series_id: DataFrame[observation_date, series_id]}.

    BLS period codes: M01..M12 = January..December; M13 = annual average (skipped).
    Values are returned as strings and may be empty strings for missing data.
    """
    body: dict = {
        "seriesid":  series_ids[:25],   # BLS limit per request
        "startyear": str(start_year),
        "endyear":   str(end_year),
    }
    if api_key:
        body["registrationkey"] = api_key

    resp = requests.post(
        f"{BLS_API_BASE}/timeseries/data/",
        json=body,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API error: {data.get('message', data.get('status'))}")

    result: dict[str, pd.DataFrame] = {}
    for series_blob in data.get("Results", {}).get("series", []):
        sid = series_blob["seriesID"]
        rows = []
        for pt in series_blob.get("data", []):
            period = pt.get("period", "")
            if period == "M13":          # skip annual average
                continue
            if not period.startswith("M"):
                continue
            try:
                month_num = int(period[1:])
                obs_date  = pd.Timestamp(f"{pt['year']}-{month_num:02d}-01")
                value     = pd.to_numeric(pt.get("value", ""), errors="coerce")
                rows.append({"observation_date": obs_date, sid: value})
            except (ValueError, KeyError):
                continue
        if rows:
            df = (pd.DataFrame(rows)
                  .sort_values("observation_date")
                  .reset_index(drop=True))
            result[sid] = df
        else:
            result[sid] = pd.DataFrame(columns=["observation_date", sid])
    return result

def refresh_bls(api_key: str | None = None) -> list[dict]:
    """
    Incremental BLS refresh for all series in BLS_SERIES_IDS.
    start_year = max(1990, year of last observation) to overlap by ~1 year
    and capture any revisions.
    Returns list of per-series status dicts.
    """
    results = []
    current_year = datetime.now().year

    for series_id, label in BLS_SERIES_IDS.items():
        t0 = time.time()
        status = {"series_id": series_id, "source": "BLS",
                  "status": "ok", "new_rows": 0, "error": None}
        try:
            csv_path = _sector_csv_path("BLS", f"{series_id}.csv")
            existing = _load_sector_existing(csv_path, series_id)

            if existing.empty or existing["observation_date"].isna().all():
                start_year = 1990
            else:
                last_year  = existing["observation_date"].max().year
                start_year = max(1990, last_year)  # one-year overlap for revisions

            log.info("    BLS  %-20s  [%s]  start=%d", series_id, label, start_year)
            fetched = fetch_bls_series([series_id], start_year, current_year, api_key)
            new_rows = fetched.get(series_id, pd.DataFrame())
            status["new_rows"] = _merge_and_save_sector(existing, new_rows, series_id, csv_path)

        except Exception as exc:
            status["status"] = "error"
            status["error"]  = str(exc)
            log.warning("    BLS  %s: FAILED — %s", series_id, exc)

        status["elapsed"] = round(time.time() - t0, 1)
        results.append(status)
        time.sleep(0.5)   # be polite to BLS API

    return results

# ── BEA Fetcher ───────────────────────────────────────────────────────────────

def fetch_bea_gdp_by_industry(api_key: str,
                               table_name: str = "1",
                               frequency: str = "Q") -> pd.DataFrame:
    """
    Fetch GDPbyIndustry data for ALL years in one call.
    Returns a long-format DataFrame with the raw BEA columns.
    """
    params = {
        "UserID":      api_key,
        "method":      "GetData",
        "DataSetName": "GDPbyIndustry",
        "TableName":   table_name,
        "Frequency":   frequency,
        "Year":        "ALL",
        "ResultFormat":"JSON",
    }
    resp = requests.get(BEA_API_BASE, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    try:
        records = data["BEAAPI"]["Results"]["GDPbyIndustry"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Unexpected BEA response structure: {exc}") from exc

    return pd.DataFrame(records)

def pivot_bea_to_series(raw_df: pd.DataFrame,
                         industry_map: dict[str, str]) -> dict[str, pd.DataFrame]:
    """
    Pivot BEA long-format DataFrame into one DataFrame per industry.
    Filters to industry IDs in industry_map and converts quarterly dates.
    """
    quarter_to_month = {"I": 1, "II": 4, "III": 7, "IV": 10}
    result: dict[str, pd.DataFrame] = {}

    for industry_id, col_name in industry_map.items():
        mask = raw_df["IndustryID"].astype(str).str.strip() == industry_id
        sub  = raw_df[mask].copy()
        if sub.empty:
            log.warning("    BEA  IndustryID '%s' not found in response", industry_id)
            continue

        rows = []
        for _, row in sub.iterrows():
            try:
                quarter     = str(row.get("Quarter", "")).strip()
                month_num   = quarter_to_month.get(quarter)
                if month_num is None:
                    continue
                year        = str(row.get("Year", "")).strip()
                obs_date    = pd.Timestamp(f"{year}-{month_num:02d}-01")
                # DataValue may have commas: "2,345.6"
                raw_val     = str(row.get("DataValue", "")).replace(",", "").strip()
                value       = pd.to_numeric(raw_val, errors="coerce")
                rows.append({"observation_date": obs_date, col_name: value})
            except (ValueError, KeyError):
                continue

        if rows:
            df = (pd.DataFrame(rows)
                  .drop_duplicates("observation_date")
                  .sort_values("observation_date")
                  .reset_index(drop=True))
            result[col_name] = df
        else:
            result[col_name] = pd.DataFrame(columns=["observation_date", col_name])

    return result

def refresh_bea(api_key: str) -> list[dict]:
    """
    BEA refresh: fetches full history (one API call), pivots per industry,
    and incrementally merges into existing CSVs.
    Skips the merge if BEA has no data newer than what's already stored.
    """
    results = []
    t0_total = time.time()

    try:
        log.info("    BEA  Fetching GDPbyIndustry (ALL years) ...")
        raw_df = fetch_bea_gdp_by_industry(api_key)
        industry_dfs = pivot_bea_to_series(raw_df, BEA_INDUSTRY_MAP)
    except Exception as exc:
        log.warning("    BEA  fetch failed — %s", exc)
        # Return one error entry representing the whole BEA call
        return [{"series_id": "BEA_GDPbyIndustry", "source": "BEA",
                  "status": "error", "new_rows": 0,
                  "elapsed": round(time.time() - t0_total, 1), "error": str(exc)}]

    for col_name, new_df in industry_dfs.items():
        t0 = time.time()
        status = {"series_id": col_name, "source": "BEA",
                  "status": "ok", "new_rows": 0, "error": None}
        try:
            csv_path = _sector_csv_path("BEA", f"{col_name}.csv")
            existing = _load_sector_existing(csv_path, col_name)

            # Skip write if BEA has nothing newer than what we have
            if not existing.empty and not new_df.empty:
                existing_max = existing["observation_date"].max()
                new_max      = new_df["observation_date"].max()
                if existing_max >= new_max:
                    log.info("    BEA  %-35s  no new data", col_name)
                    status["new_rows"] = 0
                    results.append(status)
                    status["elapsed"] = round(time.time() - t0, 1)
                    continue

            status["new_rows"] = _merge_and_save_sector(existing, new_df, col_name, csv_path)
            log.info("    BEA  %-35s  +%d rows", col_name, status["new_rows"])

        except Exception as exc:
            status["status"] = "error"
            status["error"]  = str(exc)
            log.warning("    BEA  %s: FAILED — %s", col_name, exc)

        status["elapsed"] = round(time.time() - t0, 1)
        results.append(status)

    return results

# ── World Bank Fetcher ────────────────────────────────────────────────────────

def fetch_worldbank_indicator(indicator_code: str,
                               country: str = "US") -> pd.DataFrame:
    """
    Fetch all years for a World Bank indicator for one country.
    Handles multi-page responses automatically.
    Returns annual DataFrame[observation_date, indicator_code].
    """
    records: list[dict] = []
    page = 1

    while True:
        resp = requests.get(
            f"{WB_API_BASE}/country/{country}/indicator/{indicator_code}",
            params={"format": "json", "per_page": 1000, "page": page},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list) or len(data) < 2:
            break

        meta    = data[0]
        entries = data[1] or []
        records.extend(entries)

        total_pages = meta.get("pages", 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.3)

    rows = []
    for entry in records:
        try:
            year  = str(entry.get("date", "")).strip()
            value = entry.get("value")
            if value is None:
                continue
            obs_date = pd.Timestamp(f"{year}-01-01")
            rows.append({"observation_date": obs_date, indicator_code: float(value)})
        except (ValueError, TypeError):
            continue

    if not rows:
        return pd.DataFrame(columns=["observation_date", indicator_code])

    return (pd.DataFrame(rows)
            .sort_values("observation_date")
            .reset_index(drop=True))

def forward_fill_annual_to_monthly(df_annual: pd.DataFrame,
                                    series_col: str) -> pd.DataFrame:
    """
    Convert an annual DataFrame to monthly by forward-filling.
    Identical pattern to the CAPUTLB50001SQ quarterly→monthly ffill
    in business_env_model.py:59-65, applied to annual data.
    """
    if df_annual.empty:
        return pd.DataFrame(columns=["observation_date", series_col])

    start = df_annual["observation_date"].min()
    end   = pd.Timestamp.today().to_period("M").to_timestamp()

    monthly_idx = pd.DataFrame(
        {"observation_date": pd.date_range(start, end, freq="MS")}
    )
    merged = (monthly_idx
              .merge(df_annual, on="observation_date", how="left")
              .assign(**{series_col: lambda d: d[series_col].ffill()}))
    return merged[["observation_date", series_col]]

def refresh_worldbank(country: str = "US") -> list[dict]:
    """
    World Bank refresh for all indicators in WB_INDICATORS.
    Fetches full annual history, forward-fills to monthly, merges incrementally.
    """
    results = []

    for indicator_code, label in WB_INDICATORS.items():
        t0 = time.time()
        status = {"series_id": indicator_code, "source": "WorldBank",
                  "status": "ok", "new_rows": 0, "error": None}
        try:
            log.info("    WB   %-30s  [%s]", indicator_code, label)
            annual_df  = fetch_worldbank_indicator(indicator_code, country)
            monthly_df = forward_fill_annual_to_monthly(annual_df, indicator_code)

            csv_path = _sector_csv_path("WorldBank", f"{indicator_code}.csv")
            existing = _load_sector_existing(csv_path, indicator_code)

            # Skip if no new annual observations
            if not existing.empty and not annual_df.empty:
                existing_max = existing["observation_date"].max()
                new_max      = monthly_df["observation_date"].max()
                if existing_max >= new_max:
                    log.info("    WB   %-30s  no new data", indicator_code)
                    results.append(status)
                    status["elapsed"] = round(time.time() - t0, 1)
                    continue

            status["new_rows"] = _merge_and_save_sector(
                existing, monthly_df, indicator_code, csv_path
            )
            log.info("    WB   %-30s  +%d rows", indicator_code, status["new_rows"])

        except Exception as exc:
            status["status"] = "error"
            status["error"]  = str(exc)
            log.warning("    WB   %s: FAILED — %s", indicator_code, exc)

        status["elapsed"] = round(time.time() - t0, 1)
        results.append(status)
        time.sleep(0.5)

    return results

# ── Trading Economics Fetcher (stub — requires commercial credentials) ─────────

def refresh_trading_economics(client_key: str, client_secret: str) -> list[dict]:
    """
    Trading Economics refresh stub.
    Requires a paid API plan.  Extend this function with specific indicator
    calls once credentials are available.
    """
    log.info("    TE   Trading Economics refresh not yet implemented (commercial API).")
    return [{"series_id": "TE_stub", "source": "TradingEconomics",
              "status": "skipped", "new_rows": 0, "elapsed": 0.0, "error": None}]
