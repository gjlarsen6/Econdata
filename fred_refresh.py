"""
fred_refresh.py — Weekly FRED Data Refresh + Model Update + Summary Table

Usage:
    python3 fred_refresh.py

Reads FRED_API_KEY from environment or a .env file in this directory.
Designed to be invoked weekly (e.g., via cron: 0 8 * * 1 python3 /path/fred_refresh.py).

What it does:
  1. Loads fred_ingestion_map_full_production.json for the series catalogue.
  2. For every series, calls the FRED observations API and appends any rows
     newer than the last date already in the local CSV — full history is
     re-fetched on first run (no local file yet).
  3. Saves updated CSVs; creates the file + directory if neither exists.
  4. Re-trains all four LightGBM model scripts (business_env, consumer_demand,
     cost_of_capital, risk).  Each script saves a results JSON to outputs/.
  5. Reads the results JSONs and prints a unified summary table.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from tabulate import tabulate

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

INGESTION_MAP_PATH = DATA_DIR / "fred_ingestion_map_full_production.json"
FRED_API_BASE      = "https://api.stlouisfed.org/fred/series/observations"
API_RATE_SLEEP     = 0.6   # seconds between FRED calls (stays under 120 req/min)

# Map each series_id → (directory, filename) for the local CSV
# Preserves the existing naming conventions used in the project.
SERIES_FILE_MAP: dict[str, tuple[str, str]] = {
    # BusinessEnvironment
    "INDPRO":          ("BusinessEnvironment", "INDPRO.csv"),
    "TCU":             ("BusinessEnvironment", "TCU_capacityutilization.csv"),
    "CAPUTLB50001SQ":  ("BusinessEnvironment", "CAPUTLB50001SQ.csv"),
    "PAYEMS":          ("BusinessEnvironment", "Payroll_PAYEMS.csv"),
    # ConsumerDemand
    "DSPIC96":         ("ConsumerDemand", "DSPIC96.csv"),
    "PCE":             ("ConsumerDemand", "PCE.csv"),
    "PCEPILFE":        ("ConsumerDemand", "PersConsume_noFoodEnergyPCEPILFE.csv"),
    "RSAFS":           ("ConsumerDemand", "RSAFS.csv"),
    "RRSFS":           ("ConsumerDemand", "RealRetailandFoodSalesRRSFS.csv"),
    "UMCSENT":         ("ConsumerDemand", "UMCSENT.csv"),
    # CostOfCapital
    "DFF":             ("CostOfCapital", "DFF.csv"),
    "DPRIME":          ("CostOfCapital", "DPRIME.csv"),
    "FEDFUNDS":        ("CostOfCapital", "FEDFUNDS.csv"),
    "PRIME":           ("CostOfCapital", "PRIME.csv"),
    "T10Y2Y":          ("CostOfCapital", "T10Y2Y.csv"),
    "T10Y3M":          ("CostOfCapital", "T10Y3M.csv"),
    # RiskLeadingInd
    "RECPROUSM156N":   ("RiskLeadingInd", "RECPROUSM156N.csv"),
}

# Series that appear in two directories (UMCSENT is in both ConsumerDemand and RiskLeadingInd)
SERIES_EXTRA_COPIES: dict[str, list[tuple[str, str]]] = {
    "UMCSENT": [("RiskLeadingInd", "UMCSENT.csv")],
}

# Sector data directory and script
SECTOR_DATA_DIR       = DATA_DIR / "Sector"
SECTOR_MODEL_SCRIPT   = "sector_model.py"
SECTOR_RESULTS_PATTERN = "results_sector_*.json"

# VentureCapital (Crunchbase) data directory and script
VC_DATA_DIR        = DATA_DIR / "VentureCapital"
VC_MODEL_SCRIPT    = "vc_model.py"
VC_RESULTS_PATTERN = "results_vc_*.json"

# LightGBM model scripts to run after data refresh
MODEL_SCRIPTS = [
    "business_env_model.py",
    "consumer_demand_model.py",
    "cost_of_capital_model.py",
    "risk_model.py",
]

# Results JSON written by each model script
RESULTS_FILES = [
    OUTPUT_DIR / "results_business_env.json",
    OUTPUT_DIR / "results_consumer_demand.json",
    OUTPUT_DIR / "results_cost_of_capital.json",
    OUTPUT_DIR / "results_risk.json",
]

# ── API key ───────────────────────────────────────────────────────────────────

def _load_api_key() -> str:
    # Try .env file first
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            # Parse manually
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("FRED_API_KEY"):
                    _, _, val = line.partition("=")
                    os.environ["FRED_API_KEY"] = val.strip().strip('"').strip("'")

    key = os.getenv("FRED_API_KEY", "").strip()
    if not key:
        log.error(
            "FRED_API_KEY not set.\n"
            "  • Set it as an environment variable:  export FRED_API_KEY=your_key\n"
            "  • Or add FRED_API_KEY=your_key to a .env file in this directory.\n"
            "  • Free keys: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        sys.exit(1)
    return key

# ── FRED API call ─────────────────────────────────────────────────────────────

def fetch_observations(series_id: str, api_key: str,
                        observation_start: str | None = None) -> pd.DataFrame:
    """
    Call the FRED observations endpoint and return a tidy DataFrame
    with columns [observation_date, <series_id>].
    Missing FRED values ('.') are converted to NaN.
    """
    params: dict = {
        "series_id": series_id,
        "api_key":   api_key,
        "file_type": "json",
    }
    if observation_start:
        params["observation_start"] = observation_start

    resp = requests.get(FRED_API_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    obs = data.get("observations", [])
    if not obs:
        return pd.DataFrame(columns=["observation_date", series_id])

    df = pd.DataFrame(obs)[["date", "value"]].rename(columns={"date": "observation_date"})
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df[series_id] = pd.to_numeric(df["value"].replace(".", float("nan")), errors="coerce")
    return df[["observation_date", series_id]]

# ── CSV helpers ───────────────────────────────────────────────────────────────

def _csv_path(directory: str, filename: str) -> Path:
    p = DATA_DIR / directory / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def load_existing(path: Path, series_id: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["observation_date", series_id])
    df = pd.read_csv(path, parse_dates=["observation_date"])
    if series_id not in df.columns:
        df[series_id] = float("nan")
    return df[["observation_date", series_id]]

def merge_and_save(existing: pd.DataFrame, new_rows: pd.DataFrame,
                    series_id: str, path: Path) -> int:
    """
    Combine existing + new rows.  Prefer new values where dates overlap
    (handles FRED revisions).  Returns the number of net-new date rows added.
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
    new_count = len(combined) - len(existing)
    combined.to_csv(path, index=False)
    return max(new_count, 0)

# ── Per-series refresh ────────────────────────────────────────────────────────

def refresh_series(series_id: str, api_key: str,
                    ingestion_map: dict) -> dict:
    """
    Refresh one FRED series.  Returns a status dict.
    """
    t0 = time.time()
    result = {"series_id": series_id, "status": "ok",
              "new_rows": 0, "error": None}

    try:
        directory, filename = SERIES_FILE_MAP.get(
            series_id,
            ("NewSeries", f"{series_id}.csv"),   # fallback for unknown series
        )
        primary_path = _csv_path(directory, filename)

        # Determine incremental start date
        existing = load_existing(primary_path, series_id)
        if existing.empty or existing["observation_date"].isna().all():
            obs_start = None          # full history
        else:
            last_dt   = existing["observation_date"].max()
            obs_start = (last_dt - timedelta(days=30)).strftime("%Y-%m-%d")
            # Go back 30 days to capture any revisions to the most recent data

        freq_label = ingestion_map.get(series_id, {}).get("frequency", "?")
        log.info(f"  {series_id:<18} [{freq_label:<12}]  start={obs_start or 'full'}")

        new_rows = fetch_observations(series_id, api_key, obs_start)
        time.sleep(API_RATE_SLEEP)

        result["new_rows"] = merge_and_save(existing, new_rows, series_id, primary_path)

        # Write extra copies (e.g., UMCSENT → RiskLeadingInd)
        for extra_dir, extra_file in SERIES_EXTRA_COPIES.get(series_id, []):
            extra_path    = _csv_path(extra_dir, extra_file)
            extra_existing = load_existing(extra_path, series_id)
            merge_and_save(extra_existing, new_rows, series_id, extra_path)

    except Exception as exc:
        result["status"] = "error"
        result["error"]  = str(exc)
        log.warning(f"  {series_id}: FAILED — {exc}")

    result["elapsed"] = round(time.time() - t0, 1)
    return result

# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "FRED Weekly Refresh + optional Sector API refresh.\n"
            "FRED data refresh always runs. Sector APIs are opt-in via --sector."
        )
    )
    p.add_argument(
        "--sector",
        nargs="*",
        choices=["bls", "bea", "worldbank", "tradingeconomics", "all"],
        default=[],
        metavar="SOURCE",
        help=(
            "Sector APIs to refresh (bls, bea, worldbank, tradingeconomics, all). "
            "E.g.: --sector bls worldbank  or  --sector all. "
            "FRED refresh always runs regardless of this flag."
        ),
    )
    p.add_argument(
        "--skip-models",
        action="store_true",
        default=False,
        help="Skip all model re-training (data refresh only).",
    )
    p.add_argument(
        "--crunchbase",
        action="store_true",
        default=False,
        help=(
            "Enable Crunchbase VC data refresh and modeling (AI, Fintech, Healthcare). "
            "Requires CRUNCHBASE_API_KEY in environment or .env file. "
            "FRED refresh always runs regardless of this flag."
        ),
    )
    return p.parse_args()

# ── Model runner ──────────────────────────────────────────────────────────────

def run_model_script(script: str) -> dict:
    """Run a model script as a subprocess; return status + elapsed time."""
    log.info(f"  Running {script} ...")
    t0  = time.time()
    ret = subprocess.run(
        [sys.executable, str(BASE_DIR / script)],
        capture_output=True, text=True, cwd=str(BASE_DIR),
    )
    elapsed = round(time.time() - t0, 1)
    ok = ret.returncode == 0
    if not ok:
        log.warning(f"  {script} exited {ret.returncode}:\n{ret.stderr[-800:]}")
    return {"script": script, "ok": ok,
            "elapsed": elapsed, "stderr": ret.stderr}

# ── Sector data refresh ───────────────────────────────────────────────────────

def refresh_sector_data(enabled_sources: list[str]) -> list[dict]:
    """
    Import sector_apis and call the appropriate refresh function for each
    enabled source.  Missing API keys cause a warning + skip rather than
    a crash — FRED pipeline always completes regardless.

    Returns a flat list of per-series status dicts (same shape as
    refresh_series() results) for the refresh-status display table.
    """
    try:
        import sector_apis  # lazy import — not required for FRED-only runs
    except ImportError:
        log.error("sector_apis.py not found — sector refresh unavailable.")
        return []

    keys = sector_apis.load_sector_api_keys()

    if "all" in enabled_sources:
        enabled_sources = ["bls", "bea", "worldbank", "tradingeconomics"]

    results: list[dict] = []

    if "bls" in enabled_sources:
        log.info("  Refreshing BLS sector data ...")
        results.extend(sector_apis.refresh_bls(api_key=keys.get("BLS_API_KEY")))

    if "bea" in enabled_sources:
        bea_key = keys.get("BEA_API_KEY")
        if not bea_key:
            log.warning("  BEA skipped — BEA_API_KEY not set in environment or .env file.")
        else:
            log.info("  Refreshing BEA sector data ...")
            results.extend(sector_apis.refresh_bea(api_key=bea_key))

    if "worldbank" in enabled_sources:
        log.info("  Refreshing World Bank sector data ...")
        results.extend(sector_apis.refresh_worldbank())

    if "tradingeconomics" in enabled_sources:
        te_key    = keys.get("TE_CLIENT_KEY")
        te_secret = keys.get("TE_CLIENT_SECRET")
        if not te_key or not te_secret:
            log.warning(
                "  Trading Economics skipped — TE_CLIENT_KEY / TE_CLIENT_SECRET not set."
            )
        else:
            log.info("  Refreshing Trading Economics sector data ...")
            results.extend(
                sector_apis.refresh_trading_economics(te_key, te_secret)
            )

    return results


def detect_new_sector_csvs_needing_models() -> dict[str, list[str]]:
    """
    Scan data/Sector/{BLS,BEA,WorldBank}/ for CSV files.
    For each CSV, check whether a corresponding .joblib file already exists.
    Returns {source_key: [series_ids_without_models]}.
    """
    source_map = {"BLS": "bls", "BEA": "bea", "WorldBank": "worldbank"}
    needs: dict[str, list[str]] = {}

    for dir_name, source_key in source_map.items():
        source_dir = SECTOR_DATA_DIR / dir_name
        if not source_dir.exists():
            continue
        missing = []
        for csv_file in sorted(source_dir.glob("*.csv")):
            stem       = csv_file.stem
            model_path = OUTPUT_DIR / f"sector_{source_key}_model_{stem}.joblib"
            if not model_path.exists():
                missing.append(stem)
        if missing:
            needs[source_key] = missing

    return needs


def run_sector_model_script(sources: list[str]) -> dict:
    """Run sector_model.py as a subprocess for the given source(s)."""
    log.info("  Running %s --source %s ...", SECTOR_MODEL_SCRIPT, " ".join(sources))
    t0  = time.time()
    ret = subprocess.run(
        [sys.executable, str(BASE_DIR / SECTOR_MODEL_SCRIPT), "--source"] + sources,
        capture_output=True, text=True, cwd=str(BASE_DIR),
    )
    elapsed = round(time.time() - t0, 1)
    ok = ret.returncode == 0
    if not ok:
        log.warning(
            "  %s exited %d:\n%s",
            SECTOR_MODEL_SCRIPT, ret.returncode, ret.stderr[-800:],
        )
    return {"script": SECTOR_MODEL_SCRIPT, "ok": ok,
            "elapsed": elapsed, "stderr": ret.stderr}

# ── Crunchbase VC data refresh ────────────────────────────────────────────────

def refresh_crunchbase_data(api_key: str | None) -> list[dict]:
    """
    Import crunchbase_apis and call refresh_crunchbase().
    Missing API key or missing module causes warning + empty return rather than
    a crash — FRED pipeline always completes regardless.

    Pattern mirrors refresh_sector_data() exactly.
    """
    try:
        import crunchbase_apis  # lazy import — not required for non-crunchbase runs
    except ImportError:
        log.error("crunchbase_apis.py not found — Crunchbase VC refresh unavailable.")
        return []

    if not api_key:
        log.warning(
            "  Crunchbase skipped — CRUNCHBASE_API_KEY not set in environment or .env file."
        )
        return []

    log.info("  Refreshing Crunchbase VC data (AI, Fintech, Healthcare) ...")
    return crunchbase_apis.refresh_crunchbase(api_key)


def detect_new_vc_csvs_needing_models() -> list[str]:
    """
    Scan data/VentureCapital/ for agg_{segment}_weekly.csv files.
    Return list of segment names that have a CSV but no corresponding .joblib model.
    """
    metric_cols = [
        "company_count", "round_count", "capital_raised_usd",
        "median_round_size_usd", "lead_investor_count",
    ]
    missing = []
    for segment in ("ai", "fintech", "healthcare"):
        csv_path = VC_DATA_DIR / f"agg_{segment}_weekly.csv"
        if not csv_path.exists():
            continue
        model_exists = any(
            (OUTPUT_DIR / f"vc_model_{segment}_{col}.joblib").exists()
            for col in metric_cols
        )
        if not model_exists:
            missing.append(segment)
    return missing


def run_vc_model_script(segments: list[str]) -> dict:
    """Run vc_model.py as a subprocess for the given segment(s)."""
    log.info("  Running %s --segment %s ...", VC_MODEL_SCRIPT, " ".join(segments))
    t0  = time.time()
    ret = subprocess.run(
        [sys.executable, str(BASE_DIR / VC_MODEL_SCRIPT), "--segment"] + segments,
        capture_output=True, text=True, cwd=str(BASE_DIR),
    )
    elapsed = round(time.time() - t0, 1)
    ok = ret.returncode == 0
    if not ok:
        log.warning(
            "  %s exited %d:\n%s",
            VC_MODEL_SCRIPT, ret.returncode, ret.stderr[-800:],
        )
    return {"script": VC_MODEL_SCRIPT, "ok": ok,
            "elapsed": elapsed, "stderr": ret.stderr}

# ── Summary table ─────────────────────────────────────────────────────────────

def _trend(fc_list: list) -> str:
    if len(fc_list) < 2:
        return " —"
    delta = fc_list[-1]["mid"] - fc_list[0]["mid"]
    rel   = abs(delta) / (abs(fc_list[0]["mid"]) + 1e-9)
    if rel < 0.005:
        return " →"
    return " ↑" if delta > 0 else " ↓"

def _fmt(val, decimals=2) -> str:
    if val is None:
        return "—"
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:,.2f}M"
    if abs(val) >= 1_000:
        return f"{val:,.{decimals}f}"
    return f"{val:.{decimals}f}"

def print_summary_table(results_files: list[Path], title: str = "MACRO MODEL SUMMARY — FRED Weekly Refresh"):
    rows = []
    for rf in results_files:
        if not rf.exists():
            log.warning(f"Results file not found: {rf}")
            continue
        with open(rf) as fh:
            payload = json.load(fh)

        group   = payload["group"]
        run_at  = payload["run_at"]

        for s in payload["series"]:
            fc    = s["forecast"]
            val   = s.get("validation", {})
            unit  = s["unit"]

            rows.append({
                "Group":        group,
                "Series":       s["series_id"],
                "Label":        s["label"][:28],
                "Unit":         unit[:14],
                "Last Date":    s["last_date"] or "—",
                "Last Value":   _fmt(s["last_value"]),
                "Val MAE":      _fmt(val.get("mae"),  3) if val else "—",
                "Val R²":       _fmt(val.get("r2"),   4) if val else "—",
                "+1M":          _fmt(fc[0]["mid"])  if len(fc) > 0 else "—",
                "+3M":          _fmt(fc[2]["mid"])  if len(fc) > 2 else "—",
                "+6M":          _fmt(fc[5]["mid"])  if len(fc) > 5 else "—",
                "+12M":         _fmt(fc[-1]["mid"]) if fc else "—",
                "Trend":        _trend(fc),
                "Run At":       run_at,
            })

    if not rows:
        log.error("No results to display.")
        return

    # Sort by group then series
    rows.sort(key=lambda r: (r["Group"], r["Series"]))

    print("\n")
    print("=" * 130)
    print(f"  {title}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 130)
    print(tabulate(rows, headers="keys", tablefmt="rounded_outline",
                   stralign="right", numalign="right"))
    print()
    print("  Columns: +1M/+3M/+6M/+12M = forecast midpoints at those horizons")
    print("  Trend  : ↑ rising  ↓ falling  → stable (<0.5% change over 12 months)")
    print("  Val R² : negative values are normal for trending non-stationary series")
    print("           (MAE is the more meaningful metric for those)")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    log.info("=" * 65)
    log.info("FRED Weekly Refresh starting")
    if args.sector:
        log.info("Sector APIs enabled: %s", args.sector)
    if args.crunchbase:
        log.info("Crunchbase VC refresh enabled (AI, Fintech, Healthcare)")
    if args.skip_models:
        log.info("--skip-models: model re-training disabled")
    log.info("=" * 65)

    api_key = _load_api_key()

    # Load ingestion map
    with open(INGESTION_MAP_PATH) as fh:
        ingestion_map_raw = json.load(fh)
    series_catalogue = ingestion_map_raw["series"]   # {series_id: metadata}

    # Compute total step count based on enabled options
    total_steps = 3
    if args.sector:
        total_steps += 2
    if args.crunchbase:
        total_steps += 2

    # ── Step 1: Refresh all FRED series ──────────────────────────────────────
    log.info("\n[1/%d] Refreshing FRED data (%d series) ...",
             total_steps, len(series_catalogue))
    refresh_results = []
    for series_id in series_catalogue:
        r = refresh_series(series_id, api_key, series_catalogue)
        refresh_results.append(r)

    # Detect FRED series with no model group mapping
    known_dirs = {"BusinessEnvironment", "ConsumerDemand",
                   "CostOfCapital", "RiskLeadingInd"}
    for series_id in series_catalogue:
        directory, _ = SERIES_FILE_MAP.get(series_id, ("NewSeries", ""))
        if directory not in known_dirs:
            log.warning(
                f"  Series {series_id} maps to new directory '{directory}'. "
                "Add it to a model script and to SERIES_FILE_MAP."
            )

    # Print FRED refresh status
    refresh_table = [
        {
            "Series":   r["series_id"],
            "Status":   r["status"],
            "New Rows": r["new_rows"],
            "Elapsed":  f"{r['elapsed']}s",
            "Error":    (r["error"] or "")[:60],
        }
        for r in refresh_results
    ]
    print("\n")
    print(tabulate(refresh_table, headers="keys",
                   tablefmt="rounded_outline", stralign="right"))

    ok_count  = sum(1 for r in refresh_results if r["status"] == "ok")
    err_count = len(refresh_results) - ok_count
    log.info("FRED data refresh complete: %d OK  %d errors", ok_count, err_count)

    # ── Step 1b: Optional sector data refresh ─────────────────────────────────
    sector_refresh_results: list[dict] = []
    if args.sector:
        sector_step = 2
        log.info("\n[%d/%d] Refreshing sector API data (%s) ...",
                 sector_step, total_steps, args.sector)
        sector_refresh_results = refresh_sector_data(args.sector)

        if sector_refresh_results:
            sector_table = [
                {
                    "Series":   r.get("series_id", "?"),
                    "Source":   r.get("source", "?"),
                    "Status":   r["status"],
                    "New Rows": r.get("new_rows", 0),
                    "Elapsed":  f"{r.get('elapsed', 0)}s",
                    "Error":    (r.get("error") or "")[:60],
                }
                for r in sector_refresh_results
            ]
            print("\n")
            print(tabulate(sector_table, headers="keys",
                           tablefmt="rounded_outline", stralign="right"))

        sector_ok  = sum(1 for r in sector_refresh_results if r["status"] == "ok")
        sector_err = len(sector_refresh_results) - sector_ok
        log.info("Sector data refresh complete: %d OK  %d errors/skipped",
                 sector_ok, sector_err)

    # ── Step 1c: Optional Crunchbase VC data refresh ──────────────────────────
    crunchbase_refresh_results: list[dict] = []
    if args.crunchbase:
        cb_step = 2 + (1 if args.sector else 0)
        log.info("\n[%d/%d] Refreshing Crunchbase VC data ...", cb_step, total_steps)
        cb_api_key = os.getenv("CRUNCHBASE_API_KEY", "").strip() or None
        crunchbase_refresh_results = refresh_crunchbase_data(cb_api_key)

        if crunchbase_refresh_results:
            cb_table = [
                {
                    "Series":   r.get("series_id", "?"),
                    "Source":   r.get("source", "?"),
                    "Status":   r["status"],
                    "New Rows": r.get("new_rows", 0),
                    "Elapsed":  f"{r.get('elapsed', 0)}s",
                    "Error":    (r.get("error") or "")[:60],
                }
                for r in crunchbase_refresh_results
            ]
            print("\n")
            print(tabulate(cb_table, headers="keys",
                           tablefmt="rounded_outline", stralign="right"))

        cb_ok  = sum(1 for r in crunchbase_refresh_results if r["status"] == "ok")
        cb_err = len(crunchbase_refresh_results) - cb_ok
        log.info("Crunchbase VC refresh complete: %d OK  %d errors/skipped",
                 cb_ok, cb_err)

    # ── Step 2: Re-train FRED LightGBM models ────────────────────────────────
    fred_step = 2 + (1 if args.sector else 0) + (1 if args.crunchbase else 0)
    model_results: list[dict] = []

    if args.skip_models:
        log.info("\n[%d/%d] Skipping FRED model re-training (--skip-models).",
                 fred_step, total_steps)
    else:
        log.info("\n[%d/%d] Re-training FRED LightGBM models ...", fred_step, total_steps)
        for script in MODEL_SCRIPTS:
            mr = run_model_script(script)
            model_results.append(mr)
            status = "OK" if mr["ok"] else "FAILED"
            log.info("  %-35s  %s  (%ss)", script, status, mr["elapsed"])

    # ── Step 2b: Train sector models (create new ones if needed) ─────────────
    sector_model_result: dict | None = None
    if args.sector and not args.skip_models:
        new_series = detect_new_sector_csvs_needing_models()
        if new_series:
            log.info("  New sector series without models: %s", new_series)

        # Always re-train sector models for all enabled sources
        enabled_normalized = (
            ["bls", "bea", "worldbank"]
            if "all" in args.sector
            else [s for s in args.sector
                  if s in ("bls", "bea", "worldbank")]
        )
        if enabled_normalized:
            sector_model_step = fred_step + 1
            log.info("\n[%d/%d] Training sector LightGBM models (%s) ...",
                     sector_model_step, total_steps, enabled_normalized)
            sector_model_result = run_sector_model_script(enabled_normalized)
            status = "OK" if sector_model_result["ok"] else "FAILED"
            log.info("  %-35s  %s  (%ss)",
                     SECTOR_MODEL_SCRIPT, status, sector_model_result["elapsed"])

    # ── Step 2c: Train VC models (create new ones if needed) ─────────────────
    vc_model_result: dict | None = None
    if args.crunchbase and not args.skip_models:
        new_vc = detect_new_vc_csvs_needing_models()
        if new_vc:
            log.info("  New VC segments without models: %s", new_vc)

        vc_model_step = fred_step + (1 if args.sector else 0) + 1
        log.info("\n[%d/%d] Training VC LightGBM models ...", vc_model_step, total_steps)
        vc_model_result = run_vc_model_script(["all"])
        status = "OK" if vc_model_result["ok"] else "FAILED"
        log.info("  %-35s  %s  (%ss)",
                 VC_MODEL_SCRIPT, status, vc_model_result["elapsed"])

    # ── Step 3: Print unified summary table ───────────────────────────────────
    summary_step = total_steps
    log.info("\n[%d/%d] Building unified summary table ...",
             summary_step, total_steps)

    all_results_files = list(RESULTS_FILES)
    if args.sector:
        sector_results = sorted(OUTPUT_DIR.glob(SECTOR_RESULTS_PATTERN))
        all_results_files.extend(sector_results)
    if args.crunchbase:
        vc_results = sorted(OUTPUT_DIR.glob(VC_RESULTS_PATTERN))
        all_results_files.extend(vc_results)

    if args.sector and args.crunchbase:
        table_title = "MACRO + SECTOR + VC MODEL SUMMARY — FRED Weekly Refresh"
    elif args.sector:
        table_title = "MACRO + SECTOR MODEL SUMMARY — FRED Weekly Refresh"
    elif args.crunchbase:
        table_title = "MACRO + VC MODEL SUMMARY — FRED Weekly Refresh"
    else:
        table_title = "MACRO MODEL SUMMARY — FRED Weekly Refresh"
    print_summary_table(all_results_files, title=table_title)

    # Final status
    all_fred_ok   = all(mr["ok"] for mr in model_results)
    all_sector_ok = (sector_model_result is None or sector_model_result["ok"])
    all_vc_ok     = (vc_model_result is None or vc_model_result["ok"])
    all_ok        = all_fred_ok and all_sector_ok and all_vc_ok and err_count == 0
    extras = []
    if args.sector:
        extras.append(f"{len(sector_refresh_results)} sector series refreshed")
    if args.crunchbase:
        extras.append(f"{len(crunchbase_refresh_results)} VC segments refreshed")
    log.info(
        "\nRefresh %s  (%d FRED series refreshed, %d FRED models updated%s)",
        "COMPLETED SUCCESSFULLY" if all_ok else "COMPLETED WITH WARNINGS",
        ok_count,
        sum(1 for m in model_results if m["ok"]),
        (", " + ", ".join(extras)) if extras else "",
    )

    # Write run-log entry
    log_entry = {
        "run_at":                       datetime.now().isoformat(timespec="seconds"),
        "series_ok":                    ok_count,
        "series_errors":                err_count,
        "models_ok":                    sum(1 for m in model_results if m["ok"]),
        "models_failed":                sum(1 for m in model_results if not m["ok"]),
        "sector_sources_enabled":       args.sector,
        "sector_series_refreshed":      len(sector_refresh_results),
        "sector_model_ok":              sector_model_result["ok"] if sector_model_result else None,
        "crunchbase_enabled":           args.crunchbase,
        "crunchbase_segments_refreshed":len(crunchbase_refresh_results),
        "vc_model_ok":                  vc_model_result["ok"] if vc_model_result else None,
        "refresh_details":              refresh_results,
        "sector_refresh_details":       sector_refresh_results,
        "crunchbase_refresh_details":   crunchbase_refresh_results,
    }
    log_path = OUTPUT_DIR / "refresh_log.json"
    existing_log: list = []
    if log_path.exists():
        try:
            existing_log = json.loads(log_path.read_text())
            if not isinstance(existing_log, list):
                existing_log = [existing_log]
        except Exception:
            existing_log = []
    existing_log.append(log_entry)
    log_path.write_text(json.dumps(existing_log[-52:], indent=2))
    log.info("Run log appended → %s", log_path)


if __name__ == "__main__":
    main()
