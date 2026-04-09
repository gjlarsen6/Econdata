"""
crunchbase_apis.py — Crunchbase VC data fetcher for AI, Fintech, Healthcare segments.

Saves incremental weekly snapshot data to data/VentureCapital/ CSVs in the same
additive pattern used by all other data sources in this project.
Data is never deleted — new rows are merged in and existing rows are preserved.

Usage (called from fred_refresh.py with --crunchbase flag):
    import crunchbase_apis
    api_key = crunchbase_apis.load_crunchbase_api_key()
    results = crunchbase_apis.refresh_crunchbase(api_key)

Requires CRUNCHBASE_API_KEY in environment or .env file.
Rate limit: 200 calls/minute — enforced via RATE_LIMIT_SLEEP between each call.
"""

import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent
VC_DATA_DIR = BASE_DIR / "data" / "VentureCapital"

CB_API_BASE      = "https://api.crunchbase.com/v4/data"
RATE_LIMIT_SLEEP = 0.31   # 200 calls/min → 300ms between calls; 0.31s gives margin

SEGMENTS = ["ai", "fintech", "healthcare"]

SEGMENT_QUERIES: dict[str, list[str]] = {
    "ai": [
        "artificial intelligence",
        "generative ai",
        "machine learning",
        "ai infrastructure",
        "computer vision",
        "natural language processing",
    ],
    "fintech": [
        "fintech",
        "payments",
        "digital banking",
        "insurtech",
        "embedded finance",
        "lending",
    ],
    "healthcare": [
        "health care",
        "healthcare",
        "health tech",
        "biotech",
        "medical device",
        "digital health",
    ],
}

ROUND_SEARCH_FIELD_IDS = [
    "identifier",
    "announced_on",
    "funded_organization_identifier",
    "money_raised",
    "investment_type",
    "num_investors",
    "lead_investor_identifiers",
]

ORG_SEARCH_FIELD_IDS = [
    "identifier",
    "categories",
    "category_groups",
    "location_identifiers",
    "short_description",
    "rank_org_company",
    "funding_total",
    "last_funding_at",
    "num_funding_rounds",
    "status",
]

# ── API key ───────────────────────────────────────────────────────────────────

def load_crunchbase_api_key() -> Optional[str]:
    """
    Load CRUNCHBASE_API_KEY from environment (or .env already loaded by
    fred_refresh.py). Returns None if missing — caller decides whether to skip.
    """
    return os.getenv("CRUNCHBASE_API_KEY", "").strip() or None

# ── CSV helpers (mirror sector_apis.py patterns) ──────────────────────────────

def _vc_csv_path(filename: str) -> Path:
    """Return Path to data/VentureCapital/{filename}, creating parent dirs."""
    p = VC_DATA_DIR / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _load_vc_existing(path: Path) -> pd.DataFrame:
    """Load existing VC CSV or return an empty DataFrame."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def _merge_and_save_vc(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
    key_col: str,
    path: Path,
    keep: str = "last",
) -> int:
    """
    Concat existing + new rows, deduplicate on key_col (prefer newer value when
    keep='last'), sort by key_col, save CSV.  Returns net-new row count.
    Existing rows are never deleted — this is always additive.
    """
    if new_rows.empty:
        if not existing.empty:
            existing.to_csv(path, index=False)
        return 0
    combined = (
        pd.concat([existing, new_rows], ignore_index=True)
        .drop_duplicates(subset=[key_col], keep=keep)
        .sort_values(key_col)
        .reset_index(drop=True)
    )
    net_new = max(len(combined) - len(existing), 0)
    combined.to_csv(path, index=False)
    return net_new

# ── Snapshot date ─────────────────────────────────────────────────────────────

def _get_snapshot_date() -> date:
    """
    Return the Monday of the current ISO week.
    Used as the canonical observation_date for all weekly agg rows.
    Idempotent: running twice in the same week produces one row (dedup on date).
    """
    today = date.today()
    return today - timedelta(days=today.weekday())   # weekday() == 0 for Monday

# ── Category resolution ───────────────────────────────────────────────────────

def autocomplete_categories(query: str, api_key: str) -> list[dict]:
    """
    GET /autocompletes?query={query}&collection_ids=categories,category_groups&limit=10
    Returns list of {uuid, permalink, category_name} dicts.
    Returns [] on any error — callers skip gracefully.
    Sleeps RATE_LIMIT_SLEEP after each call.
    """
    url = f"{CB_API_BASE}/autocompletes"
    params = {
        "query":          query,
        "collection_ids": "categories,category_groups",
        "limit":          10,
        "user_key":       api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        entities = data.get("entities", [])
        return [
            {
                "uuid":          e.get("identifier", {}).get("uuid", ""),
                "permalink":     e.get("identifier", {}).get("permalink", ""),
                "category_name": e.get("identifier", {}).get("value", query),
                "collection_id": e.get("identifier", {}).get("collection_id", ""),
            }
            for e in entities
            if e.get("identifier", {}).get("uuid")
        ]
    except Exception as exc:
        log.warning("    CB   autocomplete '%s' failed: %s", query, exc)
        return []
    finally:
        time.sleep(RATE_LIMIT_SLEEP)


def resolve_all_segment_categories(api_key: str) -> dict[str, list[str]]:
    """
    For all 3 segments × up to 6 queries each, call autocomplete and collect UUIDs.
    Deduplicates UUIDs within each segment.
    Saves/updates dim_category.csv (additive — new UUIDs only).
    Returns {segment: [uuid_list]} for use as search predicate values.
    """
    dim_path     = _vc_csv_path("dim_category.csv")
    existing_dim = _load_vc_existing(dim_path)
    existing_uuids: set[str] = (
        set(existing_dim["category_uuid"].tolist())
        if not existing_dim.empty and "category_uuid" in existing_dim.columns
        else set()
    )

    snapshot_date   = _get_snapshot_date().isoformat()
    segment_uuids:  dict[str, list[str]] = {}
    new_dim_rows:   list[dict] = []

    for segment in SEGMENTS:
        queries     = SEGMENT_QUERIES[segment]
        seen_uuids: set[str] = set()
        uuids:      list[str] = []

        for query in queries:
            log.info("    CB   autocomplete segment=%s query='%s'", segment, query)
            results = autocomplete_categories(query, api_key)
            for r in results:
                uuid = r["uuid"]
                if uuid and uuid not in seen_uuids:
                    seen_uuids.add(uuid)
                    uuids.append(uuid)
                    if uuid not in existing_uuids:
                        new_dim_rows.append({
                            "observation_date": snapshot_date,
                            "category_uuid":   uuid,
                            "category_name":   r["category_name"],
                            "segment":         segment,
                        })
                        existing_uuids.add(uuid)

        segment_uuids[segment] = uuids
        log.info("    CB   segment=%s resolved %d unique category UUIDs", segment, len(uuids))

    # Persist new entries to dim_category.csv
    if new_dim_rows:
        new_df = pd.DataFrame(new_dim_rows)
        _merge_and_save_vc(existing_dim, new_df, "category_uuid", dim_path)
        log.info("    CB   dim_category.csv: +%d new entries", len(new_dim_rows))

    return segment_uuids

# ── Funding rounds search ─────────────────────────────────────────────────────

def search_funding_rounds(
    api_key: str,
    category_uuids: list[str],
    start_date: str,
    segment: str,
    max_rounds: int = 500,
) -> pd.DataFrame:
    """
    POST /searches/funding_rounds for all rounds with announced_on >= start_date
    in any of the given category_uuids (OR logic via single includes predicate).

    Paginates using after_id (uuid of last entity) until exhausted or max_rounds reached.
    Returns DataFrame with fact_funding_round schema.
    """
    if not category_uuids:
        return pd.DataFrame()

    url    = f"{CB_API_BASE}/searches/funding_rounds"
    params = {"user_key": api_key}
    # Crunchbase includes operator is OR across values; batch first 20 UUIDs
    uuid_batch = category_uuids[:20]

    body: dict = {
        "field_ids": ROUND_SEARCH_FIELD_IDS,
        "order":     [{"field_id": "announced_on", "sort": "desc"}],
        "query": [
            {
                "type":        "predicate",
                "field_id":    "announced_on",
                "operator_id": "gte",
                "values":      [start_date],
            },
            {
                "type":        "predicate",
                "field_id":    "funded_organization_categories",
                "operator_id": "includes",
                "values":      uuid_batch,
            },
        ],
        "limit": 100,
    }

    snapshot_str = _get_snapshot_date().isoformat()
    all_rows: list[dict] = []
    after_id:  Optional[str] = None
    page_count = 0

    while len(all_rows) < max_rounds:
        if after_id:
            body["after_id"] = after_id

        try:
            resp = requests.post(url, json=body, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("    CB   funding_rounds search page %d failed: %s", page_count, exc)
            break
        finally:
            time.sleep(RATE_LIMIT_SLEEP)

        entities = data.get("entities", [])
        page_count += 1

        if not entities:
            break

        for e in entities:
            props      = e.get("properties", {})
            money      = props.get("money_raised") or {}
            money_usd  = money.get("value_usd") if isinstance(money, dict) else None
            funded_org = props.get("funded_organization_identifier") or {}
            ident      = props.get("identifier") or {}
            lead_inv   = props.get("lead_investor_identifiers") or []
            all_rows.append({
                "observation_date":  snapshot_str,
                "round_uuid":        ident.get("uuid", ""),
                "org_uuid":          funded_org.get("uuid", ""),
                "announced_on":      props.get("announced_on", ""),
                "money_raised_usd":  money_usd,
                "investment_type":   props.get("investment_type", ""),
                "num_investors":     props.get("num_investors", 0),
                "has_lead_investor": 1 if lead_inv else 0,
                "segment":           segment,
            })

        # Pagination: use last entity uuid as after_id cursor
        last_ident = (entities[-1].get("identifier") or {})
        after_id   = last_ident.get("uuid")
        if not after_id or len(entities) < 100:
            break   # reached last page

    log.info("    CB   segment=%s fetched %d rounds (%d pages)", segment, len(all_rows), page_count)
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

# ── Organization search ───────────────────────────────────────────────────────

def search_organizations(
    api_key: str,
    category_uuids: list[str],
    segment: str,
    max_orgs: int = 200,
) -> pd.DataFrame:
    """
    POST /searches/organizations for top companies by rank_org_company (asc)
    in any of the given category_uuids.

    Returns DataFrame with dim_organization schema.
    """
    if not category_uuids:
        return pd.DataFrame()

    url        = f"{CB_API_BASE}/searches/organizations"
    params     = {"user_key": api_key}
    uuid_batch = category_uuids[:20]
    page_limit = min(100, max_orgs)

    body: dict = {
        "field_ids": ORG_SEARCH_FIELD_IDS,
        "order":     [{"field_id": "rank_org_company", "sort": "asc"}],
        "query": [
            {
                "type":        "predicate",
                "field_id":    "categories",
                "operator_id": "includes",
                "values":      uuid_batch,
            }
        ],
        "limit": page_limit,
    }

    snapshot_str = _get_snapshot_date().isoformat()
    all_rows: list[dict] = []
    after_id: Optional[str] = None

    while len(all_rows) < max_orgs:
        if after_id:
            body["after_id"] = after_id

        try:
            resp = requests.post(url, json=body, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("    CB   org search failed: %s", exc)
            break
        finally:
            time.sleep(RATE_LIMIT_SLEEP)

        entities = data.get("entities", [])
        if not entities:
            break

        for e in entities:
            props         = e.get("properties", {})
            ident         = props.get("identifier") or {}
            funding_total = props.get("funding_total") or {}
            all_rows.append({
                "observation_date":  snapshot_str,
                "org_uuid":          ident.get("uuid", ""),
                "org_name":          ident.get("value", ""),
                "rank_org_company":  props.get("rank_org_company"),
                "funding_total_usd": (
                    funding_total.get("value_usd")
                    if isinstance(funding_total, dict) else None
                ),
                "last_funding_at":   props.get("last_funding_at", ""),
                "status":            props.get("status", ""),
                "segment":           segment,
            })

        last_ident = (entities[-1].get("identifier") or {})
        after_id   = last_ident.get("uuid")
        if not after_id or len(entities) < page_limit:
            break

    log.info("    CB   segment=%s fetched %d orgs", segment, len(all_rows))
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

# ── Weekly aggregate metrics ──────────────────────────────────────────────────

def aggregate_segment_metrics(
    rounds_df: pd.DataFrame,
    orgs_df: pd.DataFrame,
    segment: str,
    snapshot_date: date,
) -> dict:
    """
    Compute one weekly snapshot row of segment metrics from raw rounds and orgs.
    Returns a dict ready to append to agg_{segment}_weekly.csv.
    """
    row: dict = {"observation_date": snapshot_date.isoformat()}

    row["company_count"] = (
        orgs_df["org_uuid"].nunique()
        if not orgs_df.empty and "org_uuid" in orgs_df.columns
        else 0
    )

    if not rounds_df.empty and "money_raised_usd" in rounds_df.columns:
        r = rounds_df.copy()
        r["money_raised_usd"] = pd.to_numeric(r["money_raised_usd"], errors="coerce")
        row["round_count"]           = len(r)
        row["capital_raised_usd"]    = float(r["money_raised_usd"].sum(skipna=True))
        row["median_round_size_usd"] = float(r["money_raised_usd"].median(skipna=True))
        row["lead_investor_count"]   = int(
            r["has_lead_investor"].sum()
            if "has_lead_investor" in r.columns else 0
        )
    else:
        row["round_count"]           = 0
        row["capital_raised_usd"]    = 0.0
        row["median_round_size_usd"] = float("nan")
        row["lead_investor_count"]   = 0

    return row

# ── Main entry point ──────────────────────────────────────────────────────────

def refresh_crunchbase(api_key: str) -> list[dict]:
    """
    Main orchestrator. Called from fred_refresh.py's refresh_crunchbase_data().

    Steps:
      1. Resolve all segment category UUIDs via autocomplete → dim_category.csv
      2. For each segment:
         a. Search funding rounds (last 90 days) → fact_funding_round.csv
         b. Search top organizations → dim_organization.csv
         c. Compute weekly aggregate metrics → agg_{segment}_weekly.csv
      3. Return list[dict] status per segment (same shape as sector_apis results).

    The snapshot date is the Monday of the current week — idempotent across
    multiple runs in the same week (dedup on observation_date).
    """
    snapshot_date    = _get_snapshot_date()
    # 90-day lookback for rounds gives meaningful per-week metrics without
    # excessive API calls; each weekly run refreshes the trailing quarter
    round_start_date = (snapshot_date - timedelta(days=90)).isoformat()

    results: list[dict] = []

    # ── Step 1: Resolve categories ────────────────────────────────────────────
    log.info("    CB   Resolving segment categories via autocomplete ...")
    try:
        segment_uuids = resolve_all_segment_categories(api_key)
    except Exception as exc:
        log.warning("    CB   Category resolution failed: %s", exc)
        segment_uuids = {s: [] for s in SEGMENTS}

    # Load shared dimension/fact tables once
    org_dim_path    = _vc_csv_path("dim_organization.csv")
    round_fact_path = _vc_csv_path("fact_funding_round.csv")
    existing_orgs   = _load_vc_existing(org_dim_path)
    existing_rounds = _load_vc_existing(round_fact_path)

    # ── Step 2: Per-segment data collection ──────────────────────────────────
    for segment in SEGMENTS:
        t0 = time.time()
        status: dict = {
            "series_id": f"vc_{segment}_weekly",
            "source":    "Crunchbase",
            "status":    "ok",
            "new_rows":  0,
            "error":     None,
        }
        try:
            uuids = segment_uuids.get(segment, [])
            if not uuids:
                log.warning(
                    "    CB   segment=%s: no category UUIDs resolved — skipping", segment
                )
                status["status"] = "skipped"
                status["error"]  = "no category UUIDs resolved"
                status["elapsed"] = round(time.time() - t0, 1)
                results.append(status)
                continue

            # 2a: Fetch funding rounds
            log.info(
                "    CB   segment=%s: fetching funding rounds (start=%s) ...",
                segment, round_start_date,
            )
            rounds_df = search_funding_rounds(
                api_key, uuids, round_start_date, segment
            )

            # 2b: Fetch organizations
            log.info("    CB   segment=%s: fetching organizations ...", segment)
            orgs_df = search_organizations(api_key, uuids, segment)

            # 2c: Compute weekly aggregate row
            metrics    = aggregate_segment_metrics(rounds_df, orgs_df, segment, snapshot_date)
            agg_path   = _vc_csv_path(f"agg_{segment}_weekly.csv")
            existing_agg = _load_vc_existing(agg_path)
            new_agg_df   = pd.DataFrame([metrics])
            status["new_rows"] = _merge_and_save_vc(
                existing_agg, new_agg_df, "observation_date", agg_path
            )
            log.info(
                "    CB   segment=%s agg: +%d rows  "
                "(round_count=%d  capital=$%.0fM)",
                segment, status["new_rows"],
                metrics.get("round_count", 0),
                (metrics.get("capital_raised_usd", 0) or 0) / 1_000_000,
            )

            # Update dim_organization (dedup on org_uuid, keep latest snapshot)
            if not orgs_df.empty:
                _merge_and_save_vc(
                    existing_orgs, orgs_df, "org_uuid", org_dim_path, keep="last"
                )
                existing_orgs = _load_vc_existing(org_dim_path)

            # Append fact_funding_round (dedup on round_uuid)
            if not rounds_df.empty:
                _merge_and_save_vc(
                    existing_rounds, rounds_df, "round_uuid", round_fact_path
                )
                existing_rounds = _load_vc_existing(round_fact_path)

        except Exception as exc:
            status["status"] = "error"
            status["error"]  = str(exc)
            log.warning("    CB   segment=%s FAILED: %s", segment, exc)

        status["elapsed"] = round(time.time() - t0, 1)
        results.append(status)

    return results
