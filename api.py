"""
api.py — Read-only FastAPI server for macroeconomic forecast results.

Reads pre-computed LightGBM outputs from the outputs/ directory.
No authentication, no writes, no model retraining.

Run:
    python3 api.py
    uvicorn api:app --reload --port 8100 --no-server-header

Interactive docs:
    http://localhost:8100/docs   (Swagger UI)
    http://localhost:8100/redoc  (ReDoc)
"""

from __future__ import annotations

import json
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# series_id path parameters must be alphanumeric + underscores, max 30 chars
_SERIES_ID_RE = re.compile(r"^[A-Z0-9_]{1,30}$")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"

REQUIRED_RESULTS: dict[str, Path] = {
    "business_env":    OUTPUT_DIR / "results_business_env.json",
    "consumer_demand": OUTPUT_DIR / "results_consumer_demand.json",
    "cost_of_capital": OUTPUT_DIR / "results_cost_of_capital.json",
    "risk":            OUTPUT_DIR / "results_risk.json",
}

OPTIONAL_RESULTS: dict[str, Path] = {
    "sector_bls":       OUTPUT_DIR / "results_sector_bls.json",
    "sector_bea":       OUTPUT_DIR / "results_sector_bea.json",
    "sector_worldbank": OUTPUT_DIR / "results_sector_worldbank.json",
    "vc_ai":            OUTPUT_DIR / "results_vc_ai.json",
    "vc_fintech":       OUTPUT_DIR / "results_vc_fintech.json",
    "vc_healthcare":    OUTPUT_DIR / "results_vc_healthcare.json",
}

ALL_RESULTS: dict[str, Path] = {**REQUIRED_RESULTS, **OPTIONAL_RESULTS}

# ── Pydantic models ───────────────────────────────────────────────────────────

class ValidationMetrics(BaseModel):
    mae: float
    rmse: float
    r2: float


class ForecastPoint(BaseModel):
    month: str
    mid: float
    lo: float
    hi: float


class SeriesResult(BaseModel):
    series_id: str
    label: str
    unit: str
    last_date: str | None = None
    last_value: float | None = None
    validation: ValidationMetrics | None = None
    forecast: list[ForecastPoint]


class GroupResponse(BaseModel):
    group: str
    run_at: str
    series_count: int
    series: list[SeriesResult]


class EndpointInfo(BaseModel):
    path: str
    description: str
    group: str | None = None
    series: list[str] | None = None
    available: bool


class SummaryResponse(BaseModel):
    generated_at: str
    endpoints: list[EndpointInfo]


# ── Endpoint descriptor table ─────────────────────────────────────────────────
# Each entry maps to one results file and one API endpoint.

_ENDPOINT_DESCRIPTORS: list[dict] = [
    {
        "path": "/api/business-env",
        "description": (
            "12-month LightGBM forecasts for Industrial Production Index (INDPRO), "
            "Total Capacity Utilization (TCU), and Nonfarm Payroll (PAYEMS). "
            "Includes median forecast plus 80% prediction interval (lo/hi)."
        ),
        "group": "Business Environment",
        "key": "business_env",
        "optional": False,
    },
    {
        "path": "/api/consumer-demand",
        "description": (
            "12-month forecasts for Real Disposable Personal Income (DSPIC96), "
            "Personal Consumption Expenditures (PCE), Core PCE Price Index (PCEPILFE), "
            "Nominal Retail & Food Services Sales (RSAFS), "
            "Real Retail & Food Services Sales (RRSFS), "
            "and University of Michigan Consumer Sentiment (UMCSENT)."
        ),
        "group": "Consumer Demand",
        "key": "consumer_demand",
        "optional": False,
    },
    {
        "path": "/api/cost-of-capital",
        "description": (
            "12-month forecasts for the Federal Funds Effective Rate (DFF), "
            "Bank Prime Loan Rate (DPRIME), "
            "10Y−3M Treasury yield-curve spread (T10Y3M), "
            "and 10Y−2Y yield-curve spread (T10Y2Y)."
        ),
        "group": "Cost of Capital",
        "key": "cost_of_capital",
        "optional": False,
    },
    {
        "path": "/api/risk",
        "description": (
            "12-month forecasts for the Chauvet-Piger Smoothed Recession Probability "
            "(RECPROUSM156N) and University of Michigan Consumer Sentiment (UMCSENT) "
            "as leading risk indicators."
        ),
        "group": "Risk & Leading Indicators",
        "key": "risk",
        "optional": False,
    },
    {
        "path": "/api/sector/bls",
        "description": (
            "Optional: 12-month LightGBM forecasts for BLS industry-level employment "
            "sector series. Generate with: python fred_refresh.py --sector bls"
        ),
        "group": "Sector — BLS Employment",
        "key": "sector_bls",
        "optional": True,
    },
    {
        "path": "/api/sector/bea",
        "description": (
            "Optional: 12-month forecasts for BEA GDP-by-Industry sector series. "
            "Generate with: python fred_refresh.py --sector bea"
        ),
        "group": "Sector — BEA GDP by Industry",
        "key": "sector_bea",
        "optional": True,
    },
    {
        "path": "/api/sector/worldbank",
        "description": (
            "Optional: 12-month forecasts for World Bank cross-country sector indicators. "
            "Generate with: python fred_refresh.py --sector worldbank"
        ),
        "group": "Sector — World Bank",
        "key": "sector_worldbank",
        "optional": True,
    },
    {
        "path": "/api/vc/ai",
        "description": (
            "Optional: 12-month Crunchbase AI segment VC activity forecasts — "
            "company count, round count, capital raised, median round size, lead investor count. "
            "Generate with: python fred_refresh.py --crunchbase"
        ),
        "group": "Venture Capital — AI",
        "key": "vc_ai",
        "optional": True,
    },
    {
        "path": "/api/vc/fintech",
        "description": (
            "Optional: 12-month Crunchbase Fintech segment VC activity forecasts. "
            "Generate with: python fred_refresh.py --crunchbase"
        ),
        "group": "Venture Capital — Fintech",
        "key": "vc_fintech",
        "optional": True,
    },
    {
        "path": "/api/vc/healthcare",
        "description": (
            "Optional: 12-month Crunchbase Healthcare segment VC activity forecasts. "
            "Generate with: python fred_refresh.py --crunchbase"
        ),
        "group": "Venture Capital — Healthcare",
        "key": "vc_healthcare",
        "optional": True,
    },
]

# ── Shared file loader ────────────────────────────────────────────────────────

def _load_result(path: Path, key: str, optional: bool = False) -> GroupResponse:
    """Load a results JSON file and return a GroupResponse.

    Raises HTTPException 404 if the file is missing, 500 if it cannot be parsed.
    """
    if not path.exists():
        if optional:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Results file for '{key}' not found ({path.name}). "
                    "Run the pipeline with the appropriate flag to generate it — "
                    "see /api/summary for the required command."
                ),
            )
        raise HTTPException(
            status_code=404,
            detail=(
                f"Results file for '{key}' not found ({path.name}). "
                "Run `python fred_refresh.py` to generate it."
            ),
        )
    try:
        raw = json.loads(path.read_text())
        return GroupResponse(
            group=raw["group"],
            run_at=raw["run_at"],
            series_count=len(raw["series"]),
            series=raw["series"],
        )
    except Exception:
        log.exception("Failed to parse %s", path)
        raise HTTPException(
            status_code=500,
            detail=f"Results file for '{key}' could not be parsed. Check server logs.",
        )


def _validate_series_id(series_id: str) -> str:
    """Uppercase and validate series_id. Raises 400 if it contains unexpected characters."""
    sid = series_id.upper()
    if not _SERIES_ID_RE.match(sid):
        raise HTTPException(
            status_code=400,
            detail="series_id must be 1–30 alphanumeric characters or underscores.",
        )
    return sid


def _find_series(group: GroupResponse, series_id: str) -> SeriesResult:
    """Return a single SeriesResult by series_id. 404 if absent."""
    sid = _validate_series_id(series_id)
    match = next((s for s in group.series if s.series_id == sid), None)
    if match is None:
        available = [s.series_id for s in group.series]
        raise HTTPException(
            status_code=404,
            detail=f"Series '{sid}' not found in group '{group.group}'. Available: {available}",
        )
    return match


# ── App startup ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("econdata API starting — outputs dir: %s", OUTPUT_DIR)
    for key, path in REQUIRED_RESULTS.items():
        status = "FOUND" if path.exists() else "MISSING — run fred_refresh.py"
        log.info("  [required] %-20s  %s", key, status)
    for key, path in OPTIONAL_RESULTS.items():
        status = "found" if path.exists() else "not present"
        log.info("  [optional] %-20s  %s", key, status)
    yield


app = FastAPI(
    title="Macroeconomic Forecasting API",
    description=(
        "Read-only REST API exposing pre-computed LightGBM forecast results "
        "for macroeconomic indicators. Run `python fred_refresh.py` to refresh "
        "data and retrain models. See `/api/summary` for a full endpoint listing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Security middleware ───────────────────────────────────────────────────────

# CORS: restrict to localhost origins only (internal tool).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET"],
    allow_headers=[],
)


@app.middleware("http")
async def security_headers(request: Request, call_next) -> Response:
    """Add defensive HTTP security headers to every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = "default-src 'none'"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


# ── /api/summary ──────────────────────────────────────────────────────────────

@app.get(
    "/api/summary",
    response_model=SummaryResponse,
    tags=["Meta"],
    summary="List all API endpoints and their current availability",
)
def get_summary() -> SummaryResponse:
    """
    Returns a directory of every forecast endpoint — what it retrieves,
    which series it covers, and whether its backing result file is currently
    on disk (i.e. the model has been run).

    Optional endpoints (sector, VC) show `available: false` until the
    pipeline is run with the appropriate flag.
    """
    endpoint_infos: list[EndpointInfo] = []

    for desc in _ENDPOINT_DESCRIPTORS:
        path_obj = ALL_RESULTS[desc["key"]]
        available = path_obj.exists()
        series_ids: list[str] | None = None
        if available:
            try:
                raw = json.loads(path_obj.read_text())
                series_ids = [s["series_id"] for s in raw.get("series", [])]
            except Exception:
                pass  # don't crash summary if a file is corrupt
        endpoint_infos.append(
            EndpointInfo(
                path=desc["path"],
                description=desc["description"],
                group=desc["group"],
                series=series_ids,
                available=available,
            )
        )

    # Add per-series sub-endpoints for the four required groups
    sub_endpoints = [
        ("/api/business-env/{series_id}",    "Single-series result from the Business Environment group (INDPRO, TCU, PAYEMS)."),
        ("/api/consumer-demand/{series_id}", "Single-series result from Consumer Demand group (DSPIC96, PCE, PCEPILFE, RSAFS, RRSFS, UMCSENT)."),
        ("/api/cost-of-capital/{series_id}", "Single-series result from Cost of Capital group (DFF, DPRIME, T10Y3M, T10Y2Y)."),
        ("/api/risk/{series_id}",            "Single-series result from Risk & Leading Indicators group (RECPROUSM156N, UMCSENT)."),
    ]
    for path, description in sub_endpoints:
        # available if parent group file exists
        parent_key = path.split("/")[2].replace("-", "_")
        parent_path = REQUIRED_RESULTS.get(parent_key)
        endpoint_infos.append(
            EndpointInfo(
                path=path,
                description=description,
                group=None,
                series=None,
                available=parent_path.exists() if parent_path else False,
            )
        )

    # Self-reference
    endpoint_infos.append(
        EndpointInfo(
            path="/api/summary",
            description="This endpoint. Lists all API endpoints and their current availability.",
            group=None,
            series=None,
            available=True,
        )
    )

    return SummaryResponse(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        endpoints=endpoint_infos,
    )


# ── Business Environment ──────────────────────────────────────────────────────

@app.get(
    "/api/business-env",
    response_model=GroupResponse,
    tags=["Core Models"],
    summary="Business Environment forecasts (INDPRO, TCU, PAYEMS)",
)
def get_business_env() -> GroupResponse:
    """
    12-month LightGBM forecasts for:
    - **INDPRO** — Industrial Production Index (2017=100)
    - **TCU** — Total Capacity Utilization (%)
    - **PAYEMS** — Nonfarm Payroll Employment (thousands)

    Each series includes the last observed value, 24-month validation metrics
    (MAE, RMSE, R²), and a 12-month probabilistic forecast (median + 80% PI).
    """
    return _load_result(REQUIRED_RESULTS["business_env"], "business_env")


@app.get(
    "/api/business-env/{series_id}",
    response_model=SeriesResult,
    tags=["Core Models"],
    summary="Single Business Environment series forecast",
)
def get_business_env_series(series_id: str) -> SeriesResult:
    """
    Retrieve forecasts for one series from the Business Environment group.

    Valid `series_id` values: **INDPRO**, **TCU**, **PAYEMS**
    """
    group = _load_result(REQUIRED_RESULTS["business_env"], "business_env")
    return _find_series(group, series_id)


# ── Consumer Demand ───────────────────────────────────────────────────────────

@app.get(
    "/api/consumer-demand",
    response_model=GroupResponse,
    tags=["Core Models"],
    summary="Consumer Demand forecasts (DSPIC96, PCE, PCEPILFE, RSAFS, RRSFS, UMCSENT)",
)
def get_consumer_demand() -> GroupResponse:
    """
    12-month LightGBM forecasts for:
    - **DSPIC96** — Real Disposable Personal Income (billions chained $)
    - **PCE** — Personal Consumption Expenditures (billions $)
    - **PCEPILFE** — Core PCE Price Index ex food & energy (2017=100)
    - **RSAFS** — Nominal Retail & Food Services Sales (millions $)
    - **RRSFS** — Real Retail & Food Services Sales (millions chained $)
    - **UMCSENT** — University of Michigan Consumer Sentiment (1966:Q1=100)
    """
    return _load_result(REQUIRED_RESULTS["consumer_demand"], "consumer_demand")


@app.get(
    "/api/consumer-demand/{series_id}",
    response_model=SeriesResult,
    tags=["Core Models"],
    summary="Single Consumer Demand series forecast",
)
def get_consumer_demand_series(series_id: str) -> SeriesResult:
    """
    Retrieve forecasts for one series from the Consumer Demand group.

    Valid `series_id` values: **DSPIC96**, **PCE**, **PCEPILFE**, **RSAFS**, **RRSFS**, **UMCSENT**
    """
    group = _load_result(REQUIRED_RESULTS["consumer_demand"], "consumer_demand")
    return _find_series(group, series_id)


# ── Cost of Capital ───────────────────────────────────────────────────────────

@app.get(
    "/api/cost-of-capital",
    response_model=GroupResponse,
    tags=["Core Models"],
    summary="Cost of Capital forecasts (DFF, DPRIME, T10Y3M, T10Y2Y)",
)
def get_cost_of_capital() -> GroupResponse:
    """
    12-month LightGBM forecasts for:
    - **DFF** — Federal Funds Effective Rate (%)
    - **DPRIME** — Bank Prime Loan Rate (%)
    - **T10Y3M** — 10-Year minus 3-Month Treasury yield spread (%pts)
    - **T10Y2Y** — 10-Year minus 2-Year Treasury yield spread (%pts)

    Yield-curve spreads are key recession leading indicators.
    """
    return _load_result(REQUIRED_RESULTS["cost_of_capital"], "cost_of_capital")


@app.get(
    "/api/cost-of-capital/{series_id}",
    response_model=SeriesResult,
    tags=["Core Models"],
    summary="Single Cost of Capital series forecast",
)
def get_cost_of_capital_series(series_id: str) -> SeriesResult:
    """
    Retrieve forecasts for one series from the Cost of Capital group.

    Valid `series_id` values: **DFF**, **DPRIME**, **T10Y3M**, **T10Y2Y**
    """
    group = _load_result(REQUIRED_RESULTS["cost_of_capital"], "cost_of_capital")
    return _find_series(group, series_id)


# ── Risk & Leading Indicators ─────────────────────────────────────────────────

@app.get(
    "/api/risk",
    response_model=GroupResponse,
    tags=["Core Models"],
    summary="Risk & Leading Indicator forecasts (RECPROUSM156N, UMCSENT)",
)
def get_risk() -> GroupResponse:
    """
    12-month LightGBM forecasts for:
    - **RECPROUSM156N** — Chauvet-Piger Smoothed Recession Probability (%)
    - **UMCSENT** — University of Michigan Consumer Sentiment (1966:Q1=100)

    These series serve as leading risk indicators for the macro environment.
    """
    return _load_result(REQUIRED_RESULTS["risk"], "risk")


@app.get(
    "/api/risk/{series_id}",
    response_model=SeriesResult,
    tags=["Core Models"],
    summary="Single Risk & Leading Indicator series forecast",
)
def get_risk_series(series_id: str) -> SeriesResult:
    """
    Retrieve forecasts for one series from the Risk & Leading Indicators group.

    Valid `series_id` values: **RECPROUSM156N**, **UMCSENT**
    """
    group = _load_result(REQUIRED_RESULTS["risk"], "risk")
    return _find_series(group, series_id)


# ── Sector — BLS ──────────────────────────────────────────────────────────────

@app.get(
    "/api/sector/bls",
    response_model=GroupResponse,
    tags=["Sector Models (Optional)"],
    summary="BLS industry employment sector forecasts",
)
def get_sector_bls() -> GroupResponse:
    """
    12-month LightGBM forecasts for BLS industry-level employment series.

    **Requires prior run:** `python fred_refresh.py --sector bls`

    Returns 404 with an instructive message if the results file has not been generated.
    """
    return _load_result(OPTIONAL_RESULTS["sector_bls"], "sector_bls", optional=True)


@app.get(
    "/api/sector/bea",
    response_model=GroupResponse,
    tags=["Sector Models (Optional)"],
    summary="BEA GDP-by-Industry sector forecasts",
)
def get_sector_bea() -> GroupResponse:
    """
    12-month LightGBM forecasts for BEA GDP-by-Industry series.

    **Requires prior run:** `python fred_refresh.py --sector bea`
    """
    return _load_result(OPTIONAL_RESULTS["sector_bea"], "sector_bea", optional=True)


@app.get(
    "/api/sector/worldbank",
    response_model=GroupResponse,
    tags=["Sector Models (Optional)"],
    summary="World Bank cross-country sector forecasts",
)
def get_sector_worldbank() -> GroupResponse:
    """
    12-month LightGBM forecasts for World Bank cross-country sector indicators.

    **Requires prior run:** `python fred_refresh.py --sector worldbank`
    """
    return _load_result(OPTIONAL_RESULTS["sector_worldbank"], "sector_worldbank", optional=True)


# ── Venture Capital ───────────────────────────────────────────────────────────

@app.get(
    "/api/vc/ai",
    response_model=GroupResponse,
    tags=["Venture Capital Models (Optional)"],
    summary="Crunchbase AI segment VC activity forecasts",
)
def get_vc_ai() -> GroupResponse:
    """
    12-month LightGBM forecasts for the Crunchbase **AI** venture capital segment:
    company count, 90-day rolling round count, capital raised (USD),
    median round size (USD), and lead investor count.

    **Requires prior run:** `python fred_refresh.py --crunchbase`
    """
    return _load_result(OPTIONAL_RESULTS["vc_ai"], "vc_ai", optional=True)


@app.get(
    "/api/vc/fintech",
    response_model=GroupResponse,
    tags=["Venture Capital Models (Optional)"],
    summary="Crunchbase Fintech segment VC activity forecasts",
)
def get_vc_fintech() -> GroupResponse:
    """
    12-month LightGBM forecasts for the Crunchbase **Fintech** venture capital segment.

    **Requires prior run:** `python fred_refresh.py --crunchbase`
    """
    return _load_result(OPTIONAL_RESULTS["vc_fintech"], "vc_fintech", optional=True)


@app.get(
    "/api/vc/healthcare",
    response_model=GroupResponse,
    tags=["Venture Capital Models (Optional)"],
    summary="Crunchbase Healthcare segment VC activity forecasts",
)
def get_vc_healthcare() -> GroupResponse:
    """
    12-month LightGBM forecasts for the Crunchbase **Healthcare** venture capital segment.

    **Requires prior run:** `python fred_refresh.py --crunchbase`
    """
    return _load_result(OPTIONAL_RESULTS["vc_healthcare"], "vc_healthcare", optional=True)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8100,
        reload=False,
        log_level="info",
        server_header=False,   # suppress "server: uvicorn" disclosure header
    )
