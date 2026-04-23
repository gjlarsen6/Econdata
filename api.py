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

import pandas as pd
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
DATA_DIR   = BASE_DIR / "data"

REQUIRED_RESULTS: dict[str, Path] = {
    "business_env":    OUTPUT_DIR / "results_business_env.json",
    "consumer_demand": OUTPUT_DIR / "results_consumer_demand.json",
    "cost_of_capital": OUTPUT_DIR / "results_cost_of_capital.json",
    "risk":            OUTPUT_DIR / "results_risk.json",
}

OPTIONAL_RESULTS: dict[str, Path] = {
    # Sector — BLS/BEA/WorldBank (original)
    "sector_bls":           OUTPUT_DIR / "results_sector_bls.json",
    "sector_bea":           OUTPUT_DIR / "results_sector_bea.json",
    "sector_worldbank":     OUTPUT_DIR / "results_sector_worldbank.json",
    # Sector — BLS subgroups (Priority 3)
    "sector_bls_wages":     OUTPUT_DIR / "results_sector_bls_wages.json",
    "sector_bls_hours":     OUTPUT_DIR / "results_sector_bls_hours.json",
    "sector_jolts":         OUTPUT_DIR / "results_sector_jolts.json",
    # Sector — ETFs (Priority 4)
    "sector_etf":           OUTPUT_DIR / "results_sector_etf.json",
    # Venture Capital
    "vc_ai":                OUTPUT_DIR / "results_vc_ai.json",
    "vc_fintech":           OUTPUT_DIR / "results_vc_fintech.json",
    "vc_healthcare":        OUTPUT_DIR / "results_vc_healthcare.json",
    # Phase 0 — free FRED extensions
    "market_risk":          OUTPUT_DIR / "results_market_risk.json",
    "commodities":          OUTPUT_DIR / "results_commodities.json",
    "yield_curve":          OUTPUT_DIR / "results_yield_curve.json",
    "financial_stress":     OUTPUT_DIR / "results_financial_stress.json",
    "regime":               OUTPUT_DIR / "regime_history.json",
    # Phase 1 — Industrial / ISM PMI / Capacity Util / Credit (industrial_model.py)
    "industrial_production":      OUTPUT_DIR / "results_industrial_production.json",
    "industrial_ism_pmi":         OUTPUT_DIR / "results_industrial_ism_pmi.json",
    "industrial_capacity_util":   OUTPUT_DIR / "results_industrial_capacity_util.json",
    "industrial_credit":          OUTPUT_DIR / "results_industrial_credit.json",
    # Phase 2 — Financial News ML
    "news_sentiment":       OUTPUT_DIR / "results_news_sentiment.json",
    "news_volume":          OUTPUT_DIR / "results_news_volume.json",
}

ALL_RESULTS: dict[str, Path] = {**REQUIRED_RESULTS, **OPTIONAL_RESULTS}

# Map series_id → CSV path (for /api/series/{id}/history)
_HISTORY_FILE_MAP: dict[str, Path] = {
    # BusinessEnvironment
    "INDPRO":          DATA_DIR / "BusinessEnvironment" / "INDPRO.csv",
    "TCU":             DATA_DIR / "BusinessEnvironment" / "TCU_capacityutilization.csv",
    "PAYEMS":          DATA_DIR / "BusinessEnvironment" / "Payroll_PAYEMS.csv",
    "CAPUTLB50001SQ":  DATA_DIR / "BusinessEnvironment" / "CAPUTLB50001SQ.csv",
    # ConsumerDemand
    "DSPIC96":         DATA_DIR / "ConsumerDemand" / "DSPIC96.csv",
    "PCE":             DATA_DIR / "ConsumerDemand" / "PCE.csv",
    "PCEPILFE":        DATA_DIR / "ConsumerDemand" / "PersConsume_noFoodEnergyPCEPILFE.csv",
    "RSAFS":           DATA_DIR / "ConsumerDemand" / "RSAFS.csv",
    "RRSFS":           DATA_DIR / "ConsumerDemand" / "RealRetailandFoodSalesRRSFS.csv",
    "UMCSENT":         DATA_DIR / "ConsumerDemand" / "UMCSENT.csv",
    # CostOfCapital
    "DFF":             DATA_DIR / "CostOfCapital" / "DFF.csv",
    "DPRIME":          DATA_DIR / "CostOfCapital" / "DPRIME.csv",
    "FEDFUNDS":        DATA_DIR / "CostOfCapital" / "FEDFUNDS.csv",
    "PRIME":           DATA_DIR / "CostOfCapital" / "PRIME.csv",
    "T10Y2Y":          DATA_DIR / "CostOfCapital" / "T10Y2Y.csv",
    "T10Y3M":          DATA_DIR / "CostOfCapital" / "T10Y3M.csv",
    # RiskLeadingInd
    "RECPROUSM156N":   DATA_DIR / "RiskLeadingInd" / "RECPROUSM156N.csv",
    # MarketRisk
    "VIXCLS":          DATA_DIR / "MarketRisk" / "VIXCLS.csv",
    "BAMLH0A0HYM2":    DATA_DIR / "MarketRisk" / "BAMLH0A0HYM2.csv",
    "BAMLC0A0CM":      DATA_DIR / "MarketRisk" / "BAMLC0A0CM.csv",
    "DTWEXBGS":        DATA_DIR / "MarketRisk" / "DTWEXBGS.csv",
    # Commodities
    "DCOILWTICO":      DATA_DIR / "Commodities" / "DCOILWTICO.csv",
    "NASDAQQGLDI":     DATA_DIR / "Commodities" / "NASDAQQGLDI.csv",
    # YieldCurve
    "DGS1MO":          DATA_DIR / "YieldCurve" / "DGS1MO.csv",
    "DGS3MO":          DATA_DIR / "YieldCurve" / "DGS3MO.csv",
    "DGS6MO":          DATA_DIR / "YieldCurve" / "DGS6MO.csv",
    "DGS1":            DATA_DIR / "YieldCurve" / "DGS1.csv",
    "DGS2":            DATA_DIR / "YieldCurve" / "DGS2.csv",
    "DGS5":            DATA_DIR / "YieldCurve" / "DGS5.csv",
    "DGS10":           DATA_DIR / "YieldCurve" / "DGS10.csv",
    "DGS30":           DATA_DIR / "YieldCurve" / "DGS30.csv",
    # IndustrialProduction (Phase 1)
    "IPMAN":           DATA_DIR / "IndustrialProduction" / "IPMAN.csv",
    "IPUTIL":          DATA_DIR / "IndustrialProduction" / "IPUTIL.csv",
    "IPMINE":          DATA_DIR / "IndustrialProduction" / "IPMINE.csv",
    "IPCONGD":         DATA_DIR / "IndustrialProduction" / "IPCONGD.csv",
    "IPBUSEQ":         DATA_DIR / "IndustrialProduction" / "IPBUSEQ.csv",
    "IPMAT":           DATA_DIR / "IndustrialProduction" / "IPMAT.csv",
    "IPDCONGD":        DATA_DIR / "IndustrialProduction" / "IPDCONGD.csv",
    "IPNCONGD":        DATA_DIR / "IndustrialProduction" / "IPNCONGD.csv",
    # ISMIndicators — Census Bureau manufacturing orders (Phase 1)
    "NEWORDER":        DATA_DIR / "ISMIndicators" / "NEWORDER.csv",
    "DGORDER":         DATA_DIR / "ISMIndicators" / "DGORDER.csv",
    "AMTUNO":          DATA_DIR / "ISMIndicators" / "AMTUNO.csv",
    "MNFCTRIRSA":      DATA_DIR / "ISMIndicators" / "MNFCTRIRSA.csv",
    # CapacityUtilSector (Phase 1)
    "MCUMFN":          DATA_DIR / "CapacityUtilSector" / "MCUMFN.csv",
    "CAPUTLG211S":     DATA_DIR / "CapacityUtilSector" / "CAPUTLG211S.csv",
    "CAPUTLG331S":     DATA_DIR / "CapacityUtilSector" / "CAPUTLG331S.csv",
    # CreditIndicators (Phase 1)
    "BUSLOANS":        DATA_DIR / "CreditIndicators" / "BUSLOANS.csv",
    "REALLN":          DATA_DIR / "CreditIndicators" / "REALLN.csv",
    "CONSUMER":        DATA_DIR / "CreditIndicators" / "CONSUMER.csv",
    "WPU054":          DATA_DIR / "CreditIndicators" / "WPU054.csv",
    "WPU01":           DATA_DIR / "CreditIndicators" / "WPU01.csv",
}

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


class HealthResponse(BaseModel):
    status: str
    ts: str


class RegimeForecastPoint(BaseModel):
    month: str
    fsi: float
    regime: str


class RegimeHistoryPoint(BaseModel):
    date: str
    fsi: float
    regime: str


class RegimeResponse(BaseModel):
    generated_at: str
    current_regime: str
    current_fsi: float
    regime_distribution: dict[str, float]
    forecast: list[RegimeForecastPoint]
    history: list[RegimeHistoryPoint]


class HistoryPoint(BaseModel):
    observation_date: str
    value: float | None


class HistoryResponse(BaseModel):
    series_id: str
    row_count: int
    observations: list[HistoryPoint]


# ── Phase 1: Financial News models ────────────────────────────────────────────

class NewsArticle(BaseModel):
    timestamp: str
    source_name: str
    sector: str
    ticker: str | None = None
    headline: str
    sentiment: float | None = None
    sentiment_label: str | None = None
    macro_tag: str | None = None
    market_impact_score: float | None = None


class BriefingResponse(BaseModel):
    date: str
    generated_at: str
    article_count: int
    top_stories: list[NewsArticle]
    sector_mood: dict[str, float]
    macro_signals: list[str]
    alerts: list[str]
    stale: bool = False   # True if briefing file is older than BRIEFING_STALE_HOURS (36h)


# ── Phase 2: Financial News ML models ─────────────────────────────────────────

class NewsColdStartResponse(BaseModel):
    status: str = "cold_start"
    days_collected: int
    min_required: int
    series: list = []


class NewsGroupResponse(BaseModel):
    group: str
    run_at: str
    series: list[SeriesResult]


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
        "path": "/api/sector/bls-wages",
        "description": (
            "Optional: 12-month forecasts for BLS average hourly earnings by sector (6 series). "
            "Generate with: python fred_refresh.py --sector bls_wages"
        ),
        "group": "Sector — BLS Wages",
        "key": "sector_bls_wages",
        "optional": True,
    },
    {
        "path": "/api/sector/bls-hours",
        "description": (
            "Optional: 12-month forecasts for BLS average weekly hours by sector (3 series). "
            "Generate with: python fred_refresh.py --sector bls_hours"
        ),
        "group": "Sector — BLS Hours",
        "key": "sector_bls_hours",
        "optional": True,
    },
    {
        "path": "/api/sector/jolts",
        "description": (
            "Optional: 12-month forecasts for JOLTS job openings by sector (7 series). "
            "Generate with: python fred_refresh.py --sector jolts"
        ),
        "group": "Sector — JOLTS Job Openings",
        "key": "sector_jolts",
        "optional": True,
    },
    {
        "path": "/api/sector/etf",
        "description": (
            "Optional: 12-month forecasts for S&P 500 sector ETF prices (11 tickers, monthly close). "
            "Generate with: python fred_refresh.py --sector etf"
        ),
        "group": "Sector — ETFs",
        "key": "sector_etf",
        "optional": True,
    },
    # ── Phase 1: Industrial / ISM / Capacity / Credit ─────────────────────────
    {
        "path": "/api/industrial/production",
        "description": (
            "12-month LightGBM forecasts for 8 IP sector breakdowns: "
            "Manufacturing, Utilities, Mining, Consumer Goods, Business Equipment, "
            "Materials, Durable Consumer Goods, Nondurable Consumer Goods. "
            "Generated automatically by fred_refresh.py."
        ),
        "group": "Industrial Production",
        "key": "industrial_production",
        "optional": True,
    },
    {
        "path": "/api/industrial/ism-pmi",
        "description": (
            "12-month LightGBM forecasts for ISM Manufacturing and Services PMI composite "
            "and sub-indices (New Orders, Production, Employment, Vendor Deliveries). "
            "Generated automatically by fred_refresh.py."
        ),
        "group": "ISM PMI Leading Indicators",
        "key": "industrial_ism_pmi",
        "optional": True,
    },
    {
        "path": "/api/industrial/capacity-utilization",
        "description": (
            "12-month LightGBM forecasts for sector-level capacity utilization: "
            "Manufacturing, Mining, and Durable Goods. "
            "Generated automatically by fred_refresh.py."
        ),
        "group": "Capacity Utilization by Sector",
        "key": "industrial_capacity_util",
        "optional": True,
    },
    {
        "path": "/api/industrial/credit",
        "description": (
            "12-month LightGBM forecasts for commercial/real estate/consumer loans "
            "and PPI commodity prices (fuels, farm products). "
            "Generated automatically by fred_refresh.py."
        ),
        "group": "Credit & PPI Indicators",
        "key": "industrial_credit",
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
    # ── Phase 0: Free FRED Extensions ────────────────────────────────────────
    {
        "path": "/api/market/vix",
        "description": (
            "12-month LightGBM forecast for the CBOE Volatility Index (VIX). "
            "Generated automatically by fred_refresh.py (no extra API keys required)."
        ),
        "group": "Market Risk",
        "key": "market_risk",
        "optional": True,
    },
    {
        "path": "/api/market/spreads",
        "description": (
            "12-month LightGBM forecasts for US credit spreads: "
            "ICE BofA High Yield OAS (BAMLH0A0HYM2) and IG Corporate OAS (BAMLC0A0CM). "
            "Generated automatically by fred_refresh.py."
        ),
        "group": "Market Risk",
        "key": "market_risk",
        "optional": True,
    },
    {
        "path": "/api/market/dollar",
        "description": (
            "12-month LightGBM forecast for the Nominal Broad USD Index (DTWEXBGS). "
            "Generated automatically by fred_refresh.py."
        ),
        "group": "Market Risk",
        "key": "market_risk",
        "optional": True,
    },
    {
        "path": "/api/commodities/oil",
        "description": (
            "12-month LightGBM forecast for WTI Crude Oil price (DCOILWTICO, $/bbl). "
            "Generated automatically by fred_refresh.py."
        ),
        "group": "Commodities",
        "key": "commodities",
        "optional": True,
    },
    {
        "path": "/api/commodities/gold",
        "description": (
            "12-month LightGBM forecast for Gold price index (NASDAQQGLDI). "
            "Generated automatically by fred_refresh.py."
        ),
        "group": "Commodities",
        "key": "commodities",
        "optional": True,
    },
    {
        "path": "/api/market/yield-curve",
        "description": (
            "Full US Treasury yield curve: 8 constant-maturity tenors "
            "(1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y) with 12-month LightGBM forecasts. "
            "Generated automatically by fred_refresh.py."
        ),
        "group": "Treasury Yield Curve",
        "key": "yield_curve",
        "optional": True,
    },
    {
        "path": "/api/market/stress",
        "description": (
            "Financial Stress Index (FSI): composite of VIX, HY/IG credit spreads, "
            "recession probability, and yield curve inversion. "
            "Scale 0 (calm) to 1 (extreme stress). Includes 12-month forecast. "
            "Generated automatically by fred_refresh.py."
        ),
        "group": "Financial Stress Index",
        "key": "financial_stress",
        "optional": True,
    },
    {
        "path": "/api/market/regime",
        "description": (
            "Market Regime classifier: current label (expansion/slowdown/contraction/stress/recovery), "
            "FSI value, regime distribution, 12-month forecast, and 10-year history. "
            "Generated automatically by fred_refresh.py."
        ),
        "group": "Market Regime",
        "key": "regime",
        "optional": True,
    },
    # ── Phase 1: Financial News ───────────────────────────────────────────────
    {
        "path": "/api/financial-news/briefing",
        "description": (
            "Today's top stories, sector mood scores, macro signal bullets, and high-impact alerts. "
            "Generated by briefing.py after: python fred_refresh.py --news daily. "
            "Includes stale=true flag if briefing is older than 36 hours."
        ),
        "group": "Financial News (Phase 1)",
        "key": None,   # no single results file — availability checked via _latest_briefing_path()
        "optional": True,
    },
    {
        "path": "/api/financial-news/top-stories",
        "description": (
            "Top 10 highest market-impact articles from today's news ingestion, "
            "sorted by impact score. Generate with: python fred_refresh.py --news daily"
        ),
        "group": "Financial News (Phase 1)",
        "key": None,
        "optional": True,
    },
    {
        "path": "/api/financial-news/alerts",
        "description": (
            "High-impact articles (market_impact_score ≥ 0.75) from today's news. "
            "Returns [] if none exceed threshold. Generate with: python fred_refresh.py --news daily"
        ),
        "group": "Financial News (Phase 1)",
        "key": None,
        "optional": True,
    },
    # ── Phase 2: Financial News ML ────────────────────────────────────────────
    {
        "path": "/api/financial-news/sentiment",
        "description": (
            "Sector sentiment trends + 12-month LightGBM forecast. "
            "Series: MACRO_SENT, EQUITIES_SENT, FINTECH_SENT, VC_SENT. "
            "Returns cold_start status until ≥30 days of news data are collected. "
            "Generate with: python news_model.py"
        ),
        "group": "Financial News ML (Phase 2)",
        "key": "news_sentiment",
        "optional": True,
    },
    {
        "path": "/api/financial-news/sentiment/{id}",
        "description": (
            "Single sentiment series (MACRO_SENT, EQUITIES_SENT, FINTECH_SENT, VC_SENT). "
            "Returns 404 if series_id not found or results not yet generated."
        ),
        "group": "Financial News ML (Phase 2)",
        "key": "news_sentiment",
        "optional": True,
    },
    {
        "path": "/api/financial-news/volume",
        "description": (
            "Article volume trends by sector + 12-month forecast. "
            "Series: TOTAL_VOL, MACRO_VOL, EQUITIES_VOL, FINTECH_VOL. "
            "Returns cold_start status until ≥30 days of news data are collected. "
            "Generate with: python news_model.py"
        ),
        "group": "Financial News ML (Phase 2)",
        "key": "news_volume",
        "optional": True,
    },
]

# ── Phase 1: briefing path helpers (used by get_summary + route handlers) ────

BRIEFING_STALE_HOURS = 36


def _latest_briefing_path() -> Path | None:
    """Return the most recent daily_briefing_*.json in OUTPUT_DIR, or None."""
    candidates = sorted(OUTPUT_DIR.glob("daily_briefing_*.json"), reverse=True)
    return candidates[0] if candidates else None


def _briefing_is_stale(path: Path) -> bool:
    """Return True if the briefing file's mtime is older than BRIEFING_STALE_HOURS."""
    import time as _time
    age_seconds = _time.time() - path.stat().st_mtime
    return age_seconds > BRIEFING_STALE_HOURS * 3600


def _load_briefing_json(path: Path) -> dict:
    """Load and parse a briefing JSON file. Raises HTTPException 500 on parse failure."""
    try:
        return json.loads(path.read_text())
    except Exception:
        log.exception("Failed to parse briefing %s", path)
        raise HTTPException(status_code=500, detail="Failed to parse briefing file.")


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
        key = desc["key"]
        series_ids: list[str] | None = None

        if key is None:
            # Phase 1 news endpoints — availability via latest briefing file
            available = _latest_briefing_path() is not None
        else:
            path_obj  = ALL_RESULTS[key]
            available = path_obj.exists()
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


@app.get(
    "/api/sector/bls-wages",
    response_model=GroupResponse,
    tags=["Sector Models (Optional)"],
    summary="BLS average hourly earnings by sector forecasts",
)
def get_sector_bls_wages() -> GroupResponse:
    """
    12-month LightGBM forecasts for BLS average hourly earnings by sector (6 series).

    **Requires prior run:** `python fred_refresh.py --sector bls_wages`
    """
    return _load_result(OPTIONAL_RESULTS["sector_bls_wages"], "sector_bls_wages", optional=True)


@app.get(
    "/api/sector/bls-hours",
    response_model=GroupResponse,
    tags=["Sector Models (Optional)"],
    summary="BLS average weekly hours by sector forecasts",
)
def get_sector_bls_hours() -> GroupResponse:
    """
    12-month LightGBM forecasts for BLS average weekly hours by sector (3 series).

    **Requires prior run:** `python fred_refresh.py --sector bls_hours`
    """
    return _load_result(OPTIONAL_RESULTS["sector_bls_hours"], "sector_bls_hours", optional=True)


@app.get(
    "/api/sector/jolts",
    response_model=GroupResponse,
    tags=["Sector Models (Optional)"],
    summary="JOLTS job openings by sector forecasts",
)
def get_sector_jolts() -> GroupResponse:
    """
    12-month LightGBM forecasts for JOLTS job openings by sector (7 series).

    **Requires prior run:** `python fred_refresh.py --sector jolts`
    """
    return _load_result(OPTIONAL_RESULTS["sector_jolts"], "sector_jolts", optional=True)


@app.get(
    "/api/sector/etf",
    response_model=GroupResponse,
    tags=["Sector Models (Optional)"],
    summary="S&P 500 sector ETF price forecasts",
)
def get_sector_etf() -> GroupResponse:
    """
    12-month LightGBM forecasts for S&P 500 sector ETF monthly close prices (11 tickers):
    XLK, XLF, XLV, XLE, XLI, XLP, XLY, XLU, XLRE, XLB, XLC.

    **Requires prior run:** `python fred_refresh.py --sector etf`
    """
    return _load_result(OPTIONAL_RESULTS["sector_etf"], "sector_etf", optional=True)


# ── Industrial Models (Phase 1) ───────────────────────────────────────────────

@app.get(
    "/api/industrial/production",
    response_model=GroupResponse,
    tags=["Industrial Models (Phase 1)"],
    summary="Industrial production sector breakdowns — 12-month forecast",
)
def get_industrial_production() -> GroupResponse:
    """
    12-month LightGBM forecasts for 8 IP sector breakdowns (IPMAN, IPUTIL, IPMINE,
    IPCONGD, IPBUSEQ, IPMAT, IPDCONGD, IPNCONGD). History starts 1939–1919.

    Generated automatically by `industrial_model.py` on every `fred_refresh.py` run.
    """
    return _load_result(
        OPTIONAL_RESULTS["industrial_production"], "industrial_production", optional=True
    )


@app.get(
    "/api/industrial/ism-pmi",
    response_model=GroupResponse,
    tags=["Industrial Models (Phase 1)"],
    summary="Manufacturing orders leading indicators — 12-month forecast",
)
def get_industrial_ism_pmi() -> GroupResponse:
    """
    12-month LightGBM forecasts for Census Bureau manufacturing orders:
    Nondefense Capital Goods (NEWORDER), Durable Goods (DGORDER),
    Unfilled Orders (AMTUNO), Total Manufacturing SA (MNFCTRIRSA).

    NEWORDER leads business investment by 3–6 months — strongest leading indicator here.

    Generated automatically by `industrial_model.py` on every `fred_refresh.py` run.
    """
    return _load_result(
        OPTIONAL_RESULTS["industrial_ism_pmi"], "industrial_ism_pmi", optional=True
    )


@app.get(
    "/api/industrial/capacity-utilization",
    response_model=GroupResponse,
    tags=["Industrial Models (Phase 1)"],
    summary="Sector-level capacity utilization — 12-month forecast",
)
def get_industrial_capacity_util() -> GroupResponse:
    """
    12-month LightGBM forecasts for sector-level capacity utilization:
    Manufacturing (MCUMFN), Mining (CAPUTLG211S), Primary Metals (CAPUTLG331S).

    Generated automatically by `industrial_model.py` on every `fred_refresh.py` run.
    """
    return _load_result(
        OPTIONAL_RESULTS["industrial_capacity_util"], "industrial_capacity_util", optional=True
    )


@app.get(
    "/api/industrial/credit",
    response_model=GroupResponse,
    tags=["Industrial Models (Phase 1)"],
    summary="Credit and PPI sector indicators — 12-month forecast",
)
def get_industrial_credit() -> GroupResponse:
    """
    12-month LightGBM forecasts for commercial and industrial loans (BUSLOANS),
    real estate loans (REALLN), consumer loans (CONSUMER),
    PPI fuels (WPU054), and PPI farm products (WPU01).

    Generated automatically by `industrial_model.py` on every `fred_refresh.py` run.
    """
    return _load_result(
        OPTIONAL_RESULTS["industrial_credit"], "industrial_credit", optional=True
    )


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


@app.get(
    "/api/financial-news/briefing",
    response_model=BriefingResponse,
    tags=["Financial News (Phase 1)"],
    summary="Daily financial news briefing",
)
def get_briefing() -> BriefingResponse:
    """
    Today's full briefing: date, article count, top 10 stories by market impact score,
    sector mood scores (avg sentiment per sector), macro signal bullets, high-impact
    alerts, and a `stale` flag (true if briefing is older than 36 hours).

    **Requires prior run:** `python fred_refresh.py --news daily`

    Returns 404 if no briefing has been generated yet.
    """
    path = _latest_briefing_path()
    if path is None:
        raise HTTPException(
            status_code=404,
            detail="No briefing found. Run: python fred_refresh.py --news daily",
        )
    data = _load_briefing_json(path)
    data["stale"] = _briefing_is_stale(path)
    return BriefingResponse(**data)


@app.get(
    "/api/financial-news/top-stories",
    response_model=list[NewsArticle],
    tags=["Financial News (Phase 1)"],
    summary="Top 10 market-impact articles today",
)
def get_top_stories() -> list[NewsArticle]:
    """
    Top 10 articles sorted by `market_impact_score` from today's briefing.
    Each article includes headline, source, sector, ticker, sentiment, and score.

    **Requires prior run:** `python fred_refresh.py --news daily`
    """
    path = _latest_briefing_path()
    if path is None:
        raise HTTPException(
            status_code=404,
            detail="No briefing found. Run: python fred_refresh.py --news daily",
        )
    data = _load_briefing_json(path)
    return [NewsArticle(**a) for a in data.get("top_stories", [])]


@app.get(
    "/api/financial-news/alerts",
    response_model=list[str],
    tags=["Financial News (Phase 1)"],
    summary="High-impact news alerts (score ≥ 0.75)",
)
def get_alerts() -> list[str]:
    """
    Headlines where `market_impact_score ≥ 0.75`, formatted as
    `"{source}: {headline}"`, sorted by score descending. Returns `[]` if
    none exceed the threshold (always HTTP 200 once a briefing exists).

    **Requires prior run:** `python fred_refresh.py --news daily`
    """
    path = _latest_briefing_path()
    if path is None:
        raise HTTPException(
            status_code=404,
            detail="No briefing found. Run: python fred_refresh.py --news daily",
        )
    data = _load_briefing_json(path)
    return data.get("alerts", [])


# ── Phase 2: Financial News ML helpers ────────────────────────────────────────

def _load_news_result(key: str) -> dict:
    """Load a Phase 2 news results JSON. Raises HTTPException 404 if missing."""
    path = OPTIONAL_RESULTS[key]
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Results for '{key}' not found ({path.name}). "
                "Run: python news_model.py"
            ),
        )
    try:
        return json.loads(path.read_text())
    except Exception:
        log.exception("Failed to parse %s", path)
        raise HTTPException(
            status_code=500,
            detail=f"Results file for '{key}' could not be parsed. Check server logs.",
        )


# ── Phase 2: Financial News ML endpoints ──────────────────────────────────────

@app.get(
    "/api/financial-news/sentiment",
    tags=["Financial News ML (Phase 2)"],
    summary="Sector sentiment trends + 12-month LightGBM forecast",
)
def get_news_sentiment() -> NewsGroupResponse | NewsColdStartResponse:
    """
    12-month LightGBM forecasts for financial news sector sentiment:
    **MACRO_SENT**, **EQUITIES_SENT**, **FINTECH_SENT**, **VC_SENT**.

    Returns `{"status": "cold_start", ...}` until ≥30 days of news data have been
    collected. Run `python news_model.py` after accumulating sufficient data.

    Returns 404 if `outputs/results_news_sentiment.json` has not been generated.
    """
    raw = _load_news_result("news_sentiment")
    if raw.get("status") == "cold_start":
        return NewsColdStartResponse(
            days_collected=raw.get("days_collected", 0),
            min_required=raw.get("min_required", 30),
        )
    return NewsGroupResponse(
        group=raw["group"],
        run_at=raw["run_at"],
        series=raw.get("series", []),
    )


@app.get(
    "/api/financial-news/sentiment/{series_id}",
    response_model=SeriesResult,
    tags=["Financial News ML (Phase 2)"],
    summary="Single sentiment series forecast",
)
def get_news_sentiment_series(series_id: str) -> SeriesResult:
    """
    Single series from the sentiment group.

    Valid `series_id` values: **MACRO_SENT**, **EQUITIES_SENT**,
    **FINTECH_SENT**, **VC_SENT**

    Returns 404 if the series is not found or results have not been generated.
    """
    sid = _validate_series_id(series_id)
    raw = _load_news_result("news_sentiment")
    if raw.get("status") == "cold_start":
        raise HTTPException(
            status_code=404,
            detail=(
                f"News sentiment model is in cold_start state "
                f"({raw.get('days_collected', 0)}/{raw.get('min_required', 30)} days). "
                "Run: python news_model.py after ≥30 days of data."
            ),
        )
    match = next((s for s in raw.get("series", []) if s.get("series_id") == sid), None)
    if match is None:
        available = [s.get("series_id") for s in raw.get("series", [])]
        raise HTTPException(
            status_code=404,
            detail=f"Series '{sid}' not found. Available: {available}",
        )
    return SeriesResult(**match)


@app.get(
    "/api/financial-news/volume",
    tags=["Financial News ML (Phase 2)"],
    summary="Article volume by sector + 12-month forecast",
)
def get_news_volume() -> NewsGroupResponse | NewsColdStartResponse:
    """
    12-month LightGBM forecasts for article volume by sector:
    **TOTAL_VOL**, **MACRO_VOL**, **EQUITIES_VOL**, **FINTECH_VOL**.

    Returns `{"status": "cold_start", ...}` until ≥30 days of news data have been
    collected. Run `python news_model.py` after accumulating sufficient data.

    Returns 404 if `outputs/results_news_volume.json` has not been generated.
    """
    raw = _load_news_result("news_volume")
    if raw.get("status") == "cold_start":
        return NewsColdStartResponse(
            days_collected=raw.get("days_collected", 0),
            min_required=raw.get("min_required", 30),
        )
    return NewsGroupResponse(
        group=raw["group"],
        run_at=raw["run_at"],
        series=raw.get("series", []),
    )


# ── Health check ─────────────────────────────────────────────────────────────

@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["Meta"],
    summary="Liveness health check",
)
def get_health() -> HealthResponse:
    """Lightweight liveness check — no file I/O, always fast."""
    return HealthResponse(
        status="ok",
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


# ── Shared helpers for Phase 0 endpoints ──────────────────────────────────────

def _load_optional(key: str) -> GroupResponse:
    """Load an optional results file, returning 404 with instructions if missing."""
    path = OPTIONAL_RESULTS[key]
    return _load_result(path, key, optional=True)


def _filter_series(group: GroupResponse, series_ids: list[str]) -> GroupResponse:
    """Return a GroupResponse containing only the requested series_ids."""
    matched = [s for s in group.series if s.series_id in series_ids]
    if not matched:
        raise HTTPException(
            status_code=404,
            detail=f"None of {series_ids} found in group '{group.group}'.",
        )
    return GroupResponse(
        group=group.group,
        run_at=group.run_at,
        series_count=len(matched),
        series=matched,
    )


# ── Market Risk ───────────────────────────────────────────────────────────────

@app.get(
    "/api/market/vix",
    response_model=SeriesResult,
    tags=["Market Risk (Phase 0)"],
    summary="VIX volatility index forecast",
)
def get_market_vix() -> SeriesResult:
    """
    12-month LightGBM forecast for **VIXCLS** (CBOE Volatility Index).

    Generated automatically by `python fred_refresh.py` — no additional API keys required.
    Returns 404 if `outputs/results_market_risk.json` has not been generated yet.
    """
    group = _load_optional("market_risk")
    return _find_series(group, "VIXCLS")


@app.get(
    "/api/market/spreads",
    response_model=GroupResponse,
    tags=["Market Risk (Phase 0)"],
    summary="HY and IG credit spread forecasts",
)
def get_market_spreads() -> GroupResponse:
    """
    12-month LightGBM forecasts for US credit spreads:
    - **BAMLH0A0HYM2** — ICE BofA US High Yield OAS (%)
    - **BAMLC0A0CM** — ICE BofA US Investment Grade Corporate OAS (%)

    Wider spreads indicate credit stress. Generated automatically by `python fred_refresh.py`.
    """
    group = _load_optional("market_risk")
    return _filter_series(group, ["BAMLH0A0HYM2", "BAMLC0A0CM"])


@app.get(
    "/api/market/dollar",
    response_model=SeriesResult,
    tags=["Market Risk (Phase 0)"],
    summary="USD broad index forecast",
)
def get_market_dollar() -> SeriesResult:
    """
    12-month LightGBM forecast for **DTWEXBGS** (Nominal Broad U.S. Dollar Index, Jan 2006=100).

    Generated automatically by `python fred_refresh.py`.
    """
    group = _load_optional("market_risk")
    return _find_series(group, "DTWEXBGS")


# ── Commodities ───────────────────────────────────────────────────────────────

@app.get(
    "/api/commodities/oil",
    response_model=SeriesResult,
    tags=["Commodities (Phase 0)"],
    summary="WTI crude oil price forecast",
)
def get_commodities_oil() -> SeriesResult:
    """
    12-month LightGBM forecast for **DCOILWTICO** (WTI Crude Oil, $/bbl).

    Generated automatically by `python fred_refresh.py`.
    """
    group = _load_optional("commodities")
    return _find_series(group, "DCOILWTICO")


@app.get(
    "/api/commodities/gold",
    response_model=SeriesResult,
    tags=["Commodities (Phase 0)"],
    summary="Gold price forecast",
)
def get_commodities_gold() -> SeriesResult:
    """
    12-month LightGBM forecast for **NASDAQQGLDI** (Gold Price Index, NASDAQ).

    Generated automatically by `python fred_refresh.py`.
    """
    group = _load_optional("commodities")
    return _find_series(group, "NASDAQQGLDI")


# ── Yield Curve ───────────────────────────────────────────────────────────────

@app.get(
    "/api/market/yield-curve",
    response_model=GroupResponse,
    tags=["Market Risk (Phase 0)"],
    summary="Full Treasury yield curve forecasts (8 tenors)",
)
def get_yield_curve() -> GroupResponse:
    """
    12-month LightGBM forecasts for all 8 Treasury constant-maturity tenors:
    **DGS1MO**, **DGS3MO**, **DGS6MO**, **DGS1**, **DGS2**, **DGS5**, **DGS10**, **DGS30**.

    Generated automatically by `python fred_refresh.py`.
    """
    return _load_optional("yield_curve")


@app.get(
    "/api/market/yield-curve/{series_id}",
    response_model=SeriesResult,
    tags=["Market Risk (Phase 0)"],
    summary="Single Treasury tenor forecast",
)
def get_yield_curve_series(series_id: str) -> SeriesResult:
    """
    Retrieve the forecast for one Treasury tenor.

    Valid `series_id` values: **DGS1MO**, **DGS3MO**, **DGS6MO**, **DGS1**,
    **DGS2**, **DGS5**, **DGS10**, **DGS30**
    """
    group = _load_optional("yield_curve")
    return _find_series(group, series_id)


# ── Financial Stress Index ────────────────────────────────────────────────────

@app.get(
    "/api/market/stress",
    response_model=SeriesResult,
    tags=["Market Risk (Phase 0)"],
    summary="Financial Stress Index forecast",
)
def get_market_stress() -> SeriesResult:
    """
    12-month LightGBM forecast for the **Financial Stress Index (FSI)**.

    FSI = equal-weighted percentile-rank composite of VIX, HY spread, IG spread,
    recession probability, and yield curve inversion.
    Scale: 0 (calm) → 1 (extreme stress).
    Thresholds: < 0.25 expansion | 0.25–0.45 slowdown | 0.45–0.65 contraction | > 0.65 stress.

    Generated automatically by `python fred_refresh.py`.
    """
    group = _load_optional("financial_stress")
    return _find_series(group, "FSI")


# ── Market Regime ─────────────────────────────────────────────────────────────

@app.get(
    "/api/market/regime",
    response_model=RegimeResponse,
    tags=["Market Risk (Phase 0)"],
    summary="Market regime classification + history",
)
def get_market_regime() -> RegimeResponse:
    """
    Current market regime label, FSI value, 12-month regime forecast,
    and last 10 years of monthly regime history.

    Regime labels: **expansion** | **slowdown** | **contraction** | **stress** | **recovery**

    Generated automatically by `python fred_refresh.py`.
    """
    path = OPTIONAL_RESULTS["regime"]
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "regime_history.json not found. "
                "Run `python fred_refresh.py` to generate it."
            ),
        )
    try:
        raw = json.loads(path.read_text())
        return RegimeResponse(
            generated_at=raw["generated_at"],
            current_regime=raw["current_regime"],
            current_fsi=raw["current_fsi"],
            regime_distribution=raw["regime_distribution"],
            forecast=[RegimeForecastPoint(**p) for p in raw["forecast"]],
            history=[RegimeHistoryPoint(**h) for h in raw["history"]],
        )
    except Exception:
        log.exception("Failed to parse regime_history.json")
        raise HTTPException(
            status_code=500,
            detail="regime_history.json could not be parsed. Check server logs.",
        )


# ── Raw series history ────────────────────────────────────────────────────────

@app.get(
    "/api/series/{series_id}/history",
    response_model=HistoryResponse,
    tags=["Meta"],
    summary="Raw historical observations for any fetched series",
)
def get_series_history(series_id: str) -> HistoryResponse:
    """
    Return raw historical observations from the local CSV for any series that has
    been fetched by `fred_refresh.py`.

    Supported series: all FRED series (INDPRO, TCU, PAYEMS, DFF, VIXCLS, BAMLH0A0HYM2,
    DCOILWTICO, NASDAQQGLDI, DGS1MO … DGS30, etc.)
    """
    sid  = _validate_series_id(series_id)
    path = _HISTORY_FILE_MAP.get(sid)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Series '{sid}' is not in the history file map. "
                "Supported series: " + ", ".join(sorted(_HISTORY_FILE_MAP))
            ),
        )
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"CSV file for '{sid}' not found ({path}). "
                "Run `python fred_refresh.py` to fetch data."
            ),
        )
    try:
        df    = pd.read_csv(path, parse_dates=["observation_date"])
        col   = sid if sid in df.columns else df.columns[1]
        def _to_val(v: object) -> float | None:
            try:
                f = float(v)  # type: ignore[arg-type]
                return None if pd.isna(f) else round(f, 6)
            except (TypeError, ValueError):
                return None

        dates: list = df["observation_date"].dt.strftime("%Y-%m-%d").tolist()
        obs = [
            HistoryPoint(observation_date=str(d), value=_to_val(v))
            for d, v in zip(dates, df[col])
        ]
        return HistoryResponse(series_id=sid, row_count=len(obs), observations=obs)
    except Exception:
        log.exception("Failed to read history for %s", sid)
        raise HTTPException(
            status_code=500,
            detail=f"Could not read history for '{sid}'. Check server logs.",
        )


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
