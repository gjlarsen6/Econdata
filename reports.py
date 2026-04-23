"""
reports.py — Generate a Markdown forecast report from all model result JSON files.

Usage:
    python3 reports.py

Output:
    reports/report_<YYYYMMDD_HHMMSS>.md
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent / "outputs"
REPORTS_DIR = Path(__file__).parent / "reports"

# All known result groups in display order. Files that don't exist are skipped.
RESULT_FILES = [
    # ── Core macro models ──────────────────────────────────────────────────────
    ("results_business_env",             "Business Environment"),
    ("results_consumer_demand",          "Consumer Demand"),
    ("results_cost_of_capital",          "Cost of Capital"),
    ("results_yield_curve",              "Treasury Yield Curve"),
    ("results_financial_stress",         "Financial Stress Index"),
    ("results_market_risk",              "Market Risk"),
    ("results_commodities",              "Commodities"),
    ("results_risk",                     "Risk & Leading Indicators"),
    # ── Industrial / Phase 1 ──────────────────────────────────────────────────
    ("results_industrial_production",    "Industrial Production by Sector"),
    ("results_industrial_ism_pmi",       "Manufacturing Orders Leading Indicators"),
    ("results_industrial_capacity_util", "Capacity Utilization by Sector"),
    ("results_industrial_credit",        "Credit & PPI Sector Indicators"),
    # ── Sector labor & structural ─────────────────────────────────────────────
    ("results_sector_bls",               "BLS Sector Employment"),
    ("results_sector_bls_wages",         "BLS Avg Hourly Earnings by Sector"),
    ("results_sector_bls_hours",         "BLS Avg Weekly Hours by Sector"),
    ("results_sector_worldbank",         "World Bank Sector GDP"),
    ("results_sector_bea",               "BEA Industry GDP"),
    ("results_sector_jolts",             "JOLTS Labor Turnover"),
    ("results_sector_etf",               "S&P 500 Sector ETFs"),
]

# Signal paragraphs keyed by group name. {placeholders} are filled with live data.
_SIGNALS: dict[str, str] = {
    "Business Environment": (
        "Industrial production is forecast to {indpro_dir} from {indpro_last} to {indpro_12m} "
        "over 12 months. Nonfarm payrolls are projected to {payems_dir}, with capacity utilization "
        "holding near {tcu_12m}%. Directional signals point to a manufacturing-led slowdown. "
        "Payroll and production models have high negative R² values, indicating extrapolation "
        "beyond the training distribution — treat magnitudes as approximate."
    ),
    "Consumer Demand": (
        "Consumer sentiment is the lone bright spot, recovering from {umcsent_last} toward "
        "{umcsent_12m} by year-end. However, actual spending metrics — PCE, disposable income, "
        "and retail sales — are all contracting in nominal terms. Core PCE easing toward "
        "{pcepilfe_12m} signals disinflation, which supports real purchasing power but is "
        "consistent with a demand-driven slowdown, not a supply shock."
    ),
    "Cost of Capital": (
        "The Fed funds rate is forecast to edge down from {dff_last}% to {dff_12m}% — a modest "
        "easing bias rather than an aggressive cut cycle. The 10Y−3M yield spread is steepening "
        "from {t10y3m_last}% to {t10y3m_12m}%, reflecting short rates falling faster than long "
        "rates — a classic post-inversion curve normalization. Bank prime rate tracks the funds "
        "rate closely and is expected to remain near current levels."
    ),
    "Treasury Yield Curve": (
        "A bull steepener is underway: front-end yields (1M through 2Y) are forecast to fall "
        "100–120bps over 12 months, while the long end (10Y, 30Y) stays anchored. The 10-year "
        "is forecast at {dgs10_12m}% and the 30-year at {dgs30_12m}% — persistent long-term "
        "yields reflect ongoing fiscal and inflation risk. The 2-year is forecast to decline to "
        "{dgs2_12m}%, widening the 10Y−2Y spread and confirming curve re-normalization."
    ),
    "Financial Stress Index": (
        "The FSI currently reads {fsi_last} — elevated relative to the neutral baseline near 0. "
        "The model forecasts a decline to {fsi_12m} and a flatline there, suggesting the stress "
        "episode is peaking but not fully unwinding. The flat forecast trajectory reflects model "
        "uncertainty in high-volatility regimes."
    ),
    "Market Risk": (
        "VIX is forecast to decline from {vix_last} to {vix_12m}, suggesting equity volatility "
        "is peaking. However, credit markets tell a more cautious story: HY spreads are forecast "
        "to widen from {hy_last}% to {hy_12m}%, and IG spreads from {ig_last}% to {ig_12m}%. "
        "Credit spread widening ahead of equity calm is a classic early-warning pattern. The USD "
        "broad index strengthening to {usd_12m} would add pressure on emerging markets and "
        "US multinationals. Credit spread models have limited BAML history (from 2023 only) — "
        "treat as directional signals."
    ),
    "Commodities": (
        "WTI crude at {wti_last}/bbl is forecast to drop sharply to ~{wti_12m}/bbl — a "
        "~{wti_pct}% decline consistent with the demand slowdown implied by other models. "
        "Gold (NASDAQQGLDI) at {gold_last} is forecast to decline to a low of {gold_3m} at "
        "3 months before partially recovering to {gold_12m} by year-end. Gold declining "
        "alongside macro stress may reflect a reversal from speculative premium or limited "
        "model sensitivity to safe-haven dynamics."
    ),
    "Risk & Leading Indicators": (
        "This is the most consequential signal: the recession probability model (RECPROUSM156N) "
        "starts from {rec_last}% and accelerates nonlinearly to {rec_6m}% at 6 months and "
        "{rec_12m}% at 12 months. A {rec_12m}% recession probability within a year is a "
        "material risk flag. The model has very high uncertainty (R² = {rec_r2}), indicating "
        "it is extrapolating outside its training distribution — treat as a directional risk "
        "signal, not a precise estimate. Consumer sentiment (also in this group) continues "
        "its recovery toward {umcsent2_12m}."
    ),
    "Industrial Production by Sector": (
        "Manufacturing IP ({ipman_last}) is forecast to {ipman_dir} to {ipman_12m} over 12 months. "
        "Mining IP ({ipmine_last}) and Utilities ({iputil_last}) are running well above pre-pandemic "
        "baselines and are forecast to {ipmine_dir} and {iputil_dir} respectively. Business "
        "Equipment IP ({ipbuseq_last} → {ipbuseq_12m}) is a proxy for capex momentum — "
        "declining business equipment output is an early signal of investment retrenchment. "
        "Consumer goods IP ({ipcongd_last} → {ipcongd_12m}) tracks final demand for manufactured goods."
    ),
    "Manufacturing Orders Leading Indicators": (
        "Durable goods new orders ({dgorder_last}M) are forecast to {dgorder_dir} to "
        "{dgorder_12m}M — the headline signal for manufacturing pipeline health. "
        "Nondefense capital goods orders (NEWORDER: {neworder_last}M → {neworder_12m}M) "
        "lead business investment by 3–6 months and are the most forward-looking series here. "
        "Unfilled orders ({amtuno_last}M → {amtuno_12m}M) indicate whether the production "
        "backlog is building or draining. Total manufacturing orders (SA) at {mnfctrirsa_last}M "
        "provide a broad cross-check. All are freely available Census Bureau series (replaced "
        "unlicensed ISM data)."
    ),
    "Capacity Utilization by Sector": (
        "Manufacturing capacity utilization (MCUMFN: {mcumfn_last}%) is forecast to "
        "{mcumfn_dir} to {mcumfn_12m}% over 12 months. Readings below 80% historically "
        "signal that manufacturers will defer capital expenditures — the current level "
        "suggests restrained investment ahead. Mining utilization ({caputlg211s_last}%) "
        "is running at elevated levels, reflecting the continued energy extraction buildout."
    ),
    "Credit & PPI Sector Indicators": (
        "Commercial & industrial loans ({busloans_last}) are forecast to {busloans_dir} — "
        "C&I loan growth leads business investment by 2–3 quarters. Real estate loans "
        "({realln_last} → {realln_12m}) reflect mortgage and commercial property credit "
        "conditions. Consumer loans ({consumer_last} → {consumer_12m}) track household "
        "borrowing momentum. PPI farm products ({wpu_farm_last}) "
        "capture commodity input cost pressure on producers — rising PPI ahead of soft demand "
        "compresses margins."
    ),
    "BLS Employment": (
        "Manufacturing employment ({mfg_emp_last}K) is forecast to {mfg_emp_dir} to "
        "{mfg_emp_12m}K, tracking the broader industrial slowdown. Construction "
        "({const_emp_last}K → {const_emp_12m}K) is sensitive to interest rate levels — "
        "elevated rates are a headwind. Professional & Business Services ({profsvc_emp_last}K "
        "→ {profsvc_emp_12m}K) is the largest private-sector cyclical indicator and typically "
        "leads aggregate payrolls. Government employment ({govt_emp_last}K) provides structural "
        "support to the headline. Information sector ({info_emp_last}K → {info_emp_12m}K) "
        "reflects ongoing tech-sector restructuring. Leisure & Hospitality ({leisure_emp_last}K) "
        "remains a post-pandemic recovery story."
    ),
    "BLS Avg Hourly Earnings by Sector": (
        "Wage growth is most concentrated in Information ({info_wages_last}/hr → "
        "{info_wages_12m}/hr) and Construction ({const_wages_last}/hr → {const_wages_12m}/hr). "
        "Manufacturing wages ({mfg_wages_last}/hr → {mfg_wages_12m}/hr) lag the knowledge "
        "sectors, reflecting the long-run bifurcation in labor market compensation. "
        "Financial sector wages ({finance_wages_last}/hr → {finance_wages_12m}/hr) are a "
        "leading indicator of white-collar labor market conditions. Persistent wage growth "
        "above 3% in any sector keeps core services inflation elevated — the Fed's primary "
        "inflation concern."
    ),
    "BLS Avg Weekly Hours by Sector": (
        "Weekly hours worked lead employment levels by 1–2 months: employers cut hours "
        "before headcount. Manufacturing hours ({mfg_hours_last} hrs/wk → {mfg_hours_12m} "
        "hrs/wk) near or below 40 hours signals utilization pressure. Construction hours "
        "({const_hours_last} hrs/wk → {const_hours_12m} hrs/wk) reflect weather and project "
        "pipeline timing. Professional services hours ({profsvc_hours_last} hrs/wk) are a "
        "proxy for white-collar workload — declining hours in this sector often precede "
        "layoffs in high-cost labor markets."
    ),
    "World Bank Sector GDP": (
        "The services sector accounts for {svc_gdp_pct_last}% of US GDP — forecast to "
        "{svc_gdp_pct_dir} to {svc_gdp_pct_12m}% over 12 months. Manufacturing's GDP share "
        "({mfg_gdp_pct_last}%) continues its long structural decline, while total industry "
        "(including mining, construction, utilities) holds at {ind_gdp_pct_last}%. "
        "These World Bank annual series update with a significant lag and are most useful "
        "for tracking structural composition trends rather than cyclical signals."
    ),
    "BEA Industry GDP": (
        "BEA industry-level GDP provides quarterly value-added data for all 61 NAICS "
        "industries. Key sectors to monitor: Finance & Insurance, Real Estate, and "
        "Professional Services together account for over 40% of private-sector GDP. "
        "Manufacturing value-added tracks closely with Industrial Production indices. "
        "These data are released with a 2–3 quarter lag."
    ),
    "JOLTS Labor Turnover": (
        "Job openings, hires, and quits rates are the leading edges of labor market "
        "rebalancing. Declining openings signal reduced labor demand before payroll "
        "growth slows. The quits rate — often called the 'take-this-job-and-shove-it "
        "index' — is a proxy for worker confidence and wage pressure. A falling quits "
        "rate signals workers are less willing to voluntarily leave, indicating labor "
        "market cooling and reduced upward wage pressure."
    ),
    "S&P 500 Sector ETFs": (
        "Sector ETF price momentum provides market-implied views on sector rotation. "
        "Defensive sectors (XLU: Utilities, XLP: Consumer Staples, XLV: Healthcare) "
        "outperforming cyclicals (XLY: Consumer Discretionary, XLI: Industrials) signals "
        "risk-off positioning consistent with late-cycle dynamics. Technology (XLK) and "
        "Communications (XLC) weighting in the S&P 500 makes their relative performance "
        "a dominant driver of index-level returns."
    ),
}

# Series ID → short alias for use in signal templates
_ALIASES: dict[str, str] = {
    # Core macro
    "VIXCLS":        "vix",
    "BAMLH0A0HYM2":  "hy",
    "BAMLC0A0CM":    "ig",
    "DTWEXBGS":      "usd",
    "DCOILWTICO":    "wti",
    "NASDAQQGLDI":   "gold",
    "RECPROUSM156N": "rec",
    "UMCSENT":       "umcsent",
    "INDPRO":        "indpro",
    "PAYEMS":        "payems",
    "TCU":           "tcu",
    "PCEPILFE":      "pcepilfe",
    "DFF":           "dff",
    "T10Y3M":        "t10y3m",
    "DGS2":          "dgs2",
    "DGS10":         "dgs10",
    "DGS30":         "dgs30",
    "FSI":           "fsi",
    # Industrial production
    "IPMAN":         "ipman",
    "IPUTIL":        "iputil",
    "IPMINE":        "ipmine",
    "IPCONGD":       "ipcongd",
    "IPBUSEQ":       "ipbuseq",
    "IPMAT":         "ipmat",
    "IPDCONGD":      "ipdcongd",
    "IPNCONGD":      "ipncongd",
    # Manufacturing orders (Census Bureau replacements for ISM)
    "NEWORDER":      "neworder",
    "DGORDER":       "dgorder",
    "AMTUNO":        "amtuno",
    "MNFCTRIRSA":    "mnfctrirsa",
    # Capacity utilization
    "MCUMFN":        "mcumfn",
    "CAPUTLG211S":   "caputlg211s",
    "CAPUTLG331S":   "cap_util_metals",   # new correct ID
    "CAPUTLB58SQ":   "cap_util_metals",   # old ID → same alias
    # Credit & PPI
    "BUSLOANS":      "busloans",
    "REALLN":        "realln",
    "CONSUMER":      "consumer",
    "WPU054":        "wpu_fuel",    # new correct ID
    "WPU05":         "wpu_fuel",    # old ID → same alias
    "WPU01":         "wpu_farm",    # new correct ID
    "WPU10":         "wpu_farm",    # old ID → same alias
    # BLS employment
    "CES1000000001": "mining_emp",
    "CES2000000001": "const_emp",
    "CES3000000001": "mfg_emp",
    "CES4000000001": "trade_emp",
    "CES5000000001": "info_emp",
    "CES5500000001": "finance_emp",
    "CES6000000001": "profsvc_emp",
    "CES9000000001": "govt_emp",
    "CEU6500000001": "edhealth_emp",
    "CEU7000000001": "leisure_emp",
    # BLS wages
    "CES2000000008": "const_wages",
    "CES3000000008": "mfg_wages",
    "CES4000000008": "trade_wages",
    "CES5000000008": "info_wages",
    "CES5500000008": "finance_wages",
    "CES6000000008": "profsvc_wages",
    # BLS hours
    "CES2000000007": "const_hours",
    "CES3000000007": "mfg_hours",
    "CES6000000007": "profsvc_hours",
    # World Bank
    "NV.IND.MANF.ZS": "mfg_gdp_pct",
    "NV.IND.TOTL.ZS":  "ind_gdp_pct",
    "NV.SRV.TOTL.ZS":  "svc_gdp_pct",
}


def _fmt(val: float | None, unit: str = "") -> str:
    """Format a value for display in a Markdown table."""
    if val is None:
        return "—"
    u = unit.lower()
    if "k persons" in u:
        return f"{val:,.0f}K"
    if u.startswith("b ") or u == "b $" or u == "b chained $":
        return f"${val:,.1f}B"
    if u.startswith("m ") or u == "m $" or u == "m chained $":
        return f"${val:,.0f}M"
    if unit in ("%", "%pts"):
        return f"{val:.2f}%"
    if val >= 10_000:
        return f"{val:,.0f}"
    if val >= 100:
        return f"{val:.2f}"
    if val >= 1:
        return f"{val:.4f}"
    return f"{val:.4f}"


def _fmt_plain(val: float | None, unit: str = "") -> str:
    """Same as _fmt but without markdown — used in signal paragraphs."""
    return _fmt(val, unit)


def _trend(last: float | None, fc12: float | None, series_id: str, unit: str) -> str:
    """Derive a trend label from last value vs +12M forecast."""
    if last is None or fc12 is None or last == 0:
        return "→ Stable"
    pct = (fc12 - last) / abs(last) * 100

    if series_id == "RECPROUSM156N":
        return "⚠ Rising" if fc12 > 10 else ("↑ Rising" if pct > 50 else "→ Low")
    if series_id == "FSI":
        return "↓ Easing" if pct < -2 else ("↑ Rising" if pct > 2 else "→ Stable")
    if series_id == "VIXCLS":
        return "↓ Calming" if pct < -2 else ("↑ Rising" if pct > 2 else "→ Stable")
    if unit in ("%", "%pts"):
        diff = fc12 - last
        if diff < -0.10:
            return "↓ Declining"
        if diff > 0.10:
            return "↑ Rising"
        return "→ Stable"
    if pct < -2:
        return "↓ Declining"
    if pct > 2:
        return "↑ Rising"
    return "→ Stable"


def _get_fc(series_data: dict, month_index: int) -> float | None:
    """Return the mid value at position month_index in the forecast list (0 = +1M)."""
    fc_list = series_data.get("forecast", [])
    if len(fc_list) > month_index:
        return fc_list[month_index].get("mid")
    return None


def _low_confidence(series_list: list[dict]) -> bool:
    """True if any series in the group has R² < -5."""
    for s in series_list:
        r2 = s.get("validation", {}).get("r2")
        if r2 is not None and r2 < -5:
            return True
    return False


def build_group_section(data: dict, section_num: int) -> str:
    group = data.get("group", "Unknown")
    series_list = data.get("series", [])

    lines: list[str] = []
    lines.append(f"## {section_num}. {group}")
    lines.append("")

    first_last_date = series_list[0].get("last_date", "") if series_list else ""
    lines.append(f"| Series | Label | Last ({first_last_date}) | +1M | +3M | +6M | +12M | Trend |")
    lines.append("|--------|-------|------|-----|-----|-----|------|-------|")

    for s in series_list:
        sid = s.get("series_id", "")
        label = s.get("label", sid)
        unit = s.get("unit", "")
        last_val = s.get("last_value")
        fc1 = _get_fc(s, 0)
        fc3 = _get_fc(s, 2)
        fc6 = _get_fc(s, 5)
        fc12 = _get_fc(s, 11)
        trend = _trend(last_val, fc12, sid, unit)

        row = (
            f"| `{sid}` | {label} | {_fmt(last_val, unit)} "
            f"| {_fmt(fc1, unit)} | {_fmt(fc3, unit)} | {_fmt(fc6, unit)} | {_fmt(fc12, unit)} "
            f"| {trend} |"
        )
        lines.append(row)

    lines.append("")

    # Signal paragraph
    signal_template = _SIGNALS.get(group, "").replace("%%", "%")
    if signal_template:
        subs: dict[str, str] = {}
        for s in series_list:
            sid = s.get("series_id", "")
            unit = s.get("unit", "")
            last_val = s.get("last_value")
            fc3 = _get_fc(s, 2)
            fc6 = _get_fc(s, 5)
            fc12 = _get_fc(s, 11)
            r2 = s.get("validation", {}).get("r2")

            # Register under both the lowercased series ID and its alias
            alias = _ALIASES.get(sid, sid.lower())
            for key in {sid.lower(), alias}:
                subs[f"{key}_last"] = _fmt_plain(last_val, unit)
                subs[f"{key}_3m"]   = _fmt_plain(fc3, unit)
                subs[f"{key}_6m"]   = _fmt_plain(fc6, unit)
                subs[f"{key}_12m"]  = _fmt_plain(fc12, unit)
                subs[f"{key}_r2"]   = f"{r2:.1f}" if r2 is not None else "n/a"
                if last_val is not None and fc12 is not None and last_val != 0:
                    pct = (fc12 - last_val) / abs(last_val) * 100
                    subs[f"{key}_dir"] = "decline" if pct < -2 else ("rise" if pct > 2 else "hold steady")

            # Consumer Sentiment appears in two groups — support umcsent2 alias in Risk group
            if sid == "UMCSENT":
                for suffix in ("_last", "_3m", "_6m", "_12m"):
                    subs[f"umcsent2{suffix}"] = subs.get(f"umcsent{suffix}", "—")

            # WTI: pct drop for commodities signal
            if sid == "DCOILWTICO" and last_val and fc12:
                drop = abs((fc12 - last_val) / last_val * 100)
                subs["wti_pct"] = f"{drop:.0f}"

            # Loan balances are in Billions of Dollars — show as $X.XB for signal clarity
            if sid in ("BUSLOANS", "REALLN", "CONSUMER") and last_val is not None:
                subs[f"{alias}_last"] = f"${last_val:,.0f}B"
                if fc12 is not None:
                    subs[f"{alias}_12m"] = f"${fc12:,.0f}B"

        # Use defaultdict so missing series render as "—" instead of raising KeyError
        safe_subs: dict = defaultdict(lambda: "—", subs)
        signal_text = signal_template.format_map(safe_subs).replace("%%", "%")

        lines.append(f"**Signal:** {signal_text}")
        lines.append("")

    if _low_confidence(series_list):
        lines.append(
            "*Note: One or more series in this group has R² < −5, indicating the model is "
            "extrapolating outside its training distribution. Treat forecasts as directional "
            "signals rather than precise estimates.*"
        )
        lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def build_narrative(all_data: list[dict]) -> str:
    """Build the overall macro narrative section using key values from all groups."""
    lookup: dict[str, dict] = {}
    for data in all_data:
        for s in data.get("series", []):
            lookup[s["series_id"]] = s

    def fc12(sid: str) -> float | None:
        s = lookup.get(sid)
        return _get_fc(s, 11) if s else None

    def fc6(sid: str) -> float | None:
        s = lookup.get(sid)
        return _get_fc(s, 5) if s else None

    def last(sid: str) -> float | None:
        s = lookup.get(sid)
        return s.get("last_value") if s else None

    # Helper: return first non-None value from a list of candidate series IDs
    def first_last(*sids: str) -> float | None:
        for sid in sids:
            v = last(sid)
            if v is not None:
                return v
        return None

    def first_fc12(*sids: str) -> float | None:
        for sid in sids:
            v = fc12(sid)
            if v is not None:
                return v
        return None

    rec_12m    = fc12("RECPROUSM156N")
    rec_6m_val = fc6("RECPROUSM156N")
    rec_last   = last("RECPROUSM156N")
    dff_12m    = fc12("DFF")
    t10y3m_12m = fc12("T10Y3M")
    vix_12m    = fc12("VIXCLS")
    vix_last   = last("VIXCLS")
    indpro_12m = fc12("INDPRO")
    indpro_last = last("INDPRO")
    umcsent_12m = fc12("UMCSENT")
    umcsent_last = last("UMCSENT")
    hy_12m     = fc12("BAMLH0A0HYM2")
    hy_last    = last("BAMLH0A0HYM2")
    wti_12m    = fc12("DCOILWTICO")
    wti_last   = last("DCOILWTICO")

    # Industrial / sector signals
    ipman_last  = last("IPMAN")
    ipman_12m   = fc12("IPMAN")
    mcumfn_last = last("MCUMFN")
    mcumfn_12m  = fc12("MCUMFN")
    busloans_last = last("BUSLOANS")
    busloans_12m  = fc12("BUSLOANS")
    mfg_emp_last  = last("CES3000000001")
    mfg_emp_12m   = fc12("CES3000000001")
    # Durable goods orders — use whichever series is present
    dgorder_last = first_last("DGORDER", "NAPMNEWO")
    dgorder_12m  = first_fc12("DGORDER", "NAPMNEWO")

    lines: list[str] = []
    lines.append("## Overall Macro Narrative")
    lines.append("")
    lines.append(
        "The models collectively describe a **late-cycle deceleration with rising tail risk**. "
        "Key signals across all model groups:"
    )
    lines.append("")

    bullets: list[str] = []

    # Recession risk
    if rec_12m is not None:
        level = "serious warning" if rec_12m > 20 else ("elevated" if rec_12m > 10 else "low but rising")
        rec_last_str = f"{rec_last:.2f}%" if rec_last is not None else "current"
        rec_6m_str   = f"{rec_6m_val:.1f}%" if rec_6m_val is not None else "—"
        bullets.append(
            f"**Recession risk is {level}.** The recession probability model forecasts "
            f"{rec_6m_str} at 6 months and **{rec_12m:.1f}% at 12 months** — a sharp nonlinear "
            f"acceleration from the {rec_last_str} baseline. Even discounting model uncertainty, "
            f"the trajectory warrants attention."
        )

    # Growth — macro IP
    if indpro_last and indpro_12m:
        chg = indpro_12m - indpro_last
        bullets.append(
            f"**Growth is decelerating.** Industrial production is forecast to "
            f"{'fall' if chg < 0 else 'rise'} from {indpro_last:.2f} to {indpro_12m:.2f} over 12 months. "
            f"Nonfarm payrolls, PCE, and retail sales are all contracting in the model forecasts, "
            f"pointing to a broad demand slowdown."
        )

    # Industrial production sub-sectors
    if ipman_last and ipman_12m:
        chg = ipman_12m - ipman_last
        direction = "contract" if chg < 0 else "expand"
        bullets.append(
            f"**Manufacturing IP is forecast to {direction}.** IP: Manufacturing moves from "
            f"{ipman_last:.2f} to {ipman_12m:.2f} — a {abs(chg):.2f}-point "
            f"{'decline' if chg < 0 else 'gain'}. Sub-sector divergences "
            f"(Mining vs Consumer Goods vs Business Equipment) reveal which segments are most "
            f"exposed to the demand slowdown."
        )

    # Capacity utilization
    if mcumfn_last:
        invest_signal = "restraining" if mcumfn_last < 80 else "supporting"
        bullets.append(
            f"**Capacity utilization is {invest_signal} capex.** Manufacturing utilization "
            f"(MCUMFN) is at {mcumfn_last:.1f}% — "
            f"{'below' if mcumfn_last < 80 else 'near'} the 80% threshold that historically "
            f"triggers capital investment decisions. "
            f"{'Firms are unlikely to add capacity in this environment.' if mcumfn_last < 80 else 'Firms may still invest to meet demand.'}"
        )

    # Durable goods orders
    if dgorder_last and dgorder_12m:
        chg_pct = (dgorder_12m - dgorder_last) / abs(dgorder_last) * 100
        bullets.append(
            f"**Manufacturing order book is {'weakening' if chg_pct < -2 else ('strengthening' if chg_pct > 2 else 'stable')}.** "
            f"Durable goods orders (DGORDER) are forecast to move from "
            f"${dgorder_last:,.0f}M to ${dgorder_12m:,.0f}M "
            f"({'−' if chg_pct < 0 else '+'}{abs(chg_pct):.1f}%) — this leads actual "
            f"manufacturing output by 2–3 months."
        )

    # Credit conditions
    if busloans_last and busloans_12m:
        chg = busloans_12m - busloans_last
        bullets.append(
            f"**Credit conditions are {'tightening' if chg < 0 else 'easing'}.** "
            f"C&I loans are forecast to {'decline' if chg < 0 else 'grow'} from "
            f"${busloans_last:,.0f}B to ${busloans_12m:,.0f}B (Billions of Dollars). "
            f"Loan contraction typically precedes business investment declines by 1–2 quarters."
        )

    # Labor market — sector employment
    if mfg_emp_last and mfg_emp_12m:
        chg = mfg_emp_12m - mfg_emp_last
        bullets.append(
            f"**Sector-level employment signals confirm the industrial slowdown.** "
            f"Manufacturing employment is forecast to {'fall' if chg < 0 else 'rise'} from "
            f"{mfg_emp_last:,.0f}K to {mfg_emp_12m:,.0f}K. Professional services and "
            f"government remain more resilient, providing a partial offset to goods-sector weakness."
        )

    # Fed / rates
    if dff_12m is not None:
        bullets.append(
            f"**The Fed is easing modestly.** The funds rate is forecast at {dff_12m:.2f}% in "
            f"12 months. The yield curve is re-steepening (10Y−3M spread widening to "
            f"{t10y3m_12m:.2f}%pts), with short-term yields falling 100+ bps — the market "
            f"is pricing in more cuts than the Fed is delivering."
        )

    # Credit vs equity divergence
    if vix_last and vix_12m and hy_last and hy_12m:
        bullets.append(
            f"**Credit is flashing caution while equities calm.** VIX is forecast to decline "
            f"from {vix_last:.1f} to {vix_12m:.1f} (equity volatility peaking), but HY credit "
            f"spreads are forecast to widen from {hy_last:.2f}% to {hy_12m:.2f}% — credit "
            f"markets are pricing in rising default risk ahead of equities."
        )

    # Consumer sentiment
    if umcsent_last and umcsent_12m:
        bullets.append(
            f"**Consumer sentiment is recovering.** UMCSENT rises from {umcsent_last:.1f} to "
            f"{umcsent_12m:.1f} by year-end — the one consistent bright spot. However, actual "
            f"spending (PCE, retail sales) is still contracting, suggesting sentiment may be "
            f"leading a real recovery or simply reflecting equity market stabilization."
        )

    # Commodities
    if wti_last and wti_12m:
        drop = (wti_12m - wti_last) / wti_last * 100
        bullets.append(
            f"**Commodity prices are normalizing.** WTI crude is forecast to fall "
            f"~{abs(drop):.0f}% from ${wti_last:.2f}/bbl to ${wti_12m:.2f}/bbl, consistent "
            f"with the demand slowdown implied by other models. PPI commodity prices also "
            f"bear watching as leading input cost signals."
        )

    for b in bullets:
        lines.append(f"- {b}")
        lines.append("")

    lines.append("### Bottom Line")
    lines.append("")
    if rec_12m is not None and rec_12m > 20:
        bottom = (
            f"The models favor a soft landing as the near-term base case — consumer sentiment "
            f"is recovering, financial stress is easing, and VIX is declining. However, the "
            f"12-month recession probability rising to **{rec_12m:.1f}%**, combined with "
            f"contracting production, payrolls, and consumer spending, signals that the window "
            f"for avoiding a harder landing is narrowing. Credit spread widening and declining "
            f"C&I loan growth are the most actionable near-term risk indicators to watch."
        )
    else:
        bottom = (
            "The models indicate a moderate slowdown with low near-term recession risk. "
            "Monitor credit spreads, durable goods orders, manufacturing capacity utilization, "
            "and the recession probability trajectory as the primary leading indicators."
        )
    lines.append(bottom)
    lines.append("")

    lines.append(
        "*Model quality note: Most series have negative R² values, reflecting the difficulty of "
        "forecasting macro variables in post-2020 regime-changing data. All forecasts should be "
        "treated as directional signals with wide uncertainty bands, not precise point estimates.*"
    )
    lines.append("")

    return "\n".join(lines)


def generate_report() -> Path:
    """Read all available model result files, build a Markdown report, and save it."""
    REPORTS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"report_{timestamp}.md"

    # Load all existing result files in display order
    loaded: list[tuple[str, dict]] = []   # (display_name, data)
    for fname, display in RESULT_FILES:
        path = OUTPUTS_DIR / f"{fname}.json"
        if path.exists():
            data = json.loads(path.read_text())
            loaded.append((display, data))

    if not loaded:
        raise FileNotFoundError(f"No result files found in {OUTPUTS_DIR}")

    run_at = loaded[0][1].get("run_at", "unknown")

    lines: list[str] = []
    lines.append("# Macroeconomic Model Forecast Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Models last run:** {run_at}  ")
    lines.append(f"**Groups included:** {len(loaded)}  ")
    lines.append(f"**Source files:** `outputs/results_*.json`  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Table of Contents")
    lines.append("")

    for i, (display, _) in enumerate(loaded, 1):
        anchor = display.lower().replace(" ", "-").replace("&", "").replace("--", "-")
        lines.append(f"{i}. [{display}](#{i}-{anchor})")
    lines.append(f"{len(loaded) + 1}. [Overall Macro Narrative](#overall-macro-narrative)")
    lines.append("")
    lines.append("---")
    lines.append("")

    all_data = [data for _, data in loaded]

    section_num = 1
    for _, data in loaded:
        lines.append(build_group_section(data, section_num))
        section_num += 1

    lines.append(build_narrative(all_data))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {report_path}")
    return report_path


if __name__ == "__main__":
    generate_report()
