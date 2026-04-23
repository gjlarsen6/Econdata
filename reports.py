"""
reports.py — Generate a Markdown forecast report from all model result JSON files.

Usage:
    python3 reports.py

Output:
    reports/report_<YYYYMMDD_HHMMSS>.md
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent / "outputs"
REPORTS_DIR = Path(__file__).parent / "reports"

RESULT_FILES = [
    ("results_business_env",     "Business Environment"),
    ("results_consumer_demand",  "Consumer Demand"),
    ("results_cost_of_capital",  "Cost of Capital"),
    ("results_yield_curve",      "Treasury Yield Curve"),
    ("results_financial_stress", "Financial Stress Index"),
    ("results_market_risk",      "Market Risk"),
    ("results_commodities",      "Commodities"),
    ("results_risk",             "Risk & Leading Indicators"),
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
        # absolute change for rates
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

    # Table header
    # Determine the last_date from first series for column label
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
        # Short alias overrides so templates can use readable names like {vix_last}
        _ALIASES: dict[str, str] = {
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
        }

        # Build substitution dict from this group's series
        subs: dict[str, str] = {}
        for s in series_list:
            sid = s.get("series_id", "")
            unit = s.get("unit", "")
            last_val = s.get("last_value")
            fc3 = _get_fc(s, 2)
            fc6 = _get_fc(s, 5)
            fc12 = _get_fc(s, 11)
            r2 = s.get("validation", {}).get("r2")

            # Register under both the raw lowercased ID and any alias
            for key in {sid.lower(), _ALIASES.get(sid, sid.lower())}:
                subs[f"{key}_last"] = _fmt_plain(last_val, unit)
                subs[f"{key}_3m"] = _fmt_plain(fc3, unit)
                subs[f"{key}_6m"] = _fmt_plain(fc6, unit)
                subs[f"{key}_12m"] = _fmt_plain(fc12, unit)
                subs[f"{key}_r2"] = f"{r2:.1f}" if r2 is not None else "n/a"
                if last_val is not None and fc12 is not None and last_val != 0:
                    pct = (fc12 - last_val) / abs(last_val) * 100
                    subs[f"{key}_dir"] = "decline" if pct < -2 else ("rise" if pct > 2 else "hold steady")

            # Consumer Sentiment appears in two groups — support umcsent2 alias for Risk group
            if sid == "UMCSENT":
                for suffix in ("_last", "_3m", "_6m", "_12m"):
                    subs[f"umcsent2{suffix}"] = subs.get(f"umcsent{suffix}", "—")

            # WTI special: pct drop
            if sid == "DCOILWTICO" and last_val and fc12:
                drop = abs((fc12 - last_val) / last_val * 100)
                subs["wti_pct"] = f"{drop:.0f}"

        try:
            signal_text = signal_template.format_map(subs)
            # format_map may double % signs when values already contain %; clean up
            signal_text = signal_text.replace("%%", "%")
        except KeyError:
            signal_text = signal_template  # fallback: show raw template

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
    # Index all series by series_id for easy lookup
    lookup: dict[str, dict] = {}
    for data in all_data:
        for s in data.get("series", []):
            lookup[s["series_id"]] = s

    def fc12(sid: str) -> float | None:
        s = lookup.get(sid)
        return _get_fc(s, 11) if s else None

    def last(sid: str) -> float | None:
        s = lookup.get(sid)
        return s.get("last_value") if s else None

    rec_12m = fc12("RECPROUSM156N")
    rec_6m = _get_fc(lookup.get("RECPROUSM156N", {}), 5)
    dff_12m = fc12("DFF")
    vix_12m = fc12("VIXCLS")
    vix_last = last("VIXCLS")
    indpro_12m = fc12("INDPRO")
    indpro_last = last("INDPRO")
    umcsent_12m = fc12("UMCSENT")
    umcsent_last = last("UMCSENT")
    hy_12m = fc12("BAMLH0A0HYM2")
    hy_last = last("BAMLH0A0HYM2")
    t10y3m_12m = fc12("T10Y3M")

    lines: list[str] = []
    lines.append("## Overall Macro Narrative")
    lines.append("")
    lines.append(
        "The models collectively describe a **late-cycle deceleration with rising tail risk**. "
        "Key signals across all eight model groups:"
    )
    lines.append("")

    bullets: list[str] = []

    # Recession risk
    if rec_12m is not None:
        level = "serious warning" if rec_12m > 20 else ("elevated" if rec_12m > 10 else "low but rising")
        bullets.append(
            f"**Recession risk is {level}.** The recession probability model forecasts "
            f"{rec_6m:.1f}% at 6 months and **{rec_12m:.1f}% at 12 months** — a sharp nonlinear "
            f"acceleration from the current {last('RECPROUSM156N'):.2f}% baseline. Even discounting "
            f"model uncertainty, the trajectory warrants attention."
        )

    # Growth
    if indpro_last and indpro_12m:
        chg = indpro_12m - indpro_last
        bullets.append(
            f"**Growth is decelerating.** Industrial production is forecast to "
            f"{'fall' if chg < 0 else 'rise'} from {indpro_last:.2f} to {indpro_12m:.2f} over 12 months. "
            f"Nonfarm payrolls, PCE, and retail sales are all contracting in the model forecasts, "
            f"pointing to a broad demand slowdown."
        )

    # Fed / rates
    if dff_12m is not None:
        bullets.append(
            f"**The Fed is easing modestly.** The funds rate is forecast at {dff_12m:.2f}% in "
            f"12 months. The yield curve is re-steepening (10Y−3M spread widening to "
            f"{t10y3m_12m:.2f}% pts), with short-term yields falling 100+ bps — the market "
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
            f"{umcsent_12m:.1f} by year-end — the one consistent bright spot across the "
            f"consumer demand models. However, actual spending (PCE, retail sales) is still "
            f"contracting, suggesting sentiment may be leading a real recovery or simply "
            f"reflecting equity market stabilization."
        )

    # Commodities
    wti_12m = fc12("DCOILWTICO")
    wti_last = last("DCOILWTICO")
    if wti_last and wti_12m:
        drop = (wti_12m - wti_last) / wti_last * 100
        bullets.append(
            f"**Commodity prices are normalizing.** WTI crude is forecast to fall "
            f"~{abs(drop):.0f}% from ${wti_last:.2f}/bbl to ${wti_12m:.2f}/bbl, consistent "
            f"with the demand slowdown implied by other models. Gold is forecast to partially "
            f"retrace its recent highs."
        )

    for b in bullets:
        lines.append(f"- {b}")
        lines.append("")

    # Bottom line
    lines.append("### Bottom Line")
    lines.append("")
    if rec_12m is not None and rec_12m > 20:
        bottom = (
            f"The models favor a soft landing as the near-term base case — consumer sentiment "
            f"is recovering, financial stress is easing, and VIX is declining. However, the "
            f"12-month recession probability rising to **{rec_12m:.1f}%**, combined with "
            f"contracting production, payrolls, and consumer spending, signals that the window "
            f"for avoiding a harder landing is narrowing. Credit spread widening is the most "
            f"actionable near-term risk indicator to watch."
        )
    else:
        bottom = (
            "The models indicate a moderate slowdown with low near-term recession risk. "
            "Monitor credit spreads, payroll trends, and the recession probability trajectory "
            "as the primary leading indicators."
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
    """Read all model result files, build a Markdown report, and save it to reports/."""
    REPORTS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"report_{timestamp}.md"

    all_data: list[dict] = []
    for fname, _ in RESULT_FILES:
        path = OUTPUTS_DIR / f"{fname}.json"
        if path.exists():
            all_data.append(json.loads(path.read_text()))

    if not all_data:
        raise FileNotFoundError(f"No result files found in {OUTPUTS_DIR}")

    # Determine overall run timestamp from first file
    run_at = all_data[0].get("run_at", "unknown") if all_data else "unknown"

    lines: list[str] = []
    lines.append("# Macroeconomic Model Forecast Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Models last run:** {run_at}  ")
    lines.append(f"**Source files:** `outputs/results_*.json`  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Table of Contents")
    lines.append("")
    for i, (_, display) in enumerate(RESULT_FILES, 1):
        anchor = display.lower().replace(" ", "-").replace("&", "").replace("--", "-")
        lines.append(f"{i}. [{display}](#{i}-{anchor})")
    lines.append(f"{len(RESULT_FILES) + 1}. [Overall Macro Narrative](#overall-macro-narrative)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Load each group in display order
    section_num = 1
    for fname, _ in RESULT_FILES:
        path = OUTPUTS_DIR / f"{fname}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        lines.append(build_group_section(data, section_num))
        section_num += 1

    lines.append(build_narrative(all_data))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {report_path}")
    return report_path


if __name__ == "__main__":
    generate_report()
