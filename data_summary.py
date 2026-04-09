"""
Economic Data Source Summary
Reads all FRED data files across four directories, computes stats live,
prints a formatted summary table, and suggests LightGBM feature strategies
for sales forecasting and customer pipeline models.
"""

import os
import pandas as pd
from tabulate import tabulate

# ── Data source manifest ──────────────────────────────────────────────────────
# (directory, filename, series_id, economic_concept, frequency)
SOURCES = [
    # BusinessEnvironment
    ("BusinessEnvironment", "INDPRO.csv",                          "INDPRO",          "Industrial Production Index (2017=100)",           "Monthly"),
    ("BusinessEnvironment", "TCU_capacityutilization.csv",         "TCU",             "Capacity Utilization — Total Industry",             "Monthly"),
    ("BusinessEnvironment", "Payroll_PAYEMS.csv",                  "PAYEMS",          "Total Nonfarm Payroll (thousands of persons)",      "Monthly"),
    ("BusinessEnvironment", "CAPUTLB50001SQ.csv",                  "CAPUTLB50001SQ",  "Capacity Utilization — Quarterly Version",          "Quarterly"),
    # ConsumerDemand
    ("ConsumerDemand",      "DSPIC96.csv",                         "DSPIC96",         "Real Disposable Personal Income (B chained $)",    "Monthly"),
    ("ConsumerDemand",      "PCE.csv",                             "PCE",             "Personal Consumption Expenditures (B $)",           "Monthly"),
    ("ConsumerDemand",      "PersConsume_noFoodEnergyPCEPILFE.csv","PCEPILFE",        "Core PCE Price Index (ex food & energy, 2017=100)", "Monthly"),
    ("ConsumerDemand",      "RSAFS.csv",                           "RSAFS",           "Nominal Retail & Food Services Sales (M $)",        "Monthly"),
    ("ConsumerDemand",      "RealRetailandFoodSalesRRSFS.csv",     "RRSFS",           "Real Retail & Food Services Sales (M chained $)",  "Monthly"),
    ("ConsumerDemand",      "UMCSENT.csv",                         "UMCSENT",         "U. of Michigan Consumer Sentiment (1966:Q1=100)",   "Monthly"),
    # CostOfCapital
    ("CostOfCapital",       "DFF.csv",                             "DFF",             "Federal Funds Effective Rate (%)",                  "Daily"),
    ("CostOfCapital",       "DPRIME.csv",                          "DPRIME",          "Bank Prime Loan Rate — Daily (%)",                  "Daily"),
    ("CostOfCapital",       "FEDFUNDS.csv",                        "FEDFUNDS",        "Federal Funds Rate — Monthly Average (%)",          "Monthly"),
    ("CostOfCapital",       "PRIME.csv",                           "PRIME",           "Prime Rate — Event-Based Changes Only (%)",         "Event"),
    ("CostOfCapital",       "T10Y2Y.csv",                          "T10Y2Y",          "Yield Curve Spread: 10-Year minus 2-Year (%)",      "Daily"),
    ("CostOfCapital",       "T10Y3M.csv",                          "T10Y3M",          "Yield Curve Spread: 10-Year minus 3-Month (%)",     "Daily"),
    # RiskLeadingInd
    ("RiskLeadingInd",      "RECPROUSM156N.csv",                   "RECPROUSM156N",   "Chauvet-Piger Smoothed Recession Probability (%)",  "Monthly"),
    ("RiskLeadingInd",      "UMCSENT.csv",                         "UMCSENT",         "Consumer Sentiment — duplicate in risk dir",        "Monthly"),
]

BASE_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Build summary rows ────────────────────────────────────────────────────────
rows = []
for directory, filename, series, concept, freq in SOURCES:
    path = os.path.join(BASE_DIR, directory, filename)
    try:
        df = pd.read_csv(path, parse_dates=["observation_date"])
        df = df.dropna(subset=[series])
        n_obs     = len(df)
        date_min  = df["observation_date"].min().strftime("%Y-%m")
        date_max  = df["observation_date"].max().strftime("%Y-%m")
        latest    = df[series].iloc[-1]
        latest_str = f"{latest:,.2f}"
    except Exception as e:
        n_obs, date_min, date_max, latest_str = "ERR", "—", "—", str(e)

    rows.append({
        "Directory":       directory,
        "File":            filename,
        "Series":          series,
        "Economic Concept":concept,
        "Freq":            freq,
        "Date Range":      f"{date_min} → {date_max}",
        "# Obs":           f"{n_obs:,}" if isinstance(n_obs, int) else n_obs,
        "Latest Value":    latest_str,
    })

# ── Print table ───────────────────────────────────────────────────────────────
print("=" * 140)
print("FRED ECONOMIC DATA SOURCE SUMMARY")
print("=" * 140)
print(tabulate(rows, headers="keys", tablefmt="rounded_outline", stralign="left"))

# ── Suggestions ───────────────────────────────────────────────────────────────
suggestions = """
════════════════════════════════════════════════════════════════════════════════════════════
SUGGESTIONS: Using Macroeconomic Data to Improve LightGBM Sales & Pipeline Models
════════════════════════════════════════════════════════════════════════════════════════════

─── A. Feature Engineering Recommendations (apply to all series) ─────────────────────────

  1. Temporal Alignment
     • Daily series (DFF, DPRIME, T10Y2Y, T10Y3M): resample to month-end using mean()
       before joining to monthly sales/pipeline records.
     • Quarterly series (CAPUTLB50001SQ): forward-fill to monthly using ffill().
     • Event-based series (PRIME): forward-fill from each change date to next change.

  2. Lag Features  — economic data leads business outcomes by weeks to months
     • Create lags at 1, 3, 6, and 12 months for each series.
     • Example: UMCSENT_lag3 captures consumer confidence 3 months prior to a sale.

  3. Rolling Statistics  — capture trend and volatility, not just point-in-time values
     • 3-month, 6-month, 12-month rolling mean and standard deviation.
     • Month-over-month % change (momentum): e.g., INDPRO_mom = INDPRO.pct_change(1)
     • Year-over-year % change: e.g., PAYEMS_yoy = PAYEMS.pct_change(12)

  4. Binary Signal Features  — encode regime changes as flags LightGBM can split on cleanly
     • Yield curve inverted:  T10Y2Y_inv  = 1 if T10Y2Y < 0 else 0
     • Deep inversion:        T10Y3M_inv  = 1 if T10Y3M < -0.5 else 0
     • Recession elevated:    RECPRO_high = 1 if RECPROUSM156N > 20 else 0
     • Rate hiking cycle:     DFF_rising  = 1 if DFF > DFF_lag3 else 0

─── B. Sales Forecasting Model ───────────────────────────────────────────────────────────

  Goal: predict future sales volume or revenue given macroeconomic conditions.

  Recommended primary features (high signal for consumer/business spending):
    • RSAFS / RRSFS     — retail sales momentum is a direct proxy for market demand
    • PCE               — broadest measure of consumer spending; tracks total market size
    • DSPIC96           — real disposable income drives purchasing power and willingness to buy
    • UMCSENT (lag 1-3) — sentiment leads actual spending; a falling index signals softness
    • PAYEMS_yoy        — employment growth drives income and discretionary spending

  Recommended secondary features:
    • INDPRO_yoy        — industrial production growth signals B2B demand tailwinds
    • TCU               — high capacity utilization means suppliers are stretched → price pressure
    • FEDFUNDS          — rising rates compress consumer credit availability → softer demand
    • PCEPILFE_mom      — core inflation momentum signals margin pressure and price sensitivity

  Suggested feature construction:
    sales_features = [
        "RSAFS_lag1", "RSAFS_roll3_mean", "RSAFS_yoy",
        "PCE_lag1", "PCE_roll6_mean",
        "DSPIC96_lag1", "DSPIC96_yoy",
        "UMCSENT_lag1", "UMCSENT_lag3", "UMCSENT_roll6_mean",
        "PAYEMS_yoy", "INDPRO_yoy",
        "FEDFUNDS_lag1", "PCEPILFE_mom",
    ]

─── C. Customer Pipeline / Deal Conversion Model ─────────────────────────────────────────

  Goal: predict whether pipeline opportunities convert, or forecast time-to-close,
        weighted by macro risk conditions.

  Recommended primary features (high signal for deal risk and buyer urgency):
    • DFF / FEDFUNDS     — financing cost directly affects customer capex decisions;
                           rising rates slow deal velocity, especially in capital-intensive sectors
    • T10Y3M             — the most reliable recession predictor; an inverted curve signals
                           that buyers will delay discretionary purchases
    • T10Y3M_inv (flag)  — binary inversion flag; easy for LightGBM to split on
    • RECPROUSM156N      — probability-weighted recession risk; use as a deal risk multiplier
    • RECPRO_high (flag) — high-risk regime flag (>20% probability)
    • UMCSENT_lag1       — buyer confidence; low sentiment correlates with slower pipeline movement

  Recommended secondary features:
    • PCEPILFE_mom       — inflation acceleration erodes budget certainty → delayed decisions
    • DSPIC96_lag3       — lagged income growth signals whether customers have budget headroom
    • DFF_rising (flag)  — rate hiking cycle flag; procurement teams freeze capex when rates rise
    • PAYEMS_yoy         — workforce growth at customer companies signals expanding budgets

  Suggested feature construction:
    pipeline_features = [
        "DFF_lag1", "DFF_lag3", "DFF_rising",
        "FEDFUNDS_lag1", "FEDFUNDS_roll3_mean",
        "T10Y3M_lag1", "T10Y3M_roll3_mean", "T10Y3M_inv",
        "T10Y2Y_lag1", "T10Y2Y_inv",
        "RECPROUSM156N_lag1", "RECPRO_high",
        "UMCSENT_lag1", "UMCSENT_lag3",
        "PCEPILFE_mom", "DSPIC96_lag3", "PAYEMS_yoy",
    ]

─── D. LightGBM-Specific Implementation Notes ────────────────────────────────────────────

  • Use monotone_constraints for features with known directional relationships:
      - DSPIC96 → sales: positive constraint (more income = more sales)
      - RECPROUSM156N → deal conversion: negative constraint (higher risk = fewer conversions)
      - DFF → deal conversion: negative constraint (higher rates = slower deals)

  • Leverage LightGBM's native handling of missing values — forward-filled NaNs from
    publication lags (e.g., UMCSENT released with 1-month delay) can be left as NaN
    and LightGBM will learn optimal split directions for them.

  • Use early_stopping with a macro-aware validation split: hold out the most recent
    12-24 months as validation rather than a random split, to respect time ordering
    and avoid leaking future economic conditions into training.

  • Consider adding interaction features between economic regime and your sales/pipeline
    data, e.g.: units_sold * UMCSENT_lag1, deal_size * RECPROUSM156N.
"""

print(suggestions)
