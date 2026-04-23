# Sector Analysis Expansion Plan

## Current State

The sector framework has **13 series across 3 sources**. No data has been fetched yet — `data/Sector/` does not exist. All ingestion and model infrastructure is ready to run.

| Source | Current Series | Gap |
|---|---|---|
| BLS (`sector_apis.py`) | 6 employment series | Only 6 of ~300+ BLS series |
| BEA (`sector_apis.py`) | 4 GDP-by-industry series | Only 4 of 61 BEA industries |
| World Bank (`sector_apis.py`) | 3 annual % GDP shares | Annual frequency, low forecasting value |

---

## Priority 1 — Quick Wins: Extend Existing Dicts (no new infrastructure)

### 1a. BLS: 4 Missing Employment Sectors

**File:** `sector_apis.py` → `BLS_SERIES_IDS` dict  
**Effort:** Add 4 lines to the dict. Existing `refresh_bls()` handles everything else.

| Series ID | Label | Why it matters |
|---|---|---|
| `CES1000000001` | Mining & Logging Employment | Energy sector employment proxy |
| `CES2000000001` | Construction Employment | Housing/capex leading indicator |
| `CES5000000001` | Information Employment | Tech sector employment |
| `CES9000000001` | Government Employment | Fiscal employment baseline |

Brings BLS to 10 sectors — full private-sector NAICS coverage.

### 1b. BEA: 6 Additional Industries

**File:** `sector_apis.py` → `BEA_INDUSTRY_MAP` dict  
**Effort:** Add 6 lines. `fetch_bea_gdp_by_industry()` already fetches all 61 industries in one call — only the pivot filter needs updating.

| BEA IndustryID | Column Name | Why it matters |
|---|---|---|
| `11` | `BEA_Agriculture` | Commodity/food sector GDP |
| `22` | `BEA_Utilities` | Energy/grid sector GDP |
| `23` | `BEA_Construction` | Capital investment proxy |
| `51` | `BEA_Information` | Tech sector GDP |
| `62` | `BEA_Healthcare` | Largest US service sector |
| `71-72` | `BEA_Arts_Hospitality` | Consumer cyclical |

Nearly triples BEA coverage at zero additional API cost.

---

## Priority 2 — FRED Additions (leverage existing ingestion pipeline)

Add series to `data/fred_ingestion_map_full_production.json`. They will be fetched by `fred_refresh.py` automatically and can be modeled via a new `industrial_model.py` or added to `market_model.py`.

### 2a. Industrial Production by Sector

| FRED Series | Label | History |
|---|---|---|
| `IPMAN` | IP: Manufacturing | 1919– |
| `IPUTIL` | IP: Utilities | 1939– |
| `IPMINE` | IP: Mining | 1919– |
| `IPCONGD` | IP: Consumer Goods | 1939– |
| `IPBUSEQ` | IP: Business Equipment | 1939– |
| `IPMAT` | IP: Materials | 1939– |
| `IPDCONGD` | IP: Durable Consumer Goods | 1939– |
| `IPNCONGD` | IP: Nondurable Consumer Goods | 1939– |

Complements total `INDPRO` (already tracked) with sector-level breakdowns. Long history → strong model training data.

### 2b. ISM PMI — Key Leading Indicators

ISM PMI moves 1–3 months before actual activity data. New Orders sub-index is the most forward-looking, predicting GDP direction 2–3 months out.

| FRED Series | Label | Signal |
|---|---|---|
| `NAPM` | ISM Manufacturing PMI | >50 = expansion |
| `NMFCI` | ISM Non-Manufacturing (Services) PMI | >50 = expansion |
| `NAPMPROD` | ISM Manufacturing: Production | Activity sub-index |
| `NAPMNEWO` | ISM Manufacturing: New Orders | **Best leading sub-index** |
| `NAPMEMPL` | ISM Manufacturing: Employment | Employment forward signal |
| `NAPMVNDR` | ISM: Vendor Deliveries | Supply chain stress proxy |

### 2c. Capacity Utilization by Sector

| FRED Series | Label |
|---|---|
| `MCUMFN` | Manufacturing Capacity Utilization |
| `CAPUTLG211S` | Mining Capacity Utilization |
| `CAPUTLB58SQ` | Durable Goods Capacity Utilization |

---

## Priority 3 — BLS Subgroups (new SOURCE_CONFIG entries)

Add new subgroup keys to `SOURCE_CONFIG` in `sector_model.py` (`bls_wages`, `bls_hours`, `bls_jolts`). Fetchers use the same `refresh_bls()` / `fetch_bls_series()` logic — only new series IDs and a new `SOURCE_CONFIG` block are needed.

### 3a. BLS: Average Hourly Earnings by Sector

Sector-level wage data is a direct input-cost inflation signal. Series suffix `008` = avg hourly earnings.

| Series ID | Label |
|---|---|
| `CES3000000008` | Manufacturing Avg Hourly Earnings |
| `CES4000000008` | Trade/Transport Avg Hourly Earnings |
| `CES5500000008` | Financial Avg Hourly Earnings |
| `CES6000000008` | Professional Services Avg Hourly Earnings |
| `CES2000000008` | Construction Avg Hourly Earnings |
| `CES5000000008` | Information Avg Hourly Earnings |

### 3b. BLS: Average Weekly Hours by Sector

Weekly hours are a leading employment indicator — firms cut hours before headcount. Series suffix `007` = avg weekly hours.

| Series ID | Label |
|---|---|
| `CES3000000007` | Manufacturing Avg Weekly Hours |
| `CES6000000007` | Professional Services Avg Weekly Hours |
| `CES2000000007` | Construction Avg Weekly Hours |

### 3c. JOLTS: Job Openings by Sector

Job openings turn before payrolls do — a forward-looking labor market signal. Same BLS API endpoint, already wired up in `refresh_bls()`.

| Series ID | Label |
|---|---|
| `JTS3000JOL` | Job Openings: Manufacturing |
| `JTS4000JOL` | Job Openings: Trade/Transport |
| `JTS5500JOL` | Job Openings: Financial |
| `JTS6000JOL` | Job Openings: Professional Services |
| `JTS6500JOL` | Job Openings: Education & Health |
| `JTS7000JOL` | Job Openings: Leisure & Hospitality |
| `JTS2300JOL` | Job Openings: Construction |

Pair with employment levels to compute vacancy-to-employment ratio per sector — a tight labor market signal.

---

## Priority 4 — S&P 500 Sector ETFs via yfinance (new fetcher)

**Why:** ETF prices give a **market-implied**, forward-looking view of sector strength that leads lagging government data by months. The project already uses `yfinance` in `enrichment_apis.py`.

**What to build:**
1. New fetcher in `sector_apis.py`: `refresh_sector_etfs(symbols, from_date)` — thin yfinance wrapper, saves monthly close prices to `data/Sector/ETF/{ticker}.csv`
2. New `SOURCE_CONFIG` entry in `sector_model.py`: `"etf"` → `data/Sector/ETF/`
3. New `--sector etf` option wired into `fred_refresh.py`

| Ticker | Sector |
|---|---|
| `XLK` | Technology |
| `XLF` | Financials |
| `XLV` | Healthcare |
| `XLE` | Energy |
| `XLI` | Industrials |
| `XLP` | Consumer Staples |
| `XLY` | Consumer Discretionary |
| `XLU` | Utilities |
| `XLRE` | Real Estate |
| `XLB` | Materials |
| `XLC` | Communication Services |

---

## Priority 5 — Additional FRED Credit / Cost Indicators

| FRED Series | Label | Sector Proxy |
|---|---|---|
| `BUSLOANS` | Commercial & Industrial Loans | Business credit |
| `REALLN` | Real Estate Loans | Real estate credit |
| `CONSUMER` | Consumer Loans | Consumer credit |
| `WPU05` | PPI: Fuels & Related | Energy input costs |
| `WPU10` | PPI: Farm Products | Agriculture input costs |

---

## Implementation Order

| Step | Item | File(s) to change | New series | Effort |
|---|---|---|---|---|
| 1 | BLS: 4 missing employment sectors | `sector_apis.py` | +4 | Trivial — add dict entries |
| 2 | BEA: 6 additional industries | `sector_apis.py` | +6 | Trivial — add dict entries |
| 3 | FRED: ISM PMI + sub-indices | `fred_ingestion_map_full_production.json` | +6 | Low — add JSON entries |
| 4 | FRED: Industrial production by sector | `fred_ingestion_map_full_production.json` | +8 | Low — add JSON entries |
| 5 | FRED: Capacity utilization by sector | `fred_ingestion_map_full_production.json` | +3 | Low — add JSON entries |
| 6 | BLS: Hourly earnings by sector | `sector_apis.py`, `sector_model.py` | +6 | Low — new BLS subgroup |
| 7 | BLS: Weekly hours by sector | `sector_apis.py`, `sector_model.py` | +3 | Low — new BLS subgroup |
| 8 | JOLTS: Job openings by sector | `sector_apis.py`, `sector_model.py` | +7 | Low — new BLS subgroup |
| 9 | Sector ETFs via yfinance | `sector_apis.py`, `sector_model.py`, `fred_refresh.py` | +11 | Moderate — new fetcher |
| 10 | FRED: Credit / PPI by sector | `fred_ingestion_map_full_production.json` | +5 | Low — add JSON entries |

**Total new series at completion: ~59 additional series across all priorities.**

---

## Notes

- Steps 1–2 require only editing two Python dicts. Run `python3 fred_refresh.py --sector all` immediately after to fetch data and train models.
- Steps 3–5 require adding entries to the FRED ingestion JSON map and deciding which model group they belong to (new `industrial_model.py` recommended for IP series, or extend `market_model.py`).
- Steps 6–8 use the existing BLS API infrastructure; only new `SOURCE_CONFIG` blocks in `sector_model.py` are required.
- Step 9 (sector ETFs) is the highest-value addition for forward-looking analysis but requires the most new code (~50 lines for the fetcher).
- World Bank series (annual frequency, forward-filled to monthly) have limited forecasting value and are not prioritized for expansion.
