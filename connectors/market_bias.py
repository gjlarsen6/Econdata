"""
market_bias.py — SPY SMA crossover enrichment connector.

Ported from:
  olddatapipeline/orig_ClickAI_Connect_5b-OFA_USMktBiasTOSales_PREP.py
    — cumsum_sma(), running_mean_uniform_filter1d()
  olddatapipeline/orig_ClickAI_Connect_5c-OFA_USMktBiasTOSales.py
    — crossover detection logic (findUSMktBiasEvent)

Adds two columns to an events DataFrame:
  spy_cross_9_20  — "Y" when SPY 9-day SMA > 20-day SMA, else "N"
  spy_cross_20_50 — "Y" when SPY 20-day SMA > 50-day SMA, else "N"

Missing market dates (weekends, holidays) are filled forward from the
last known signal (replacing the old global prev_w_evt_* carry-forward).

SPY daily close prices are fetched via yfinance and cached to
  data/Sector/ETF/SPY_daily.csv
in the same [observation_date, SPY] format used by refresh_sector_etfs().
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .base_connector import BaseConnector

log = logging.getLogger(__name__)

_DEFAULT_SPY_PATH = Path(__file__).parent.parent / "data" / "Sector" / "ETF" / "SPY_daily.csv"
_SPY_START = "1993-01-01"


# ── Utility functions (ported from 5b PREP script) ────────────────────────────

def cumsum_sma(array: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average via cumulative sum (ported from 5b PREP script).

    Returns an array of length ``len(array) - period + 1``.
    Equivalent to pandas ``Series.rolling(period).mean().dropna()``.
    """
    ret = np.cumsum(array, dtype=float)
    ret[period:] = ret[period:] - ret[:-period]
    return ret[period - 1:] / period


# ── Connector ─────────────────────────────────────────────────────────────────

class MarketBiasConnector(BaseConnector):
    """Adds SPY SMA crossover signal columns to an events DataFrame.

    Parameters
    ----------
    date_col : str
        Name of the date column in the events DataFrame (default: ``"date"``).
    spy_path : Path | None
        Path to the cached SPY daily CSV.  Defaults to
        ``data/Sector/ETF/SPY_daily.csv`` relative to the project root.
        If the file does not exist it will be fetched and saved automatically.
    from_date : str
        Earliest date to fetch when no local cache exists (default: ``"1993-01-01"``).
    """

    def __init__(
        self,
        date_col: str = "date",
        spy_path: Path | None = None,
        from_date: str = _SPY_START,
    ) -> None:
        self.date_col = date_col
        self.spy_path = Path(spy_path) if spy_path else _DEFAULT_SPY_PATH
        self.from_date = from_date

    # ── load ──────────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Return SPY daily close prices as a DataFrame with columns [date, close].

        Fetches from yfinance on first call and caches to ``self.spy_path``.
        On subsequent calls the local CSV is used; the last 30 days are
        refreshed to pick up any corrections.
        """
        self.spy_path.parent.mkdir(parents=True, exist_ok=True)

        start_date = self.from_date
        existing: pd.DataFrame | None = None

        if self.spy_path.exists():
            existing = pd.read_csv(self.spy_path, parse_dates=["observation_date"])
            existing.columns = ["date", "close"]
            if not existing.empty:
                last_date = existing["date"].max()
                start_date = (last_date - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
                log.info("MarketBias: SPY cache exists, refreshing from %s", start_date)

        fresh = self._fetch_spy(start_date)
        if fresh is None or fresh.empty:
            if existing is not None:
                return existing.sort_values("date").reset_index(drop=True)
            raise RuntimeError("Could not fetch SPY data and no local cache exists.")

        if existing is not None:
            combined = pd.concat([existing, fresh], ignore_index=True)
            combined = combined.drop_duplicates(subset="date", keep="last")
        else:
            combined = fresh

        combined = combined.sort_values("date").reset_index(drop=True)

        # Persist cache in the standard project format
        out = combined.rename(columns={"date": "observation_date", "close": "SPY"})
        out.to_csv(self.spy_path, index=False)
        log.info("MarketBias: SPY cache updated → %s (%d rows)", self.spy_path, len(out))

        return combined

    def _fetch_spy(self, start_date: str) -> pd.DataFrame | None:
        try:
            import yfinance as yf
        except ImportError:
            log.error("yfinance not installed — pip install yfinance")
            return None

        t0 = time.time()
        try:
            tkr = yf.Ticker("SPY")
            hist = tkr.history(start=start_date, interval="1d", auto_adjust=True)
            if hist.empty:
                return None
            hist = hist[["Close"]].copy()
            hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
            hist = hist.reset_index()
            hist.columns = ["date", "close"]
            log.info("MarketBias: fetched %d SPY rows in %.1fs", len(hist), time.time() - t0)
            return hist
        except Exception as exc:
            log.warning("MarketBias: yfinance fetch failed — %s", exc)
            return None

    # ── enrich ────────────────────────────────────────────────────────────────

    def enrich(self, events: pd.DataFrame) -> pd.DataFrame:
        """Join SPY SMA crossover signals onto *events* by date.

        New columns added:
          ``spy_cross_9_20``  — "Y"/"N" (9-day SMA above 20-day SMA)
          ``spy_cross_20_50`` — "Y"/"N" (20-day SMA above 50-day SMA)

        Missing dates (weekends, holidays) are filled forward from the last
        known market day (replacing the global prev_w_evt_* pattern in the
        original script).
        """
        spy = self.load()
        signals = self._compute_signals(spy)

        df = events.copy()
        df["_merge_date"] = pd.to_datetime(df[self.date_col]).dt.normalize()

        # Left-join so every event row is preserved
        merged = df.merge(
            signals.rename(columns={"date": "_merge_date"}),
            on="_merge_date",
            how="left",
        )
        merged = merged.sort_values("_merge_date")

        # Carry forward last known signal for event dates with no market data
        merged["spy_cross_9_20"] = merged["spy_cross_9_20"].ffill()
        merged["spy_cross_20_50"] = merged["spy_cross_20_50"].ffill()

        merged = merged.drop(columns=["_merge_date"])
        return merged.reset_index(drop=True)

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_signals(spy: pd.DataFrame) -> pd.DataFrame:
        """Compute 9/20/50-day SMAs and crossover signals from daily close prices."""
        df = spy.copy().sort_values("date")
        close = df["close"].values

        df["sma_9"]  = pd.Series(close, index=df.index).rolling(9,  min_periods=9).mean()
        df["sma_20"] = pd.Series(close, index=df.index).rolling(20, min_periods=20).mean()
        df["sma_50"] = pd.Series(close, index=df.index).rolling(50, min_periods=50).mean()

        has_9_20  = df["sma_9"].notna() & df["sma_20"].notna()
        has_20_50 = df["sma_20"].notna() & df["sma_50"].notna()

        df["spy_cross_9_20"]  = pd.NA
        df["spy_cross_20_50"] = pd.NA
        df.loc[has_9_20,  "spy_cross_9_20"]  = (df.loc[has_9_20,  "sma_9"]  > df.loc[has_9_20,  "sma_20"]).map({True: "Y", False: "N"})
        df.loc[has_20_50, "spy_cross_20_50"] = (df.loc[has_20_50, "sma_20"] > df.loc[has_20_50, "sma_50"]).map({True: "Y", False: "N"})

        return pd.DataFrame(df[["date", "spy_cross_9_20", "spy_cross_20_50"]])
