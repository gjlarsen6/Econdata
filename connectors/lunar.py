"""
lunar.py — Lunar phase enrichment connector.

Ported from olddatapipeline/orig_ClickAI_Connect_99-MoonTOSales.py
Original algorithm by Sean B. Palmer, inamidst.com
Ref: https://en.wikipedia.org/wiki/Lunar_phase#Lunar_phase_calculation

Adds a `moon_phase` column (string) to an events DataFrame.
"""

import decimal
import math
from datetime import datetime

import pandas as pd

from .base_connector import BaseConnector

dec = decimal.Decimal


def position(now: datetime | None = None) -> decimal.Decimal:
    """Return the fractional lunation position (0.0–1.0) for *now*."""
    if now is None:
        now = datetime.now()
    diff = now - datetime(2001, 1, 1)
    days = dec(diff.days) + (dec(diff.seconds) / dec(86400))
    lunations = dec("0.20439731") + (days * dec("0.03386319269"))
    return lunations % dec(1)


def phase(pos: decimal.Decimal) -> str:
    """Map a lunation position to one of the 8 named lunar phases."""
    index = math.floor((pos * dec(8)) + dec("0.5"))
    return {
        0: "New Moon",
        1: "Waxing Crescent",
        2: "First Quarter",
        3: "Waxing Gibbous",
        4: "Full Moon",
        5: "Waning Gibbous",
        6: "Last Quarter",
        7: "Waning Crescent",
    }[int(index) & 7]


def moon_phase_for_date(dt: datetime) -> str:
    """Convenience wrapper: return the phase name for a datetime."""
    return phase(position(dt))


class LunarConnector(BaseConnector):
    """Adds a `moon_phase` column to an events DataFrame.

    Parameters
    ----------
    date_col : str
        Name of the date column in the events DataFrame (default: ``"date"``).
    """

    def __init__(self, date_col: str = "date") -> None:
        self.date_col = date_col

    def load(self) -> None:
        # No external data source — phase is computed purely from the date.
        return None

    def enrich(self, events: pd.DataFrame) -> pd.DataFrame:
        """Add ``moon_phase`` column to *events*."""
        df = events.copy()
        dates = pd.to_datetime(df[self.date_col])
        df["moon_phase"] = dates.apply(lambda d: moon_phase_for_date(d.to_pydatetime()))
        return df
