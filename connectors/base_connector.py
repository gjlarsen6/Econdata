"""
base_connector.py — Abstract base class for enrichment connectors.

Each connector loads an external data source and joins it onto an events
DataFrame by date key, adding one or more new signal columns.
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseConnector(ABC):
    """Abstract enrichment connector.

    Subclasses must implement `load()` (fetch/prepare source data) and
    `enrich()` (join source data onto the events DataFrame).
    """

    @abstractmethod
    def load(self) -> pd.DataFrame | None:
        """Load and return source data as a DataFrame (or None if not applicable)."""
        ...

    @abstractmethod
    def enrich(self, events: pd.DataFrame) -> pd.DataFrame:
        """Join source signals onto *events* by date and return the augmented DataFrame."""
        ...
