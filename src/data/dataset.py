"""In-memory containers for the workshop data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

import pandas as pd

from src.utils.config import DEFAULT_N_ASSETS

PandasObj = TypeVar("PandasObj", pd.Series, pd.DataFrame)


@dataclass(frozen=True)
class DateWindow:
    """Closed date interval ``[start, end]``.

    Attributes:
        name: Human-readable label (e.g. ``"train"``).
        start: First date included.
        end: Last date included.
    """

    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        """Validate the interval.

        Raises:
            ValueError: If ``start`` is after ``end``.
        """
        if self.start > self.end:
            raise ValueError(
                f"Window '{self.name}': start {self.start.date()} is after "
                f"end {self.end.date()}."
            )

    @classmethod
    def from_metadata(cls, name: str, spec: dict[str, str]) -> DateWindow:
        """Build a window from a ``{"start": ..., "end": ...}`` mapping.

        Args:
            name: Label of the window.
            spec: Mapping with ISO ``start`` and ``end`` dates.

        Returns:
            The corresponding window.

        Raises:
            ValueError: If a key is missing.
        """
        try:
            return cls(name, pd.Timestamp(spec["start"]),
                       pd.Timestamp(spec["end"]))
        except KeyError as exc:
            raise ValueError(
                f"Window '{name}' is missing key {exc} in metadata."
            ) from exc

    def slice(self, obj: PandasObj) -> PandasObj:
        """Restrict a date-indexed Series/DataFrame to this window.

        Args:
            obj: Object with a ``DatetimeIndex``.

        Returns:
            The rows whose date lies inside the window.
        """
        mask = (obj.index >= self.start) & (obj.index <= self.end)
        return obj[mask]


@dataclass(frozen=True)
class MarketDataset:
    """Everything downloaded from the workshop API.

    Attributes:
        metadata: Raw metadata (universe, windows, target, sectors...).
        prices_wide: Adjusted close prices, index = date, columns = ticker.
        benchmark: Benchmark (RSP) adjusted price series indexed by date.
    """

    metadata: dict[str, Any]
    prices_wide: pd.DataFrame
    benchmark: pd.Series

    @property
    def train_window(self) -> DateWindow:
        """Training window (the only data the autoencoder sees)."""
        return DateWindow.from_metadata(
            "train", self.metadata["windows"]["train"]
        )

    @property
    def validation_window(self) -> DateWindow:
        """Validation window (model selection only)."""
        return DateWindow.from_metadata(
            "validation", self.metadata["windows"]["validation"]
        )

    @property
    def n_assets(self) -> int:
        """Number of stocks in the tracker required by the competition."""
        return int(self.metadata.get("n_assets", DEFAULT_N_ASSETS))

    @property
    def sectors(self) -> dict[str, str]:
        """Mapping ticker -> GICS sector (empty if not published)."""
        return dict(self.metadata.get("sectors", {}))
