"""Small helper classes that mimic the subset of pandas used in the project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence
from datetime import datetime, timedelta


def _ensure_length(values: Dict[str, List[float]]) -> int:
    lengths = {len(v) for v in values.values()}
    if len(lengths) != 1:
        raise ValueError("Column lengths mismatch in time series table")
    return lengths.pop() if lengths else 0


@dataclass
class TimeSeriesTable:
    """Simple table structure storing equally spaced time series data."""

    index: List[datetime]
    columns: List[str]
    data: Dict[str, List[float]]

    def __post_init__(self) -> None:
        n_rows = len(self.index)
        for col in self.columns:
            if col not in self.data:
                raise KeyError(f"Missing column data for {col!r}")
            if len(self.data[col]) != n_rows:
                raise ValueError(f"Column {col!r} has wrong length")

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self) -> Iterator[str]:
        return iter(self.columns)

    def __getitem__(self, key: str) -> List[float]:
        return self.data[key]

    def as_rows(self) -> Iterator[Dict[str, float]]:
        for i in range(len(self.index)):
            yield {col: self.data[col][i] for col in self.columns}

    def copy(self) -> "TimeSeriesTable":
        return TimeSeriesTable(self.index[:], self.columns[:], {k: v[:] for k, v in self.data.items()})

    def to_dict(self) -> Dict[str, List[float]]:
        return {k: v[:] for k, v in self.data.items()}

    def subset(self, cols: Sequence[str]) -> "TimeSeriesTable":
        return TimeSeriesTable(self.index[:], list(cols), {c: self.data[c][:] for c in cols})

    def ensure_frequency(self, dt_hours: float) -> None:
        if len(self.index) < 2:
            return
        expected = timedelta(hours=dt_hours)
        for prev, curr in zip(self.index, self.index[1:]):
            if curr - prev != expected:
                raise ValueError("Time index is not evenly spaced")

    def column_stats(self, name: str) -> Dict[str, float]:
        series = self.data[name]
        if not series:
            return {"name": name, "n": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
        total = sum(series)
        return {
            "name": name,
            "n": len(series),
            "min": float(min(series)),
            "max": float(max(series)),
            "mean": float(total / len(series)),
        }


def forward_fill(values: List[float]) -> List[float]:
    out = []
    last = None
    for v in values:
        if v is None or v != v:  # NaN check
            out.append(last)
        else:
            out.append(v)
            last = v
    return out


def backward_fill(values: List[float]) -> List[float]:
    out = values[:]
    next_val = None
    for i in range(len(out) - 1, -1, -1):
        v = out[i]
        if v is None or v != v:
            out[i] = next_val
        else:
            next_val = v
    return out


def fill_gaps(values: List[float]) -> List[float]:
    return backward_fill(forward_fill(values))

