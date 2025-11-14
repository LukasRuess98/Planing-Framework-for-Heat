from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from energis.io import exporter
from energis.io.exporter import write_timeseries_csv
from energis.utils.timeseries import TimeSeriesTable


def _build_table() -> TimeSeriesTable:
    start = datetime(2024, 1, 1)
    index = [start + timedelta(hours=i) for i in range(3)]
    columns = ["strompreis_EUR_MWh", "grid_co2_kg_MWh"]
    data = {
        "strompreis_EUR_MWh": [1.234, 0.0, 1e-9],
        "grid_co2_kg_MWh": [Decimal("2.5"), 3.75, 0.0],
    }
    return TimeSeriesTable(index=index, columns=columns, data=data)


def test_write_timeseries_csv_writes_both_decimal_variants(tmp_path: Path) -> None:
    table = _build_table()
    extra = {"calc": ["1,5", Decimal("4.250"), 0.0]}
    primary = tmp_path / "primary.csv"
    alternate = tmp_path / "alternate.csv"

    write_timeseries_csv(
        str(primary),
        table,
        extra,
        decimal_separator=",",
        alternate_path=str(alternate),
        alternate_decimal_separator=".",
    )

    primary_lines = primary.read_text(encoding="utf-8").splitlines()
    alternate_lines = alternate.read_text(encoding="utf-8").splitlines()

    assert primary_lines[0] == "timestamp;strompreis_EUR_MWh;grid_co2_kg_MWh;calc"
    assert alternate_lines[0] == primary_lines[0]
    assert "1,234" in primary_lines[1]
    assert "1.234" in alternate_lines[1]
    assert primary_lines[3].endswith("0")
    assert alternate_lines[3].endswith("0")


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1,25", "1,25"),
        ("", ""),
        (" 2.50 ", "2,5"),
        (1.5, "1,5"),
    ],
)
def test_fmt_value_normalises_strings_and_numbers(value: object, expected: str) -> None:
    assert exporter._fmt_value(value, decimal_separator=",") == expected


def test_fmt_value_falls_back_to_text_for_non_numeric() -> None:
    assert exporter._fmt_value("foo", decimal_separator=",") == "foo"


def test_extract_pyomo_series_handles_invalid_data(monkeypatch, caplog) -> None:
    from energis.run import orchestrator

    monkeypatch.setattr(orchestrator, "HAVE_PYOMO", True)
    monkeypatch.setattr(orchestrator, "pyo", SimpleNamespace(value=lambda obj: obj))

    class DummyVar:
        def __init__(self):
            self._values = {
                0: 1.0,
                1: ValueError("bad"),
                2: float("nan"),
                3: 5e-10,
            }
            self.name = "dummy"

        def __getitem__(self, key):
            val = self._values[key]
            if isinstance(val, Exception):
                raise val
            return val

    times = [0, 1, 2, 3]
    with caplog.at_level("WARNING"):
        result = orchestrator._extract_pyomo_series(DummyVar(), times, "dummy")

    assert result == [1.0, 0.0, 0.0, 0.0]
    assert "dummy[1]" in caplog.text

