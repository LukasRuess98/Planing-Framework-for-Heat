from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile
import json

import pytest

from energis.io import exporter
from energis.io.exporter import write_timeseries_csv
from energis.utils.timeseries import TimeSeriesTable
from energis.run import orchestrator


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


def test_run_all_creates_export_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    config = {
        "site": {"input_xlsx": "dummy.xlsx"},
        "run": {"dt_h": 1.0, "solver": "dummy"},
        "scenario": {"title": "Demo", "mode": "PF"},
        "system": {
            "heat_pumps": [
                {"id": "HP1", "max_th_mw": 20.0, "min_th_mw": 1.0, "investment": {}},
            ],
        },
    }

    table = TimeSeriesTable(
        index=[datetime(2024, 1, 1), datetime(2024, 1, 1, 1)],
        columns=["waermebedarf_MWth"],
        data={"waermebedarf_MWth": [5.0, 6.0]},
    )

    series = OrderedDict({"P_buy_MW": [0.0, 0.0]})
    summary = OrderedDict(
        {
            "objective": OrderedDict({"OBJ_value_EUR": 1.0}),
            "heat_pump_HP1": OrderedDict(
                {
                    "Thermal_capacity_MW": 5.0,
                    "Build_binary": 1.0,
                }
            ),
        }
    )
    costs = {"objective.OBJ_value_EUR": 1.0}

    monkeypatch.setattr(orchestrator, "load_and_merge", lambda *_args: config)
    monkeypatch.setattr(orchestrator, "load_input_excel", lambda *args, **kwargs: table)
    monkeypatch.setattr(orchestrator, "build_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        orchestrator,
        "_collect_timeseries_and_summary",
        lambda *args, **kwargs: (series, summary, costs),
    )
    monkeypatch.setattr(orchestrator, "export_plots", lambda *args, **kwargs: [])

    result = orchestrator.run_all([])

    assert result["scenario_xlsx"] is not None
    xlsx_path = Path(result["scenario_xlsx"])
    assert xlsx_path.exists()

    manifest_path = Path(result["manifest_json"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["scenario_title"] == "Demo"
    assert manifest["flags"]["has_design"] is True

    design_path = Path(result["pf_design_json"])
    design = json.loads(design_path.read_text(encoding="utf-8"))
    assert design["heat_pumps"]["HP1"]["capacity_mw"] == pytest.approx(5.0)

    with ZipFile(xlsx_path, "r") as archive:
        workbook_xml = archive.read("xl/workbook.xml").decode("utf-8")
    assert "Timeseries" in workbook_xml

