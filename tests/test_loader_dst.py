from datetime import datetime, timedelta

import pytest

from energis.config.merge import load_and_merge
from energis.io.loader import load_input_excel
from energis.utils.xlsx import write_simple_xlsx


def _date_range(start: str, periods: int, freq_hours: int = 1):
    base = datetime.fromisoformat(start)
    return [base + timedelta(hours=freq_hours * i) for i in range(periods)]

def test_merge_and_loader_smoke(tmp_path):
    # minimal merge smoke test
    cfg = load_and_merge([
        "configs/base.yaml",
        "configs/tech_catalog.yaml",
        "configs/sites/default.site.yaml",
        "configs/systems/baseline.system.yaml",
        "configs/scenarios/pf_then_rh.scenario.yaml",
    ])
    assert "site" in cfg and "system" in cfg

    # synthesize tiny excel to check DST pipeline (no pyomo needed)
    xls = tmp_path/"Import_Data.xlsx"
    idx = _date_range("2023-03-26T00:00:00", periods=6)
    headers = [
        "Datum",
        "Day_Ahead_Price €/MWh",
        "Wärmebedarf MW",
        "CO2_consumption_based kgCO2/MWh",
        "WRG1Q MW",
        "WRG1_T °C",
        "WRG2Q MW",
        "WRG2_T °C",
        "WRG3Q MW",
        "WRG3_T °C",
        "WRG4Q MW",
        "WRG4_T °C",
    ]
    rows = [
        [idx[i], 50 + i % 3, 10, 340 + 5 * (i % 3), 2, 50, 1, 45, 0, 40, 0, 35]
        for i in range(len(idx))
    ]
    write_simple_xlsx(str(xls), headers, rows)

    cfg["site"]["input_xlsx"] = str(xls)
    data = load_input_excel(cfg["site"]["input_xlsx"], cfg["site"], dt_hours=1.0)
    assert len(data) == 6
    assert {"strompreis_EUR_MWh","waermebedarf_MWth","grid_co2_kg_MWh"}.issubset(set(data.columns))
