from datetime import datetime, timedelta
import os

import pytest

try:
    import pyomo.environ as pyo
    HAVE_PYOMO = True
except Exception:
    HAVE_PYOMO = False

from energis.config.merge import load_and_merge
from energis.io.loader import load_input_excel
from energis.models.system_builder import build_model
from energis.utils.xlsx import write_simple_xlsx


def _date_range(start: str, periods: int, freq_hours: int = 1):
    base = datetime.fromisoformat(start)
    return [base + timedelta(hours=freq_hours * i) for i in range(periods)]

@pytest.mark.skipif(not HAVE_PYOMO, reason="Pyomo not available")
def test_bus_constraints_exist(tmp_path):
    # minimal data
    xls = tmp_path/"Import_Data.xlsx"
    idx = _date_range("2023-01-01T00:00:00", periods=3)
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
        [idx[i], 50 + 5 * i, 10, 350, 2, 50, 1, 45, 0, 40, 0, 35]
        for i in range(len(idx))
    ]
    write_simple_xlsx(str(xls), headers, rows)

    cfg = load_and_merge([
        "configs/base.yaml",
        "configs/tech_catalog.yaml",
        "configs/sites/default.site.yaml",
        "configs/systems/baseline.system.yaml",
        "configs/scenarios/perfect_forecast_full_year.scenario.yaml",
    ])
    cfg["site"]["input_xlsx"] = str(xls)
    data = load_input_excel(cfg["site"]["input_xlsx"], cfg["site"], dt_hours=1.0)
    m = build_model(data, cfg, dt_h=1.0)
    assert hasattr(m, "el_balance")
    assert hasattr(m, "ht_balance")
