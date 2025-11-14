import pytest, os
try:
    import pyomo.environ as pyo
    HAVE_PYOMO=True
except Exception:
    HAVE_PYOMO=False

from energis.config.merge import load_and_merge
from energis.io.loader import load_input_excel
from energis.models.system_builder import build_model
import pandas as pd

@pytest.mark.skipif(not HAVE_PYOMO, reason="Pyomo not available")
def test_bus_constraints_exist(tmp_path):
    # minimal data
    xls = tmp_path/"Import_Data.xlsx"
    idx = pd.date_range("2023-01-01 00:00", periods=3, freq="H")
    df = pd.DataFrame({
        "Datum": idx,
        "Day_Ahead_Price €/MWh": [50,55,60],
        "Wärmebedarf MW": [10,10,10],
        "CO2_consumption_based kgCO2/MWh": [350,350,350],
        "WRG1Q MW":[2,2,2],
        "WRG1_T °C":[50,50,50],
        "WRG2Q MW":[1,1,1],
        "WRG2_T °C":[45,45,45],
        "WRG3Q MW":[0,0,0],
        "WRG3_T °C":[40,40,40],
        "WRG4Q MW":[0,0,0],
        "WRG4_T °C":[35,35,35],
    })
    with pd.ExcelWriter(xls) as xw:
        df.to_excel(xw, index=False)

    cfg = load_and_merge([
        "configs/base.yaml",
        "configs/tech_catalog.yaml",
        "configs/sites/default.site.yaml",
        "configs/systems/baseline.system.yaml",
        "configs/scenarios/pf_then_rh.scenario.yaml",
    ])
    cfg["site"]["input_xlsx"] = str(xls)
    data = load_input_excel(cfg["site"]["input_xlsx"], cfg["site"], dt_hours=1.0)
    m = build_model(data, cfg, dt_h=1.0)
    assert hasattr(m, "el_balance")
    assert hasattr(m, "ht_balance")
